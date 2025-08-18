#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
클래스별로 10장씩 이미지를 샘플링하고 배경을 제거하여 저장하는 스크립트
"""

import json
import os
import shutil
import random
from collections import defaultdict
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2

# background_removal 모듈 import
from lib.utils.background_removal import BgRemovalConfig, process_image

def load_json_data(json_path):
    """JSON 파일을 로드합니다."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def sample_images_by_class(data, samples_per_class=10, random_seed=42, target_classes=None):
    """클래스별로 지정된 개수만큼 이미지를 샘플링합니다."""
    random.seed(random_seed)
    
    # 타겟 클래스가 지정되지 않으면 모든 클래스 사용
    if target_classes is None:
        target_classes = set()  # 빈 집합으로 초기화 (모든 클래스 포함)
    
    # 클래스별로 이미지 그룹화 (file_path가 유효한 데이터만)
    class_images = defaultdict(list)
    skipped_count = 0
    
    for item in data['data']:
        class_name = item['class']
        file_path = item.get('file_path', '')
        
        # 타겟 클래스가 지정되어 있고, 현재 클래스가 타겟에 없으면 스킵
        if target_classes and class_name not in target_classes:
            continue
        
        # file_path가 빈 리스트이거나 빈 문자열이거나 None인 경우 스킵
        if not file_path or file_path == [] or file_path == "":
            skipped_count += 1
            continue
            
        # file_path가 리스트인 경우 첫 번째 요소 사용 (혹시 모를 경우)
        if isinstance(file_path, list):
            if len(file_path) > 0:
                file_path = file_path[0]
            else:
                skipped_count += 1
                continue
        
        # file_path가 문자열인지 확인
        if not isinstance(file_path, str):
            skipped_count += 1
            continue
            
        # 유효한 이미지 파일 경로인지 확인
        if not file_path.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
            skipped_count += 1
            continue
            
        class_images[class_name].append(item)
    
    if skipped_count > 0:
        print(f"Warning: {skipped_count}개의 유효하지 않은 데이터를 스킵했습니다.")
    
    # 클래스별로 샘플링
    sampled_images = {}
    for class_name, images in class_images.items():
        if len(images) >= samples_per_class:
            sampled = random.sample(images, samples_per_class)
        else:
            # 해당 클래스의 이미지가 부족한 경우 전체 사용
            sampled = images
            print(f"Warning: 클래스 '{class_name}'의 이미지가 {samples_per_class}개보다 적습니다. ({len(images)}개 사용)")
        
        sampled_images[class_name] = sampled
        print(f"클래스 '{class_name}': {len(sampled_images[class_name])}개 샘플링")
    
    return sampled_images

def copy_images_with_background_removal(sampled_images, output_dir, bg_config, create_class_folders=True, save_original=True):
    """샘플링된 이미지들을 배경 제거 후 출력 디렉토리로 저장합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 원본과 배경 제거된 이미지를 위한 별도 폴더 생성
    if save_original:
        original_dir = output_path / "original"
        processed_dir = output_path / "background_removed"
        original_dir.mkdir(exist_ok=True)
        processed_dir.mkdir(exist_ok=True)
    else:
        processed_dir = output_path
    
    total_processed = 0
    total_copied = 0
    processed_images = {}  # 클래스별로 처리된 이미지 정보 저장
    
    for class_name, images in sampled_images.items():
        if create_class_folders:
            if save_original:
                original_class_dir = original_dir / class_name
                processed_class_dir = processed_dir / class_name
                original_class_dir.mkdir(exist_ok=True)
                processed_class_dir.mkdir(exist_ok=True)
            else:
                original_class_dir = None
                processed_class_dir = processed_dir / class_name
                processed_class_dir.mkdir(exist_ok=True)
        else:
            if save_original:
                original_class_dir = original_dir
                processed_class_dir = processed_dir
            else:
                original_class_dir = None
                processed_class_dir = processed_dir
        
        print(f"\n클래스 '{class_name}' 이미지 처리 중...")
        
        processed_images[class_name] = []
        
        for i, img_data in enumerate(images):
            src_path = str(Path(img_data['file_path']))
            
            if not Path(src_path).exists():
                print(f"Warning: 이미지 파일을 찾을 수 없습니다: {src_path}")
                continue
            
            # 기본 파일명 생성
            base_filename = f"{img_data['patient_id']}_{i+1:02d}"
            
            # 원본 이미지 복사 (save_original이 True인 경우)
            original_saved_path = None
            if save_original and original_class_dir is not None:
                if create_class_folders:
                    original_filename = f"{base_filename}_original.jpg"
                    original_dst_path = str(original_class_dir / original_filename)
                else:
                    original_filename = f"{class_name}_{base_filename}_original.jpg"
                    original_dst_path = str(original_class_dir / original_filename)
                
                try:
                    # 원본 이미지를 JPG로 변환하여 저장
                    import cv2
                    original_img = cv2.imread(src_path)
                    if original_img is not None:
                        cv2.imwrite(original_dst_path, original_img)
                        original_saved_path = original_dst_path
                        total_copied += 1
                    else:
                        print(f"  Warning: 원본 이미지를 읽을 수 없습니다: {src_path}")
                except Exception as e:
                    print(f"  Warning: 원본 이미지 복사 실패: {e}")
            
            # 배경 제거된 이미지 저장
            if create_class_folders:
                processed_filename = f"{base_filename}_bg_removed.jpg"
                processed_dst_path = str(processed_class_dir / processed_filename)
            else:
                processed_filename = f"{class_name}_{base_filename}_bg_removed.jpg"
                processed_dst_path = str(processed_class_dir / processed_filename)
            
            try:
                # 배경 제거 처리
                success, error_msg = process_image(src_path, processed_dst_path, bg_config)
                
                if success:
                    total_processed += 1
                    status_msg = f"  ✓ {i+1:2d}/{len(images)}: {img_data['patient_id']} - 배경 제거 완료"
                    if save_original and original_saved_path:
                        status_msg += " (원본도 저장됨)"
                    print(status_msg)
                    
                    # 처리된 이미지 정보 저장 (히스토그램 분석용)
                    processed_images[class_name].append({
                        'original_path': src_path,
                        'original_saved_path': original_saved_path if save_original else None,
                        'processed_path': processed_dst_path,
                        'patient_id': img_data['patient_id'],
                        'index': i
                    })
                else:
                    print(f"  ✗ {i+1:2d}/{len(images)}: {img_data['patient_id']} - 배경 제거 실패: {error_msg}")
                    
            except Exception as e:
                print(f"  ✗ {i+1:2d}/{len(images)}: {img_data['patient_id']} - 오류: {e}")
    
    if save_original:
        print(f"\n원본 이미지 복사 완료: {total_copied}개")
    print(f"배경 제거 완료: {total_processed}개")
    
    return total_processed, processed_images

def create_histogram_comparison(processed_images, output_dir):
    """클래스별로 한 개씩 이미지를 선택해서 히스토그램 비교를 생성합니다."""
    print("\n=== 히스토그램 비교 생성 중 ===")
    
    # matplotlib 한글 폰트 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    for class_name, images in processed_images.items():
        if not images:
            continue
            
        # 첫 번째 이미지 선택 (또는 랜덤 선택)
        selected_img = images[0]
        
        try:
            # 원본 이미지 로드 (저장된 원본이 있으면 그것을 사용, 없으면 원래 경로 사용)
            original_path = selected_img.get('original_saved_path') or selected_img['original_path']
            original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            if original_img is None:
                print(f"Warning: 원본 이미지를 로드할 수 없습니다: {original_path}")
                continue
                
            # 배경 제거된 이미지 로드
            processed_img = cv2.imread(selected_img['processed_path'], cv2.IMREAD_GRAYSCALE)
            if processed_img is None:
                print(f"Warning: 처리된 이미지를 로드할 수 없습니다: {selected_img['processed_path']}")
                continue
            
            # 히스토그램 계산
            hist_original = cv2.calcHist([original_img], [0], None, [256], [0, 256])
            hist_processed = cv2.calcHist([processed_img], [0], None, [256], [0, 256])
            
            # 히스토그램 정규화 (0-1 범위)
            hist_original = hist_original.flatten() / np.sum(hist_original)
            hist_processed = hist_processed.flatten() / np.sum(hist_processed)
            
            # 히스토그램 비교 플롯 생성
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Class: {class_name} - Histogram Comparison (Patient: {selected_img["patient_id"]})', fontsize=16)
            
            # 원본 이미지
            axes[0, 0].imshow(original_img, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # 배경 제거된 이미지
            axes[0, 1].imshow(processed_img, cmap='gray')
            axes[0, 1].set_title('Background Removed Image')
            axes[0, 1].axis('off')
            
            # 히스토그램 비교
            axes[1, 0].plot(hist_original, color='blue', alpha=0.7, label='Original')
            axes[1, 0].plot(hist_processed, color='red', alpha=0.7, label='Background Removed')
            axes[1, 0].set_title('Histogram Comparison')
            axes[1, 0].set_xlabel('Pixel Value (0-255)')
            axes[1, 0].set_ylabel('Normalized Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 히스토그램 차이
            hist_diff = hist_processed - hist_original
            axes[1, 1].bar(range(256), hist_diff, color='green', alpha=0.7)
            axes[1, 1].set_title('Histogram Difference (Processed - Original)')
            axes[1, 1].set_xlabel('Pixel Value (0-255)')
            axes[1, 1].set_ylabel('Normalized Frequency Difference')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 통계 정보 추가
            stats_text = f"""
Statistics:
Original Mean: {np.mean(original_img.astype(np.float64)):.2f}
Processed Mean: {np.mean(processed_img.astype(np.float64)):.2f}
Original Std: {np.std(original_img.astype(np.float64)):.2f}
Processed Std: {np.std(processed_img.astype(np.float64)):.2f}
Original Max: {np.max(original_img)}
Processed Max: {np.max(processed_img)}
Original Min: {np.min(original_img)}
Processed Min: {np.min(processed_img)}
            """
            
            # 통계 정보를 텍스트로 추가
            fig.text(0.02, 0.02, stats_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # 히스토그램 이미지 저장
            histogram_path = Path(output_dir) / f"histogram_{class_name}.png"
            plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 클래스 '{class_name}': 히스토그램 저장 완료 - {histogram_path}")
            
        except Exception as e:
            print(f"  ✗ 클래스 '{class_name}': 히스토그램 생성 실패 - {e}")
    
    print("히스토그램 비교 생성 완료!")

def create_summary_report(sampled_images, output_dir, bg_config, total_processed, save_original=True):
    """샘플링 결과 요약 리포트를 생성합니다."""
    report_path = Path(output_dir) / "sampling_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 클래스별 이미지 샘플링 및 배경 제거 결과 ===\n\n")
        
        f.write("배경 제거 설정:\n")
        f.write(f"  - 방법: {bg_config.method}\n")
        if bg_config.method == "fixed":
            f.write(f"  - 임계값: {bg_config.fixed_thresh}\n")
        elif bg_config.method == "percentile":
            f.write(f"  - 백분위: {bg_config.percentile}%\n")
        elif bg_config.method == "hsv_value":
            f.write(f"  - HSV V 채널 임계값: {bg_config.hsv_value_thresh}\n")
            f.write(f"  - 피부 영역 보호: {bg_config.protect_skin}\n")
            f.write(f"  - 뼈/관절 영역 보호: {bg_config.protect_bone}\n")
        f.write(f"  - 형태학 커널: {bg_config.morph_kernel}\n")
        f.write(f"  - 최대 연결 성분만 유지: {bg_config.keep_largest_only}\n")
        f.write(f"  - 타이트 크롭: {bg_config.tight_crop}\n")
        f.write(f"  - 최소 객체 면적: {bg_config.min_object_area}\n\n")
        
        f.write("처리 결과:\n")
        f.write(f"  - 총 처리된 이미지 수: {total_processed}\n\n")
        
        total_samples = 0
        for class_name, images in sampled_images.items():
            f.write(f"클래스: {class_name}\n")
            f.write(f"  - 샘플링된 이미지 수: {len(images)}\n")
            f.write(f"  - 이미지 목록:\n")
            
            for i, img_data in enumerate(images):
                f.write(f"    {i+1:2d}. {img_data['patient_id']} - {img_data['file_path']}\n")
            
            f.write("\n")
            total_samples += len(images)
        
        f.write(f"\n총 샘플링된 이미지 수: {total_samples}\n")
        f.write(f"출력 디렉토리: {output_dir}\n")
        
        if save_original:
            f.write(f"원본 이미지 저장 위치: {output_dir}/original/\n")
            f.write(f"배경 제거된 이미지 저장 위치: {output_dir}/background_removed/\n")
        else:
            f.write(f"배경 제거된 이미지만 저장됨\n")
        
        f.write(f"히스토그램 비교 이미지: histogram_*.png\n")
    
    print(f"샘플링 리포트가 생성되었습니다: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='클래스별로 이미지를 샘플링하고 배경을 제거하여 저장합니다.')
    parser.add_argument('--json_path', type=str, default='data/json/RA_hand_merge_cau.json',
                       help='JSON 데이터 파일 경로')
    parser.add_argument('--output_dir', type=str, default='sampled_images_bg_removed',
                       help='출력 디렉토리')
    parser.add_argument('--samples_per_class', type=int, default=10,
                       help='클래스당 샘플링할 이미지 수')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='랜덤 시드')
    parser.add_argument('--create_class_folders', action='store_true',
                       help='클래스별로 폴더를 생성합니다')
    
    # 배경 제거 관련 인자들
    parser.add_argument('--bg_method', type=str, default='hsv_value', choices=['fixed', 'otsu', 'percentile', 'hsv_value'],
                       help='배경 제거 방법')
    parser.add_argument('--bg_thresh', type=int, default=10,
                       help='fixed 방법일 때 임계값 (0-255)')
    parser.add_argument('--bg_percentile', type=float, default=2.0,
                       help='percentile 방법일 때 하위 백분위 (0-100)')
    parser.add_argument('--bg_hsv_thresh', type=int, default=15,
                       help='HSV V 채널 임계값 (hsv_value 방법 사용시)')
    parser.add_argument('--bg_protect_skin', action='store_true', default=True,
                       help='피부 영역 보호 (기본값: True)')
    parser.add_argument('--no_protect_skin', dest='bg_protect_skin', action='store_false',
                       help='피부 영역 보호 비활성화')
    parser.add_argument('--bg_protect_bone', action='store_true', default=True,
                       help='뼈/관절 영역 보호 (기본값: True)')
    parser.add_argument('--no_protect_bone', dest='bg_protect_bone', action='store_false',
                       help='뼈/관절 영역 보호 비활성화')
    parser.add_argument('--bg_morph_kernel', type=int, default=5,
                       help='형태학 커널 크기 (홀수)')
    parser.add_argument('--bg_keep_largest', action='store_true', default=True,
                       help='가장 큰 연결 성분만 유지')
    parser.add_argument('--bg_tight_crop', action='store_true',
                       help='마스크 bbox로 타이트 크롭')
    parser.add_argument('--bg_min_area', type=int, default=5000,
                       help='최소 객체 면적 (픽셀)')
    
    # 타겟 클래스 선택 인자 추가
    parser.add_argument('--target_classes', type=str, nargs='+', 
                       default=['oa', 'ra', 'normal'],
                       help='처리할 클래스 목록 (기본값: oa, ra, normal)')
    
    # 원본 이미지 저장 여부
    parser.add_argument('--save_original', action='store_true', default=True,
                       help='원본 이미지도 함께 저장 (기본값: True)')
    parser.add_argument('--no_save_original', dest='save_original', action='store_false',
                       help='원본 이미지를 저장하지 않음')
    
    args = parser.parse_args()
    
    print("=== 이미지 클래스별 샘플링 및 배경 제거 시작 ===")
    print(f"JSON 파일: {args.json_path}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"클래스당 샘플 수: {args.samples_per_class}")
    print(f"랜덤 시드: {args.random_seed}")
    print(f"클래스별 폴더 생성: {args.create_class_folders}")
    print(f"처리할 클래스: {args.target_classes}")
    print(f"원본 이미지 저장: {args.save_original}")
    print()
    
    # 배경 제거 설정 생성 (HSV 기반 배경 제거 사용)
    bg_config = BgRemovalConfig(
        method=args.bg_method,  # 커맨드 라인에서 지정된 방법 사용
        fixed_thresh=args.bg_thresh,
        percentile=args.bg_percentile,
        hsv_value_thresh=args.bg_hsv_thresh,  # HSV V 채널 임계값 (기본값: 15)
        protect_skin=args.bg_protect_skin,  # 피부 영역 보호
        protect_bone=args.bg_protect_bone,  # 뼈/관절 영역 보호
        morph_kernel=args.bg_morph_kernel,
        keep_largest_only=False if args.bg_method == "hsv_value" else args.bg_keep_largest,  # HSV 방식에서는 형태학적 후처리 생략
        tight_crop=args.bg_tight_crop,
        min_object_area=args.bg_min_area,
        normalize_to_uint8=True,
        save_original=False
    )
    
    print("배경 제거 설정:")
    print(f"  - 방법: {bg_config.method}")
    if bg_config.method == "fixed":
        print(f"  - 임계값: {bg_config.fixed_thresh}")
    elif bg_config.method == "percentile":
        print(f"  - 백분위: {bg_config.percentile}%")
    elif bg_config.method == "hsv_value":
        print(f"  - HSV V 채널 임계값: {bg_config.hsv_value_thresh}")
        print(f"  - 피부 영역 보호: {bg_config.protect_skin}")
        print(f"  - 뼈/관절 영역 보호: {bg_config.protect_bone}")
    print(f"  - 형태학 커널: {bg_config.morph_kernel}")
    print(f"  - 최대 연결 성분만 유지: {bg_config.keep_largest_only}")
    print(f"  - 타이트 크롭: {bg_config.tight_crop}")
    print(f"  - 최소 객체 면적: {bg_config.min_object_area}")
    print()
    
    # JSON 데이터 로드
    try:
        data = load_json_data(args.json_path)
        print(f"JSON 데이터 로드 완료: {data['meta']['total_records']}개 레코드")
    except Exception as e:
        print(f"Error: JSON 파일을 로드할 수 없습니다: {e}")
        return
    
    # 클래스별 이미지 샘플링
    sampled_images = sample_images_by_class(
        data, 
        samples_per_class=args.samples_per_class,
        random_seed=args.random_seed,
        target_classes=set(args.target_classes)
    )
    
    print(f"\n총 {len(sampled_images)}개 클래스에서 샘플링 완료")
    
    # 이미지 복사 및 배경 제거
    if args.save_original:
        print("\n이미지 배경 제거 및 원본 저장 중...")
    else:
        print("\n이미지 배경 제거 중...")
    total_processed, processed_images = copy_images_with_background_removal(
        sampled_images, 
        args.output_dir, 
        bg_config,
        create_class_folders=args.create_class_folders,
        save_original=args.save_original
    )
    
    print(f"\n총 {total_processed}개 이미지 처리 완료")
    
    # 히스토그램 비교 생성
    create_histogram_comparison(processed_images, args.output_dir)
    
    # 요약 리포트 생성
    create_summary_report(sampled_images, args.output_dir, bg_config, total_processed, args.save_original)
    
    print(f"\n=== 샘플링 및 배경 제거 완료 ===")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"히스토그램 비교 이미지들이 저장되었습니다.")

if __name__ == "__main__":
    main() 
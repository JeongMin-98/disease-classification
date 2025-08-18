#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fold 6의 test indices를 사용해서 해당 이미지들의 원본과 배경 제거된 버전을 클래스별로 생성하는 스크립트
"""

import json
import os
import shutil
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

def load_kfold_indices(kfold_json_path):
    """K-fold test indices JSON 파일을 로드합니다."""
    with open(kfold_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_fold6_test_images(data, fold6_indices, output_dir, bg_config, create_class_folders=True):
    """Fold 6의 test indices에 해당하는 이미지들을 추출하여 배경 제거 후 저장합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 원본과 배경 제거된 이미지를 위한 별도 폴더 생성
    original_dir = output_path / "original"
    processed_dir = output_path / "background_removed"
    original_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    # Fold 6 test indices를 set으로 변환 (빠른 검색을 위해)
    fold6_indices_set = set(fold6_indices)
    
    # 클래스별로 이미지 그룹화
    class_images = {}
    total_processed = 0
    total_copied = 0
    
    print(f"Fold 6 test indices: {len(fold6_indices)}개")
    print(f"데이터 총 레코드 수: {len(data['data'])}개")
    
    # Fold 6 test indices에 해당하는 이미지들만 추출
    fold6_images = []
    for idx in fold6_indices:
        if idx < len(data['data']):
            item = data['data'][idx]
            fold6_images.append(item)
        else:
            print(f"Warning: 인덱스 {idx}가 데이터 범위를 벗어났습니다.")
    
    print(f"Fold 6에서 추출된 이미지 수: {len(fold6_images)}개")
    
    # 클래스별로 그룹화
    for item in fold6_images:
        class_name = item['class']
        if class_name not in class_images:
            class_images[class_name] = []
        class_images[class_name].append(item)
    
    print(f"클래스별 분포:")
    for class_name, images in class_images.items():
        print(f"  - {class_name}: {len(images)}개")
    
    # 각 클래스별로 이미지 처리
    for class_name, images in class_images.items():
        if create_class_folders:
            original_class_dir = original_dir / class_name
            processed_class_dir = processed_dir / class_name
            original_class_dir.mkdir(exist_ok=True)
            processed_class_dir.mkdir(exist_ok=True)
        else:
            original_class_dir = original_dir
            processed_class_dir = processed_dir
        
        print(f"\n클래스 '{class_name}' 이미지 처리 중... ({len(images)}개)")
        
        for i, img_data in enumerate(images):
            src_path = str(Path(img_data['file_path']))
            
            if not Path(src_path).exists():
                print(f"Warning: 이미지 파일을 찾을 수 없습니다: {src_path}")
                continue
            
            # 파일명 생성 (patient_id와 인덱스 사용)
            base_filename = f"{img_data['patient_id']}_{i+1:02d}"
            
            # 원본 이미지 저장
            if create_class_folders:
                original_filename = f"{base_filename}_original.jpg"
                original_dst_path = str(original_class_dir / original_filename)
            else:
                original_filename = f"{class_name}_{base_filename}_original.jpg"
                original_dst_path = str(original_class_dir / original_filename)
            
            try:
                # 원본 이미지를 JPG로 변환하여 저장
                original_img = cv2.imread(src_path)
                if original_img is not None:
                    cv2.imwrite(original_dst_path, original_img)
                    total_copied += 1
                else:
                    print(f"  Warning: 원본 이미지를 읽을 수 없습니다: {src_path}")
                    continue
            except Exception as e:
                print(f"  Warning: 원본 이미지 복사 실패: {e}")
                continue
            
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
                    print(f"  ✓ {i+1:2d}/{len(images)}: {img_data['patient_id']} - 배경 제거 완료")
                else:
                    print(f"  ✗ {i+1:2d}/{len(images)}: {img_data['patient_id']} - 배경 제거 실패: {error_msg}")
                    
            except Exception as e:
                print(f"  ✗ {i+1:2d}/{len(images)}: {img_data['patient_id']} - 오류: {e}")
    
    print(f"\n=== Fold 6 Test Images 처리 완료 ===")
    print(f"원본 이미지 저장 완료: {total_copied}개")
    print(f"배경 제거 완료: {total_processed}개")
    
    return total_processed, total_copied

def create_fold6_summary_report(class_images, output_dir, bg_config, total_processed, total_copied):
    """Fold 6 test images 처리 결과 요약 리포트를 생성합니다."""
    report_path = Path(output_dir) / "fold6_test_images_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Fold 6 Test Images 처리 결과 ===\n\n")
        
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
        f.write(f"  - 총 처리된 이미지 수: {total_processed}\n")
        f.write(f"  - 총 복사된 원본 이미지 수: {total_copied}\n\n")
        
        f.write("클래스별 분포:\n")
        total_samples = 0
        for class_name, images in class_images.items():
            f.write(f"클래스: {class_name}\n")
            f.write(f"  - 이미지 수: {len(images)}\n")
            f.write(f"  - 이미지 목록:\n")
            
            for i, img_data in enumerate(images):
                f.write(f"    {i+1:2d}. {img_data['patient_id']} - {img_data['file_path']}\n")
            
            f.write("\n")
            total_samples += len(images)
        
        f.write(f"\n총 이미지 수: {total_samples}\n")
        f.write(f"출력 디렉토리: {output_dir}\n")
        f.write(f"원본 이미지 저장 위치: {output_dir}/original/\n")
        f.write(f"배경 제거된 이미지 저장 위치: {output_dir}/background_removed/\n")
    
    print(f"Fold 6 요약 리포트가 생성되었습니다: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Fold 6의 test indices에 해당하는 이미지들을 추출하여 배경 제거 후 저장합니다.')
    parser.add_argument('--json_path', type=str, default='data/json/RA_hand_merge_cau.json',
                       help='JSON 데이터 파일 경로')
    parser.add_argument('--kfold_json_path', type=str, 
                       default='kfold_indices_hsv/kfold_test_indices_ra_hand_classifier_OA_Normal_224_vgg19bn_hsv_realtime_kfold_seed_42.json',
                       help='K-fold test indices JSON 파일 경로')
    parser.add_argument('--output_dir', type=str, default='fold6_test_images',
                       help='출력 디렉토리')
    parser.add_argument('--create_class_folders', action='store_true', default=True,
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
    
    args = parser.parse_args()
    
    print("=== Fold 6 Test Images 추출 및 배경 제거 시작 ===")
    print(f"JSON 파일: {args.json_path}")
    print(f"K-fold indices 파일: {args.kfold_json_path}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"클래스별 폴더 생성: {args.create_class_folders}")
    print()
    
    # 배경 제거 설정 생성 (HSV 기반 배경 제거 사용)
    bg_config = BgRemovalConfig(
        method=args.bg_method,
        fixed_thresh=args.bg_thresh,
        percentile=args.bg_percentile,
        hsv_value_thresh=args.bg_hsv_thresh,
        protect_skin=args.bg_protect_skin,
        protect_bone=args.bg_protect_bone,
        morph_kernel=args.bg_morph_kernel,
        keep_largest_only=False if args.bg_method == "hsv_value" else args.bg_keep_largest,
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
    
    # K-fold indices 로드
    try:
        kfold_data = load_kfold_indices(args.kfold_json_path)
        fold6_indices = kfold_data['fold_6']['indices']
        print(f"Fold 6 test indices 로드 완료: {len(fold6_indices)}개")
    except Exception as e:
        print(f"Error: K-fold indices 파일을 로드할 수 없습니다: {e}")
        return
    
    # Fold 6 test images 추출 및 처리
    print("\nFold 6 test images 처리 중...")
    total_processed, total_copied = extract_fold6_test_images(
        data, 
        fold6_indices,
        args.output_dir, 
        bg_config,
        create_class_folders=args.create_class_folders
    )
    
    # 클래스별 분포 계산
    class_images = {}
    for idx in fold6_indices:
        if idx < len(data['data']):
            item = data['data'][idx]
            class_name = item['class']
            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].append(item)
    
    # 요약 리포트 생성
    create_fold6_summary_report(class_images, args.output_dir, bg_config, total_processed, total_copied)
    
    print(f"\n=== Fold 6 Test Images 처리 완료 ===")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"원본 이미지: {args.output_dir}/original/")
    print(f"배경 제거된 이미지: {args.output_dir}/background_removed/")

if __name__ == "__main__":
    main()

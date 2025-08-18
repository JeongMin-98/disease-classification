#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
원본과 배경제거된 이미지들의 CAM을 비교하는 스크립트
"""

import _init_path
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
import time

# pytorch_grad_cam 라이브러리 import
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 프로젝트 모듈 import
from models import create as create_model
from config import cfg, update_config
import torchvision.transforms as transforms

def load_and_preprocess_image(image_path, cfg):
    """이미지를 로드하고 전처리합니다."""
    # cfg에서 이미지 크기 가져오기
    image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', [224, 224])
    if isinstance(image_size, int):
        target_size = (image_size, image_size)
    elif isinstance(image_size, (list, tuple)):
        if len(image_size) == 2:
            target_size = tuple(image_size)
        else:
            target_size = (224, 224)
    else:
        target_size = (224, 224)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # BGR -> RGB 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]  # (height, width)
    
    # 모델 입력용으로 리사이즈
    image_resized = cv2.resize(image, target_size)
    
    # PIL Image로 변환 (전처리용)
    pil_image = Image.fromarray(image_resized)
    
    # 전처리 파이프라인
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 전처리된 이미지
    input_tensor = transform(pil_image).unsqueeze(0)
    
    return input_tensor, image_resized, original_size

def get_gradcam_target_layer(model):
    """GradCAM을 위한 타겟 레이어를 반환합니다."""
    # 모델 구조에 따라 타겟 레이어 결정
    if hasattr(model, 'model'):
        # wrapper 모델인 경우
        model_arch = model.model
    else:
        model_arch = model
    
    # ResNet 계열
    if hasattr(model_arch, 'layer4'):
        target_layer = model_arch.layer4[-1]
    elif hasattr(model_arch, 'layer3'):
        target_layer = model_arch.layer3[-1]
    # VGG 계열
    elif hasattr(model_arch, 'features'):
        target_layer = model_arch.features[-1]
    else:
        # 기본값: 마지막 레이어
        target_layer = list(model_arch.children())[-1]
    
    return target_layer 

def inference_and_cam(model, image_path, target_layer, class_names, device, cfg):
    """이미지에 대해 Inference와 CAM을 수행합니다."""
    # 이미지 로드 및 전처리
    input_tensor, original_image, original_size = load_and_preprocess_image(image_path, cfg)
    input_tensor = input_tensor.to(device)
    
    # 모델 예측
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # GradCAM 생성
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        # targets 파라미터 없이 호출 (기본값 사용)
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]  # 첫 번째 차원 제거
    
    # 결과 정보
    result_info = {
        'predicted_class': predicted_class,
        'predicted_class_name': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy(),
        'cam': grayscale_cam,
        'original_image': original_image,
        'original_size': original_size
    }
    
    return result_info

def inference_and_cam_comparison(model, original_path, bg_removed_path, target_layer, class_names, device, cfg):
    """원본과 배경제거 이미지에 대해 Inference와 CAM을 수행하고 비교합니다."""
    
    # 원본 이미지 처리
    try:
        original_result = inference_and_cam(model, original_path, target_layer, class_names, device, cfg)
        original_result['image_type'] = 'Original'
        original_result['image_path'] = original_path
    except Exception as e:
        print(f"    Warning: 원본 이미지 처리 실패: {e}")
        return None
    
    # 배경제거 이미지 처리
    try:
        bg_removed_result = inference_and_cam(model, bg_removed_path, target_layer, class_names, device, cfg)
        bg_removed_result['image_type'] = 'Background Removed'
        bg_removed_result['image_path'] = bg_removed_path
    except Exception as e:
        print(f"    Warning: 배경제거 이미지 처리 실패: {e}")
        return None
    
    return {
        'original': original_result,
        'bg_removed': bg_removed_result,
        'patient_id': Path(original_path).stem.split('_')[0] if '_' in Path(original_path).stem else Path(original_path).stem
    }

def create_comparison_visualization(comparison_result, output_path, class_names):
    """원본과 배경제거 이미지의 CAM을 비교하여 시각화합니다."""
    original = comparison_result['original']
    bg_removed = comparison_result['bg_removed']
    patient_id = comparison_result['patient_id']
    
    # 3x2 서브플롯 생성 (원본 vs 배경제거)
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle(f'Original vs Background Removed - CAM Comparison\nPatient ID: {patient_id}', 
                 fontsize=18, fontweight='bold')
    
    # 열 제목
    axes[0, 0].text(0.5, 1.1, 'Original Image', transform=axes[0, 0].transAxes, 
                    fontsize=14, fontweight='bold', ha='center')
    axes[0, 1].text(0.5, 1.1, 'Background Removed Image', transform=axes[0, 1].transAxes, 
                    fontsize=14, fontweight='bold', ha='center')
    
    # 첫 번째 행: 원본 이미지들
    axes[0, 0].imshow(original['original_image'])
    axes[0, 0].set_title(f"Pred: {original['predicted_class_name']} (Conf: {original['confidence']:.3f})")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(bg_removed['original_image'])
    axes[0, 1].set_title(f"Pred: {bg_removed['predicted_class_name']} (Conf: {bg_removed['confidence']:.3f})")
    axes[0, 1].axis('off')
    
    # 두 번째 행: CAM
    im1 = axes[1, 0].imshow(original['cam'], cmap='jet')
    axes[1, 0].set_title('Class Activation Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(bg_removed['cam'], cmap='jet')
    axes[1, 1].set_title('Class Activation Map')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 세 번째 행: CAM 오버레이
    try:
        # 원본 이미지 CAM 오버레이
        original_normalized = original['original_image'].astype(np.float32) / 255.0
        overlay1 = show_cam_on_image(original_normalized, original['cam'], use_rgb=True)
        axes[2, 0].imshow(overlay1)
        axes[2, 0].set_title('CAM Overlay')
    except Exception as e:
        print(f"    Warning: 원본 CAM 오버레이 생성 실패: {e}")
        axes[2, 0].imshow(original['original_image'])
        axes[2, 0].set_title('CAM Overlay (Failed)')
    axes[2, 0].axis('off')
    
    try:
        # 배경제거 이미지 CAM 오버레이
        bg_removed_normalized = bg_removed['original_image'].astype(np.float32) / 255.0
        overlay2 = show_cam_on_image(bg_removed_normalized, bg_removed['cam'], use_rgb=True)
        axes[2, 1].imshow(overlay2)
        axes[2, 1].set_title('CAM Overlay')
    except Exception as e:
        print(f"    Warning: 배경제거 CAM 오버레이 생성 실패: {e}")
        axes[2, 1].imshow(bg_removed['original_image'])
        axes[2, 1].set_title('CAM Overlay (Failed)')
    axes[2, 1].axis('off')
    
    # 통계 정보 추가
    stats_text = f"""
Comparison Statistics:
Original - Pred: {original['predicted_class_name']}, Conf: {original['confidence']:.3f}
BG Removed - Pred: {bg_removed['predicted_class_name']}, Conf: {bg_removed['confidence']:.3f}

Prediction Match: {'✓' if original['predicted_class'] == bg_removed['predicted_class'] else '✗'}
Confidence Diff: {abs(original['confidence'] - bg_removed['confidence']):.3f}

Original Probabilities: {', '.join([f'{class_names[i]}: {prob:.3f}' for i, prob in enumerate(original['probabilities'])])}
BG Removed Probabilities: {', '.join([f'{class_names[i]}: {prob:.3f}' for i, prob in enumerate(bg_removed['probabilities'])])}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 비교 시각화 저장 완료: {output_path}") 

def find_matching_images(original_dir, bg_removed_dir):
    """원본과 배경제거 디렉토리에서 매칭되는 이미지 쌍을 찾습니다."""
    matching_pairs = []
    
    # 원본 이미지들 찾기
    original_images = list(original_dir.glob('*.jpg')) + list(original_dir.glob('*.png'))
    
    for original_path in original_images:
        # 파일명에서 patient_id와 index 추출
        # 예: CAU001_01_original.jpg -> CAU001_01_bg_removed.jpg
        original_stem = original_path.stem
        if '_original' in original_stem:
            base_name = original_stem.replace('_original', '')
            bg_removed_name = f"{base_name}_bg_removed.jpg"
        else:
            # _original이 없는 경우, 파일명 패턴을 다시 확인
            base_name = original_stem
            bg_removed_name = f"{base_name}_bg_removed.jpg"
        
        bg_removed_path = bg_removed_dir / bg_removed_name
        
        if bg_removed_path.exists():
            matching_pairs.append((str(original_path), str(bg_removed_path)))
        else:
            # 다른 확장자도 시도
            for ext in ['.png', '.tif', '.tiff']:
                bg_removed_path_alt = bg_removed_dir / f"{base_name}_bg_removed{ext}"
                if bg_removed_path_alt.exists():
                    matching_pairs.append((str(original_path), str(bg_removed_path_alt)))
                    break
            else:
                print(f"  Warning: 매칭되는 배경제거 이미지를 찾을 수 없습니다: {original_path.name}")
    
    return matching_pairs

def process_class_comparison(model, class_name, sampled_dir, target_layer, class_names, device, output_dir, cfg):
    """특정 클래스의 원본과 배경제거 이미지들을 비교합니다."""
    print(f"\n=== 클래스 '{class_name}' 비교 처리 중 ===")
    
    # 디렉토리 경로
    original_class_dir = sampled_dir / "original" / class_name
    bg_removed_class_dir = sampled_dir / "background_removed" / class_name
    
    if not original_class_dir.exists():
        print(f"  Warning: 원본 이미지 디렉토리를 찾을 수 없습니다: {original_class_dir}")
        return 0
    
    if not bg_removed_class_dir.exists():
        print(f"  Warning: 배경제거 이미지 디렉토리를 찾을 수 없습니다: {bg_removed_class_dir}")
        return 0
    
    # 매칭되는 이미지 쌍 찾기
    matching_pairs = find_matching_images(original_class_dir, bg_removed_class_dir)
    
    if not matching_pairs:
        print(f"  Warning: 클래스 '{class_name}'에서 매칭되는 이미지 쌍을 찾을 수 없습니다.")
        return 0
    
    print(f"  총 {len(matching_pairs)}개 이미지 쌍 발견")
    
    # 결과 저장 디렉토리 생성
    class_output_dir = Path(output_dir) / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    # 각 이미지 쌍에 대해 처리
    for i, (original_path, bg_removed_path) in enumerate(matching_pairs):
        original_name = Path(original_path).name
        bg_removed_name = Path(bg_removed_path).name
        print(f"  [{i+1:2d}/{len(matching_pairs)}] {original_name} vs {bg_removed_name} 비교 중...")
        
        try:
            # 비교 분석 수행
            comparison_result = inference_and_cam_comparison(
                model, original_path, bg_removed_path, target_layer, class_names, device, cfg
            )
            
            if comparison_result is None:
                continue
            
            # 결과 시각화 및 저장
            output_filename = f"{comparison_result['patient_id']}_comparison.png"
            output_path = class_output_dir / output_filename
            
            create_comparison_visualization(comparison_result, output_path, class_names)
            
            # 결과 요약 출력
            original_result = comparison_result['original']
            bg_removed_result = comparison_result['bg_removed']
            
            print(f"    - 원본: {original_result['predicted_class_name']} (신뢰도: {original_result['confidence']:.3f})")
            print(f"    - 배경제거: {bg_removed_result['predicted_class_name']} (신뢰도: {bg_removed_result['confidence']:.3f})")
            
            prediction_match = "일치" if original_result['predicted_class'] == bg_removed_result['predicted_class'] else "불일치"
            confidence_diff = abs(original_result['confidence'] - bg_removed_result['confidence'])
            print(f"    - 예측 결과: {prediction_match}, 신뢰도 차이: {confidence_diff:.3f}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"    ✗ 오류 발생: {e}")
            continue
    
    return processed_count

def create_summary_report(output_dir, total_processed, target_classes):
    """CAM 비교 결과 요약 리포트를 생성합니다."""
    report_path = Path(output_dir) / "cam_comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 원본 vs 배경제거 CAM 비교 결과 요약 ===\n\n")
        
        f.write("처리 결과:\n")
        f.write(f"  - 총 처리된 이미지 쌍 수: {total_processed}\n")
        f.write(f"  - 처리된 클래스: {', '.join(target_classes)}\n\n")
        
        f.write("출력 구조:\n")
        for class_name in target_classes:
            f.write(f"  - {class_name}/: 클래스별 비교 결과 이미지들\n")
        f.write(f"  - cam_comparison_report.txt: 이 리포트 파일\n\n")
        
        f.write(f"출력 디렉토리: {output_dir}\n")
        f.write(f"각 이미지 쌍별로 다음 정보가 포함된 비교 시각화 파일이 생성됩니다:\n")
        f.write(f"  - 원본 이미지 vs 배경제거 이미지\n")
        f.write(f"  - 각각의 Class Activation Map (CAM)\n")
        f.write(f"  - 각각의 CAM 오버레이\n")
        f.write(f"  - 예측 결과 및 신뢰도 비교 통계\n")
    
    print(f"CAM 비교 리포트가 생성되었습니다: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='원본과 배경제거된 이미지들의 CAM을 비교합니다.')
    parser.add_argument('--cfg', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--sampled_dir', type=str, default='sampled_images_bg_removed',
                       help='샘플링된 이미지들이 저장된 디렉토리 (original과 background_removed 폴더 포함)')
    parser.add_argument('--output_dir', type=str, default='cam_comparison_results',
                       help='결과를 저장할 디렉토리')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='학습된 모델 경로')
    parser.add_argument('--target_classes', nargs='+', default=['oa', 'normal'],
                       help='처리할 클래스들')
    parser.add_argument('--device', type=str, default='cuda',
                       help='사용할 디바이스 (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=== 원본 vs 배경제거 CAM 비교 시작 ===")
    print(f"설정 파일: {args.cfg}")
    print(f"샘플링된 이미지 디렉토리: {args.sampled_dir}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"모델 경로: {args.model_path}")
    print(f"처리할 클래스: {args.target_classes}")
    print(f"사용 디바이스: {args.device}")
    print()
    
    # 설정 업데이트
    update_config(cfg, args)
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    try:
        print("모델 로딩 중...")
        model_name = getattr(cfg.MODEL, 'NAME', 'VGG19_BN')
        print(f"모델 이름: {model_name}")
        
        model = create_model(model_name, cfg)
        
        # 체크포인트 로드
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.float()
        model.eval()
        
        print("모델 로딩 완료!")
        print(f"최고 검증 손실: {checkpoint['best_val_loss']:.4f}")
        
    except Exception as e:
        print(f"Error: 모델 로딩 실패 - {e}")
        return
    
    # 클래스 이름 정의 (설정에서 가져오기)
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
    class_names = target_classes if target_classes else ['oa', 'normal']
    print(f"클래스 이름: {class_names}")
    
    # 타겟 레이어 결정
    target_layer = get_gradcam_target_layer(model)
    print(f"GradCAM 타겟 레이어: {target_layer}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 샘플링된 이미지 디렉토리 확인
    sampled_dir = Path(args.sampled_dir)
    if not sampled_dir.exists():
        print(f"Error: 샘플링된 이미지 디렉토리를 찾을 수 없습니다: {args.sampled_dir}")
        return
    
    # 지정된 클래스들 처리
    total_processed = 0
    for class_name in args.target_classes:
        processed_count = process_class_comparison(
            model, class_name, sampled_dir, target_layer, class_names, device, output_dir, cfg
        )
        total_processed += processed_count
    
    print(f"\n총 {total_processed}개 이미지 쌍 처리 완료")
    
    # 요약 리포트 생성
    create_summary_report(output_dir, total_processed, args.target_classes)
    
    print(f"\n=== 원본 vs 배경제거 CAM 비교 완료 ===")
    print(f"결과 저장 디렉토리: {output_dir}")

if __name__ == "__main__":
    main() 
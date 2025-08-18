#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
샘플링된 OA, Normal 이미지들로 Inference를 수행하고 원본/배경제거 이미지의 CAM을 비교하는 스크립트
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

def get_image_size_from_config(cfg):
    """설정에서 이미지 크기를 가져옵니다."""
    image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', [224, 224])
    if isinstance(image_size, int):
        return (image_size, image_size)
    elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return tuple(image_size)
    else:
        print(f"Warning: Invalid IMAGE_SIZE format: {image_size}, using default (224, 224)")
        return (224, 224)

def load_and_preprocess_image(image_path, cfg):
    """이미지를 로드하고 전처리합니다."""
    # 설정에서 이미지 크기 가져오기
    target_size = get_image_size_from_config(cfg)
    
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
    
    # 설정에서 정규화 파라미터 가져오기
    mean = getattr(cfg.DATASET, 'MEAN', [0.485, 0.456, 0.406])
    std = getattr(cfg.DATASET, 'STD', [0.229, 0.224, 0.225])
    
    # 전처리 파이프라인
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
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

def create_comparison_visualization(original_result, bg_removed_result, output_path, class_names, original_path, bg_removed_path):
    """원본과 배경제거 이미지의 CAM 비교 시각화를 생성합니다."""
    # 3x2 서브플롯 생성
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle(f'Original vs Background Removed - CAM Comparison\n{Path(original_path).name}', 
                 fontsize=18, fontweight='bold')
    
    # 첫 번째 행: 원본 이미지들
    axes[0, 0].imshow(original_result['original_image'])
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(bg_removed_result['original_image'])
    axes[0, 1].set_title('Background Removed Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 두 번째 행: CAM들
    im1 = axes[1, 0].imshow(original_result['cam'], cmap='jet')
    axes[1, 0].set_title('Original CAM', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(bg_removed_result['cam'], cmap='jet')
    axes[1, 1].set_title('Background Removed CAM', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 세 번째 행: CAM 오버레이들
    try:
        # 원본 이미지 CAM 오버레이
        original_normalized = original_result['original_image'].astype(np.float32) / 255.0
        original_overlay = show_cam_on_image(original_normalized, original_result['cam'], use_rgb=True)
        axes[2, 0].imshow(original_overlay)
        axes[2, 0].set_title('Original CAM Overlay', fontsize=14, fontweight='bold')
    except Exception as e:
        print(f"  Warning: 원본 CAM 오버레이 생성 실패: {e}")
        axes[2, 0].imshow(original_result['original_image'])
        axes[2, 0].set_title('Original Image (Overlay Failed)', fontsize=14)
    axes[2, 0].axis('off')
    
    try:
        # 배경제거 이미지 CAM 오버레이
        bg_removed_normalized = bg_removed_result['original_image'].astype(np.float32) / 255.0
        bg_removed_overlay = show_cam_on_image(bg_removed_normalized, bg_removed_result['cam'], use_rgb=True)
        axes[2, 1].imshow(bg_removed_overlay)
        axes[2, 1].set_title('Background Removed CAM Overlay', fontsize=14, fontweight='bold')
    except Exception as e:
        print(f"  Warning: 배경제거 CAM 오버레이 생성 실패: {e}")
        axes[2, 1].imshow(bg_removed_result['original_image'])
        axes[2, 1].set_title('Background Removed Image (Overlay Failed)', fontsize=14)
    axes[2, 1].axis('off')
    
    # 예측 결과 텍스트 추가
    prediction_text = f"""
    Original Image Prediction:
    - Class: {original_result['predicted_class_name']}
    - Confidence: {original_result['confidence']:.3f}
    
    Background Removed Prediction:
    - Class: {bg_removed_result['predicted_class_name']}
    - Confidence: {bg_removed_result['confidence']:.3f}
    
    Prediction Match: {'✓' if original_result['predicted_class'] == bg_removed_result['predicted_class'] else '✗'}
    Confidence Difference: {abs(original_result['confidence'] - bg_removed_result['confidence']):.3f}
    """
    
    fig.text(0.02, 0.02, prediction_text, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 비교 시각화 저장 완료: {output_path}")

def process_image_pair(model, original_path, bg_removed_path, target_layer, class_names, device, output_dir, cfg):
    """원본과 배경제거 이미지 쌍에 대해 CAM 비교를 수행합니다."""
    try:
        # 원본 이미지 처리
        original_result = inference_and_cam(
            model, str(original_path), target_layer, class_names, device, cfg
        )
        
        # 배경제거 이미지 처리
        bg_removed_result = inference_and_cam(
            model, str(bg_removed_path), target_layer, class_names, device, cfg
        )
        
        # 비교 시각화 생성
        output_filename = f"{original_path.stem}_cam_comparison.png"
        output_path = output_dir / output_filename
        
        create_comparison_visualization(
            original_result, bg_removed_result, output_path, class_names, 
            str(original_path), str(bg_removed_path)
        )
        
        # 결과 요약 반환
        comparison_info = {
            'original_prediction': original_result['predicted_class_name'],
            'original_confidence': original_result['confidence'],
            'bg_removed_prediction': bg_removed_result['predicted_class_name'],
            'bg_removed_confidence': bg_removed_result['confidence'],
            'prediction_match': original_result['predicted_class'] == bg_removed_result['predicted_class'],
            'confidence_diff': abs(original_result['confidence'] - bg_removed_result['confidence'])
        }
        
        return comparison_info
        
    except Exception as e:
        print(f"    ✗ 오류 발생: {e}")
        return None

def process_class_images(model, class_name, sampled_dir, target_layer, class_names, device, output_dir, cfg):
    """특정 클래스의 모든 이미지에 대해 원본/배경제거 CAM 비교를 수행합니다."""
    print(f"\n=== 클래스 '{class_name}' 처리 중 ===")
    
    # 원본과 배경제거 디렉토리 경로
    original_dir = sampled_dir / "original" / class_name
    bg_removed_dir = sampled_dir / "background_removed" / class_name
    
    # 디렉토리 존재 확인
    if not original_dir.exists():
        print(f"  Warning: 원본 이미지 디렉토리를 찾을 수 없습니다: {original_dir}")
        return 0, []
    
    if not bg_removed_dir.exists():
        print(f"  Warning: 배경제거 이미지 디렉토리를 찾을 수 없습니다: {bg_removed_dir}")
        return 0, []
    
    # 이미지 파일들 찾기
    original_files = list(original_dir.glob('*_original.jpg'))
    bg_removed_files = list(bg_removed_dir.glob('*_bg_removed.jpg'))
    
    print(f"  원본 이미지 {len(original_files)}개, 배경제거 이미지 {len(bg_removed_files)}개 발견")
    
    # 결과 저장 디렉토리 생성
    class_output_dir = Path(output_dir) / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    comparison_results = []
    
    # 이미지 쌍 매칭 및 처리
    for original_file in original_files:
        # 매칭되는 배경제거 파일 찾기
        base_name = original_file.stem.replace('_original', '')
        bg_removed_file = bg_removed_dir / f"{base_name}_bg_removed.jpg"
        
        if not bg_removed_file.exists():
            print(f"  Warning: 매칭되는 배경제거 이미지를 찾을 수 없습니다: {bg_removed_file}")
            continue
        
        print(f"  [{processed_count+1}] {base_name} 처리 중...")
        
        # 이미지 쌍 처리
        comparison_info = process_image_pair(
            model, original_file, bg_removed_file, target_layer, 
            class_names, device, class_output_dir, cfg
        )
        
        if comparison_info:
            comparison_info['base_name'] = base_name
            comparison_results.append(comparison_info)
            processed_count += 1
            
            # 결과 요약 출력
            print(f"    - 원본: {comparison_info['original_prediction']} ({comparison_info['original_confidence']:.3f})")
            print(f"    - 배경제거: {comparison_info['bg_removed_prediction']} ({comparison_info['bg_removed_confidence']:.3f})")
            print(f"    - 예측 일치: {'✓' if comparison_info['prediction_match'] else '✗'}")
    
    return processed_count, comparison_results

def create_summary_report(output_dir, all_comparison_results, target_classes):
    """CAM 비교 결과 요약 리포트를 생성합니다."""
    report_path = Path(output_dir) / "cam_comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 원본 vs 배경제거 CAM 비교 결과 요약 ===\n\n")
        
        total_processed = sum(len(results) for results in all_comparison_results.values())
        f.write(f"총 처리된 이미지 쌍: {total_processed}개\n")
        f.write(f"처리된 클래스: {', '.join(target_classes)}\n\n")
        
        for class_name, results in all_comparison_results.items():
            if not results:
                continue
                
            f.write(f"=== 클래스: {class_name} ===\n")
            f.write(f"처리된 이미지 쌍: {len(results)}개\n")
            
            # 예측 일치 통계
            matches = sum(1 for r in results if r['prediction_match'])
            match_rate = matches / len(results) * 100
            f.write(f"예측 일치율: {matches}/{len(results)} ({match_rate:.1f}%)\n")
            
            # 신뢰도 차이 통계
            confidence_diffs = [r['confidence_diff'] for r in results]
            avg_confidence_diff = np.mean(confidence_diffs)
            f.write(f"평균 신뢰도 차이: {avg_confidence_diff:.3f}\n")
            f.write(f"최대 신뢰도 차이: {max(confidence_diffs):.3f}\n")
            f.write(f"최소 신뢰도 차이: {min(confidence_diffs):.3f}\n\n")
            
            # 개별 결과
            f.write("개별 결과:\n")
            for i, result in enumerate(results):
                f.write(f"  {i+1:2d}. {result['base_name']}\n")
                f.write(f"      원본: {result['original_prediction']} ({result['original_confidence']:.3f})\n")
                f.write(f"      배경제거: {result['bg_removed_prediction']} ({result['bg_removed_confidence']:.3f})\n")
                f.write(f"      일치: {'✓' if result['prediction_match'] else '✗'}, 차이: {result['confidence_diff']:.3f}\n")
            f.write("\n")
        
        f.write(f"출력 디렉토리: {output_dir}\n")
        f.write(f"각 클래스별로 비교 시각화 파일들이 생성되었습니다.\n")
    
    print(f"CAM 비교 리포트가 생성되었습니다: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='원본과 배경제거 이미지의 CAM을 비교합니다.')
    parser.add_argument('--cfg', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--sampled_dir', type=str, default='sampled_images_bg_removed',
                       help='샘플링된 이미지들이 저장된 디렉토리')
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
    
    # 설정에서 이미지 크기 확인
    image_size = get_image_size_from_config(cfg)
    print(f"설정된 이미지 크기: {image_size}")
    
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
    all_comparison_results = {}
    
    for class_name in args.target_classes:
        processed_count, comparison_results = process_class_images(
            model, class_name, sampled_dir, target_layer, class_names, device, output_dir, cfg
        )
        total_processed += processed_count
        all_comparison_results[class_name] = comparison_results
    
    print(f"\n총 {total_processed}개 이미지 쌍 처리 완료")
    
    # 요약 리포트 생성
    create_summary_report(output_dir, all_comparison_results, args.target_classes)
    
    print(f"\n=== 원본 vs 배경제거 CAM 비교 완료 ===")
    print(f"결과 저장 디렉토리: {output_dir}")

if __name__ == "__main__":
    main() 
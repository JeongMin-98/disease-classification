#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
샘플링된 OA, Normal 이미지들로 Inference를 수행하고 CAM을 생성하는 스크립트
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

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """이미지를 로드하고 전처리합니다."""
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
    input_tensor, original_image, original_size = load_and_preprocess_image(image_path)
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

def create_visualization(result_info, output_path, class_names):
    """결과를 시각화하여 저장합니다."""
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Inference & CAM Results - {Path(result_info["original_image_path"]).name}', 
                 fontsize=16, fontweight='bold')
    
    # 원본 이미지
    axes[0, 0].imshow(result_info['original_image'])
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # CAM
    axes[0, 1].imshow(result_info['cam'], cmap='jet')
    axes[0, 1].set_title('Class Activation Map (CAM)', fontsize=12)
    axes[0, 1].axis('off')
    
    # CAM 오버레이
    try:
        # 원본 이미지를 0-1 범위로 정규화
        original_normalized = result_info['original_image'].astype(np.float32) / 255.0
        overlay = show_cam_on_image(original_normalized, result_info['cam'], use_rgb=True)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('CAM Overlay', fontsize=12)
    except Exception as e:
        print(f"  Warning: CAM 오버레이 생성 실패: {e}")
        axes[1, 0].imshow(result_info['original_image'])
        axes[1, 0].set_title('Original Image (Overlay Failed)', fontsize=12)
    axes[1, 0].axis('off')
    
    # 예측 결과 및 확률
    class_probs = result_info['probabilities']
    
    # 이진 분류: 두 클래스의 확률만 표시
    if len(class_probs) == 2:
        # 두 클래스의 확률을 막대 그래프로 표시
        y_pos = np.arange(2)
        labels = [class_names[0], class_names[1]]
        
        # 막대 그래프 생성 (각 클래스별로 다른 색상)
        colors = ['lightcoral' if i == result_info['predicted_class'] else 'lightblue' for i in range(2)]
        axes[1, 1].barh(y_pos, class_probs, color=colors)
        
        # ticks 설정
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(labels)
        
        axes[1, 1].set_xlabel('Probability')
        axes[1, 1].set_title('Binary Classification Probabilities')
        axes[1, 1].set_xlim(0, 1)
        
        # 확률 값 표시
        for i, prob in enumerate(class_probs):
            axes[1, 1].text(prob + 0.01, i, f'{prob:.3f}', 
                            va='center', fontweight='bold')
    else:
        # 다중 클래스인 경우 (기존 로직 유지)
        # 유효한 확률이 있는 클래스들만 필터링
        valid_indices = np.where(class_probs > 0)[0]
        if len(valid_indices) == 0:
            # 모든 확률이 0인 경우
            top_k = min(3, len(class_names))
            top_indices = np.arange(top_k)
            top_probs = np.zeros(top_k)
        else:
            top_k = min(3, len(valid_indices))
            top_indices = valid_indices[np.argsort(class_probs[valid_indices])[-top_k:][::-1]]
            top_probs = class_probs[top_indices]
        
        # y_pos와 labels 준비
        y_pos = np.arange(top_k)
        labels = [class_names[i] if i < len(class_names) else f'Class_{i}' for i in top_indices]
        
        # 막대 그래프 생성
        axes[1, 1].barh(y_pos, top_probs, color='skyblue')
        
        # ticks 설정 (안전하게)
        if len(y_pos) > 0:
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(labels)
        
        axes[1, 1].set_xlabel('Probability')
        axes[1, 1].set_title(f'Top {top_k} Predictions')
        axes[1, 1].set_xlim(0, 1)
        
        # 확률 값 표시
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            if prob > 0:  # 확률이 0보다 큰 경우에만 표시
                axes[1, 1].text(prob + 0.01, i, f'{prob:.3f}', 
                                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 시각화 저장 완료: {output_path}")

def process_class_images(model, class_dir, target_layer, class_names, device, output_dir, cfg):
    """특정 클래스의 모든 이미지에 대해 Inference와 CAM을 수행합니다."""
    class_name = class_dir.name
    print(f"\n=== 클래스 '{class_name}' 처리 중 ===")
    
    # 이미지 파일들 찾기
    image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
    
    if not image_files:
        print(f"  Warning: 클래스 '{class_name}'에 이미지가 없습니다.")
        return 0
    
    print(f"  총 {len(image_files)}개 이미지 발견")
    
    # 결과 저장 디렉토리 생성
    class_output_dir = Path(output_dir) / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    # 각 이미지에 대해 처리
    for i, image_path in enumerate(image_files):
        print(f"  [{i+1:2d}/{len(image_files)}] {image_path.name} 처리 중...")
        
        try:
            # Inference 및 CAM 생성
            result_info = inference_and_cam(
                model, str(image_path), target_layer, class_names, device, cfg
            )
            
            # 원본 이미지 경로 추가
            result_info['original_image_path'] = str(image_path)
            
            # 결과 시각화 및 저장
            output_filename = f"{image_path.stem}_inference_cam.png"
            output_path = class_output_dir / output_filename
            
            create_visualization(result_info, output_path, class_names)
            
            # 결과 요약 출력
            print(f"    - 예측 클래스: {result_info['predicted_class_name']}")
            print(f"    - 신뢰도: {result_info['confidence']:.3f}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"    ✗ 오류 발생: {e}")
            continue
    
    return processed_count

def create_summary_report(output_dir, total_processed, target_classes):
    """Inference & CAM 결과 요약 리포트를 생성합니다."""
    report_path = Path(output_dir) / "inference_cam_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Inference & CAM 결과 요약 ===\n\n")
        
        f.write("처리 결과:\n")
        f.write(f"  - 총 처리된 이미지 수: {total_processed}\n")
        f.write(f"  - 처리된 클래스: {', '.join(target_classes)}\n\n")
        
        f.write("출력 구조:\n")
        for class_name in target_classes:
            f.write(f"  - {class_name}/: 클래스별 결과 이미지들\n")
        f.write(f"  - inference_cam_report.txt: 이 리포트 파일\n\n")
        
        f.write(f"출력 디렉토리: {output_dir}\n")
        f.write(f"각 이미지별로 다음 정보가 포함된 시각화 파일이 생성됩니다:\n")
        f.write(f"  - 원본 이미지\n")
        f.write(f"  - Class Activation Map (CAM)\n")
        f.write(f"  - CAM 오버레이\n")
        f.write(f"  - Top 3 예측 확률\n")
    
    print(f"Inference & CAM 리포트가 생성되었습니다: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='샘플링된 이미지들로 Inference와 CAM을 수행합니다.')
    parser.add_argument('--cfg', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--sampled_dir', type=str, default='sampled_images_bg_removed',
                       help='샘플링된 이미지들이 저장된 디렉토리')
    parser.add_argument('--output_dir', type=str, default='inference_cam_results',
                       help='결과를 저장할 디렉토리')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='학습된 모델 경로')
    parser.add_argument('--target_classes', nargs='+', default=['oa', 'normal'],
                       help='처리할 클래스들')
    parser.add_argument('--device', type=str, default='cuda',
                       help='사용할 디바이스 (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=== Inference & CAM 생성 시작 ===")
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
    for class_name in target_classes:
        class_dir = sampled_dir / class_name
        if class_dir.exists():
            processed_count = process_class_images(
                model, class_dir, target_layer, class_names, device, output_dir, cfg
            )
            total_processed += processed_count
        else:
            print(f"Warning: 클래스 '{class_name}' 디렉토리를 찾을 수 없습니다.")
    
    print(f"\n총 {total_processed}개 이미지 처리 완료")
    
    # 요약 리포트 생성
    create_summary_report(output_dir, total_processed, target_classes)
    
    print(f"\n=== Inference & CAM 생성 완료 ===")
    print(f"결과 저장 디렉토리: {output_dir}")

if __name__ == "__main__":
    main() 
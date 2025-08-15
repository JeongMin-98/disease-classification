#!/usr/bin/env python3
"""
GradCAM 분석 스크립트
저장된 best_model.pth를 로드하여 GradCAM을 수행합니다.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
import _init_path
from models.vgg19 import VGG
from dataset import MedicalImageDataset
from config import cfg, update_config
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def load_model(checkpoint_path, cfg, device):
    """체크포인트에서 모델을 로드합니다."""
    # 모델 초기화
    model = VGG(
        num_classes=cfg.MODEL.NUM_CLASSES,
        Target_Classes=cfg.DATASET.TARGET_CLASSES,
        freeze_layers=cfg.MODEL.FREEZE_LAYERS
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"모델이 {checkpoint_path}에서 로드되었습니다.")
    print(f"최고 검증 손실: {checkpoint['best_val_loss']:.4f}")
    
    return model

def get_gradcam_target_layer(model):
    """GradCAM을 위한 타겟 레이어를 반환합니다."""
    # VGG19의 마지막 convolutional layer (features의 마지막 레이어)
    target_layer = model.model.features[-1]
    return target_layer

def preprocess_image_for_gradcam(image_path, target_size=(224, 224)):
    """이미지를 GradCAM용으로 전처리합니다."""
    # 이미지 로드 및 리사이즈
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    
    # 정규화 (ImageNet 통계 사용)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 0-1 범위로 정규화
    image_normalized = image.astype(np.float32) / 255.0
    
    # ImageNet 통계로 정규화
    image_normalized = (image_normalized - mean) / std
    
    return image_normalized, image

def generate_gradcam(model, image_path, target_class=None, device='cuda'):
    """GradCAM을 생성합니다."""
    # 이미지 전처리
    input_image, original_image = preprocess_image_for_gradcam(image_path)
    
    # 모델 입력용 텐서 생성
    input_tensor = torch.from_numpy(input_image).unsqueeze(0).to(device)
    input_tensor = input_tensor.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    input_tensor = input_tensor.float()  # float32로 변환
    
    # 타겟 레이어 설정
    target_layer = get_gradcam_target_layer(model)
    
    # GradCAM 생성
    if target_class is None:
        # 모델 예측 결과 사용
        with torch.no_grad():
            output = model(input_tensor)
            # Binary classification의 경우 특별 처리
            if output.shape[1] == 1:
                # Binary classification: sigmoid 사용 (1개 출력)
                predicted_class = (torch.sigmoid(output) > 0.5).int().item()
                confidence = torch.sigmoid(output).max().item()
            else:
                # Multi-class classification: softmax 사용 (2개 이상 출력)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = F.softmax(output, dim=1).max().item()
    else:
        predicted_class = target_class
        confidence = None
    
    # GradCAM 생성 (최신 문법 사용)
    grayscale_cam = None  # 초기화
    try:
        targets = [ClassifierOutputTarget(predicted_class)]
        
        # GradCAM 초기화 및 실행
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
    except Exception as e:
        print(f"GradCAM 생성 오류: {e}")
        # 오류 발생 시 더미 히트맵 생성
        dummy_size = (224, 224)  # 기본 크기
        grayscale_cam = np.zeros(dummy_size)
    
    # 원본 이미지와 GradCAM 오버레이
    try:
        # 원본 이미지를 0-1 범위로 정규화
        original_normalized = original_image.astype(np.float32) / 255.0
        visualization = show_cam_on_image(original_normalized, grayscale_cam, use_rgb=True)
    except Exception as e:
        print(f"오버레이 생성 오류: {e}")
        # 오류 발생 시 원본 이미지만 사용
        visualization = original_image
    
    return visualization, predicted_class, confidence, original_image, grayscale_cam

def save_gradcam_results(visualization, original_image, grayscale_cam, predicted_class, 
                        confidence, image_path, output_dir):
    """GradCAM 결과를 저장합니다."""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}_gradcam.png")
    
    # 결과 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 원본 이미지
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GradCAM 히트맵
    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # 오버레이된 이미지
    axes[2].imshow(visualization)
    title = f'GradCAM Overlay\nPredicted: Class {predicted_class}'
    if confidence is not None:
        title += f'\nConfidence: {confidence:.3f}'
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"GradCAM 결과가 {output_path}에 저장되었습니다.")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='GradCAM 분석')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='체크포인트 파일 경로 (예: experiments/exp_name/best_model.pth)')
    parser.add_argument('--image', type=str, required=True,
                       help='분석할 이미지 경로')
    parser.add_argument('--config', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--output_dir', type=str, default='gradcam_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--target_class', type=int, default=None,
                       help='특정 클래스에 대한 GradCAM 생성 (기본값: 예측된 클래스)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='사용할 디바이스 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 설정 로드
    cfg.merge_from_file(args.config)
    cfg.freeze()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    model = load_model(args.checkpoint, cfg, device)
    
    # GradCAM 생성
    visualization, predicted_class, confidence, original_image, grayscale_cam = generate_gradcam(
        model, args.image, args.target_class, str(device)
    )
    
    # 결과 저장
    output_path = save_gradcam_results(
        visualization, original_image, grayscale_cam, predicted_class, 
        confidence, args.image, args.output_dir
    )
    
    print(f"\n분석 완료!")
    print(f"예측된 클래스: {predicted_class}")
    if confidence is not None:
        print(f"신뢰도: {confidence:.3f}")
    print(f"결과 저장 위치: {output_path}")

if __name__ == '__main__':
    main() 
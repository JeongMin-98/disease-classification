#!/usr/bin/env python3
"""
K-fold indices를 사용해서 각 fold의 best model로 batch CAM을 생성하는 스크립트
배치 단위로 처리하여 성능 향상
원본과 배경 제거된 이미지의 CAM 비교 기능 추가
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
import re
from pathlib import Path
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Subset
import time

from models import create as create_model
from dataset import MedicalImageDataset
from config import cfg, update_config

from utils.background_removal import BgRemovalConfig, process_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def apply_background_removal(image_path, args):
    """이미지에 배경 제거를 적용합니다."""
    try:
        # 임시 출력 경로 생성
        temp_dir = "/tmp/bg_removal"
        os.makedirs(temp_dir, exist_ok=True)
        
        input_filename = Path(image_path).name
        temp_output_path = os.path.join(temp_dir, f"bg_removed_{input_filename}")
        
        # 배경 제거 설정 (HSV 기반 배경 제거 사용)
        bg_config = BgRemovalConfig(
            method=args.bg_method,  # 커맨드 라인에서 지정된 방법 사용
            fixed_thresh=args.bg_thresh,
            percentile=2.0,  # percentile 방법의 기본값
            hsv_value_thresh=args.bg_hsv_thresh,  # HSV V 채널 임계값 (기본값: 15)
            protect_skin=args.bg_protect_skin,  # 피부 영역 보호
            protect_bone=args.bg_protect_bone,  # 뼈/관절 영역 보호
            morph_kernel=args.bg_morph_kernel,
            keep_largest_only=False if args.bg_method == "hsv_value" else True,  # HSV 방식에서는 형태학적 후처리 생략
            tight_crop=False,
            fill_value=0,
            min_object_area=args.bg_min_area,
            normalize_to_uint8=True
        )
        
        # 배경 제거 처리
        success, error_msg = process_image(image_path, temp_output_path, bg_config)
        
        if success:
            return temp_output_path
        else:
            print(f"배경 제거 실패: {error_msg}")
            return image_path  # 실패 시 원본 경로 반환
            
    except Exception as e:
        print(f"배경 제거 중 오류 발생: {e}")
        return image_path  # 오류 시 원본 경로 반환

def select_random_test_images(test_indices, dataset, num_samples=10):
    """테스트 세트에서 랜덤하게 이미지를 선택합니다."""
    selected_indices = random.sample(test_indices, min(num_samples, len(test_indices)))
    selected_info = []
    
    for idx in selected_indices:
        if idx < len(dataset.db_rec):
            db_record = dataset.db_rec[idx]
            selected_info.append({
                'index': idx,
                'image_path': db_record['file_path'],
                'label': db_record['label'],
                'patient_id': db_record.get('patient_id', 'unknown')
            })
    
    return selected_info

def parse_wandb_runs_from_log(log_file_path):
    """
    로그 파일에서 wandb run 폴더명을 파싱합니다.
    """
    if not Path(log_file_path).exists():
        print(f"로그 파일을 찾을 수 없습니다: {log_file_path}")
        return {}
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    wandb_runs = {}
    
    # 패턴: wandb: Find logs at: ./wandb/run-20250810_211304-i8hoewlc/logs
    run_pattern = r'wandb: Find logs at: \./wandb/(run-[a-zA-Z0-9_-]+)/logs'
    matches = re.findall(run_pattern, log_content)
    
    print(f"로그에서 {len(matches)}개의 wandb run 폴더를 발견했습니다.")
    
    # fold 순서대로 매핑 (run 순서가 fold 순서와 일치한다고 가정)
    for i, run_folder in enumerate(matches):
        if i < 7:  # 7-fold 가정
            run_id = run_folder.split('-')[-1]  # i8hoewlc 부분 추출
            wandb_runs[f'fold_{i}'] = {
                'run_name': f"fold{i}_run",
                'run_id': run_id,
                'run_folder': run_folder
            }
            print(f"Fold {i} wandb run 매핑: {run_folder} -> {run_id}")
    
    return wandb_runs

def load_model(checkpoint_path, cfg, device):
    """체크포인트에서 모델을 로드합니다."""
    model_name = getattr(cfg.MODEL, 'NAME', 'VGG19_BN')
    print(f"모델 이름: {model_name}")
    
    model = create_model(model_name, cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.float()
    model.eval()
    
    print(f"모델이 {checkpoint_path}에서 로드되었습니다.")
    print(f"최고 검증 손실: {checkpoint['best_val_loss']:.4f}")
    
    return model

def get_gradcam_target_layer(model):
    """GradCAM을 위한 타겟 레이어를 반환합니다."""
    model_name = getattr(cfg.MODEL, 'NAME', 'VGG19_BN')
    
    try:
        if model_name.startswith('VGG'):
            if hasattr(model.model, 'features'):
                target_layer = model.model.features[-1]
            else:
                raise AttributeError(f"VGG 모델에 features 속성이 없습니다: {type(model.model)}")
        elif model_name.startswith('ResNet'):
            if hasattr(model.model, 'layer4'):
                target_layer = model.model.layer4[-1]
            elif hasattr(model.model, 'layer3'):
                target_layer = model.model.layer3[-1]
            else:
                raise AttributeError(f"ResNet 모델에 layer4 또는 layer3 속성이 없습니다: {type(model.model)}")
        else:
            if hasattr(model.model, 'features'):
                target_layer = model.model.features[-1]
            else:
                raise AttributeError(f"알 수 없는 모델 구조: {type(model.model)}")
        
        return target_layer
        
    except Exception as e:
        print(f"GradCAM 타겟 레이어 선택 중 오류: {e}")
        return model.model

def preprocess_batch_for_gradcam(image_paths, cfg, target_size=None):
    """배치 단위로 이미지를 GradCAM용으로 전처리합니다."""
    batch_images = []
    batch_resized_images = []  # 모델 입력 크기로 리사이즈된 이미지들
    batch_original_sizes = []
    
    # cfg에서 이미지 크기 가져오기
    if target_size is None:
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
        if isinstance(image_size, int):
            target_size = (image_size, image_size)
        else:
            target_size = image_size
    
    # 정규화 파라미터
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for image_path in image_paths:
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]  # (height, width)
            
            # 모델 입력용으로 리사이즈
            image_resized = cv2.resize(image, target_size)
            
            # 0-1 범위로 정규화
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # ImageNet 통계로 정규화
            image_normalized = (image_normalized - mean) / std
            
            batch_images.append(image_normalized)
            batch_resized_images.append(image_resized)
            batch_original_sizes.append(original_size)
            
        except Exception as e:
            print(f"이미지 전처리 오류 ({image_path}): {e}")
            # 오류 발생 시 더미 이미지 생성
            dummy_image = np.zeros((*target_size, 3), dtype=np.float32)
            batch_images.append(dummy_image)
            batch_resized_images.append(dummy_image)
            batch_original_sizes.append(target_size)
    
    return np.array(batch_images), batch_resized_images, batch_original_sizes

def generate_batch_gradcam(model, image_paths, target_classes, cfg, device='cuda'):
    """배치 단위로 GradCAM을 생성합니다."""
    batch_size = len(image_paths)
    
    # 이미지 전처리
    input_images, resized_images, original_sizes = preprocess_batch_for_gradcam(image_paths, cfg)
    
    print(f"=== 배치 GradCAM 생성 과정 크기 정보 ===")
    print(f"1. 배치 크기: {batch_size}")
    print(f"2. 전처리 후 input_images 크기: {input_images.shape}")
    print(f"3. 모델 입력 크기로 리사이즈된 이미지들: {[img.shape for img in resized_images]}")
    
    # 모델 입력용 텐서 생성
    input_tensor = torch.from_numpy(input_images).to(device)
    input_tensor = input_tensor.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    input_tensor = input_tensor.float()  # float32로 변환
    
    print(f"4. 모델 입력 텐서 크기: {input_tensor.shape}")
    
    # 타겟 레이어 설정
    target_layer = get_gradcam_target_layer(model)
    
    # 모델 예측
    with torch.no_grad():
        outputs = model(input_tensor)
        print(f"5. 모델 출력 크기: {outputs.shape}")
        
        # 이진 분류 처리
        if outputs.shape[1] == 1:
            # 이진 분류 (sigmoid 출력)
            predicted_classes = (outputs.squeeze() > 0).long()
            confidences = torch.sigmoid(outputs.squeeze())
        else:
            # 다중 분류 (softmax 출력)
            predicted_classes = torch.argmax(outputs, dim=1)
            confidences = F.softmax(outputs, dim=1).max(dim=1)[0]
    
    # GradCAM 생성 (배치 단위)
    batch_gradcams = []
    try:
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            # 각 이미지별로 GradCAM 생성
            for i, target_class in enumerate(target_classes):
                try:
                    # target_class가 문자열인 경우 인덱스로 변환
                    if isinstance(target_class, str):
                        if target_class in ['oa', 'normal']:
                            target_class_idx = 0 if target_class == 'oa' else 1
                        else:
                            target_class_idx = 0  # 기본값
                    else:
                        target_class_idx = int(target_class)
                    
                    # 이진 분류의 경우 특별 처리
                    if outputs.shape[1] == 1:
                        # 이진 분류: sigmoid 출력에서 타겟 클래스에 대한 GradCAM 생성
                        targets = [ClassifierOutputTarget(0)]  # 첫 번째 출력 (sigmoid)
                        grayscale_cam = cam(input_tensor=input_tensor[i:i+1], targets=targets)
                        grayscale_cam = grayscale_cam[0, :]  # 첫 번째 차원 제거
                    else:
                        # 다중 분류: 특정 클래스에 대한 GradCAM
                        targets = [ClassifierOutputTarget(target_class_idx)]
                        grayscale_cam = cam(input_tensor=input_tensor[i:i+1], targets=targets)
                        grayscale_cam = grayscale_cam[0, :]  # 첫 번째 차원 제거
                    
                    batch_gradcams.append(grayscale_cam)
                    
                except Exception as e:
                    print(f"이미지 {i} GradCAM 생성 오류: {e}")
                    # 오류 발생 시 더미 히트맵 생성
                    image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
                    if isinstance(image_size, int):
                        dummy_size = (image_size, image_size)
                    else:
                        dummy_size = image_size
                    dummy_gradcam = np.zeros(dummy_size)
                    batch_gradcams.append(dummy_gradcam)
    
    except Exception as e:
        print(f"배치 GradCAM 생성 오류: {e}")
        # 오류 발생 시 모든 이미지에 대해 더미 히트맵 생성
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
        if isinstance(image_size, int):
            dummy_size = (image_size, image_size)
        else:
            dummy_size = image_size
        
        for _ in range(batch_size):
            dummy_gradcam = np.zeros(dummy_size)
            batch_gradcams.append(dummy_gradcam)
    
    print(f"6. 생성된 GradCAM 수: {len(batch_gradcams)}")
    print(f"7. GradCAM 크기: {[gc.shape for gc in batch_gradcams]}")
    print(f"=== 배치 GradCAM 생성 완료 ===\n")
    
    return batch_gradcams, predicted_classes, confidences, resized_images, original_sizes

def save_comparison_gradcam_results(original_gradcam, bg_removed_gradcam, original_image, bg_removed_image,
                                  predicted_class, confidence, true_class, image_path, output_dir, idx, cfg, 
                                  patient_id=None, is_correct=None):
    """원본과 배경 제거된 이미지의 GradCAM 비교 결과를 저장합니다."""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 기본값 설정
    if patient_id is None:
        patient_id = 'unknown'
    if is_correct is None:
        is_correct = True
    
    # cfg에서 모델 입력 크기 가져오기
    model_input_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
    if isinstance(model_input_size, int):
        model_input_size = (model_input_size, model_input_size)
    
    # 예측 클래스 이름 변환
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    # 파일명 생성
    image_name = Path(image_path).stem
    accuracy_status = "correct" if is_correct else "incorrect"
    output_path = os.path.join(output_dir, f"{idx:03d}_{image_name}_{accuracy_status}_comparison.png")
    
    # 예측 클래스 이름 변환
    if isinstance(predicted_class, int):
        pred_class_name = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)
    else:
        pred_class_name = str(predicted_class)
    
    # true_class가 이미 문자열인지 확인
    if isinstance(true_class, int):
        true_class_name = class_names[true_class] if true_class < len(class_names) else str(true_class)
    else:
        true_class_name = true_class
    
    # 확률 계산
    confidence_val = confidence.item() if hasattr(confidence, 'item') else confidence
    
    if len(class_names) == 2:
        if confidence_val <= 0.5:
            disease_probability = 1 - confidence_val
            normal_probability = confidence_val
        else:
            disease_probability = 1 - confidence_val
            normal_probability = confidence_val
    else:
        disease_probability = confidence_val
        normal_probability = 1 - confidence_val
    
    # 판독문 생성
    report_text = f"Patient ID: {patient_id}\n"
    report_text += f"True Diagnosis: {true_class_name.upper()}\n"
    report_text += f"Model Prediction: {pred_class_name.upper()}\n"
    report_text += f"Confidence: {confidence_val:.3f}\n"
    if len(class_names) == 2:
        report_text += f"{class_names[0].upper()} Probability: {disease_probability:.3f}\n"
        report_text += f"{class_names[1].upper()} Probability: {normal_probability:.3f}\n"
    report_text += f"Status: {'CORRECT' if is_correct else 'INCORRECT'}\n"
    report_text += f"Comparison: Original vs Background Removed"
    
    # GradCAM 오버레이 생성
    try:
        # 원본 이미지 오버레이
        original_normalized = original_image.astype(np.float32) / 255.0
        original_visualization = show_cam_on_image(original_normalized, original_gradcam, use_rgb=True)
        
        # 배경 제거된 이미지 오버레이
        bg_removed_normalized = bg_removed_image.astype(np.float32) / 255.0
        bg_removed_visualization = show_cam_on_image(bg_removed_normalized, bg_removed_gradcam, use_rgb=True)
    except Exception as e:
        print(f"오버레이 생성 오류: {e}")
        original_visualization = original_image
        bg_removed_visualization = bg_removed_image
    
    # 결과 시각화 (2x3 레이아웃)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 첫 번째 행: 원본 이미지
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title(f'Original Image\nTrue: {true_class_name.upper()}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_gradcam, cmap='jet')
    axes[0, 1].set_title(f'Original GradCAM\nRange: [{original_gradcam.min():.3f}, {original_gradcam.max():.3f}]')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(original_visualization)
    axes[0, 2].set_title(f'Original Overlay\nPred: {pred_class_name.upper()}\nConf: {confidence_val:.3f}')
    axes[0, 2].axis('off')
    
    # 두 번째 행: 배경 제거된 이미지
    axes[1, 0].imshow(bg_removed_image)
    axes[1, 0].set_title(f'Background Removed\nTrue: {true_class_name.upper()}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(bg_removed_gradcam, cmap='jet')
    axes[1, 1].set_title(f'BG Removed GradCAM\nRange: [{bg_removed_gradcam.min():.3f}, {bg_removed_gradcam.max():.3f}]')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(bg_removed_visualization)
    axes[1, 2].set_title(f'BG Removed Overlay\nPred: {pred_class_name.upper()}\nConf: {confidence_val:.3f}')
    axes[1, 2].axis('off')
    
    # 판독문을 전체 플롯 중앙 아래에 추가
    fig.text(0.5, 0.02, report_text, ha='center', va='bottom', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def save_batch_gradcam_results(batch_gradcams, resized_images, predicted_classes, confidences, 
                             true_classes, image_paths, output_dir, start_idx, cfg, patient_ids=None, 
                             is_corrects=None, batch_size=None):
    """배치 단위로 GradCAM 결과를 저장합니다."""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 기본값 설정
    if patient_ids is None:
        patient_ids = ['unknown'] * len(image_paths)
    if is_corrects is None:
        is_corrects = [True] * len(image_paths)
    if batch_size is None:
        batch_size = len(image_paths)
    
    # cfg에서 모델 입력 크기 가져오기
    model_input_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
    if isinstance(model_input_size, int):
        model_input_size = (model_input_size, model_input_size)
    
    # 예측 클래스 이름 변환 - INCLUDE_CLASSES 사용
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    saved_paths = []
    
    # 각 이미지별로 결과 저장
    for i in range(len(image_paths)):
        try:
            # 파일명 생성 (정확도 정보 포함)
            image_name = Path(image_paths[i]).stem
            accuracy_status = "correct" if is_corrects[i] else "incorrect"
            output_path = os.path.join(output_dir, f"{start_idx + i:03d}_{image_name}_{accuracy_status}_gradcam.png")
            
            # 이미지 크기 정보
            image_size = resized_images[i].shape[:2]  
            gradcam_size = batch_gradcams[i].shape[:2]  
            
            # 예측 클래스 이름 변환
            if isinstance(predicted_classes[i], int):
                pred_class_name = class_names[predicted_classes[i]] if predicted_classes[i] < len(class_names) else str(predicted_classes[i])
            else:
                pred_class_name = str(predicted_classes[i])
            
            # true_class가 이미 문자열인지 확인
            if isinstance(true_classes[i], int):
                true_class_name = class_names[true_classes[i]] if true_classes[i] < len(class_names) else str(true_classes[i])
            else:
                true_class_name = true_classes[i]
            
            # 이진 분류에서 각 클래스의 확률을 명확하게 계산
            confidence = confidences[i].item() if hasattr(confidences[i], 'item') else confidences[i]
            
            if len(class_names) == 2:
                # 이진 분류에서 각 클래스의 확률을 명확하게 표시
                if confidence <= 0.5:
                    disease_probability = 1 - confidence  # OA일 확률 (높음)
                    normal_probability = confidence       # Normal일 확률 (낮음)
                    print(f"Binary classification: sigmoid {confidence:.3f} ≤ 0.5 -> Disease present ({class_names[0]})")
                else:
                    disease_probability = 1 - confidence  # OA일 확률 (낮음)
                    normal_probability = confidence       # Normal일 확률 (높음)
                    print(f"Binary classification: sigmoid {confidence:.3f} > 0.5 -> No disease ({class_names[1]})")
            else:
                disease_probability = confidence
                normal_probability = 1 - confidence
            
            # 판독문 생성
            report_text = f"Patient ID: {patient_ids[i] or 'Unknown'}\n"
            report_text += f"True Diagnosis: {true_class_name.upper()}\n"
            report_text += f"Model Prediction: {pred_class_name.upper()}\n"
            report_text += f"Confidence: {confidence:.3f}\n"
            if len(class_names) == 2:
                report_text += f"{class_names[0].upper()} Probability: {disease_probability:.3f}\n"
                report_text += f"{class_names[1].upper()} Probability: {normal_probability:.3f}\n"
            report_text += f"Status: {'CORRECT' if is_corrects[i] else 'INCORRECT'}\n"
            report_text += f"Image Size: {image_size[1]}x{image_size[0]} (Unified to training size)\n"
            report_text += f"Model Input: {model_input_size[0]}x{model_input_size[1]}"
            
            # GradCAM 오버레이 생성
            try:
                # 리사이즈된 이미지를 0-1 범위로 정규화
                resized_normalized = resized_images[i].astype(np.float32) / 255.0
                visualization = show_cam_on_image(resized_normalized, batch_gradcams[i], use_rgb=True)
            except Exception as e:
                print(f"오버레이 생성 오류: {e}")
                visualization = resized_images[i]
            
            # 결과 시각화
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 학습 크기로 리사이즈된 이미지
            axes[0].imshow(resized_images[i])
            axes[0].set_title(f'Resized Original Image (Training Size)\nTrue: {true_class_name.upper()}\nSize: {image_size[1]}x{image_size[0]}')
            axes[0].axis('off')
            
            # GradCAM 히트맵 (학습 크기)
            axes[1].imshow(batch_gradcams[i], cmap='jet')
            axes[1].set_title(f'GradCAM Heatmap (Training Size)\nTarget: {true_class_name.upper()}\nSize: {gradcam_size[1]}x{gradcam_size[0]}\nRange: [{batch_gradcams[i].min():.3f}, {batch_gradcams[i].max():.3f}]')
            axes[1].axis('off')
            
            # 오버레이된 이미지 (학습 크기)
            axes[2].imshow(visualization)
            title = f'GradCAM Overlay (Training Size)\nTrue: {true_class_name.upper()} | Pred: {pred_class_name.upper()}'
            title += f'\nConfidence: {confidence:.3f}'
            title += f'\nStatus: {"CORRECT" if is_corrects[i] else "INCORRECT"}'
            title += f'\nSize: {image_size[1]}x{image_size[0]} (All images unified size)'
            axes[2].set_title(title)
            axes[2].axis('off')
            
            # 판독문을 전체 플롯 중앙 아래에 추가
            fig.text(0.5, 0.02, report_text, ha='center', va='bottom', 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # 판독문을 위한 공간 확보
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(output_path)
            
        except Exception as e:
            print(f"이미지 {i} 저장 오류: {e}")
            saved_paths.append(None)
    
    return saved_paths

def find_best_model_in_wandb(wandb_dir, fold_idx, wandb_runs=None):
    """wandb 폴더에서 특정 fold의 best model을 찾습니다."""
    best_model_path = None
    
    # wandb_runs에서 해당 fold의 run 정보 찾기
    fold_key = f'fold_{fold_idx}'
    if wandb_runs and fold_key in wandb_runs:
        run_folder = wandb_runs[fold_key]['run_folder']
        best_model_candidate = Path(wandb_dir) / run_folder / "files" / "best_model.pth"
        
        if best_model_candidate.exists():
            best_model_path = str(best_model_candidate)
            print(f"Fold {fold_idx}의 best model 발견 (wandb run): {best_model_path}")
            return best_model_path
        else:
            print(f"Fold {fold_idx}의 wandb run에서 best_model.pth를 찾을 수 없습니다: {best_model_candidate}")
    
    # fallback: 기존 방식으로 검색
    print(f"wandb run 정보가 없어 기존 방식으로 검색합니다...")
    for run_folder in Path(wandb_dir).glob("run-*"):
        run_name = run_folder.name
        
        # fold 정보가 포함된 run 폴더인지 확인
        if f"fold{fold_idx}" in run_name.lower() or f"fold_{fold_idx}" in run_name.lower():
            # best_model.pth 파일 찾기
            best_model_candidate = run_folder / "files" / "best_model.pth"
            if best_model_candidate.exists():
                best_model_path = str(best_model_candidate)
                print(f"Fold {fold_idx}의 best model 발견 (fallback): {best_model_path}")
                break
    
    return best_model_path

def generate_single_gradcam(model, image_path, true_class, cfg, device):
    """단일 이미지에 대해 GradCAM을 생성합니다."""
    try:
        # 이미지 전처리
        input_images, resized_images, _ = preprocess_batch_for_gradcam([image_path], cfg)
        
        # 모델 입력용 텐서 생성
        input_tensor = torch.from_numpy(input_images).to(device)
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        input_tensor = input_tensor.float()
        
        # 타겟 레이어 설정
        target_layer = get_gradcam_target_layer(model)
        
        # 모델 예측
        with torch.no_grad():
            outputs = model(input_tensor)
            
            if outputs.shape[1] == 1:
                # 이진 분류
                predicted = (outputs.squeeze() > 0).long()
                confidence = torch.sigmoid(outputs.squeeze())
            else:
                # 다중 분류
                predicted = torch.argmax(outputs, dim=1)
                confidence = F.softmax(outputs, dim=1).max(dim=1)[0]
        
        # GradCAM 생성
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            if outputs.shape[1] == 1:
                targets = [ClassifierOutputTarget(0)]
            else:
                targets = [ClassifierOutputTarget(true_class)]
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
        
        # 값들을 안전하게 추출
        pred_val = predicted.item() if predicted.dim() == 0 else predicted[0].item()
        conf_val = confidence.item() if confidence.dim() == 0 else confidence[0].item()
        
        return grayscale_cam, pred_val, conf_val, resized_images[0]
        
    except Exception as e:
        print(f"GradCAM 생성 중 오류: {e}")
        # 더미 데이터 반환
        dummy_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
        if isinstance(dummy_size, int):
            dummy_size = (dummy_size, dummy_size)
        dummy_gradcam = np.zeros(dummy_size)
        dummy_image = np.zeros((*dummy_size, 3), dtype=np.uint8)
        return dummy_gradcam, 0, 0.5, dummy_image

def process_fold_comparison_gradcam(model, dataset, test_indices, cfg, device, output_dir, fold_idx, args, num_samples=10):
    """특정 fold에서 원본과 배경 제거된 이미지의 GradCAM 비교를 수행합니다."""
    print(f"\n=== Fold {fold_idx} 원본 vs 배경제거 GradCAM 비교 (샘플 수: {num_samples}) ===")
    
    # 랜덤 샘플 선택
    selected_images = select_random_test_images(test_indices, dataset, num_samples)
    
    if not selected_images:
        print(f"Fold {fold_idx}에서 처리할 이미지가 없습니다.")
        return {}
    
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    # 출력 디렉토리 생성
    comparison_output_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_output_dir, exist_ok=True)
    
    # 클래스별 폴더 생성 (옵션)
    if args.create_class_folders:
        for class_name in class_names:
            class_dir = os.path.join(comparison_output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        misclass_dir = os.path.join(comparison_output_dir, "misclassified")
        os.makedirs(misclass_dir, exist_ok=True)
    
    model.eval()
    processed_count = 0
    comparison_results = []
    
    for i, image_info in enumerate(tqdm(selected_images, desc=f"Fold {fold_idx} 비교 이미지 처리 중")):
        try:
            image_path = image_info['image_path']
            true_label = image_info['label']
            patient_id = image_info['patient_id']
            
            print(f"  처리 중: {Path(image_path).name} (true_label: {true_label})")
            
            # 원본 이미지 GradCAM 생성
            original_gradcam, original_pred, original_conf, original_resized = generate_single_gradcam(
                model, image_path, true_label, cfg, device
            )
            
            # 배경 제거된 이미지 생성
            bg_removed_path = apply_background_removal(image_path, args)
            
            # 배경 제거된 이미지 GradCAM 생성
            bg_removed_gradcam, bg_removed_pred, bg_removed_conf, bg_removed_resized = generate_single_gradcam(
                model, bg_removed_path, true_label, cfg, device
            )
            
            # 예측 결과 비교
            original_correct = (original_pred == true_label)
            bg_removed_correct = (bg_removed_pred == true_label)
            
            # 출력 디렉토리 결정
            if args.create_class_folders:
                if original_correct:
                    true_class_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
                    save_dir = os.path.join(comparison_output_dir, true_class_name)
                else:
                    save_dir = os.path.join(comparison_output_dir, "misclassified")
            else:
                save_dir = comparison_output_dir
            
            # 결과 저장
            output_path = save_comparison_gradcam_results(
                original_gradcam, bg_removed_gradcam,
                original_resized, bg_removed_resized,
                original_pred, original_conf,
                true_label, image_path, save_dir, i, cfg,
                patient_id, original_correct
            )
            
            # 결과 기록
            comparison_results.append({
                'image_path': image_path,
                'patient_id': patient_id,
                'true_label': true_label,
                'original_prediction': original_pred,
                'original_confidence': original_conf,
                'original_correct': original_correct,
                'bg_removed_prediction': bg_removed_pred,
                'bg_removed_confidence': bg_removed_conf,
                'bg_removed_correct': bg_removed_correct,
                'output_path': output_path
            })
            
            processed_count += 1
            
            print(f"    원본: 예측={original_pred}, 신뢰도={original_conf:.3f}, 정답={'O' if original_correct else 'X'}")
            print(f"    배경제거: 예측={bg_removed_pred}, 신뢰도={bg_removed_conf:.3f}, 정답={'O' if bg_removed_correct else 'X'}")
            
            # 임시 배경 제거 파일 정리
            if bg_removed_path != image_path and os.path.exists(bg_removed_path):
                try:
                    os.remove(bg_removed_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"이미지 {i} 처리 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Fold {fold_idx} 비교 완료: {processed_count}개 이미지 처리됨")
    
    return {
        'processed_count': processed_count,
        'comparison_results': comparison_results,
        'output_dir': comparison_output_dir
    }

def process_fold_test_set_batch(model, dataset, test_indices, cfg, device, output_base_dir, fold_idx, batch_size=8):
    """배치 단위로 특정 fold의 test set을 처리합니다."""
    print(f"\n=== Fold {fold_idx} Test Set 배치 처리 중 (배치 크기: {batch_size}) ===")
    
    # test set 생성
    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    # 출력 디렉토리 생성
    fold_output_dir = os.path.join(output_base_dir, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # 클래스별 디렉토리 생성
    for class_name in class_names:
        os.makedirs(os.path.join(fold_output_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(fold_output_dir, "misclassified"), exist_ok=True)
    
    model.eval()
    total_images = 0
    correct_images = 0
    incorrect_images = 0
    
    # 배치별로 처리
    for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Fold {fold_idx} 배치 처리 중")):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                images, labels = batch
            else:
                continue
        elif isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        else:
            continue
        
        images = images.to(device)
        labels = labels.to(device)
        
        # 모델 예측
        with torch.no_grad():
            outputs = model(images)
            
            if outputs.shape[1] == 1:
                predicted = (outputs.squeeze() > 0).long()
                confidence = torch.sigmoid(outputs.squeeze())
            else:
                predicted = torch.argmax(outputs, dim=1)
                confidence = F.softmax(outputs, dim=1).max(dim=1)[0]
        
        # 실제 이미지 경로들 가져오기
        batch_size_actual = test_loader.batch_size or 1
        start_idx = batch_idx * batch_size_actual
        end_idx = min(start_idx + batch_size_actual, len(test_indices))
        
        image_paths = []
        patient_ids = []
        true_classes = []
        is_corrects = []
        
        for i in range(len(labels)):
            actual_idx = test_indices[start_idx + i]
            if actual_idx < len(dataset.db_rec):
                db_record = dataset.db_rec[actual_idx]
                image_paths.append(db_record['file_path'])
                patient_ids.append(db_record.get('patient_id', 'unknown'))
            else:
                image_paths.append(f"unknown_path_{batch_idx}_{i}")
                patient_ids.append('unknown')
            
            true_label_idx = int(labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i])
            true_class = class_names[true_label_idx] if true_label_idx < len(class_names) else str(true_label_idx)
            true_classes.append(true_class)
            
            pred_label_idx = int(predicted[i].item() if isinstance(predicted[i], torch.Tensor) else predicted[i])
            is_correct = (true_label_idx == pred_label_idx)
            is_corrects.append(is_correct)
            
            if is_correct:
                correct_images += 1
            else:
                incorrect_images += 1
        
        try:
            # 배치 단위로 GradCAM 생성
            batch_gradcams, predicted_classes, confidences, resized_images, original_sizes = generate_batch_gradcam(
                model, image_paths, true_classes, cfg, str(device)
            )
            
            # 각 이미지별로 출력 디렉토리 결정 및 저장
            for i, (true_class, is_correct) in enumerate(zip(true_classes, is_corrects)):
                if is_correct:
                    output_dir = os.path.join(fold_output_dir, true_class)
                else:
                    output_dir = os.path.join(fold_output_dir, "misclassified")
                
                # 결과 저장
                save_batch_gradcam_results(
                    [batch_gradcams[i]], [resized_images[i]], [predicted_classes[i]], [confidences[i]], 
                    [true_class], [image_paths[i]], output_dir, total_images, cfg, 
                    [patient_ids[i]], [is_correct], 1
                )
                
                total_images += 1
                
        except Exception as e:
            print(f"배치 처리 중 오류 발생: {str(e)}")
            # 오류 발생 시 개별 이미지 처리로 fallback
            for i, (true_class, is_correct) in enumerate(zip(true_classes, is_corrects)):
                try:
                    if is_correct:
                        output_dir = os.path.join(fold_output_dir, true_class)
                    else:
                        output_dir = os.path.join(fold_output_dir, "misclassified")
                    
                    # 더미 GradCAM 생성
                    image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
                    if isinstance(image_size, int):
                        dummy_size = (image_size, image_size)
                    else:
                        dummy_size = image_size
                    dummy_gradcam = np.zeros(dummy_size)
                    
                    # 결과 저장 (오류 시 더미 이미지 생성)
                    dummy_image = np.zeros((*dummy_size, 3), dtype=np.uint8)
                    save_batch_gradcam_results(
                        [dummy_gradcam], [dummy_image], [predicted_classes[i]], [confidences[i]], 
                        [true_class], [image_paths[i]], output_dir, total_images, cfg, 
                        [patient_ids[i]], [is_correct], 1
                    )
                    
                    total_images += 1
                    
                except Exception as e2:
                    print(f"개별 이미지 처리 중 오류 발생: {str(e2)}")
                    continue
    
    print(f"Fold {fold_idx} 완료: 총 {total_images}개, 정확: {correct_images}개, 오류: {incorrect_images}개")
    
    return {
        'total': total_images,
        'correct': correct_images,
        'incorrect': incorrect_images,
        'accuracy': correct_images / total_images * 100 if total_images > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser(description='K-fold Test Set Batch GradCAM 분석 (배치 처리 버전)')
    parser.add_argument('--cfg', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--kfold_indices', type=str, required=True,
                       help='K-fold indices JSON 파일 경로')
    parser.add_argument('--log', type=str, required=True,
                       help='로그 파일 경로 (wandb run 정보 파싱용)')
    parser.add_argument('--wandb_dir', type=str, default='wandb',
                       help='wandb 폴더 경로')
    parser.add_argument('--output_dir', type=str, default='batch_gradcam_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--device', type=str, default='cuda',
                       help='사용할 디바이스 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='배치 크기 (GradCAM 생성용)')
    parser.add_argument('--remove_bg', action='store_true',
                       help='배경 제거된 이미지와 원본 이미지의 CAM 비교 수행')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='비교할 랜덤 샘플 수 (--remove_bg 사용시)')
    parser.add_argument('--bg_method', type=str, default='hsv_value',
                       help='배경 제거 방법 (fixed/otsu/percentile/hsv_value)')
    parser.add_argument('--bg_thresh', type=int, default=3,
                       help='배경 제거 임계값 (fixed 방법 사용시)')
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
    parser.add_argument('--bg_morph_kernel', type=int, default=1,
                       help='형태학적 연산 커널 크기')
    parser.add_argument('--bg_min_area', type=int, default=500,
                       help='최소 객체 영역 크기')
    parser.add_argument('--create_class_folders', action='store_true',
                       help='클래스별 폴더 생성')
    
    args = parser.parse_args()
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 랜덤 시드 설정
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 설정 업데이트
    update_config(cfg, args)
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    print(f"배치 크기: {args.batch_size}")
    
    # K-fold indices 로드
    with open(args.kfold_indices, 'r') as f:
        fold_indices = json.load(f)
    
    print(f"로드된 fold 수: {len(fold_indices)}")
    
    # wandb run 정보 파싱
    print(f"\n로그 파일에서 wandb run 정보 파싱 중...")
    wandb_runs = parse_wandb_runs_from_log(args.log)
    print(f"파싱된 wandb run 수: {len(wandb_runs)}")
    
    # 데이터셋 생성
    print("데이터셋 생성 중...")
    dataset = MedicalImageDataset(cfg)
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
    
    target_count = getattr(cfg.DATASET, 'TARGET_COUNT_PER_CLASS', None)
    if target_count:
        print(f"데이터 균등화 적용: 클래스당 {target_count}개")
        dataset.balance_dataset(target_count_per_class=target_count)
    
    # 각 fold 처리
    fold_results = {}
    
    for fold_name, fold_info in fold_indices.items():
        fold_idx = int(fold_name.split('_')[1])
        test_indices = fold_info['indices']
        
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx} 처리 중...")
        print(f"Test indices 수: {len(test_indices)}")
        
        # best model 찾기 (wandb run 정보 사용)
        best_model_path = find_best_model_in_wandb(args.wandb_dir, fold_idx, wandb_runs)
        
        if best_model_path is None:
            print(f"Fold {fold_idx}의 best model을 찾을 수 없습니다. 건너뜁니다.")
            continue
        
        # 모델 로드
        model = load_model(best_model_path, cfg, device)
        
        # fold 처리
        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
        
        if args.remove_bg:
            # 배경 제거 비교 모드
            fold_result = process_fold_comparison_gradcam(
                model, dataset, test_indices, cfg, device, fold_output_dir, fold_idx, args, args.num_samples
            )
        else:
            # 기본 배치 처리 모드
            fold_result = process_fold_test_set_batch(
                model, dataset, test_indices, cfg, device, fold_output_dir, fold_idx, args.batch_size
            )
        
        fold_results[fold_name] = fold_result
        
        # 모델 메모리 해제
        del model
        torch.cuda.empty_cache()
    
    # 전체 결과 요약
    print(f"\n{'='*60}")
    if args.remove_bg:
        print("배경 제거 비교 결과 요약")
    else:
        print("전체 결과 요약")
    print(f"{'='*60}")
    
    if args.remove_bg:
        # 배경 제거 비교 모드 결과
        total_processed = sum(result.get('processed_count', 0) for result in fold_results.values())
        print(f"전체 처리된 비교 이미지 수: {total_processed}")
        
        print(f"\nFold별 비교 결과:")
        for fold_name, result in fold_results.items():
            processed = result.get('processed_count', 0)
            print(f"  {fold_name}: {processed}개 이미지 비교 완료")
        
        # 결과 저장
        summary = {
            'mode': 'background_removal_comparison',
            'total_processed': total_processed,
            'num_samples_per_fold': args.num_samples,
            'fold_results': fold_results,
            'wandb_runs': wandb_runs,
            'output_dir': args.output_dir,
            'processing_time': time.time() - start_time
        }
    else:
        # 기본 배치 처리 모드 결과
        total_images = sum(result.get('total', 0) for result in fold_results.values())
        total_correct = sum(result.get('correct', 0) for result in fold_results.values())
        total_incorrect = sum(result.get('incorrect', 0) for result in fold_results.values())
        
        print(f"전체 이미지 수: {total_images}")
        print(f"전체 정확한 분류: {total_correct}")
        print(f"전체 잘못된 분류: {total_incorrect}")
        if total_images > 0:
            print(f"전체 정확도: {total_correct/total_images*100:.2f}%")
        
        print(f"\nFold별 결과:")
        for fold_name, result in fold_results.items():
            total = result.get('total', 0)
            correct = result.get('correct', 0)
            accuracy = result.get('accuracy', 0)
            print(f"  {fold_name}: {correct}/{total} ({accuracy:.2f}%)")
        
        # 결과 저장
        summary = {
            'mode': 'batch_processing',
            'total_images': total_images,
            'total_correct': total_correct,
            'total_incorrect': total_incorrect,
            'overall_accuracy': total_correct/total_images*100 if total_images > 0 else 0,
            'fold_results': fold_results,
            'wandb_runs': wandb_runs,
            'output_dir': args.output_dir,
            'batch_size': args.batch_size,
            'processing_time': time.time() - start_time
        }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n요약 정보가 {summary_path}에 저장되었습니다.")
    print(f"결과 저장 위치: {args.output_dir}")
    print(f"총 처리 시간: {time.time() - start_time:.2f}초")

if __name__ == '__main__':
    main() 
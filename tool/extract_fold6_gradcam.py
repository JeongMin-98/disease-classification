#!/usr/bin/env python3
"""
6번 fold의 test indices를 사용해서 각 모델의 best_model.pth로 성능 평가 및 GradCAM 생성
1. Dataset inference로 성능 지표 계산 (Confusion Matrix, Recall, Precision, Accuracy)
2. HSV 기반 배경 처리 + 원본/배경제거 이미지 각각에 대해 CAM 생성
3. 오분류 이미지 상세 분석
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
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

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
            method=args.bg_method,
            fixed_thresh=args.bg_thresh,
            percentile=2.0,
            hsv_value_thresh=args.bg_hsv_thresh,
            protect_skin=args.bg_protect_skin,
            protect_bone=args.bg_protect_bone,
            morph_kernel=args.bg_morph_kernel,
            keep_largest_only=False if args.bg_method == "hsv_value" else True,
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
            return image_path
            
    except Exception as e:
        print(f"배경 제거 중 오류 발생: {e}")
        return image_path

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

def preprocess_batch_for_inference(image_paths, cfg, target_size=None):
    """배치 단위로 이미지를 추론용으로 전처리합니다."""
    batch_images = []
    
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
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, target_size)
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_normalized = (image_normalized - mean) / std
            
            batch_images.append(image_normalized)
            
        except Exception as e:
            print(f"이미지 전처리 오류 ({image_path}): {e}")
            dummy_image = np.zeros((*target_size, 3), dtype=np.float32)
            batch_images.append(dummy_image)
    
    return np.array(batch_images)

def evaluate_model_performance(model, dataset, test_indices, cfg, device, model_name):
    """모델 성능을 평가하고 성능 지표를 계산합니다."""
    print(f"\n=== {model_name} 모델 성능 평가 ===")
    
    # test set 생성
    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    all_image_paths = []
    all_patient_ids = []
    
    # 추론 수행
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"{model_name} 추론 중")):
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            elif isinstance(batch, dict):
                images = batch['image']
                labels = batch['label']
            else:
                continue
            
            images = images.to(device)
            labels = labels.to(device)
            
            # 모델 예측
            outputs = model(images)
            
            if outputs.shape[1] == 1:
                # 이진 분류
                predicted = (outputs.squeeze() > 0).long()
                confidence = torch.sigmoid(outputs.squeeze())
            else:
                # 다중 분류
                predicted = torch.argmax(outputs, dim=1)
                confidence = F.softmax(outputs, dim=1).max(dim=1)[0]
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            
            # 이미지 경로와 patient ID 수집
            batch_size_actual = test_loader.batch_size or 1
            start_idx = len(all_image_paths)
            for i in range(len(labels)):
                actual_idx = test_indices[start_idx + i]
                if actual_idx < len(dataset.db_rec):
                    db_record = dataset.db_rec[actual_idx]
                    all_image_paths.append(db_record['file_path'])
                    all_patient_ids.append(db_record.get('patient_id', 'unknown'))
                else:
                    all_image_paths.append(f"unknown_path_{start_idx + i}")
                    all_patient_ids.append('unknown')
    
    # 성능 지표 계산
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_confidences = np.array(all_confidences)
    
    # Accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    
    # Precision, Recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # 클래스별 세부 지표
    class_report = classification_report(all_true_labels, all_predictions, 
                                       target_names=['OA', 'Normal'], output_dict=True)
    
    # 결과 출력
    print(f"\n{model_name} 성능 지표:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  Predicted")
    print(f"Actual  OA  Normal")
    print(f"OA      {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"Normal  {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    # 오분류/정분류 이미지 분리
    correct_indices = []
    incorrect_indices = []
    
    for i, (pred, true) in enumerate(zip(all_predictions, all_true_labels)):
        if pred == true:
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)
    
    print(f"\n이미지 분류 결과:")
    print(f"  정분류: {len(correct_indices)}개")
    print(f"  오분류: {len(incorrect_indices)}개")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'confidences': all_confidences,
        'image_paths': all_image_paths,
        'patient_ids': all_patient_ids,
        'correct_indices': correct_indices,
        'incorrect_indices': incorrect_indices
    }

def save_confusion_matrix(cm, model_name, output_dir):
    """Confusion Matrix를 시각화하여 저장합니다."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['OA', 'Normal'], yticklabels=['OA', 'Normal'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_path

def generate_gradcam_for_image(model, image_path, true_class, cfg, device):
    """단일 이미지에 대해 GradCAM을 생성합니다."""
    try:
        # 이미지 전처리
        input_image, resized_image, _ = preprocess_image_for_gradcam(image_path, cfg)
        
        # 모델 입력용 텐서 생성
        input_tensor = torch.from_numpy(input_image).unsqueeze(0).to(device)
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        input_tensor = input_tensor.float()
        
        # 타겟 레이어 설정
        target_layer = get_gradcam_target_layer(model)
        
        # 모델 예측
        with torch.no_grad():
            outputs = model(input_tensor)
            
            if outputs.shape[1] == 1:
                predicted = (outputs.squeeze() > 0).long()
                confidence = torch.sigmoid(outputs.squeeze())
            else:
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
        
        return grayscale_cam, pred_val, conf_val, resized_image
        
    except Exception as e:
        print(f"GradCAM 생성 중 오류: {e}")
        dummy_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
        if isinstance(dummy_size, int):
            dummy_size = (dummy_size, dummy_size)
        dummy_gradcam = np.zeros(dummy_size)
        dummy_image = np.zeros((*dummy_size, 3), dtype=np.uint8)
        return dummy_gradcam, 0, 0.5, dummy_image

def preprocess_image_for_gradcam(image_path, cfg, target_size=None):
    """이미지를 GradCAM용으로 전처리합니다."""
    if target_size is None:
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
        if isinstance(image_size, int):
            target_size = (image_size, image_size)
        else:
            target_size = image_size
    
    # 정규화 파라미터
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        image_resized = cv2.resize(image, target_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - mean) / std
        
        return image_normalized, image_resized, original_size
        
    except Exception as e:
        print(f"이미지 전처리 오류 ({image_path}): {e}")
        dummy_image = np.zeros((*target_size, 3), dtype=np.float32)
        return dummy_image, dummy_image, target_size

def save_gradcam_comparison(original_gradcam, bg_removed_gradcam, original_image, bg_removed_image,
                           predicted_class, confidence, true_class, image_path, output_dir, idx, cfg, 
                           patient_id=None, is_correct=None):
    """원본과 배경 제거된 이미지의 GradCAM 비교 결과를 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    if patient_id is None:
        patient_id = 'unknown'
    if is_correct is None:
        is_correct = True
    
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    image_name = Path(image_path).stem
    
    # 클래스명 가져오기
    if isinstance(true_class, int):
        true_class_name = class_names[true_class] if true_class < len(class_names) else str(true_class)
    else:
        true_class_name = str(true_class)
    
    if isinstance(predicted_class, int):
        pred_class_name = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)
    else:
        pred_class_name = str(predicted_class)
    
    # 파일명에 환자 ID, True Label, Prediction 포함
    output_filename = f"{patient_id}_{true_class_name}_{pred_class_name}_comparison.png"
    output_path = os.path.join(output_dir, output_filename)
    
    if isinstance(predicted_class, int):
        pred_class_name = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)
    else:
        pred_class_name = str(predicted_class)
    
    if isinstance(true_class, int):
        true_class_name = class_names[true_class] if true_class < len(class_names) else str(true_class)
    else:
        true_class_name = true_class
    
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
        original_normalized = original_image.astype(np.float32) / 255.0
        original_visualization = show_cam_on_image(original_normalized, original_gradcam, use_rgb=True)
        
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

def process_gradcam_for_images(model, dataset, image_indices, cfg, device, output_dir, model_name, args, 
                             performance_results, image_type="all"):
    """지정된 이미지들에 대해 GradCAM을 생성합니다."""
    print(f"\n=== {model_name} {image_type} 이미지 GradCAM 생성 ===")
    
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    # 출력 디렉토리 생성 - 단순화된 구조
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # misclassified 폴더와 클래스별 폴더 생성
    misclassified_dir = os.path.join(model_output_dir, "misclassified")
    os.makedirs(misclassified_dir, exist_ok=True)
    
    for class_name in class_names:
        class_dir = os.path.join(model_output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    model.eval()
    processed_count = 0
    comparison_results = []
    
    # 배치 크기 설정
    batch_size = args.batch_size
    
    # 배치별로 처리
    for batch_start in tqdm(range(0, len(image_indices), batch_size), desc=f"{image_type} 이미지 배치 처리 중"):
        batch_end = min(batch_start + batch_size, len(image_indices))
        batch_indices = image_indices[batch_start:batch_end]
        
        print(f"\n  배치 {batch_start//batch_size + 1}: 인덱스 {batch_start}-{batch_end-1} 처리 중...")
        
        # 배치 데이터 준비
        batch_image_paths = []
        batch_true_labels = []
        batch_patient_ids = []
        
        for idx in batch_indices:
            if idx >= len(dataset.db_rec):
                continue
                
            db_record = dataset.db_rec[idx]
            batch_image_paths.append(db_record['file_path'])
            batch_true_labels.append(db_record['label'])
            batch_patient_ids.append(db_record.get('patient_id', 'unknown'))
        
        if not batch_image_paths:
            continue
        
        # 각 이미지별로 GradCAM 생성
        for i, (image_path, true_label, patient_id) in enumerate(zip(batch_image_paths, batch_true_labels, batch_patient_ids)):
            try:
                # 원본 이미지 GradCAM 생성
                original_gradcam, original_pred, original_conf, original_resized = generate_gradcam_for_image(
                    model, image_path, true_label, cfg, device
                )
                
                # 배경 제거된 이미지 생성 및 GradCAM 생성
                bg_removed_path = apply_background_removal(image_path, args)
                bg_removed_gradcam, bg_removed_pred, bg_removed_conf, bg_removed_resized = generate_gradcam_for_image(
                    model, bg_removed_path, true_label, cfg, device
                )
                
                # 예측 결과 확인
                original_correct = (original_pred == true_label)
                
                # 출력 디렉토리 결정 - 단순화된 구조
                if original_correct:
                    true_class_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
                    save_dir = os.path.join(model_output_dir, true_class_name)
                else:
                    save_dir = misclassified_dir
                
                # 결과 저장
                output_path = save_gradcam_comparison(
                    original_gradcam, bg_removed_gradcam,
                    original_resized, bg_removed_resized,
                    original_pred, original_conf,
                    true_label, image_path, save_dir, batch_start + i, cfg,
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
                    'bg_removed_correct': (bg_removed_pred == true_label),
                    'output_path': output_path
                })
                
                processed_count += 1
                
                print(f"    이미지 {i+1}: 원본 예측={original_pred}, 신뢰도={original_conf:.3f}, 정답={'O' if original_correct else 'X'}")
                print(f"              배경제거 예측={bg_removed_pred}, 신뢰도={bg_removed_conf:.3f}, 정답={'O' if (bg_removed_pred == true_label) else 'X'}")
                
                # 임시 배경 제거 파일 정리
                if bg_removed_path != image_path and os.path.exists(bg_removed_path):
                    try:
                        os.remove(bg_removed_path)
                    except:
                        pass
                        
            except Exception as e:
                print(f"이미지 {i} 처리 중 오류 발생: {str(e)}")
                continue
    
    # 정확도 계산
    original_correct_count = sum(1 for result in comparison_results if result['original_correct'])
    bg_removed_correct_count = sum(1 for result in comparison_results if result['bg_removed_correct'])
    
    original_accuracy = (original_correct_count / processed_count * 100) if processed_count > 0 else 0
    bg_removed_accuracy = (bg_removed_correct_count / processed_count * 100) if processed_count > 0 else 0
    
    # 환자별 통계 계산
    patient_stats = {}
    for result in comparison_results:
        patient_id = result['patient_id']
        if patient_id not in patient_stats:
            patient_stats[patient_id] = {
                'total_images': 0,
                'correct_original': 0,
                'correct_bg_removed': 0
            }
        
        patient_stats[patient_id]['total_images'] += 1
        if result['original_correct']:
            patient_stats[patient_id]['correct_original'] += 1
        if result['bg_removed_correct']:
            patient_stats[patient_id]['correct_bg_removed'] += 1
    
    print(f"{model_name} {image_type} 이미지 GradCAM 생성 완료: {processed_count}개 이미지 처리됨")
    print(f"  원본 이미지 정확도: {original_correct_count}/{processed_count} ({original_accuracy:.2f}%)")
    print(f"  배경제거 이미지 정확도: {bg_removed_correct_count}/{processed_count} ({bg_removed_accuracy:.2f}%)")
    print(f"  처리된 환자 수: {len(patient_stats)}명")
    
    return {
        'processed_count': processed_count,
        'original_correct_count': original_correct_count,
        'bg_removed_correct_count': bg_removed_correct_count,
        'original_accuracy': original_accuracy,
        'bg_removed_accuracy': bg_removed_accuracy,
        'patient_stats': patient_stats,
        'comparison_results': comparison_results,
        'output_dir': model_output_dir
    }

def main():
    parser = argparse.ArgumentParser(description='6번 fold의 test indices를 사용해서 모델 성능 평가 및 GradCAM 생성')
    parser.add_argument('--cfg', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--fold_number', type=int, default=6,
                       help='분석할 fold 번호 (기본값: 6)')
    parser.add_argument('--output_dir', type=str, default='fold6_analysis_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--device', type=str, default='cuda',
                       help='사용할 디바이스 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='배치 크기 (기본값: 16)')
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
    
    # train_kfold.py와 동일한 방식으로 K-fold indices 생성 (검증용)
    print("K-fold indices 생성 중... (train_kfold.py와 동일한 방식)")
    
    # MedicalImageDataset 생성 (균등화 전)
    temp_dataset = MedicalImageDataset(cfg)
    
    # train_kfold.py와 동일한 K-fold 분할 함수 사용
    from sklearn.model_selection import StratifiedKFold, train_test_split
    
    def create_kfold_splits(dataset, n_splits=7, random_state=42):
        """train_kfold.py와 동일한 K-fold 분할 함수"""
        all_indices = list(range(len(dataset)))
        
        # 라벨 정보 추출
        if hasattr(dataset, 'db_rec'):
            labels = [dataset.db_rec[i]['label'] for i in all_indices]
        else:
            labels = [0] * len(all_indices)
        
        # StratifiedKFold 사용하여 클래스별 균등 분배 보장
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        fold_splits = []
        
        for fold_idx, (train_val_indices, test_indices) in enumerate(skf.split(all_indices, labels)):
            # train_val_indices를 다시 train과 val로 분할
            val_ratio = 0.15 / (0.7 + 0.15)  # 약 0.176
            
            # train_val_indices의 라벨 추출
            train_val_labels = [labels[i] for i in train_val_indices]
            
            # train과 val도 stratified split으로 분할
            train_indices, val_indices = train_test_split(
                train_val_indices, 
                test_size=val_ratio,
                random_state=random_state,
                stratify=train_val_labels
            )
            
            fold_splits.append({
                'fold': fold_idx,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices
            })
        
        return fold_splits
    
    # K-fold 분할 생성 (seed 42로 고정)
    kfold_size = 7
    fold_splits = create_kfold_splits(temp_dataset, n_splits=kfold_size, random_state=args.seed)
    
    # 지정된 fold의 test indices 가져오기
    target_fold = None
    for fold_info in fold_splits:
        if fold_info['fold'] == args.fold_number:
            target_fold = fold_info
            break
    
    if target_fold is None:
        print(f"지정된 fold {args.fold_number}를 찾을 수 없습니다.")
        print(f"사용 가능한 fold: {[f['fold'] for f in fold_splits]}")
        sys.exit(1)
    
    test_indices = target_fold['test_indices']
    print(f"Fold {args.fold_number} test indices 수: {len(test_indices)}")
    print(f"  Train indices: {len(target_fold['train_indices'])}")
    print(f"  Val indices: {len(target_fold['val_indices'])}")
    print(f"  Test indices: {len(test_indices)}")
    
    # 임시 데이터셋 메모리 해제
    del temp_dataset
    
    # 데이터셋 생성 - train_kfold.py와 동일한 방식
    print("데이터셋 생성 중...")
    dataset = MedicalImageDataset(cfg)
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
    
    # train_kfold.py와 동일한 데이터 균등화 적용
    target_count = getattr(cfg.DATASET, 'TARGET_COUNT_PER_CLASS', None)
    if target_count:
        print(f"데이터 균등화 적용: 클래스당 {target_count}개")
        dataset.balance_dataset(target_count_per_class=target_count)
        print(f"균등화 후 데이터셋 크기: {len(dataset)}")
    
    # 모델별 best_model.pth 경로 정의
    model_paths = {
        'vgg19bn_rgb': 'wandb/run-20250811_020827-uaj8lelt/files/best_model.pth',  # 일반 모델 Fold 6
        'vgg19bn_hsv': 'wandb/run-20250816_232559-svyp2x77/files/best_model.pth'   # HSV 모델 Fold 0 (가장 성능 좋음)
    }
    
    # 각 모델 처리
    all_results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n{'='*60}")
        print(f"{model_name} 모델 처리 중...")
        print(f"모델 경로: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            continue
        
        # 모델 로드
        model = load_model(model_path, cfg, device)
        
        # 1단계: 모델 성능 평가
        performance_results = evaluate_model_performance(model, dataset, test_indices, cfg, device, model_name)
        
        # Confusion Matrix 저장
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        cm_path = save_confusion_matrix(performance_results['confusion_matrix'], model_name, model_output_dir)
        print(f"Confusion Matrix 저장됨: {cm_path}")
        
        # 2단계: HSV 기반 배경 처리 + CAM 생성 (정분류 이미지)
        correct_gradcam_results = process_gradcam_for_images(
            model, dataset, performance_results['correct_indices'], cfg, device, 
            args.output_dir, model_name, args, performance_results, "correct"
        )
        
        # 3단계: 오분류 이미지 상세 분석
        incorrect_gradcam_results = process_gradcam_for_images(
            model, dataset, performance_results['incorrect_indices'], cfg, device, 
            args.output_dir, model_name, args, performance_results, "incorrect"
        )
        
        # 결과 통합
        all_results[model_name] = {
            'performance': performance_results,
            'correct_gradcam': correct_gradcam_results,
            'incorrect_gradcam': incorrect_gradcam_results,
            'confusion_matrix_path': cm_path
        }
        
        # 모델 메모리 해제
        del model
        torch.cuda.empty_cache()
    
    # 전체 결과 요약
    print(f"\n{'='*60}")
    print("전체 결과 요약")
    print(f"{'='*60}")
    
    for model_name, results in all_results.items():
        performance = results['performance']
        correct_gradcam = results['correct_gradcam']
        incorrect_gradcam = results['incorrect_gradcam']
        
        print(f"\n{model_name}:")
        print(f"  성능 지표:")
        print(f"    Accuracy: {performance['accuracy']:.4f}")
        print(f"    Precision: {performance['precision']:.4f}")
        print(f"    Recall: {performance['recall']:.4f}")
        print(f"    F1-Score: {performance['f1_score']:.4f}")
        print(f"  정분류 이미지 GradCAM: {correct_gradcam['processed_count']}개")
        print(f"  오분류 이미지 GradCAM: {incorrect_gradcam['processed_count']}개")
        print(f"  결과 저장 위치: {correct_gradcam['output_dir']}")
        
        # 환자별 통계 출력
        if 'patient_stats' in correct_gradcam:
            print(f"  정분류 환자별 통계:")
            for patient_id, stats in correct_gradcam['patient_stats'].items():
                print(f"    Patient {patient_id}: {stats['correct_original']}/{stats['total_images']} (원본), {stats['correct_bg_removed']}/{stats['total_images']} (배경제거)")
        
        if 'patient_stats' in incorrect_gradcam:
            print(f"  오분류 환자별 통계:")
            for patient_id, stats in incorrect_gradcam['patient_stats'].items():
                print(f"    Patient {patient_id}: {stats['correct_original']}/{stats['total_images']} (원본), {stats['correct_bg_removed']}/{stats['total_images']} (배경제거)")
    
    # JSON 직렬화를 위해 ndarray를 list로 변환하는 함수
    def convert_ndarray_to_list(obj):
        """ndarray를 list로 변환하여 JSON 직렬화 가능하게 만듭니다."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_ndarray_to_list(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    # 결과 저장 (ndarray를 list로 변환)
    summary = {
        'fold_number': args.fold_number,
        'test_indices_count': len(test_indices),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'all_results': convert_ndarray_to_list(all_results),
        'output_dir': args.output_dir,
        'processing_time': time.time() - start_time
    }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n요약 정보가 {summary_path}에 저장되었습니다.")
    print(f"결과 저장 위치: {args.output_dir}")
    print(f"총 처리 시간: {time.time() - start_time:.2f}초")
    print(f"\n사용법:")
    print(f"  python {sys.argv[0]} --cfg <config.yaml> --fold_number <fold_num> --seed <seed>")
    print(f"  예시: python {sys.argv[0]} --cfg experiments/config.yaml --fold_number 6 --seed 42")

if __name__ == '__main__':
    main()

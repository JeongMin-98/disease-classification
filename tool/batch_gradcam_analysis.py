#!/usr/bin/env python3
"""
배치 GradCAM 분석 스크립트
실제 모델이 올바르게 분류한 이미지들에 대해 GradCAM을 수행합니다.
"""

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
import random
from torch.utils.data import random_split

import _init_path
from models import create as create_model
from dataset import MedicalImageDataset
from config import cfg, update_config
from pytorch_grad_cam  import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def load_model(checkpoint_path, cfg, device):
    """체크포인트에서 모델을 로드합니다."""

    # cfg에서 모델 이름 가져오기
    model_name = getattr(cfg.MODEL, 'NAME', 'VGG19_BN')
    print(f"모델 이름: {model_name}")
    
    # 모델 생성
    model = create_model(model_name, cfg)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.float()  # 모델을 float32로 변환
    model.eval()
    
    print(f"모델이 {checkpoint_path}에서 로드되었습니다.")
    print(f"최고 검증 손실: {checkpoint['best_val_loss']:.4f}")
    
    return model

def get_gradcam_target_layer(model):
    """GradCAM을 위한 타겟 레이어를 반환합니다."""
    model_name = getattr(cfg.MODEL, 'NAME', 'VGG19_BN')
    print(f"GradCAM 타겟 레이어 선택 - 모델: {model_name}")
    
    try:
        if model_name.startswith('VGG'):
            # VGG19의 마지막 convolutional layer (features의 마지막 레이어)
            if hasattr(model.model, 'features'):
                target_layer = model.model.features[-1]
                print(f"VGG 모델 - features[-1] 선택됨")
            else:
                raise AttributeError(f"VGG 모델에 features 속성이 없습니다: {type(model.model)}")
                
        elif model_name.startswith('ResNet'):
            # ResNet의 마지막 convolutional layer (layer4의 마지막 레이어)
            if hasattr(model.model, 'layer4'):
                target_layer = model.model.layer4[-1]
                print(f"ResNet 모델 - layer4[-1] 선택됨")
            else:
                # ResNet의 다른 가능한 구조들
                if hasattr(model.model, 'layer3'):
                    target_layer = model.model.layer3[-1]
                    print(f"ResNet 모델 - layer3[-1] 선택됨 (layer4 없음)")
                else:
                    raise AttributeError(f"ResNet 모델에 layer4 또는 layer3 속성이 없습니다: {type(model.model)}")
        else:
            # 기본값으로 VGG 방식 사용
            if hasattr(model.model, 'features'):
                target_layer = model.model.features[-1]
                print(f"기본 모델 - features[-1] 선택됨")
            else:
                raise AttributeError(f"알 수 없는 모델 구조: {type(model.model)}")
        
        print(f"타겟 레이어 타입: {type(target_layer)}")
        return target_layer
        
    except Exception as e:
        print(f"GradCAM 타겟 레이어 선택 중 오류: {e}")
        print(f"모델 구조 디버깅:")
        print(f"  model type: {type(model)}")
        print(f"  model.model type: {type(model.model)}")
        if hasattr(model.model, '__dict__'):
            print(f"  model.model attributes: {list(model.model.__dict__.keys())}")
        
        # 마지막 수단으로 모델의 첫 번째 레이어 사용
        print("기본 타겟 레이어 사용")
        return model.model

def preprocess_image_for_gradcam(image_path, cfg, target_size=None):
    """이미지를 GradCAM용으로 전처리합니다."""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]  # (height, width)
    
    # cfg에서 이미지 크기 가져오기
    if target_size is None:
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
        if isinstance(image_size, int):
            target_size = (image_size, image_size)
        else:
            target_size = image_size
    
    # 모델 입력용으로 리사이즈
    image_resized = cv2.resize(image, target_size)
    
    # 정규화 (ImageNet 통계 사용)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 0-1 범위로 정규화
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # ImageNet 통계로 정규화
    image_normalized = (image_normalized - mean) / std
    
    return image_normalized, image, original_size

def generate_gradcam(model, image_path, target_class, cfg, device='cuda'):
    """GradCAM을 생성합니다."""
    # 이미지 전처리
    input_image, original_image, original_size = preprocess_image_for_gradcam(image_path, cfg)
    
    # 모델 입력용 텐서 생성
    input_tensor = torch.from_numpy(input_image).unsqueeze(0).to(device)
    input_tensor = input_tensor.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    input_tensor = input_tensor.float()  # float32로 변환
    
    # 타겟 레이어 설정
    target_layer = get_gradcam_target_layer(model)
    
    # 모델 예측
    with torch.no_grad():
        output = model(input_tensor)
        
        # 이진 분류 처리
        if output.shape[1] == 1:
            # 이진 분류 (sigmoid 출력)
            predicted_class = (output.squeeze() > 0).long().item()
            confidence = torch.sigmoid(output.squeeze()).item()
        else:
            # 다중 분류 (softmax 출력)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = F.softmax(output, dim=1).max().item()
    
    # GradCAM 생성 (최신 문법 사용)
    grayscale_cam = None
    try:
        # cfg에서 타겟 클래스 인덱스 가져오기 - INCLUDE_CLASSES 사용
        target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.Target_Classes
        class_names = target_classes if target_classes else ['oa', 'normal']
        
        # target_class가 문자열인 경우 인덱스로 변환
        if isinstance(target_class, str):
            if target_class in class_names:
                target_class_idx = class_names.index(target_class)
            else:
                # cfg의 INCLUDE_CLASSES에서 찾기
                if target_class in target_classes:
                    target_class_idx = target_classes.index(target_class)
                else:
                    print(f"Warning: target_class '{target_class}' not found in class_names {class_names}")
                    target_class_idx = 0  # 기본값
        else:
            # target_class가 이미 인덱스인 경우
            target_class_idx = int(target_class)
        
        # 이진 분류인 경우 타겟 인덱스를 0으로 설정 (출력이 1개뿐이므로)
        if output.shape[1] == 1:
            target_class_idx = 0
        
        print(f"Debug: target_class={target_class}, target_class_idx={target_class_idx}, class_names={class_names}")
        
        # GradCAM 초기화 및 실행
        try:
            with GradCAM(model=model, target_layers=[target_layer]) as cam:
                # 이진 분류의 경우 특별 처리
                if output.shape[1] == 1:
                    # 이진 분류: 첫 번째 출력에 대한 GradCAM (인덱스 0)
                    targets = [ClassifierOutputTarget(target_class_idx)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # type: ignore
                else:
                    # 다중 분류: 특정 클래스에 대한 GradCAM
                    targets = [ClassifierOutputTarget(target_class_idx)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # type: ignore
                
                grayscale_cam = grayscale_cam[0, :]
        except Exception as e:
            print(f"GradCAM 실행 오류: {e}")
            # 대안: 모델의 출력을 직접 사용하여 히트맵 생성
            grayscale_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
    except Exception as e:
        print(f"GradCAM 생성 오류: {e}")
        # 오류 발생 시 더미 히트맵 생성
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
        if isinstance(image_size, int):
            dummy_size = (image_size, image_size)
        else:
            dummy_size = image_size
        grayscale_cam = np.zeros(dummy_size)
    
    # GradCAM을 원본 이미지 크기로 리사이즈 (안전성 검사 추가)
    if grayscale_cam is not None and grayscale_cam.size > 0:
        try:
            grayscale_cam_resized = cv2.resize(grayscale_cam, (original_size[1], original_size[0]))
        except Exception as e:
            print(f"GradCAM 리사이즈 오류: {e}")
            # 오류 발생 시 더미 히트맵 생성
            grayscale_cam_resized = np.zeros((original_size[0], original_size[1]))
    else:
        # grayscale_cam이 None이거나 빈 경우
        grayscale_cam_resized = np.zeros((original_size[0], original_size[1]))
    
    # 원본 이미지와 GradCAM 오버레이
    try:
        # 원본 이미지를 0-1 범위로 정규화
        original_normalized = original_image.astype(np.float32) / 255.0
        visualization = show_cam_on_image(original_normalized, grayscale_cam_resized, use_rgb=True)
    except Exception as e:
        print(f"오버레이 생성 오류: {e}")
        # 오류 발생 시 원본 이미지만 사용
        visualization = original_image
    
    return visualization, predicted_class, confidence, original_image, grayscale_cam_resized

def save_gradcam_results(visualization, original_image, grayscale_cam, predicted_class, 
                        confidence, true_class, image_path, output_dir, image_idx, cfg, patient_id=None, is_correct=True):
    """GradCAM 결과를 저장합니다."""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성 (정확도 정보 포함)
    image_name = Path(image_path).stem
    accuracy_status = "correct" if is_correct else "incorrect"
    output_path = os.path.join(output_dir, f"{image_idx:03d}_{image_name}_{accuracy_status}_gradcam.png")
    
    # 원본 이미지 크기 정보
    original_size = original_image.shape[:2]
    gradcam_size = grayscale_cam.shape[:2]
    
    # cfg에서 모델 입력 크기 가져오기
    model_input_size = getattr(cfg.DATASET, 'IMAGE_SIZE', 224)
    if isinstance(model_input_size, int):
        model_input_size = (model_input_size, model_input_size)
    
    # 예측 클래스 이름 변환 - INCLUDE_CLASSES 사용
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.Target_Classes
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    if isinstance(predicted_class, int):
        pred_class_name = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)
    else:
        pred_class_name = str(predicted_class)
    
    # true_class가 이미 문자열인지 확인
    if isinstance(true_class, int):
        true_class_name = class_names[true_class] if true_class < len(class_names) else str(true_class)
    else:
        true_class_name = true_class
    
    # 판독문 생성
    report_text = f"Patient ID: {patient_id or 'Unknown'}\n"
    report_text += f"True Diagnosis: {true_class_name.upper()}\n"
    report_text += f"Model Prediction: {pred_class_name.upper()}\n"
    if confidence is not None:
        report_text += f"Confidence: {confidence:.3f}\n"
    report_text += f"Status: {'CORRECT' if is_correct else 'INCORRECT'}\n"
    report_text += f"Image Size: {original_size[1]}x{original_size[0]}\n"
    report_text += f"Model Input: {model_input_size[0]}x{model_input_size[1]}"
    
    # 결과 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 원본 이미지
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\nTrue: {true_class_name.upper()}\nSize: {original_size[1]}x{original_size[0]}')
    axes[0].axis('off')
    
    # GradCAM 히트맵 (원본 크기로 리사이즈된 것)
    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title(f'GradCAM Heatmap\nTarget: {true_class_name.upper()}\nSize: {gradcam_size[1]}x{gradcam_size[0]}')
    axes[1].axis('off')
    
    # 오버레이된 이미지
    axes[2].imshow(visualization)
    title = f'GradCAM Overlay\nTrue: {true_class_name.upper()} | Pred: {pred_class_name.upper()}'
    if confidence is not None:
        title += f'\nConfidence: {confidence:.3f}'
    title += f'\nStatus: {"CORRECT" if is_correct else "INCORRECT"}'
    title += f'\nOriginal: {original_size[1]}x{original_size[0]} | Model Input: {model_input_size[0]}x{model_input_size[1]}'
    axes[2].set_title(title)
    axes[2].axis('off')
    
    # 판독문을 전체 플롯 중앙 아래에 추가
    fig.text(0.5, 0.02, report_text, ha='center', va='bottom', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 판독문을 위한 공간 확보
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def load_dataset_info(json_path):
    """데이터셋 정보를 로드합니다."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_stratified_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    클래스 비율을 유지하면서 데이터셋을 train/val/test로 분할
    """
    total_size = len(dataset)
    
    # 라벨 정보 수집
    labels = []
    for idx in range(total_size):
        try:
            if hasattr(dataset, 'db_rec') and idx < len(dataset.db_rec):
                label = dataset.db_rec[idx]['label']
            else:
                sample = dataset[idx]
                if isinstance(sample, dict):
                    label = sample['label']
                else:
                    _, label = sample
                if isinstance(label, torch.Tensor):
                    label = label.item()
            labels.append(label)
        except Exception as e:
            print(f"Error getting label for index {idx}: {e}")
            labels.append(0)  # 기본값
    
    labels = np.array(labels)
    
    # 먼저 train과 temp로 분할
    train_indices, temp_indices = train_test_split(
        range(total_size),
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state
    )
    
    # temp를 val과 test로 분할
    temp_labels = labels[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_state
    )
    
    # Subset 객체 생성
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    return train_set, val_set, test_set

def get_all_test_images(model, dataloader, device):
    """테스트셋의 모든 이미지에 대해 예측 결과를 수집합니다."""
    model.eval()
    
    # cfg에서 클래스 이름 가져오기 - INCLUDE_CLASSES 사용
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.Target_Classes
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    all_images = []
    
    # 디버깅을 위한 통계
    total_images = 0
    class_counts = {class_name: 0 for class_name in class_names}
    correct_counts = {class_name: 0 for class_name in class_names}
    
    # 예측 분포 확인
    prediction_distribution = {class_name: {pred_class: 0 for pred_class in class_names} for class_name in class_names}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="테스트셋 전체 분석 중")):
            # 배치 데이터 처리
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    images, labels = batch
                    image_paths = [f"batch_{batch_idx}_sample_{i}" for i in range(len(images))]
                elif len(batch) == 3:
                    images, labels, image_paths = batch
                else:
                    continue
            elif isinstance(batch, dict):
                images = batch['image']
                labels = batch['label']
                image_paths = [f"batch_{batch_idx}_sample_{i}" for i in range(len(images))]
            else:
                continue
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # 이진 분류 처리
            if outputs.shape[1] == 1:
                # 이진 분류 (sigmoid 출력)
                predicted = (outputs.squeeze() > 0).long()
                confidence = torch.sigmoid(outputs.squeeze())
            else:
                # 다중 분류 (softmax 출력)
                predicted = torch.argmax(outputs, dim=1)
                confidence = F.softmax(outputs, dim=1).max(dim=1)[0]
            
            # 각 이미지에 대해 결과 수집
            for i, (true_label, pred_label, conf) in enumerate(zip(labels, predicted, confidence)):
                total_images += 1
                
                # 클래스 이름 가져오기 - 정수 인덱스로 변환
                true_label_idx = int(true_label.item() if isinstance(true_label, torch.Tensor) else true_label)
                pred_label_idx = int(pred_label.item() if isinstance(pred_label, torch.Tensor) else pred_label)
                
                true_class = class_names[true_label_idx] if true_label_idx < len(class_names) else str(true_label_idx)
                pred_class = class_names[pred_label_idx] if pred_label_idx < len(class_names) else str(pred_label_idx)
                
                # 예측 분포 업데이트
                prediction_distribution[true_class][pred_class] += 1
                
                # 통계 업데이트
                class_counts[true_class] += 1
                is_correct = (true_label_idx == pred_label_idx)
                if is_correct:
                    correct_counts[true_class] += 1
                
                # 이미지 정보 생성
                sample_info = {
                    'true_label': true_label_idx,
                    'pred_label': pred_label_idx,
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'confidence': conf.item() if isinstance(conf, torch.Tensor) else conf,
                    'is_correct': is_correct,
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'patient_id': 'unknown'
                }
                
                # 데이터셋에서 실제 이미지 경로와 환자 ID 가져오기
                if hasattr(dataloader.dataset, 'dataset') and hasattr(dataloader.dataset.dataset, 'db_rec'):
                    # Subset인 경우
                    actual_idx = dataloader.dataset.indices[batch_idx * dataloader.batch_size + i]
                    if actual_idx < len(dataloader.dataset.dataset.db_rec):
                        db_record = dataloader.dataset.dataset.db_rec[actual_idx]
                        sample_info['path'] = db_record['file_path']
                        sample_info['patient_id'] = db_record.get('patient_id', 'unknown')
                elif hasattr(dataloader.dataset, 'db_rec'):
                    # 직접 데이터셋인 경우
                    actual_idx = batch_idx * dataloader.batch_size + i
                    if actual_idx < len(dataloader.dataset.db_rec):
                        db_record = dataloader.dataset.db_rec[actual_idx]
                        sample_info['path'] = db_record['file_path']
                        sample_info['patient_id'] = db_record.get('patient_id', 'unknown')
                else:
                    # 경로를 찾을 수 없는 경우
                    sample_info['path'] = f"unknown_path_{batch_idx}_{i}"
                
                all_images.append(sample_info)
    
    # 디버깅 정보 출력
    print(f"\n=== 전체 테스트셋 분석 결과 ===")
    print(f"총 이미지 수: {total_images}")
    print(f"클래스별 총 개수: {class_counts}")
    print(f"클래스별 올바른 분류 개수: {correct_counts}")
    accuracy_info = {k: v/class_counts[k]*100 if class_counts[k] > 0 else 0 for k, v in correct_counts.items()}
    print(f"클래스별 정확도: {accuracy_info}")
    print(f"전체 정확도: {sum(correct_counts.values())/total_images*100:.2f}%")
    print(f"예측 분포:")
    for true_class in class_names:
        print(f"  {true_class} -> {prediction_distribution[true_class]}")
    print("==============================\n")
    
    return all_images

def main():
    parser = argparse.ArgumentParser(description='전체 테스트셋 GradCAM 분석')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='체크포인트 파일 경로')
    parser.add_argument('--cfg', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--output_dir', type=str, default='full_test_gradcam_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--device', type=str, default='cuda',
                       help='사용할 디바이스 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    
    args = parser.parse_args()
    
    # 랜덤 시드 설정
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 설정 업데이트
    update_config(cfg, args)
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    model = load_model(args.checkpoint, cfg, device)
    
    # 훈련 시와 동일한 방식으로 데이터셋 생성
    print("데이터셋 생성 중...")
    dataset = MedicalImageDataset(cfg)
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.Target_Classes
    
    # 클래스 불균형 해결 (훈련 시와 동일)
    target_count = getattr(cfg.DATASET, 'TARGET_COUNT_PER_CLASS', None)
    if target_count:
        print(f"데이터 균등화 적용: 클래스당 {target_count}개")
        dataset.balance_dataset(target_count_per_class=target_count)
    
    # Stratified Split을 사용하여 데이터 분할 (훈련 시와 동일)
    use_stratified_split = getattr(cfg.DATASET, 'USE_STRATIFIED_SPLIT', True)
    if use_stratified_split:
        print("Using Stratified Split for train/val/test division")
        train_set, val_set, test_set = create_stratified_splits(
            dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15, 
            random_state=args.seed
        )
    else:
        print("Using Random Split for train/val/test division")
        # 기존 방식 (random_split)
        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # val과 test set에 대해 augmentation 비활성화 (훈련 시와 동일)
    val_dataset = MedicalImageDataset(cfg, is_train=False)
    test_dataset = MedicalImageDataset(cfg, is_train=False)
    
    # Subset의 인덱스를 새로운 데이터셋에 적용
    val_indices = val_set.indices
    test_indices = test_set.indices
    
    val_set = Subset(val_dataset, val_indices)
    test_set = Subset(test_dataset, test_indices)
    
    # 테스트 데이터로더 생성
    test_loader = DataLoader(test_set, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False)
    
    # 전체 테스트셋 분석
    print("전체 테스트셋 분석 중...")
    all_test_images = get_all_test_images(model, test_loader, device)
    
    # cfg에서 클래스 이름 가져오기 - INCLUDE_CLASSES 사용
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.Target_Classes
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    print(f"전체 테스트셋 이미지 수: {len(all_test_images)}개")
    
    # GradCAM 생성
    print("\n전체 테스트셋에 대해 GradCAM 생성 중...")
    total_images = 0
    correct_images = 0
    incorrect_images = 0
    
    # 클래스별로 결과 저장
    for class_name in class_names:
        class_output_dir = os.path.join(args.output_dir, class_name)
        correct_output_dir = os.path.join(class_output_dir, 'correct')
        incorrect_output_dir = os.path.join(class_output_dir, 'incorrect')
        
        # 해당 클래스의 이미지들만 필터링
        class_images = [img for img in all_test_images if img['true_class'] == class_name]
        
        print(f"\n{class_name.upper()} 클래스 처리 중... (총 {len(class_images)}개)")
        
        for idx, image_info in enumerate(tqdm(class_images, desc=f"{class_name} GradCAM")):
            try:
                # 디버그 정보 출력
                print(f"Debug: Processing {class_name} image {idx+1}/{len(class_images)}")
                print(f"  true_class: {image_info['true_class']}")
                print(f"  pred_class: {image_info['pred_class']}")
                print(f"  is_correct: {image_info['is_correct']}")
                print(f"  confidence: {image_info['confidence']:.3f}")
                
                # GradCAM 생성 - true_class를 전달
                visualization, predicted_class, confidence, original_image, grayscale_cam = generate_gradcam(
                    model, image_info['path'], image_info['true_class'], cfg, str(device)
                )
                
                # 올바른/잘못된 분류에 따라 다른 디렉토리에 저장
                if image_info['is_correct']:
                    output_dir = correct_output_dir
                    correct_images += 1
                else:
                    output_dir = incorrect_output_dir
                    incorrect_images += 1
                
                # 결과 저장
                output_path = save_gradcam_results(
                    visualization, original_image, grayscale_cam, image_info['pred_class'], 
                    image_info['confidence'], image_info['true_class'], image_info['path'], 
                    output_dir, idx, cfg, image_info['patient_id'], image_info['is_correct']
                )
                
                total_images += 1
                
            except Exception as e:
                print(f"이미지 처리 중 오류 발생: {image_info['path']} - {str(e)}")
                continue
    
    print(f"\n분석 완료!")
    print(f"총 처리된 이미지 수: {total_images}개")
    print(f"올바르게 분류된 이미지: {correct_images}개")
    print(f"잘못 분류된 이미지: {incorrect_images}개")
    print(f"결과 저장 위치: {args.output_dir}")
    
    # 결과 요약 저장
    summary = {
        'total_images': total_images,
        'correct_images': correct_images,
        'incorrect_images': incorrect_images,
        'accuracy': correct_images / total_images * 100 if total_images > 0 else 0,
        'class_summary': {},
        'output_dir': args.output_dir
    }
    
    # 클래스별 요약 정보
    for class_name in class_names:
        class_images = [img for img in all_test_images if img['true_class'] == class_name]
        class_correct = sum(1 for img in class_images if img['is_correct'])
        class_total = len(class_images)
        summary['class_summary'][class_name] = {
            'total': class_total,
            'correct': class_correct,
            'incorrect': class_total - class_correct,
            'accuracy': class_correct / class_total * 100 if class_total > 0 else 0
        }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"요약 정보가 {summary_path}에 저장되었습니다.")
    
    # 클래스별 결과 요약 출력
    print(f"\n=== 클래스별 결과 요약 ===")
    for class_name, class_info in summary['class_summary'].items():
        print(f"{class_name.upper()}: {class_info['correct']}/{class_info['total']} ({class_info['accuracy']:.2f}%)")
    print(f"전체: {correct_images}/{total_images} ({summary['accuracy']:.2f}%)")
    print("===========================")

if __name__ == '__main__':
    main() 
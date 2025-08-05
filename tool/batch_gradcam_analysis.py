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
from models.vgg19 import VGG
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

    num_classes = len(cfg.DATASET.TARGET_CLASSES)
    # 모델 초기화
    model = VGG(
        num_classes=num_classes,
        Target_Classes=cfg.DATASET.TARGET_CLASSES,
        freeze_layers=cfg.MODEL.FREEZE_LAYERS
    )
    
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
    # VGG19의 마지막 convolutional layer (features의 마지막 레이어)
    target_layer = model.model.features[-1]
    return target_layer

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
                    target_class_idx = 0  # 기본값
        else:
            target_class_idx = target_class
        
        # 이진 분류인 경우 타겟 인덱스를 0으로 설정 (출력이 1개뿐이므로)
        if output.shape[1] == 1:
            target_class_idx = 0
        
        # 타겟 설정 - 이진 분류의 경우 특별 처리
        if output.shape[1] == 1:
            # 이진 분류: 첫 번째 출력에 대한 GradCAM (인덱스 0)
            targets = [ClassifierOutputTarget(0)]
        else:
            # 다중 분류: 해당 클래스에 대한 GradCAM
            targets = [ClassifierOutputTarget(target_class_idx)]
        
        # GradCAM 초기화 및 실행
        try:
            with GradCAM(model=model, target_layers=[target_layer]) as cam:
                # 이진 분류의 경우 특별 처리
                if output.shape[1] == 1:
                    # 이진 분류: 모델의 출력을 직접 사용
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                else:
                    # 다중 분류: 특정 클래스에 대한 GradCAM
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                
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
                        confidence, true_class, image_path, output_dir, image_idx, cfg, patient_id=None):
    """GradCAM 결과를 저장합니다."""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_idx:03d}_{image_name}_gradcam.png")
    
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
    
    # 판독문 생성
    report_text = f"Patient ID: {patient_id or 'Unknown'}\n"
    report_text += f"Diagnosis: {true_class.upper()}\n"
    report_text += f"Model Prediction: {pred_class_name.upper()}\n"
    if confidence is not None:
        report_text += f"Confidence: {confidence:.3f}\n"
    report_text += f"Image Size: {original_size[1]}x{original_size[0]}\n"
    report_text += f"Model Input: {model_input_size[0]}x{model_input_size[1]}"
    
    # 결과 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 원본 이미지
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\nTrue: {true_class}\nSize: {original_size[1]}x{original_size[0]}')
    axes[0].axis('off')
    
    # GradCAM 히트맵 (원본 크기로 리사이즈된 것)
    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title(f'GradCAM Heatmap\nTarget: {true_class}\nSize: {gradcam_size[1]}x{gradcam_size[0]}')
    axes[1].axis('off')
    
    # 오버레이된 이미지
    axes[2].imshow(visualization)
    title = f'GradCAM Overlay\nTrue: {true_class} | Pred: {pred_class_name.upper()}'
    if confidence is not None:
        title += f'\nConfidence: {confidence:.3f}'
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

def get_correctly_classified_images(model, dataloader, device, max_images_per_class=32):
    """올바르게 분류된 이미지들을 찾습니다."""
    model.eval()
    
    # cfg에서 클래스 이름 가져오기 - INCLUDE_CLASSES 사용
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.Target_Classes
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    correct_images = {class_name: [] for class_name in class_names}
    incorrect_images = {class_name: [] for class_name in class_names}  # 잘못 분류된 이미지도 수집
    
    # 디버깅을 위한 통계
    total_images = 0
    class_counts = {class_name: 0 for class_name in class_names}
    correct_counts = {class_name: 0 for class_name in class_names}
    
    # 예측 분포 확인
    prediction_distribution = {class_name: {pred_class: 0 for pred_class in class_names} for class_name in class_names}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="올바른 분류 찾는 중")):
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
            else:
                # 다중 분류 (softmax 출력)
                predicted = torch.argmax(outputs, dim=1)
            
            # 올바르게 분류된 이미지들 찾기
            correct_mask = (predicted == labels)
            
            for i, (is_correct, true_label, pred_label) in enumerate(zip(correct_mask, labels, predicted)):
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
                if is_correct:
                    correct_counts[true_class] += 1
                
                # 이미지 정보 생성
                sample_info = {
                    'true_label': true_label_idx,
                    'pred_label': pred_label_idx,
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'patient_id': 'unknown'  # 기본값
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
                
                if is_correct:
                    if len(correct_images[true_class]) < max_images_per_class:
                        correct_images[true_class].append(sample_info)
                else:
                    # 잘못 분류된 이미지도 수집 (normal 클래스가 올바르게 분류되지 않을 경우를 대비)
                    if len(incorrect_images[true_class]) < max_images_per_class:
                        incorrect_images[true_class].append(sample_info)
            
            # 충분한 이미지를 찾았으면 중단
            if all(len(correct_images[class_name]) >= max_images_per_class for class_name in class_names):
                break
    
    # normal 클래스가 올바르게 분류된 이미지가 없으면 잘못 분류된 이미지 사용
    for class_name in class_names:
        if len(correct_images[class_name]) == 0 and len(incorrect_images[class_name]) > 0:
            print(f"Warning: {class_name} 클래스에 올바르게 분류된 이미지가 없어서 잘못 분류된 이미지를 사용합니다.")
            correct_images[class_name] = incorrect_images[class_name][:max_images_per_class]
    
    # 디버깅 정보 출력
    print(f"\n=== 디버깅 정보 ===")
    print(f"총 이미지 수: {total_images}")
    print(f"클래스별 총 개수: {class_counts}")
    print(f"클래스별 올바른 분류 개수: {correct_counts}")
    accuracy_info = {k: v/class_counts[k]*100 if class_counts[k] > 0 else 0 for k, v in correct_counts.items()}
    print(f"클래스별 정확도: {accuracy_info}")
    found_images = {k: len(v) for k, v in correct_images.items()}
    print(f"찾은 이미지: {found_images}")
    print(f"예측 분포:")
    for true_class in class_names:
        print(f"  {true_class} -> {prediction_distribution[true_class]}")
    print("==================\n")
    
    return correct_images

def main():
    parser = argparse.ArgumentParser(description='배치 GradCAM 분석')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='체크포인트 파일 경로')
    parser.add_argument('--cfg', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--output_dir', type=str, default='batch_gradcam_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--max_images_per_class', type=int, default=32,
                       help='클래스당 최대 이미지 수')
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
    
    # 올바르게 분류된 이미지들 찾기
    print("올바르게 분류된 이미지들을 찾는 중...")
    correct_images = get_correctly_classified_images(
        model, test_loader, device, args.max_images_per_class
    )
    
    # cfg에서 클래스 이름 가져오기 - INCLUDE_CLASSES 사용
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.Target_Classes
    class_names = target_classes if target_classes else ['oa', 'normal']
    
    print(f"찾은 이미지 수:")
    for class_name in class_names:
        print(f"  {class_name.upper()} 클래스: {len(correct_images[class_name])}개")
    
    # GradCAM 생성
    print("\nGradCAM 생성 중...")
    total_images = 0
    
    for class_name, images in correct_images.items():
        print(f"\n{class_name.upper()} 클래스 처리 중...")
        class_output_dir = os.path.join(args.output_dir, class_name)
        
        for idx, image_info in enumerate(tqdm(images, desc=f"{class_name} GradCAM")):
            try:
                # GradCAM 생성
                visualization, predicted_class, confidence, original_image, grayscale_cam = generate_gradcam(
                    model, image_info['path'], image_info['true_label'], cfg, str(device)
                )
                
                # 결과 저장
                output_path = save_gradcam_results(
                    visualization, original_image, grayscale_cam, predicted_class, 
                    confidence, class_name, image_info['path'], class_output_dir, idx, cfg, image_info['patient_id']
                )
                
                total_images += 1
                
            except Exception as e:
                print(f"이미지 처리 중 오류 발생: {image_info['path']} - {str(e)}")
                continue
    
    print(f"\n분석 완료!")
    print(f"총 처리된 이미지 수: {total_images}개")
    print(f"결과 저장 위치: {args.output_dir}")
    
    # 결과 요약 저장
    summary = {
        'total_images': total_images,
        'class_images': {class_name: len(correct_images[class_name]) for class_name in class_names},
        'output_dir': args.output_dir
    }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"요약 정보가 {summary_path}에 저장되었습니다.")

if __name__ == '__main__':
    main() 
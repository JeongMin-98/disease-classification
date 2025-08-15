import _init_path
import os
import logging
import wandb
import torch
from torch.utils.data import random_split, DataLoader, Subset
from models import create
from core.base_trainer import BaseTrainer
from core.original_trainer import OriginalTrainer
from utils.utils import set_seed
from dataset import MedicalImageDataset
from dataset.balanced_sampler import BalancedBatchSampler, StratifiedBatchSampler
from config import cfg, update_config
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from sweep_configs import get_sweep_config, create_custom_sweep_config, SWEEP_CONFIGS


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
            logging.warning(f"Error getting label for index {idx}: {e}")
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


def analyze_batch_distribution(dataloader, dataset, num_batches=5, logger=None):
    """
    DataLoader의 배치당 클래스 분포를 분석하고 로깅
    """
    if logger is None:
        logger = logging.getLogger("batch_analysis")
    
    logger.info(f"=== 배치 분포 분석 (처음 {num_batches}개 배치) ===")
    
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_count >= num_batches:
            break
        
        # MedicalImageDataset은 딕셔너리를 반환하므로 'label' 키로 접근
        if isinstance(batch, dict):
            labels = batch['label']
        else:
            # 기존 방식 (튜플인 경우)
            _, labels = batch
            
        # 배치 내 클래스별 샘플 수 계산
        batch_class_counts = defaultdict(int)
        for label in labels:
            if isinstance(label, torch.Tensor):
                label = label.item()
            batch_class_counts[label] += 1
        
        # 클래스별 분포를 정렬하여 로깅
        class_distribution = dict(sorted(batch_class_counts.items()))
        logger.info(f"배치 {batch_idx}: 클래스 분포 = {class_distribution}")
        
        batch_count += 1
    
    logger.info("=== 배치 분포 분석 완료 ===")


def train_sweep():
    """
    Wandb sweep을 위한 메인 훈련 함수
    """
    # wandb 초기화 (sweep agent로 실행됨)
    wandb.init()
    
    # sweep에서 설정된 하이퍼파라미터 가져오기
    config = wandb.config
    
    # config 업데이트
    update_config(cfg, config)
    set_seed(config.seed)
    
    # cfg 파일의 내용을 logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_sweep")
    logger.info("실행 config 내용:")
    for k, v in cfg.items() if hasattr(cfg, 'items') else cfg.__dict__.items():
        logger.info(f"{k}: {v}")

    # device를 cfg로부터 받아서 명시적으로 지정
    device = getattr(cfg, 'DEVICE', 'GPU' if torch.cuda.is_available() else 'cpu')
    device = "cuda" if device == "GPU" else "cpu"
    logger.info(f"사용할 device: {device}")

    # 데이터셋 생성 및 클래스 필터링
    dataset = MedicalImageDataset(cfg)
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.Target_Classes

    # 클래스 불균형 해결
    target_count = getattr(cfg.DATASET, 'TARGET_COUNT_PER_CLASS', None)
    if target_count:
        logger.info(f"=== 데이터 균등화 전 ===")
        logger.info(f"원본 데이터셋 크기: {len(dataset)}")
        # 원본 클래스 분포 확인
        original_summary = dataset.summary()
        logger.info(f"원본 클래스별 샘플 수: {original_summary}")
    
    dataset.balance_dataset(target_count_per_class=target_count)
    
    if target_count:
        logger.info(f"=== 데이터 균등화 후 ===")
        logger.info(f"균등화된 데이터셋 크기: {len(dataset)}")
        logger.info(f"클래스당 목표 샘플 수: {target_count}")
        # 균등화 후 클래스 분포 확인
        balanced_summary = dataset.summary()
        logger.info(f"균등화 후 클래스별 샘플 수: {balanced_summary}")
        logger.info("=" * 50)

    # Stratified Split을 사용하여 데이터 분할 (클래스 비율 유지)
    use_stratified_split = getattr(cfg.DATASET, 'USE_STRATIFIED_SPLIT', True)
    if use_stratified_split:
        logger.info("Using Stratified Split for train/val/test division")
        train_set, val_set, test_set = create_stratified_splits(
            dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15, 
            random_state=config.seed
        )
    else:
        logger.info("Using Random Split for train/val/test division")
        # 기존 방식 (random_split)
        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # val과 test set에 대해 augmentation 비활성화
    val_dataset = MedicalImageDataset(cfg, is_train=False)
    test_dataset = MedicalImageDataset(cfg, is_train=False)
    
    # Subset의 인덱스를 새로운 데이터셋에 적용
    val_indices = val_set.indices
    test_indices = test_set.indices
    
    from torch.utils.data import Subset
    val_set = Subset(val_dataset, val_indices)
    test_set = Subset(test_dataset, test_indices)

    # 배치 크기 조정 (클래스 수로 나누어 떨어지도록)
    num_classes = len(target_classes)
    original_batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
    adjusted_batch_size = (original_batch_size // num_classes) * num_classes
    if adjusted_batch_size != original_batch_size:
        logger.info(f"Batch size adjusted from {original_batch_size} to {adjusted_batch_size} for balanced sampling")
        cfg.TRAIN.BATCH_SIZE_PER_GPU = adjusted_batch_size

    # Sampler 선택
    use_balanced_sampling = getattr(cfg.TRAIN, 'USE_BALANCED_SAMPLING', True)
    sampling_type = getattr(cfg.TRAIN, 'SAMPLING_TYPE', 'balanced')
    
    # 배치 분포 로깅 옵션 (성능 향상을 위해 기본적으로 비활성화)
    log_batch_distribution = getattr(cfg.TRAIN, 'LOG_BATCH_DISTRIBUTION', False)
    
    if use_balanced_sampling:
        if sampling_type == 'balanced':
            train_sampler = BalancedBatchSampler(
                train_set, 
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, 
                shuffle=True, 
                drop_last=True,
                log_batch_distribution=log_batch_distribution
            )
            logger.info("Using BalancedBatchSampler (equal samples per class)")
        elif sampling_type == 'stratified':
            train_sampler = StratifiedBatchSampler(
                train_set, 
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, 
                shuffle=True, 
                drop_last=True,
                log_batch_distribution=log_batch_distribution
            )
            logger.info("Using StratifiedBatchSampler (maintains class ratios)")
        else:
            raise ValueError(f"Unknown sampling type: {sampling_type}")
    else:
        train_sampler = None
        logger.info("Using standard DataLoader (no balanced sampling)")

    # DataLoader
    if train_sampler is not None:
        train_loader = DataLoader(
            train_set, 
            batch_sampler=train_sampler,
            num_workers=cfg.WORKERS
        )
    else:
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, 
            shuffle=True,
            num_workers=cfg.WORKERS
        )
    val_loader = DataLoader(val_set, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False)

    # 배치 분포 분석 및 로깅 (로깅 옵션이 활성화된 경우에만)
    if log_batch_distribution:
        logger.info("=== 훈련 데이터 배치 분포 확인 ===")
        analyze_batch_distribution(train_loader, train_set, num_batches=5, logger=logger)
        
        logger.info("=== 검증 데이터 배치 분포 확인 ===")
        analyze_batch_distribution(val_loader, val_set, num_batches=3, logger=logger)
        
        logger.info("=== 테스트 데이터 배치 분포 확인 ===")
        analyze_batch_distribution(test_loader, test_set, num_batches=3, logger=logger)

    # 모델 생성 (VGG19)
    freeze_layers = getattr(cfg.MODEL, 'FREEZE_LAYERS', None)
    if freeze_layers:
        logger.info(f"Freezing layers: {freeze_layers}")
    model = create('VGG19', Target_Classes=target_classes, pretrained=True, freeze_layers=freeze_layers)
    model = model.to(device)

    trainer = OriginalTrainer(cfg, model, device, cfg.OUTPUT_DIR)

    # wandb run_name, tags 자동 생성
    lr = getattr(cfg.TRAIN, 'LR', None) or getattr(cfg.TRAIN, 'LEARNING_RATE', None) or getattr(cfg.TRAIN, 'BASE_LR', None)
    scheduler = getattr(cfg.TRAIN, 'SCHEDULER', None)
    scheduler_tag = scheduler if scheduler else 'none'
    optimizer = getattr(cfg.TRAIN, 'OPTIMIZER', None)
    model_name = 'VGG19'
    dataset_target_class = ','.join(target_classes) if target_classes else 'all'
    dataset_type = cfg.DATASET.TYPE
    run_name = f"{model_name}_opt-{optimizer}_sch-{scheduler}_lr-{lr}_cls-{dataset_target_class}_type-{dataset_type}"
    tags = [model_name, optimizer, scheduler_tag, f"lr:{lr}", f"cls:{dataset_target_class}", f"type:{dataset_type}"]

    trainer.init_wandb(run_name, tags)

    # 학습
    test_acc = trainer.train(train_loader, val_loader, test_loader, num_epochs=cfg.TRAIN.END_EPOCH)
    logger.info(f'최종 테스트 정확도: {test_acc:.2f}%')
    
    # sweep 결과를 wandb에 로깅
    wandb.log({"test_accuracy": test_acc})


def main():
    """
    Sweep 설정 및 실행
    """
    parser = argparse.ArgumentParser(description='Wandb Sweep을 사용한 하이퍼파라미터 최적화')
    
    # 기본 인자들
    parser.add_argument('--cfg', type=str, 
                       default='experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml',
                       help='실험에 사용할 config yaml 경로')
    parser.add_argument('--sweep_config', type=str, default='base',
                       choices=list(SWEEP_CONFIGS.keys()),
                       help=f'사용할 sweep 설정 (기본값: base). 선택가능: {list(SWEEP_CONFIGS.keys())}')
    parser.add_argument('--project', type=str, default='disease-classification',
                       help='Wandb 프로젝트 이름')
    parser.add_argument('--count', type=int, default=20,
                       help='실행할 실험 수')
    parser.add_argument('--custom_params', type=str, default=None,
                       help='커스텀 파라미터 (JSON 형식 문자열)')
    
    # 커스텀 sweep 설정을 위한 인자들
    parser.add_argument('--method', type=str, choices=['grid', 'random', 'bayes'],
                       help='Sweep 방법 (커스텀 설정 시 사용)')
    parser.add_argument('--sweep_name', type=str,
                       help='Sweep 이름 (커스텀 설정 시 사용)')
    parser.add_argument('--metric', type=str, default='test_accuracy',
                       help='최적화할 메트릭 이름 (커스텀 설정 시 사용)')
    
    args = parser.parse_args()
    
    # 커스텀 파라미터 파싱
    custom_parameters = None
    if args.custom_params:
        import json
        try:
            custom_parameters = json.loads(args.custom_params)
        except json.JSONDecodeError:
            print("Error: custom_params는 유효한 JSON 형식이어야 합니다.")
            return
    
    # Sweep 설정 가져오기
    if args.method and args.sweep_name:
        # 커스텀 sweep 설정 생성
        sweep_config = create_custom_sweep_config(
            method=args.method,
            name=args.sweep_name,
            metric_name=args.metric,
            parameters=custom_parameters
        )
        print(f"커스텀 sweep 설정 사용: {args.sweep_name}")
    else:
        # 미리 정의된 sweep 설정 사용
        sweep_config = get_sweep_config(args.sweep_config, custom_parameters)
        print(f"미리 정의된 sweep 설정 사용: {args.sweep_config}")
    
    # cfg 파일 경로를 sweep 설정에 추가
    if 'cfg' not in sweep_config['parameters']:
        sweep_config['parameters']['cfg'] = {'values': [args.cfg]}
    else:
        # 기존 cfg 값들을 args.cfg로 덮어쓰기
        sweep_config['parameters']['cfg'] = {'values': [args.cfg]}
    
    print(f"Sweep 설정:")
    print(f"  - 방법: {sweep_config['method']}")
    print(f"  - 이름: {sweep_config['name']}")
    print(f"  - 메트릭: {sweep_config['metric']['name']}")
    print(f"  - 파라미터 수: {len(sweep_config['parameters'])}")
    print(f"  - 실행할 실험 수: {args.count}")
    print(f"  - Wandb 프로젝트: {args.project}")
    
    # Sweep 생성
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"Sweep ID: {sweep_id}")
    
    # Sweep agent 시작
    wandb.agent(sweep_id, train_sweep, count=args.count)


if __name__ == "__main__":
    main() 
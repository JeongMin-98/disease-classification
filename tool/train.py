import _init_path
import os
import logging
import wandb
import torch
from torch.utils.data import random_split, DataLoader, Subset
from models import create
from core.base_trainer import BaseTrainer
from core.original_trainer import OriginalTrainer
from utils.utils import (
    set_seed, 
    create_balanced_stratified_splits, 
    create_stratified_splits, 
    analyze_batch_distribution,
    verify_split_distribution
)
from dataset import MedicalImageDataset
from dataset.balanced_sampler import BalancedBatchSampler, StratifiedBatchSampler
from config import cfg, update_config
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np


def main(args):
    # config 업데이트
    update_config(cfg, args)
    set_seed(args.seed)
    
    # cfg 파일의 내용을 logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train")
    logger.info("실행 config 내용:")
    for k, v in cfg.items() if hasattr(cfg, 'items') else cfg.__dict__.items():
        logger.info(f"{k}: {v}")

    # device를 cfg로부터 받아서 명시적으로 지정
    device = getattr(cfg, 'DEVICE', 'GPU' if torch.cuda.is_available() else 'cpu')
    device = "cuda" if device == "GPU" else "cpu"
    logger.info(f"사용할 device: {device}")

    # logger 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train")

    # 데이터셋을 한 번만 생성
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

    # 데이터 분할 방식 선택
    use_balanced_split = getattr(cfg.DATASET, 'USE_BALANCED_SPLIT', True)
    use_stratified_split = getattr(cfg.DATASET, 'USE_STRATIFIED_SPLIT', True)
    
    if use_balanced_split:
        logger.info("Using Balanced Stratified Split (equal class distribution in each split)")
        train_set, val_set, test_set = create_balanced_stratified_splits(
            dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15, 
            random_state=args.seed,
            verify=True,
            logger=logger
        )
    elif use_stratified_split:
        logger.info("Using Stratified Split (maintains class ratios)")
        train_set, val_set, test_set = create_stratified_splits(
            dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15, 
            random_state=args.seed
        )
    else:
        logger.info("Using Random Split for train/val/test division")
        # 기존 방식 (random_split)
        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # 분할 결과 로깅
    logger.info(f"데이터 분할 결과:")
    logger.info(f"  Train set: {len(train_set)} samples")
    logger.info(f"  Val set: {len(val_set)} samples")
    logger.info(f"  Test set: {len(test_set)} samples")
    logger.info(f"  Total: {len(train_set) + len(val_set) + len(test_set)} samples")

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
                log_batch_distribution=log_batch_distribution  # 로깅 옵션 적용
            )
            logger.info("Using BalancedBatchSampler (equal samples per class)")
        elif sampling_type == 'stratified':
            train_sampler = StratifiedBatchSampler(
                train_set, 
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, 
                shuffle=True, 
                drop_last=True,
                log_batch_distribution=log_batch_distribution  # 로깅 옵션 적용
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
    model_name = 'VGG19'  # 현재 모델이 고정되어 있으므로
    dataset_target_class = ','.join(target_classes) if target_classes else 'all'
    dataset_type = cfg.DATASET.TYPE
    run_name = f"{model_name}_opt-{optimizer}_sch-{scheduler}_lr-{lr}_cls-{dataset_target_class}_type-{dataset_type}"
    tags = [model_name, optimizer, scheduler_tag, f"lr:{lr}", f"cls:{dataset_target_class}", f"type:{dataset_type}"]

    trainer.init_wandb(run_name, tags)

    # 학습
    test_acc = trainer.train(train_loader, val_loader, test_loader, num_epochs=cfg.TRAIN.END_EPOCH)
    logger.info(f'최종 테스트 정확도: {test_acc:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml', help='실험에 사용할 config yaml 경로')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    args = parser.parse_args()
    main(args)

"""
WandB 체크포인트 기능 사용법:

1. 실험 중에는 로컬에만 체크포인트가 저장됩니다 (성능 최적화)
2. 실험 종료 시에만 best_model과 final_model이 wandb에 업로드됩니다
3. wandb에서 체크포인트를 다운로드하려면:

   # best_model 다운로드
   checkpoint_path = BaseTrainer.download_checkpoint_from_wandb(
       run_id="your_run_id", 
       checkpoint_type="best_model"
   )
   
   # final_model 다운로드  
   checkpoint_path = BaseTrainer.download_checkpoint_from_wandb(
       run_id="your_run_id", 
       checkpoint_type="final_model"
   )

4. 다운로드한 체크포인트를 로드하려면:
   trainer = OriginalTrainer(cfg, model, device, output_dir)
   trainer.load_checkpoint(checkpoint_path)
""" 
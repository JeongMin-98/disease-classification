#!/usr/bin/env python3
"""
오분류 케이스 분석을 위한 독립 스크립트
기존 체크포인트를 로드하여 테스트 세트의 오분류 케이스를 분석하고 저장합니다.

재현성 보장:
- train.py와 동일한 시드 사용
- 동일한 데이터 분할 방식 적용
- 동일한 테스트 세트 구성
"""

import _init_path
import os
import argparse
import torch
from torch.utils.data import DataLoader
from models import create
from core.original_trainer import OriginalTrainer
from dataset import MedicalImageDataset
from config import cfg, update_config
from utils.utils import (
    set_seed, 
    create_balanced_stratified_splits, 
    create_stratified_splits
)
import logging

def main(args):
    # Config 업데이트
    update_config(cfg, args)
    set_seed(args.seed)
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("analyze_misclassified")
    
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 데이터셋 생성 (train.py와 동일한 방식)
    dataset = MedicalImageDataset(cfg)
    target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.Target_Classes
    
    # 클래스 불균형 해결 (train.py와 동일)
    target_count = getattr(cfg.DATASET, 'TARGET_COUNT_PER_CLASS', None)
    if target_count:
        logger.info(f"=== 데이터 균등화 전 ===")
        logger.info(f"원본 데이터셋 크기: {len(dataset)}")
        original_summary = dataset.summary()
        logger.info(f"원본 클래스별 샘플 수: {original_summary}")
    
    dataset.balance_dataset(target_count_per_class=target_count)
    
    if target_count:
        logger.info(f"=== 데이터 균등화 후 ===")
        logger.info(f"균등화된 데이터셋 크기: {len(dataset)}")
        logger.info(f"클래스당 목표 샘플 수: {target_count}")
        balanced_summary = dataset.summary()
        logger.info(f"균등화 후 클래스별 샘플 수: {balanced_summary}")
        logger.info("=" * 50)

    # 데이터 분할 방식 선택 (train.py와 동일)
    use_balanced_split = getattr(cfg.DATASET, 'USE_BALANCED_SPLIT', True)
    use_stratified_split = getattr(cfg.DATASET, 'USE_STRATIFIED_SPLIT', True)
    
    if use_balanced_split:
        logger.info("Using Balanced Stratified Split (equal class distribution in each split)")
        train_set, val_set, test_set = create_balanced_stratified_splits(
            dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15, 
            random_state=args.seed,  # 동일한 시드 사용
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
            random_state=args.seed  # 동일한 시드 사용
        )
    else:
        logger.info("Using Random Split for train/val/test division")
        from torch.utils.data import random_split
        # 시드 설정 후 분할
        torch.manual_seed(args.seed)
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
    
    # 테스트 데이터로더 생성 (train.py와 동일)
    test_loader = DataLoader(
        test_set, 
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, 
        shuffle=False,  # 테스트 시에는 셔플하지 않음
        num_workers=cfg.WORKERS
    )
    
    # 모델 생성
    model = create('VGG19', Target_Classes=target_classes, pretrained=False)
    model = model.to(device)
    
    # Trainer 생성
    trainer = OriginalTrainer(cfg, model, device, args.output_dir or "./misclassified_analysis")
    
    # 체크포인트 로드
    if not args.checkpoint:
        logger.error("체크포인트 경로를 지정해주세요 (--checkpoint)")
        return
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        return
    
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint)
    
    # 재현성 확인을 위한 추가 로깅
    logger.info("=" * 60)
    logger.info("재현성 보장 정보:")
    logger.info(f"  사용된 시드: {args.seed}")
    logger.info(f"  데이터 분할 방식: {'Balanced Stratified' if use_balanced_split else 'Stratified' if use_stratified_split else 'Random'}")
    logger.info(f"  테스트 세트 크기: {len(test_set)} samples")
    logger.info(f"  배치 크기: {cfg.TEST.BATCH_SIZE_PER_GPU}")
    logger.info("=" * 60)
    
    # 오분류 케이스 분석 및 저장
    logger.info("Starting misclassified cases analysis...")
    misclassified_dir, metadata_path = trainer.save_misclassified_cases(
        test_loader, 
        model_type=args.model_type,
        output_dir=args.output_dir
    )
    
    logger.info("Analysis completed!")
    logger.info(f"Results saved to: {misclassified_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze misclassified cases")
    parser.add_argument('--cfg', type=str, required=True, 
                        help='config yaml 파일 경로')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='분석할 체크포인트 파일 경로')
    parser.add_argument('--model_type', type=str, default='best', 
                        choices=['best', 'final'], 
                        help='모델 타입 (best 또는 final)')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='결과 저장 디렉토리 (기본값: ./misclassified_analysis)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='랜덤 시드 (train.py와 동일한 시드 사용 권장)')
    
    args = parser.parse_args()
    main(args) 
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
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import json
from pathlib import Path


def verify_fold_class_distribution(fold_splits, dataset, logger):
    """
    각 fold의 클래스 분포를 검증하여 균등 분배가 되었는지 확인합니다.
    """
    logger.info("=== Fold별 클래스 분포 검증 ===")
    
    if not hasattr(dataset, 'db_rec'):
        logger.warning("dataset.db_rec가 없어 클래스 분포 검증을 건너뜁니다.")
        return
    
    # 클래스별 전체 샘플 수 계산
    all_labels = [dataset.db_rec[i]['label'] for i in range(len(dataset))]
    unique_classes = list(set(all_labels))
    total_per_class = {cls: all_labels.count(cls) for cls in unique_classes}
    
    logger.info(f"전체 데이터 클래스별 샘플 수: {total_per_class}")
    
    for fold_info in fold_splits:
        fold_idx = fold_info['fold']
        
        # 각 fold의 train/val/test에서 클래스별 샘플 수 계산
        train_labels = [dataset.db_rec[i]['label'] for i in fold_info['train_indices']]
        val_labels = [dataset.db_rec[i]['label'] for i in fold_info['val_indices']]
        test_labels = [dataset.db_rec[i]['label'] for i in fold_info['test_indices']]
        
        train_per_class = {cls: train_labels.count(cls) for cls in unique_classes}
        val_per_class = {cls: val_labels.count(cls) for cls in unique_classes}
        test_per_class = {cls: test_labels.count(cls) for cls in unique_classes}
        
        logger.info(f"Fold {fold_idx}:")
        logger.info(f"  Train: {train_per_class}")
        logger.info(f"  Val:   {val_per_class}")
        logger.info(f"  Test:  {test_per_class}")
        
        # 균등 분배 검증
        train_balanced = len(set(train_per_class.values())) <= 1
        val_balanced = len(set(val_per_class.values())) <= 1
        test_balanced = len(set(test_per_class.values())) <= 1
        
        if train_balanced and val_balanced and test_balanced:
            logger.info(f"  ✓ Fold {fold_idx}: 모든 분할에서 균등 분배 확인")
        else:
            logger.warning(f"  ⚠ Fold {fold_idx}: 일부 분할에서 불균등 분배 발견")
    
    logger.info("=" * 50)


def create_kfold_splits(dataset, n_splits=7, random_state=42):
    """
    K-fold 교차 검증을 위한 데이터 분할을 생성합니다.
    각 fold에서 서로 다른 test set을 사용하며, 모든 데이터가 한 번씩 test set이 됩니다.
    StratifiedKFold를 사용하여 각 fold에서 클래스별 균등 분배를 보장합니다.
    """
    from sklearn.model_selection import StratifiedKFold
    
    # 전체 데이터셋의 인덱스와 라벨
    all_indices = list(range(len(dataset)))
    
    # 라벨 정보 추출 (dataset.db_rec에서)
    if hasattr(dataset, 'db_rec'):
        labels = [dataset.db_rec[i]['label'] for i in all_indices]
    else:
        # db_rec가 없는 경우 기본 라벨 사용
        labels = [0] * len(all_indices)
    
    # StratifiedKFold 사용하여 클래스별 균등 분배 보장
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_splits = []
    
    for fold_idx, (train_val_indices, test_indices) in enumerate(skf.split(all_indices, labels)):
        # train_val_indices를 다시 train과 val로 분할
        # val 비율을 15%로 맞춤 (전체 데이터 기준)
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


def main(args):
    # config 업데이트
    update_config(cfg, args)
    set_seed(args.seed)
    
    # cfg 파일의 내용을 logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_kfold")
    logger.info("실행 config 내용:")
    for k, v in cfg.items() if hasattr(cfg, 'items') else cfg.__dict__.items():
        logger.info(f"{k}: {v}")

    # device를 cfg로부터 받아서 명시적으로 지정
    device = getattr(cfg, 'DEVICE', 'GPU' if torch.cuda.is_available() else 'cpu')
    device = "cuda" if device == "GPU" else "cpu"
    logger.info(f"사용할 device: {device}")

    # K-fold 설정 (7-fold로 고정)
    kfold_size = 7
    logger.info(f"K-fold 크기: {kfold_size} (7-fold)")

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

    # K-fold 분할 생성
    logger.info(f"=== K-fold 분할 생성 (k={kfold_size}) ===")
    fold_splits = create_kfold_splits(dataset, n_splits=kfold_size, random_state=args.seed)
    
    # 각 fold의 분할 결과 로깅
    logger.info("K-fold 분할 결과:")
    all_test_indices = set()
    for fold_info in fold_splits:
        fold_idx = fold_info['fold']
        train_size = len(fold_info['train_indices'])
        val_size = len(fold_info['val_indices'])
        test_size = len(fold_info['test_indices'])
        total_size = train_size + val_size + test_size
        
        logger.info(f"Fold {fold_idx}: Train={train_size}, Val={val_size}, Test={test_size}, Total={total_size}")
        
        # test indices를 수집하여 중복 확인
        test_indices = set(fold_info['test_indices'])
        all_test_indices.update(test_indices)
    
    # 각 fold의 클래스 분포 검증
    verify_fold_class_distribution(fold_splits, dataset, logger)
    
    # K-fold 교차 검증 검증
    total_data_size = len(dataset)
    expected_test_size_per_fold = total_data_size // kfold_size
    logger.info(f"\nK-fold 교차 검증 검증:")
    logger.info(f"전체 데이터 크기: {total_data_size}")
    logger.info(f"fold당 예상 test 크기: {expected_test_size_per_fold}")
    logger.info(f"실제 test set에 포함된 고유 데이터 수: {len(all_test_indices)}")
    logger.info(f"모든 데이터가 test set에 포함됨: {len(all_test_indices) == total_data_size}")
    
    if len(all_test_indices) != total_data_size:
        logger.warning("경고: 일부 데이터가 test set에 포함되지 않았습니다!")
    else:
        logger.info("✓ 모든 데이터가 정확히 한 번씩 test set에 포함되었습니다.")

    # 배치 크기 조정 (클래스 수로 나누어 떨어지도록)
    num_classes = len(target_classes)
    original_batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
    adjusted_batch_size = (original_batch_size // num_classes) * num_classes
    if adjusted_batch_size != original_batch_size:
        logger.info(f"Batch size adjusted from {original_batch_size} to {adjusted_batch_size} for balanced sampling")
        cfg.TRAIN.BATCH_SIZE_PER_GPU = adjusted_batch_size

    # K-fold 실험 결과 저장
    kfold_results = {
        'config': {
            'kfold_size': kfold_size,
            'seed': args.seed,
            'model_name': getattr(cfg.MODEL, 'NAME', 'VGG19'),
            'target_classes': target_classes,
            'batch_size': cfg.TRAIN.BATCH_SIZE_PER_GPU
        },
        'folds': []
    }

    # K-fold 실험에서 체크포인트 저장 옵션
    save_final_models = getattr(cfg.KFOLD, 'SAVE_FINAL_MODELS', False)  # 기본적으로 False
    save_best_models = getattr(cfg.KFOLD, 'SAVE_BEST_MODELS', True)     # 기본적으로 True
    keep_only_best_fold = getattr(cfg.KFOLD, 'KEEP_ONLY_BEST_FOLD', False)  # 최고 성능 fold만 저장
    keep_best_and_worst = getattr(cfg.KFOLD, 'KEEP_BEST_AND_WORST', True)   # 최고/최저 성능 fold만 저장
    
    logger.info(f"체크포인트 저장 설정:")
    logger.info(f"  Save final models: {save_final_models}")
    logger.info(f"  Save best models: {save_best_models}")
    logger.info(f"  Keep only best fold: {keep_only_best_fold}")
    logger.info(f"  Keep best and worst folds: {keep_best_and_worst}")

    # 각 fold에 대해 실험 수행
    fold_accuracies = []  # 성능 추적용
    for fold_info in fold_splits:
        fold_idx = fold_info['fold']
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx} 시작")
        logger.info(f"{'='*60}")
        
        # 현재 fold의 test set 정보 로깅
        test_indices = fold_info['test_indices']
        logger.info(f"Fold {fold_idx} Test Set:")
        logger.info(f"  Test indices: {test_indices[:10]}{'...' if len(test_indices) > 10 else ''}")
        logger.info(f"  Test set 크기: {len(test_indices)}")
        
        # cfg 파일의 AUGMENT 설정을 사용하여 augmentation 제어
        use_augment = getattr(cfg.DATASET, 'AUGMENT', True)
        logger.info(f"Fold {fold_idx}: Augmentation 사용 여부: {use_augment}")
        
        # Train용과 Val/Test용을 분리하여 생성
        if use_augment:
            train_dataset = MedicalImageDataset(cfg, is_train=True)      # augmentation 적용
            val_test_dataset = MedicalImageDataset(cfg, is_train=False)  # augmentation 미적용
            logger.info(f"Fold {fold_idx}: Train set에만 augmentation 적용")
        else:
            # augmentation을 사용하지 않는 경우
            train_dataset = MedicalImageDataset(cfg, is_train=False)
            val_test_dataset = MedicalImageDataset(cfg, is_train=False)
            logger.info(f"Fold {fold_idx}: 모든 set에 augmentation 미적용")
        
        # 각 데이터셋에 균등화 적용
        train_dataset.balance_dataset(target_count_per_class=target_count)
        val_test_dataset.balance_dataset(target_count_per_class=target_count)
        
        # Subset 생성
        train_set = Subset(train_dataset, fold_info['train_indices'])
        val_set = Subset(val_test_dataset, fold_info['val_indices'])
        test_set = Subset(val_test_dataset, fold_info['test_indices'])
        
        logger.info(f"Fold {fold_idx} 데이터 분할:")
        logger.info(f"  Train set: {len(train_set)} samples")
        logger.info(f"  Val set: {len(val_set)} samples")
        logger.info(f"  Test set: {len(test_set)} samples")

        # 각 fold의 클래스 분포 확인 (균등화 확인)
        if hasattr(train_dataset, 'summary'):
            train_summary = train_dataset.summary()
            logger.info(f"Fold {fold_idx} Train set 클래스 분포: {train_summary}")
        if hasattr(val_test_dataset, 'summary'):
            val_test_summary = val_test_dataset.summary()
            logger.info(f"Fold {fold_idx} Val/Test set 클래스 분포: {val_test_summary}")

        # Sampler 선택
        use_balanced_sampling = getattr(cfg.TRAIN, 'USE_BALANCED_SAMPLING', True)
        sampling_type = getattr(cfg.TRAIN, 'SAMPLING_TYPE', 'balanced')
        
        # 배치 분포 로깅 옵션 (K-fold에서는 기본적으로 활성화)
        log_batch_distribution = getattr(cfg.TRAIN, 'LOG_BATCH_DISTRIBUTION', True)  # K-fold에서는 True로 변경
        
        if use_balanced_sampling:
            if sampling_type == 'balanced':
                train_sampler = BalancedBatchSampler(
                    train_set, 
                    batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, 
                    shuffle=True, 
                    drop_last=True,
                    log_batch_distribution=log_batch_distribution
                )
                logger.info(f"Fold {fold_idx}: Using BalancedBatchSampler (equal samples per class)")
            elif sampling_type == 'stratified':
                train_sampler = StratifiedBatchSampler(
                    train_set, 
                    batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, 
                    shuffle=True, 
                    drop_last=True,
                    log_batch_distribution=log_batch_distribution
                )
                logger.info(f"Fold {fold_idx}: Using StratifiedBatchSampler (maintains class ratios)")
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

        # 배치 분포 분석 및 로깅
        if log_batch_distribution:
            logger.info(f"=== Fold {fold_idx} 훈련 데이터 배치 분포 확인 ===")
            analyze_batch_distribution(train_loader, train_set, num_batches=3, logger=logger)
            
            logger.info(f"=== Fold {fold_idx} 검증 데이터 배치 분포 확인 ===")
            analyze_batch_distribution(val_loader, val_set, num_batches=2, logger=logger)
            
            logger.info(f"=== Fold {fold_idx} 테스트 데이터 배치 분포 확인 ===")
            analyze_batch_distribution(test_loader, test_set, num_batches=2, logger=logger)

        # 모델 생성
        model_name = getattr(cfg.MODEL, 'NAME', 'VGG19')
        pretrained = getattr(cfg.MODEL, 'PRETRAINED', True)
        freeze_layers = getattr(cfg.MODEL, 'FREEZE_LAYERS', None)
        
        logger.info(f"Creating model: {model_name}")
        if freeze_layers:
            logger.info(f"Freezing layers: {freeze_layers}")
        
        model = create(model_name, Target_Classes=target_classes, pretrained=pretrained, freeze_layers=freeze_layers)
        model = model.to(device)

        # 출력 디렉토리 설정 (fold별로 분리)
        fold_output_dir = os.path.join(cfg.OUTPUT_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)

        trainer = OriginalTrainer(cfg, model, device, fold_output_dir)
        trainer.set_wandb_project("Medical-Classification-Kfold")

        # wandb run_name, tags 자동 생성 (fold 정보 포함)
        lr = getattr(cfg.TRAIN, 'LR', None) or getattr(cfg.TRAIN, 'LEARNING_RATE', None) or getattr(cfg.TRAIN, 'BASE_LR', None)
        scheduler = getattr(cfg.TRAIN, 'SCHEDULER', None)
        scheduler_tag = scheduler if scheduler else 'none'
        optimizer = getattr(cfg.TRAIN, 'OPTIMIZER', None)
        dataset_target_class = ','.join(target_classes) if target_classes else 'all'
        dataset_type = cfg.DATASET.TYPE
        run_name = f"fold{fold_idx}_{model_name}_opt-{optimizer}_sch-{scheduler}_lr-{lr}_cls-{dataset_target_class}_type-{dataset_type}"
        tags = [f"fold{fold_idx}", model_name, optimizer, scheduler_tag, f"lr:{lr}", f"cls:{dataset_target_class}", f"type:{dataset_type}", "kfold"]

        trainer.init_wandb(run_name, tags)

        # 학습
        test_acc = trainer.train(train_loader, val_loader, test_loader, num_epochs=cfg.TRAIN.END_EPOCH)
        logger.info(f'Fold {fold_idx} 최종 테스트 정확도: {test_acc:.2f}%')

        # fold 완료 후 메모리 정리 (OOM 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Fold {fold_idx} 완료 후 GPU 메모리 정리됨")
        
        # fold 결과 저장
        fold_result = {
            'fold': fold_idx,
            'train_size': len(train_set),
            'val_size': len(val_set),
            'test_size': len(test_set),
            'test_accuracy': test_acc,
            'output_dir': fold_output_dir
        }
        kfold_results['folds'].append(fold_result)
        fold_accuracies.append(test_acc)

        # 체크포인트 관리 (용량 절약)
        if not save_final_models:
            # final_model 삭제 (best_model만 유지)
            final_model_path = os.path.join(fold_output_dir, 'final_model.pth')
            if os.path.exists(final_model_path):
                os.remove(final_model_path)
                logger.info(f"Fold {fold_idx}: final_model.pth 삭제됨 (용량 절약)")

        # wandb run 종료
        if wandb.run is not None:
            wandb.finish()
        
        # trainer 객체 메모리 해제
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Fold {fold_idx} trainer 객체 메모리 해제됨")

    # 최고/최저 성능 fold만 유지하는 옵션
    if keep_best_and_worst and len(fold_accuracies) > 0:
        best_fold_idx = np.argmax(fold_accuracies)
        worst_fold_idx = np.argmin(fold_accuracies)
        
        logger.info(f"최고 성능 fold: {best_fold_idx} (정확도: {fold_accuracies[best_fold_idx]:.2f}%)")
        logger.info(f"최저 성능 fold: {worst_fold_idx} (정확도: {fold_accuracies[worst_fold_idx]:.2f}%)")
        
        # 최고/최저 성능 fold가 아닌 것들의 체크포인트 삭제
        for fold_info in fold_splits:
            fold_idx = fold_info['fold']
            if fold_idx not in [best_fold_idx, worst_fold_idx]:
                fold_output_dir = os.path.join(cfg.OUTPUT_DIR, f"fold_{fold_idx}")
                if os.path.exists(fold_output_dir):
                    import shutil
                    shutil.rmtree(fold_output_dir)
                    logger.info(f"Fold {fold_idx} 디렉토리 삭제됨 (최고/최저 성능 fold만 유지)")
        
        logger.info(f"유지된 fold: {best_fold_idx} (최고), {worst_fold_idx} (최저)")
        logger.info(f"삭제된 fold: {[i for i in range(len(fold_accuracies)) if i not in [best_fold_idx, worst_fold_idx]]}")

    # K-fold 실험 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info("K-fold 실험 결과 요약")
    logger.info(f"{'='*60}")
    
    accuracies = [fold['test_accuracy'] for fold in kfold_results['folds']]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    logger.info(f"각 fold 정확도: {accuracies}")
    logger.info(f"평균 정확도: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"최고 정확도: {max(accuracies):.2f}%")
    logger.info(f"최저 정확도: {min(accuracies):.2f}%")
    
    # 결과를 JSON 파일로 저장
    results_file = os.path.join(cfg.OUTPUT_DIR, 'kfold_results.json')
    kfold_results['summary'] = {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'max_accuracy': max(accuracies),
        'min_accuracy': min(accuracies),
        'all_accuracies': accuracies
    }
    
    with open(results_file, 'w') as f:
        json.dump(kfold_results, f, indent=2)
    
    logger.info(f"K-fold 실험 결과가 {results_file}에 저장되었습니다.")
    
    return mean_acc, std_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml', help='실험에 사용할 config yaml 경로')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    args = parser.parse_args()
    
    mean_acc, std_acc = main(args)
    print(f"\n최종 K-fold 실험 결과: {mean_acc:.2f}% ± {std_acc:.2f}%")

"""
K-fold 실험 사용법:

1. 기본 7-fold 실험:
   python tool/train_kfold.py --cfg experiments/your_config.yaml

2. 시드 변경:
   python tool/train_kfold.py --cfg experiments/your_config.yaml --seed 123

3. 결과 확인:
   - 각 fold의 결과는 cfg.OUTPUT_DIR/fold_X/ 에 저장됩니다
   - 전체 결과 요약은 cfg.OUTPUT_DIR/kfold_results.json 에 저장됩니다
   - wandb에서 각 fold별 실험을 확인할 수 있습니다

4. 특징:
   - 7-fold 교차 검증이 자동으로 수행됩니다
   - 각 fold에서 train/val/test 분할이 자동으로 이루어집니다
   - 모든 fold의 결과가 자동으로 요약됩니다
""" 
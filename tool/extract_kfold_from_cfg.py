#!/usr/bin/env python3
"""
cfg 파일과 log 파일을 사용해서 k-fold test indices를 추출하는 스크립트
"""

import _init_path
import json
import logging
import argparse
import re
import numpy as np
from pathlib import Path

def extract_log_indices(log_file_path):
    """
    로그 파일에서 각 fold의 test indices를 추출합니다.
    """
    if not Path(log_file_path).exists():
        print(f"로그 파일을 찾을 수 없습니다: {log_file_path}")
        return {}
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # fold별 test indices 패턴 찾기
    fold_pattern = r'Fold (\d+) Test Set:\s*\n.*?Test indices: \[(.*?)\]\.\.\.\s*\n.*?Test set 크기: (\d+)'
    matches = re.findall(fold_pattern, log_content, re.DOTALL)
    
    log_indices = {}
    
    for fold_num, indices_str, size in matches:
        fold_num = int(fold_num)
        size = int(size)
        
        # indices 문자열을 리스트로 변환
        try:
            indices = [int(x) for x in indices_str.split() if x.strip().isdigit()]
            log_indices[f'fold_{fold_num}'] = {
                'indices': indices,
                'size': size
            }
            print(f"로그에서 Fold {fold_num}: {len(indices)}개 indices 추출 (예상 크기: {size})")
        except Exception as e:
            print(f"로그 파싱 오류 Fold {fold_num}: {e}")
    
    return log_indices

def compare_indices(cfg_indices, log_indices):
    """
    cfg에서 생성한 indices와 로그의 indices를 비교합니다.
    로그의 첫 10개가 cfg indices에 포함되어 있는지 확인합니다.
    """
    print("\n=== 로그와 비교 결과 ===")
    
    if not log_indices:
        print("로그에서 indices를 추출할 수 없습니다.")
        return
    
    for fold_name in cfg_indices.keys():
        if fold_name in log_indices:
            cfg_size = cfg_indices[fold_name]['size']
            log_size = log_indices[fold_name]['size']
            cfg_indices_list = set(cfg_indices[fold_name]['indices'])
            log_indices_list = log_indices[fold_name]['indices']  # 리스트로 유지
            
            print(f"\n{fold_name}:")
            print(f"  CFG 크기: {cfg_size}, 로그 크기: {log_size}")
            
            if cfg_size == log_size:
                print(f"  ✓ 크기 일치")
            else:
                print(f"  ⚠ 크기 불일치")
            
            # 로그의 첫 10개 indices
            log_first_10 = log_indices_list[:10]
            print(f"  로그 첫 10개: {log_first_10}")
            
            # 로그의 첫 10개가 cfg indices에 포함되어 있는지 확인
            included_count = 0
            for idx in log_first_10:
                if idx in cfg_indices_list:
                    included_count += 1
            
            inclusion_ratio = included_count / len(log_first_10) * 100
            print(f"  로그 첫 10개 중 CFG에 포함된 개수: {included_count}/10 ({inclusion_ratio:.1f}%)")
            
            if inclusion_ratio == 100:
                print(f"  ✓ 로그 첫 10개 모두 CFG에 포함됨 - 올바른 fold!")
            elif inclusion_ratio > 80:
                print(f"  ⚠ 로그 첫 10개 대부분 CFG에 포함됨 - 높은 일치도")
            elif inclusion_ratio > 50:
                print(f"  ⚠ 로그 첫 10개 절반 정도 CFG에 포함됨 - 중간 일치도")
            else:
                print(f"  ❌ 로그 첫 10개 대부분 CFG에 포함되지 않음 - 낮은 일치도")
            
            # CFG에서 해당 fold의 첫 10개 indices도 표시
            cfg_first_10 = list(cfg_indices_list)[:10]
            print(f"  CFG 첫 10개: {cfg_first_10}")
            
        else:
            print(f"\n{fold_name}: 로그에서 찾을 수 없음")

def convert_numpy_types(obj):
    """
    numpy 타입을 Python 기본 타입으로 변환하여 JSON 직렬화 가능하게 만듭니다.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

def main():
    try:
        # 필요한 모듈들 import
        from config import cfg, update_config
        from dataset import MedicalImageDataset
        from utils.utils import set_seed, create_kfold_splits, verify_fold_class_distribution
        
        # argument parser
        parser = argparse.ArgumentParser(description='K-fold test indices 추출')
        parser.add_argument('--cfg', type=str, default='experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_vgg19bn_kfold.yaml',
                          help='설정 파일 경로')
        parser.add_argument('--log', type=str, default='log/224/oa_normal_hand_vgg19bn_kfold.log',
                          help='로그 파일 경로')
        parser.add_argument('--output_dir', type=str, default='kfold_test_indices',
                          help='결과 저장 디렉토리')
        parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
        args = parser.parse_args()
        
        print("K-fold test indices 추출 중...")
        print("=" * 50)
        print(f"설정 파일: {args.cfg}")
        print(f"로그 파일: {args.log}")
        print(f"시드: {args.seed}")
        
        # 설정 파일 로드
        if not Path(args.cfg).exists():
            print(f"설정 파일을 찾을 수 없습니다: {args.cfg}")
            return
        
        # cfg 업데이트
        update_config(cfg, args)
        
        # 기본 설정
        set_seed(args.seed)
        
        # 데이터셋 생성
        dataset = MedicalImageDataset(cfg)
        target_classes = cfg.DATASET.INCLUDE_CLASSES or cfg.DATASET.TARGET_CLASSES
        
        print(f"원본 데이터셋 크기: {len(dataset)}")
        print(f"타겟 클래스: {target_classes}")
        print(f"JSON 파일: {cfg.DATASET.JSON}")
        
        # 클래스 불균형 해결
        target_count = getattr(cfg.DATASET, 'TARGET_COUNT_PER_CLASS', None)
        if target_count:
            print(f"클래스당 목표 샘플 수: {target_count}")
            dataset.balance_dataset(target_count_per_class=target_count)
            print(f"균등화 후 데이터셋 크기: {len(dataset)}")
        
        # K-fold 분할 생성
        kfold_size = cfg.KFOLD.KFOLD_SIZE
        print(f"\nK-fold 분할 생성 (k={kfold_size})")
        fold_splits = create_kfold_splits(dataset, n_splits=kfold_size, random_state=args.seed)
        
        # 각 fold의 분할 결과 로깅
        print("K-fold 분할 결과:")
        all_test_indices = set()
        fold_test_indices = {}
        
        for fold_info in fold_splits:
            fold_idx = fold_info['fold']
            train_size = len(fold_info['train_indices'])
            val_size = len(fold_info['val_indices'])
            test_size = len(fold_info['test_indices'])
            total_size = train_size + val_size + test_size
            
            print(f"Fold {fold_idx}: Train={train_size}, Val={val_size}, Test={test_size}, Total={total_size}")
            
            # test indices를 수집하여 중복 확인
            test_indices = set(fold_info['test_indices'])
            all_test_indices.update(test_indices)
            
            # JSON 저장을 위해 indices를 리스트로 변환하고 numpy 타입 변환
            fold_test_indices[f'fold_{fold_idx}'] = {
                'indices': convert_numpy_types(list(fold_info['test_indices'])),
                'size': test_size
            }
        
        # 각 fold의 클래스 분포 검증
        print("\n=== 클래스 분포 검증 ===")
        verify_fold_class_distribution(fold_splits, dataset, None)
        
        # K-fold 교차 검증 검증
        total_data_size = len(dataset)
        expected_test_size_per_fold = total_data_size // kfold_size
        print(f"\nK-fold 교차 검증 검증:")
        print(f"전체 데이터 크기: {total_data_size}")
        print(f"fold당 예상 test 크기: {expected_test_size_per_fold}")
        print(f"실제 test set에 포함된 고유 데이터 수: {len(all_test_indices)}")
        print(f"모든 데이터가 test set에 포함됨: {len(all_test_indices) == total_data_size}")
        
        if len(all_test_indices) != total_data_size:
            print("경고: 일부 데이터가 test set에 포함되지 않았습니다!")
        else:
            print("✓ 모든 데이터가 정확히 한 번씩 test set에 포함되었습니다.")
        
        # 로그에서 indices 추출
        print(f"\n=== 로그 파일에서 indices 추출 ===")
        log_indices = extract_log_indices(args.log)
        
        # 로그와 비교
        compare_indices(fold_test_indices, log_indices)
        
        # 결과 디렉토리 생성
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 실험 이름 추출 (설정 파일 경로에서)
        cfg_path = Path(args.cfg)
        experiment_name = cfg_path.stem  # 파일명에서 확장자 제거
        
        # JSON 파일로 저장 (실험 이름 포함)
        output_file = f"{args.output_dir}/kfold_test_indices_{experiment_name}_seed_{args.seed}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fold_test_indices, f, indent=2, ensure_ascii=False)
        
        print(f"\nK-fold test indices가 {output_file}에 저장되었습니다.")
        print(f"실험 이름: {experiment_name}")
        
        # 요약 정보 출력
        print("\n=== 요약 정보 ===")
        total_test_samples = sum(fold_info['size'] for fold_info in fold_test_indices.values())
        print(f"전체 test 샘플 수: {total_test_samples}")
        
        for fold_name, fold_info in fold_test_indices.items():
            print(f"{fold_name}: {fold_info['size']}개 샘플")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
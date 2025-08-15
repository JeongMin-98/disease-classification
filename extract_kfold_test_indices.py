#!/usr/bin/env python3
"""
로그 파일에서 k-fold test indices를 추출하는 스크립트
"""

import re
import json
import numpy as np
from pathlib import Path

def extract_kfold_test_indices(log_file_path):
    """
    로그 파일에서 각 fold의 test indices를 추출합니다.
    
    Args:
        log_file_path (str): 로그 파일 경로
        
    Returns:
        dict: fold별 test indices를 담은 딕셔너리
    """
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # fold별 test indices 패턴 찾기
    fold_pattern = r'Fold (\d+) Test Set:\s*\n.*?Test indices: \[(.*?)\]\.\.\.\s*\n.*?Test set 크기: (\d+)'
    matches = re.findall(fold_pattern, log_content, re.DOTALL)
    
    fold_test_indices = {}
    
    for fold_num, indices_str, size in matches:
        fold_num = int(fold_num)
        size = int(size)
        
        # indices 문자열을 numpy 배열로 변환
        try:
            # 공백과 숫자만 추출하여 배열 생성
            indices = np.array([int(x) for x in indices_str.split() if x.strip().isdigit()])
            
            # 실제 크기와 일치하는지 확인
            if len(indices) == size:
                fold_test_indices[f'fold_{fold_num}'] = {
                    'indices': indices.tolist(),
                    'size': size
                }
                print(f"Fold {fold_num}: {size}개 샘플, indices: {indices[:10].tolist()}...")
            else:
                print(f"Warning: Fold {fold_num}의 indices 개수 불일치 - 예상: {size}, 실제: {len(indices)}")
                
        except Exception as e:
            print(f"Error parsing Fold {fold_num}: {e}")
            continue
    
    return fold_test_indices

def save_kfold_indices(fold_test_indices, output_file):
    """
    추출된 k-fold indices를 JSON 파일로 저장합니다.
    
    Args:
        fold_test_indices (dict): fold별 test indices
        output_file (str): 출력 파일 경로
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fold_test_indices, f, indent=2, ensure_ascii=False)
    
    print(f"\nK-fold test indices가 {output_file}에 저장되었습니다.")

def main():
    # 로그 파일 경로
    log_file = "log/224/oa_normal_hand_vgg19bn_kfold.log"
    
    if not Path(log_file).exists():
        print(f"로그 파일을 찾을 수 없습니다: {log_file}")
        return
    
    print("K-fold test indices 추출 중...")
    print("=" * 50)
    
    # test indices 추출
    fold_test_indices = extract_kfold_test_indices(log_file)
    
    if not fold_test_indices:
        print("추출된 test indices가 없습니다.")
        return
    
    print(f"\n총 {len(fold_test_indices)}개 fold의 test indices를 추출했습니다.")
    
    # JSON 파일로 저장
    output_file = "kfold_test_indices.json"
    save_kfold_indices(fold_test_indices, output_file)
    
    # 요약 정보 출력
    print("\n=== 요약 정보 ===")
    total_test_samples = sum(fold_info['size'] for fold_info in fold_test_indices.values())
    print(f"전체 test 샘플 수: {total_test_samples}")
    
    for fold_name, fold_info in fold_test_indices.items():
        print(f"{fold_name}: {fold_info['size']}개 샘플")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Wandb Sweep 사용 예시 스크립트
다양한 설정으로 sweep 실험을 실행하는 방법을 보여줍니다.
"""

import subprocess
import sys
import json

def run_sweep_command(cmd_args):
    """sweep 명령어를 실행하는 함수"""
    cmd = ["python", "train_sweep.py"] + cmd_args
    print(f"실행 명령어: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"반환 코드: {result.returncode}")
    except Exception as e:
        print(f"명령어 실행 중 오류: {e}")
    
    print("=" * 50)

def main():
    """다양한 sweep 설정 예시들을 실행"""
    
    print("Wandb Sweep 사용 예시")
    print("=" * 50)
    
    # 1. 기본 설정으로 빠른 테스트
    print("\n1. 빠른 테스트 (quick_test 설정)")
    run_sweep_command([
        "--cfg", "experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml",
        "--sweep_config", "quick_test",
        "--count", "4",
        "--project", "sweep-example"
    ])
    
    # 2. 학습률 최적화
    print("\n2. 학습률 최적화")
    run_sweep_command([
        "--cfg", "experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml",
        "--sweep_config", "learning_rate",
        "--count", "10",
        "--project", "sweep-example"
    ])
    
    # 3. 배치 크기 최적화
    print("\n3. 배치 크기 최적화")
    run_sweep_command([
        "--cfg", "experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml",
        "--sweep_config", "batch_size",
        "--count", "5",
        "--project", "sweep-example"
    ])
    
    # 4. 데이터 균등화 효과 테스트
    print("\n4. 데이터 균등화 효과 테스트")
    run_sweep_command([
        "--cfg", "experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml",
        "--sweep_config", "data_balancing",
        "--count", "10",
        "--project", "sweep-example"
    ])
    
    # 5. 커스텀 설정 예시
    print("\n5. 커스텀 설정 예시")
    custom_params = {
        'TRAIN.LR': {'values': [0.001, 0.01]},
        'TRAIN.BATCH_SIZE_PER_GPU': {'values': [16, 32]},
        'TRAIN.OPTIMIZER': {'values': ['adam']},
        'TRAIN.END_EPOCH': {'values': [20]}
    }
    
    run_sweep_command([
        "--cfg", "experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml",
        "--method", "grid",
        "--sweep_name", "custom_test",
        "--metric", "test_accuracy",
        "--custom_params", json.dumps(custom_params),
        "--count", "4",
        "--project", "sweep-example"
    ])
    
    # 6. 전체 최적화 (대규모 실험)
    print("\n6. 전체 하이퍼파라미터 최적화 (대규모)")
    run_sweep_command([
        "--cfg", "experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml",
        "--sweep_config", "full_optimization",
        "--count", "50",
        "--project", "sweep-example"
    ])

if __name__ == "__main__":
    print("이 스크립트는 sweep 설정 예시들을 보여줍니다.")
    print("실제로 실행하려면 아래 명령어 중 하나를 선택하세요:\n")
    
    print("1. 빠른 테스트:")
    print("python train_sweep.py --sweep_config quick_test --count 4")
    
    print("\n2. 학습률 최적화:")
    print("python train_sweep.py --sweep_config learning_rate --count 10")
    
    print("\n3. 배치 크기 최적화:")
    print("python train_sweep.py --sweep_config batch_size --count 5")
    
    print("\n4. 데이터 균등화 테스트:")
    print("python train_sweep.py --sweep_config data_balancing --count 10")
    
    print("\n5. 커스텀 설정:")
    print('python train_sweep.py --method grid --sweep_name custom_test --custom_params \'{"TRAIN.LR": {"values": [0.001, 0.01]}}\' --count 4')
    
    print("\n6. 전체 최적화:")
    print("python train_sweep.py --sweep_config full_optimization --count 50")
    
    print("\n사용 가능한 sweep 설정들:")
    print("- base: 기본 설정")
    print("- quick_test: 빠른 테스트용")
    print("- learning_rate: 학습률 최적화")
    print("- batch_size: 배치 크기 최적화")
    print("- data_balancing: 데이터 균등화 테스트")
    print("- full_optimization: 전체 하이퍼파라미터 최적화") 
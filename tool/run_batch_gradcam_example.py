#!/usr/bin/env python3
"""
배치 GradCAM 사용 예시 스크립트
"""

import os
import subprocess
import sys

def run_batch_gradcam_example():
    """배치 GradCAM 실행 예시"""
    
    # 예시 명령어들
    examples = [
        {
            "description": "기본 배치 GradCAM 분석 (클래스당 32개씩)",
            "command": [
                "python", "batch_gradcam_analysis.py",
                "--checkpoint", "wandb/run-20241201_123456/best_model.pth",
                "--cfg", "experiments/image_exp/foot/vgg/foot_classifier_filtered_oa_normal_0702_sgdm_freeze.yaml",
                "--output_dir", "batch_gradcam_results",
                "--max_images_per_class", "32"
            ]
        },
        {
            "description": "클래스당 16개씩 분석",
            "command": [
                "python", "batch_gradcam_analysis.py",
                "--checkpoint", "wandb/run-20241201_123456/best_model.pth",
                "--cfg", "experiments/image_exp/foot/vgg/foot_classifier_filtered_oa_normal_0702_sgdm_freeze.yaml",
                "--output_dir", "batch_gradcam_results_16",
                "--max_images_per_class", "16"
            ]
        },
        {
            "description": "CPU에서 실행",
            "command": [
                "python", "batch_gradcam_analysis.py",
                "--checkpoint", "wandb/run-20241201_123456/best_model.pth",
                "--cfg", "experiments/image_exp/foot/vgg/foot_classifier_filtered_oa_normal_0702_sgdm_freeze.yaml",
                "--output_dir", "batch_gradcam_results_cpu",
                "--device", "cpu",
                "--max_images_per_class", "32"
            ]
        },
        {
            "description": "다른 랜덤 시드 사용",
            "command": [
                "python", "batch_gradcam_analysis.py",
                "--checkpoint", "wandb/run-20241201_123456/best_model.pth",
                "--cfg", "experiments/image_exp/foot/vgg/foot_classifier_filtered_oa_normal_0702_sgdm_freeze.yaml",
                "--output_dir", "batch_gradcam_results_seed123",
                "--seed", "123",
                "--max_images_per_class", "32"
            ]
        }
    ]
    
    print("=== 배치 GradCAM 사용 예시 ===\n")
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print("   명령어:")
        print("   " + " ".join(example['command']))
        print()
    
    print("=== 사용 방법 ===")
    print("1. 위의 명령어에서 경로들을 실제 경로로 수정하세요.")
    print("2. --checkpoint: wandb local 저장 위치의 best_model.pth 파일 경로")
    print("   예: wandb/run-20241201_123456/best_model.pth")
    print("3. --cfg: 사용한 설정 파일 경로")
    print("4. --output_dir: 결과 저장 디렉토리 (기본값: batch_gradcam_results)")
    print("5. --max_images_per_class: 클래스당 최대 이미지 수 (기본값: 32)")
    print("6. --device: 사용할 디바이스 (cuda/cpu, 기본값: cuda)")
    print("7. --seed: 랜덤 시드 (기본값: 42)")
    print()
    
    print("=== 주요 개선사항 ===")
    print("✓ 훈련 시와 동일한 테스트 데이터셋 사용")
    print("✓ 동일한 데이터 분할 방식 (Stratified Split)")
    print("✓ 동일한 데이터 균등화 적용")
    print("✓ 실제 이미지 파일 경로 추출")
    print("✓ 테스트 셋에서 올바르게 분류된 이미지만 선택")
    print()
    
    print("=== 결과 구조 ===")
    print("결과는 다음과 같은 구조로 저장됩니다:")
    print("batch_gradcam_results/")
    print("├── oa/")
    print("│   ├── 000_이미지명_gradcam.png")
    print("│   ├── 001_이미지명_gradcam.png")
    print("│   └── ...")
    print("├── normal/")
    print("│   ├── 000_이미지명_gradcam.png")
    print("│   ├── 001_이미지명_gradcam.png")
    print("│   └── ...")
    print("└── summary.json")
    print()
    
    print("=== 각 GradCAM 이미지 내용 ===")
    print("각 GradCAM 결과 이미지는 다음을 포함합니다:")
    print("- 원본 이미지 (True 클래스 표시)")
    print("- GradCAM 히트맵 (타겟 클래스에 대한)")
    print("- 오버레이된 이미지 (True/Pred 클래스 및 신뢰도)")
    print()
    
    print("=== 주의사항 ===")
    print("- 테스트 데이터셋에서 올바르게 분류된 이미지만 선택됩니다.")
    print("- OA와 Normal 클래스 각각 지정된 수만큼 이미지를 선택합니다.")
    print("- 이미지 선택은 배치 순서대로 진행되며, 충분한 이미지를 찾으면 중단됩니다.")
    print("- 랜덤 시드를 설정하면 동일한 이미지들이 선택됩니다.")
    print("- 훈련 시와 동일한 데이터 분할을 사용하므로 일관성 있는 결과를 얻을 수 있습니다.")

if __name__ == '__main__':
    run_batch_gradcam_example() 
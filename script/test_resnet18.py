#!/usr/bin/env python3
"""
ResNet18 모델 테스트 스크립트
모델이 제대로 로드되고 forward pass가 작동하는지 확인합니다.
"""

import sys
import os
sys.path.append('lib')

import torch
from models import create

def test_resnet18():
    print("=" * 50)
    print("ResNet18 Model Test")
    print("=" * 50)
    
    # 사용 가능한 모델 확인
    from models import names
    print(f"Available models: {names()}")
    
    # ResNet18 모델 생성 테스트
    print("\n1. Testing ResNet18 creation...")
    try:
        # 이진 분류용 (2클래스 -> 1 출력)
        model_binary = create('ResNet18', Target_Classes=['oa', 'normal'], pretrained=True)
        print("✅ ResNet18 binary classification model created successfully")
        print(f"   Output features: {model_binary.model.fc.out_features}")
        
        # 다중 분류용 (3클래스)
        model_multi = create('ResNet18', Target_Classes=['class1', 'class2', 'class3'], pretrained=True)
        print("✅ ResNet18 multi-class model created successfully")
        print(f"   Output features: {model_multi.model.fc.out_features}")
        
        # Freeze 레이어 테스트 (conv1, bn1, layer1)
        model_frozen = create('ResNet18', Target_Classes=['oa', 'normal'], pretrained=True, freeze_layers=[1, 3])
        print("✅ ResNet18 with frozen layers (conv1, bn1, layer1) created successfully")
        
    except Exception as e:
        print(f"❌ Error creating ResNet18: {e}")
        return False
    
    # Forward pass 테스트
    print("\n2. Testing forward pass...")
    try:
        # 가짜 입력 데이터 생성 (batch_size=2, channels=3, height=224, width=224)
        dummy_input = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output_binary = model_binary(dummy_input)
            output_multi = model_multi(dummy_input)
            
        print(f"✅ Binary model output shape: {output_binary.shape}")
        print(f"✅ Multi-class model output shape: {output_multi.shape}")
        
        # 출력 형태 확인
        assert output_binary.shape == (2, 1), f"Expected (2, 1), got {output_binary.shape}"
        assert output_multi.shape == (2, 3), f"Expected (2, 3), got {output_multi.shape}"
        
        print("✅ Forward pass successful!")
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        return False
    
    # 파라미터 수 확인
    print("\n3. Model parameters:")
    total_params = sum(p.numel() for p in model_binary.parameters())
    trainable_params = sum(p.numel() for p in model_binary.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Frozen 모델의 파라미터 확인
    frozen_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"   Frozen model trainable parameters: {frozen_trainable:,}")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! ResNet18 is ready to use.")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_resnet18()
    if not success:
        sys.exit(1) 
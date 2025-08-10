#!/bin/bash

# 7-fold 실험 실행 스크립트
# 모든 실험이 순차적으로 실행되며, OOM 방지를 위해 배치 크기를 조정했습니다.

echo "=== 7-fold 손 발 질환 이진분류기 실험 시작 ==="
echo "실험 순서:"
echo "1. 224x224 VGG19"
echo "2. 224x224 ResNet18"
echo "3. 1024x1024 VGG19"
echo "4. 1024x1024 ResNet18"
echo "=========================="

# 기본값 설정
SEED=${1:-42}

echo "시드: $SEED"
echo "=========================="

# 1. 224x224 VGG19 실험
echo "=== 1. 224x224 VGG19 실험 시작 ==="
python tool/train_kfold.py --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_vgg19_kfold.yaml --seed $SEED
if [ $? -ne 0 ]; then
    echo "224x224 VGG19 실험 실패"
    exit 1
fi
echo "=== 1. 224x224 VGG19 실험 완료 ==="

# 2. 224x224 ResNet18 실험
echo "=== 2. 224x224 ResNet18 실험 시작 ==="
python tool/train_kfold.py --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_resnet18_kfold.yaml --seed $SEED
if [ $? -ne 0 ]; then
    echo "224x224 ResNet18 실험 실패"
    exit 1
fi
echo "=== 2. 224x224 ResNet18 실험 완료 ==="

# 3. 1024x1024 VGG19 실험
echo "=== 3. 1024x1024 VGG19 실험 시작 ==="
python tool/train_kfold.py --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_1024_vgg19_kfold.yaml --seed $SEED
if [ $? -ne 0 ]; then
    echo "1024x1024 VGG19 실험 실패"
    exit 1
fi
echo "=== 3. 1024x1024 VGG19 실험 완료 ==="

# 4. 1024x1024 ResNet18 실험
echo "=== 4. 1024x1024 ResNet18 실험 시작 ==="
python tool/train_kfold.py --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_1024_resnet18_kfold.yaml --seed $SEED
if [ $? -ne 0 ]; then
    echo "1024x1024 ResNet18 실험 실패"
    exit 1
fi
echo "=== 4. 1024x1024 ResNet18 실험 완료 ==="

echo "=== 모든 7-fold 실험 완료 ==="
echo "결과는 다음 디렉토리에 저장되었습니다:"
echo "- experiments/results/kfold_224_vgg19_oa_normal"
echo "- experiments/results/kfold_224_resnet18_oa_normal"
echo "- experiments/results/kfold_1024_vgg19_oa_normal"
echo "- experiments/results/kfold_1024_resnet18_oa_normal"
echo "각 디렉토리에는 fold_0, fold_1, ..., fold_6 하위 디렉토리가 있습니다." 
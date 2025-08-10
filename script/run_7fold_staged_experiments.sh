#!/bin/bash

# 7-fold 실험 단계별 병렬 실행 스크립트
# 배치 크기: 16으로 통일
# 단계별 실행: VGG19BN 224x224 -> ResNet18 224x224 -> VGG19BN 1024x1024 -> ResNet18 1024x1024

echo "=== 7-fold 실험 단계별 병렬 실행 시작 ==="
echo "실험 순서:"
echo "1. 224x224 VGG19BN (손 발 질환)"
echo "2. 224x224 ResNet18 (손 발 질환)"
echo "3. 1024x1024 VGG19BN (손 발 질환)"
echo "4. 1024x1024 ResNet18 (손 발 질환)"
echo "=========================="

# 기본값 설정
MAX_WORKERS=${1:-2}
SEED=${2:-42}

echo "최대 동시 실행 수: $MAX_WORKERS"
echo "시드: $SEED"
echo "=========================="

# 의존성 패키지 설치 확인
echo "의존성 패키지 확인 중..."
python -c "import GPUtil" 2>/dev/null || {
    echo "GPUtil이 설치되지 않았습니다. 설치를 진행합니다..."
    pip install GPUtil
}

python -c "import psutil" 2>/dev/null || {
    echo "psutil이 설치되지 않았습니다. 설치를 진행합니다..."
    pip install psutil
}

# 실험 실행
echo "=== Python 스크립트로 실험 시작 ==="
python run_parallel_7fold_experiments.py --max-workers $MAX_WORKERS --seed $SEED

if [ $? -eq 0 ]; then
    echo "=== 모든 실험 완료 ==="
    echo "결과는 다음 디렉토리에 저장되었습니다:"
    echo "- experiments/results/kfold_224_vgg19bn_oa_normal"
    echo "- experiments/results/kfold_224_resnet18_oa_normal"
    echo "- experiments/results/kfold_1024_vgg19bn_oa_normal"
    echo "- experiments/results/kfold_1024_resnet18_oa_normal"
    echo ""
    echo "상세 결과는 experiments/results/ 디렉토리의 JSON 파일을 확인하세요."
else
    echo "=== 실험 실행 중 오류 발생 ==="
    exit 1
fi 
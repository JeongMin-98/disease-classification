#!/bin/bash

# 7-fold 실험 단계별 병렬 실행 스크립트 (nohup 백그라운드 실행)
# 배치 크기: 16으로 통일
# 단계별 실행: 각 단계에서 손과 발 데이터를 동시에 병렬로 실행
# 1. 224x224 VGG19BN (손 + 발 동시 실행)
# 2. 224x224 ResNet18 (손 + 발 동시 실행)
# 3. 1024x1024 VGG19BN (손 + 발 동시 실행)
# 4. 1024x1024 ResNet18 (손 + 발 동시 실행)

echo "=== 7-fold 실험 단계별 병렬 실행 시작 (백그라운드) ==="
echo "실험 순서 (각 단계에서 손과 발 데이터를 동시에 병렬로 실행):"
echo "1. 224x224 VGG19BN (손 + 발 동시 실행)"
echo "2. 224x224 ResNet18 (손 + 발 동시 실행)"
echo "3. 1024x1024 VGG19BN (손 + 발 동시 실행)"
echo "4. 1024x1024 ResNet18 (손 + 발 동시 실행)"
echo "=========================="

# 기본값 설정
MAX_WORKERS=${1:-2}
SEED=${2:-42}

echo "최대 동시 실행 수: $MAX_WORKERS"
echo "시드: $SEED"
echo "=========================="

# 타임스탬프 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 로그 파일 경로 설정
STDOUT_LOG="$LOG_DIR/experiments_stdout_${TIMESTAMP}.log"
STDERR_LOG="$LOG_DIR/experiments_stderr_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/experiments_pid_${TIMESTAMP}.txt"

echo "로그 파일:"
echo "- 표준 출력: $STDOUT_LOG"
echo "- 표준 오류: $STDERR_LOG"
echo "- 프로세스 ID: $PID_FILE"
echo "- 실험별 상세 로그: $LOG_DIR/ (각 실험별로 별도 파일 생성)"
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

# 실험 실행 (nohup으로 백그라운드 실행)
echo "=== Python 스크립트로 실험 시작 (백그라운드) ==="
echo "실험을 백그라운드에서 실행합니다. SSH 연결이 끊어져도 계속 진행됩니다."
echo "각 단계에서 손과 발 데이터가 동시에 병렬로 실행됩니다."
echo ""

nohup python run_parallel_7fold_experiments.py --max-workers $MAX_WORKERS --seed $SEED > $STDOUT_LOG 2> $STDERR_LOG &

# 프로세스 ID 저장
EXPERIMENT_PID=$!
echo $EXPERIMENT_PID > $PID_FILE

echo "=== 실험이 백그라운드에서 시작되었습니다 ==="
echo "프로세스 ID: $EXPERIMENT_PID"
echo "PID 파일: $PID_FILE"
echo ""
echo "실험 진행 상황을 확인하려면:"
echo "1. 실시간 로그 확인: tail -f $STDOUT_LOG"
echo "2. 오류 로그 확인: tail -f $STDERR_LOG"
echo "3. 프로세스 상태 확인: ps aux | grep $EXPERIMENT_PID"
echo "4. 실험 중단: kill $EXPERIMENT_PID"
echo ""
echo "=== 실험별 상세 로그 확인 방법 ==="
echo "각 실험의 상세 진행사항은 다음 파일들에서 확인할 수 있습니다:"
echo "- 메인 로그: $LOG_DIR/main_experiments.log"
echo "- 실험별 로그: $LOG_DIR/[실험명].log"
echo ""
echo "실시간 실험별 로그 확인 예시:"
echo "  tail -f $LOG_DIR/224x224_VGG19BN_Hand_OA_Normal.log"
echo "  tail -f $LOG_DIR/224x224_VGG19BN_Foot_OA_Normal.log"
echo ""
echo "실험 완료 후 결과는 다음 디렉토리에 저장됩니다:"
echo "=== 손 데이터 ==="
echo "- experiments/results/kfold_224_vgg19bn_hand_oa_normal"
echo "- experiments/results/kfold_224_resnet18_hand_oa_normal"
echo "- experiments/results/kfold_1024_vgg19bn_hand_oa_normal"
echo "- experiments/results/kfold_1024_resnet18_hand_oa_normal"
echo ""
echo "=== 발 데이터 ==="
echo "- experiments/results/kfold_224_vgg19bn_foot_oa_normal"
echo "- experiments/results/kfold_224_resnet18_foot_oa_normal"
echo "- experiments/results/kfold_1024_vgg19bn_foot_oa_normal"
echo "- experiments/results/kfold_1024_resnet18_foot_oa_normal"
echo ""
echo "상세 결과는 experiments/results/ 디렉토리의 JSON 파일을 확인하세요."
echo ""
echo "=== 백그라운드 실행 완료 ===" 
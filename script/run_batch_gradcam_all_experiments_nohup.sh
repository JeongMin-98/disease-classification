#!/bin/bash

# SSH 연결이 끊어져도 계속 실행되는 전체 실험 일괄 GradCAM 실행 스크립트 (nohup 버전)
# 8개 실험을 순차적으로 실행

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 시작 시간 기록
START_TIME=$(date)
log_info "nohup 전체 실험 일괄 GradCAM 실행 시작: $START_TIME"

# 기본 설정
BASE_DIR=$(pwd)
TOOL_DIR="$BASE_DIR/tool"
EXPERIMENTS_DIR="$BASE_DIR/experiments/image_exp"
LOG_DIR="$BASE_DIR/log"
OUTPUT_BASE_DIR="$BASE_DIR/batch_gradcam_results"
KFOLD_INDICES_FILE="$BASE_DIR/kfold_test_indices_seed_42.json"
WANDB_DIR="$BASE_DIR/wandb"
DEVICE="cuda"
SEED=42
NOHUP_LOG_DIR="nohup_logs"

# 필요한 디렉토리 확인
log_info "필요한 디렉토리 및 파일 확인 중..."

if [ ! -d "$TOOL_DIR" ]; then
    log_error "tool 디렉토리를 찾을 수 없습니다: $TOOL_DIR"
    exit 1
fi

if [ ! -f "$KFOLD_INDICES_FILE" ]; then
    log_error "K-fold indices 파일을 찾을 수 없습니다: $KFOLD_INDICES_FILE"
    log_info "extract_kfold_from_cfg.py를 먼저 실행하여 indices를 생성하세요."
    exit 1
fi

if [ ! -d "$WANDB_DIR" ]; then
    log_warning "wandb 디렉토리를 찾을 수 없습니다: $WANDB_DIR"
    log_info "wandb 디렉토리가 없으면 모델을 찾을 수 없을 수 있습니다."
fi

# 디렉토리 생성
mkdir -p "$NOHUP_LOG_DIR"
mkdir -p "$OUTPUT_BASE_DIR"

# PID 파일 생성
PID_FILE="$NOHUP_LOG_DIR/gradcam_all_experiments.pid"
echo "$$" > "$PID_FILE"
log_info "메인 프로세스 PID: $$ (PID 파일: $PID_FILE)"

# 8개 실험 정의
declare -a EXPERIMENTS=(
    # (1024, 224) x (resnet, vgg) x (foot, hand)
    "1024_resnet18_foot:experiments/image_exp/foot/foot_classifier_OA_Normal_1024_resnet18_kfold.yaml:log/1024/oa_normal_foot_resnet18_kfold.log"
    "1024_resnet18_hand:experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_1024_resnet18_kfold.yaml:log/1024/oa_normal_hand_resnet18_kfold.log"
    "1024_vgg19bn_foot:experiments/image_exp/foot/foot_classifier_OA_Normal_1024_vgg19bn_kfold.yaml:log/1024/oa_normal_foot_vgg19bn_kfold.log"
    "1024_vgg19bn_hand:experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_1024_vgg19bn_kfold.yaml:log/1024/oa_normal_hand_vgg19bn_kfold.log"
    "224_resnet18_foot:experiments/image_exp/foot/foot_classifier_OA_Normal_224_resnet18_kfold.yaml:log/224/oa_normal_foot_resnet18_kfold.log"
    "224_resnet18_hand:experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_resnet18_kfold.yaml:log/224/oa_normal_hand_resnet18_kfold.log"
    "224_vgg19bn_foot:experiments/image_exp/foot/foot_classifier_OA_Normal_224_vgg19bn_kfold.yaml:log/224/oa_normal_foot_vgg19bn_kfold.log"
    "224_vgg19bn_hand:experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_vgg19bn_kfold.yaml:log/224/oa_normal_hand_vgg19bn_kfold.log"
)

# 전체 결과 요약
TOTAL_RESULTS=()
SUCCESS_COUNT=0
FAILED_COUNT=0

# 각 실험에 대해 nohup으로 GradCAM 실행
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name cfg_file log_file <<< "$experiment"
    
    log_info "=========================================="
    log_info "실험 시작: $exp_name"
    log_info "설정 파일: $cfg_file"
    log_info "로그 파일: $log_file"
    log_info "=========================================="
    
    # 파일 존재 확인
    if [ ! -f "$cfg_file" ]; then
        log_error "설정 파일을 찾을 수 없습니다: $cfg_file"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        TOTAL_RESULTS+=("$exp_name:FAILED:설정 파일 없음")
        continue
    fi
    
    if [ ! -f "$log_file" ]; then
        log_error "로그 파일을 찾을 수 없습니다: $log_file"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        TOTAL_RESULTS+=("$exp_name:FAILED:로그 파일 없음")
        continue
    fi
    
    # 출력 디렉토리 설정
    EXP_OUTPUT_DIR="$OUTPUT_BASE_DIR/$exp_name"
    mkdir -p "$EXP_OUTPUT_DIR"
    
    # nohup 로그 파일 설정
    NOHUP_LOG="$NOHUP_LOG_DIR/${exp_name}_nohup.log"
    
    # GradCAM을 nohup으로 실행
    log_info "nohup으로 GradCAM 실행 중: $exp_name"
    
    # 해당 실험의 k-fold indices 파일 경로 (설정 파일명에서 실험 이름 추출)
    cfg_path=$(basename "$cfg_file" .yaml)
    EXP_KFOLD_INDICES_FILE="$KFOLD_INDICES_DIR/kfold_test_indices_${cfg_path}_seed_${SEED}.json"
    
    if [ ! -f "$EXP_KFOLD_INDICES_FILE" ]; then
        log_error "해당 실험의 k-fold indices 파일을 찾을 수 없습니다: $EXP_KFOLD_INDICES_FILE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        TOTAL_RESULTS+=("$exp_name:FAILED:k-fold indices 파일 없음")
        continue
    fi
    
    nohup python "$TOOL_DIR/batch_gradcam_kfold.py" \
        --cfg "$cfg_file" \
        --kfold_indices "$EXP_KFOLD_INDICES_FILE" \
        --log "$log_file" \
        --wandb_dir "$WANDB_DIR" \
        --output_dir "$EXP_OUTPUT_DIR" \
        --device "$DEVICE" \
        --seed "$SEED" > "$NOHUP_LOG" 2>&1 &
    
    local_pid=$!
    echo "$local_pid" > "$NOHUP_LOG_DIR/${exp_name}_pid.txt"
    
    log_success "실험 시작 완료: $exp_name (PID: $local_pid)"
    log_info "nohup 로그: $NOHUP_LOG"
    log_info "PID 파일: $NOHUP_LOG_DIR/${exp_name}_pid.txt"
    
    TOTAL_RESULTS+=("$exp_name:STARTED:PID $local_pid")
    
    log_info "=========================================="
    echo ""
    
    # 다음 실험 시작 전 잠시 대기 (시스템 부하 방지)
    sleep 5
done

# 실행 완료 요약
log_info "=========================================="
log_info "nohup 전체 실험 실행 완료"
log_info "=========================================="
log_info "시작 시간: $START_TIME"
log_info "완료 시간: $(date)"
log_info "총 실행된 실험: ${#EXPERIMENTS[@]}개"

echo ""
log_info "실행된 실험들:"
for result in "${TOTAL_RESULTS[@]}"; do
    IFS=':' read -r exp_name status detail <<< "$result"
    if [ "$status" = "STARTED" ]; then
        echo -e "  ${GREEN}✓${NC} $exp_name: $detail"
    else
        echo -e "  ${RED}✗${NC} $exp_name: $detail"
    fi
done

echo ""
log_info "중요 정보:"
log_info "1. SSH 연결이 끊어져도 모든 작업이 계속 실행됩니다."
log_info "2. 각 실험의 PID는 $NOHUP_LOG_DIR/*_pid.txt 파일에 저장됩니다."
log_info "3. nohup 로그는 $NOHUP_LOG_DIR/*_nohup.log 파일에 저장됩니다."
log_info "4. 메인 프로세스 PID: $$ (PID 파일: $PID_FILE)"
log_info "5. 작업 상태 확인: ./check_gradcam_status.sh"
log_info "6. 실시간 모니터링: ./monitor_gradcam_progress.sh"
log_info "7. 개별 로그 확인: ./tail_gradcam_logs.sh"

# PID 파일 정리
rm -f "$PID_FILE"

log_success "모든 실험이 nohup으로 백그라운드에서 실행 중입니다!"
log_info "터미널을 닫아도 작업이 계속 실행됩니다."
log_info "각 실험의 진행 상황은 $NOHUP_LOG_DIR 폴더에서 확인할 수 있습니다." 
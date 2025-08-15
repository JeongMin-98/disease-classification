#!/bin/bash

# SSH 연결이 끊어져도 계속 실행되는 병렬 GradCAM 실행 스크립트 (nohup 버전)

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

# 사용법 출력
usage() {
    echo "사용법: $0 [OPTIONS] <experiment_names...>"
    echo ""
    echo "옵션:"
    echo "  --max-jobs <N>         최대 동시 실행 작업 수 (기본값: 2)"
    echo "  --device <device>      사용할 디바이스 (기본값: cuda)"
    echo "  --seed <seed>          랜덤 시드 (기본값: 42)"
    echo "  --output-dir <path>    출력 디렉토리 (기본값: batch_gradcam_results)"
    echo "  --log-dir <path>       nohup 로그 디렉토리 (기본값: nohup_logs)"
    echo "  --help                 이 도움말 출력"
    echo ""
    echo "사용 가능한 실험 이름:"
    echo "  1024_resnet18_foot    1024x1024 ResNet18 Foot"
    echo "  1024_resnet18_hand    1024x1024 ResNet18 Hand"
    echo "  1024_vgg19bn_foot     1024x1024 VGG19-BN Foot"
    echo "  1024_vgg19bn_hand     1024x1024 VGG19-BN Hand"
    echo "  224_resnet18_foot     224x224 ResNet18 Foot"
    echo "  224_resnet18_hand     224x224 ResNet18 Hand"
    echo "  224_vgg19bn_foot      224x224 VGG19-BN Foot"
    echo "  224_vgg19bn_hand      224x224 VGG19-BN Hand"
    echo ""
    echo "예시:"
    echo "  $0 1024_resnet18_foot 224_vgg19bn_hand"
    echo "  $0 --max-jobs 4 1024_resnet18_foot 1024_vgg19bn_foot 224_resnet18_hand"
    echo ""
    echo "주의: 이 스크립트는 nohup으로 실행되어 SSH 연결이 끊어져도 계속 실행됩니다."
}

# 기본 설정
BASE_DIR=$(pwd)
MAX_JOBS=2
DEVICE="cuda"
SEED=42
OUTPUT_DIR="batch_gradcam_results"
NOHUP_LOG_DIR="nohup_logs"

# 실험별 설정 파일과 로그 파일 매핑
declare -A EXPERIMENT_CONFIGS=(
    ["1024_resnet18_foot"]="experiments/image_exp/foot/foot_classifier_OA_Normal_1024_resnet18_kfold.yaml:log/1024/oa_normal_foot_resnet18_kfold.log"
    ["1024_resnet18_hand"]="experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_1024_resnet18_kfold.yaml:log/1024/oa_normal_hand_resnet18_kfold.log"
    ["1024_vgg19bn_foot"]="experiments/image_exp/foot/foot_classifier_OA_Normal_1024_vgg19bn_kfold.yaml:log/1024/oa_normal_foot_vgg19bn_kfold.log"
    ["1024_vgg19bn_hand"]="experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_1024_vgg19bn_kfold.yaml:log/1024/oa_normal_hand_vgg19bn_kfold.log"
    ["224_resnet18_foot"]="experiments/image_exp/foot/foot_classifier_OA_Normal_224_resnet18_kfold.yaml:log/224/oa_normal_foot_resnet18_kfold.log"
    ["224_resnet18_hand"]="experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_resnet18_kfold.yaml:log/224/oa_normal_hand_resnet18_kfold.log"
    ["224_vgg19bn_foot"]="experiments/image_exp/foot/foot_classifier_OA_Normal_224_vgg19bn_kfold.yaml:log/224/oa_normal_foot_vgg19bn_kfold.log"
    ["224_vgg19bn_hand"]="experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_vgg19bn_kfold.yaml:log/224/oa_normal_hand_vgg19bn_kfold.log"
)

# 명령행 인수 파싱
EXPERIMENT_NAMES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            NOHUP_LOG_DIR="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        -*)
            log_error "알 수 없는 옵션: $1"
            usage
            exit 1
            ;;
        *)
            EXPERIMENT_NAMES+=("$1")
            shift
            ;;
    esac
done

# 실험 이름이 지정되지 않은 경우
if [ ${#EXPERIMENT_NAMES[@]} -eq 0 ]; then
    log_error "실험 이름을 하나 이상 지정해야 합니다."
    usage
    exit 1
fi

# 실험 이름 유효성 검사
INVALID_EXPERIMENTS=()
for exp_name in "${EXPERIMENT_NAMES[@]}"; do
    if [[ ! ${EXPERIMENT_CONFIGS[$exp_name]+_} ]]; then
        INVALID_EXPERIMENTS+=("$exp_name")
    fi
done

if [ ${#INVALID_EXPERIMENTS[@]} -gt 0 ]; then
    log_error "알 수 없는 실험 이름들: ${INVALID_EXPERIMENTS[*]}"
    echo ""
    echo "사용 가능한 실험 이름:"
    for exp_name in "${!EXPERIMENT_CONFIGS[@]}"; do
        echo "  $exp_name"
    done
    exit 1
fi

# 시작 시간 기록
START_TIME=$(date)
log_info "nohup 병렬 GradCAM 실행 시작: $START_TIME"
log_info "최대 동시 실행 작업 수: $MAX_JOBS"
log_info "실행할 실험: ${EXPERIMENT_NAMES[*]}"
log_info "출력 디렉토리: $OUTPUT_DIR"
log_info "nohup 로그 디렉토리: $NOHUP_LOG_DIR"

# 필요한 파일 확인
KFOLD_INDICES_DIR="$BASE_DIR/kfold_indices"
WANDB_DIR="$BASE_DIR/wandb"

if [ ! -d "$KFOLD_INDICES_DIR" ]; then
    log_error "K-fold indices 디렉토리를 찾을 수 없습니다: $KFOLD_INDICES_DIR"
    log_info "generate_all_kfold_indices.sh를 먼저 실행하여 indices를 생성하세요."
    exit 1
fi

# 디렉토리 생성
mkdir -p "$NOHUP_LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# PID 파일 생성
PID_FILE="$NOHUP_LOG_DIR/gradcam_parallel.pid"
echo "$$" > "$PID_FILE"
log_info "메인 프로세스 PID: $$ (PID 파일: $PID_FILE)"

# 작업 실행 함수 (nohup으로 실행)
run_experiment_nohup() {
    local exp_name="$1"
    local cfg_file="$2"
    local log_file="$3"
    local output_dir="$BASE_DIR/$OUTPUT_DIR/$exp_name"
    local nohup_log="$NOHUP_LOG_DIR/${exp_name}_nohup.log"
    
    log_info "[$exp_name] nohup으로 GradCAM 실행 시작..."
    
    # 출력 디렉토리 생성
    mkdir -p "$output_dir"
    
    # 해당 실험의 k-fold indices 파일 경로 (설정 파일명에서 실험 이름 추출)
    local cfg_path=$(basename "$cfg_file" .yaml)
    local exp_kfold_indices_file="$KFOLD_INDICES_DIR/kfold_test_indices_${cfg_path}_seed_${SEED}.json"
    
    if [ ! -f "$exp_kfold_indices_file" ]; then
        log_error "[$exp_name] 해당 실험의 k-fold indices 파일을 찾을 수 없습니다: $exp_kfold_indices_file"
        return 1
    fi
    
    # nohup으로 실행
    nohup python "$BASE_DIR/tool/batch_gradcam_kfold.py" \
        --cfg "$cfg_file" \
        --kfold_indices "$exp_kfold_indices_file" \
        --log "$log_file" \
        --wandb_dir "$WANDB_DIR" \
        --output_dir "$output_dir" \
        --device "$DEVICE" \
        --seed "$SEED" > "$nohup_log" 2>&1 &
    
    local pid=$!
    echo "$pid" >> "$NOHUP_LOG_DIR/${exp_name}_pid.txt"
    
    log_success "[$exp_name] nohup 실행 완료! PID: $pid"
    log_info "[$exp_name] nohup 로그: $nohup_log"
    log_info "[$exp_name] PID 파일: $NOHUP_LOG_DIR/${exp_name}_pid.txt"
    
    return $pid
}

# 메인 실행
log_info "=========================================="
log_info "nohup 병렬 실행 시작"
log_info "=========================================="

# 각 실험을 nohup으로 실행
for exp_name in "${EXPERIMENT_NAMES[@]}"; do
    IFS=':' read -r cfg_file log_file <<< "${EXPERIMENT_CONFIGS[$exp_name]}"
    
    # 파일 존재 확인
    if [ ! -f "$cfg_file" ]; then
        log_error "설정 파일을 찾을 수 없습니다: $cfg_file"
        continue
    fi
    
    if [ ! -f "$log_file" ]; then
        log_error "로그 파일을 찾을 수 없습니다: $log_file"
        continue
    fi
    
    # nohup으로 실행
    run_experiment_nohup "$exp_name" "$cfg_file" "$log_file"
    
    # 동시 실행 작업 수 제한
    current_jobs=$(ls "$NOHUP_LOG_DIR"/*_pid.txt 2>/dev/null | wc -l)
    if [ $current_jobs -ge $MAX_JOBS ]; then
        log_info "최대 동시 실행 작업 수($MAX_JOBS)에 도달했습니다. 잠시 대기..."
        sleep 10
    fi
done

# 실행 완료 요약
log_info "=========================================="
log_info "nohup 병렬 실행 완료"
log_info "=========================================="
log_info "시작 시간: $START_TIME"
log_info "완료 시간: $(date)"
log_info "총 실행된 실험: ${#EXPERIMENT_NAMES[@]}개"
log_info "최대 동시 실행 작업 수: $MAX_JOBS"

echo ""
log_info "실행된 실험들:"
for exp_name in "${EXPERIMENT_NAMES[@]}"; do
    echo -e "  ${GREEN}✓${NC} $exp_name"
done

echo ""
log_info "중요 정보:"
log_info "1. SSH 연결이 끊어져도 작업이 계속 실행됩니다."
log_info "2. 각 실험의 PID는 $NOHUP_LOG_DIR/*_pid.txt 파일에 저장됩니다."
log_info "3. nohup 로그는 $NOHUP_LOG_DIR/*_nohup.log 파일에 저장됩니다."
log_info "4. 메인 프로세스 PID: $$ (PID 파일: $PID_FILE)"
log_info "5. 작업 상태 확인: ./check_gradcam_status.sh"
log_info "6. 실시간 모니터링: ./monitor_gradcam_progress.sh"

# PID 파일 정리
rm -f "$PID_FILE"

log_success "모든 실험이 nohup으로 백그라운드에서 실행 중입니다!"
log_info "터미널을 닫아도 작업이 계속 실행됩니다." 
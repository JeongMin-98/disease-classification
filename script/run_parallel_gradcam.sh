#!/bin/bash

# 병렬 GradCAM 실행 스크립트 (여러 실험을 동시에 실행)

set -e

# Conda 환경 활성화
echo -e "\033[0;34m[INFO]\033[0m Conda 환경 활성화 중..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disease_classification

if [ $? -ne 0 ]; then
    echo -e "\033[0;31m[ERROR]\033[0m Conda 환경 'disease_classification'을 활성화할 수 없습니다."
    echo "사용 가능한 환경 목록:"
    conda env list
    exit 1
fi

echo -e "\033[0;32m[SUCCESS]\033[0m Conda 환경 'disease_classification' 활성화 완료"
echo "Python 경로: $(which python)"
echo "Python 버전: $(python --version)"
echo ""

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

# 사용법 출력
usage() {
    echo "사용법: $0 [OPTIONS] <experiment_names...>"
    echo ""
    echo "옵션:"
    echo "  --max-jobs <N>         최대 동시 실행 작업 수 (기본값: 2)"
    echo "  --device <device>      사용할 디바이스 (기본값: cuda)"
    echo "  --seed <seed>          랜덤 시드 (기본값: 42)"
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
}

# 기본 설정
BASE_DIR=$(pwd)
MAX_JOBS=2
DEVICE="cuda"
SEED=42

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
log_info "병렬 GradCAM 실행 시작: $START_TIME"
log_info "최대 동시 실행 작업 수: $MAX_JOBS"
log_info "실행할 실험: ${EXPERIMENT_NAMES[*]}"

# 필요한 파일 확인
KFOLD_INDICES_DIR="$BASE_DIR/kfold_indices"
WANDB_DIR="$BASE_DIR/wandb"

if [ ! -d "$KFOLD_INDICES_DIR" ]; then
    log_error "K-fold indices 디렉토리를 찾을 수 없습니다: $KFOLD_INDICES_DIR"
    log_info "generate_all_kfold_indices.sh를 먼저 실행하여 indices를 생성하세요."
    exit 1
fi

# 작업 실행 함수
run_experiment() {
    local exp_name="$1"
    local cfg_file="$2"
    local log_file="$3"
    local output_dir="$BASE_DIR/batch_gradcam_results/final_exp/$exp_name"
    
    log_info "[$exp_name] GradCAM 실행 시작..."
    
    # 출력 디렉토리 생성
    mkdir -p "$output_dir"
    
    # 로그 파일 설정
    local log_file_path="$output_dir/execution.log"
    
    # 해당 실험의 k-fold indices 파일 경로 (설정 파일명에서 실험 이름 추출)
    local cfg_path=$(basename "$cfg_file" .yaml)
    local exp_kfold_indices_file="$KFOLD_INDICES_DIR/kfold_test_indices_${cfg_path}_seed_${SEED}.json"
    
    if [ ! -f "$exp_kfold_indices_file" ]; then
        log_error "[$exp_name] 해당 실험의 k-fold indices 파일을 찾을 수 없습니다: $exp_kfold_indices_file"
        return 1
    fi
    
    # GradCAM 실행
    if python "$BASE_DIR/tool/batch_gradcam_kfold.py" \
        --cfg "$cfg_file" \
        --kfold_indices "$exp_kfold_indices_file" \
        --log "$log_file" \
        --wandb_dir "$WANDB_DIR" \
        --output_dir "$output_dir" \
        --device "$DEVICE" \
        --seed "$SEED" > "$log_file_path" 2>&1; then
        
        log_success "[$exp_name] GradCAM 실행 완료!"
        
        # 결과 요약 확인
        local summary_file="$output_dir/summary.json"
        if [ -f "$summary_file" ]; then
            local total_images=$(python -c "import json; data=json.load(open('$summary_file')); print(data.get('total_images', 'N/A'))")
            local overall_accuracy=$(python -c "import json; data=json.load(open('$summary_file')); print(f\"{data.get('overall_accuracy', 'N/A'):.2f}%\" if data.get('overall_accuracy') is not None else 'N/A')")
            log_info "[$exp_name] 결과: $total_images개 이미지, 정확도: $overall_accuracy"
        fi
        
        return 0
    else
        log_error "[$exp_name] GradCAM 실행 실패! 로그: $log_file_path"
        return 1
    fi
}

# 병렬 실행
log_info "병렬 실행 시작..."

# 백그라운드 작업 추적
declare -A PIDS
declare -A RESULTS
SUCCESS_COUNT=0
FAILED_COUNT=0

# 각 실험을 백그라운드에서 실행
for exp_name in "${EXPERIMENT_NAMES[@]}"; do
    IFS=':' read -r cfg_file log_file <<< "${EXPERIMENT_CONFIGS[$exp_name]}"
    
    # 현재 실행 중인 작업 수 확인
    while [ ${#PIDS[@]} -ge $MAX_JOBS ]; do
        # 완료된 작업 확인
        for pid in "${!PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                local exp_name_pid="${PIDS[$pid]}"
                local exit_code=$?
                
                if [ $exit_code -eq 0 ]; then
                    log_success "[$exp_name_pid] 작업 완료"
                    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                    RESULTS["$exp_name_pid"]="SUCCESS"
                else
                    log_error "[$exp_name_pid] 작업 실패 (종료 코드: $exit_code)"
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                    RESULTS["$exp_name_pid"]="FAILED"
                fi
                
                unset PIDS["$pid"]
            fi
        done
        
        # 잠시 대기
        sleep 1
    done
    
    # 새 작업 시작
    log_info "[$exp_name] 작업 시작 (현재 실행 중: ${#PIDS[@]})"
    run_experiment "$exp_name" "$cfg_file" "$log_file" &
    local pid=$!
    PIDS["$pid"]="$exp_name"
    
    log_info "[$exp_name] PID: $pid"
done

# 남은 작업 완료 대기
log_info "모든 작업 시작 완료. 완료 대기 중..."
while [ ${#PIDS[@]} -gt 0 ]; do
    for pid in "${!PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local exp_name_pid="${PIDS[$pid]}"
            local exit_code=$?
            
            if [ $exit_code -eq 0 ]; then
                log_success "[$exp_name_pid] 작업 완료"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                RESULTS["$exp_name_pid"]="SUCCESS"
            else
                log_error "[$exp_name_pid] 작업 실패 (종료 코드: $exit_code)"
                FAILED_COUNT=$((FAILED_COUNT + 1))
                RESULTS["$exp_name_pid"]="FAILED"
            fi
            
            unset PIDS["$pid"]
        fi
    done
    
    if [ ${#PIDS[@]} -gt 0 ]; then
        sleep 2
    fi
done

# 전체 결과 요약
log_info "=========================================="
log_info "병렬 실행 결과 요약"
log_info "=========================================="
log_info "시작 시간: $START_TIME"
log_info "종료 시간: $(date)"
log_info "성공: $SUCCESS_COUNT개"
log_info "실패: $FAILED_COUNT개"
log_info "총 실험: ${#EXPERIMENT_NAMES[@]}개"

echo ""
log_info "상세 결과:"
for exp_name in "${EXPERIMENT_NAMES[@]}"; do
    if [ "${RESULTS[$exp_name]}" = "SUCCESS" ]; then
        echo -e "  ${GREEN}✓${NC} $exp_name: 성공"
    else
        echo -e "  ${RED}✗${NC} $exp_name: 실패"
    fi
done

# 결과 요약 파일 생성
SUMMARY_FILE="$BASE_DIR/batch_gradcam_results/parallel_execution_summary.txt"
mkdir -p "$(dirname "$SUMMARY_FILE")"

{
    echo "병렬 GradCAM 실행 결과 요약"
    echo "================================"
    echo "시작 시간: $START_TIME"
    echo "종료 시간: $(date)"
    echo "최대 동시 실행 작업 수: $MAX_JOBS"
    echo "성공: $SUCCESS_COUNT개"
    echo "실패: $FAILED_COUNT개"
    echo "총 실험: ${#EXPERIMENT_NAMES[@]}개"
    echo ""
    echo "상세 결과:"
    for exp_name in "${EXPERIMENT_NAMES[@]}"; do
        if [ "${RESULTS[$exp_name]}" = "SUCCESS" ]; then
            echo "  ✓ $exp_name: 성공"
        else
            echo "  ✗ $exp_name: 실패"
        fi
    done
} > "$SUMMARY_FILE"

log_success "결과 요약이 저장되었습니다: $SUMMARY_FILE"

if [ $FAILED_COUNT -eq 0 ]; then
    log_success "모든 실험이 성공적으로 완료되었습니다!"
    exit 0
else
    log_warning "$FAILED_COUNT개의 실험이 실패했습니다. 로그를 확인하세요."
    exit 1
fi 
#!/bin/bash

# 개별 실험에 대한 GradCAM 실행 스크립트

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
    echo "사용법: $0 [OPTIONS] <experiment_name>"
    echo ""
    echo "옵션:"
    echo "  --cfg <path>           설정 파일 경로"
    echo "  --log <path>           로그 파일 경로"
    echo "  --output <path>        출력 디렉토리 (기본값: batch_gradcam_results/<experiment_name>)"
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
    echo "  $0 1024_resnet18_foot"
    echo "  $0 --cfg custom.yaml --log custom.log 224_vgg19bn_hand"
}

# 기본 설정
BASE_DIR=$(pwd)
TOOL_DIR="$BASE_DIR/tool"
KFOLD_INDICES_DIR="$BASE_DIR/kfold_indices"
WANDB_DIR="$BASE_DIR/wandb"
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
CFG_FILE=""
LOG_FILE=""
OUTPUT_DIR=""
EXPERIMENT_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cfg)
            CFG_FILE="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
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
            if [ -z "$EXPERIMENT_NAME" ]; then
                EXPERIMENT_NAME="$1"
            else
                log_error "실험 이름은 하나만 지정할 수 있습니다: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# 실험 이름이 지정되지 않은 경우
if [ -z "$EXPERIMENT_NAME" ]; then
    log_error "실험 이름을 지정해야 합니다."
    usage
    exit 1
fi

# 실험 이름이 유효한지 확인
if [[ ! ${EXPERIMENT_CONFIGS[$EXPERIMENT_NAME]+_} ]]; then
    log_error "알 수 없는 실험 이름: $EXPERIMENT_NAME"
    echo ""
    echo "사용 가능한 실험 이름:"
    for exp_name in "${!EXPERIMENT_CONFIGS[@]}"; do
        echo "  $exp_name"
    done
    exit 1
fi

# 설정 파일과 로그 파일이 지정되지 않은 경우 기본값 사용
if [ -z "$CFG_FILE" ] || [ -z "$LOG_FILE" ]; then
    IFS=':' read -r default_cfg default_log <<< "${EXPERIMENT_CONFIGS[$EXPERIMENT_NAME]}"
    
    if [ -z "$CFG_FILE" ]; then
        CFG_FILE="$default_cfg"
        log_info "기본 설정 파일 사용: $CFG_FILE"
    fi
    
    if [ -z "$LOG_FILE" ]; then
        LOG_FILE="$default_log"
        log_info "기본 로그 파일 사용: $LOG_FILE"
    fi
fi

# 출력 디렉토리가 지정되지 않은 경우 기본값 사용
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$BASE_DIR/batch_gradcam_results/$EXPERIMENT_NAME"
    log_info "기본 출력 디렉토리 사용: $OUTPUT_DIR"
fi

# 파일 존재 확인
log_info "파일 존재 확인 중..."

if [ ! -f "$CFG_FILE" ]; then
    log_error "설정 파일을 찾을 수 없습니다: $CFG_FILE"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    log_error "로그 파일을 찾을 수 없습니다: $LOG_FILE"
    exit 1
fi

if [ ! -d "$KFOLD_INDICES_DIR" ]; then
    log_error "K-fold indices 디렉토리를 찾을 수 없습니다: $KFOLD_INDICES_DIR"
    log_info "generate_all_kfold_indices.sh를 먼저 실행하여 indices를 생성하세요."
    exit 1
fi

if [ ! -d "$WANDB_DIR" ]; then
    log_warning "wandb 디렉토리를 찾을 수 없습니다: $WANDB_DIR"
    log_info "wandb 디렉토리가 없으면 모델을 찾을 수 없을 수 있습니다."
fi

# 실행 정보 출력
log_info "=========================================="
log_info "GradCAM 실행 정보"
log_info "=========================================="
log_info "실험 이름: $EXPERIMENT_NAME"
log_info "설정 파일: $CFG_FILE"
log_info "로그 파일: $LOG_FILE"
log_info "출력 디렉토리: $OUTPUT_DIR"
log_info "K-fold indices: $KFOLD_INDICES_FILE"
log_info "Wandb 디렉토리: $WANDB_DIR"
log_info "디바이스: $DEVICE"
log_info "시드: $SEED"
log_info "=========================================="

# GradCAM 실행
log_info "GradCAM 실행 중..."

    # 해당 실험의 k-fold indices 파일 경로 (설정 파일명에서 실험 이름 추출)
    cfg_path=$(basename "$CFG_FILE" .yaml)
    EXP_KFOLD_INDICES_FILE="$KFOLD_INDICES_DIR/kfold_test_indices_${cfg_path}_seed_${SEED}.json"
    
    if [ ! -f "$EXP_KFOLD_INDICES_FILE" ]; then
        log_error "해당 실험의 k-fold indices 파일을 찾을 수 없습니다: $EXP_KFOLD_INDICES_FILE"
        exit 1
    fi
    
    if python "$TOOL_DIR/batch_gradcam_kfold.py" \
        --cfg "$CFG_FILE" \
        --kfold_indices "$EXP_KFOLD_INDICES_FILE" \
        --log "$LOG_FILE" \
        --wandb_dir "$WANDB_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --seed "$SEED"; then
    
    log_success "GradCAM 실행 완료!"
    
    # 결과 요약 파일 확인
    SUMMARY_FILE="$OUTPUT_DIR/summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        log_info "결과 요약: $SUMMARY_FILE"
        
        # 간단한 통계 출력
        TOTAL_IMAGES=$(python -c "import json; data=json.load(open('$SUMMARY_FILE')); print(data.get('total_images', 'N/A'))")
        OVERALL_ACCURACY=$(python -c "import json; data=json.load(open('$SUMMARY_FILE')); print(f\"{data.get('overall_accuracy', 'N/A'):.2f}%\" if data.get('overall_accuracy') is not None else 'N/A')")
        
        log_info "결과 통계:"
        log_info "  총 이미지: $TOTAL_IMAGES"
        log_info "  전체 정확도: $OVERALL_ACCURACY"
        
        # Fold별 결과 출력
        log_info "Fold별 결과:"
        python -c "
import json
data = json.load(open('$SUMMARY_FILE'))
for fold_name, result in data.get('fold_results', {}).items():
    print(f\"    {fold_name}: {result.get('correct', 0)}/{result.get('total', 0)} ({result.get('accuracy', 0):.2f}%)\")
"
    fi
    
    log_info "출력 디렉토리: $OUTPUT_DIR"
    
else
    log_error "GradCAM 실행 실패!"
    exit 1
fi 
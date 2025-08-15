#!/bin/bash

# 8개 실험에 대한 배치 GradCAM 실행 스크립트
# (1024, 224) x (resnet, vgg) x (foot, hand)

set -e  # 오류 발생 시 스크립트 중단

# Conda 환경 활성화
log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_info "Conda 환경 활성화 중..."
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

# 시작 시간 기록
START_TIME=$(date)
log_info "배치 GradCAM 실행 시작: $START_TIME"

# 기본 설정
BASE_DIR=$(pwd)
TOOL_DIR="$BASE_DIR/tool"
EXPERIMENTS_DIR="$BASE_DIR/experiments/image_exp"
LOG_DIR="$BASE_DIR/log"
OUTPUT_BASE_DIR="$BASE_DIR/batch_gradcam_results"
KFOLD_INDICES_DIR="$BASE_DIR/kfold_indices"
WANDB_DIR="$BASE_DIR/wandb"
DEVICE="cuda"
SEED=42

# 필요한 디렉토리 확인
log_info "필요한 디렉토리 및 파일 확인 중..."

if [ ! -d "$TOOL_DIR" ]; then
    log_error "tool 디렉토리를 찾을 수 없습니다: $TOOL_DIR"
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

# 각 실험에 대해 GradCAM 실행
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
    
    # GradCAM 실행
    log_info "GradCAM 실행 중: $exp_name"
    
    # 해당 실험의 k-fold indices 파일 경로 (설정 파일명에서 실험 이름 추출)
    cfg_path=$(basename "$cfg_file" .yaml)
    EXP_KFOLD_INDICES_FILE="$KFOLD_INDICES_DIR/kfold_test_indices_${cfg_path}_seed_${SEED}.json"
    
    if [ ! -f "$EXP_KFOLD_INDICES_FILE" ]; then
        log_error "해당 실험의 k-fold indices 파일을 찾을 수 없습니다: $EXP_KFOLD_INDICES_FILE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        TOTAL_RESULTS+=("$exp_name:FAILED:k-fold indices 파일 없음")
        continue
    fi
    
    if python "$TOOL_DIR/batch_gradcam_kfold.py" \
        --cfg "$cfg_file" \
        --kfold_indices "$EXP_KFOLD_INDICES_FILE" \
        --log "$log_file" \
        --wandb_dir "$WANDB_DIR" \
        --output_dir "$EXP_OUTPUT_DIR" \
        --device "$DEVICE" \
        --seed "$SEED"; then
        
        log_success "실험 완료: $exp_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        TOTAL_RESULTS+=("$exp_name:SUCCESS:$EXP_OUTPUT_DIR")
        
        # 결과 요약 파일 확인
        SUMMARY_FILE="$EXP_OUTPUT_DIR/summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            log_info "결과 요약: $SUMMARY_FILE"
            # 간단한 통계 출력
            TOTAL_IMAGES=$(python -c "import json; data=json.load(open('$SUMMARY_FILE')); print(data.get('total_images', 'N/A'))")
            OVERALL_ACCURACY=$(python -c "import json; data=json.load(open('$SUMMARY_FILE')); print(f\"{data.get('overall_accuracy', 'N/A'):.2f}%\" if data.get('overall_accuracy') is not None else 'N/A')")
            log_info "  총 이미지: $TOTAL_IMAGES, 전체 정확도: $OVERALL_ACCURACY"
        fi
        
    else
        log_error "실험 실패: $exp_name"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        TOTAL_RESULTS+=("$exp_name:FAILED:실행 오류")
    fi
    
    log_info "=========================================="
    echo ""
done

# 전체 결과 요약
log_info "=========================================="
log_info "전체 실행 결과 요약"
log_info "=========================================="
log_info "시작 시간: $START_TIME"
log_info "종료 시간: $(date)"
log_info "성공: $SUCCESS_COUNT개"
log_info "실패: $FAILED_COUNT개"
log_info "총 실험: ${#EXPERIMENTS[@]}개"

echo ""
log_info "상세 결과:"
for result in "${TOTAL_RESULTS[@]}"; do
    IFS=':' read -r exp_name status detail <<< "$result"
    if [ "$status" = "SUCCESS" ]; then
        echo -e "  ${GREEN}✓${NC} $exp_name: $detail"
    else
        echo -e "  ${RED}✗${NC} $exp_name: $detail"
    fi
done

# 결과 요약 파일 생성
SUMMARY_FILE="$OUTPUT_BASE_DIR/batch_execution_summary.txt"
{
    echo "배치 GradCAM 실행 결과 요약"
    echo "================================"
    echo "시작 시간: $START_TIME"
    echo "종료 시간: $(date)"
    echo "성공: $SUCCESS_COUNT개"
    echo "실패: $FAILED_COUNT개"
    echo "총 실험: ${#EXPERIMENTS[@]}개"
    echo ""
    echo "상세 결과:"
    for result in "${TOTAL_RESULTS[@]}"; do
        IFS=':' read -r exp_name status detail <<< "$result"
        if [ "$status" = "SUCCESS" ]; then
            echo "  ✓ $exp_name: $detail"
        else
            echo "  ✗ $exp_name: $detail"
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
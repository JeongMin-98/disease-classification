#!/bin/bash

# 8개 실험에 대해 각각 k-fold indices를 생성하는 배치 스크립트

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
    echo "사용법: $0 [OPTIONS]"
    echo ""
    echo "옵션:"
    echo "  --output-dir <path>    출력 디렉토리 (기본값: kfold_indices)"
    echo "  --seed <seed>          랜덤 시드 (기본값: 42)"
    echo "  --help                 이 도움말 출력"
    echo ""
    echo "이 스크립트는 8개 실험에 대해 각각 k-fold indices를 생성합니다."
    echo "각 실험의 설정 파일과 로그 파일을 사용하여 indices를 생성합니다."
}

# 기본 설정
BASE_DIR=$(pwd)
OUTPUT_DIR="kfold_indices"
SEED=42

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

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
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
            log_error "알 수 없는 인수: $1"
            usage
            exit 1
            ;;
    esac
done

# 시작 시간 기록
START_TIME=$(date)
log_info "8개 실험에 대한 k-fold indices 생성 시작: $START_TIME"
log_info "출력 디렉토리: $OUTPUT_DIR"
log_info "시드: $SEED"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 전체 결과 요약
TOTAL_RESULTS=()
SUCCESS_COUNT=0
FAILED_COUNT=0

# 각 실험에 대해 k-fold indices 생성
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name cfg_file log_file <<< "$experiment"
    
    log_info "=========================================="
    log_info "실험 처리 중: $exp_name"
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
    
    # 출력 파일 경로 설정 (실험 이름은 설정 파일에서 추출됨)
    # extract_kfold_from_cfg.py가 자동으로 실험 이름을 포함한 파일명으로 저장
    output_file="$OUTPUT_DIR/kfold_test_indices_*_seed_${SEED}.json"
    
    # k-fold indices 생성
    log_info "k-fold indices 생성 중: $exp_name"
    
    if python "$BASE_DIR/tool/extract_kfold_from_cfg.py" \
        --cfg "$cfg_file" \
        --log "$log_file" \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED"; then
        
        # 생성된 파일 확인 (실험 이름이 포함된 파일명으로 저장됨)
        # 설정 파일에서 실험 이름 추출
        cfg_path=$(basename "$cfg_file" .yaml)
        expected_file="$OUTPUT_DIR/kfold_test_indices_${cfg_path}_seed_${SEED}.json"
        
        if [ -f "$expected_file" ]; then
            log_success "k-fold indices 생성 완료: $exp_name"
            log_info "출력 파일: $expected_file"
            
            # 파일 크기 확인
            file_size=$(stat -c%s "$expected_file" 2>/dev/null || echo "0")
            log_info "파일 크기: ${file_size} bytes"
            
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            TOTAL_RESULTS+=("$exp_name:SUCCESS:$expected_file")
        else
            log_error "출력 파일을 찾을 수 없습니다: $expected_file"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            TOTAL_RESULTS+=("$exp_name:FAILED:출력 파일 없음")
        fi
        
    else
        log_error "k-fold indices 생성 실패: $exp_name"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        TOTAL_RESULTS+=("$exp_name:FAILED:실행 오류")
    fi
    
    log_info "=========================================="
    echo ""
    
    # 다음 실험 전 잠시 대기
    sleep 2
done

# 전체 결과 요약
log_info "=========================================="
log_info "k-fold indices 생성 완료 요약"
log_info "=========================================="
log_info "시작 시간: $START_TIME"
log_info "완료 시간: $(date)"
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

# 생성된 파일 목록
echo ""
log_info "생성된 k-fold indices 파일들:"
if [ -d "$OUTPUT_DIR" ]; then
    for file in "$OUTPUT_DIR"/kfold_test_indices_*_seed_*.json; do
        if [ -f "$file" ]; then
            file_size=$(stat -c%s "$file" 2>/dev/null || echo "0")
            echo "  $(basename "$file") (${file_size} bytes)"
        fi
    done
fi

# 결과 요약 파일 생성
SUMMARY_FILE="$OUTPUT_DIR/generation_summary.txt"
{
    echo "k-fold indices 생성 결과 요약"
    echo "================================"
    echo "시작 시간: $START_TIME"
    echo "완료 시간: $(date)"
    echo "시드: $SEED"
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
    log_success "모든 실험에 대한 k-fold indices가 성공적으로 생성되었습니다!"
    log_info "생성된 파일들은 $OUTPUT_DIR 디렉토리에 저장되었습니다."
    exit 0
else
    log_warning "$FAILED_COUNT개의 실험이 실패했습니다. 로그를 확인하세요."
    exit 1
fi 
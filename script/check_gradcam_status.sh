#!/bin/bash

# GradCAM 실행 상태 간단 확인 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 기본 설정
OUTPUT_DIR="batch_gradcam_results"

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "사용법: $0 [--output-dir <path>]"
            echo "  --output-dir <path>  출력 디렉토리 (기본값: batch_gradcam_results)"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

# 실험 목록 정의
declare -a EXPERIMENTS=(
    "1024_resnet18_foot"
    "1024_resnet18_hand"
    "1024_vgg19bn_foot"
    "1024_vgg19bn_hand"
    "224_resnet18_foot"
    "224_resnet18_hand"
    "224_vgg19bn_foot"
    "224_vgg19bn_hand"
)

echo -e "${BLUE}=== GradCAM 실행 상태 확인 ===${NC}"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "확인 시간: $(date)"
echo ""

# 출력 디렉토리 존재 확인
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}❌ 출력 디렉토리를 찾을 수 없습니다: $OUTPUT_DIR${NC}"
    echo "먼저 GradCAM 스크립트를 실행하여 결과 디렉토리를 생성하세요."
    exit 1
fi

# 각 실험 상태 확인
total_completed=0
total_running=0
total_pending=0
total_failed=0

echo -e "${BLUE}실험별 상태:${NC}"
for exp_name in "${EXPERIMENTS[@]}"; do
    exp_dir="$OUTPUT_DIR/$exp_name"
    
    if [ -d "$exp_dir" ]; then
        if [ -f "$exp_dir/summary.json" ]; then
            # 완료된 경우
            total_images=$(python -c "import json; data=json.load(open('$exp_dir/summary.json')); print(data.get('total_images', 'N/A'))" 2>/dev/null || echo "N/A")
            accuracy=$(python -c "import json; data=json.load(open('$exp_dir/summary.json')); print(f\"{data.get('overall_accuracy', 'N/A'):.2f}%\" if data.get('overall_accuracy') is not None else 'N/A')" 2>/dev/null || echo "N/A")
            echo -e "  ${GREEN}✓${NC} $exp_name: 완료 (이미지: $total_images, 정확도: $accuracy)"
            total_completed=$((total_completed + 1))
        elif [ -f "$exp_dir/execution.log" ]; then
            # 실행 중인지 확인
            last_modified=$(stat -c %Y "$exp_dir/execution.log" 2>/dev/null || echo "0")
            current_time=$(date +%s)
            time_diff=$((current_time - last_modified))
            
            if [ $time_diff -lt 300 ]; then  # 5분 이내
                echo -e "  ${BLUE}🔄${NC} $exp_name: 실행 중 (마지막 활동: ${time_diff}초 전)"
                total_running=$((total_running + 1))
            else
                echo -e "  ${RED}⚠️${NC} $exp_name: 멈춤 (마지막 활동: ${time_diff}초 전)"
                total_failed=$((total_failed + 1))
            fi
        else
            echo -e "  ${YELLOW}⏳${NC} $exp_name: 대기 중"
            total_pending=$((total_pending + 1))
        fi
    else
        echo -e "  ${YELLOW}⏳${NC} $exp_name: 대기 중"
        total_pending=$((total_pending + 1))
    fi
done

echo ""
echo -e "${PURPLE}=== 전체 요약 ===${NC}"
echo -e "  ${GREEN}완료: $total_completed개${NC}"
echo -e "  ${BLUE}실행 중: $total_running개${NC}"
echo -e "  ${YELLOW}대기 중: $total_pending개${NC}"
echo -e "  ${RED}실패/멈춤: $total_failed개${NC}"

total_experiments=${#EXPERIMENTS[@]}
if [ $total_experiments -gt 0 ]; then
    progress_percent=$((total_completed * 100 / total_experiments))
    echo -e "  ${PURPLE}전체 진행률: $progress_percent% ($total_completed/$total_experiments)${NC}"
fi

# 실행 중인 프로세스 확인
echo ""
echo -e "${BLUE}=== 실행 중인 GradCAM 프로세스 ===${NC}"
gradcam_processes=$(ps aux | grep -E "(batch_gradcam_kfold|gradcam)" | grep -v grep)

if [ -n "$gradcam_processes" ]; then
    echo "$gradcam_processes"
else
    echo "실행 중인 GradCAM 프로세스가 없습니다."
fi

# GPU 사용량 확인 (nvidia-smi가 있는 경우)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo -e "${BLUE}=== GPU 사용량 ===${NC}"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r name util mem_used mem_total; do
        echo "  $name: GPU $util%, 메모리 ${mem_used}MB/${mem_total}MB"
    done
fi 
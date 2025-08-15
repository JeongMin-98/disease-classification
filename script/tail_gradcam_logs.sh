#!/bin/bash

# GradCAM 실행 로그 실시간 tail 스크립트

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
LOG_NAME="execution.log"
FOLLOW=true
LINES=50

# 사용법 출력
usage() {
    echo "사용법: $0 [OPTIONS] [experiment_name]"
    echo ""
    echo "옵션:"
    echo "  --output-dir <path>    출력 디렉토리 (기본값: batch_gradcam_results)"
    echo "  --log-name <name>      로그 파일명 (기본값: execution.log)"
    echo "  --lines <N>            표시할 라인 수 (기본값: 50)"
    echo "  --no-follow            실시간 tail 비활성화 (한 번만 표시)"
    echo "  --help                 이 도움말 출력"
    echo ""
    echo "인수:"
    echo "  experiment_name        특정 실험의 로그만 표시 (예: 1024_resnet18_foot)"
    echo "                         지정하지 않으면 모든 실행 중인 로그를 표시"
    echo ""
    echo "예시:"
    echo "  $0                     # 모든 실행 중인 로그를 실시간으로 표시"
    echo "  $0 1024_resnet18_foot # 특정 실험의 로그만 표시"
    echo "  $0 --lines 100        # 최근 100줄 표시"
    echo "  $0 --no-follow        # 실시간 tail 비활성화"
}

# 명령행 인수 파싱
EXPERIMENT_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-name)
            LOG_NAME="$2"
            shift 2
            ;;
        --lines)
            LINES="$2"
            shift 2
            ;;
        --no-follow)
            FOLLOW=false
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        -*)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            usage
            exit 1
            ;;
        *)
            if [ -z "$EXPERIMENT_NAME" ]; then
                EXPERIMENT_NAME="$1"
            else
                echo -e "${RED}실험 이름은 하나만 지정할 수 있습니다: $1${NC}"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# 출력 디렉토리 존재 확인
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}❌ 출력 디렉토리를 찾을 수 없습니다: $OUTPUT_DIR${NC}"
    echo "먼저 GradCAM 스크립트를 실행하여 결과 디렉토리를 생성하세요."
    exit 1
fi

# 특정 실험의 로그만 표시
if [ -n "$EXPERIMENT_NAME" ]; then
    exp_dir="$OUTPUT_DIR/$EXPERIMENT_NAME"
    log_file="$exp_dir/$LOG_NAME"
    
    if [ ! -d "$exp_dir" ]; then
        echo -e "${RED}❌ 실험 디렉토리를 찾을 수 없습니다: $exp_dir${NC}"
        exit 1
    fi
    
    if [ ! -f "$log_file" ]; then
        echo -e "${RED}❌ 로그 파일을 찾을 수 없습니다: $log_file${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}=== $EXPERIMENT_NAME 로그 ===${NC}"
    echo "로그 파일: $log_file"
    echo "표시 라인: $LINES"
    echo "실시간 tail: $([ "$FOLLOW" = true ] && echo "활성화" || echo "비활성화")"
    echo ""
    
    if [ "$FOLLOW" = true ]; then
        tail -f -n "$LINES" "$log_file"
    else
        tail -n "$LINES" "$log_file"
    fi
    
    exit 0
fi

# 모든 실행 중인 로그 찾기
echo -e "${BLUE}=== 실행 중인 GradCAM 로그 모니터링 ===${NC}"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "로그 파일명: $LOG_NAME"
echo "표시 라인: $LINES"
echo "실시간 tail: $([ "$FOLLOW" = true ] && echo "활성화" || echo "비활성화")"
echo ""

# 실행 중인 로그 파일들 찾기
running_logs=()
for exp_dir in "$OUTPUT_DIR"/*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        log_file="$exp_dir/$LOG_NAME"
        
        if [ -f "$log_file" ]; then
            # 로그 파일이 최근에 수정되었는지 확인 (5분 이내)
            last_modified=$(stat -c %Y "$log_file" 2>/dev/null || echo "0")
            current_time=$(date +%s)
            time_diff=$((current_time - last_modified))
            
            if [ $time_diff -lt 300 ]; then  # 5분 이내
                running_logs+=("$log_file")
                echo -e "${GREEN}✓${NC} $exp_name: $log_file (${time_diff}초 전 활동)"
            fi
        fi
    fi
done

if [ ${#running_logs[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠️ 실행 중인 로그 파일이 없습니다.${NC}"
    echo "GradCAM 스크립트를 실행하거나 다른 로그 파일명을 지정해보세요."
    exit 0
fi

echo ""
echo -e "${BLUE}총 ${#running_logs[@]}개의 실행 중인 로그 파일을 발견했습니다.${NC}"
echo ""

# 여러 로그 파일을 동시에 tail
if [ ${#running_logs[@]} -eq 1 ]; then
    # 로그 파일이 하나인 경우
    log_file="${running_logs[0]}"
    exp_name=$(basename "$(dirname "$log_file")")
    
    echo -e "${BLUE}=== $exp_name 로그 모니터링 ===${NC}"
    if [ "$FOLLOW" = true ]; then
        tail -f -n "$LINES" "$log_file"
    else
        tail -n "$LINES" "$log_file"
    fi
else
    # 여러 로그 파일인 경우
    echo -e "${BLUE}여러 로그 파일을 동시에 모니터링합니다.${NC}"
    echo "각 로그는 다른 터미널에서 개별적으로 확인하는 것을 권장합니다."
    echo ""
    
    for log_file in "${running_logs[@]}"; do
        exp_name=$(basename "$(dirname "$log_file")")
        echo -e "${GREEN}=== $exp_name ===${NC}"
        echo "로그 파일: $log_file"
        echo "명령어: tail -f -n $LINES \"$log_file\""
        echo ""
    done
    
    echo -e "${YELLOW}여러 로그를 동시에 tail하려면 새 터미널을 열어서 각각 실행하세요.${NC}"
    echo ""
    echo "예시:"
    for log_file in "${running_logs[@]}"; do
        exp_name=$(basename "$(dirname "$log_file")")
        echo "  # 터미널 1: $exp_name"
        echo "  tail -f -n $LINES \"$log_file\""
    done
fi 
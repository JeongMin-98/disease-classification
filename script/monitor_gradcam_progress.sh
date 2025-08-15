#!/bin/bash

# GradCAM 실행 진행 상황 모니터링 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

log_progress() {
    echo -e "${PURPLE}[PROGRESS]${NC} $1"
}

# 사용법 출력
usage() {
    echo "사용법: $0 [OPTIONS]"
    echo ""
    echo "옵션:"
    echo "  --watch <seconds>      자동 새로고침 간격 (초, 기본값: 10)"
    echo "  --output-dir <path>    출력 디렉토리 (기본값: batch_gradcam_results)"
    echo "  --show-logs            실시간 로그 표시"
    echo "  --show-processes       실행 중인 프로세스 정보 표시"
    echo "  --help                 이 도움말 출력"
    echo ""
    echo "예시:"
    echo "  $0                     # 기본 모니터링 (10초마다 새로고침)"
    echo "  $0 --watch 5          # 5초마다 새로고침"
    echo "  $0 --show-logs        # 실시간 로그 표시"
    echo "  $0 --show-processes   # 프로세스 정보 표시"
}

# 기본 설정
WATCH_INTERVAL=10
OUTPUT_DIR="batch_gradcam_results"
SHOW_LOGS=false
SHOW_PROCESSES=false

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --watch)
            WATCH_INTERVAL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --show-logs)
            SHOW_LOGS=true
            shift
            ;;
        --show-processes)
            SHOW_PROCESSES=true
            shift
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

# 화면 클리어 함수
clear_screen() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}    GradCAM 실행 진행 상황 모니터링    ${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo "마지막 업데이트: $(date)"
    echo "새로고침 간격: ${WATCH_INTERVAL}초"
    echo "출력 디렉토리: $OUTPUT_DIR"
    echo ""
}

# 프로세스 정보 표시
show_processes() {
    if [ "$SHOW_PROCESSES" = true ]; then
        echo -e "${YELLOW}=== 실행 중인 GradCAM 프로세스 ===${NC}"
        
        # Python 프로세스 중 GradCAM 관련 프로세스 찾기
        local gradcam_processes=$(ps aux | grep -E "(batch_gradcam_kfold|gradcam)" | grep -v grep)
        
        if [ -n "$gradcam_processes" ]; then
            echo "$gradcam_processes"
        else
            echo "실행 중인 GradCAM 프로세스가 없습니다."
        fi
        
        echo ""
    fi
}

# 실험 진행 상황 표시
show_experiment_progress() {
    echo -e "${BLUE}=== 실험별 진행 상황 ===${NC}"
    
    local total_completed=0
    local total_failed=0
    local total_running=0
    local total_pending=0
    
    for exp_name in "${EXPERIMENTS[@]}"; do
        local exp_dir="$OUTPUT_DIR/$exp_name"
        local status=""
        local details=""
        
        if [ -d "$exp_dir" ]; then
            # summary.json 파일 확인
            if [ -f "$exp_dir/summary.json" ]; then
                status="COMPLETED"
                local total_images=$(python -c "import json; data=json.load(open('$exp_dir/summary.json')); print(data.get('total_images', 'N/A'))" 2>/dev/null || echo "N/A")
                local accuracy=$(python -c "import json; data=json.load(open('$exp_dir/summary.json')); print(f\"{data.get('overall_accuracy', 'N/A'):.2f}%\" if data.get('overall_accuracy') is not None else 'N/A')" 2>/dev/null || echo "N/A")
                details="✓ 완료 (이미지: $total_images, 정확도: $accuracy)"
                total_completed=$((total_completed + 1))
            else
                # 실행 중인지 확인 (execution.log 파일 확인)
                if [ -f "$exp_dir/execution.log" ]; then
                    local last_modified=$(stat -c %Y "$exp_dir/execution.log" 2>/dev/null || echo "0")
                    local current_time=$(date +%s)
                    local time_diff=$((current_time - last_modified))
                    
                    if [ $time_diff -lt 300 ]; then  # 5분 이내에 수정된 경우 실행 중으로 간주
                        status="RUNNING"
                        details="🔄 실행 중 (마지막 활동: ${time_diff}초 전)"
                        total_running=$((total_running + 1))
                    else
                        status="STALLED"
                        details="⚠️ 멈춤 (마지막 활동: ${time_diff}초 전)"
                        total_failed=$((total_failed + 1))
                    fi
                else
                    status="PENDING"
                    details="⏳ 대기 중"
                    total_pending=$((total_pending + 1))
                fi
            fi
        else
            status="PENDING"
            details="⏳ 대기 중"
            total_pending=$((total_pending + 1))
        fi
        
        # 상태에 따른 색상 설정
        case $status in
            "COMPLETED")
                echo -e "  ${GREEN}✓${NC} $exp_name: $details"
                ;;
            "RUNNING")
                echo -e "  ${BLUE}🔄${NC} $exp_name: $details"
                ;;
            "STALLED")
                echo -e "  ${RED}⚠️${NC} $exp_name: $details"
                ;;
            "PENDING")
                echo -e "  ${YELLOW}⏳${NC} $exp_name: $details"
                ;;
        esac
    done
    
    echo ""
    echo -e "${CYAN}=== 전체 진행 상황 요약 ===${NC}"
    echo -e "  ${GREEN}완료: $total_completed개${NC}"
    echo -e "  ${BLUE}실행 중: $total_running개${NC}"
    echo -e "  ${YELLOW}대기 중: $total_pending개${NC}"
    echo -e "  ${RED}실패/멈춤: $total_failed개${NC}"
    
    local total_experiments=${#EXPERIMENTS[@]}
    local progress_percent=$((total_completed * 100 / total_experiments))
    echo -e "  ${PURPLE}전체 진행률: $progress_percent% ($total_completed/$total_experiments)${NC}"
}

# 실시간 로그 표시
show_realtime_logs() {
    if [ "$SHOW_LOGS" = true ]; then
        echo -e "${YELLOW}=== 실시간 로그 (최근 10줄) ===${NC}"
        
        # 가장 최근에 수정된 execution.log 파일 찾기
        local latest_log=$(find "$OUTPUT_DIR" -name "execution.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        
        if [ -n "$latest_log" ] && [ -f "$latest_log" ]; then
            echo "최신 로그 파일: $latest_log"
            echo "---"
            tail -10 "$latest_log" 2>/dev/null || echo "로그를 읽을 수 없습니다."
        else
            echo "실행 중인 로그 파일을 찾을 수 없습니다."
        fi
        
        echo ""
    fi
}

# 시스템 리소스 정보 표시
show_system_resources() {
    echo -e "${CYAN}=== 시스템 리소스 ===${NC}"
    
    # GPU 사용량 (nvidia-smi가 있는 경우)
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BLUE}GPU 사용량:${NC}"
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r name util mem_used mem_total; do
            echo "  $name: GPU $util%, 메모리 ${mem_used}MB/${mem_total}MB"
        done
    else
        echo "GPU 정보를 가져올 수 없습니다 (nvidia-smi 없음)"
    fi
    
    # 메모리 사용량
    local mem_info=$(free -h | grep Mem)
    local mem_total=$(echo $mem_info | awk '{print $2}')
    local mem_used=$(echo $mem_info | awk '{print $3}')
    local mem_available=$(echo $mem_info | awk '{print $7}')
    echo -e "${BLUE}메모리:${NC} 사용: $mem_used / 전체: $mem_total (가용: $mem_available)"
    
    # 디스크 사용량
    local disk_usage=$(df -h . | tail -1)
    local disk_total=$(echo $disk_usage | awk '{print $2}')
    local disk_used=$(echo $disk_usage | awk '{print $3}')
    local disk_available=$(echo $disk_usage | awk '{print $4}')
    local disk_percent=$(echo $disk_usage | awk '{print $5}')
    echo -e "${BLUE}디스크:${NC} 사용: $disk_used / 전체: $disk_total (가용: $disk_available, $disk_percent)"
    
    echo ""
}

# 메인 모니터링 루프
main_monitoring() {
    while true; do
        clear_screen
        show_processes
        show_experiment_progress
        show_realtime_logs
        show_system_resources
        
        echo -e "${CYAN}========================================${NC}"
        echo "Ctrl+C를 누르면 종료됩니다."
        echo "새로고침까지 ${WATCH_INTERVAL}초 대기 중..."
        
        sleep $WATCH_INTERVAL
    done
}

# Ctrl+C 처리
trap 'echo -e "\n${GREEN}모니터링을 종료합니다.${NC}"; exit 0' INT

# 메인 실행
if [ ! -d "$OUTPUT_DIR" ]; then
    log_error "출력 디렉토리를 찾을 수 없습니다: $OUTPUT_DIR"
    log_info "먼저 GradCAM 스크립트를 실행하여 결과 디렉토리를 생성하세요."
    exit 1
fi

log_info "GradCAM 진행 상황 모니터링을 시작합니다..."
log_info "출력 디렉토리: $OUTPUT_DIR"
log_info "새로고침 간격: ${WATCH_INTERVAL}초"

main_monitoring 
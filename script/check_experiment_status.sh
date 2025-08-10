#!/bin/bash

# 실험 상태 확인 및 관리 스크립트

LOG_DIR="logs"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 함수: 실험 상태 확인
check_experiment_status() {
    echo -e "${BLUE}=== 실험 상태 확인 ===${NC}"
    
    # PID 파일들 찾기
    PID_FILES=$(find $LOG_DIR -name "experiments_pid_*.txt" 2>/dev/null | sort -r)
    
    if [ -z "$PID_FILES" ]; then
        echo -e "${YELLOW}실행 중인 실험이 없습니다.${NC}"
        return
    fi
    
    for pid_file in $PID_FILES; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            timestamp=$(basename "$pid_file" | sed 's/experiments_pid_\(.*\)\.txt/\1/')
            
            # 프로세스 상태 확인
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${GREEN}✓ 실험 실행 중${NC}"
                echo "  PID: $pid"
                echo "  시작 시간: $timestamp"
                echo "  로그 파일: $LOG_DIR/experiments_stdout_${timestamp}.log"
                echo "  오류 로그: $LOG_DIR/experiments_stderr_${timestamp}.log"
                
                # CPU 및 메모리 사용량 확인
                if command -v ps > /dev/null; then
                    cpu_mem=$(ps -p $pid -o %cpu,%mem --no-headers 2>/dev/null)
                    if [ ! -z "$cpu_mem" ]; then
                        echo "  CPU/메모리 사용량: $cpu_mem"
                    fi
                fi
                
                # 실험별 로그 파일 확인
                echo -e "${CYAN}  실험별 상세 로그:${NC}"
                experiment_logs=$(find $LOG_DIR -name "*.log" -not -name "experiments_*" -not -name "main_experiments.log" 2>/dev/null | head -5)
                if [ ! -z "$experiment_logs" ]; then
                    for log in $experiment_logs; do
                        log_name=$(basename "$log" .log)
                        echo "    - $log_name.log"
                    done
                    if [ $(find $LOG_DIR -name "*.log" -not -name "experiments_*" -not -name "main_experiments.log" 2>/dev/null | wc -l) -gt 5 ]; then
                        echo "    ... (더 많은 로그 파일이 있습니다)"
                    fi
                else
                    echo "    - 아직 실험별 로그가 생성되지 않았습니다"
                fi
            else
                echo -e "${RED}✗ 실험 종료됨${NC}"
                echo "  PID: $pid (종료됨)"
                echo "  시작 시간: $timestamp"
                echo "  로그 파일: $LOG_DIR/experiments_stdout_${timestamp}.log"
            fi
            echo ""
        fi
    done
}

# 함수: 실시간 로그 확인
show_live_logs() {
    echo -e "${BLUE}=== 실시간 로그 확인 ===${NC}"
    
    # 가장 최근 PID 파일 찾기
    latest_pid_file=$(find $LOG_DIR -name "experiments_pid_*.txt" 2>/dev/null | sort -r | head -1)
    
    if [ -z "$latest_pid_file" ]; then
        echo -e "${YELLOW}실행 중인 실험이 없습니다.${NC}"
        return
    fi
    
    timestamp=$(basename "$latest_pid_file" | sed 's/experiments_pid_\(.*\)\.txt/\1/')
    log_file="$LOG_DIR/experiments_stdout_${timestamp}.log"
    
    if [ -f "$log_file" ]; then
        echo "로그 파일: $log_file"
        echo "실시간 로그를 확인합니다. (Ctrl+C로 종료)"
        echo "=========================="
        tail -f "$log_file"
    else
        echo -e "${RED}로그 파일을 찾을 수 없습니다: $log_file${NC}"
    fi
}

# 함수: 실험별 로그 확인
show_experiment_logs() {
    echo -e "${BLUE}=== 실험별 로그 확인 ===${NC}"
    
    # 실험별 로그 파일들 찾기
    experiment_logs=$(find $LOG_DIR -name "*.log" -not -name "experiments_*" -not -name "main_experiments.log" 2>/dev/null | sort)
    
    if [ -z "$experiment_logs" ]; then
        echo -e "${YELLOW}실험별 로그 파일이 없습니다.${NC}"
        return
    fi
    
    echo "사용 가능한 실험별 로그 파일:"
    echo ""
    
    i=1
    for log in $experiment_logs; do
        log_name=$(basename "$log" .log)
        size=$(du -h "$log" 2>/dev/null | cut -f1)
        echo "  $i. $log_name.log (크기: $size)"
        i=$((i+1))
    done
    
    echo ""
    echo "실험별 로그를 확인하려면:"
    echo "  tail -f $LOG_DIR/[실험명].log"
    echo ""
    echo "예시:"
    echo "  tail -f $LOG_DIR/224x224_VGG19BN_Hand_OA_Normal.log"
    echo "  tail -f $LOG_DIR/224x224_VGG19BN_Foot_OA_Normal.log"
}

# 함수: 실험 중단
stop_experiment() {
    echo -e "${BLUE}=== 실험 중단 ===${NC}"
    
    # 가장 최근 PID 파일 찾기
    latest_pid_file=$(find $LOG_DIR -name "experiments_pid_*.txt" 2>/dev/null | sort -r | head -1)
    
    if [ -z "$latest_pid_file" ]; then
        echo -e "${YELLOW}실행 중인 실험이 없습니다.${NC}"
        return
    fi
    
    pid=$(cat "$latest_pid_file")
    timestamp=$(basename "$latest_pid_file" | sed 's/experiments_pid_\(.*\)\.txt/\1/')
    
    if ps -p $pid > /dev/null 2>&1; then
        echo "실험을 중단합니다. (PID: $pid)"
        echo "정말 중단하시겠습니까? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            kill $pid
            echo -e "${GREEN}실험이 중단되었습니다.${NC}"
        else
            echo "실험 중단이 취소되었습니다."
        fi
    else
        echo -e "${YELLOW}실험이 이미 종료되었습니다.${NC}"
    fi
}

# 함수: 로그 파일 목록
list_logs() {
    echo -e "${BLUE}=== 로그 파일 목록 ===${NC}"
    
    if [ ! -d "$LOG_DIR" ]; then
        echo -e "${YELLOW}로그 디렉토리가 없습니다.${NC}"
        return
    fi
    
    echo -e "${CYAN}메인 로그 파일:${NC}"
    main_logs=$(find $LOG_DIR -name "experiments_stdout_*.log" -o -name "main_experiments.log" | sort -r)
    
    if [ ! -z "$main_logs" ]; then
        for log_file in $main_logs; do
            timestamp=$(basename "$log_file" | sed 's/experiments_stdout_\(.*\)\.log/\1/' | sed 's/main_experiments.log/main/')
            size=$(du -h "$log_file" 2>/dev/null | cut -f1)
            echo "  $timestamp (크기: $size)"
        done
    else
        echo "  - 메인 로그 파일이 없습니다"
    fi
    
    echo ""
    echo -e "${CYAN}실험별 상세 로그 파일:${NC}"
    experiment_logs=$(find $LOG_DIR -name "*.log" -not -name "experiments_*" -not -name "main_experiments.log" | sort)
    
    if [ ! -z "$experiment_logs" ]; then
        for log_file in $experiment_logs; do
            log_name=$(basename "$log_file" .log)
            size=$(du -h "$log_file" 2>/dev/null | cut -f1)
            echo "  $log_name (크기: $size)"
        done
    else
        echo "  - 실험별 로그 파일이 없습니다"
    fi
}

# 함수: 도움말
show_help() {
    echo -e "${BLUE}=== 실험 관리 스크립트 사용법 ===${NC}"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  status    - 실험 상태 확인 (기본값)"
    echo "  logs      - 실시간 로그 확인"
    echo "  exp-logs  - 실험별 로그 파일 목록 및 확인 방법"
    echo "  stop      - 실행 중인 실험 중단"
    echo "  list      - 모든 로그 파일 목록"
    echo "  help      - 이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0 status    # 실험 상태 확인"
    echo "  $0 logs      # 실시간 로그 확인"
    echo "  $0 exp-logs  # 실험별 로그 확인"
    echo "  $0 stop      # 실험 중단"
    echo "  $0 list      # 로그 파일 목록"
    echo ""
    echo "실험별 로그 확인 예시:"
    echo "  tail -f $LOG_DIR/224x224_VGG19BN_Hand_OA_Normal.log"
    echo "  tail -f $LOG_DIR/224x224_VGG19BN_Foot_OA_Normal.log"
}

# 메인 로직
case "${1:-status}" in
    "status")
        check_experiment_status
        ;;
    "logs")
        show_live_logs
        ;;
    "exp-logs")
        show_experiment_logs
        ;;
    "stop")
        stop_experiment
        ;;
    "list")
        list_logs
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo -e "${RED}알 수 없는 옵션: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac 
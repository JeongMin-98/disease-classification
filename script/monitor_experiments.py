#!/usr/bin/env python3
"""
실험 진행 상황을 실시간으로 모니터링하는 스크립트
"""

import os
import sys
import time
import json
import logging
import re
from datetime import datetime
from pathlib import Path
import psutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentMonitor:
    """실험 모니터링 클래스"""
    
    def __init__(self):
        self.log_dir = Path("logs")
        self.results_dir = Path("experiments/results")
        
    def find_running_experiments(self):
        """실행 중인 실험 찾기"""
        running_experiments = []
        
        # PID 파일들 찾기
        pid_files = list(self.log_dir.glob("experiments_pid_*.txt"))
        
        for pid_file in pid_files:
            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    
                    timestamp = pid_file.stem.replace('experiments_pid_', '')
                    
                    # 프로세스 상태 확인
                    if psutil.pid_exists(pid):
                        process = psutil.Process(pid)
                        running_experiments.append({
                            'pid': pid,
                            'timestamp': timestamp,
                            'start_time': datetime.fromtimestamp(process.create_time()),
                            'cpu_percent': process.cpu_percent(),
                            'memory_percent': process.memory_percent(),
                            'status': process.status(),
                            'is_running': True
                        })
                    else:
                        # 프로세스가 종료된 경우에도 정보 표시
                        running_experiments.append({
                            'pid': pid,
                            'timestamp': timestamp,
                            'start_time': datetime.fromtimestamp(pid_file.stat().st_mtime),
                            'cpu_percent': 0.0,
                            'memory_percent': 0.0,
                            'status': '종료됨',
                            'is_running': False
                        })
                except Exception as e:
                    logger.warning(f"PID 파일 {pid_file} 읽기 실패: {e}")
        
        return running_experiments
    
    def get_experiment_logs(self, timestamp):
        """특정 타임스탬프의 실험 로그 가져오기"""
        stdout_log = self.log_dir / f"experiments_stdout_{timestamp}.log"
        stderr_log = self.log_dir / f"experiments_stderr_{timestamp}.log"
        
        logs = {'stdout': [], 'stderr': []}
        
        if stdout_log.exists():
            with open(stdout_log, 'r', encoding='utf-8') as f:
                logs['stdout'] = f.readlines()[-50:]  # 마지막 50줄
        
        if stderr_log.exists():
            with open(stderr_log, 'r', encoding='utf-8') as f:
                logs['stderr'] = f.readlines()[-100:]  # 마지막 100줄 (더 많은 로그 읽기)
        
        return logs
    
    def parse_progress_from_logs(self, logs):
        """로그에서 진행 상황 파싱"""
        progress_info = {}
        
        # stdout과 stderr 모두에서 진행 상황 파싱
        all_lines = logs['stdout'] + logs['stderr']
        
        for line in all_lines:
            line = line.strip()
            
            # 실험 이름 추출 (대괄호 안의 내용)
            if '[' in line and ']' in line:
                experiment_name = line[line.find('[')+1:line.find(']')]
                if experiment_name not in progress_info:
                    progress_info[experiment_name] = {
                        'status': '실행 중',
                        'current_epoch': '0/0',
                        'current_fold': '0/7',
                        'best_accuracy': '0.0000',
                        'current_accuracy': '0.0000',
                        'current_loss': '0.0000',
                        'last_update': ''
                    }
            
            # 진행 상황 정보 추출
            if '진행상황:' in line:
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if experiment_name in progress_info:
                            if '상태' in key:
                                progress_info[experiment_name]['status'] = value
                            elif '에포크' in key:
                                progress_info[experiment_name]['current_epoch'] = value
                            elif '폴드' in key:
                                progress_info[experiment_name]['current_fold'] = value
                            elif '최고정확도' in key:
                                progress_info[experiment_name]['best_accuracy'] = value
                            elif '현재정확도' in key:
                                progress_info[experiment_name]['current_accuracy'] = value
                            elif '현재손실' in key:
                                progress_info[experiment_name]['current_loss'] = value
                            elif '경과시간' in key:
                                progress_info[experiment_name]['last_update'] = value
            
            # Validation loss decreased 메시지에서 손실값 추출
            if 'Validation loss decreased' in line:
                loss_match = re.search(r'Validation loss decreased \([^)]* --> ([\d.]+)\)', line)
                if loss_match:
                    current_loss = float(loss_match.group(1))
                    for exp_name in progress_info:
                        progress_info[exp_name]['current_loss'] = f"{current_loss:.4f}"
                        # 손실값이 개선되었으므로 상태를 업데이트
                        progress_info[exp_name]['status'] = '훈련 중'
            
            # train_acc와 val_acc 정보 추출
            if 'train_acc:' in line and 'val_acc:' in line:
                acc_match = re.search(r'train_acc: ([\d.]+), val_acc: ([\d.]+)', line)
                if acc_match:
                    train_acc = float(acc_match.group(1))
                    val_acc = float(acc_match.group(2))
                    for exp_name in progress_info:
                        progress_info[exp_name]['current_accuracy'] = f"{val_acc:.4f}"
                        # 정확도가 개선되었는지 확인
                        current_best = float(progress_info[exp_name]['best_accuracy'])
                        if val_acc > current_best:
                            progress_info[exp_name]['best_accuracy'] = f"{val_acc:.4f}"
            
            # train_loss와 val_loss 정보 추출
            if 'train_loss:' in line and 'val_loss:' in line:
                loss_match = re.search(r'train_loss: ([\d.]+), val_loss: ([\d.]+)', line)
                if loss_match:
                    val_loss = float(loss_match.group(2))
                    for exp_name in progress_info:
                        progress_info[exp_name]['current_loss'] = f"{val_loss:.4f}"
            
            # 에포크 정보 추출 (다른 패턴도 확인)
            if 'epoch' in line.lower() and '/' in line:
                epoch_match = re.search(r'epoch[:\s]*(\d+)/(\d+)', line.lower())
                if epoch_match:
                    for exp_name in progress_info:
                        progress_info[exp_name]['current_epoch'] = f"{epoch_match.group(1)}/{epoch_match.group(2)}"
            
            # 폴드 정보 추출
            if 'fold' in line.lower() and '/' in line:
                fold_match = re.search(r'fold[:\s]*(\d+)/(\d+)', line.lower())
                if fold_match:
                    for exp_name in progress_info:
                        progress_info[exp_name]['current_fold'] = f"{fold_match.group(1)}/{fold_match.group(2)}"
            
            # Train Epoch 또는 Val Epoch 정보 추출
            if 'Train Epoch' in line or 'Val Epoch' in line:
                epoch_match = re.search(r'Epoch (\d+):', line)
                if epoch_match:
                    epoch_num = int(epoch_match.group(1))
                    for exp_name in progress_info:
                        # 기본적으로 100 에포크로 가정하거나 기존 값 유지
                        current_epoch = progress_info[exp_name]['current_epoch']
                        if current_epoch == '0/0':
                            progress_info[exp_name]['current_epoch'] = f"{epoch_num}/100"
                        else:
                            # 기존 총 에포크 수 유지
                            total_epochs = current_epoch.split('/')[1] if '/' in current_epoch else '100'
                            progress_info[exp_name]['current_epoch'] = f"{epoch_num}/{total_epochs}"
        
        return progress_info
    
    def display_progress(self, running_experiments):
        """진행 상황 표시"""
        if not running_experiments:
            print("실행 중인 실험이 없습니다.")
            return
        
        print(f"\n=== 실험 진행 상황 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
        
        for exp in running_experiments:
            status_icon = "🟢" if exp['is_running'] else "🔴"
            print(f"\n{status_icon} 실험 PID: {exp['pid']}")
            print(f"   시작 시간: {exp['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   경과 시간: {datetime.now() - exp['start_time']}")
            
            if exp['is_running']:
                print(f"   CPU 사용률: {exp['cpu_percent']:.1f}%")
                print(f"   메모리 사용률: {exp['memory_percent']:.1f}%")
                print(f"   상태: {exp['status']}")
            else:
                print(f"   상태: {exp['status']}")
            
            # 로그에서 진행 상황 파싱
            logs = self.get_experiment_logs(exp['timestamp'])
            progress_info = self.parse_progress_from_logs(logs)
            
            if progress_info:
                print("   📈 진행 상황:")
                for exp_name, progress in progress_info.items():
                    print(f"      {exp_name}:")
                    print(f"        상태: {progress['status']}")
                    print(f"        에포크: {progress['current_epoch']}")
                    print(f"        폴드: {progress['current_fold']}")
                    print(f"        최고 정확도: {progress['best_accuracy']}")
                    print(f"        현재 정확도: {progress['current_accuracy']}")
                    print(f"        현재 손실: {progress['current_loss']}")
                    if progress['last_update']:
                        print(f"        마지막 업데이트: {progress['last_update']}")
            else:
                print("   📈 진행 상황: 정보 없음")
        
        print("\n" + "="*60)
    
    def monitor_continuously(self, interval=30):
        """연속 모니터링"""
        print(f"실험 진행 상황을 {interval}초마다 모니터링합니다. (Ctrl+C로 종료)")
        
        try:
            while True:
                running_experiments = self.find_running_experiments()
                self.display_progress(running_experiments)
                
                if not running_experiments:
                    print("실행 중인 실험이 없습니다. 30초 후 다시 확인합니다...")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n모니터링을 종료합니다.")
    
    def show_recent_logs(self, lines=20):
        """최근 로그 표시"""
        # 가장 최근 PID 파일 찾기
        pid_files = list(self.log_dir.glob("experiments_pid_*.txt"))
        if not pid_files:
            print("실행 중인 실험이 없습니다.")
            return
        
        latest_pid_file = max(pid_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest_pid_file.stem.replace('experiments_pid_', '')
        
        logs = self.get_experiment_logs(timestamp)
        
        print(f"\n=== 최근 로그 (타임스탬프: {timestamp}) ===")
        
        if logs['stdout']:
            print(f"\n📄 표준 출력 (최근 {lines}줄):")
            for line in logs['stdout'][-lines:]:
                print(f"  {line.rstrip()}")
        
        if logs['stderr']:
            print(f"\n⚠️  오류 로그 (최근 {lines}줄):")
            for line in logs['stderr'][-lines:]:
                print(f"  {line.rstrip()}")
        
        print("="*60)

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='실험 진행 상황 모니터링')
    parser.add_argument('--mode', choices=['monitor', 'logs'], default='monitor',
                       help='모니터링 모드 (monitor: 연속 모니터링, logs: 최근 로그)')
    parser.add_argument('--interval', type=int, default=30,
                       help='모니터링 간격 (초, 기본값: 30)')
    parser.add_argument('--lines', type=int, default=20,
                       help='로그 표시 줄 수 (기본값: 20)')
    
    args = parser.parse_args()
    
    monitor = ExperimentMonitor()
    
    if args.mode == 'monitor':
        monitor.monitor_continuously(args.interval)
    elif args.mode == 'logs':
        monitor.show_recent_logs(args.lines)

if __name__ == "__main__":
    main() 
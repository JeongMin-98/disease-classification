#!/usr/bin/env python3
"""
7-fold 실험을 단계별로 병렬 실행하는 스크립트
GPU 메모리를 모니터링하고 OOM 발생 시 기록으로 남깁니다.
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# GPUtil import (선택적)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("GPUtil이 설치되지 않았습니다. GPU 모니터링이 제한됩니다.")

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로그 디렉토리 생성
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 메인 로거 설정
def setup_logger(name, log_file):
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 메인 로거 설정
main_logger = setup_logger('main', log_dir / 'main_experiments.log')

class GPUMonitor:
    """GPU 메모리 모니터링 클래스"""
    
    def __init__(self):
        self.gpu_info = {}
        self.update_gpu_info()
    
    def update_gpu_info(self):
        """GPU 정보 업데이트"""
        if not GPUTIL_AVAILABLE:
            return
            
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.gpu_info[gpu.id] = {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'utilization': gpu.load * 100 if gpu.load else 0
                }
        except Exception as e:
            main_logger.warning(f"GPU 정보 업데이트 실패: {e}")
    
    def get_gpu_memory_usage(self):
        """GPU 메모리 사용량 반환"""
        self.update_gpu_info()
        return self.gpu_info
    
    def log_gpu_status(self):
        """GPU 상태 로깅"""
        if not GPUTIL_AVAILABLE:
            main_logger.info("GPU 모니터링을 사용할 수 없습니다 (GPUtil 미설치)")
            return
            
        gpu_info = self.get_gpu_memory_usage()
        for gpu_id, info in gpu_info.items():
            main_logger.info(f"GPU {gpu_id} ({info['name']}): "
                           f"메모리 {info['memory_used']}/{info['memory_total']}MB "
                           f"({info['memory_used']/info['memory_total']*100:.1f}%) "
                           f"사용률: {info['utilization']:.1f}%")

class ExperimentProgressTracker:
    """실험 진행 상황 추적 클래스"""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_fold = 0
        self.total_folds = 7
        self.best_accuracy = 0.0
        self.current_accuracy = 0.0
        self.current_loss = 0.0
        self.status = "시작됨"
        
        # 실험별 로거 설정
        safe_name = re.sub(r'[^\w\-_]', '_', experiment_name)
        self.logger = setup_logger(f'experiment_{safe_name}', log_dir / f'{safe_name}.log')
        
    def update_progress(self, line):
        """진행 상황 업데이트"""
        # 에포크 정보 추출 (다양한 패턴 지원)
        epoch_patterns = [
            r'epoch\s+(\d+)/(\d+)',
            r'epoch[:\s]*(\d+)/(\d+)',
            r'(\d+)/(\d+)\s*epoch',
            r'epoch\s*(\d+)\s*/\s*(\d+)'
        ]
        
        for pattern in epoch_patterns:
            epoch_match = re.search(pattern, line.lower())
            if epoch_match:
                self.current_epoch = int(epoch_match.group(1))
                self.total_epochs = int(epoch_match.group(2))
                self.status = f"에포크 {self.current_epoch}/{self.total_epochs} 실행 중"
                break
        
        # 폴드 정보 추출 (다양한 패턴 지원)
        fold_patterns = [
            r'fold\s+(\d+)/(\d+)',
            r'fold[:\s]*(\d+)/(\d+)',
            r'(\d+)/(\d+)\s*fold',
            r'fold\s*(\d+)\s*/\s*(\d+)',
            r'k-fold\s*(\d+)\s*/\s*(\d+)'
        ]
        
        for pattern in fold_patterns:
            fold_match = re.search(pattern, line.lower())
            if fold_match:
                self.current_fold = int(fold_match.group(1))
                self.total_folds = int(fold_match.group(2))
                self.status = f"폴드 {self.current_fold}/{self.total_folds} 실행 중"
                break
        
        # 정확도 정보 추출 (다양한 패턴 지원)
        accuracy_patterns = [
            r'accuracy[:\s]*([\d.]+)',
            r'acc[:\s]*([\d.]+)',
            r'정확도[:\s]*([\d.]+)',
            r'accuracy\s*=\s*([\d.]+)',
            r'acc\s*=\s*([\d.]+)'
        ]
        
        for pattern in accuracy_patterns:
            accuracy_match = re.search(pattern, line.lower())
            if accuracy_match:
                try:
                    accuracy_value = float(accuracy_match.group(1))
                    self.current_accuracy = accuracy_value
                    if accuracy_value > self.best_accuracy:
                        self.best_accuracy = accuracy_value
                        self.logger.info(f"새로운 최고 정확도 달성: {accuracy_value:.4f}")
                except ValueError:
                    pass
                break
        
        # 손실 정보 추출 (다양한 패턴 지원)
        loss_patterns = [
            r'loss[:\s]*([\d.]+)',
            r'손실[:\s]*([\d.]+)',
            r'loss\s*=\s*([\d.]+)',
            r'train_loss[:\s]*([\d.]+)',
            r'val_loss[:\s]*([\d.]+)'
        ]
        
        for pattern in loss_patterns:
            loss_match = re.search(pattern, line.lower())
            if loss_match:
                try:
                    self.current_loss = float(loss_match.group(1))
                except ValueError:
                    pass
                break
        
        # 완료 상태 확인
        completion_patterns = [
            r'completed',
            r'finished',
            r'완료',
            r'종료',
            r'done',
            r'success'
        ]
        
        for pattern in completion_patterns:
            if re.search(pattern, line.lower()):
                self.status = "완료됨"
                break
        
        # 오류 상태 확인
        error_patterns = [
            r'error',
            r'오류',
            r'failed',
            r'실패',
            r'exception',
            r'out of memory',
            r'oom'
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, line.lower()):
                if 'out of memory' in line.lower() or 'oom' in line.lower():
                    self.status = "OOM 발생"
                else:
                    self.status = "오류 발생"
                break
        
        # 실험별 로그에 기록 (중요한 정보만)
        if any([
            epoch_match for epoch_match in [re.search(pattern, line.lower()) for pattern in epoch_patterns]
        ]) or any([
            fold_match for fold_match in [re.search(pattern, line.lower()) for pattern in fold_patterns]
        ]) or any([
            accuracy_match for accuracy_match in [re.search(pattern, line.lower()) for pattern in accuracy_patterns]
        ]) or any([
            loss_match for loss_match in [re.search(pattern, line.lower()) for pattern in loss_patterns]
        ]) or any([
            completion_match for completion_match in [re.search(pattern, line.lower()) for pattern in completion_patterns]
        ]) or any([
            error_match for error_match in [re.search(pattern, line.lower()) for pattern in error_patterns]
        ]):
            self.logger.info(f"진행상황 업데이트: {line.strip()}")
        
    def get_progress_summary(self):
        """진행 상황 요약"""
        elapsed_time = datetime.now() - self.start_time
        return {
            'status': self.status,
            'elapsed_time': str(elapsed_time).split('.')[0],
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_fold': self.current_fold,
            'total_folds': self.total_folds,
            'best_accuracy': self.best_accuracy,
            'current_accuracy': self.current_accuracy,
            'current_loss': self.current_loss
        }
        
    def log_progress(self):
        """진행 상황 로깅"""
        summary = self.get_progress_summary()
        
        # 진행률 계산
        epoch_progress = f"{summary['current_epoch']}/{summary['total_epochs']}" if summary['total_epochs'] > 0 else "0/0"
        fold_progress = f"{summary['current_fold']}/{summary['total_folds']}" if summary['total_folds'] > 0 else "0/0"
        
        # 진행률 퍼센트 계산
        epoch_percent = (summary['current_epoch'] / summary['total_epochs'] * 100) if summary['total_epochs'] > 0 else 0
        fold_percent = (summary['current_fold'] / summary['total_folds'] * 100) if summary['total_folds'] > 0 else 0
        
        progress_msg = (f"[{self.experiment_name}] 진행상황: "
                       f"상태={summary['status']}, "
                       f"경과시간={summary['elapsed_time']}, "
                       f"에포크={epoch_progress} ({epoch_percent:.1f}%), "
                       f"폴드={fold_progress} ({fold_percent:.1f}%), "
                       f"최고정확도={summary['best_accuracy']:.4f}, "
                       f"현재정확도={summary['current_accuracy']:.4f}, "
                       f"현재손실={summary['current_loss']:.4f}")
        
        self.logger.info(progress_msg)
        main_logger.info(progress_msg)

class ExperimentRunner:
    """실험 실행 클래스"""
    
    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.gpu_monitor = GPUMonitor()
        self.experiment_log = []
        self.results_dir = Path("experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.progress_trackers = {}  # 실험별 진행 상황 추적기
        
        # 실험 설정 - 단계별로 그룹화 (손과 발 데이터를 동시에 실행)
        self.experiment_groups = {
            'stage1_224_vgg19bn': [
                {
                    'name': '224x224_VGG19BN_Hand_OA_Normal',
                    'config': 'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_vgg19bn_kfold.yaml',
                    'output_dir': 'experiments/results/kfold_224_vgg19bn_hand_oa_normal',
                    'batch_size': 16,
                    'image_size': '224x224',
                    'data_type': 'hand'
                },
                {
                    'name': '224x224_VGG19BN_Foot_OA_Normal',
                    'config': 'experiments/image_exp/foot/foot_classifier_OA_Normal_224_vgg19bn_kfold.yaml',
                    'output_dir': 'experiments/results/kfold_224_vgg19bn_foot_oa_normal',
                    'batch_size': 16,
                    'image_size': '224x224',
                    'data_type': 'foot'
                }
            ],
            'stage2_224_resnet18': [
                {
                    'name': '224x224_ResNet18_Hand_OA_Normal',
                    'config': 'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_resnet18_kfold.yaml',
                    'output_dir': 'experiments/results/kfold_224_resnet18_hand_oa_normal',
                    'batch_size': 16,
                    'image_size': '224x224',
                    'data_type': 'hand'
                },
                {
                    'name': '224x224_ResNet18_Foot_OA_Normal',
                    'config': 'experiments/image_exp/foot/foot_classifier_OA_Normal_224_resnet18_kfold.yaml',
                    'output_dir': 'experiments/results/kfold_224_resnet18_foot_oa_normal',
                    'batch_size': 16,
                    'image_size': '224x224',
                    'data_type': 'foot'
                }
            ],
            'stage3_1024_vgg19bn': [
                {
                    'name': '1024x1024_VGG19BN_Hand_OA_Normal',
                    'config': 'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_1024_vgg19bn_kfold.yaml',
                    'output_dir': 'experiments/results/kfold_1024_vgg19bn_hand_oa_normal',
                    'batch_size': 16,
                    'image_size': '1024x1024',
                    'data_type': 'hand'
                },
                {
                    'name': '1024x1024_VGG19BN_Foot_OA_Normal',
                    'config': 'experiments/image_exp/foot/foot_classifier_OA_Normal_1024_vgg19bn_kfold.yaml',
                    'output_dir': 'experiments/results/kfold_1024_vgg19bn_foot_oa_normal',
                    'batch_size': 16,
                    'image_size': '1024x1024',
                    'data_type': 'foot'
                }
            ],
            'stage4_1024_resnet18': [
                {
                    'name': '1024x1024_ResNet18_Hand_OA_Normal',
                    'config': 'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_1024_resnet18_kfold.yaml',
                    'output_dir': 'experiments/results/kfold_1024_resnet18_hand_oa_normal',
                    'batch_size': 16,
                    'image_size': '1024x1024',
                    'data_type': 'hand'
                },
                {
                    'name': '1024x1024_ResNet18_Foot_OA_Normal',
                    'config': 'experiments/image_exp/foot/foot_classifier_OA_Normal_1024_resnet18_kfold.yaml',
                    'output_dir': 'experiments/results/kfold_1024_resnet18_foot_oa_normal',
                    'batch_size': 16,
                    'image_size': '1024x1024',
                    'data_type': 'foot'
                }
            ]
        }
    
    def run_single_experiment(self, experiment, seed=42):
        """단일 실험 실행"""
        start_time = datetime.now()
        experiment_name = experiment['name']
        config_path = experiment['config']
        output_dir = experiment['output_dir']
        
        main_logger.info(f"=== {experiment_name} 실험 시작 ===")
        main_logger.info(f"설정 파일: {config_path}")
        main_logger.info(f"출력 디렉토리: {output_dir}")
        main_logger.info(f"배치 크기: {experiment['batch_size']}")
        main_logger.info(f"데이터 타입: {experiment['data_type']}")
        
        # 진행 상황 추적기 생성
        progress_tracker = ExperimentProgressTracker(experiment_name)
        self.progress_trackers[experiment_name] = progress_tracker
        
        # 실험별 로그에 시작 정보 기록
        progress_tracker.logger.info(f"=== {experiment_name} 실험 시작 ===")
        progress_tracker.logger.info(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        progress_tracker.logger.info(f"설정 파일: {config_path}")
        progress_tracker.logger.info(f"출력 디렉토리: {output_dir}")
        progress_tracker.logger.info(f"배치 크기: {experiment['batch_size']}")
        progress_tracker.logger.info(f"데이터 타입: {experiment['data_type']}")
        progress_tracker.logger.info(f"시드: {seed}")
        
        # GPU 상태 로깅
        self.gpu_monitor.log_gpu_status()
        
        # 실험 실행
        cmd = [
            'python', 'tool/train_kfold.py',
            '--cfg', config_path,
            '--seed', str(seed)
        ]
        
        try:
            # 실험 실행
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 실시간 출력 모니터링
            stdout_lines = []
            stderr_lines = []
            last_progress_log = time.time()
            last_gpu_log = time.time()
            
            progress_tracker.logger.info(f"프로세스 시작 (PID: {process.pid})")
            
            while True:
                stdout_line = process.stdout.readline() if process.stdout else None
                stderr_line = process.stderr.readline() if process.stderr else None
                
                if stdout_line:
                    line = stdout_line.strip()
                    stdout_lines.append(line)
                    main_logger.info(f"[{experiment_name}] {line}")
                    
                    # 실험별 로그에 기록
                    progress_tracker.logger.info(f"STDOUT: {line}")
                    
                    # 진행 상황 업데이트
                    progress_tracker.update_progress(line)
                    
                    # 30초마다 진행 상황 로깅
                    current_time = time.time()
                    if current_time - last_progress_log >= 30:
                        progress_tracker.log_progress()
                        last_progress_log = current_time
                    
                    # 5분마다 GPU 상태 로깅
                    if current_time - last_gpu_log >= 300:
                        self.gpu_monitor.log_gpu_status()
                        last_gpu_log = current_time
                
                if stderr_line:
                    line = stderr_line.strip()
                    stderr_lines.append(line)
                    main_logger.warning(f"[{experiment_name}] {line}")
                    
                    # 실험별 로그에 기록
                    progress_tracker.logger.warning(f"STDERR: {line}")
                    
                    # 진행 상황 업데이트
                    progress_tracker.update_progress(line)
                
                # 프로세스 종료 확인
                if process.poll() is not None:
                    break
            
            # 남은 출력 읽기
            remaining_stdout, remaining_stderr = process.communicate()
            stdout_lines.extend(remaining_stdout.strip().split('\n') if remaining_stdout else [])
            stderr_lines.extend(remaining_stderr.strip().split('\n') if remaining_stderr else [])
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            # 결과 분석
            success = process.returncode == 0
            oom_occurred = any('out of memory' in line.lower() or 'oom' in line.lower() 
                             for line in stderr_lines)
            
            # 실험별 로그에 완료 정보 기록
            progress_tracker.logger.info(f"=== {experiment_name} 실험 완료 ===")
            progress_tracker.logger.info(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            progress_tracker.logger.info(f"소요 시간: {duration}")
            progress_tracker.logger.info(f"성공 여부: {'성공' if success else '실패'}")
            progress_tracker.logger.info(f"반환 코드: {process.returncode}")
            if oom_occurred:
                progress_tracker.logger.error("OOM 발생")
            
            result = {
                'experiment_name': experiment_name,
                'config_path': config_path,
                'output_dir': output_dir,
                'batch_size': experiment['batch_size'],
                'image_size': experiment['image_size'],
                'data_type': experiment['data_type'],
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'success': success,
                'return_code': process.returncode,
                'oom_occurred': oom_occurred,
                'stdout_lines': stdout_lines[-10:],  # 마지막 10줄만 저장
                'stderr_lines': stderr_lines[-10:],  # 마지막 10줄만 저장
                'final_progress': progress_tracker.get_progress_summary()
            }
            
            if success:
                main_logger.info(f"=== {experiment_name} 실험 성공 완료 (소요시간: {duration}) ===")
                progress_tracker.status = "완료됨"
            else:
                if oom_occurred:
                    main_logger.error(f"=== {experiment_name} 실험 실패: OOM 발생 (소요시간: {duration}) ===")
                    progress_tracker.status = "OOM 발생"
                else:
                    main_logger.error(f"=== {experiment_name} 실험 실패 (소요시간: {duration}) ===")
                    progress_tracker.status = "실패"
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = end_time - start_time
            
            progress_tracker.logger.error(f"실험 실행 중 예외 발생: {e}")
            
            result = {
                'experiment_name': experiment_name,
                'config_path': config_path,
                'output_dir': output_dir,
                'batch_size': experiment['batch_size'],
                'image_size': experiment['image_size'],
                'data_type': experiment['data_type'],
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'success': False,
                'return_code': -1,
                'oom_occurred': False,
                'error': str(e),
                'final_progress': progress_tracker.get_progress_summary()
            }
            
            main_logger.error(f"=== {experiment_name} 실험 실행 중 예외 발생: {e} ===")
            return result
    
    def log_all_progress(self):
        """모든 실험의 진행 상황을 로깅"""
        if not self.progress_trackers:
            return
            
        main_logger.info("=== 모든 실험 진행 상황 ===")
        for experiment_name, tracker in self.progress_trackers.items():
            tracker.log_progress()
        main_logger.info("==========================")
    
    def run_experiment_group(self, group_name, experiments, seed=42):
        """실험 그룹 실행 (손과 발 데이터를 동시에 병렬 실행)"""
        main_logger.info(f"=== {group_name} 단계 시작 ===")
        main_logger.info(f"실험 수: {len(experiments)}")
        main_logger.info("손과 발 데이터를 동시에 병렬로 실행합니다.")
        
        start_time = datetime.now()
        
        # 실험별 로그에 그룹 시작 정보 기록
        for experiment in experiments:
            if experiment['name'] in self.progress_trackers:
                self.progress_trackers[experiment['name']].logger.info(f"=== {group_name} 그룹 시작 ===")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 실험들을 동시에 실행
            future_to_experiment = {
                executor.submit(self.run_single_experiment, experiment, seed): experiment 
                for experiment in experiments
            }
            
            # 완료된 실험들을 수집
            for future in as_completed(future_to_experiment):
                experiment = future_to_experiment[future]
                try:
                    result = future.result()
                    self.experiment_log.append(result)
                    
                    # 실험별 로그에 그룹 완료 정보 기록
                    if experiment['name'] in self.progress_trackers:
                        self.progress_trackers[experiment['name']].logger.info(f"=== {group_name} 그룹 내 실험 완료 ===")
                        
                except Exception as e:
                    main_logger.error(f"{experiment['name']} 실험 실행 중 예외 발생: {e}")
                    self.experiment_log.append({
                        'experiment_name': experiment['name'],
                        'config_path': experiment['config'],
                        'output_dir': experiment['output_dir'],
                        'batch_size': experiment['batch_size'],
                        'image_size': experiment['image_size'],
                        'data_type': experiment['data_type'],
                        'start_time': datetime.now().isoformat(),
                        'end_time': datetime.now().isoformat(),
                        'duration_seconds': 0,
                        'success': False,
                        'return_code': -1,
                        'oom_occurred': False,
                        'error': str(e)
                    })
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        main_logger.info(f"=== {group_name} 단계 완료 (소요시간: {duration}) ===")
        
        # 실험별 로그에 그룹 완료 정보 기록
        for experiment in experiments:
            if experiment['name'] in self.progress_trackers:
                self.progress_trackers[experiment['name']].logger.info(f"=== {group_name} 그룹 완료 (소요시간: {duration}) ===")
        
        return duration
    
    def run_all_experiments(self, seed=42):
        """모든 실험을 단계별로 실행"""
        main_logger.info("=== 7-fold 실험 단계별 병렬 실행 시작 ===")
        main_logger.info(f"최대 동시 실행 수: {self.max_workers}")
        main_logger.info(f"시드: {seed}")
        
        # 시작 전 GPU 상태 로깅
        self.gpu_monitor.log_gpu_status()
        
        total_start_time = datetime.now()
        
        # 단계별로 실행 (각 단계에서 손과 발 데이터를 동시에 병렬 실행)
        stage_order = [
            'stage1_224_vgg19bn',
            'stage2_224_resnet18', 
            'stage3_1024_vgg19bn',
            'stage4_1024_resnet18'
        ]
        
        for stage in stage_order:
            if stage in self.experiment_groups:
                experiments = self.experiment_groups[stage]
                stage_duration = self.run_experiment_group(stage, experiments, seed)
                
                # 다음 단계 시작 전 잠시 대기
                if stage != stage_order[-1]:  # 마지막 단계가 아니면
                    main_logger.info("다음 단계 시작 전 30초 대기...")
                    time.sleep(30)
        
        total_end_time = datetime.now()
        total_duration = total_end_time - total_start_time
        
        # 결과 요약
        self.save_results(total_duration)
        
        main_logger.info("=== 모든 실험 완료 ===")
        main_logger.info(f"총 소요시간: {total_duration}")
        
        return self.experiment_log
    
    def save_results(self, total_duration):
        """결과 저장"""
        results = {
            'total_duration_seconds': total_duration.total_seconds(),
            'start_time': datetime.now().isoformat(),
            'experiments': self.experiment_log,
            'summary': self.generate_summary()
        }
        
        # JSON 파일로 저장
        results_file = self.results_dir / f"parallel_experiments_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 요약 로그 파일로 저장
        summary_file = self.results_dir / f"parallel_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_summary_text())
        
        main_logger.info(f"결과가 저장되었습니다:")
        main_logger.info(f"- 상세 결과: {results_file}")
        main_logger.info(f"- 요약 결과: {summary_file}")
    
    def generate_summary(self):
        """결과 요약 생성"""
        total_experiments = len(self.experiment_log)
        successful_experiments = sum(1 for exp in self.experiment_log if exp.get('success', False))
        oom_experiments = sum(1 for exp in self.experiment_log if exp.get('oom_occurred', False))
        
        return {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'failed_experiments': total_experiments - successful_experiments,
            'oom_experiments': oom_experiments,
            'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0
        }
    
    def generate_summary_text(self):
        """요약 텍스트 생성"""
        summary = self.generate_summary()
        
        text = f"""=== 7-fold 실험 단계별 병렬 실행 결과 요약 ===
실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

전체 실험 수: {summary['total_experiments']}
성공한 실험 수: {summary['successful_experiments']}
실패한 실험 수: {summary['failed_experiments']}
OOM 발생 실험 수: {summary['oom_experiments']}
성공률: {summary['success_rate']:.1%}

=== 개별 실험 결과 ===
"""
        
        for exp in self.experiment_log:
            status = "성공" if exp.get('success', False) else "실패"
            oom_status = " (OOM 발생)" if exp.get('oom_occurred', False) else ""
            duration = exp.get('duration_seconds', 0)
            data_type = exp.get('data_type', 'unknown')
            
            # 진행 상황 정보 추가
            final_progress = exp.get('final_progress', {})
            progress_info = ""
            if final_progress:
                progress_info = f" (최고정확도: {final_progress.get('best_accuracy', 'N/A')})"
            
            text += f"{exp['experiment_name']} ({data_type}): {status}{oom_status} (소요시간: {duration:.1f}초){progress_info}\n"
        
        return text

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='7-fold 실험을 단계별로 병렬 실행')
    parser.add_argument('--max-workers', type=int, default=2, 
                       help='최대 동시 실행 수 (기본값: 2)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='랜덤 시드 (기본값: 42)')
    
    args = parser.parse_args()
    
    # 실험 시작 시간 기록
    start_time = datetime.now()
    main_logger.info(f"=== 7-fold 실험 단계별 병렬 실행 시작 ===")
    main_logger.info(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main_logger.info(f"최대 동시 실행 수: {args.max_workers}")
    main_logger.info(f"시드: {args.seed}")
    main_logger.info(f"로그 디렉토리: {log_dir}")
    
    # 실험 실행
    runner = ExperimentRunner(max_workers=args.max_workers)
    results = runner.run_all_experiments(seed=args.seed)
    
    # 실험 종료 시간 기록
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    main_logger.info(f"=== 7-fold 실험 단계별 병렬 실행 완료 ===")
    main_logger.info(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main_logger.info(f"총 소요 시간: {total_duration}")
    
    # 최종 요약 출력
    summary = runner.generate_summary()
    main_logger.info(f"=== 최종 결과 요약 ===")
    main_logger.info(f"전체 실험 수: {summary['total_experiments']}")
    main_logger.info(f"성공한 실험 수: {summary['successful_experiments']}")
    main_logger.info(f"실패한 실험 수: {summary['failed_experiments']}")
    main_logger.info(f"OOM 발생 실험 수: {summary['oom_experiments']}")
    main_logger.info(f"성공률: {summary['success_rate']:.1%}")
    
    print(f"\n=== 최종 결과 요약 ===")
    print(f"전체 실험 수: {summary['total_experiments']}")
    print(f"성공한 실험 수: {summary['successful_experiments']}")
    print(f"실패한 실험 수: {summary['failed_experiments']}")
    print(f"OOM 발생 실험 수: {summary['oom_experiments']}")
    print(f"성공률: {summary['success_rate']:.1%}")
    print(f"총 소요 시간: {total_duration}")
    print(f"로그 파일 위치: {log_dir}")

if __name__ == "__main__":
    main() 
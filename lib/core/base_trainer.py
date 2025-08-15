import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.cuda.amp import GradScaler, autocast
import wandb
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from core.optim_utils import optimizer_builder, scheduler_builder
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from utils.utils import EarlyStopping
import json
import shutil
from PIL import Image

class BaseTrainer(ABC):
    """
    모든 trainer의 기본 클래스
    공통적인 훈련 로직을 구현하고, 데이터셋별 특화 로직은 하위 클래스에서 구현
    """
    
    def __init__(self, cfg, model, device, output_dir):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # 훈련 상태
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.final_model_state = None  # 마지막 에포크의 모델 상태 저장
        
        # Mixed Precision 설정
        self.use_amp = getattr(cfg.TRAIN, 'USE_AMP', True)  # 기본값 True
        if self.use_amp and device == 'cuda':
            self.scaler = GradScaler()
            self.logger.info("Mixed Precision Training enabled")
        else:
            self.scaler = None
            self.logger.info("Mixed Precision Training disabled")
        
        # 옵티마이저 및 스케줄러 설정
        self.optimizer = optimizer_builder(cfg, model)
        # StepLR, MultiStepLR 등에서 step/milestones가 리스트면 첫 번째(정렬 후 첫 번째) 값만 사용
        if hasattr(cfg.TRAIN, 'LR_STEP') and isinstance(cfg.TRAIN.LR_STEP, list):
            lr_step = sorted(cfg.TRAIN.LR_STEP)[0]
            cfg.defrost()
            cfg.TRAIN.LR_STEP = lr_step
            cfg.freeze()
        self.scheduler = scheduler_builder(cfg, self.optimizer)
        
        # WandB 초기화
        self.wandb_run = None
        
        # EarlyStopping 인스턴스 선언
        patience = getattr(cfg.TRAIN, 'PATIENCE', 5)
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, min_epochs=cfg.TRAIN.MIN_EPOCHS)
        
    def init_wandb(self, run_name: str, tags: list):
        """WandB 초기화"""
        self.wandb_run = wandb.init(
            project=self._get_wandb_project(),
            name=run_name,
            config=dict(self.cfg),
            tags=tags,
            reinit=True
        )
    
    @abstractmethod
    def _get_wandb_project(self) -> str:
        """WandB 프로젝트명 반환 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치 데이터 준비 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def _compute_loss(self, outputs, labels) -> torch.Tensor:
        """손실 계산 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def _compute_accuracy(self, outputs, labels) -> float:
        """정확도 계산 (하위 클래스에서 구현)"""
        # 이진 분류일 때 안전하게 동작하는 기본 구현 제공
        outputs = outputs.detach().cpu().squeeze()
        labels = labels.detach().cpu().squeeze()
        if outputs.ndim > 1:
            # 다중 분류: argmax
            predicted = outputs.argmax(dim=1)
        else:
            # 이진 분류: 0/1로 변환
            predicted = (outputs > 0).long()
        correct = (predicted == labels.long()).sum().item()
        return correct
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """한 에포크 훈련"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Gradient clipping 설정
        use_grad_clip = getattr(self.cfg.TRAIN, 'USE_GRAD_CLIP', False)
        grad_clip_norm = getattr(self.cfg.TRAIN, 'GRAD_CLIP_NORM', float('inf'))
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {self.current_epoch}")):
            # 배치 데이터 준비
            inputs, labels = self._prepare_batch(batch)
            
            # Mixed Precision 훈련
            if self.use_amp and self.scaler is not None:
                with autocast():
                    # 순전파
                    outputs = self.model(inputs)
                    # 손실 계산
                    loss = self._compute_loss(outputs, labels)
                
                # 역전파 (scaler 사용)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (Mixed Precision)
                if use_grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 일반 훈련
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, labels)
                loss.backward()
                
                # Gradient clipping (일반)
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                
                self.optimizer.step()
            
            # 통계 업데이트
            train_loss += loss.item()
            train_correct += self._compute_accuracy(outputs, labels)
            train_total += labels.size(0)
        
        # 평균 계산
        avg_loss = train_loss / len(train_loader)
        avg_acc = 100. * train_correct / train_total
        
        return avg_loss, avg_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """한 에포크 검증"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {self.current_epoch}"):
                # 배치 데이터 준비
                inputs, labels = self._prepare_batch(batch)
                
                # Mixed Precision 검증
                if self.use_amp and self.scaler is not None:
                    with autocast():
                        # 순전파
                        outputs = self.model(inputs)
                        # 손실 및 정확도 계산
                        loss = self._compute_loss(outputs, labels)
                else:
                    # 일반 검증
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, labels)
                
                val_loss += loss.item()
                val_correct += self._compute_accuracy(outputs, labels)
                val_total += labels.size(0)
        
        # 평균 계산
        avg_loss = val_loss / len(val_loader)
        avg_acc = 100. * val_correct / val_total
        
        return avg_loss, avg_acc
    
    def test_epoch(self, test_loader):
        """테스트 에포크"""
        self.model.eval()
        
        # 메모리 누수 방지: 테스트 시작 전 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("Test 시작 전 GPU 메모리 정리됨")
        
        test_correct = 0
        test_total = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Test"):
                # 배치 데이터 준비
                inputs, labels = self._prepare_batch(batch)
                
                # 순전파
                outputs = self.model(inputs)
                
                # 정확도 계산
                test_correct += self._compute_accuracy(outputs, labels)
                test_total += labels.size(0)

                # 이진 분류 기준 예측값/정답 저장
                if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
                    preds = (outputs.squeeze() > 0).long().cpu().numpy()
                else:
                    preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                # 배치 처리 후 메모리 정리
                del inputs, labels, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 테스트 완료 후 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("Test 완료 후 GPU 메모리 정리됨")
        
        # 평균 정확도 계산
        test_acc = 100. * test_correct / test_total
        # 정밀도, 재현율 계산
        test_precision = precision_score(all_labels, all_preds, average='binary' if len(set(all_labels))==2 else 'macro', zero_division='warn')
        test_recall = recall_score(all_labels, all_preds, average='binary' if len(set(all_labels))==2 else 'macro', zero_division='warn')
        return {'test_acc': test_acc, 'test_precision': test_precision, 'test_recall': test_recall}
    
    def update_scheduler(self, val_loss: float):
        """스케줄러 업데이트"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.debug(f"Learning rate updated to: {current_lr:.6f}")
            else:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.debug(f"Learning rate updated to: {current_lr:.6f}")
        else:
            # Scheduler가 None인 경우 고정 학습률 사용
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.current_epoch == 0:  # 첫 에포크에서만 로깅
                self.logger.info(f"No scheduler used. Fixed learning rate: {current_lr:.6f}")
    
    def save_checkpoint(self, filepath: str):
        """체크포인트 저장 (로컬만)"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state,
            'final_model_state': self.final_model_state,  # final_model_state 추가
            'cfg': self.cfg
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 로컬에만 저장 (실험 중에는 wandb 업로드하지 않음)
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def _upload_checkpoint_to_wandb(self, filepath: str, checkpoint_type: str):
        """체크포인트를 wandb에 업로드 (실험 종료 시에만 사용)"""
        if self.wandb_run is not None:
            try:
                # 파일명 추출
                filename = os.path.basename(filepath)
                
                # wandb에 아티팩트로 업로드
                artifact = wandb.Artifact(
                    name=f"{checkpoint_type}-{self.wandb_run.id}",
                    type="model",
                    description=f"{checkpoint_type} model checkpoint from epoch {self.current_epoch}"
                )
                
                artifact.add_file(filepath, name=filename)
                self.wandb_run.log_artifact(artifact)
                
                self.logger.info(f"{checkpoint_type} checkpoint uploaded to wandb: {filename}")
                
            except Exception as e:
                self.logger.warning(f"Failed to upload {checkpoint_type} checkpoint to wandb: {e}")
    
    def save_best_model(self, val_loss: float):
        """최고 모델 저장 (로컬에만 저장)"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            
            # 로컬에 체크포인트 저장
            if self.wandb_run is not None:
                # wandb run의 디렉토리 사용
                best_model_path = os.path.join(self.wandb_run.dir, 'best_model.pth')
            elif self.output_dir is not None:
                # output_dir이 있으면 사용
                best_model_path = os.path.join(self.output_dir, 'best_model.pth')
            else:
                # 기본값으로 현재 디렉토리 사용
                best_model_path = 'best_model.pth'
            
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            self.save_checkpoint(best_model_path)
            
            self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            self.logger.info(f"Checkpoint saved to: {best_model_path}")
            
            # wandb에 best model 메트릭만 로깅 (체크포인트는 업로드하지 않음)
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "best_val_loss": val_loss,
                    "best_model_epoch": self.current_epoch
                })
    
    def load_best_model(self):
        """최고 모델 로드"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Best model loaded")
        else:
            self.logger.warning("No best model state available")
    
    def load_final_model(self):
        """최종 모델 로드"""
        if self.final_model_state is not None:
            self.model.load_state_dict(self.final_model_state)
            self.logger.info("Final model loaded")
        else:
            self.logger.warning("No final model state available")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """메트릭 로깅"""
        if self.wandb_run is not None:
            self.wandb_run.log(metrics)
        
        # 콘솔 로깅
        log_str = f"Epoch {self.current_epoch}: "
        for key, value in metrics.items():
            if key != "epoch":
                log_str += f"{key}: {value:.4f}, "
        self.logger.info(log_str.rstrip(", "))
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              test_loader: DataLoader, num_epochs: int) -> float:
        """전체 훈련 과정"""
        self.logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 훈련
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 스케줄러 업데이트
            self.update_scheduler(val_loss)
            
            # 최고 모델 저장
            self.save_best_model(val_loss)
            
            # 마지막 에포크의 모델 상태 저장 (final_model)
            self.final_model_state = self.model.state_dict().copy()
            
            # EarlyStopping 체크
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # 로깅
            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch
            }
            self.log_metrics(metrics)
        
        # 최고 모델로 테스트
        self.load_best_model()
        best_test_metrics = self.test_epoch(test_loader)
        
        # 최종 모델로 테스트
        self.load_final_model()
        final_test_metrics = {'final_test_acc': 0, 'final_test_precision': 0, 'final_test_recall': 0}
        test_metrics = self.test_epoch(test_loader)
        final_test_metrics['final_test_acc'] = test_metrics['test_acc']
        final_test_metrics['final_test_precision'] = test_metrics['test_precision']
        final_test_metrics['final_test_recall'] = test_metrics['test_recall']   
        
        # 최종 결과 로깅
        if self.wandb_run is not None:
            self.wandb_run.log(best_test_metrics)
            self.wandb_run.log(final_test_metrics)
        
        self.logger.info(f"Best Model - Test Accuracy: {best_test_metrics['test_acc']:.2f}% | Precision: {best_test_metrics['test_precision']:.4f} | Recall: {best_test_metrics['test_recall']:.4f}")
        self.logger.info(f"Final Model - Test Accuracy: {final_test_metrics['final_test_acc']:.2f}% | Precision: {final_test_metrics['final_test_precision']:.4f} | Recall: {final_test_metrics['final_test_recall']:.4f}")
        
        # 최종 체크포인트 저장
        if self.wandb_run is not None:
            # wandb run의 디렉토리 사용
            final_checkpoint_path = os.path.join(self.wandb_run.dir, 'final_model.pth')
        elif self.output_dir is not None:
            # output_dir이 있으면 사용
            final_checkpoint_path = os.path.join(self.output_dir, 'final_model.pth')
        else:
            # 기본값으로 현재 디렉토리 사용
            final_checkpoint_path = 'final_model.pth'
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(final_checkpoint_path), exist_ok=True)
        self.save_checkpoint(final_checkpoint_path)
        self.logger.info(f"Final checkpoint saved to: {final_checkpoint_path}")
        
        # 실험 종료 시에만 wandb에 체크포인트 업로드
        if self.wandb_run is not None:
            # best_model 업로드
            if self.best_model_state is not None:
                best_model_path = os.path.join(self.wandb_run.dir, 'best_model.pth')
                if os.path.exists(best_model_path):
                    self._upload_checkpoint_to_wandb(best_model_path, "best_model")
                else:
                    self.logger.warning("Best model checkpoint file not found")
            
            # final_model 업로드
            if os.path.exists(final_checkpoint_path):
                self._upload_checkpoint_to_wandb(final_checkpoint_path, "final_model")
            else:
                self.logger.warning("Final model checkpoint file not found")
        
        # 오분류 케이스 저장 (옵션)
        save_misclassified = getattr(self.cfg.TRAIN, 'SAVE_MISCLASSIFIED_CASES', True)
        if save_misclassified:
            self.logger.info("Saving misclassified cases...")
            
            # Best model 오분류 케이스 저장
            self.load_best_model()
            best_misclassified_dir, best_metadata_path = self.save_misclassified_cases(test_loader, "best")
            self.logger.info(f"Best model misclassified cases saved to: {best_misclassified_dir}")
            self.logger.info(f"Best model misclassified metadata saved to: {best_metadata_path}")
            if self.wandb_run is not None:
                # wandb에 오분류 케이스 디렉토리와 메타데이터 업로드
                if os.path.exists(best_misclassified_dir):
                    try:
                        # 올바른 wandb Artifact 생성 방법
                        artifact = wandb.Artifact(
                            name=f"misclassified_cases_best_{self.wandb_run.id}",
                            type="misclassified_cases",
                            description="Best model misclassified cases"
                        )
                        artifact.add_dir(best_misclassified_dir)
                        self.wandb_run.log_artifact(artifact)
                        self.logger.info("Best model misclassified cases uploaded to wandb")
                    except Exception as e:
                        self.logger.warning(f"Failed to upload misclassified cases to wandb: {e}")
                if os.path.exists(best_metadata_path):
                    self.wandb_run.save(best_metadata_path)
            
            # Final model 오분류 케이스 저장
            self.load_final_model()
            final_misclassified_dir, final_metadata_path = self.save_misclassified_cases(test_loader, "final")
            
            # Final model misclassified cases를 wandb에 업로드
            if self.wandb_run is not None and os.path.exists(final_misclassified_dir):
                try:
                    # Final model misclassified cases artifact 생성
                    artifact = wandb.Artifact(
                        name=f"misclassified_cases_final_{self.wandb_run.id}",
                        type="misclassified_cases",
                        description="Final model misclassified cases"
                    )
                    artifact.add_dir(final_misclassified_dir)
                    self.wandb_run.log_artifact(artifact)
                    self.logger.info("Final model misclassified cases uploaded to wandb")
                except Exception as e:
                    self.logger.warning(f"Failed to upload final misclassified cases to wandb: {e}")
            
            if os.path.exists(final_metadata_path):
                self.wandb_run.save(final_metadata_path)
            
            self.logger.info("Misclassified cases analysis completed!")
        
        # WandB 종료
        if self.wandb_run is not None:
            self.wandb_run.finish()
        
        return best_test_metrics['test_acc']  # 기존과의 호환성을 위해 best model의 정확도 반환
    
    def load_checkpoint(self, filepath: str):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_model_state = checkpoint['best_model_state']
        self.final_model_state = checkpoint.get('final_model_state', None)  # final_model_state 로드
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {filepath}") 
    
    @classmethod
    def download_checkpoint_from_wandb(cls, run_id: str, checkpoint_type: str = "best_model", project_name: Optional[str] = None):
        """
        wandb에서 체크포인트를 다운로드하는 클래스 메서드
        
        Args:
            run_id: wandb run ID
            checkpoint_type: "best_model" 또는 "final_model"
            project_name: wandb 프로젝트명 (None이면 기본값 사용)
        
        Returns:
            str: 다운로드된 체크포인트 파일 경로
        """
        try:
            import wandb
            
            # wandb API 초기화
            api = wandb.Api()
            
            # 프로젝트명이 없으면 기본값 사용
            if project_name is None:
                project_name = "medical-classification-original"  # 기본값
            
            # run 가져오기
            run = api.run(f"{project_name}/{run_id}")
            
            # 아티팩트 찾기
            artifact_name = f"{checkpoint_type}-{run_id}"
            artifact = api.artifact(f"{project_name}/{artifact_name}:latest")
            
            # 체크포인트 다운로드
            download_dir = artifact.download()
            
            # 체크포인트 파일 경로 찾기
            checkpoint_files = [f for f in os.listdir(download_dir) if f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_path = os.path.join(download_dir, checkpoint_files[0])
                print(f"Checkpoint downloaded to: {checkpoint_path}")
                return checkpoint_path
            else:
                raise FileNotFoundError("No checkpoint file found in artifact")
                
        except Exception as e:
            print(f"Failed to download checkpoint from wandb: {e}")
            return None
    
    def save_misclassified_cases(self, test_loader: DataLoader, model_type: str = "best", output_dir: Optional[str] = None):
        """
        테스트 세트에서 오분류된 케이스들을 저장하는 메서드
        
        Args:
            test_loader: 테스트 데이터로더
            model_type: "best" 또는 "final" 모델 타입
            output_dir: 저장할 디렉토리 (None이면 wandb.run.dir 또는 self.output_dir 사용)
        """
        self.model.eval()
        misclassified_cases = []
        
        # 저장 디렉토리 설정
        if output_dir is None:
            if self.wandb_run is not None:
                base_output_dir = self.wandb_run.dir
            elif self.output_dir is not None:
                base_output_dir = self.output_dir
            else:
                base_output_dir = "."
        else:
            base_output_dir = output_dir
        
        # 오분류 케이스 저장 폴더 생성
        misclassified_dir = os.path.join(base_output_dir, f"misclassified_cases_{model_type}")
        images_dir = os.path.join(misclassified_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        self.logger.info(f"Analyzing misclassified cases with {model_type} model...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Analyzing {model_type} model misclassifications")):
                # 배치 데이터 준비
                inputs, labels = self._prepare_batch(batch)
                
                # 순전파
                outputs = self.model(inputs)
                
                # 예측값 계산
                if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
                    # 이진 분류
                    predictions = (outputs.squeeze() > 0).long().cpu()
                    probabilities = torch.sigmoid(outputs.squeeze()).cpu()
                else:
                    # 다중 분류
                    predictions = outputs.argmax(dim=1).cpu()
                    probabilities = torch.softmax(outputs, dim=1).cpu()
                
                labels_cpu = labels.cpu()
                
                # 오분류 케이스 찾기
                misclassified_mask = (predictions != labels_cpu)
                
                for i in range(len(inputs)):
                    if misclassified_mask[i]:
                        # 원본 배치에서 메타데이터 추출
                        if isinstance(batch, dict):
                            # MedicalImageDataset의 경우
                            patient_id = batch.get('patient_id', [None])[i] if 'patient_id' in batch else f"unknown_{batch_idx}_{i}"
                            original_image = batch.get('image', inputs)[i]
                        else:
                            # 다른 데이터셋의 경우
                            patient_id = f"patient_{batch_idx}_{i}"
                            original_image = inputs[i]
                        
                        true_label = int(labels_cpu[i].item())
                        pred_label = int(predictions[i].item())
                        
                        # 확률 정보
                        if len(probabilities.shape) == 1:
                            # 이진 분류
                            confidence = float(probabilities[i].item())
                            prob_info = {
                                "predicted_class_prob": confidence if pred_label == 1 else 1 - confidence,
                                "true_class_prob": confidence if true_label == 1 else 1 - confidence
                            }
                        else:
                            # 다중 분류
                            confidence = float(probabilities[i][pred_label].item())
                            prob_info = {
                                "predicted_class_prob": confidence,
                                "true_class_prob": float(probabilities[i][true_label].item()),
                                "all_class_probs": probabilities[i].tolist()
                            }
                        
                        # 클래스명 변환 (가능한 경우)
                        true_class_name = self._get_class_name(true_label)
                        pred_class_name = self._get_class_name(pred_label)
                        
                        # 이미지 저장
                        image_filename = f"{patient_id}_true-{true_class_name}_pred-{pred_class_name}_conf-{confidence:.3f}.png"
                        image_path = os.path.join(images_dir, image_filename)
                        
                        # 이미지를 PIL 형태로 변환하여 저장
                        self._save_tensor_as_image(original_image, image_path)
                        
                        # 메타데이터 저장
                        case_info = {
                            "patient_id": patient_id,
                            "image_filename": image_filename,
                            "true_label": true_label,
                            "predicted_label": pred_label,
                            "true_class_name": true_class_name,
                            "predicted_class_name": pred_class_name,
                            "confidence": confidence,
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                            **prob_info
                        }
                        
                        misclassified_cases.append(case_info)
        
        # JSON 파일로 메타데이터 저장
        metadata_path = os.path.join(misclassified_dir, f"misclassified_metadata_{model_type}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(misclassified_cases, f, indent=2, ensure_ascii=False)
        
        # 요약 정보 로깅
        total_misclassified = len(misclassified_cases)
        try:
            total_samples = len(test_loader.dataset)  # type: ignore
        except:
            # dataset에 __len__이 없는 경우 대안 방법
            total_samples = sum(1 for _ in test_loader.dataset)
        
        misclassification_rate = (total_misclassified / total_samples) * 100
        
        self.logger.info(f"=== {model_type.upper()} Model Misclassification Analysis ===")
        self.logger.info(f"Total test samples: {total_samples}")
        self.logger.info(f"Misclassified samples: {total_misclassified}")
        self.logger.info(f"Misclassification rate: {misclassification_rate:.2f}%")
        self.logger.info(f"Misclassified images saved to: {images_dir}")
        self.logger.info(f"Metadata saved to: {metadata_path}")
        
        # wandb에 요약 정보 로깅
        if self.wandb_run is not None:
            self.wandb_run.log({
                f"{model_type}_misclassification_rate": misclassification_rate,
                f"{model_type}_misclassified_count": total_misclassified,
                f"{model_type}_total_test_samples": total_samples
            })
        
        return misclassified_dir, metadata_path
    
    def _get_class_name(self, label_idx: int) -> str:
        """라벨 인덱스를 클래스명으로 변환"""
        # 데이터셋에서 클래스명 매핑 정보 가져오기 시도
        if hasattr(self.cfg, 'DATASET') and hasattr(self.cfg.DATASET, 'INCLUDE_CLASSES'):
            classes = self.cfg.DATASET.INCLUDE_CLASSES
            if label_idx < len(classes):
                return classes[label_idx]
        
        return f"class_{label_idx}"
    
    def _save_tensor_as_image(self, tensor: torch.Tensor, save_path: str):
        """텐서를 이미지 파일로 저장"""
        try:
            # 텐서를 numpy로 변환
            if tensor.dim() == 3:  # [C, H, W]
                # 정규화 해제 (평균과 표준편차 사용)
                if hasattr(self.cfg.DATASET, 'MEAN') and hasattr(self.cfg.DATASET, 'STD'):
                    mean = torch.tensor(self.cfg.DATASET.MEAN).view(-1, 1, 1)
                    std = torch.tensor(self.cfg.DATASET.STD).view(-1, 1, 1)
                    tensor = tensor * std + mean
                
                # [0, 1] 범위로 클리핑
                tensor = torch.clamp(tensor, 0, 1)
                
                # numpy로 변환하고 [H, W, C]로 변경
                if tensor.shape[0] == 1:  # Grayscale
                    image_array = tensor.squeeze(0).numpy()
                    image_array = (image_array * 255).astype(np.uint8)
                    image = Image.fromarray(image_array, mode='L')
                else:  # RGB
                    image_array = tensor.permute(1, 2, 0).numpy()
                    image_array = (image_array * 255).astype(np.uint8)
                    image = Image.fromarray(image_array, mode='RGB')
            else:
                # 2D 텐서인 경우
                image_array = tensor.numpy()
                image_array = (image_array * 255).astype(np.uint8)
                image = Image.fromarray(image_array, mode='L')
            
            # 이미지 저장
            image.save(save_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to save image {save_path}: {e}")
            # 실패한 경우 텐서를 직접 저장
            torch.save(tensor, save_path.replace('.png', '.pt')) 
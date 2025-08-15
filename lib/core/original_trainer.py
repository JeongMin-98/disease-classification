import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
from .base_trainer import BaseTrainer

class OriginalTrainer(BaseTrainer):
    """
    원본 이미지를 사용하는 훈련을 위한 trainer
    MedicalImageDataset과 함께 사용
    """
    
    def __init__(self, cfg, model, device, output_dir):
        super().__init__(cfg, model, device, output_dir)
        self.is_binary = len(cfg.DATASET.INCLUDE_CLASSES) == 2
        self.wandb_project_name = None
    def set_wandb_project(self, project_name: str):
        """
        wandb 프로젝트명을 외부에서 설정할 수 있도록 하는 메서드
        """
        self.wandb_project_name = project_name

    def _get_wandb_project(self) -> str:
        """
        wandb 프로젝트명 반환 (외부에서 지정된 경우 우선, 없으면 기본값)
        """
        if self.wandb_project_name is not None:
            return self.wandb_project_name
        return "medical-classification-original"
    
    
    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치 데이터 준비 - MedicalImageDataset용"""
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        return images, labels
    
    def _compute_loss(self, outputs, labels) -> torch.Tensor:
        """손실 계산"""
        if self.is_binary:
            # 이진 분류인 경우 BCEWithLogitsLoss 사용
            # outputs: [batch, 1] 또는 [batch]로 squeeze
            # labels: [batch] 또는 [batch, 1]로 float
            return nn.BCEWithLogitsLoss()(outputs.squeeze(), labels.float())
        else:
            # 다중 분류인 경우 CrossEntropyLoss 사용
            return nn.CrossEntropyLoss()(outputs, labels)
    
    def _compute_accuracy(self, outputs, labels) -> float:
        """정확도 계산"""
        outputs = outputs.detach().cpu().squeeze()
        labels = labels.detach().cpu().squeeze()
        if self.is_binary:
            # 이진 분류: 0/1로 변환
            predicted = (outputs > 0).long()
        else:
            # 다중 분류: argmax
            predicted = outputs.argmax(dim=1)
        correct = (predicted == labels.long()).sum().item()
        return correct 
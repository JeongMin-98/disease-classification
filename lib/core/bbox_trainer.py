import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
from .base_trainer import BaseTrainer

class BBoxTrainer(BaseTrainer):
    """
    BBox crop 이미지를 사용하는 훈련을 위한 trainer
    MedicalImageDatasetWithPatches와 함께 사용 (bbox crop 이미지만 사용)
    """
    
    def __init__(self, cfg, model, device, output_dir, dataset):
        super().__init__(cfg, model, device, output_dir)
        self.dataset = dataset
        self.is_binary = dataset.is_binary if hasattr(dataset, 'is_binary') else len(cfg.DATASET.TARGET_CLASSES) == 2
    
    def _get_wandb_project(self) -> str:
        """WandB 프로젝트명 반환"""
        return "medical-classification-bbox"
    
    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치 데이터 준비 - MedicalImageDatasetWithPatches용 (bbox crop 이미지만 사용)"""
        # batch[0]: merged bbox crop image
        # batch[1]: patches (사용하지 않음)
        # batch[2]: label
        images = batch[0].to(self.device)  # merged bbox crop image
        labels = batch[2].to(self.device)  # label
        return images, labels
    
    def _compute_loss(self, outputs, labels) -> torch.Tensor:
        """손실 계산"""
        if self.is_binary:
            # 이진 분류인 경우 BCEWithLogitsLoss 사용
            return nn.BCEWithLogitsLoss()(outputs.squeeze(), labels.float())
        else:
            # 다중 분류인 경우 CrossEntropyLoss 사용
            return nn.CrossEntropyLoss()(outputs, labels)
    
    def _compute_accuracy(self, outputs, labels) -> float:
        """정확도 계산"""
        if self.is_binary:
            # 이진 분류
            predicted = (outputs > 0).float()
            return predicted.eq(labels).sum().item()
        else:
            # 다중 분류
            _, predicted = outputs.max(1)
            return predicted.eq(labels).sum().item() 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
from .base_trainer import BaseTrainer

class PatchesTrainer(BaseTrainer):
    """
    BBox crop + 관절 키포인트 패치를 사용하는 훈련을 위한 trainer
    MedicalImageDatasetWithPatches와 함께 사용 (bbox crop 이미지 + patches 모두 사용)
    """
    
    def __init__(self, cfg, model, device, output_dir, dataset):
        super().__init__(cfg, model, device, output_dir)
        self.dataset = dataset
        self.is_binary = dataset.is_binary if hasattr(dataset, 'is_binary') else len(cfg.DATASET.TARGET_CLASSES) == 2
        
        # 패치를 처리할 수 있는 모델인지 확인
        self.use_patches = getattr(cfg.DATASET, 'USE_PATCH', True)
        self.concat_patches = getattr(cfg.DATASET, 'CONCAT_PATCH', True)
    
    def _get_wandb_project(self) -> str:
        """WandB 프로젝트명 반환"""
        return "medical-classification-patches"
    
    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치 데이터 준비 - MedicalImageDatasetWithPatches용 (bbox crop 이미지 + patches)"""
        # batch[0]: merged bbox crop image
        # batch[1]: patches tensor
        # batch[2]: label
        images = batch[0].to(self.device)  # merged bbox crop image
        patches = batch[1].to(self.device)  # patches tensor
        labels = batch[2].to(self.device)  # label
        
        if self.use_patches and self.concat_patches:
            # 패치를 이미지와 결합하여 사용
            # 여기서는 간단히 이미지만 사용하도록 구현
            # 실제로는 모델이 patches를 처리할 수 있도록 수정 필요
            return images, labels
        else:
            # 이미지만 사용
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
    
    def get_patch_info(self, batch):
        """패치 정보 반환 (디버깅용)"""
        patches = batch[1]
        return {
            'patch_shape': patches.shape,
            'patch_dtype': patches.dtype,
            'patch_device': patches.device
        } 
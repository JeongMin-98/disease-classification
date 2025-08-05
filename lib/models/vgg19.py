import torch.nn as nn
from torchvision.models import vgg19_bn
from torchvision.models import VGG19_BN_Weights
import logging

class VGG(nn.Module):
    def __init__(self, num_classes=None, pretrained=True, Target_Classes=None, freeze_layers=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Target_Classes가 있으면 그 길이로 num_classes를 자동 설정
        if Target_Classes is not None:
            num_classes = len(Target_Classes)
        if num_classes is None:
            num_classes = 1000
        if num_classes == 2:
            num_classes = 1
        self.model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        
        # 특정 레이어 freeze
        if freeze_layers is not None:
            self.freeze_layers(freeze_layers)
        
        # 마지막 fc 레이어를 num_classes에 맞게 교체
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def freeze_layers(self, layer_range):
        """
        특정 범위의 레이어들을 freeze
        
        Args:
            layer_range: freeze할 레이어 범위 [시작, 끝] (예: [1, 6] -> 1번째부터 6번째까지)
        """
        if len(layer_range) != 2:
            self.logger.warning(f"layer_range should be [start, end], got {layer_range}")
            return
            
        start_layer, end_layer = layer_range
        
        # VGG19의 features 레이어 구조 확인
        features_children = list(self.model.features.children())
        self.logger.info(f"VGG19 features has {len(features_children)} layers")
        
        # 지정된 범위의 레이어들을 freeze
        for idx in range(start_layer - 1, end_layer):  # 0-based indexing으로 변환
            if idx < len(features_children):
                # 해당 레이어의 파라미터들을 freeze
                layer = features_children[idx]
                for param in layer.parameters():
                    param.requires_grad = False
                self.logger.info(f"Layer {idx + 1} ({type(layer).__name__}) frozen")
            else:
                self.logger.warning(f"Layer {idx + 1} does not exist in VGG19 features")
        
        # Freeze된 파라미터 수 확인
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Frozen parameters: {frozen_params:,}")
        self.logger.info(f"Trainable parameters: {total_params - frozen_params:,}")

    def forward(self, x):
        return self.model(x)

import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import logging

class ResNet18(nn.Module):
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
        
        # ResNet18 모델 로드
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.logger.info("ResNet18 loaded with ImageNet pretrained weights")
        else:
            self.model = resnet18(weights=None)
            self.logger.info("ResNet18 loaded without pretrained weights")
        
        # 특정 레이어 freeze
        if freeze_layers is not None:
            self.freeze_layers(freeze_layers)
        
        # 마지막 fc 레이어를 num_classes에 맞게 교체
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.logger.info(f"ResNet18 final layer changed to output {num_classes} classes")

    def freeze_layers(self, layer_range):
        """
        특정 범위의 레이어들을 freeze
        
        Args:
            layer_range: freeze할 레이어 범위 [시작, 끝] (예: [1, 3] -> conv1, bn1, layer1 freeze)
                        ResNet18 구조 순서:
                        1: conv1 (initial conv layer)
                        2: bn1 (initial batch norm)  
                        3: layer1 (first residual block group)
                        4: layer2 (second residual block group)
                        5: layer3 (third residual block group)
                        6: layer4 (fourth residual block group)
        """
        if len(layer_range) != 2:
            self.logger.warning(f"layer_range should be [start, end], got {layer_range}")
            return
            
        start_layer, end_layer = layer_range
        
        # ResNet18의 레이어 구조 (순서대로)
        layer_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        
        self.logger.info(f"ResNet18 layer structure: {[(i+1, name) for i, name in enumerate(layer_names)]}")
        
        # 지정된 범위의 레이어들을 freeze
        frozen_layers = []
        for idx in range(start_layer - 1, end_layer):  # 0-based indexing으로 변환
            if idx < len(layer_names):
                layer_name = layer_names[idx]
                layer = getattr(self.model, layer_name)
                
                # 해당 레이어의 파라미터들을 freeze
                for param in layer.parameters():
                    param.requires_grad = False
                
                frozen_layers.append(f"{idx + 1}:{layer_name}")
                self.logger.info(f"Layer {idx + 1} ({layer_name}) frozen")
            else:
                self.logger.warning(f"Layer {idx + 1} does not exist in ResNet18")
        
        # Freeze된 파라미터 수 확인
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable_params = total_params - frozen_params
        
        self.logger.info(f"Frozen layers: {frozen_layers}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    def forward(self, x):
        return self.model(x) 
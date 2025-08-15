import torch.nn as nn
from torchvision.models import vgg19_bn, vgg19
from torchvision.models import VGG19_BN_Weights, VGG19_Weights
import logging

class VGG(nn.Module):
    """
    VGG19 모델 클래스 (Batch Normalization 옵션 지원)
    
    Args:
        num_classes (int, optional): 출력 클래스 수. Target_Classes가 주어지면 자동 설정됩니다.
        pretrained (bool): 사전 훈련된 가중치 사용 여부. 기본값: True
        Target_Classes (list, optional): 타겟 클래스 리스트. 주어지면 num_classes가 자동 설정됩니다.
        freeze_layers (list, optional): freeze할 레이어 범위 [시작, 끝]. 예: [1, 10]
        batch_norm (bool): Batch Normalization 사용 여부. 기본값: True (VGG19_BN 사용)
    
    Examples:
        >>> # VGG19_BN 사용 (기본값)
        >>> model = VGG(num_classes=10, batch_norm=True)
        
        >>> # 기본 VGG19 사용 (Batch Normalization 없음)
        >>> model = VGG(num_classes=10, batch_norm=False)
        
        >>> # 첫 번째 블록 freeze (VGG19_BN의 경우)
        >>> model = VGG(num_classes=10, freeze_layers=[1, 6], batch_norm=True)
        
        >>> # 첫 번째 블록 freeze (기본 VGG19의 경우)
        >>> model = VGG(num_classes=10, freeze_layers=[1, 4], batch_norm=False)
        
        >>> # Target_Classes로 자동 클래스 수 설정
        >>> model = VGG(Target_Classes=['healthy', 'disease'], batch_norm=True)
    
    Note:
        - VGG19_BN: Conv → BatchNorm → ReLU 패턴 (약 53개 레이어)
        - VGG19: Conv → ReLU 패턴 (약 37개 레이어)
        - freeze_layers 범위는 모델 타입에 따라 다르게 설정해야 합니다.
    """
    def __init__(self, num_classes=None, pretrained=True, Target_Classes=None, freeze_layers=None, batch_norm=False):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.batch_norm = batch_norm
        
        # Target_Classes가 있으면 그 길이로 num_classes를 자동 설정
        if Target_Classes is not None:
            num_classes = len(Target_Classes)
        if num_classes is None:
            num_classes = 1000
        if num_classes == 2:
            num_classes = 1
            
        # batch_norm 여부에 따라 다른 모델 사용
        if batch_norm:
            self.model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT if pretrained else None)
            self.logger.info("Using VGG19 with Batch Normalization")
        else:
            self.model = vgg19(weights=VGG19_Weights.DEFAULT if pretrained else None)
            self.logger.info("Using VGG19 without Batch Normalization")
        
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
        
        Note:
            VGG19 구조에 따른 레이어 수 차이:
            - VGG19 (기본): Conv → ReLU 패턴 (약 37개 레이어)
            - VGG19_BN: Conv → BatchNorm → ReLU 패턴 (약 53개 레이어)
            
            레이어 인덱스는 features의 children() 순서를 따릅니다.
        """
        if len(layer_range) != 2:
            self.logger.warning(f"layer_range should be [start, end], got {layer_range}")
            return
            
        start_layer, end_layer = layer_range
        
        # VGG19/VGG19_BN의 features 레이어 구조 확인
        features_children = list(self.model.features.children())
        model_type = "VGG19_BN" if self.batch_norm else "VGG19"
        self.logger.info(f"{model_type} features has {len(features_children)} layers")
        
        # 처음 몇 개 레이어의 구조를 로깅하여 사용자가 이해할 수 있도록 함
        self.logger.info("First 10 layers structure:")
        for i, layer in enumerate(features_children[:10]):
            self.logger.info(f"  {i:2d}: {type(layer).__name__}")
        if len(features_children) > 10:
            self.logger.info(f"  ... and {len(features_children) - 10} more layers")
        
        # 지정된 범위의 레이어들을 freeze
        frozen_layer_types = []
        for idx in range(start_layer - 1, end_layer):  # 0-based indexing으로 변환
            if idx < len(features_children):
                # 해당 레이어의 파라미터들을 freeze
                layer = features_children[idx]
                for param in layer.parameters():
                    param.requires_grad = False
                
                layer_type = type(layer).__name__
                frozen_layer_types.append(layer_type)
                self.logger.info(f"Layer {idx + 1:2d} ({layer_type}) frozen")
            else:
                self.logger.warning(f"Layer {idx + 1} does not exist in {model_type} features")
        
        # 어떤 타입의 레이어들이 freeze되었는지 요약
        from collections import Counter
        layer_counts = Counter(frozen_layer_types)
        self.logger.info(f"Frozen layer types: {dict(layer_counts)}")
        
        # Freeze된 파라미터 수 확인
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Frozen parameters: {frozen_params:,}")
        self.logger.info(f"Trainable parameters: {total_params - frozen_params:,}")
        self.logger.info(f"Frozen ratio: {frozen_params/total_params*100:.1f}%")

    def forward(self, x):
        return self.model(x)

class VGG19_BN(VGG):
    """
    VGG19 with Batch Normalization
    
    VGG 클래스를 상속받아 batch_norm=True로 고정한 클래스입니다.
    
    Args:
        num_classes (int, optional): 출력 클래스 수. Target_Classes가 주어지면 자동 설정됩니다.
        pretrained (bool): 사전 훈련된 가중치 사용 여부. 기본값: True
        Target_Classes (list, optional): 타겟 클래스 리스트. 주어지면 num_classes가 자동 설정됩니다.
        freeze_layers (list, optional): freeze할 레이어 범위 [시작, 끝]. 예: [1, 10]
    
    Examples:
        >>> # VGG19_BN 사용
        >>> model = VGG19_BN(num_classes=10)
        
        >>> # 첫 번째 블록 freeze
        >>> model = VGG19_BN(num_classes=10, freeze_layers=[1, 6])
        
        >>> # Target_Classes로 자동 클래스 수 설정
        >>> model = VGG19_BN(Target_Classes=['healthy', 'disease'])
    """
    def __init__(self, num_classes=None, pretrained=True, Target_Classes=None, freeze_layers=None):
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            Target_Classes=Target_Classes,
            freeze_layers=freeze_layers,
            batch_norm=True  # VGG19_BN은 항상 batch_norm=True
        )

# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by Jeongmin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------
from .vgg19 import VGG, VGG19_BN
from .resnet18 import ResNet18

__factory = {
    'VGG19' : VGG,           # VGG19 without Batch Normalization
    'VGG19_BN' : VGG19_BN,   # VGG19 with Batch Normalization
    'VGG16' : VGG,           # VGG16 without Batch Normalization (using VGG class)
    'VGG16_BN' : VGG19_BN,   # VGG16 with Batch Normalization (using VGG19_BN class)
    'ResNet18' : ResNet18,
}

def names():
    return sorted(__factory.keys())

def create(name, cfg=None, *args, **kwargs):
    """
    모델을 생성합니다.
    
    Args:
        name (str): 모델 이름 (VGG19, VGG19_BN, VGG16, VGG16_BN, ResNet18)
        cfg: 설정 객체 (선택사항)
        *args, **kwargs: 추가 인자들
    
    Returns:
        생성된 모델 인스턴스
    
    Examples:
        >>> # Config 없이 직접 생성
        >>> model = create('VGG19_BN', num_classes=10)
        
        >>> # Config를 사용하여 생성
        >>> model = create('VGG19_BN', cfg)
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    
    # Config가 제공된 경우 설정에서 파라미터 추출
    if cfg is not None:
        # VGG 모델의 경우
        if name.startswith('VGG'):
            model_kwargs = {
                'num_classes': getattr(cfg.MODEL, 'NUM_CLASSES', None),
                'pretrained': getattr(cfg.MODEL, 'PRETRAINED', True),
                'Target_Classes': getattr(cfg.DATASET, 'TARGET_CLASSES', None),
                'freeze_layers': getattr(cfg.MODEL, 'FREEZE_LAYERS', None),
            }
            
            # 사용자가 제공한 kwargs로 덮어쓰기
            model_kwargs.update(kwargs)
            
            return __factory[name](**model_kwargs)
        
        # ResNet의 경우
        elif name.startswith('ResNet'):
            model_kwargs = {
                'num_classes': getattr(cfg.MODEL, 'NUM_CLASSES', None),
                'pretrained': getattr(cfg.MODEL, 'PRETRAINED', True),
                'Target_Classes': getattr(cfg.DATASET, 'TARGET_CLASSES', None),
                'freeze_layers': getattr(cfg.MODEL, 'FREEZE_LAYERS', None),
            }
            
            model_kwargs.update(kwargs)
            return __factory[name](**model_kwargs)
    
    # Config 없이 직접 생성
    return __factory[name](*args, **kwargs)
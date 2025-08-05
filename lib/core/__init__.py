# --------------------------------------------------------
# Core trainer modules for medical image classification
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# --------------------------------------------------------

from .base_trainer import BaseTrainer
from .original_trainer import OriginalTrainer
from .bbox_trainer import BBoxTrainer
from .patches_trainer import PatchesTrainer

__all__ = [
    'BaseTrainer',
    'OriginalTrainer', 
    'BBoxTrainer',
    'PatchesTrainer'
] 
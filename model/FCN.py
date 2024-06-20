# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

import torch
import torch.nn as nn
from network import tools

class FCN(nn.Module):
    def __init__(self, cfg):
       super(FCN, self).__init__()

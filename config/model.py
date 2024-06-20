# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# Your model related params
# examples
FCN = CN()
FCN.IN.CHANNELS = 784
FCN.HIDDEN.CHANNELS = [128, 64, 10]
FCN.HIDDEN.ACTIVATION = 'ReLU'
FCN.HIDDEN.DROPOUT = 0.25
FCN.OUTPUT.CHANNELS = 2
FCN.OUTPUT.ACTIVATION = 'logSoftMax'

MODELS = {
    'FCN': FCN,
}

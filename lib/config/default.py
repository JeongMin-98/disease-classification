# --------------------------------------------------------
# 
# Written by Jeongmin Kim(jm.kim@dankook.ac.kr)
# 
# ----------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN
from .model_cfg import MODEL_EXTRAS
# # If you want to save yaml file, use below code 
# from model_cfg import MODEL_EXTRAS

_C = CN()

_C.DATA_DIR = ''
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = 0
_C.WORKERS = 4
_C.PHASE = 'train'
_C.DEVICE = "GPU"
_C.PRINT_FREQ = 1
_C.PRINT_SAMPLE_FREQ = 5


# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK (Feature Extractor)
_C.MODEL = CN()
_C.MODEL.FREEZE = CN(new_allowed=True)
_C.MODEL.NAME = 'feature_extractor'
_C.MODEL.EXTRA = MODEL_EXTRAS[_C.MODEL.NAME]
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = True
_C.MODEL.FREEZE.BACKBONE = False
_C.MODEL.FREEZE.PROJECTION = False
_C.MODEL.FREEZE.CLASSIFIER = False
_C.MODEL.FREEZE.GLOBAL_BACKBONE = False  # 글로벌 백본 freeze 옵션
_C.MODEL.FREEZE.LOCAL_BACKBONE = False   # 로컬 백본 freeze 옵션
_C.MODEL.FREEZE_LAYERS = None  # 특정 레이어 freeze (예: [1, 2, 3, 4, 5, 6])

# Params for Decoder (GPT2)
_C.DECODER = CN()
_C.DECODER.NAME = 'GPT2'
_C.DECODER.EXTRA = MODEL_EXTRAS[_C.DECODER.NAME]
_C.DECODER.INIT_WEIGHTS = True
_C.DECODER.PRETRAINED = True
_C.DECODER.USE_LORA = True
_C.DECODER.FLAMINGO = CN()
_C.DECODER.FLAMINGO.USE_FLAMINGO = False
_C.DECODER.FLAMINGO.ADAPTER_LAYERS= [4, 8, 10]

# if you want to add new params for NETWORK, Init new Params below!

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.JSON = 'data/merge/output.json'
_C.DATASET.BBOX_JSON = 'data/json/merge/final_merge_output_v2_summary.json'
_C.DATASET.TYPE = 'foot'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.INCLUDE_CLASSES = ['oa', 'normal']
_C.DATASET.AUGMENT = True
_C.DATASET.AUGMENT_RATIO = 1
_C.DATASET.BALANCE = True
_C.DATASET.BASIC_TRANSFORM = True
_C.DATASET.BBOX_CROP_FLAG = False
_C.DATASET.MEAN = [0.1147, 0.1147, 0.1147]
_C.DATASET.STD = [0.2194, 0.2194, 0.2194]
_C.DATASET.IMAGE_SIZE = [512, 512]  # 입력 이미지 해상도 [H, W] 또는 정수 (정수인 경우 [H, H]로 변환)
_C.DATASET.INPUT_CHANNELS = 3  # 입력 채널 수 (1: grayscale, 3: RGB)
# _C.DATASET.SPLIT_RATIO = {'train': 0.7, 'validation': 0.15, 'test': 0.15}
_C.DATASET.TARGET_CLASSES = ['oa', 'normal']
_C.DATASET.MULTI_RUN = False
_C.DATASET.MULTI_RUN_SEEDING = False

_C.DATASET.USE_PKL = False
_C.DATASET.USE_RAW = False
_C.DATASET.USE_PATCH = False
_C.DATASET.CONCAT_PATCH = False
_C.DATASET.REPORT = False
_C.DATASET.TRANSFORM_TYPE = 'basic'
_C.DATASET.TARGET_COUNT_PER_CLASS = 224
_C.DATASET.USE_CLAHE = False
_C.DATASET.CLAHE_CLIP_LIMIT = 2.0
_C.DATASET.CLAHE_TILE_GRID_SIZE = [8, 8]
_C.DATASET.CLAHE_NORMALIZE = False
_C.DATASET.USE_STRATIFIED_SPLIT = True  # val/test set에서도 균등한 클래스 분포 보장
_C.DATASET.USE_BALANCED_SPLIT = True  # val/test set에서도 균등한 클래스 분포 보장
_C.DATASET.VERIFY_SPLIT = True  # val/test set에서도 균등한 클래스 분포 보장

# Adaptive Histogram Equalization 설정
_C.DATASET.USE_ADAPTIVE_HISTOGRAM = False
_C.DATASET.ADAPTIVE_THRESHOLD_LOWER = 50
_C.DATASET.ADAPTIVE_THRESHOLD_UPPER = 70
_C.DATASET.ADAPTIVE_ALPHA = 0.01
_C.DATASET.ADAPTIVE_THRESHOLD = True

# 배경 제거 설정
_C.DATASET.USE_BACKGROUND_REMOVAL = False  # 배경 제거 사용 여부
_C.DATASET.BG_REMOVAL = CN()
_C.DATASET.BG_REMOVAL.METHOD = 'fixed'  # "fixed" | "otsu" | "percentile"
_C.DATASET.BG_REMOVAL.FIXED_THRESH = 10  # fixed일 때 임계값(0~255 스케일 기준)
_C.DATASET.BG_REMOVAL.PERCENTILE = 2.0  # percentile일 때 하위 % (0~100)
_C.DATASET.BG_REMOVAL.MORPH_KERNEL = 5  # 형태학 커널 크기(홀수)
_C.DATASET.BG_REMOVAL.KEEP_LARGEST_ONLY = True  # 가장 큰 연결 성분만 유지
_C.DATASET.BG_REMOVAL.TIGHT_CROP = False  # 마스크 bbox로 크롭
_C.DATASET.BG_REMOVAL.FILL_VALUE = 0  # 배경 채울 값 (보통 0)
_C.DATASET.BG_REMOVAL.MIN_OBJECT_AREA = 5000  # 너무 작은 성분 제거(픽셀)
_C.DATASET.BG_REMOVAL.NORMALIZE_TO_UINT8 = True  # 저장 전 8-bit로 정규화(시각화/학습 용이)
_C.DATASET.BG_REMOVAL.SAVE_ORIGINAL = True  # 원본 이미지도 함께 저장 (비교용)
_C.DATASET.BG_REMOVAL.OUTPUT_SUFFIX = '_bg_removed'  # 배경 제거된 이미지 파일명 접미사

# HSV 기반 배경제거 이미지 사용 설정
_C.DATASET.USE_BACKGROUND_REMOVED = False  # 배경제거 사용 여부
_C.DATASET.BACKGROUND_REMOVED_TYPE = 'folder'  # 'folder': 폴더에서 로드, 'hsv': 실시간 HSV 배경제거
_C.DATASET.BACKGROUND_REMOVED_DIR = ''  # 배경제거된 이미지 폴더 경로 (BACKGROUND_REMOVED_TYPE='folder'일 때만)

# HSV 실시간 배경제거 설정 (BACKGROUND_REMOVED_TYPE='hsv'일 때)
_C.DATASET.HSV_BG_REMOVAL = CN()
_C.DATASET.HSV_BG_REMOVAL.V_THRESHOLD = 50  # HSV V 채널 임계값
_C.DATASET.HSV_BG_REMOVAL.PROTECT_SKIN = True  # 피부 영역 보호
_C.DATASET.HSV_BG_REMOVAL.PROTECT_BONE = True  # 뼈/관절 영역 보호
_C.DATASET.HSV_BG_REMOVAL.MORPH_KERNEL = 5  # 형태학 커널 크기
_C.DATASET.HSV_BG_REMOVAL.KEEP_LARGEST_ONLY = False  # 가장 큰 연결 성분만 유지
_C.DATASET.HSV_BG_REMOVAL.TIGHT_CROP = False  # 타이트 크롭
_C.DATASET.HSV_BG_REMOVAL.MIN_OBJECT_AREA = 5000  # 최소 객체 면적

_C.KFOLD = CN()
_C.KFOLD.USE_KFOLD = True
_C.KFOLD.KFOLD_SIZE = 7
_C.KFOLD.P = 0
_C.KFOLD.TEST_SET_RATIO = 0.15
_C.KFOLD.SAVE_FINAL_MODELS = False
_C.KFOLD.SAVE_BEST_MODELS = True
_C.KFOLD.KEEP_ONLY_BEST_FOLD = False
_C.KFOLD.KEEP_BEST_AND_WORST = True


# Related Train Parameter
_C.TRAIN = CN()
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.USE_GRAD_CLIP = False
_C.TRAIN.GRADIENT_ACCUMULATION_STEPS = 2
_C.TRAIN.USE_AMP = True  # Mixed Precision Training 활성화
_C.TRAIN.USE_BALANCED_SAMPLING = True  # 균등한 클래스 분포를 위한 Balanced Sampling 활성화
_C.TRAIN.SAMPLING_TYPE = 'balanced'  # 'balanced' 또는 'stratified'
_C.TRAIN.LOG_BATCH_DISTRIBUTION = False  # 배치 분포 로깅 (디버깅 시에만 True)
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.DECAY = 5e-4
_C.TRAIN.LOSS = 'BCELoss'

_C.TRAIN.SCHEDULER = 'ReduceLROnPlateau'
_C.TRAIN.MODE = 'min'
_C.TRAIN.factor = 0.5
_C.TRAIN.PATIENCE = 5
_C.TRAIN.MIN_EPOCHS = 25
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# KFold Setting
_C.KFOLD = CN()
_C.KFOLD.USE_KFOLD = True
_C.KFOLD.KFOLD_SIZE = 5
_C.KFOLD.P = 0
_C.KFOLD.TEST_SET_RATIO = 0.15

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.TEST_SET_RATIO = 0.15
_C.TEST.SAVE_MISCLASSIFIED_CASES = True

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.GRAPH_DEBUG = True


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)

    # if args.modelDir:
    #     cfg.OUTPUT_DIR = args.modelDir
    #
    # if args.logDir:
    #     cfg.LOG_DIR = args.logDir
    #
    # if args.dataDir:
    #     cfg.DATA_DIR = args.dataDir

    # cfg.DATASET.ROOT = os.path.join(
    #     cfg.DATA_DIR, cfg.DATASET.ROOT
    # )
    #
    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )

    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

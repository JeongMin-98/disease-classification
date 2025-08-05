import cv2
import numpy as np
from PIL import Image

def create_clahe_transform(clip_limit=2.0, tile_grid_size=(8, 8)):
    def clahe_fn(img):
        # PIL -> numpy
        img_np = np.array(img)
        if img_np.ndim == 2:  # grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            img_np = clahe.apply(img_np)
        elif img_np.ndim == 3 and img_np.shape[2] == 3:  # color
            img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(img_np)
    return clahe_fn

def create_adaptive_transform():
    # 예시: CLAHE + 기타 전처리 조합
    from torchvision import transforms
    return transforms.Compose([
        create_clahe_transform(),
        transforms.ToTensor()
    ])

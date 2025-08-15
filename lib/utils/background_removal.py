# --------------------------------------------------------
# 
# Written by Jeongmin Kim(jm.kim@dankook.ac.kr)
# 
# ----------------------------------------------------

"""
배경 제거 유틸리티 모듈

X-ray 이미지에서 배경을 제거하여 손/발 영역만 추출하는 기능을 제공합니다.
config를 통해 다양한 배경 제거 방법과 파라미터를 설정할 수 있습니다.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal, Optional, Union
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

Method = Literal["fixed", "otsu", "percentile"]

@dataclass
class BgRemovalConfig:
    """배경 제거 설정을 위한 데이터클래스"""
    method: Method = "fixed"           # "fixed" | "otsu" | "percentile"
    fixed_thresh: int = 10             # fixed일 때 임계값(0~255 스케일 기준)
    percentile: float = 2.0            # percentile일 때 하위 % (0~100)
    morph_kernel: int = 5              # 형태학 커널 크기(홀수)
    keep_largest_only: bool = True     # 가장 큰 연결 성분만 유지
    tight_crop: bool = False           # 마스크 bbox로 크롭
    fill_value: int = 0                # 배경 채울 값 (보통 0)
    min_object_area: int = 5000        # 너무 작은 성분 제거(픽셀)
    normalize_to_uint8: bool = True    # 저장 전 8-bit로 정규화(시각화/학습 용이)
    save_original: bool = True         # 원본 이미지도 함께 저장 (비교용)
    output_suffix: str = '_bg_removed' # 배경 제거된 이미지 파일명 접미사

def to_uint8(gray: np.ndarray) -> np.ndarray:
    """16-bit 등 다양한 범위를 안전하게 0~255로 스케일."""
    g = gray.astype(np.float32)
    minv, maxv = np.min(g), np.max(g)
    if maxv - minv < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    g = (g - minv) / (maxv - minv) * 255.0
    return g.astype(np.uint8)

def load_image_any(path: str) -> np.ndarray:
    """
    PNG/JPG/TIF 등 일반 포맷을 16-bit까지 안전히 로드.
    (DICOM은 pydicom 등 별도 처리 권장)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    # 채널 처리
    if img.ndim == 3:  # BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # 16-bit -> 유지 (마스크 계산은 uint8로 수행)
    return gray

def compute_mask(gray: np.ndarray, cfg: BgRemovalConfig) -> np.ndarray:
    """
    입력 gray는 8-bit 또는 16-bit일 수 있음.
    마스크는 0/255 (uint8)로 반환.
    """
    # 마스크 산출은 8-bit 공간에서 수행
    gray8 = gray if gray.dtype == np.uint8 else to_uint8(gray)

    if cfg.method == "fixed":
        _, mask = cv2.threshold(gray8, cfg.fixed_thresh, 255, cv2.THRESH_BINARY)
    elif cfg.method == "otsu":
        # Otsu는 bimodal에 강함. 배경과 조직이 잘 갈라질 때 유리
        _, mask = cv2.threshold(gray8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif cfg.method == "percentile":
        t = np.percentile(gray8, cfg.percentile)
        _, mask = cv2.threshold(gray8, int(t), 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    # 형태학 보정
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    # 연결 성분 분석: 가장 큰 성분만 유지 (배경 포함 여부 주의)
    if cfg.keep_largest_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # 라벨 0은 배경. 가장 큰 전경 성분 찾기
        largest_label, largest_area = -1, -1
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_label = lbl
        kept = np.zeros_like(mask)
        if largest_label > 0 and largest_area >= cfg.min_object_area:
            kept[labels == largest_label] = 255
        mask = kept

    return mask

def apply_mask_and_optionally_crop(
    orig: np.ndarray, mask: np.ndarray, cfg: BgRemovalConfig
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """마스크 적용 후 배경을 fill_value로 채우고, 선택적으로 타이트 크롭."""
    # 원본이 3채널일 수도 있으니 처리 (여기선 주로 단일 채널 가정)
    if orig.ndim == 2:
        out = orig.copy()
        out[mask == 0] = cfg.fill_value
    else:
        out = orig.copy()
        out[mask == 0, :] = cfg.fill_value

    bbox = None
    if cfg.tight_crop:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            bbox = (x1, y1, x2, y2)
            out = out[y1:y2+1, x1:x2+1] if out.ndim == 2 else out[y1:y2+1, x1:x2+1, :]
    return out, bbox

def process_image(
    in_path: str,
    out_path: str,
    cfg: BgRemovalConfig
) -> Tuple[bool, Optional[str]]:
    """단일 이미지에 대해 배경 제거를 수행합니다."""
    try:
        gray = load_image_any(in_path)
        mask = compute_mask(gray, cfg)

        # 저장 용 시각화를 위해 8-bit로 정규화 옵션
        save_img = gray
        if cfg.normalize_to_uint8 and gray.dtype != np.uint8:
            save_img = to_uint8(gray)

        result, _ = apply_mask_and_optionally_crop(save_img, mask, cfg)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ok = cv2.imwrite(out_path, result)
        if not ok:
            return False, f"Failed to write: {out_path}"
        return True, None
    except Exception as e:
        return False, str(e)

def batch_process(
    in_dir: str,
    out_dir: str,
    cfg: BgRemovalConfig,
    exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
):
    """폴더 내 모든 이미지에 대해 배경 제거를 일괄 처리합니다."""
    os.makedirs(out_dir, exist_ok=True)
    files = [
        f for f in os.listdir(in_dir)
        if f.lower().endswith(exts)
    ]
    
    success_count = 0
    error_count = 0
    
    for fname in tqdm(files, desc="Background removal"):
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        
        # 원본 이미지도 저장 (비교용)
        if cfg.save_original:
            original_out_path = os.path.join(out_dir, f"original_{fname}")
            try:
                img = cv2.imread(in_path)
                if img is not None:
                    cv2.imwrite(original_out_path, img)
            except Exception as e:
                logger.warning(f"Failed to save original {fname}: {e}")
        
        ok, err = process_image(in_path, out_path, cfg)
        if ok:
            success_count += 1
        else:
            error_count += 1
            logger.error(f"[ERROR] {fname}: {err}")
    
    logger.info(f"Background removal completed: {success_count} success, {error_count} errors")

def config_to_bg_removal_config(cfg) -> BgRemovalConfig:
    """config 객체에서 BgRemovalConfig를 생성합니다."""
    bg_cfg = cfg.DATASET.BG_REMOVAL
    return BgRemovalConfig(
        method=bg_cfg.METHOD,
        fixed_thresh=bg_cfg.FIXED_THRESH,
        percentile=bg_cfg.PERCENTILE,
        morph_kernel=bg_cfg.MORPH_KERNEL,
        keep_largest_only=bg_cfg.KEEP_LARGEST_ONLY,
        tight_crop=bg_cfg.TIGHT_CROP,
        fill_value=bg_cfg.FILL_VALUE,
        min_object_area=bg_cfg.MIN_OBJECT_AREA,
        normalize_to_uint8=bg_cfg.NORMALIZE_TO_UINT8,
        save_original=bg_cfg.SAVE_ORIGINAL,
        output_suffix=bg_cfg.OUTPUT_SUFFIX
    )

def process_dataset_with_background_removal(
    input_dir: str,
    output_dir: str,
    cfg,
    file_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
):
    """
    config를 사용하여 데이터셋에 배경 제거를 적용합니다.
    
    Args:
        input_dir: 입력 이미지 폴더
        output_dir: 출력 이미지 폴더
        cfg: 설정 객체
        file_extensions: 처리할 파일 확장자들
    """
    if not cfg.DATASET.USE_BACKGROUND_REMOVAL:
        logger.info("Background removal is disabled in config. Skipping...")
        return
    
    logger.info("Starting background removal process...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    bg_config = config_to_bg_removal_config(cfg)
    logger.info(f"Background removal method: {bg_config.method}")
    
    batch_process(input_dir, output_dir, bg_config, file_extensions)
    logger.info("Background removal process completed!")

if __name__ == "__main__":
    # 예시 사용법
    cfg = BgRemovalConfig(
        method="fixed",        # "fixed" | "otsu" | "percentile"
        fixed_thresh=10,       # 김현일 선배님 제안 기준
        morph_kernel=5,
        keep_largest_only=True,
        tight_crop=False,      # ROI 대비실험 시 True로 두고 저장
        fill_value=0,
        normalize_to_uint8=True
    )
    
    # 단일 이미지
    # ok, err = process_image("input/sample.png", "output/sample_bg_removed.png", cfg)
    # if not ok: print("Error:", err)

    # 일괄 처리
    # batch_process("input_folder", "output_folder", cfg)
    pass 
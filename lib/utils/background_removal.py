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

Method = Literal["fixed", "otsu", "percentile", "hsv_value"]

@dataclass
class BgRemovalConfig:
    """배경 제거 설정을 위한 데이터클래스"""
    method: Method = "fixed"           # "fixed" | "otsu" | "percentile" | "hsv_value"
    fixed_thresh: int = 10             # fixed일 때 임계값(0~255 스케일 기준)
    percentile: float = 2.0            # percentile일 때 하위 % (0~100)
    hsv_value_thresh: int = 45         # HSV V 채널 임계값 (0~255)
    protect_skin: bool = True          # 피부 영역 보호 여부
    protect_bone: bool = True          # 뼈/관절 영역 보호 여부
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
    # 채널 처리는 이제 compute_mask에서 처리
    return img

def compute_mask_hsv_value(img: np.ndarray, thresh: int, protect_skin: bool = True, protect_bone: bool = True) -> np.ndarray:
    """
    HSV 색상 공간의 V(Value) 채널 기반으로 마스크 생성.
    V 값이 thresh 이하인 픽셀들을 배경으로 간주하되, 피부/관절 영역은 보호.
    """
    # 입력이 그레이스케일인 경우 RGB로 변환
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy()
    
    # BGR to HSV 변환
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # HSV 각 채널 추출
    h_channel = hsv[:, :, 0]  # Hue
    s_channel = hsv[:, :, 1]  # Saturation
    v_channel = hsv[:, :, 2]  # Value
    
    # 기본 V 채널 기반 마스크
    basic_mask = np.where(v_channel > thresh, 255, 0).astype(np.uint8)
    
    # 보호 마스크 초기화
    protection_mask = np.zeros_like(basic_mask)
    
    # 피부 톤 보호 마스크 생성 (옵션에 따라)
    if protect_skin:
        skin_mask = create_skin_protection_mask(hsv)
        protection_mask = cv2.bitwise_or(protection_mask, skin_mask)
    
    # 관절/뼈 구조 보호 마스크 생성 (옵션에 따라)
    if protect_bone:
        bone_mask = create_bone_protection_mask(hsv, v_channel)
        protection_mask = cv2.bitwise_or(protection_mask, bone_mask)
    
    # 기본 마스크에 보호 영역 추가
    # 보호 영역은 V 값에 관계없이 전경으로 유지
    final_mask = cv2.bitwise_or(basic_mask, protection_mask)
    
    return final_mask

def create_skin_protection_mask(hsv: np.ndarray) -> np.ndarray:
    """
    피부 톤 영역을 보호하는 마스크를 생성합니다.
    다양한 피부 톤 범위를 고려합니다.
    """
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    # 피부 톤 범위 정의 (HSV)
    # 범위 1: 밝은 피부 톤
    skin_lower1 = np.array([0, 20, 70])    # 주황-빨강 계열, 낮은 채도, 중간 밝기
    skin_upper1 = np.array([20, 255, 255])
    
    # 범위 2: 중간 피부 톤
    skin_lower2 = np.array([5, 30, 50])    # 주황 계열, 중간 채도
    skin_upper2 = np.array([25, 200, 200])
    
    # 범위 3: 어두운 피부 톤
    skin_lower3 = np.array([8, 40, 30])    # 갈색 계열
    skin_upper3 = np.array([30, 180, 150])
    
    # 각 범위별 마스크 생성
    skin_mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
    skin_mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
    skin_mask3 = cv2.inRange(hsv, skin_lower3, skin_upper3)
    
    # 모든 피부 톤 마스크 통합
    skin_mask = cv2.bitwise_or(cv2.bitwise_or(skin_mask1, skin_mask2), skin_mask3)
    
    # 추가적으로 S(채도)와 V(명도) 기반 피부 보호
    # 피부는 보통 중간 정도의 채도와 밝기를 가짐
    saturation_mask = np.where((s_channel >= 20) & (s_channel <= 150), 255, 0).astype(np.uint8)
    brightness_mask = np.where((v_channel >= 40) & (v_channel <= 200), 255, 0).astype(np.uint8)
    
    # 채도와 밝기 조건을 만족하는 영역 중에서 피부색에 가까운 영역
    sv_combined = cv2.bitwise_and(saturation_mask, brightness_mask)
    
    # 최종 피부 마스크
    final_skin_mask = cv2.bitwise_or(skin_mask, sv_combined)
    
    # 형태학적 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_skin_mask = cv2.morphologyEx(final_skin_mask, cv2.MORPH_OPEN, kernel)
    final_skin_mask = cv2.morphologyEx(final_skin_mask, cv2.MORPH_CLOSE, kernel)
    
    return final_skin_mask

def create_bone_protection_mask(hsv: np.ndarray, v_channel: np.ndarray) -> np.ndarray:
    """
    관절/뼈 구조(높은 밝기값)를 보호하는 마스크를 생성합니다.
    """
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    # 뼈/관절 영역 특성: 높은 밝기값, 낮은 채도
    # X-ray에서 뼈는 밝게 나타남
    high_brightness = np.where(v_channel >= 180, 255, 0).astype(np.uint8)
    low_saturation = np.where(s_channel <= 50, 255, 0).astype(np.uint8)
    
    # 밝고 채도가 낮은 영역 (뼈/관절 가능성 높음)
    bone_mask = cv2.bitwise_and(high_brightness, low_saturation)
    
    # 중간 밝기지만 구조적으로 중요한 영역도 보호
    # 관절 부분은 중간 밝기일 수 있음
    medium_brightness = np.where((v_channel >= 100) & (v_channel < 180), 255, 0).astype(np.uint8)
    very_low_saturation = np.where(s_channel <= 30, 255, 0).astype(np.uint8)
    
    # 중간 밝기 + 매우 낮은 채도 영역
    joint_mask = cv2.bitwise_and(medium_brightness, very_low_saturation)
    
    # 뼈와 관절 마스크 통합
    combined_bone_mask = cv2.bitwise_or(bone_mask, joint_mask)
    
    # 형태학적 연산으로 연결성 향상
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_bone_mask = cv2.morphologyEx(combined_bone_mask, cv2.MORPH_CLOSE, kernel)
    
    return combined_bone_mask

def compute_mask(gray_or_color: np.ndarray, cfg: BgRemovalConfig) -> np.ndarray:
    """
    입력은 grayscale 또는 color 이미지일 수 있음.
    마스크는 0/255 (uint8)로 반환.
    """
    if cfg.method == "hsv_value":
        # HSV 기반 처리는 컬러 정보가 필요하므로 별도 처리
        mask = compute_mask_hsv_value(gray_or_color, cfg.hsv_value_thresh, cfg.protect_skin, cfg.protect_bone)
    else:
        # 기존 방식들은 grayscale 기준
        if gray_or_color.ndim == 3:  # BGR
            gray = cv2.cvtColor(gray_or_color, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_or_color
            
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

def apply_hsv_background_removal(img: np.ndarray, thresh: int, fill_value: int = 0, protect_skin: bool = True, protect_bone: bool = True) -> np.ndarray:
    """
    HSV 색상 공간에서 V 채널 기반으로 배경을 제거하고 RGB로 다시 변환.
    피부/관절 영역은 보호합니다.
    
    Args:
        img: 입력 이미지 (BGR 또는 grayscale)
        thresh: HSV V 채널 임계값
        fill_value: 배경을 채울 값
    
    Returns:
        배경이 제거된 RGB 이미지
    """
    # 입력이 그레이스케일인 경우 BGR로 변환
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy()
    
    # BGR to HSV 변환
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 개선된 마스크 생성 (피부/관절 보호 포함)
    foreground_mask = compute_mask_hsv_value(img_bgr, thresh, protect_skin, protect_bone)
    
    # 전경 마스크를 배경 마스크로 변환
    background_mask = (foreground_mask == 0)
    
    # 배경 픽셀을 fill_value로 설정
    result = img_bgr.copy()
    result[background_mask] = fill_value
    
    # BGR to RGB 변환
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result_rgb

def process_image(
    in_path: str,
    out_path: str,
    cfg: BgRemovalConfig
) -> Tuple[bool, Optional[str]]:
    """단일 이미지에 대해 배경 제거를 수행합니다."""
    try:
        img = load_image_any(in_path)
        
        if cfg.method == "hsv_value":
            # HSV 기반 처리: 바로 RGB 결과 생성
            result_rgb = apply_hsv_background_removal(
                img, cfg.hsv_value_thresh, cfg.fill_value, cfg.protect_skin, cfg.protect_bone
            )
            result = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)  # 저장을 위해 BGR로 변환
        else:
            # 기존 방식: 마스크 기반 처리
            mask = compute_mask(img, cfg)
            
            # grayscale 처리
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

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
        hsv_value_thresh=getattr(bg_cfg, 'HSV_VALUE_THRESH', 15),  # 기본값 15
        protect_skin=getattr(bg_cfg, 'PROTECT_SKIN', True),  # 기본값 True
        protect_bone=getattr(bg_cfg, 'PROTECT_BONE', True),  # 기본값 True
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
    # HSV 기반 배경 제거 (피부/관절 보호 포함)
    cfg_hsv_protected = BgRemovalConfig(
        method="hsv_value",        # HSV V 채널 기반 배경 제거
        hsv_value_thresh=15,       # V 값 15 이하를 배경으로 간주
        protect_skin=True,         # 피부 영역 보호
        protect_bone=True,         # 뼈/관절 영역 보호
        morph_kernel=5,
        keep_largest_only=True,
        tight_crop=False,
        fill_value=0,
        normalize_to_uint8=True
    )
    
    # 기본 HSV 방법 (보호 기능 없음)
    cfg_hsv_basic = BgRemovalConfig(
        method="hsv_value",        # HSV V 채널 기반 배경 제거
        hsv_value_thresh=15,       # V 값 15 이하를 배경으로 간주
        protect_skin=False,        # 피부 영역 보호 비활성화
        protect_bone=False,        # 뼈/관절 영역 보호 비활성화
        morph_kernel=5,
        keep_largest_only=True,
        tight_crop=False,
        fill_value=0,
        normalize_to_uint8=True
    )
    
    # 단일 이미지
    # ok, err = process_image("input/sample.png", "output/sample_bg_removed.png", cfg)
    # if not ok: print("Error:", err)

    # 일괄 처리
    # batch_process("input_folder", "output_folder", cfg)
    pass 
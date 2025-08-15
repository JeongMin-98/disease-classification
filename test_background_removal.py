#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
배경 제거 기능 테스트 스크립트

이 스크립트는 config를 통해 배경 제거 기능을 테스트할 수 있습니다.
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lib.config.default import _C
from lib.utils.background_removal import (
    process_dataset_with_background_removal,
    BgRemovalConfig,
    process_image
)

def test_single_image(input_path: str, output_path: str, cfg):
    """단일 이미지에 대해 배경 제거를 테스트합니다."""
    print(f"Testing background removal on: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # 배경 제거 설정 생성
    bg_config = BgRemovalConfig(
        method=cfg.DATASET.BG_REMOVAL.METHOD,
        fixed_thresh=cfg.DATASET.BG_REMOVAL.FIXED_THRESH,
        percentile=cfg.DATASET.BG_REMOVAL.PERCENTILE,
        morph_kernel=cfg.DATASET.BG_REMOVAL.MORPH_KERNEL,
        keep_largest_only=cfg.DATASET.BG_REMOVAL.KEEP_LARGEST_ONLY,
        tight_crop=cfg.DATASET.BG_REMOVAL.TIGHT_CROP,
        fill_value=cfg.DATASET.BG_REMOVAL.FILL_VALUE,
        min_object_area=cfg.DATASET.BG_REMOVAL.MIN_OBJECT_AREA,
        normalize_to_uint8=cfg.DATASET.BG_REMOVAL.NORMALIZE_TO_UINT8,
        save_original=cfg.DATASET.BG_REMOVAL.SAVE_ORIGINAL,
        output_suffix=cfg.DATASET.BG_REMOVAL.OUTPUT_SUFFIX
    )
    
    # 배경 제거 실행
    success, error = process_image(input_path, output_path, bg_config)
    
    if success:
        print(f"Success! Output saved to: {output_path}")
        return True
    else:
        print(f"Error: {error}")
        return False

def test_batch_processing(input_dir: str, output_dir: str, cfg):
    """배치 처리를 테스트합니다."""
    print(f"Testing batch processing:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return False
    
    # 배경 제거 활성화
    cfg.defrost()
    cfg.DATASET.USE_BACKGROUND_REMOVAL = True
    cfg.freeze()
    
    # 배치 처리 실행
    process_dataset_with_background_removal(input_dir, output_dir, cfg)
    
    print("Batch processing completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Test background removal functionality')
    parser.add_argument('--mode', choices=['single', 'batch'], required=True,
                       help='Test mode: single image or batch processing')
    parser.add_argument('--input', required=True,
                       help='Input image path (single mode) or directory (batch mode)')
    parser.add_argument('--output', required=True,
                       help='Output image path (single mode) or directory (batch mode)')
    parser.add_argument('--config', default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--enable-bg-removal', action='store_true',
                       help='Enable background removal in config')
    
    args = parser.parse_args()
    
    # config 로드
    if os.path.exists(args.config):
        cfg = _C.clone()
        cfg.merge_from_file(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        print(f"Warning: Config file not found: {args.config}")
        print("Using default config")
        cfg = _C.clone()
    
    # 배경 제거 활성화 (명령행에서 요청된 경우)
    if args.enable_bg_removal:
        cfg.defrost()
        cfg.DATASET.USE_BACKGROUND_REMOVAL = True
        cfg.freeze()
        print("Background removal enabled in config")
    
    # 현재 설정 출력
    print("\nCurrent background removal settings:")
    print(f"  USE_BACKGROUND_REMOVAL: {cfg.DATASET.USE_BACKGROUND_REMOVAL}")
    if cfg.DATASET.USE_BACKGROUND_REMOVAL:
        bg_cfg = cfg.DATASET.BG_REMOVAL
        print(f"  Method: {bg_cfg.METHOD}")
        print(f"  Fixed threshold: {bg_cfg.FIXED_THRESH}")
        print(f"  Morph kernel: {bg_cfg.MORPH_KERNEL}")
        print(f"  Keep largest only: {bg_cfg.KEEP_LARGEST_ONLY}")
        print(f"  Tight crop: {bg_cfg.TIGHT_CROP}")
        print(f"  Fill value: {bg_cfg.FILL_VALUE}")
    
    print()
    
    # 테스트 실행
    if args.mode == 'single':
        success = test_single_image(args.input, args.output, cfg)
    elif args.mode == 'batch':
        success = test_batch_processing(args.input, args.output, cfg)
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)

if __name__ == '__main__':
    main() 
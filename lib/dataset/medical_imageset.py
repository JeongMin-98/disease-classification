import os
import json
import logging
import torch
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset
from torchvision import transforms
from utils.transform import get_basic_transforms
# from utils.transform import get_augmentation_transforms, fill_patch_grid, fill_and_pad_patch_grid, get_patch_transform
from utils.adaptive_preprocessing import create_clahe_transform, create_adaptive_transform
from utils.background_removal import process_image, config_to_bg_removal_config
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class MedicalImageDataset(BaseDataset):
    def __init__(self, cfg, transform=None, include_classes=None, num_workers=5, is_train=True):
        super().__init__()
        """
        Args:
            cfg: Configuration object containing dataset path and transform settings.
            transform (callable, optional): 직접 지정하는 transform (None이면 cfg에서 결정)
            include_classes (list, optional): A list of classes to include in the dataset.
            is_train (bool): 훈련 데이터셋인지 여부 (기본값: True)
        """
        json_file = cfg.DATASET.JSON
        json_file_name = os.path.basename(json_file)
        self.pickle_path = os.path.join(os.path.dirname(json_file), f"{json_file_name}_{cfg.DATASET.INCLUDE_CLASSES}_data.pkl")
        
        with open(json_file, 'r') as f:
            data = json.load(f)  # Load the JSON file

        self.mean, self.std = cfg.DATASET.MEAN, cfg.DATASET.STD
        self.bbox_crop_flag = cfg.DATASET.BBOX_CROP_FLAG
        logger.info(f"Dataset mean {self.mean}, std {self.std} BBOX_CROP_FLAG: {self.bbox_crop_flag}")

        self.data = []
        self.augment = cfg.DATASET.AUGMENT
        self.is_train = is_train
        self.cfg = cfg  # config 객체 저장
        
        # transform 결정: 직접 지정된 transform이 있으면 사용, 없으면 cfg에서 결정
        if transform is not None:
            self.transform = transform
        else:
            if is_train:
                self.transform = self.get_transform(cfg)  # augmentation 포함
            else:
                self.transform = self.get_test_transform(cfg)  # augmentation 없음
                logger.info("Using test transform (no augmentation)")
            
        self.include_classes = cfg.DATASET.INCLUDE_CLASSES
        self.label_to_idx = {label: idx for idx, label in enumerate(self.include_classes)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Lazy loading을 위해 메타데이터만 저장
        for record in data["data"]:
            # Ensure file_path is valid
            file_path = record.get("file_path")
            if not file_path or (isinstance(file_path, list) and len(file_path) == 0):
                logger.info(f"Skipping record with invalid file_path: {record}")
                continue

            # class에 콤마가 포함된 경우는 제외
            if ',' in record["class"]:
                logger.info(f"Skipping record with multi-class label: {record}")
                continue

            cls = record["class"].strip()
            if self.include_classes and cls in self.include_classes:
                new_record = record.copy()
                new_record["class"] = cls
                new_record["class_label"] = cls  # class_label 추가
                self.data.append(new_record)

        logger.info(f"Total samples after filtering: {len(self.data)}")
        
        # Lazy loading을 위해 db_rec는 메타데이터만 저장
        self.db_rec = self._initialize_metadata()
        
        self.class_counts = self.summary()
        self.num_classes = len(self.include_classes)

        self.show_class_count()

    def _initialize_metadata(self):
        """
        Lazy loading을 위해 메타데이터만 초기화
        """
        db_rec = []
        
        for record in self.data:
            file_path = record.get("file_path")
            if not file_path or (isinstance(file_path, list) and len(file_path) == 0):
                continue

            classes = [c.strip() for c in record["class"].split(',')]
            for c in classes:
                if self.include_classes and c in self.include_classes:
                    label_idx = self.label_to_idx[c]
                    
                    # 메타데이터만 저장 (이미지는 나중에 로드)
                    db_rec.append({
                        'patient_id': record["patient_id"],
                        'file_path': file_path,
                        'label': label_idx,
                        'class_label': c,
                        'valid_boxes': record.get("valid_boxes", [])
                    })
        
        return db_rec

    def _load_and_process_image(self, record):
        """
        이미지를 로드하고 전처리를 적용하는 함수
        """
        try:
            file_path = record['file_path']
            
            # HSV 배경제거 이미지 사용 설정이 있는 경우
            if hasattr(self.cfg.DATASET, 'USE_BACKGROUND_REMOVED') and self.cfg.DATASET.USE_BACKGROUND_REMOVED:
                bg_removed_type = getattr(self.cfg.DATASET, 'BACKGROUND_REMOVED_TYPE', 'folder')
                
                if bg_removed_type == 'folder':
                    # 기존 방식: 폴더에서 배경제거된 이미지 로드
                    bg_removed_dir = getattr(self.cfg.DATASET, 'BACKGROUND_REMOVED_DIR', '')
                    
                    if bg_removed_dir:
                        # 원본 파일명에서 patient_id 추출
                        file_name = os.path.basename(file_path)
                        name_without_ext = os.path.splitext(file_name)[0]
                        
                        # 배경제거 이미지 파일명 생성 (예: CAUHRA00802_bg_removed.jpg)
                        bg_removed_filename = f"{name_without_ext}_bg_removed.jpg"
                        
                        # 클래스명 가져오기
                        class_name = record.get('class', record.get('label'))
                        if isinstance(class_name, int):
                            # 숫자 레이블을 클래스명으로 변환
                            class_name = self.idx_to_label.get(class_name, 'unknown')
                        
                        # 배경제거 이미지 경로 생성
                        bg_removed_path = os.path.join(bg_removed_dir, class_name, bg_removed_filename)
                        
                        # 배경제거 이미지가 존재하면 사용
                        if os.path.exists(bg_removed_path):
                            file_path = bg_removed_path
                            logger.debug(f"Using background removed image: {bg_removed_path}")
                        else:
                            logger.warning(f"Background removed image not found: {bg_removed_path}, using original")
                
                elif bg_removed_type == 'hsv':
                    # HSV 실시간 배경제거는 이미지 로드 후에 적용 (아래에서 처리)
                    pass
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return None
            
            # 이미지 로드
            image = Image.open(file_path).convert("L")  # Grayscale로 로드
            
            # BBOX 크롭 적용 (필요한 경우)
            if self.bbox_crop_flag and record.get('valid_boxes'):
                valid_boxes = record['valid_boxes']
                if valid_boxes:
                    bbox = valid_boxes[0]["bbox"]
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    image = image.crop((x_min, y_min, x_max, y_max))
            
            # HSV 실시간 배경제거 적용
            if (hasattr(self.cfg.DATASET, 'USE_BACKGROUND_REMOVED') and 
                self.cfg.DATASET.USE_BACKGROUND_REMOVED and
                getattr(self.cfg.DATASET, 'BACKGROUND_REMOVED_TYPE', 'folder') == 'hsv'):
                try:
                    # HSV 배경제거 적용
                    image = self._apply_hsv_background_removal(image)
                except Exception as e:
                    logger.warning(f"HSV background removal failed for {file_path}: {e}")
                    # 배경 제거 실패 시 원본 이미지 사용
            
            # 기존 실시간 배경 제거 적용 (config에서 활성화된 경우, 단 USE_BACKGROUND_REMOVED가 False일 때만)
            elif (hasattr(self, 'cfg') and getattr(self.cfg.DATASET, 'USE_BACKGROUND_REMOVAL', False) and 
                  not getattr(self.cfg.DATASET, 'USE_BACKGROUND_REMOVED', False)):
                try:
                    # PIL Image를 numpy array로 변환
                    img_array = np.array(image)
                    
                    # 배경 제거 설정 가져오기
                    bg_config = config_to_bg_removal_config(self.cfg)
                    
                    # 배경 제거 적용 (임시 파일 없이 메모리에서 처리)
                    mask = self._compute_background_mask(img_array, bg_config)
                    img_array = self._apply_background_mask(img_array, mask, bg_config)
                    
                    # numpy array를 다시 PIL Image로 변환
                    image = Image.fromarray(img_array.astype(np.uint8))
                    
                except Exception as e:
                    logger.warning(f"Background removal failed for {file_path}: {e}")
                    # 배경 제거 실패 시 원본 이미지 사용
            
            # 전처리 적용
            if self.transform is not None:
                processed_image = self.transform(image)
            else:
                # transform이 None인 경우 기본 변환 적용
                processed_image = torch.from_numpy(np.array(image)).float()
            
            return {
                'patient_id': record['patient_id'],
                'image': processed_image,
                'label': record['label']
            }
            
        except Exception as e:
            logger.error(f"Error processing image {record.get('file_path', 'unknown')}: {str(e)}")
            return None

    def show_class_count(self):
        logger.info("Class distribution:")
        for class_name, count in self.class_counts.items():
            logger.info(f"{class_name}: {count}")
    
    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.db_rec)
    
    def __getitem__(self, idx):
        """
        Lazy loading으로 이미지를 로드하고 전처리를 적용
        """
        record = self.db_rec[idx]
        result = self._load_and_process_image(record)
        
        if result is None:
            # 에러 발생 시 더미 데이터 반환
            logger.warning(f"Failed to load image at index {idx}, returning dummy data")
            dummy_image = torch.zeros(3, 224, 224)
            return {
                'patient_id': record['patient_id'],
                'image': dummy_image,
                'label': record['label']
            }
        
        return result
        
    def summary(self):
        """
        Summarize the dataset, including the total number of samples and the number of samples for each class.
        """
        total_samples = len(self.db_rec)
        class_counts = Counter(record['label'] for record in self.db_rec)

        print(f"Total number of samples: {total_samples}")
        print("Number of samples per class:")
        for class_idx, count in class_counts.items():
            class_name = self.idx_to_label[class_idx]
            print(f"  {class_name}: {count}")
        
        return class_counts

    def _compute_background_mask(self, img_array: np.ndarray, bg_config) -> np.ndarray:
        """배경 마스크를 계산합니다."""
        from utils.background_removal import compute_mask
        
        # 그레이스케일 변환 (이미 그레이스케일이지만 확실히 하기 위해)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # 마스크 계산
        mask = compute_mask(gray, bg_config)
        return mask
    
    def _apply_background_mask(self, img_array: np.ndarray, mask: np.ndarray, bg_config) -> np.ndarray:
        """배경 마스크를 적용합니다."""
        from utils.background_removal import apply_mask_and_optionally_crop
        
        # 마스크 적용
        result, _ = apply_mask_and_optionally_crop(img_array, mask, bg_config)
        return result

    def get_labels(self):
        """
        모든 샘플의 라벨을 반환
        
        Returns:
            list: 모든 샘플의 라벨 리스트
        """
        return [record['label'] for record in self.db_rec]
    
    def get_label_indices(self, label):
        """
        특정 라벨을 가진 샘플들의 인덱스를 반환
        
        Args:
            label: 찾을 라벨 (int 또는 str)
            
        Returns:
            list: 해당 라벨을 가진 샘플들의 인덱스 리스트
        """
        if isinstance(label, str):
            # 문자열 라벨인 경우 인덱스로 변환
            if label in self.label_to_idx:
                label = self.label_to_idx[label]
            else:
                return []
        
        return [i for i, record in enumerate(self.db_rec) if record['label'] == label]
    
    def get_class_distribution(self):
        """
        클래스별 샘플 수를 딕셔너리로 반환
        
        Returns:
            dict: {class_name: count} 형태의 딕셔너리
        """
        distribution = {}
        for record in self.db_rec:
            class_name = self.idx_to_label[record['label']]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def _apply_hsv_background_removal(self, image):
        """
        HSV 기반 배경제거를 PIL 이미지에 적용합니다.
        
        Args:
            image: PIL Image 객체
            
        Returns:
            PIL Image: 배경제거가 적용된 이미지
        """
        from utils.background_removal import apply_hsv_background_removal
        
        # PIL Image를 numpy array로 변환
        img_array = np.array(image)
        
        # HSV 배경제거 설정 가져오기
        hsv_config = self.cfg.DATASET.HSV_BG_REMOVAL
        v_threshold = getattr(hsv_config, 'V_THRESHOLD', 50)
        protect_skin = getattr(hsv_config, 'PROTECT_SKIN', True)
        protect_bone = getattr(hsv_config, 'PROTECT_BONE', True)
        fill_value = 0
        
        # HSV 배경제거 적용
        result_rgb = apply_hsv_background_removal(
            img_array, 
            thresh=v_threshold,
            fill_value=fill_value,
            protect_skin=protect_skin,
            protect_bone=protect_bone
        )
        
        # RGB numpy array를 PIL Image로 변환 (grayscale로)
        result_gray = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY)
        result_image = Image.fromarray(result_gray)
        
        return result_image




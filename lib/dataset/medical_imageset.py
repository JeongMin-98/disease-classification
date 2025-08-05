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
            
            # 전처리 적용
            processed_image = self.transform(image)
            
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




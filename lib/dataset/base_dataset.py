import random
from collections import Counter
from torch.utils.data import Dataset
import logging
from torchvision import transforms
from utils.transform import get_basic_transforms
from utils.adaptive_preprocessing import create_clahe_transform, create_adaptive_transform

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.db_rec = []
        self.class_counts = None
        self.logger = logging.getLogger(__name__)

    def get_transform(self, cfg):
        """
        cfg 설정에 따라 적절한 transform을 반환합니다.
        
        Args:
            cfg: Configuration object containing transform settings
            
        Returns:
            transforms.Compose: 선택된 transform pipeline
        """
        transform_type = getattr(cfg.DATASET, 'TRANSFORM_TYPE', 'basic')
        self.logger.info(f"Using transform type: {transform_type}")
        
        if transform_type == 'clahe':
            return self.get_clahe_transform(cfg)
        elif transform_type == 'adaptive':
            return self.get_adaptive_transform(cfg)
        elif transform_type == 'basic':
            return self.get_basic_transform(cfg)
    
    def get_basic_transform(self, cfg):
        """기본 정규화 transform (정규분포)"""
        mean = cfg.DATASET.MEAN
        std = cfg.DATASET.STD
        augment = getattr(cfg.DATASET, 'AUGMENT', False)
        
        # 동적 이미지 크기 설정 (H, W 지원)
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', [224, 224])
        if isinstance(image_size, int):
            # 정수인 경우 1:1 비율 (H=W)
            image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)):
            # 리스트/튜플인 경우 (H, W) 형태
            if len(image_size) == 2:
                image_size = tuple(image_size)
            else:
                # 잘못된 형태인 경우 기본값 사용
                self.logger.warning(f"Invalid IMAGE_SIZE format: {image_size}, using default (224, 224)")
                image_size = (224, 224)
        else:
            # 기타 경우 기본값 사용
            self.logger.warning(f"Unknown IMAGE_SIZE type: {type(image_size)}, using default (224, 224)")
            image_size = (224, 224)
        
        self.logger.info(f"Image size set to: {image_size} (H: {image_size[0]}, W: {image_size[1]})")
        
        # 동적 채널 수 설정
        input_channels = getattr(cfg.DATASET, 'INPUT_CHANNELS', 3)
        
        transforms_list = [
            transforms.Grayscale(num_output_channels=input_channels),  # 동적 채널 수
            transforms.Resize(image_size),                            # 동적 크기 (H, W)
        ]
        
        # Data augmentation 적용 (AUGMENT=True인 경우)
        if augment:
            self.logger.info("Applying data augmentation: RandomHorizontalFlip, RandomRotation, RandomAffine")
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),  # 좌우 뒤집기 (50% 확률)
                transforms.RandomRotation(degrees=10),    # 회전 (±10도)
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1),  # 이동 (10% 범위)
                    scale=(0.9, 1.1),      # 크기 조정 (90-110%)
                    fill=0                 # 빈 공간을 0으로 채움
                ),
            ])
        
        # 기본 전처리
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        return transforms.Compose(transforms_list)
    
    def get_test_transform(self, cfg):
        """테스트용 transform (augmentation 없음)"""
        transform_type = getattr(cfg.DATASET, 'TRANSFORM_TYPE', 'basic')
        
        if transform_type == 'clahe':
            return self.get_clahe_test_transform(cfg)
        elif transform_type == 'adaptive':
            return self.get_adaptive_test_transform(cfg)
        else:
            return self.get_basic_test_transform(cfg)
    
    def get_basic_test_transform(self, cfg):
        """기본 테스트용 transform (augmentation 없음)"""
        mean = cfg.DATASET.MEAN
        std = cfg.DATASET.STD
        
        # 동적 이미지 크기 설정 (H, W 지원)
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', [224, 224])
        if isinstance(image_size, int):
            # 정수인 경우 1:1 비율 (H=W)
            image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)):
            # 리스트/튜플인 경우 (H, W) 형태
            if len(image_size) == 2:
                image_size = tuple(image_size)
            else:
                # 잘못된 형태인 경우 기본값 사용
                self.logger.warning(f"Invalid IMAGE_SIZE format: {image_size}, using default (224, 224)")
                image_size = (224, 224)
        else:
            # 기타 경우 기본값 사용
            self.logger.warning(f"Unknown IMAGE_SIZE type: {type(image_size)}, using default (224, 224)")
            image_size = (224, 224)
        
        # 동적 채널 수 설정
        input_channels = getattr(cfg.DATASET, 'INPUT_CHANNELS', 3)
        
        transforms_list = [
            transforms.Grayscale(num_output_channels=input_channels),  # 동적 채널 수
            transforms.Resize(image_size),                            # 동적 크기 (H, W)
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        
        return transforms.Compose(transforms_list)
    
    def get_clahe_test_transform(self, cfg):
        """CLAHE 테스트용 transform (augmentation 없음)"""
        mean = cfg.DATASET.MEAN
        std = cfg.DATASET.STD
        
        # 동적 이미지 크기 설정 (H, W 지원)
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', [224, 224])
        if isinstance(image_size, int):
            # 정수인 경우 1:1 비율 (H=W)
            image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)):
            # 리스트/튜플인 경우 (H, W) 형태
            if len(image_size) == 2:
                image_size = tuple(image_size)
            else:
                # 잘못된 형태인 경우 기본값 사용
                self.logger.warning(f"Invalid IMAGE_SIZE format: {image_size}, using default (224, 224)")
                image_size = (224, 224)
        else:
            # 기타 경우 기본값 사용
            self.logger.warning(f"Unknown IMAGE_SIZE type: {type(image_size)}, using default (224, 224)")
            image_size = (224, 224)
        
        # CLAHE 파라미터 (cfg에서 가져오거나 기본값 사용)
        clahe_params = getattr(cfg.DATASET, 'CLAHE_PARAMS', {})
        clip_limit = clahe_params.get('clip_limit', 2.0)
        tile_grid_size = clahe_params.get('tile_grid_size', (8, 8))
        
        # CLAHE transform 생성
        clahe_transform = create_clahe_transform(clip_limit, tile_grid_size)
        
        transforms_list = [
            transforms.Lambda(lambda x: clahe_transform(x)),  # CLAHE 전처리
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(image_size),  # 동적 크기 (H, W)
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        
        return transforms.Compose(transforms_list)
    
    def get_adaptive_test_transform(self, cfg):
        """Adaptive 테스트용 transform (augmentation 없음)"""
        mean = cfg.DATASET.MEAN
        std = cfg.DATASET.STD
        
        # 동적 이미지 크기 설정 (H, W 지원)
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', [224, 224])
        if isinstance(image_size, int):
            # 정수인 경우 1:1 비율 (H=W)
            image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)):
            # 리스트/튜플인 경우 (H, W) 형태
            if len(image_size) == 2:
                image_size = tuple(image_size)
            else:
                # 잘못된 형태인 경우 기본값 사용
                self.logger.warning(f"Invalid IMAGE_SIZE format: {image_size}, using default (224, 224)")
                image_size = (224, 224)
        else:
            # 기타 경우 기본값 사용
            self.logger.warning(f"Unknown IMAGE_SIZE type: {type(image_size)}, using default (224, 224)")
            image_size = (224, 224)
        
        # 적응적 히스토그램 transform 생성 (인자 없이 호출)
        adaptive_transform = create_adaptive_transform()
        
        transforms_list = [
            transforms.Lambda(lambda x: adaptive_transform(x)),  # Adaptive 전처리
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(image_size),  # 동적 크기 (H, W)
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        
        return transforms.Compose(transforms_list)
    
    def get_clahe_transform(self, cfg):
        """CLAHE 후 정규화 transform"""
        mean = cfg.DATASET.MEAN
        std = cfg.DATASET.STD
        augment = getattr(cfg.DATASET, 'AUGMENT', False)
        
        # 동적 이미지 크기 설정 (H, W 지원)
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', [224, 224])
        if isinstance(image_size, int):
            # 정수인 경우 1:1 비율 (H=W)
            image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)):
            # 리스트/튜플인 경우 (H, W) 형태
            if len(image_size) == 2:
                image_size = tuple(image_size)
            else:
                # 잘못된 형태인 경우 기본값 사용
                self.logger.warning(f"Invalid IMAGE_SIZE format: {image_size}, using default (224, 224)")
                image_size = (224, 224)
        else:
            # 기타 경우 기본값 사용
            self.logger.warning(f"Unknown IMAGE_SIZE type: {type(image_size)}, using default (224, 224)")
            image_size = (224, 224)
        
        # CLAHE 파라미터 (cfg에서 가져오거나 기본값 사용)
        clahe_params = getattr(cfg.DATASET, 'CLAHE_PARAMS', {})
        clip_limit = clahe_params.get('clip_limit', 2.0)
        tile_grid_size = clahe_params.get('tile_grid_size', (8, 8))
        
        # CLAHE transform 생성
        clahe_transform = create_clahe_transform(clip_limit, tile_grid_size)
        
        transforms_list = [
            transforms.Lambda(lambda x: clahe_transform(x)),  # CLAHE 전처리
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(image_size),  # 동적 크기 (H, W)
        ]
        
        # Data augmentation 적용 (AUGMENT=True인 경우)
        if augment:
            self.logger.info("Applying data augmentation to CLAHE transform: RandomHorizontalFlip, RandomRotation, RandomAffine")
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),  # 좌우 뒤집기 (50% 확률)
                transforms.RandomRotation(degrees=10),    # 회전 (±10도)
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1),  # 이동 (10% 범위)
                    scale=(0.9, 1.1),      # 크기 조정 (90-110%)
                    fill=0                 # 빈 공간을 0으로 채움
                ),
            ])
        
        # 기본 전처리
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        return transforms.Compose(transforms_list)
    
    def get_adaptive_transform(self, cfg):
        """적응적 히스토그램 평활화 transform"""
        mean = cfg.DATASET.MEAN
        std = cfg.DATASET.STD
        augment = getattr(cfg.DATASET, 'AUGMENT', False)
        
        # 동적 이미지 크기 설정 (H, W 지원)
        image_size = getattr(cfg.DATASET, 'IMAGE_SIZE', [224, 224])
        if isinstance(image_size, int):
            # 정수인 경우 1:1 비율 (H=W)
            image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)):
            # 리스트/튜플인 경우 (H, W) 형태
            if len(image_size) == 2:
                image_size = tuple(image_size)
            else:
                # 잘못된 형태인 경우 기본값 사용
                self.logger.warning(f"Invalid IMAGE_SIZE format: {image_size}, using default (224, 224)")
                image_size = (224, 224)
        else:
            # 기타 경우 기본값 사용
            self.logger.warning(f"Unknown IMAGE_SIZE type: {type(image_size)}, using default (224, 224)")
            image_size = (224, 224)
        
        # 적응적 히스토그램 transform 생성 (인자 없이 호출)
        adaptive_transform = create_adaptive_transform()
        
        transforms_list = [
            transforms.Lambda(lambda x: adaptive_transform(x)),  # Adaptive 전처리
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(image_size),  # 동적 크기 (H, W)
        ]
        
        # Data augmentation 적용 (AUGMENT=True인 경우)
        if augment:
            self.logger.info("Applying data augmentation to Adaptive transform: RandomHorizontalFlip, RandomRotation, RandomAffine")
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),  # 좌우 뒤집기 (50% 확률)
                transforms.RandomRotation(degrees=10),    # 회전 (±10도)
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1),  # 이동 (10% 범위)
                    scale=(0.9, 1.1),      # 크기 조정 (90-110%)
                    fill=0                 # 빈 공간을 0으로 채움
                ),
            ])
        
        # 기본 전처리
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        return transforms.Compose(transforms_list)

    def balance_dataset(self, target_count_per_class=None, min_ratio=1, max_ratio=1.05):
        """
        클래스별로 균등하게 샘플링하여 self.db_rec을 재구성한다.
        스마트 균형 맞춤: 224개로 Undersampling 가능하면 그대로, Oversampling 필요시 최소 클래스 수로 맞춤
        """
        class_counts = Counter([record['class_label'] for record in self.db_rec])
        self.logger.info(f"[balance_dataset] 샘플링 전 클래스별 개수: {dict(class_counts)}")
        balanced_data = []
        
        if target_count_per_class is not None:
            # 스마트 균형 맞춤 로직
            min_class_count = min(class_counts.values())
            
            # 모든 클래스가 target_count_per_class 이상인지 확인
            can_undersample = all(count >= target_count_per_class for count in class_counts.values())
            
            if can_undersample:
                # 모든 클래스가 충분하면 224개로 Undersampling
                self.logger.info(f"[balance_dataset] 모든 클래스가 {target_count_per_class}개 이상이므로 Undersampling 진행")
                for class_name, count in class_counts.items():
                    class_data = [record for record in self.db_rec if record['class_label'] == class_name]
                    selected_data = random.sample(class_data, target_count_per_class)
                    balanced_data.extend(selected_data)
                    self.logger.info(f"[balance_dataset] {class_name}: {len(selected_data)}개 Undersampling 완료 (원본 {len(class_data)})")
            else:
                # 일부 클래스가 부족하면 최소 클래스 수로 맞춤
                self.logger.info(f"[balance_dataset] 일부 클래스가 {target_count_per_class}개 미만이므로 최소 클래스 수({min_class_count}개)로 맞춤")
                for class_name, count in class_counts.items():
                    class_data = [record for record in self.db_rec if record['class_label'] == class_name]
                    if count > min_class_count:
                        # 다수 클래스: 최소 클래스 수에 맞춰 Undersampling
                        scale_factor = random.uniform(min_ratio, max_ratio)
                        target_count = int(min_class_count * scale_factor)
                        selected_data = random.choices(class_data, k=target_count)
                    else:
                        # 소수 클래스: 원본 그대로 유지
                        target_count = count
                        selected_data = class_data
                    balanced_data.extend(selected_data)
                    self.logger.info(f"[balance_dataset] {class_name}: {len(selected_data)}개 샘플링 완료 (원본 {len(class_data)})")
        else:
            # 기존 로직: 최소 클래스 수 기준으로 비율 샘플링
            min_class_count = min(class_counts.values())
            self.logger.info(f"[balance_dataset] min_class_count={min_class_count} 기준으로 비율 샘플링 진행")
            for class_name, count in class_counts.items():
                class_data = [record for record in self.db_rec if record['class_label'] == class_name]
                scale_factor = random.uniform(min_ratio, max_ratio) if count > min_class_count else 1
                target_count = int(min_class_count * scale_factor)
                balanced_data.extend(random.choices(class_data, k=target_count))
                self.logger.info(f"[balance_dataset] {class_name}: {target_count}개 샘플링 완료 (원본 {len(class_data)})")
        
        random.shuffle(balanced_data)
        self.db_rec = balanced_data
        self.class_counts = Counter([record['class_label'] for record in self.db_rec])
        self.logger.info(f"[balance_dataset] 샘플링 후 클래스별 개수: {dict(self.class_counts)}")

    def check_patient_overlap(self, other_dataset):
        """
        다른 데이터셋과 환자 ID가 겹치는지 확인한다.
        """
        my_patients = set([rec['patient_id'] for rec in self.db_rec])
        other_patients = set([rec['patient_id'] for rec in other_dataset.db_rec])
        overlap = my_patients & other_patients
        return overlap 
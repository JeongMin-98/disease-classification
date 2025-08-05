import random
from collections import Counter, defaultdict
import torch
import numpy as np
import logging
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balance_dataset(dataset, min_ratio=1, max_ratio=1.05, target_count_per_class=None):
    """
    Balance the dataset by ensuring all classes have data within a certain ratio of the smallest class.
    If target_count_per_class is specified, each class will have exactly that many samples.

    Args:
        dataset (MedicalImageDataset): The dataset to balance.
        min_ratio (float): Minimum scaling factor for class balancing.
        max_ratio (float): Maximum scaling factor for class balancing.
        target_count_per_class (int, optional): Target number of samples per class. If None, uses original balancing logic.

    Returns:
        dataset: A balanced dataset.
    """
    class_counts = Counter([record['label'] for record in dataset.db_rec])
    
    balanced_data = []
    
    if target_count_per_class is not None:
        # 사용자가 지정한 개수로 각 클래스 제한
        for class_name, count in class_counts.items():
            class_data = [record for record in dataset.db_rec if record['label'] == class_name]
            # 지정된 개수만큼 랜덤 선택 (원본 데이터보다 적으면 중복 허용)
            if len(class_data) >= target_count_per_class:
                selected_data = random.sample(class_data, target_count_per_class)
            else:
                # 원본 데이터보다 적으면 중복해서 채움
                selected_data = random.choices(class_data, k=target_count_per_class)
            balanced_data.extend(selected_data)
    else:
        # 기존 로직: 최소 클래스 기준으로 균형 맞추기
        min_class_count = min(class_counts.values())
        for class_name, count in class_counts.items():
            class_data = [record for record in dataset.db_rec if record['label'] == class_name]
            scale_factor = random.uniform(min_ratio, max_ratio) if count > min_class_count else 1
            target_count = int(min_class_count * scale_factor)
            balanced_data.extend(random.choices(class_data, k=target_count))
    
    random.shuffle(balanced_data)
    dataset.db_rec = balanced_data
    
    dataset.class_counts = Counter([record['label'] for record in dataset.db_rec])


def verify_split_distribution(train_set, val_set, test_set, dataset, logger=None):
    """
    데이터 분할 결과를 검증하여 클래스별 분포와 겹침 여부를 확인
    
    Args:
        train_set, val_set, test_set: 분할된 데이터셋들
        dataset: 원본 데이터셋
        logger: 로깅을 위한 logger 객체
    """
    if logger is None:
        logger = logging.getLogger("split_verification")
    
    # 각 세트의 인덱스
    train_indices = set(train_set.indices)
    val_indices = set(val_set.indices)
    test_indices = set(test_set.indices)
    
    # 겹침 확인
    train_val_overlap = train_indices & val_indices
    train_test_overlap = train_indices & test_indices
    val_test_overlap = val_indices & test_indices
    
    logger.info("=== 데이터 분할 검증 결과 ===")
    logger.info(f"Train set 크기: {len(train_set)}")
    logger.info(f"Val set 크기: {len(val_set)}")
    logger.info(f"Test set 크기: {len(test_set)}")
    logger.info(f"총 샘플 수: {len(train_set) + len(val_set) + len(test_set)}")
    
    # 겹침 검사
    if train_val_overlap:
        logger.warning(f"Train-Val 겹침: {len(train_val_overlap)}개")
    else:
        logger.info("✓ Train-Val 겹침 없음")
    
    if train_test_overlap:
        logger.warning(f"Train-Test 겹침: {len(train_test_overlap)}개")
    else:
        logger.info("✓ Train-Test 겹침 없음")
    
    if val_test_overlap:
        logger.warning(f"Val-Test 겹침: {len(val_test_overlap)}개")
    else:
        logger.info("✓ Val-Test 겹침 없음")
    
    # 클래스별 분포 확인
    if hasattr(dataset, 'get_labels'):
        all_labels = dataset.get_labels()
        train_labels = [all_labels[i] for i in train_set.indices]
        val_labels = [all_labels[i] for i in val_set.indices]
        test_labels = [all_labels[i] for i in test_set.indices]
        
        # 클래스별 카운트
        train_counts = Counter(train_labels)
        val_counts = Counter(val_labels)
        test_counts = Counter(test_labels)
        
        logger.info("=== 클래스별 분포 ===")
        for class_idx in sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())):
            class_name = dataset.idx_to_label.get(class_idx, f"Class_{class_idx}")
            logger.info(f"{class_name}: Train={train_counts[class_idx]}, Val={val_counts[class_idx]}, Test={test_counts[class_idx]}")
    
    logger.info("=== 검증 완료 ===")


def create_balanced_stratified_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42, verify=True, logger=None):
    """
    클래스별로 균등하게 분배하여 train/val/test로 분할하는 함수
    
    Args:
        dataset: 분할할 데이터셋
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        random_state: 랜덤 시드
        verify: 분할 결과 검증 여부
        logger: 로깅을 위한 logger 객체
    
    Returns:
        train_set, val_set, test_set: Subset 객체들
    """
    # dataset.get_labels() 메서드 사용
    if hasattr(dataset, 'get_labels'):
        labels = dataset.get_labels()
    else:
        # fallback: 기존 방식
        total_size = len(dataset)
        labels = []
        for idx in range(total_size):
            try:
                if hasattr(dataset, 'db_rec') and idx < len(dataset.db_rec):
                    label = dataset.db_rec[idx]['label']
                else:
                    sample = dataset[idx]
                    if isinstance(sample, dict):
                        label = sample['label']
                    else:
                        _, label = sample
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                labels.append(label)
            except Exception as e:
                logging.warning(f"Error getting label for index {idx}: {e}")
                labels.append(0)  # 기본값
    
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    # 클래스별로 인덱스 그룹화
    class_indices = {}
    for label in unique_labels:
        if hasattr(dataset, 'get_label_indices'):
            # dataset의 get_label_indices 메서드 사용
            class_indices[label] = dataset.get_label_indices(label)
        else:
            # fallback: 기존 방식
            class_indices[label] = np.where(labels == label)[0]
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # 각 클래스별로 균등하게 분할
    for label in unique_labels:
        class_idx = class_indices[label]
        class_size = len(class_idx)
        
        # 각 클래스 내에서 train/val/test 분할
        train_size = int(train_ratio * class_size)
        val_size = int(val_ratio * class_size)
        test_size = class_size - train_size - val_size
        
        # 랜덤 셔플
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(class_idx)
        
        # 분할
        train_indices.extend(shuffled_indices[:train_size])
        val_indices.extend(shuffled_indices[train_size:train_size + val_size])
        test_indices.extend(shuffled_indices[train_size + val_size:])
    
    # Subset 객체 생성
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    # 검증 (옵션)
    if verify:
        verify_split_distribution(train_set, val_set, test_set, dataset, logger)
    
    return train_set, val_set, test_set


def create_stratified_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    클래스 비율을 유지하면서 데이터셋을 train/val/test로 분할
    
    Args:
        dataset: 분할할 데이터셋
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        random_state: 랜덤 시드
    
    Returns:
        train_set, val_set, test_set: Subset 객체들
    """
    # dataset.get_labels() 메서드 사용
    if hasattr(dataset, 'get_labels'):
        labels = dataset.get_labels()
    else:
        # fallback: 기존 방식
        total_size = len(dataset)
        labels = []
        for idx in range(total_size):
            try:
                if hasattr(dataset, 'db_rec') and idx < len(dataset.db_rec):
                    label = dataset.db_rec[idx]['label']
                else:
                    sample = dataset[idx]
                    if isinstance(sample, dict):
                        label = sample['label']
                    else:
                        _, label = sample
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                labels.append(label)
            except Exception as e:
                logging.warning(f"Error getting label for index {idx}: {e}")
                labels.append(0)  # 기본값
    
    labels = np.array(labels)
    
    # 먼저 train과 temp로 분할
    train_indices, temp_indices = train_test_split(
        range(len(labels)),
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state
    )
    
    # temp를 val과 test로 분할
    temp_labels = labels[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_state
    )
    
    # Subset 객체 생성
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    return train_set, val_set, test_set


def analyze_batch_distribution(dataloader, dataset, num_batches=5, logger=None):
    """
    DataLoader의 배치당 클래스 분포를 분석하고 로깅
    
    Args:
        dataloader: 분석할 DataLoader
        dataset: 원본 데이터셋 (라벨 정보 접근용)
        num_batches: 분석할 배치 수
        logger: 로깅을 위한 logger 객체
    """
    if logger is None:
        logger = logging.getLogger("batch_analysis")
    
    logger.info(f"=== 배치 분포 분석 (처음 {num_batches}개 배치) ===")
    
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_count >= num_batches:
            break
        
        # MedicalImageDataset은 딕셔너리를 반환하므로 'label' 키로 접근
        if isinstance(batch, dict):
            labels = batch['label']
        else:
            # 기존 방식 (튜플인 경우)
            _, labels = batch
            
        # 배치 내 클래스별 샘플 수 계산
        batch_class_counts = defaultdict(int)
        for label in labels:
            if isinstance(label, torch.Tensor):
                label = label.item()
            batch_class_counts[label] += 1
        
        # 클래스별 분포를 정렬하여 로깅
        class_distribution = dict(sorted(batch_class_counts.items()))
        logger.info(f"배치 {batch_idx}: 클래스 분포 = {class_distribution}")
        
        batch_count += 1
    
    logger.info("=== 배치 분포 분석 완료 ===")


class EarlyStopping:
    """
    조기 종료(Early stopping) 콜백 클래스
    """
    def __init__(self, patience=7, verbose=False, delta=0, path=None, min_epochs=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.best_state = None
        self.min_epochs = min_epochs
        self.current_epoch = 0
        self.min_epochs_reached = False  # 최소 epoch 도달 여부 추적

    def __call__(self, val_loss, model=None):
        self.current_epoch += 1
        score = -val_loss
        
        # 최소 epoch 수에 도달했는지 확인
        if self.current_epoch >= self.min_epochs and not self.min_epochs_reached:
            self.min_epochs_reached = True
            self.counter = 0  # patience counter 리셋
            if self.verbose:
                print(f'Reached minimum epochs ({self.min_epochs}), resetting patience counter')
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # 최소 epoch 수를 지난 후에만 counter 증가
            if self.min_epochs_reached:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            # 성능이 개선되면 counter 리셋 (최소 epoch 이후에만)
            if self.min_epochs_reached:
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if model is not None:
            self.best_state = model.state_dict().copy()
        self.val_loss_min = val_loss


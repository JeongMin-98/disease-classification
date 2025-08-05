import torch
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict
import random
import logging


class BalancedBatchSampler(Sampler):
    """
    배치마다 균등한 클래스 분포를 보장하는 Sampler
    각 배치에서 모든 클래스가 동일한 수의 샘플을 가지도록 함
    """
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, log_batch_distribution=True):
        """
        Args:
            dataset: 데이터셋 (labels 속성이 있어야 함)
            batch_size: 배치 크기 (클래스 수로 나누어 떨어져야 함)
            shuffle: 셔플 여부
            drop_last: 마지막 배치를 버릴지 여부
            log_batch_distribution: 배치당 클래스 분포를 로깅할지 여부
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.log_batch_distribution = log_batch_distribution
        
        # 성능 최적화: 라벨 정보만 미리 캐시
        self._cache_labels()
        
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        
        # 배치 크기가 클래스 수로 나누어 떨어지는지 확인
        if batch_size % self.num_classes != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by number of classes ({self.num_classes})")
        
        self.samples_per_class = batch_size // self.num_classes
        
        # 각 클래스의 샘플 수 확인
        min_samples = min(len(indices) for indices in self.class_indices.values())
        if min_samples < self.samples_per_class:
            raise ValueError(f"Not enough samples in some classes. Minimum: {min_samples}, Required per class: {self.samples_per_class}")
        
        # 배치 수 계산
        self.num_batches = min(len(indices) // self.samples_per_class for indices in self.class_indices.values())
        if not drop_last:
            # drop_last=False인 경우 더 많은 배치를 만들 수 있는지 확인
            max_batches = min(len(indices) for indices in self.class_indices.values())
            self.num_batches = max_batches
        
        # 로깅 설정
        self.logger = logging.getLogger("BalancedBatchSampler")
        if self.log_batch_distribution:
            self.logger.info(f"BalancedBatchSampler 초기화:")
            self.logger.info(f"  - 클래스 수: {self.num_classes}")
            self.logger.info(f"  - 배치 크기: {self.batch_size}")
            self.logger.info(f"  - 클래스당 샘플 수: {self.samples_per_class}")
            self.logger.info(f"  - 총 배치 수: {self.num_batches}")
            self.logger.info(f"  - 클래스별 전체 샘플 수: {dict([(cls, len(indices)) for cls, indices in self.class_indices.items()])}")
    
    def _cache_labels(self):
        """라벨 정보만 미리 캐시하여 성능 최적화"""
        self.class_indices = defaultdict(list)
        self.label_cache = {}  # 인덱스별 라벨 캐시
        
        # 데이터셋의 라벨 정보만 미리 수집
        for idx in range(len(self.dataset)):
            try:
                # MedicalImageDataset의 경우 라벨 정보만 빠르게 접근
                if hasattr(self.dataset, 'db_rec') and idx < len(self.dataset.db_rec):
                    # db_rec에서 직접 라벨 정보 가져오기 (이미지 로딩 없이)
                    label = self.dataset.db_rec[idx]['label']
                else:
                    # 일반적인 경우: 샘플을 가져와서 라벨만 추출
                    sample = self.dataset[idx]
                    if isinstance(sample, dict):
                        label = sample['label']
                    else:
                        _, label = sample
                
                if isinstance(label, torch.Tensor):
                    label = label.item()
                
                self.class_indices[label].append(idx)
                self.label_cache[idx] = label
                
            except Exception as e:
                # 에러 발생 시 해당 인덱스 스킵
                logging.warning(f"Error accessing sample at index {idx}: {e}")
                continue
    
    def _log_batch_distribution(self, batch_indices, batch_idx):
        """배치당 클래스 분포를 로깅 (최적화된 버전)"""
        if not self.log_batch_distribution:
            return
        
        # 로깅 빈도 줄이기: 처음 3개 배치와 20개 배치마다만 로깅
        if batch_idx >= 3 and batch_idx % 20 != 0:
            return
        
        # 캐시된 라벨 정보 사용 (데이터셋 접근 없이)
        batch_class_counts = defaultdict(int)
        for idx in batch_indices:
            if idx in self.label_cache:
                label = self.label_cache[idx]
            else:
                # 캐시에 없는 경우에만 데이터셋 접근
                sample = self.dataset[idx]
                if isinstance(sample, dict):
                    label = sample['label']
                else:
                    _, label = sample
                if isinstance(label, torch.Tensor):
                    label = label.item()
                self.label_cache[idx] = label
            
            batch_class_counts[label] += 1
        
        # 클래스별 분포를 정렬하여 로깅
        class_distribution = dict(sorted(batch_class_counts.items()))
        self.logger.info(f"배치 {batch_idx}: 클래스 분포 = {class_distribution}")
    
    def __iter__(self):
        # 각 클래스의 인덱스를 셔플
        if self.shuffle:
            for class_idx in self.class_indices:
                random.shuffle(self.class_indices[class_idx])
        
        # 배치 생성
        for batch_idx in range(self.num_batches):
            batch_indices = []
            for class_idx in self.class_indices:
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                batch_indices.extend(self.class_indices[class_idx][start_idx:end_idx])
            
            # 배치 내에서 셔플 (선택사항)
            if self.shuffle:
                random.shuffle(batch_indices)
            
            # 배치 분포 로깅
            self._log_batch_distribution(batch_indices, batch_idx)
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


class StratifiedBatchSampler(Sampler):
    """
    Stratified sampling을 사용하여 클래스 비율을 유지하는 Sampler
    """
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, log_batch_distribution=True):
        """
        Args:
            dataset: 데이터셋 (labels 속성이 있어야 함)
            batch_size: 배치 크기
            shuffle: 셔플 여부
            drop_last: 마지막 배치를 버릴지 여부
            log_batch_distribution: 배치당 클래스 분포를 로깅할지 여부
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.log_batch_distribution = log_batch_distribution
        
        # 성능 최적화: 라벨 정보만 미리 캐시
        self._cache_labels()
        
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        
        # 전체 데이터셋 크기
        self.dataset_size = len(dataset)
        
        # 클래스별 비율 계산
        class_counts = {cls: len(indices) for cls, indices in self.class_indices.items()}
        total_count = sum(class_counts.values())
        self.class_ratios = {cls: count / total_count for cls, count in class_counts.items()}
        
        # 배치 수 계산
        if drop_last:
            self.num_batches = self.dataset_size // batch_size
        else:
            self.num_batches = (self.dataset_size + batch_size - 1) // batch_size
        
        # 로깅 설정
        self.logger = logging.getLogger("StratifiedBatchSampler")
        if self.log_batch_distribution:
            self.logger.info(f"StratifiedBatchSampler 초기화:")
            self.logger.info(f"  - 클래스 수: {self.num_classes}")
            self.logger.info(f"  - 배치 크기: {self.batch_size}")
            self.logger.info(f"  - 총 배치 수: {self.num_batches}")
            self.logger.info(f"  - 클래스별 전체 샘플 수: {class_counts}")
            self.logger.info(f"  - 클래스별 비율: {dict([(cls, f'{ratio:.3f}') for cls, ratio in self.class_ratios.items()])}")
    
    def _cache_labels(self):
        """라벨 정보만 미리 캐시하여 성능 최적화"""
        self.class_indices = defaultdict(list)
        self.label_cache = {}  # 인덱스별 라벨 캐시
        
        # 데이터셋의 라벨 정보만 미리 수집
        for idx in range(len(self.dataset)):
            try:
                # MedicalImageDataset의 경우 라벨 정보만 빠르게 접근
                if hasattr(self.dataset, 'db_rec') and idx < len(self.dataset.db_rec):
                    # db_rec에서 직접 라벨 정보 가져오기 (이미지 로딩 없이)
                    label = self.dataset.db_rec[idx]['label']
                else:
                    # 일반적인 경우: 샘플을 가져와서 라벨만 추출
                    sample = self.dataset[idx]
                    if isinstance(sample, dict):
                        label = sample['label']
                    else:
                        _, label = sample
                
                if isinstance(label, torch.Tensor):
                    label = label.item()
                
                self.class_indices[label].append(idx)
                self.label_cache[idx] = label
                
            except Exception as e:
                # 에러 발생 시 해당 인덱스 스킵
                logging.warning(f"Error accessing sample at index {idx}: {e}")
                continue
    
    def _log_batch_distribution(self, batch_indices, batch_idx):
        """배치당 클래스 분포를 로깅 (최적화된 버전)"""
        if not self.log_batch_distribution:
            return
        
        # 로깅 빈도 줄이기: 처음 3개 배치와 20개 배치마다만 로깅
        if batch_idx >= 3 and batch_idx % 20 != 0:
            return
        
        # 캐시된 라벨 정보 사용 (데이터셋 접근 없이)
        batch_class_counts = defaultdict(int)
        for idx in batch_indices:
            if idx in self.label_cache:
                label = self.label_cache[idx]
            else:
                # 캐시에 없는 경우에만 데이터셋 접근
                sample = self.dataset[idx]
                if isinstance(sample, dict):
                    label = sample['label']
                else:
                    _, label = sample
                if isinstance(label, torch.Tensor):
                    label = label.item()
                self.label_cache[idx] = label
            
            batch_class_counts[label] += 1
        
        # 클래스별 분포를 정렬하여 로깅
        class_distribution = dict(sorted(batch_class_counts.items()))
        self.logger.info(f"배치 {batch_idx}: 클래스 분포 = {class_distribution}")
    
    def __iter__(self):
        # 각 클래스의 인덱스를 셔플
        if self.shuffle:
            for class_idx in self.class_indices:
                random.shuffle(self.class_indices[class_idx])
        
        # 클래스별 인덱스 포인터
        class_pointers = {cls: 0 for cls in self.classes}
        
        # 배치 생성
        for batch_idx in range(self.num_batches):
            batch_indices = []
            
            # 각 클래스에서 비율에 맞게 샘플 선택
            for cls in self.classes:
                target_count = int(self.batch_size * self.class_ratios[cls])
                if target_count > 0:
                    start_idx = class_pointers[cls]
                    end_idx = min(start_idx + target_count, len(self.class_indices[cls]))
                    batch_indices.extend(self.class_indices[cls][start_idx:end_idx])
                    class_pointers[cls] = end_idx
                    
                    # 클래스 인덱스가 끝에 도달하면 다시 시작
                    if class_pointers[cls] >= len(self.class_indices[cls]):
                        class_pointers[cls] = 0
                        if self.shuffle:
                            random.shuffle(self.class_indices[cls])
            
            # 배치 크기가 부족하면 다른 클래스에서 보충
            while len(batch_indices) < self.batch_size:
                for cls in self.classes:
                    if len(batch_indices) >= self.batch_size:
                        break
                    if class_pointers[cls] < len(self.class_indices[cls]):
                        batch_indices.append(self.class_indices[cls][class_pointers[cls]])
                        class_pointers[cls] += 1
                    else:
                        class_pointers[cls] = 0
                        if self.shuffle:
                            random.shuffle(self.class_indices[cls])
            
            # 배치 크기 조정
            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[:self.batch_size]
            
            # 배치 내에서 셔플
            if self.shuffle:
                random.shuffle(batch_indices)
            
            # 배치 분포 로깅
            self._log_batch_distribution(batch_indices, batch_idx)
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches 
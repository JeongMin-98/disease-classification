# Wandb Sweep 사용법 가이드

## 개요

Wandb Sweep은 하이퍼파라미터 최적화를 자동화하는 강력한 도구입니다. 여러 하이퍼파라미터 조합을 자동으로 테스트하여 최적의 설정을 찾아줍니다.

## 주요 구성 요소

1. **Sweep Configuration**: 어떤 하이퍼파라미터를 테스트할지 정의
2. **Agent**: 실제 실험을 실행하는 프로세스
3. **Controller**: sweep을 관리하고 결과를 수집

## 파일 구조

```
tool/
├── train_sweep.py          # 메인 sweep 실행 파일
├── sweep_configs.py        # 다양한 sweep 설정들
├── simple_sweep_example.py # 간단한 예시
├── run_sweep_examples.py   # 사용 예시 스크립트
└── README_sweep.md         # 이 파일
```

## 빠른 시작

### 1. 기본 사용법

```bash
# 기본 설정으로 sweep 실행
python train_sweep.py --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml

# 빠른 테스트 (4번의 실험)
python train_sweep.py --sweep_config quick_test --count 4

# 학습률 최적화 (10번의 실험)
python train_sweep.py --sweep_config learning_rate --count 10
```

### 2. 사용 가능한 설정들

- **base**: 기본 설정 (Bayesian optimization)
- **quick_test**: 빠른 테스트용 (Grid search, 적은 실험 수)
- **learning_rate**: 학습률 최적화 전용
- **batch_size**: 배치 크기 최적화 전용
- **data_balancing**: 데이터 균등화 효과 테스트
- **full_optimization**: 전체 하이퍼파라미터 최적화 (대규모)

## 명령어 옵션

```bash
python train_sweep.py [옵션들]
```

### 기본 옵션

- `--cfg`: 실험에 사용할 config yaml 경로 (기본값: ra_hand_classifier_OA_Normal.yaml)
- `--sweep_config`: 사용할 sweep 설정 (기본값: base)
- `--project`: Wandb 프로젝트 이름 (기본값: disease-classification)
- `--count`: 실행할 실험 수 (기본값: 20)

### 커스텀 설정 옵션

- `--method`: Sweep 방법 (grid, random, bayes)
- `--sweep_name`: Sweep 이름
- `--metric`: 최적화할 메트릭 이름 (기본값: test_accuracy)
- `--custom_params`: 커스텀 파라미터 (JSON 형식)

## 사용 예시

### 1. 빠른 테스트
```bash
python train_sweep.py \
    --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml \
    --sweep_config quick_test \
    --count 4 \
    --project my-experiment
```

### 2. 학습률 최적화
```bash
python train_sweep.py \
    --cfg experiments/image_exp/foot_xray/foot_xray_classifier.yaml \
    --sweep_config learning_rate \
    --count 15 \
    --project foot-xray-optimization
```

### 3. 커스텀 설정
```bash
python train_sweep.py \
    --cfg experiments/image_exp/foot/foot_classifier.yaml \
    --method grid \
    --sweep_name custom_foot_test \
    --custom_params '{"TRAIN.LR": {"values": [0.001, 0.01]}, "TRAIN.BATCH_SIZE_PER_GPU": {"values": [16, 32]}}' \
    --count 4
```

### 4. 대규모 최적화
```bash
python train_sweep.py \
    --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml \
    --sweep_config full_optimization \
    --count 100 \
    --project large-scale-optimization
```

## Sweep 방법 (Method)

### 1. Grid Search
- 모든 가능한 조합을 체계적으로 테스트
- 빠르지만 조합이 많으면 시간이 오래 걸림

```python
sweep_config = {
    'method': 'grid',
    'parameters': {
        'learning_rate': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [16, 32, 64]}
    }
}
```

### 2. Random Search
- 무작위로 하이퍼파라미터 조합을 선택
- Grid search보다 효율적일 수 있음

```python
sweep_config = {
    'method': 'random',
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.1},
        'batch_size': {'values': [16, 32, 64]}
    }
}
```

### 3. Bayesian Optimization
- 이전 결과를 바탕으로 다음 조합을 지능적으로 선택
- 가장 효율적인 방법

```python
sweep_config = {
    'method': 'bayes',
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.1, 'distribution': 'log_uniform'},
        'batch_size': {'values': [16, 32, 64]}
    }
}
```

## 설정 파일 사용법

### 1. 미리 정의된 설정 사용

```python
from sweep_configs import get_sweep_config

# 기본 설정 사용
config = get_sweep_config('base')

# 커스텀 파라미터 추가
custom_params = {
    'TRAIN.LR': {'values': [0.001, 0.01]}
}
config = get_sweep_config('base', custom_params)
```

### 2. 커스텀 설정 생성

```python
from sweep_configs import create_custom_sweep_config

config = create_custom_sweep_config(
    method='bayes',
    name='my_custom_sweep',
    metric_name='test_accuracy',
    parameters={
        'TRAIN.LR': {'min': 0.0001, 'max': 0.01},
        'TRAIN.BATCH_SIZE_PER_GPU': {'values': [16, 32, 64]}
    }
)
```

## 주요 기능

### 1. Early Termination
- 성능이 좋지 않은 실험을 조기에 종료
- 시간과 리소스 절약

```python
'early_terminate': {
    'type': 'hyperband',
    'min_iter': 10
}
```

### 2. 조건부 파라미터
- 특정 조건에서만 적용되는 파라미터

```python
'parameters': {
    'optimizer': {'values': ['adam', 'sgd']},
    'momentum': {
        'values': [0.9, 0.95],
        'parent': 'optimizer',
        'parent_values': ['sgd']
    }
}
```

### 3. 커스텀 메트릭
- 여러 메트릭을 조합하여 최적화

```python
'metric': {
    'name': 'custom_metric',
    'goal': 'maximize'
}
```

## 모니터링

### 1. Wandb 대시보드
- 실시간으로 sweep 진행 상황 확인
- 각 실험의 결과 비교
- 최적 하이퍼파라미터 조합 확인

### 2. 결과 분석
```python
# 최고 성능 실험 찾기
api = wandb.Api()
sweep = api.sweep("username/project/sweep_id")
best_run = sweep.best_run
print(f"Best accuracy: {best_run.summary['test_accuracy']}")
print(f"Best config: {best_run.config}")
```

## 팁과 모범 사례

### 1. 하이퍼파라미터 범위 설정
- 너무 넓은 범위는 비효율적
- 너무 좁은 범위는 최적해를 놓칠 수 있음
- 논문이나 경험을 바탕으로 설정

### 2. 실험 수 조정
- Grid search: 모든 조합 테스트
- Random/Bayes: 20-100회 정도가 적당
- 조기 종료 조건 설정

### 3. 메트릭 선택
- 검증 정확도보다는 테스트 정확도 사용
- 과적합 방지를 위한 메트릭 고려

### 4. 리소스 관리
- GPU 메모리 고려
- 병렬 실행 시 리소스 분배
- 클라우드 비용 고려

## 문제 해결

### 1. Sweep이 멈춤
- 네트워크 연결 확인
- Wandb 로그인 상태 확인
- 리소스 부족 여부 확인

### 2. 결과가 예상과 다름
- 하이퍼파라미터 범위 재검토
- 메트릭 정의 확인
- 데이터셋 문제 확인

### 3. 성능 저하
- 조기 종료 조건 조정
- 실험 수 줄이기
- 더 효율적인 방법 사용 (Bayes > Random > Grid)

## 고급 사용법

### 1. 병렬 실행
여러 터미널에서 동일한 sweep을 실행하면 자동으로 병렬 처리됩니다:

```bash
# 터미널 1
python train_sweep.py --sweep_config learning_rate --count 10

# 터미널 2 (동시에 실행)
python train_sweep.py --sweep_config learning_rate --count 10
```

### 2. 설정 파일 수정
`sweep_configs.py` 파일을 수정하여 새로운 설정을 추가할 수 있습니다:

```python
# 새로운 설정 추가
CUSTOM_CONFIG = {
    'method': 'bayes',
    'name': 'my_custom_sweep',
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        # 커스텀 파라미터들
    }
}

# SWEEP_CONFIGS에 추가
SWEEP_CONFIGS['my_custom'] = CUSTOM_CONFIG
```

### 3. 결과 분석 스크립트
```python
import wandb
api = wandb.Api()

# 특정 sweep의 모든 실행 결과 가져오기
sweep = api.sweep("username/project/sweep_id")
runs = sweep.runs

# 결과 분석
for run in runs:
    print(f"Run: {run.name}")
    print(f"Config: {run.config}")
    print(f"Accuracy: {run.summary.get('test_accuracy', 'N/A')}")
    print("-" * 30)
``` 
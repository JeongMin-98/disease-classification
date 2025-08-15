# GradCAM 배치 실행 스크립트 사용법

이 폴더에는 8개 실험에 대해 fold별로 GradCAM을 생성하는 배치 스크립트들이 포함되어 있습니다.

## 실험 구성

8개 실험은 다음과 같이 구성됩니다:
- **이미지 크기**: 1024x1024, 224x224
- **모델**: ResNet18, VGG19-BN
- **데이터**: Foot, Hand

### 실험 목록
1. `1024_resnet18_foot` - 1024x1024 ResNet18 Foot
2. `1024_resnet18_hand` - 1024x1024 ResNet18 Hand
3. `1024_vgg19bn_foot` - 1024x1024 VGG19-BN Foot
4. `1024_vgg19bn_hand` - 1024x1024 VGG19-BN Hand
5. `224_resnet18_foot` - 224x224 ResNet18 Foot
6. `224_resnet18_hand` - 224x224 ResNet18 Hand
7. `224_vgg19bn_foot` - 224x224 VGG19-BN Foot
8. `224_vgg19bn_hand` - 224x224 VGG19-BN Hand

## 스크립트 종류

### 1. 전체 실험 일괄 실행
```bash
./run_batch_gradcam_all_experiments.sh
```
- 8개 실험을 순차적으로 실행
- 각 실험의 결과를 자동으로 수집하고 요약
- 전체 실행 결과를 `batch_gradcam_results/batch_execution_summary.txt`에 저장

### 2. 개별 실험 실행
```bash
./run_single_experiment_gradcam.sh <experiment_name>
```

**사용 예시:**
```bash
# 기본 설정으로 실행
./run_single_experiment_gradcam.sh 1024_resnet18_foot

# 커스텀 설정으로 실행
./run_single_experiment_gradcam.sh --cfg custom.yaml --log custom.log 224_vgg19bn_hand

# 도움말 보기
./run_single_experiment_gradcam.sh --help
```

**옵션:**
- `--cfg <path>`: 설정 파일 경로
- `--log <path>`: 로그 파일 경로
- `--output <path>`: 출력 디렉토리
- `--device <device>`: 사용할 디바이스 (기본값: cuda)
- `--seed <seed>`: 랜덤 시드 (기본값: 42)

### 3. 병렬 실행
```bash
./run_parallel_gradcam.sh [OPTIONS] <experiment_names...>
```

**사용 예시:**
```bash
# 2개 실험을 동시에 실행 (기본값)
./run_parallel_gradcam.sh 1024_resnet18_foot 224_vgg19bn_hand

# 최대 4개 작업을 동시에 실행
./run_parallel_gradcam.sh --max-jobs 4 1024_resnet18_foot 1024_vgg19bn_foot 224_resnet18_hand

# 도움말 보기
./run_parallel_gradcam.sh --help
```

**옵션:**
- `--max-jobs <N>`: 최대 동시 실행 작업 수 (기본값: 2)
- `--device <device>`: 사용할 디바이스 (기본값: cuda)
- `--seed <seed>`: 랜덤 시드 (기본값: 42)

### 4. SSH 연결 끊김 방지 (nohup) 실행
SSH 연결이 끊어져도 계속 실행되도록 nohup을 사용할 수 있습니다.

#### **전체 실험 nohup 실행**
```bash
./run_batch_gradcam_all_experiments_nohup.sh
```

#### **병렬 nohup 실행**
```bash
./run_parallel_gradcam_nohup.sh [OPTIONS] <experiment_names...>
```

**사용 예시:**
```bash
# 2개 실험을 nohup으로 병렬 실행
./run_parallel_gradcam_nohup.sh 1024_resnet18_foot 224_vgg19bn_hand

# 최대 4개 작업을 nohup으로 병렬 실행
./run_parallel_gradcam_nohup.sh --max-jobs 4 1024_resnet18_foot 1024_vgg19bn_foot 224_resnet18_hand
```

**nohup 옵션:**
- `--max-jobs <N>`: 최대 동시 실행 작업 수 (기본값: 2)
- `--device <device>`: 사용할 디바이스 (기본값: cuda)
- `--seed <seed>`: 랜덤 시드 (기본값: 42)
- `--output-dir <path>`: 출력 디렉토리 (기본값: batch_gradcam_results)
- `--log-dir <path>`: nohup 로그 디렉토리 (기본값: nohup_logs)

## 사전 준비사항

### 1. K-fold indices 파일 생성
각 실험마다 별도의 k-fold indices를 생성해야 합니다. 다음 스크립트를 사용하여 8개 실험에 대한 indices를 한 번에 생성할 수 있습니다:

```bash
cd /home/jmkim/disease-classification

# 8개 실험에 대한 k-fold indices 일괄 생성
./script/generate_all_kfold_indices.sh

# 또는 다른 시드와 출력 디렉토리 사용
./script/generate_all_kfold_indices.sh --seed 123 --output-dir custom_kfold_indices
```

**생성되는 파일들:**
- `kfold_indices/kfold_test_indices_foot_classifier_OA_Normal_1024_resnet18_kfold_seed_42.json`
- `kfold_indices/kfold_test_indices_ra_hand_classifier_OA_Normal_1024_resnet18_kfold_seed_42.json`
- `kfold_indices/kfold_test_indices_foot_classifier_OA_Normal_1024_vgg19bn_kfold_seed_42.json`
- `kfold_indices/kfold_test_indices_ra_hand_classifier_OA_Normal_1024_vgg19bn_kfold_seed_42.json`
- `kfold_indices/kfold_test_indices_foot_classifier_OA_Normal_224_resnet18_kfold_seed_42.json`
- `kfold_indices/kfold_test_indices_ra_hand_classifier_OA_Normal_224_resnet18_kfold_seed_42.json`
- `kfold_indices/kfold_test_indices_foot_classifier_OA_Normal_224_vgg19bn_kfold_seed_42.json`
- `kfold_indices/kfold_test_indices_ra_hand_classifier_OA_Normal_224_vgg19bn_kfold_seed_42.json`

**개별 실험 indices 생성:**
```bash
# 특정 실험만 indices 생성
python tool/extract_kfold_from_cfg.py --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_vgg19bn_kfold.yaml --log log/224/oa_normal_hand_vgg19bn_kfold.log --output_dir kfold_indices
```

### 2. 필요한 파일들 확인
- `kfold_test_indices_seed_42.json`: K-fold indices 파일
- `wandb/`: wandb 실행 결과 폴더 (모델 체크포인트 포함)
- 설정 파일들: `experiments/image_exp/` 하위의 YAML 파일들
- 로그 파일들: `log/` 하위의 로그 파일들

## 실행 순서

### 1단계: K-fold indices 생성
```bash
cd /home/jmkim/disease-classification

# 8개 실험에 대한 k-fold indices 일괄 생성
./script/generate_all_kfold_indices.sh
```

### 2단계: GradCAM 실행
다음 중 하나의 방법을 선택하여 실행:

## 실행 방법

### 방법 1: 전체 실험 일괄 실행 (권장)
```bash
cd /home/jmkim/disease-classification
./script/run_batch_gradcam_all_experiments.sh
```

### 방법 2: 개별 실험 실행
```bash
cd /home/jmkim/disease-classification

# 1024x1024 ResNet18 Foot 실험만 실행
./script/run_single_experiment_gradcam.sh 1024_resnet18_foot

# 224x224 VGG19-BN Hand 실험만 실행
./script/run_single_experiment_gradcam.sh 224_vgg19bn_hand
```

### 방법 3: 병렬 실행
```bash
cd /home/jmkim/disease-classification

# 1024 크기 실험들을 동시에 실행
./script/run_parallel_gradcam.sh 1024_resnet18_foot 1024_vgg19bn_foot

# 224 크기 실험들을 동시에 실행
./script/run_parallel_gradcam.sh 224_resnet18_foot 224_vgg19bn_foot
```

## 출력 구조

실행 후 다음과 같은 구조로 결과가 저장됩니다:

```
batch_gradcam_results/
├── 1024_resnet18_foot/
│   ├── fold_0/
│   │   ├── oa/
│   │   ├── normal/
│   │   └── misclassified/
│   ├── fold_1/
│   ├── ...
│   ├── summary.json
│   └── execution.log
├── 1024_resnet18_hand/
├── ...
├── batch_execution_summary.txt
└── parallel_execution_summary.txt (병렬 실행 시)
```

## 결과 확인

### 1. 개별 실험 결과
각 실험 폴더의 `summary.json`에서 다음 정보를 확인할 수 있습니다:
- 전체 이미지 수
- 전체 정확도
- Fold별 상세 결과

### 2. 전체 실행 요약
- `batch_execution_summary.txt`: 전체 일괄 실행 결과
- `parallel_execution_summary.txt`: 병렬 실행 결과

## 주의사항

1. **메모리 사용량**: 1024x1024 이미지는 224x224보다 메모리를 많이 사용합니다.
2. **GPU 메모리**: 여러 실험을 병렬로 실행할 때는 GPU 메모리 사용량을 고려하세요.
3. **wandb 폴더**: 모델 체크포인트가 `wandb/` 폴더에 있어야 합니다.
4. **K-fold indices**: 실행 전에 `extract_kfold_from_cfg.py`로 indices를 생성해야 합니다.

## 문제 해결

### 일반적인 오류들

1. **K-fold indices 파일을 찾을 수 없음**
   ```bash
   python tool/extract_kfold_from_cfg.py --cfg <config_file> --log <log_file>
   ```

2. **wandb 폴더를 찾을 수 없음**
   - `wandb/` 폴더가 존재하는지 확인
   - 모델 체크포인트가 올바른 위치에 있는지 확인

3. **설정 파일을 찾을 수 없음**
   - `experiments/image_exp/` 하위의 파일 경로 확인
   - 파일명이 정확한지 확인

4. **로그 파일을 찾을 수 없음**
   - `log/` 하위의 파일 경로 확인
   - 파일명이 정확한지 확인

## 모니터링 도구

GradCAM 실행 중인 작업들을 모니터링할 수 있는 다양한 도구들이 제공됩니다.

### 1. 실시간 진행 상황 모니터링
```bash
# 기본 모니터링 (10초마다 새로고침)
./monitor_gradcam_progress.sh

# 5초마다 새로고침
./monitor_gradcam_progress.sh --watch 5

# 실시간 로그와 프로세스 정보 포함
./monitor_gradcam_progress.sh --show-logs --show-processes
```

**기능:**
- 실험별 진행 상황 (완료/실행 중/대기 중/실패)
- 전체 진행률 표시
- GPU, 메모리, 디스크 사용량 모니터링
- 실시간 로그 표시 (옵션)
- 실행 중인 프로세스 정보 (옵션)

### 2. 상태 간단 확인
```bash
# 전체 상태 한 번만 확인
./check_gradcam_status.sh

# 다른 출력 디렉토리 확인
./check_gradcam_status.sh --output-dir custom_results
```

**기능:**
- 실험별 상태 요약
- 전체 진행률
- 실행 중인 프로세스 목록
- GPU 사용량

### 3. 실시간 로그 모니터링
```bash
# 모든 실행 중인 로그 실시간 표시
./tail_gradcam_logs.sh

# 특정 실험의 로그만 표시
./tail_gradcam_logs.sh 1024_resnet18_foot

# 최근 100줄만 표시 (실시간 tail 비활성화)
./tail_gradcam_logs.sh --lines 100 --no-follow
```

**기능:**
- 실행 중인 모든 로그 파일 자동 감지
- 특정 실험 로그만 모니터링
- 실시간 tail 또는 일회성 표시
- 여러 로그 동시 모니터링 가이드

### 4. nohup 실행 후 모니터링
nohup으로 실행한 작업들은 SSH 연결이 끊어져도 계속 실행됩니다.

#### **nohup 로그 확인**
```bash
# nohup 로그 디렉토리 확인
ls -la nohup_logs/

# 특정 실험의 nohup 로그 확인
tail -f nohup_logs/1024_resnet18_foot_nohup.log

# 모든 nohup 로그 동시 확인
tail -f nohup_logs/*_nohup.log
```

#### **실행 중인 프로세스 확인**
```bash
# PID 파일들 확인
ls -la nohup_logs/*_pid.txt

# 특정 실험의 PID 확인
cat nohup_logs/1024_resnet18_foot_pid.txt

# 프로세스 상태 확인
ps aux | grep batch_gradcam_kfold
```

#### **nohup 작업 관리**
```bash
# 특정 실험 중단
kill $(cat nohup_logs/1024_resnet18_foot_pid.txt)

# 모든 GradCAM 작업 중단
pkill -f batch_gradcam_kfold

# 작업 상태 확인
./check_gradcam_status.sh
```

## 성능 최적화 팁

1. **병렬 실행**: `--max-jobs` 옵션으로 동시 실행 작업 수 조절
2. **GPU 메모리**: 큰 이미지(1024x1024)는 동시 실행 수를 줄이세요
3. **배치 크기**: `cfg.TEST.BATCH_SIZE_PER_GPU`를 GPU 메모리에 맞게 조절
4. **모니터링**: `monitor_gradcam_progress.sh`로 리소스 사용량 실시간 확인 
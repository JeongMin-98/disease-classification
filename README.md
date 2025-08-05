## Installation

### Conda Environment Setup

프로젝트를 실행하기 위한 conda 환경을 설정합니다.

```bash
# conda 환경 생성
conda env create -f environment.yml

# 환경 활성화
conda activate disease-classification
```

### Manual Installation (Recommend)

수동으로 설치하는 경우:

```bash
# PyTorch 설치 (CUDA 12.1 기준)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia


# 기본 라이브러리 설치
pip install numpy scipy scikit-learn matplotlib pillow opencv-python tqdm wandb yacs grad-cam pandas seaborn

# 추가 라이브러리 설치
pip install albumentations timm tensorboard tensorboardX thop pyyaml
```

## Usage

### Training

`tool/train.py`를 사용하여 모델을 학습할 수 있습니다.

#### 기본 사용법
```bash
python tool/train.py --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml --seed 42
```

#### 매개변수
- `--cfg`: 실험에 사용할 config YAML 파일 경로 (기본값: `experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml`)
- `--seed`: 랜덤 시드 (기본값: 42)

#### 주요 기능
- **데이터 균등화**: 클래스별 샘플 수를 균등하게 조정
- **균형 잡힌 샘플링**: 각 배치에서 클래스별 균등한 샘플링
- **층화 분할**: 훈련/검증/테스트 세트에서 클래스 비율 유지
- **WandB 통합**: 실험 추적 및 로깅
- **VGG19 모델**: 사전 훈련된 VGG19 모델 사용

#### 출력
- 모델 체크포인트가 `outputs/` 디렉토리에 저장됩니다
- WandB를 통해 학습 과정이 추적됩니다
- 최종 테스트 정확도가 콘솔에 출력됩니다

#!/bin/bash

# 병렬 실험 실행에 필요한 의존성 패키지 설치 스크립트

echo "=== 병렬 실험 실행을 위한 의존성 패키지 설치 ==="

# Python 패키지 설치
echo "Python 패키지 설치 중..."

# GPUtil 설치 (GPU 모니터링용)
pip install GPUtil

# psutil 설치 (시스템 모니터링용)
pip install psutil

# 기타 필요한 패키지들
pip install numpy pandas

echo "=== 의존성 패키지 설치 완료 ==="
echo "이제 다음 명령어로 병렬 실험을 실행할 수 있습니다:"
echo "python run_parallel_7fold_experiments.py --max-workers 2 --seed 42" 
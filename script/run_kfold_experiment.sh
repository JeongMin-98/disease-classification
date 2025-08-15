#!/bin/bash

# 7-fold 실험 실행 스크립트
# 사용법: ./run_kfold_experiment.sh [config_file] [seed]

# 기본값 설정
CONFIG_FILE=${1:-"experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml"}
SEED=${2:-42}

echo "=== 7-fold 실험 시작 ==="
echo "설정 파일: $CONFIG_FILE"
echo "시드: $SEED"
echo "=========================="

# 실험 시작
python tool/train_kfold.py --cfg $CONFIG_FILE --seed $SEED

echo "=== 7-fold 실험 완료 ==="
echo "결과는 cfg.OUTPUT_DIR에 저장되었습니다."
echo "전체 결과 요약은 kfold_results.json 파일을 확인하세요."
echo "각 fold의 결과는 fold_0, fold_1, ..., fold_6 디렉토리에 저장됩니다." 
#!/bin/bash

# GradCAM 배치 실행 스크립트 사용 예시

echo "=== GradCAM 배치 실행 스크립트 사용 예시 ==="
echo ""

# 현재 디렉토리 확인
echo "현재 디렉토리: $(pwd)"
echo ""

# 1. 전체 실험 일괄 실행 (권장)
echo "1. 전체 실험 일괄 실행:"
echo "   ./run_batch_gradcam_all_experiments.sh"
echo ""

# 2. 개별 실험 실행
echo "2. 개별 실험 실행:"
echo "   # 1024x1024 ResNet18 Foot"
echo "   ./run_single_experiment_gradcam.sh 1024_resnet18_foot"
echo "   "
echo "   # 224x224 VGG19-BN Hand"
echo "   ./run_single_experiment_gradcam.sh 224_vgg19bn_hand"
echo ""

# 3. 병렬 실행
echo "3. 병렬 실행:"
echo "   # 1024 크기 실험들을 동시에 실행"
echo "   ./run_parallel_gradcam.sh 1024_resnet18_foot 1024_vgg19bn_foot"
echo "   "
echo "   # 최대 4개 작업을 동시에 실행"
echo "   ./run_parallel_gradcam.sh --max-jobs 4 1024_resnet18_foot 1024_vgg19bn_foot 224_resnet18_hand"
echo ""

# 4. 도움말 보기
echo "4. 도움말 보기:"
echo "   ./run_single_experiment_gradcam.sh --help"
echo "   ./run_parallel_gradcam.sh --help"
echo ""

# 5. 사전 준비사항
echo "5. 사전 준비사항:"
echo "   # K-fold indices 파일 생성 (필수)"
echo "   cd /home/jmkim/disease-classification"
echo "   "
echo "   # 8개 실험에 대한 k-fold indices 일괄 생성 (권장)"
echo "   ./script/generate_all_kfold_indices.sh"
echo "   "
echo "   # 또는 개별 실험 indices 생성"
echo "   python tool/extract_kfold_from_cfg.py --cfg experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal_224_vgg19bn_kfold.yaml --log log/224/oa_normal_hand_vgg19bn_kfold.log --output_dir kfold_indices"
echo ""

# 6. 사용 가능한 실험 목록
echo "6. 사용 가능한 실험 목록:"
echo "   - 1024_resnet18_foot     (1024x1024 ResNet18 Foot)"
echo "   - 1024_resnet18_hand     (1024x1024 ResNet18 Hand)"
echo "   - 1024_vgg19bn_foot      (1024x1024 VGG19-BN Foot)"
echo "   - 1024_vgg19bn_hand      (1024x1024 VGG19-BN Hand)"
echo "   - 224_resnet18_foot      (224x224 ResNet18 Foot)"
echo "   - 224_resnet18_hand      (224x224 ResNet18 Hand)"
echo "   - 224_vgg19bn_foot       (224x224 VGG19-BN Foot)"
echo "   - 224_vgg19bn_hand       (224x224 VGG19-BN Hand)"
echo ""

# 7. 모니터링 도구
echo "7. 모니터링 도구:"
echo "   # 실시간 진행 상황 모니터링"
echo "   ./monitor_gradcam_progress.sh"
echo "   "
echo "   # 상태 간단 확인"
echo "   ./check_gradcam_status.sh"
echo "   "
echo "   # 실시간 로그 모니터링"
echo "   ./tail_gradcam_logs.sh"
echo ""

# 8. SSH 연결 끊김 방지 (nohup)
echo "8. SSH 연결 끊김 방지 (nohup):"
echo "   # 전체 실험 nohup 실행"
echo "   ./run_batch_gradcam_all_experiments_nohup.sh"
echo "   "
echo "   # 병렬 nohup 실행"
echo "   ./run_parallel_gradcam_nohup.sh 1024_resnet18_foot 224_vgg19bn_hand"
echo "   "
echo "   # nohup 로그 확인"
echo "   tail -f nohup_logs/*_nohup.log"
echo ""

echo "=== 자세한 사용법은 README_gradcam_batch.md를 참조하세요 ===" 
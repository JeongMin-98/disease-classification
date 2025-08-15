import wandb
import random

def simple_train():
    """
    간단한 sweep 예시 함수
    """
    # wandb 초기화
    wandb.init()
    
    # sweep에서 설정된 하이퍼파라미터 가져오기
    config = wandb.config
    
    # 간단한 시뮬레이션 (실제로는 모델 훈련)
    epochs = config.epochs
    lr = config.learning_rate
    batch_size = config.batch_size
    
    # 가상의 훈련 과정
    for epoch in range(epochs):
        # 가상의 손실과 정확도 계산
        loss = 1.0 / (epoch + 1) + random.uniform(0, 0.1)
        accuracy = 0.8 + (epoch * 0.02) + random.uniform(-0.05, 0.05)
        
        # wandb에 로깅
        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": lr,
            "batch_size": batch_size
        })
    
    # 최종 정확도를 메트릭으로 사용
    final_accuracy = accuracy
    wandb.log({"final_accuracy": final_accuracy})
    
    return final_accuracy

def main():
    """
    간단한 sweep 설정 및 실행
    """
    # Sweep 설정
    sweep_config = {
        'method': 'grid',  # 'grid', 'random', 'bayes' 중 선택
        'name': 'simple_example_sweep',
        'metric': {
            'name': 'final_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [0.001, 0.01, 0.1]
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'epochs': {
                'values': [10, 20, 30]
            }
        }
    }
    
    # Sweep 생성
    sweep_id = wandb.sweep(sweep_config, project="sweep-example")
    print(f"Sweep ID: {sweep_id}")
    
    # Sweep agent 시작 (9번의 실험 실행)
    wandb.agent(sweep_id, simple_train, count=9)

if __name__ == "__main__":
    main() 
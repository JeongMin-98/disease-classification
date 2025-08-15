"""
Wandb Sweep 설정 파일
다양한 실험에 대한 sweep configuration들을 정의합니다.
"""

# 기본 sweep 설정
BASE_SWEEP_CONFIG = {
    'method': 'bayes',
    'name': 'disease_classification_sweep',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    }
}

# 기본 하이퍼파라미터 설정
BASE_PARAMETERS = {
    'seed': {
        'values': [42, 123, 456]
    },
    'cfg': {
        'values': [
            'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml',
            'experiments/image_exp/foot_xray/foot_xray_classifier.yaml',
            'experiments/image_exp/foot/foot_classifier.yaml'
        ]
    },
    'TRAIN.LR': {
        'min': 0.0001,
        'max': 0.01,
        'distribution': 'log_uniform'
    },
    'TRAIN.BATCH_SIZE_PER_GPU': {
        'values': [16, 32, 64]
    },
    'TRAIN.OPTIMIZER': {
        'values': ['adam', 'sgd']
    },
    'TRAIN.SCHEDULER': {
        'values': ['cosine', 'step', None]
    },
    'TRAIN.END_EPOCH': {
        'values': [50, 100]
    },
    'DATASET.TARGET_COUNT_PER_CLASS': {
        'values': [100, 200, 500, None]
    },
    'TRAIN.USE_BALANCED_SAMPLING': {
        'values': [True, False]
    },
    'TRAIN.SAMPLING_TYPE': {
        'values': ['balanced', 'stratified']
    }
}

# 빠른 테스트용 설정 (적은 실험 수)
QUICK_TEST_CONFIG = {
    'method': 'grid',
    'name': 'quick_test_sweep',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'seed': {
            'values': [42]
        },
        'cfg': {
            'values': [
                'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml'
            ]
        },
        'TRAIN.LR': {
            'values': [0.001, 0.01]
        },
        'TRAIN.BATCH_SIZE_PER_GPU': {
            'values': [16, 32]
        },
        'TRAIN.OPTIMIZER': {
            'values': ['adam']
        },
        'TRAIN.SCHEDULER': {
            'values': ['cosine', None]
        },
        'TRAIN.END_EPOCH': {
            'values': [20]
        },
        'DATASET.TARGET_COUNT_PER_CLASS': {
            'values': [100, None]
        },
        'TRAIN.USE_BALANCED_SAMPLING': {
            'values': [True]
        },
        'TRAIN.SAMPLING_TYPE': {
            'values': ['balanced']
        }
    }
}

# 학습률 최적화 전용 설정
LEARNING_RATE_SWEEP_CONFIG = {
    'method': 'bayes',
    'name': 'learning_rate_optimization',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5
    },
    'parameters': {
        'seed': {
            'values': [42, 123]
        },
        'cfg': {
            'values': [
                'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml'
            ]
        },
        'TRAIN.LR': {
            'min': 0.00001,
            'max': 0.1,
            'distribution': 'log_uniform'
        },
        'TRAIN.BATCH_SIZE_PER_GPU': {
            'values': [32]
        },
        'TRAIN.OPTIMIZER': {
            'values': ['adam']
        },
        'TRAIN.SCHEDULER': {
            'values': ['cosine']
        },
        'TRAIN.END_EPOCH': {
            'values': [50]
        },
        'DATASET.TARGET_COUNT_PER_CLASS': {
            'values': [200]
        },
        'TRAIN.USE_BALANCED_SAMPLING': {
            'values': [True]
        },
        'TRAIN.SAMPLING_TYPE': {
            'values': ['balanced']
        }
    }
}

# 배치 크기 최적화 전용 설정
BATCH_SIZE_SWEEP_CONFIG = {
    'method': 'grid',
    'name': 'batch_size_optimization',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'seed': {
            'values': [42]
        },
        'cfg': {
            'values': [
                'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml'
            ]
        },
        'TRAIN.LR': {
            'values': [0.001]
        },
        'TRAIN.BATCH_SIZE_PER_GPU': {
            'values': [8, 16, 32, 64, 128]
        },
        'TRAIN.OPTIMIZER': {
            'values': ['adam']
        },
        'TRAIN.SCHEDULER': {
            'values': ['cosine']
        },
        'TRAIN.END_EPOCH': {
            'values': [30]
        },
        'DATASET.TARGET_COUNT_PER_CLASS': {
            'values': [200]
        },
        'TRAIN.USE_BALANCED_SAMPLING': {
            'values': [True]
        },
        'TRAIN.SAMPLING_TYPE': {
            'values': ['balanced']
        }
    }
}

# 데이터 균등화 효과 테스트 설정
DATA_BALANCING_SWEEP_CONFIG = {
    'method': 'grid',
    'name': 'data_balancing_test',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'seed': {
            'values': [42, 123]
        },
        'cfg': {
            'values': [
                'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml'
            ]
        },
        'TRAIN.LR': {
            'values': [0.001]
        },
        'TRAIN.BATCH_SIZE_PER_GPU': {
            'values': [32]
        },
        'TRAIN.OPTIMIZER': {
            'values': ['adam']
        },
        'TRAIN.SCHEDULER': {
            'values': ['cosine']
        },
        'TRAIN.END_EPOCH': {
            'values': [50]
        },
        'DATASET.TARGET_COUNT_PER_CLASS': {
            'values': [50, 100, 200, 500, None]
        },
        'TRAIN.USE_BALANCED_SAMPLING': {
            'values': [True, False]
        },
        'TRAIN.SAMPLING_TYPE': {
            'values': ['balanced', 'stratified']
        }
    }
}

# 전체 하이퍼파라미터 최적화 설정 (대규모 실험)
FULL_OPTIMIZATION_CONFIG = {
    'method': 'bayes',
    'name': 'full_hyperparameter_optimization',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 15
    },
    'parameters': {
        'seed': {
            'values': [42, 123, 456, 789, 999]
        },
        'cfg': {
            'values': [
                'experiments/image_exp/ra_hand/ra_hand_classifier_OA_Normal.yaml',
                'experiments/image_exp/foot_xray/foot_xray_classifier.yaml',
                'experiments/image_exp/foot/foot_classifier.yaml'
            ]
        },
        'TRAIN.LR': {
            'min': 0.00001,
            'max': 0.1,
            'distribution': 'log_uniform'
        },
        'TRAIN.BATCH_SIZE_PER_GPU': {
            'values': [8, 16, 32, 64, 128]
        },
        'TRAIN.OPTIMIZER': {
            'values': ['adam', 'sgd', 'adamw']
        },
        'TRAIN.SCHEDULER': {
            'values': ['cosine', 'step', 'exponential', None]
        },
        'TRAIN.END_EPOCH': {
            'values': [30, 50, 100, 150]
        },
        'DATASET.TARGET_COUNT_PER_CLASS': {
            'values': [50, 100, 200, 500, 1000, None]
        },
        'TRAIN.USE_BALANCED_SAMPLING': {
            'values': [True, False]
        },
        'TRAIN.SAMPLING_TYPE': {
            'values': ['balanced', 'stratified']
        },
        'TRAIN.WEIGHT_DECAY': {
            'min': 0.0,
            'max': 0.01,
            'distribution': 'uniform'
        }
    }
}

# 설정 딕셔너리 (쉽게 접근하기 위해)
SWEEP_CONFIGS = {
    'base': BASE_SWEEP_CONFIG,
    'quick_test': QUICK_TEST_CONFIG,
    'learning_rate': LEARNING_RATE_SWEEP_CONFIG,
    'batch_size': BATCH_SIZE_SWEEP_CONFIG,
    'data_balancing': DATA_BALANCING_SWEEP_CONFIG,
    'full_optimization': FULL_OPTIMIZATION_CONFIG
}

def get_sweep_config(config_name='base', custom_parameters=None):
    """
    sweep 설정을 가져오는 함수
    
    Args:
        config_name (str): 설정 이름 ('base', 'quick_test', 'learning_rate', etc.)
        custom_parameters (dict): 커스텀 파라미터 (기존 파라미터를 덮어씀)
    
    Returns:
        dict: sweep 설정
    """
    if config_name not in SWEEP_CONFIGS:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(SWEEP_CONFIGS.keys())}")
    
    config = SWEEP_CONFIGS[config_name].copy()
    
    if custom_parameters:
        config['parameters'].update(custom_parameters)
    
    return config

def create_custom_sweep_config(method='bayes', name='custom_sweep', 
                              metric_name='test_accuracy', parameters=None):
    """
    커스텀 sweep 설정을 생성하는 함수
    
    Args:
        method (str): sweep 방법 ('grid', 'random', 'bayes')
        name (str): sweep 이름
        metric_name (str): 최적화할 메트릭 이름
        parameters (dict): 하이퍼파라미터 설정
    
    Returns:
        dict: 커스텀 sweep 설정
    """
    config = {
        'method': method,
        'name': name,
        'metric': {
            'name': metric_name,
            'goal': 'maximize'
        }
    }
    
    if method == 'bayes':
        config['early_terminate'] = {
            'type': 'hyperband',
            'min_iter': 10
        }
    
    if parameters:
        config['parameters'] = parameters
    else:
        config['parameters'] = BASE_PARAMETERS.copy()
    
    return config 
"""
경로, 샘플링 주파수, 하이퍼파라미터와 같은 상수 관리 파일
"""

import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FEMTO_Bearing')
BEARING_DIR = os.path.join(BASE_DIR, 'Cond1_1')

SAMPLING_FREQ = 25600 # 25.6 kHz
STFT_NPERSEG = 512 # 윈도우 크기
STFT_NOVERLAP = 448 # 오버랩 정도 (87.5%)

# 주파수 대역 정의 (Hz)
FREQ_BANDS = {
    'low': (0, 1000), # 회전 관련
    'mid': (1000, 5000), # 결함 초기
    'high': (5000, 12800) # 마모/이상
}

N_EARLY_FILES = 100 # 초기 파일 수 (정상)
N_LATE_FILES = 100 # 마지막 파일 수 (이상)
LABEL_NORMAL = 0
LABEL_DEGRADED = 1

# 랜덤 seed 값으로 해당 숫자를 넣으면 데이터를 일관적으로 추출하는 것이 보장됨
RANDOM_STATE = 42

# PCA
PCA_N_COMPONENTS = 3

# SVM
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
}

# Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

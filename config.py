"""
경로, 샘플링 주파수, 하이퍼파라미터와 같은 상수 관리 파일
"""

import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FEMTO_Bearing')

# 학습/테스트 조건 (리스트에 조건명을 추가·삭제하면 수량 변경 가능)
# 예: TRAIN_CONDITIONS = ['Cond1_1']  /  ['Cond1_1', 'Cond1_2', 'Cond1_5']
TRAIN_CONDITIONS = ['Cond1_1', 'Cond1_2', 'Cond1_3']
TEST_CONDITIONS = ['Cond1_4', 'Cond1_5', 'Cond1_6']

# 단일 조건 로더 기본값 (하위 호환)
BEARING_DIR = os.path.join(BASE_DIR, TRAIN_CONDITIONS[0])

SAMPLING_FREQ = 25600 # 25.6 kHz
STFT_NPERSEG = 512 # 윈도우 크기
STFT_NOVERLAP = 448 # 오버랩 정도 (87.5%)

# 주파수 대역 정의 (Hz)
FREQ_BANDS = {
    'low': (0, 1000), # 회전 관련 + 결함 기본 주파수
    'mid': (1000, 5000), # 구조 공진 대역 
    'high': (5000, 12800) # 고주파 대역 (초기 미세 결함)
}

N_EARLY_FILES = 100 # 초기 파일 수 (정상)
N_LATE_FILES = 100 # 마지막 파일 수 (이상)
LABEL_NORMAL = 0
LABEL_DEGRADED = 1

# 랜덤 seed 값으로 해당 숫자를 넣으면 데이터를 일관적으로 추출하는 것이 보장됨
RANDOM_STATE = 42

# PCA
PCA_N_COMPONENTS = 3

# SVM 기본 파라미터 (탐색하지 않는 고정값)
SVM_BASE_PARAMS = {
    'kernel': 'rbf',
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
}

# SVM 홀드아웃 검증에서 탐색할 하이퍼파라미터 후보
SVM_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.001],
}

# SVM 7:3 고정 파라미터 (Validation 없이 사용)
SVM_FIXED_PARAMS = {
    **SVM_BASE_PARAMS,
    'C': 1.0,
    'gamma': 'scale',
}

# Random Forest 기본 파라미터 (탐색하지 않는 고정값)
RF_BASE_PARAMS = {
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

# Random Forest 홀드아웃 검증에서 탐색할 하이퍼파라미터 후보
RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

# Random Forest 7:3 고정 파라미터 (Validation 없이 사용)
RF_FIXED_PARAMS = {
    **RF_BASE_PARAMS,
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
}

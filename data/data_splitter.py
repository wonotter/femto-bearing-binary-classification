import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from config import RANDOM_STATE

# Train:Validation:Test = 6:2:2
def split_622(X, y):
    # 전체 데이터에서 Test 먼저 분리
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Test를 제외하고 남은 80% 데이터에서 Train과 Validation로 분리 
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Train:Test = 7:3
def split_73(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test

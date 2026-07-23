import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('..')
from config import SVM_PARAMS
from utils.evaluation import print_evaluation_report

# SVM 분류기 래퍼 클래스: StandardScaler를 내장하여 자동으로 정규화 진행
class SVMClassifier:
    
    def __init__(self, params=None):
        self.params = params or SVM_PARAMS
        self.scaler = StandardScaler()
        self.model = SVC(**self.params)

    # 모델 학습 진행
    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        print("SVM 학습 완료")
        print(f"Support Vector 개수: {self.model.n_support_}")
        return self
    
    # 예측 수행
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    # 평가 수행
    def evaluate(self, X, y, set_name='Test'):
        y_pred = self.predict(X)
        print(f"\n SVM - {set_name} 평가")
        metrics = print_evaluation_report(y, y_pred)
        return y_pred, metrics
    
def run_svm_622(X_train, X_val, X_test, y_train, y_val, y_test):
    print("=" * 60)
    print("SVM (RBF Kernel) - 6:2:2 분할")
    print("=" * 60)

    clf = SVMClassifier()
    clf.train(X_train, y_train)
    y_val_pred, val_metrics = clf.evaluate(X_val, y_val, set_name='Validation')
    y_test_pred, test_metrics = clf.evaluate(X_test, y_test, set_name='Test')

    return {
        'model': clf,
        'val': {'y_true': y_val, 'y_pred': y_val_pred, 'metrics': val_metrics},
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
    }

def run_svm_73(X_train, X_test, y_train, y_test):
    print("=" * 60)
    print("SVM (RBF Kernel) - 7:3 분할")
    print("=" * 60)

    clf = SVMClassifier()
    clf.train(X_train, y_train)
    y_test_pred, test_metrics = clf.evaluate(X_test, y_test, set_name='Test')

    return {
        'model': clf,
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
    }

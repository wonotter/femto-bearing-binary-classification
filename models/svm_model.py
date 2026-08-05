import itertools
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('..')
from config import SVM_BASE_PARAMS, SVM_PARAM_GRID, SVM_FIXED_PARAMS
from utils.evaluation import print_evaluation_report

class SVMClassifier:

    def __init__(self, params=None):
        self.params = params or SVM_BASE_PARAMS
        self.scaler = StandardScaler()
        self.model = SVC(**self.params)

    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def decision_scores(self, X):
        """ROC-AUC 계산용 연속 점수 (decision_function)."""
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def evaluate(self, X, y, set_name='Test'):
        y_pred = self.predict(X)
        print(f"\n SVM - {set_name} 평가")
        metrics = print_evaluation_report(y, y_pred)
        return y_pred, metrics


def run_svm_622(X_train, X_val, X_test, y_train, y_val, y_test):
    print("=" * 60)
    print("2단계: SVM 하이퍼파라미터 탐색 (선택된 축 고정)")
    print("SVM (RBF Kernel) - 6:2:2 홀드아웃 검증")
    print("=" * 60)

    # 탐색할 파라미터의 키와 후보값 리스트를 분리
    param_keys = list(SVM_PARAM_GRID.keys())
    param_values = list(SVM_PARAM_GRID.values())

    best_f1 = -1
    best_params = None
    search_log = []

    # 모든 파라미터 조합을 순회하며 Validation 세트로 비교
    for combo in itertools.product(*param_values):
        trial_params = dict(zip(param_keys, combo))
        merged_params = {**SVM_BASE_PARAMS, **trial_params}

        clf = SVMClassifier(params=merged_params)
        clf.train(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)

        search_log.append({'params': trial_params, 'val_f1': val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = trial_params

    # 탐색 결과 출력
    print(f"\n총 {len(search_log)}개 조합 탐색 완료")
    print(f"{'C':>8} | {'gamma':>8} | {'Val F1':>8}")
    print("-" * 32)
    for log in search_log:
        p = log['params']
        marker = " ★" if p == best_params else ""
        print(f"{str(p['C']):>8} | {str(p['gamma']):>8} | {log['val_f1']:>8.4f}{marker}")
    print(f"\n최적 파라미터: {best_params} (Val F1: {best_f1:.4f})")

    # 최적 파라미터로 최종 모델 학습 및 Test 평가
    final_params = {**SVM_BASE_PARAMS, **best_params}
    final_clf = SVMClassifier(params=final_params)
    final_clf.train(X_train, y_train)

    y_val_pred, val_metrics = final_clf.evaluate(X_val, y_val, set_name='Validation')
    y_test_pred, test_metrics = final_clf.evaluate(X_test, y_test, set_name='Test')

    return {
        'model': final_clf,
        'best_params': best_params,
        'search_log': search_log,
        'val': {'y_true': y_val, 'y_pred': y_val_pred, 'metrics': val_metrics},
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
    }


def run_svm_73(X_train, X_test, y_train, y_test):
    print("=" * 60)
    print("SVM (RBF Kernel) - 7:3 분할 (고정 파라미터)")
    print("=" * 60)

    clf = SVMClassifier(params=SVM_FIXED_PARAMS)
    clf.train(X_train, y_train)
    y_test_pred, test_metrics = clf.evaluate(X_test, y_test, set_name='Test')

    return {
        'model': clf,
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
    }

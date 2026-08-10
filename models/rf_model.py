import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import sys
sys.path.append('..')
from config import RF_BASE_PARAMS, RF_PARAM_GRID, RF_FIXED_PARAMS
from utils.evaluation import print_evaluation_report

class RFClassifier:

    def __init__(self, params=None):
        self.params = params or RF_BASE_PARAMS
        self.model = RandomForestClassifier(**self.params)
        self.feature_names = None

    def train(self, X_train, y_train, feature_names=None):
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y, set_name='Test'):
        y_pred = self.predict(X)
        # 양성(열화) 클래스 확률로 AUC 계산
        y_score = self.predict_proba(X)[:, 1]
        print(f"\n Random Forest - {set_name} 평가")
        metrics = print_evaluation_report(y, y_pred, y_score=y_score)
        return y_pred, metrics

    def print_feature_importance(self, top_n=10):
        if self.feature_names is None:
            print("특징 이름이 설정되지 않았습니다")
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        print(f"\n Feature Importance - Top {top_n}")
        for rank, idx in enumerate(indices, 1):
            name = self.feature_names[idx]
            imp = importances[idx]
            print(f"{rank:>3} | {name:<24} | {imp:.4f}")
        print("-----------------------------------------")


def run_rf_622(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    print("=" * 60)
    print("Random Forest - 6:2:2 홀드아웃 검증")
    print("=" * 60)

    param_keys = list(RF_PARAM_GRID.keys())
    param_values = list(RF_PARAM_GRID.values())

    best_f1 = -1
    best_params = None
    search_log = []

    for combo in itertools.product(*param_values):
        trial_params = dict(zip(param_keys, combo))
        merged_params = {**RF_BASE_PARAMS, **trial_params}

        clf = RFClassifier(params=merged_params)
        clf.train(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)

        search_log.append({'params': trial_params, 'val_f1': val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = trial_params

    print(f"\n총 {len(search_log)}개 조합 탐색 완료")
    print(f"{'n_est':>6} | {'max_d':>6} | {'min_sp':>6} | {'Val F1':>8}")
    print("-" * 38)
    for log in search_log:
        p = log['params']
        marker = " ★" if p == best_params else ""
        print(
            f"{str(p['n_estimators']):>6} | "
            f"{str(p['max_depth']):>6} | "
            f"{str(p['min_samples_split']):>6} | "
            f"{log['val_f1']:>8.4f}{marker}"
        )
    print(f"\n최적 파라미터: {best_params} (Val F1: {best_f1:.4f})")

    final_params = {**RF_BASE_PARAMS, **best_params}
    final_clf = RFClassifier(params=final_params)
    final_clf.train(X_train, y_train, feature_names=feature_names)

    y_val_pred, val_metrics = final_clf.evaluate(X_val, y_val, set_name='Validation')
    y_test_pred, test_metrics = final_clf.evaluate(X_test, y_test, set_name='Test')
    final_clf.print_feature_importance(top_n=10)

    return {
        'model': final_clf,
        'best_params': best_params,
        'search_log': search_log,
        'val': {'y_true': y_val, 'y_pred': y_val_pred, 'metrics': val_metrics},
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
        'feature_importances': final_clf.model.feature_importances_,
        'feature_names': feature_names,
    }


def run_rf_73(X_train, X_test, y_train, y_test, feature_names):
    print("=" * 60)
    print("Random Forest - 7:3 분할 (고정 파라미터)")
    print("=" * 60)

    clf = RFClassifier(params=RF_FIXED_PARAMS)
    clf.train(X_train, y_train, feature_names=feature_names)
    y_test_pred, test_metrics = clf.evaluate(X_test, y_test, set_name='Test')
    clf.print_feature_importance(top_n=10)

    return {
        'model': clf,
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
        'feature_importances': clf.model.feature_importances_,
        'feature_names': feature_names,
    }

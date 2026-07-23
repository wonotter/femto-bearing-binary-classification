import numpy as np
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('..')
from config import RF_PARAMS
from utils.evaluation import print_evaluation_report

# Random Forest 분류기 래퍼 클래스: 특징 중요도 분석 기능 포함
class RFClassifier:
    
    def __init__(self, params=None):
        self.params = params or RF_PARAMS
        self.model = RandomForestClassifier(**self.params)
        self.feature_names = None

    def train(self, X_train, y_train, feature_names=None):
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        print("Random Forest 학습 완료")
        print(f"트리 수: {self.model.n_estimators}")
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y, set_name='Test'):
        y_pred = self.predict(X)
        print(f"\n Random Forest - {set_name} 평가")
        metrics = print_evaluation_report(y, y_pred)
        return y_pred, metrics
    
    # 특징 중요도 상위 n개 출력
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
    print("=" * 50)
    print("Random Forest - 6:2:2 분할")
    print("=" * 50)

    clf = RFClassifier()
    clf.train(X_train, y_train, feature_names=feature_names)
    y_val_pred, val_metrics = clf.evaluate(X_val, y_val, set_name='Validation')
    y_test_pred, test_metrics = clf.evaluate(X_test, y_test, set_name='Test')
    clf.print_feature_importance(top_n=10)

    return {
        'model': clf,
        'val': {'y_true': y_val, 'y_pred': y_val_pred, 'metrics': val_metrics},
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
        'feature_importances': clf.model.feature_importances_,
        'feature_names': feature_names,
    }

def run_rf_73(X_train, X_test, y_train, y_test, feature_names):
    print("=" * 50)
    print("Random Forest - 7:3 분할")
    print("=" * 50)

    clf = RFClassifier()
    clf.train(X_train, y_train, feature_names=feature_names)
    y_test_pred, test_metrics = clf.evaluate(X_test, y_test, set_name='Test')
    clf.print_feature_importance(top_n=10)

    return {
        'model': clf,
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
        'feature_importances': clf.model.feature_importances_,
        'feature_names': feature_names,
    }

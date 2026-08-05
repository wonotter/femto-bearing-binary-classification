"""
1단계 축 선택: H-only / V-only / Both를 같은 고정 파라미터 모델로 학습하고,
Validation F1과 ROC-AUC로 축을 결정한다.

2단계(하이퍼파라미터 탐색)는 선택된 축 고정 후 main/모델 러너에서 수행한다.
7:3 등 Validation이 없는 설정은 6:2:2에서 고른 축을 재사용한다.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from config import (
    PCA_N_COMPONENTS,
    SVM_FIXED_PARAMS,
    RF_FIXED_PARAMS,
    AXIS_CANDIDATES,
)
from features.feature_extractor import apply_pca, transform_pca
from models.svm_model import SVMClassifier
from models.rf_model import RFClassifier

AXIS_LABEL = {
    'h': '수평(h)',
    'v': '수직(v)',
    'both': '수평+수직(both)',
}


def filter_features_by_axis(X, feature_names, axis):
    """선택한 축의 특징만 남긴다. axis: 'h' | 'v' | 'both'"""
    if axis == 'both':
        return X, list(feature_names)

    if axis not in ('h', 'v'):
        raise ValueError(f"axis는 'h', 'v', 'both' 중 하나여야 합니다. 입력={axis}")

    prefix = f'{axis}_'
    keep = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
    if not keep:
        raise ValueError(f"'{prefix}'로 시작하는 특징이 없습니다. names={feature_names}")

    return X[:, keep], [feature_names[i] for i in keep]


def _safe_roc_auc(y_true, scores):
    """클래스가 한쪽에만 있으면 0.5를 반환."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def _val_f1_auc_fixed_models(X_train_pca, X_val_pca, y_train, y_val):
    """
    축 비교용: 고정 파라미터 SVM/RF의
    Validation F1, ROC-AUC를 계산한다.
    """
    svm = SVMClassifier(params=SVM_FIXED_PARAMS)
    svm.train(X_train_pca, y_train)
    svm_pred = svm.predict(X_val_pca)
    svm_f1 = float(f1_score(y_val, svm_pred))
    svm_auc = _safe_roc_auc(y_val, svm.decision_scores(X_val_pca))

    rf = RFClassifier(params=RF_FIXED_PARAMS)
    rf.train(X_train_pca, y_train)
    rf_pred = rf.predict(X_val_pca)
    rf_f1 = float(f1_score(y_val, rf_pred))
    rf_auc = _safe_roc_auc(y_val, rf.positive_scores(X_val_pca))

    # 축 점수: SVM/RF의 F1·AUC 평균 (네 값의 산술평균)
    mean_f1 = (svm_f1 + rf_f1) / 2.0
    mean_auc = (svm_auc + rf_auc) / 2.0
    combined = (mean_f1 + mean_auc) / 2.0

    return {
        'svm_val_f1': svm_f1,
        'svm_val_auc': svm_auc,
        'rf_val_f1': rf_f1,
        'rf_val_auc': rf_auc,
        'mean_val_f1': mean_f1,
        'mean_val_auc': mean_auc,
        'combined_score': combined,
    }


def select_axis_by_validation(
    X_train,
    X_val,
    y_train,
    y_val,
    feature_names,
    n_components=PCA_N_COMPONENTS,
    candidates=None,
):
    """
    H / V / Both 각각에 대해
      특징 필터 → Train PCA → 고정 파라미터 학습 → Val F1·AUC
    를 계산하고, combined_score = (평균 F1 + 평균 AUC) / 2 가
    가장 큰 축을 선택한다.

    Test는 사용하지 않는다.
    """
    if candidates is None:
        candidates = AXIS_CANDIDATES

    print("=" * 60)
    print("1단계: Validation F1 + AUC 기반 축 선택")
    print("       (하이퍼파라미터는 고정, Test 미사용)")
    print("=" * 60)

    best_axis = None
    best_score = -1.0
    axis_log = []

    for axis in candidates:
        X_tr, names = filter_features_by_axis(X_train, feature_names, axis)
        X_va, _ = filter_features_by_axis(X_val, feature_names, axis)

        X_tr_pca, pca, scaler = apply_pca(X_tr, n_components)
        X_va_pca = transform_pca(X_va, pca, scaler)

        metrics = _val_f1_auc_fixed_models(X_tr_pca, X_va_pca, y_train, y_val)
        row = {
            'axis': axis,
            'n_features': len(names),
            **metrics,
        }
        axis_log.append(row)

        marker = ""
        if metrics['combined_score'] > best_score:
            best_score = metrics['combined_score']
            best_axis = axis
            marker = " ← 현재 최고"

        print(f"\n  [{AXIS_LABEL[axis]}] 특징 {len(names)}개")
        print(
            f"    SVM  Val F1={metrics['svm_val_f1']:.4f}  "
            f"AUC={metrics['svm_val_auc']:.4f}"
        )
        print(
            f"    RF   Val F1={metrics['rf_val_f1']:.4f}  "
            f"AUC={metrics['rf_val_auc']:.4f}"
        )
        print(
            f"    평균 Val F1={metrics['mean_val_f1']:.4f}  "
            f"AUC={metrics['mean_val_auc']:.4f}  "
            f"종합={(metrics['combined_score']):.4f}{marker}"
        )

    print("-" * 60)
    print(
        f"선택 축: {AXIS_LABEL[best_axis]} "
        f"(종합 점수={best_score:.4f} = (평균F1 + 평균AUC) / 2)"
    )
    print("※ 이후 2단계에서 이 축을 고정한 채 하이퍼파라미터만 탐색합니다.")
    print("※ 7:3 / 교차조건도 이 축을 그대로 사용합니다.")
    print("=" * 60 + "\n")

    report = {
        'selected_axis': best_axis,
        'best_combined_score': best_score,
        'axis_log': axis_log,
    }
    return best_axis, report

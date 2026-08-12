import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

TARGET_NAMES = ['정상', '열화']


def _safe_auc(y_true, y_score):
    """이진 분류 AUC 계산. 한 클래스만 있으면 None 반환."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, y_score))


# 혼동 행렬과 분류 리포트를 출력
def print_evaluation_report(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f} ({np.sum(y_true == y_pred)} / {len(y_true)})")

    metrics = get_metrics_dict(y_true, y_pred, y_score=y_score)
    if metrics.get('auc') is not None:
        print(f"AUC:      {metrics['auc']:.4f}")
    elif y_score is not None:
        print("AUC:      N/A (단일 클래스만 존재)")

    print()
    print("Confusion Matrix:")
    print(f"실제: 정상 {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"실제: 열화 {cm[1][0]:>5}  {cm[1][1]:>5}")
    print()
    print("Classification Report")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))

    return metrics


def get_metrics_dict(y_true, y_pred, y_score=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

    # AUC는 확률/decision score가 있을 때만 계산
    if y_score is not None:
        metrics['auc'] = _safe_auc(y_true, y_score)
    else:
        metrics['auc'] = None

    return metrics

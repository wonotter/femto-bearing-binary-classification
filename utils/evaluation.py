import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

TARGET_NAMES = ['정상', '이상']

# 혼동 행렬과 분류 리포트를 출력
def print_evaluation_report(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f} ({np.sum(y_true == y_pred)} / {len(y_true)})")
    print()
    print("Confusion Matrix:")
    print(f"실제: 정상 {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"실제: 이상 {cm[1][0]:>5}  {cm[1][1]:>5}")
    print()
    print("Classification Report")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))

    return get_metrics_dict(y_true, y_pred)

def get_metrics_dict(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

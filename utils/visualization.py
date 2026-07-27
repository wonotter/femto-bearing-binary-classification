import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 예측 시각화 공통 색상 (정상/이상/오분류를 항상 같은 규칙으로 사용)
COLOR_NORMAL = '#4CAF50'      # 초록: 정상
COLOR_ANOMALY = '#E53935'     # 빨강: 이상
COLOR_MISCLASS = '#FF9800'    # 주황: 오분류 (이상 빨강과 구분)
INDEX_AXIS_MAX = 200          # 원본 인덱스 축 기본 표시 범위
INDEX_AXIS_TICK = 25


def _axis_limit_from_indices(sample_indices):
    """샘플 인덱스에 맞춰 x축 상한을 결정 (학습 조건 증가 시 자동 확장)."""
    if sample_indices is None or len(sample_indices) == 0:
        return INDEX_AXIS_MAX

    max_idx = int(np.max(sample_indices))
    return max(INDEX_AXIS_MAX, max_idx)


def plot_confusion_matrices(results):
    """각 모델/분할 조합별 Confusion Matrix 히트맵"""
    names = list(results.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    target_names = ['정상', '이상']

    for ax, name in zip(axes, names):
        test_data = results[name]['test']
        cm = confusion_matrix(test_data['y_true'], test_data['y_pred'])

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax
        )
        ax.set_title(name)
        ax.set_ylabel('실제')
        ax.set_xlabel('예측')

    fig.suptitle('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results):
    """모델별 Accuracy, Precision, Recall, F1 비교 막대 그래프"""
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())

    values = {key: [] for key in metric_keys}
    for name in model_names:
        metrics = results[name]['test']['metrics']
        for key in metric_keys:
            values[key].append(metrics[key])

    x = np.arange(len(model_names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, key in enumerate(metric_keys):
        bars = ax.bar(x + i * width, values[key], width, label=key.capitalize())
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('모델 성능 비교 (Test Set)', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances, feature_names, top_n=10, title=None):
    """Random Forest Feature Importance 수평 막대 그래프"""
    indices = np.argsort(importances)[::-1][:top_n]

    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    top_names = top_names[::-1]
    top_values = top_values[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top_names, top_values, color='steelblue')

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{w:.4f}', va='center', fontsize=9)

    ax.set_xlabel('Importance')
    
    if title is None:
        title = f'Feature Importance - Top {top_n}'

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_explained_variance(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(X.shape))
    pca.fit(X_scaled)

    cumulative = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(cumulative) + 1), pca.explained_variance_ratio_, label='개별')
    ax.plot(range(1, len(cumulative) + 1), cumulative, 'ro-', label='누적')
    ax.axhline(y=0.9, color='gray', linestyle='--', label='90%')
    ax.set_xlabel('주성분 번호')
    ax.set_ylabel('설명 분산 비율')
    ax.set_title('PCA Scree Plot')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_pca_loading(X, feature_names, n_components=3):
    """각 주성분별 피처 Loading 계수를 수평 막대 그래프로 시각화"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    loadings = pca.components_.T  # (피처 수, 주성분 수)

    fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 6))

    if n_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        pc_loadings = loadings[:, i]
        sorted_idx = np.argsort(np.abs(pc_loadings))[::-1]

        sorted_names = [feature_names[j] for j in sorted_idx]
        sorted_values = pc_loadings[sorted_idx]

        colors = ['#d32f2f' if v > 0 else '#1976d2' for v in sorted_values]

        ax.barh(range(len(sorted_names)), sorted_values, color=colors)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Loading 계수')
        ax.set_title(f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)')
        ax.axvline(x=0, color='gray', linewidth=0.8)
        ax.invert_yaxis()

    fig.suptitle('PCA Loading Plot (피처별 주성분 기여도)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def _sort_by_original_index(y_true, y_pred, sample_indices=None):
    """원본 인덱스 기준으로 정렬. 인덱스가 없으면 현재 순서 유지."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if sample_indices is None:
        return y_true, y_pred, np.arange(len(y_true))

    sample_indices = np.asarray(sample_indices)
    order = np.argsort(sample_indices)

    return y_true[order], y_pred[order], sample_indices[order]

def _label_colors(labels):
    """라벨 배열을 정상=초록, 이상=빨강 색상 리스트로 변환."""
    return [COLOR_NORMAL if label == 0 else COLOR_ANOMALY for label in labels]


def plot_prediction_timeline(y_true, y_pred, sample_indices=None, title='예측 타임라인'):
    """
    각 샘플의 실제/예측 라벨을 원본 인덱스(시간 순서) 기준으로 시각화.
    - 정상: 초록, 이상: 빨강
    - 오분류: 주황 X
    """
    y_true, y_pred, x = _sort_by_original_index(y_true, y_pred, sample_indices)
    correct = y_true == y_pred
    incorrect = ~correct
    axis_max = _axis_limit_from_indices(x)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # 상단: 실제 라벨 (정상=초록, 이상=빨강)
    ax = axes[0]
    ax.scatter(x, y_true, c=_label_colors(y_true), s=30, alpha=0.8, edgecolors='none')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['정상', '이상'])
    ax.set_title('실제 라벨 (Ground Truth)')
    ax.set_ylabel('클래스')
    ax.set_xlim(0, axis_max)

    # 하단: 예측 라벨 (맞춘 것은 클래스 색, 틀린 것은 주황 X)
    ax = axes[1]
    pred_normal = correct & (y_pred == 0)
    pred_anomaly = correct & (y_pred == 1)

    ax.scatter(
        x[pred_normal], y_pred[pred_normal],
        c=COLOR_NORMAL, s=30, alpha=0.8, marker='o', edgecolors='none',
        label='정상',
    )
    ax.scatter(
        x[pred_anomaly], y_pred[pred_anomaly],
        c=COLOR_ANOMALY, s=30, alpha=0.8, marker='o', edgecolors='none',
        label='이상',
    )
    ax.scatter(
        x[incorrect], y_pred[incorrect],
        c=COLOR_MISCLASS, s=70, marker='x', linewidths=2,
        label='오분류',
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['정상', '이상'])
    ax.set_title('모델 예측 결과')
    ax.set_xlabel('원본 샘플 인덱스 (시간 순서)')
    ax.set_ylabel('클래스')
    ax.set_xlim(0, axis_max)
    ax.set_xticks(np.arange(0, axis_max + 1, INDEX_AXIS_TICK))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_NORMAL,
               markersize=8, label='정상'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ANOMALY,
               markersize=8, label='이상'),
        Line2D([0], [0], marker='x', color=COLOR_MISCLASS, linestyle='None',
               markersize=8, markeredgewidth=2, label='오분류'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_prediction_scatter_pca(X_pca, y_true, y_pred, title='PCA 공간 예측 결과'):
    """
    PCA 2D 공간에서 예측 결과를 시각화.
    - 정답: 원(circle) 마커
    - 오분류: 삼각형(triangle) 마커 + 빨간 테두리
    """
    correct = y_true == y_pred
    incorrect = ~correct

    fig, ax = plt.subplots(figsize=(10, 7))

    # 정답 - 정상
    mask = correct & (y_true == 0)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLOR_NORMAL, marker='o', s=50, alpha=0.6, label='정상 (정답)')

    # 정답 - 이상
    mask = correct & (y_true == 1)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLOR_ANOMALY, marker='o', s=50, alpha=0.6, label='이상 (정답)')

    # 오분류 - 실제 정상인데 이상으로 예측
    mask = incorrect & (y_true == 0)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLOR_MISCLASS, marker='^', s=100, edgecolors='black',
               linewidths=1.2, label='정상→이상 오분류')

    # 오분류 - 실제 이상인데 정상으로 예측
    mask = incorrect & (y_true == 1)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c='#7B1FA2', marker='v', s=100, edgecolors='black',
               linewidths=1.2, label='이상→정상 오분류')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_prediction_heatmap_strip(y_true, y_pred, sample_indices=None, title='예측 결과 비교 Strip'):
    """
    원본 인덱스 순으로 정렬한 뒤, 실제/예측/오분류를 가로 Strip으로 비교.
    x축은 Timeline과 동일하게 샘플 인덱스 범위에 맞춰 표시한다.
    """
    y_true, y_pred, x = _sort_by_original_index(y_true, y_pred, sample_indices)
    axis_max = _axis_limit_from_indices(x)
    mismatch = (y_true != y_pred).astype(int)
    data = np.vstack([
        y_true.reshape(1, -1),
        y_pred.reshape(1, -1),
        mismatch.reshape(1, -1),
    ])

    fig, axes = plt.subplots(3, 1, figsize=(14, 3), sharex=True)

    cmap_label = plt.cm.colors.ListedColormap([COLOR_NORMAL, COLOR_ANOMALY])
    cmap_mismatch = plt.cm.colors.ListedColormap(['#FFFFFF', COLOR_MISCLASS])

    # Strip을 인덱스 구간에 꽉 채워 양끝 여백이 생기지 않도록 함
    extent = [0, axis_max, 0, 1]
    rows = [
        (data[0:1], cmap_label, '실제'),
        (data[1:2], cmap_label, '예측'),
        (data[2:3], cmap_mismatch, '오분류'),
    ]

    for i, (row, cmap, ylabel) in enumerate(rows):
        axes[i].imshow(row, aspect='auto', cmap=cmap, vmin=0, vmax=1, extent=extent)
        axes[i].set_yticks([0.5])
        axes[i].set_yticklabels([ylabel])
        axes[i].set_xlim(0, axis_max)
        if i == 0:
            axes[i].set_title(title, fontsize=14, fontweight='bold')

    axes[2].set_xlabel('원본 샘플 인덱스 (시간 순서)')
    axes[2].set_xticks(np.arange(0, axis_max + 1, INDEX_AXIS_TICK))

    legend_elements = [
        Patch(facecolor=COLOR_NORMAL, label='정상'),
        Patch(facecolor=COLOR_ANOMALY, label='이상'),
        Patch(facecolor=COLOR_MISCLASS, label='오분류'),
    ]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

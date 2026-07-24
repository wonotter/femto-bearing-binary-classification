import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

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
    """각 주성분에 대한 원래 피처의 기여도(Loading)를 히트맵으로 시각화"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    loadings = pca.components_.T  # (피처 수, 주성분 수)
    loading_df = np.array(loadings)

    pc_labels = [f'PC{i+1}' for i in range(n_components)]

    fig, ax = plt.subplots(figsize=(8, len(feature_names) * 0.4 + 1))
    im = ax.imshow(loading_df, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(n_components))
    ax.set_xticklabels(pc_labels)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)

    for i in range(len(feature_names)):
        for j in range(n_components):
            ax.text(j, i, f'{loading_df[i, j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(loading_df[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='Loading 계수')
    ax.set_title('PCA Loading Plot (피처별 주성분 기여도)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

from data.data_loader import prepare_dataset
from data.data_splitter import split_622, split_73
from models.svm_model import run_svm_622, run_svm_73
from models.rf_model import run_rf_622, run_rf_73
from utils.visualization import (
    plot_confusion_matrices,
    plot_model_comparison,
    plot_feature_importance,
    plot_explained_variance,
)
from config import BEARING_DIR, PCA_N_COMPONENTS
from features.feature_extractor import apply_pca, transform_pca

def main():
    print("-" * 60)
    print("FEMTO Bearing Dataset 이진 분류 실험")
    print("-" * 60)
    print()

    X, y, feature_names = prepare_dataset(BEARING_DIR)

    plot_explained_variance(X)

    pca_names = [f'PC{i + 1}' for i in range(PCA_N_COMPONENTS)]

    print("\n")
    print("실험 A: Train:Validation:Test = 6:2:2")

    X_train, X_val, X_test, y_train, y_val, y_test = split_622(X, y)

    X_train_pca, pca, scaler = apply_pca(X_train, PCA_N_COMPONENTS)
    X_val_pca = transform_pca(X_val, pca, scaler)
    X_test_pca = transform_pca(X_test, pca, scaler)

    svm_622 = run_svm_622(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test)
    print()
    rf_622 = run_rf_622(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca_names)

    print("\n")
    print("실험 B: Train:Validation:Test = 7:3")

    X_train, X_test, y_train, y_test = split_73(X, y)

    X_train_pca, pca, scaler = apply_pca(X_train, PCA_N_COMPONENTS)
    X_test_pca = transform_pca(X_test, pca, scaler)
    
    svm_73 = run_svm_73(X_train_pca, X_test_pca, y_train, y_test)
    print()
    rf_73 = run_rf_73(X_train_pca, X_test_pca, y_train, y_test, pca_names)

    # 시각화
    all_results = {
        'SVM 6:2:2': svm_622,
        'RF 6:2:2': rf_622,
        'SVM 7:3': svm_73,
        'RF 7:3': rf_73,
    }
    plot_confusion_matrices(all_results)
    plot_model_comparison(all_results)
    plot_feature_importance(
        rf_622['feature_importances'],
        rf_622['feature_names'],
        top_n=PCA_N_COMPONENTS,
        title=f'RF 6:2:2 Feature Importance (PCA {PCA_N_COMPONENTS})',
    )
    
    plot_feature_importance(
        rf_73['feature_importances'],
        rf_73['feature_names'],
        top_n=PCA_N_COMPONENTS,
        title=f'RF 7:3 Feature Importance (PCA {PCA_N_COMPONENTS})',
    )

if __name__ == "__main__":
    main()

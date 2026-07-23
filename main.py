from data.data_loader import prepare_dataset
from data.data_splitter import split_622, split_73
from models.svm_model import run_svm_622, run_svm_73
from models.rf_model import run_rf_622, run_rf_73
from utils.visualization import (
    plot_confusion_matrices,
    plot_model_comparison,
    plot_feature_importance,
)
from config import BEARING_DIR

def main():
    print("-" * 60)
    print("FEMTO Bearing Dataset 이진 분류 실험")
    print("-" * 60)
    print()

    X, y, feature_names = prepare_dataset(BEARING_DIR)


    print("\n")
    print("실험 A: Train:Validation:Test = 6:2:2")

    X_train, X_val, X_test, y_train, y_val, y_test = split_622(X, y)

    svm_622 = run_svm_622(X_train, X_val, X_test, y_train, y_val, y_test)
    print()
    rf_622 = run_rf_622(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)

    print("\n")
    print("실험 B: Train:Validation:Test = 7:3")

    X_train, X_test, y_train, y_test = split_73(X, y)
    
    svm_73 = run_svm_73(X_train, X_test, y_train, y_test)
    print()
    rf_73 = run_rf_73(X_train, X_test, y_train, y_test, feature_names)

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
        top_n=10,
        title='RF 6:2:2 Feature Importance',
    )
    
    plot_feature_importance(
        rf_73['feature_importances'],
        rf_73['feature_names'],
        top_n=10,
        title='RF 7:3 Feature Importance',
    )

if __name__ == "__main__":
    main()

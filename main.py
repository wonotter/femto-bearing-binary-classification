import os
from data.data_loader import prepare_dataset
from data.data_splitter import split_622, split_73
from models.svm_model import run_svm_622, run_svm_73
from models.rf_model import run_rf_622, run_rf_73
from utils.visualization import (
    plot_confusion_matrices,
    plot_model_comparison,
    plot_feature_importance,
    plot_explained_variance,
    plot_pca_loading,
    plot_prediction_timeline,
    plot_prediction_scatter_pca,
    plot_prediction_heatmap_strip,
)
from config import (
    BEARING_DIR, BASE_DIR, PCA_N_COMPONENTS,
    TRAIN_CONDITION, TEST_CONDITIONS,
)
from features.feature_extractor import apply_pca, transform_pca

def main():
    print("-" * 60)
    print("FEMTO Bearing Dataset 이진 분류 실험")
    print("-" * 60)
    print()

    X, y, feature_names = prepare_dataset(BEARING_DIR)

    plot_explained_variance(X) # Scree Plot: PCA 몇 개가 필요한가?
    plot_pca_loading(X, feature_names, n_components=PCA_N_COMPONENTS) # Loading Plot: 각 주성분에 대한 원래 피처의 기여도

    pca_names = [f'PC{i + 1}' for i in range(PCA_N_COMPONENTS)]

    print("\n")
    print("실험 A: Train:Validation:Test = 6:2:2")

    X_train_622, X_val_622, X_test_622, y_train_622, y_val_622, y_test_622, idx_train_622, idx_val_622, idx_test_622 = split_622(X, y)

    X_train_pca_622, pca_622, scaler_622 = apply_pca(X_train_622, PCA_N_COMPONENTS)
    X_val_pca_622 = transform_pca(X_val_622, pca_622, scaler_622)
    X_test_pca_622 = transform_pca(X_test_622, pca_622, scaler_622)

    svm_622 = run_svm_622(X_train_pca_622, X_val_pca_622, X_test_pca_622, y_train_622, y_val_622, y_test_622)
    print()
    rf_622 = run_rf_622(X_train_pca_622, X_val_pca_622, X_test_pca_622, y_train_622, y_val_622, y_test_622, pca_names)

    print("\n")
    print("실험 B: Train:Validation:Test = 7:3")

    X_train_73, X_test_73, y_train_73, y_test_73, idx_train_73, idx_test_73 = split_73(X, y)

    X_train_pca_73, pca_73, scaler_73 = apply_pca(X_train_73, PCA_N_COMPONENTS)
    X_test_pca_73 = transform_pca(X_test_73, pca_73, scaler_73)
    
    svm_73 = run_svm_73(X_train_pca_73, X_test_pca_73, y_train_73, y_test_73)
    print()
    rf_73 = run_rf_73(X_train_pca_73, X_test_pca_73, y_train_73, y_test_73, pca_names)

    # 시각화
    first_results = {
        'SVM 6:2:2': svm_622,
        'RF 6:2:2': rf_622,
        'SVM 7:3': svm_73,
        'RF 7:3': rf_73,
    }
    plot_confusion_matrices(first_results)
    plot_model_comparison(first_results)
    plot_feature_importance(
        rf_622['feature_importances'],
        rf_622['feature_names'],
        top_n=PCA_N_COMPONENTS,
        title=f'RF 6:2:2 Feature Importance (PCA = {PCA_N_COMPONENTS})',
    )
    
    plot_feature_importance(
        rf_73['feature_importances'],
        rf_73['feature_names'],
        top_n=PCA_N_COMPONENTS,
        title=f'RF 7:3 Feature Importance (PCA = {PCA_N_COMPONENTS})',
    )

    test_indices_map = {
        'SVM 6:2:2': idx_test_622,
        'RF 6:2:2': idx_test_622,
        'SVM 7:3': idx_test_73,
        'RF 7:3': idx_test_73,
    }

    for name, result in first_results.items():
        test_data = result['test']
        y_true = test_data['y_true']
        y_pred = test_data['y_pred']
        sample_indices = test_indices_map[name]

        plot_prediction_timeline(
            y_true, y_pred, 
            sample_indices=sample_indices, 
            title=f'{name} - Predicted Timeline'
        )
        plot_prediction_heatmap_strip(
            y_true, y_pred, 
            sample_indices=sample_indices, 
            title=f'{name} - Predicted Comparison Strip'
        )
        
    # PCA 산점도 (테스트 셋 기준)
    # 실험 A의 경우 X_test_pca 사용
    plot_prediction_scatter_pca(
        X_test_pca_622, y_test_622, svm_622['test']['y_pred'],
        title='SVM 6:2:2 - PCA 공간 예측 결과'
    )
    
    plot_prediction_scatter_pca(
        X_test_pca_622, y_test_622, rf_622['test']['y_pred'],
        title='RF 6:2:2 - PCA 공간 예측 결과'
    )

    # 실험 B PCA 산점도
    plot_prediction_scatter_pca(
        X_test_pca_73, y_test_73, svm_73['test']['y_pred'],
        title='SVM 7:3 - PCA 공간 예측 결과'
    )

    plot_prediction_scatter_pca(
        X_test_pca_73, y_test_73, rf_73['test']['y_pred'],
        title='RF 7:3 - PCA 공간 예측 결과'
    )

    print("\n" + "=" * 60)
    print("실험 C: 교차 조건 테스트")
    print(f"학습 조건: {TRAIN_CONDITION} (실험 A, B에서 학습된 모델 재사용)")
    print(f"테스트 조건: {TEST_CONDITIONS}")
    print("=" * 60)

    for test_cond in TEST_CONDITIONS:
        print(f"\n{'─' * 40}")
        print(f"테스트 조건: {test_cond}")
        print(f"{'─' * 40}")

        test_dir = os.path.join(BASE_DIR, test_cond)
        X_cross, y_cross, _ = prepare_dataset(test_dir)

        cond_results = {}

        # 실험 A(6:2:2) 모델로 테스트
        X_cross_622 = transform_pca(X_cross, pca_622, scaler_622)

        y_pred, metrics = svm_622['model'].evaluate(X_cross_622, y_cross, set_name=test_cond)
        cond_results[f'SVM 6:2:2→{test_cond}'] = {
            'test': {'y_true': y_cross, 'y_pred': y_pred, 'metrics': metrics}
        }

        y_pred, metrics = rf_622['model'].evaluate(X_cross_622, y_cross, set_name=test_cond)
        cond_results[f'RF 6:2:2→{test_cond}'] = {
            'test': {'y_true': y_cross, 'y_pred': y_pred, 'metrics': metrics}
        }

        # 실험 B(7:3) 모델로 테스트
        X_cross_73 = transform_pca(X_cross, pca_73, scaler_73)

        y_pred, metrics = svm_73['model'].evaluate(X_cross_73, y_cross, set_name=test_cond)
        cond_results[f'SVM 7:3→{test_cond}'] = {
            'test': {'y_true': y_cross, 'y_pred': y_pred, 'metrics': metrics}
        }

        y_pred, metrics = rf_73['model'].evaluate(X_cross_73, y_cross, set_name=test_cond)
        cond_results[f'RF 7:3→{test_cond}'] = {
            'test': {'y_true': y_cross, 'y_pred': y_pred, 'metrics': metrics}
        }

        plot_confusion_matrices(cond_results)
        plot_model_comparison(cond_results)

        for cond_name, cond_result in cond_results.items():
            test_data = cond_result['test']
            
            plot_prediction_timeline(
                test_data['y_true'], test_data['y_pred'],
                title=f'{cond_name} - 예측 타임라인'
            )
            
            plot_prediction_heatmap_strip(
                test_data['y_true'], test_data['y_pred'],
                title=f'{cond_name} - 예측 비교 Strip'
            )

        # PCA 산점도 (교차 조건 전체 데이터 기준)
        # 6:2:2 / 7:3 각각 학습 시 맞춘 scaler·PCA로 변환한 좌표 사용
        plot_prediction_scatter_pca(
            X_cross_622, y_cross,
            cond_results[f'SVM 6:2:2→{test_cond}']['test']['y_pred'],
            title=f'SVM 6:2:2→{test_cond} - PCA 공간 예측 결과'
        )
        plot_prediction_scatter_pca(
            X_cross_622, y_cross,
            cond_results[f'RF 6:2:2→{test_cond}']['test']['y_pred'],
            title=f'RF 6:2:2→{test_cond} - PCA 공간 예측 결과'
        )
        plot_prediction_scatter_pca(
            X_cross_73, y_cross,
            cond_results[f'SVM 7:3→{test_cond}']['test']['y_pred'],
            title=f'SVM 7:3→{test_cond} - PCA 공간 예측 결과'
        )
        plot_prediction_scatter_pca(
            X_cross_73, y_cross,
            cond_results[f'RF 7:3→{test_cond}']['test']['y_pred'],
            title=f'RF 7:3→{test_cond} - PCA 공간 예측 결과'
        )

if __name__ == "__main__":
    main()

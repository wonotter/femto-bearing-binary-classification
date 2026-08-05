import os
from data.data_loader import prepare_dataset, prepare_datasets
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
    plot_axis_selection_results,
)
from utils.raw_signal_visualization import plot_time_domain_analysis
from config import (
    BASE_DIR, PCA_N_COMPONENTS,
    TRAIN_CONDITIONS, TEST_CONDITIONS,
)
from features.feature_extractor import apply_pca, transform_pca
from features.axis_selector import (
    select_axis_by_validation,
    filter_features_by_axis,
    AXIS_LABEL,
)

def main():
    print("-" * 60)
    print("FEMTO Bearing Dataset 이진 분류 실험")
    print(f"학습 조건: {TRAIN_CONDITIONS}")
    print(f"테스트 조건: {TEST_CONDITIONS}")
    print("-" * 60)
    print()

    # 실험 시작 전: 학습 조건별 시간 영역 특성 추이 (실제 파일 번호)
    print("시간 영역 분석 시각화 (전체 수명 + early/late 구간 표시)")
    for cond in TRAIN_CONDITIONS:
        bearing_dir = os.path.join(BASE_DIR, cond)
        print(f"  - {cond}")
        plot_time_domain_analysis(bearing_dir=bearing_dir, bearing_name=cond)
    print()

    # 학습 조건: 수평+수직 모두 추출 (축은 Validation에서 선택)
    X, y, feature_names = prepare_datasets(TRAIN_CONDITIONS)

    pca_names = [f'PC{i + 1}' for i in range(PCA_N_COMPONENTS)]
    train_label = '+'.join(TRAIN_CONDITIONS)

    print("\n")
    print(f"실험 A: Train:Validation:Test = 6:2:2 ({train_label})")

    (
        X_train_622, X_val_622, X_test_622,
        y_train_622, y_val_622, y_test_622,
        idx_train_622, idx_val_622, idx_test_622,
    ) = split_622(X, y)

    # ----- 1단계: Val F1 + AUC로 축 선택 (Test 미사용, 파라미터 고정) -----
    selected_axis, axis_report = select_axis_by_validation(
        X_train_622, X_val_622,
        y_train_622, y_val_622,
        feature_names,
        n_components=PCA_N_COMPONENTS,
    )
    plot_axis_selection_results(
        axis_report,
        title=f'축 선택 비교 (Validation F1 + AUC) — {train_label}',
    )

    X_train_622, selected_feature_names = filter_features_by_axis(
        X_train_622, feature_names, selected_axis
    )
    X_val_622, _ = filter_features_by_axis(X_val_622, feature_names, selected_axis)
    X_test_622, _ = filter_features_by_axis(X_test_622, feature_names, selected_axis)

    print(
        f"[축 적용] {AXIS_LABEL[selected_axis]} → "
        f"특징 {len(feature_names)}개 → {len(selected_feature_names)}개"
    )
    print(f"사용 특징: {selected_feature_names}\n")

    plot_explained_variance(X_train_622)
    plot_pca_loading(
        X_train_622, selected_feature_names, n_components=PCA_N_COMPONENTS
    )

    # ----- 2단계: 선택 축 고정 후 PCA + 하이퍼파라미터 탐색 -----
    X_train_pca_622, pca_622, scaler_622 = apply_pca(X_train_622, PCA_N_COMPONENTS)
    X_val_pca_622 = transform_pca(X_val_622, pca_622, scaler_622)
    X_test_pca_622 = transform_pca(X_test_622, pca_622, scaler_622)

    svm_622 = run_svm_622(
        X_train_pca_622, X_val_pca_622, X_test_pca_622,
        y_train_622, y_val_622, y_test_622,
    )
    svm_622['selected_axis'] = selected_axis
    svm_622['axis_report'] = axis_report
    print()
    rf_622 = run_rf_622(
        X_train_pca_622, X_val_pca_622, X_test_pca_622,
        y_train_622, y_val_622, y_test_622, pca_names,
    )
    rf_622['selected_axis'] = selected_axis
    rf_622['axis_report'] = axis_report

    print("\n")
    print(f"실험 B: Train:Test = 7:3 ({train_label})")
    print(
        f"※ Validation이 없으므로 실험 A(6:2:2)에서 선택한 축 "
        f"[{AXIS_LABEL[selected_axis]}]을 그대로 사용합니다."
    )

    X_train_73, X_test_73, y_train_73, y_test_73, idx_train_73, idx_test_73 = split_73(X, y)

    X_train_73, _ = filter_features_by_axis(X_train_73, feature_names, selected_axis)
    X_test_73, _ = filter_features_by_axis(X_test_73, feature_names, selected_axis)

    X_train_pca_73, pca_73, scaler_73 = apply_pca(X_train_73, PCA_N_COMPONENTS)
    X_test_pca_73 = transform_pca(X_test_73, pca_73, scaler_73)

    svm_73 = run_svm_73(X_train_pca_73, X_test_pca_73, y_train_73, y_test_73)
    svm_73['selected_axis'] = selected_axis
    print()
    rf_73 = run_rf_73(
        X_train_pca_73, X_test_pca_73, y_train_73, y_test_73, pca_names
    )
    rf_73['selected_axis'] = selected_axis

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

    plot_prediction_scatter_pca(
        X_test_pca_622, y_test_622, svm_622['test']['y_pred'],
        title=f'SVM 6:2:2 ({train_label}) - PCA 공간 예측 결과'
    )

    plot_prediction_scatter_pca(
        X_test_pca_622, y_test_622, rf_622['test']['y_pred'],
        title=f'RF 6:2:2 ({train_label}) - PCA 공간 예측 결과'
    )

    plot_prediction_scatter_pca(
        X_test_pca_73, y_test_73, svm_73['test']['y_pred'],
        title=f'SVM 7:3 ({train_label}) - PCA 공간 예측 결과'
    )

    plot_prediction_scatter_pca(
        X_test_pca_73, y_test_73, rf_73['test']['y_pred'],
        title=f'RF 7:3 ({train_label}) - PCA 공간 예측 결과'
    )

    print("\n" + "=" * 60)
    print("실험 C: 교차 조건 테스트")
    print(f"학습 조건: {TRAIN_CONDITIONS} (실험 A, B에서 학습된 모델 재사용)")
    print(f"테스트 조건: {TEST_CONDITIONS}")
    print(f"사용 축: {AXIS_LABEL[selected_axis]} (실험 A Validation F1+AUC로 선택)")
    print("=" * 60)

    for test_cond in TEST_CONDITIONS:
        print(f"\n{'─' * 40}")
        print(f"테스트 조건: {test_cond}")
        print(f"{'─' * 40}")

        test_dir = os.path.join(BASE_DIR, test_cond)
        X_cross, y_cross, cross_names = prepare_dataset(test_dir)
        X_cross, _ = filter_features_by_axis(X_cross, cross_names, selected_axis)

        cond_results = {}

        X_cross_622 = transform_pca(X_cross, pca_622, scaler_622)

        y_pred, metrics = svm_622['model'].evaluate(X_cross_622, y_cross, set_name=test_cond)
        cond_results[f'SVM 6:2:2→{test_cond}'] = {
            'test': {'y_true': y_cross, 'y_pred': y_pred, 'metrics': metrics}
        }

        y_pred, metrics = rf_622['model'].evaluate(X_cross_622, y_cross, set_name=test_cond)
        cond_results[f'RF 6:2:2→{test_cond}'] = {
            'test': {'y_true': y_cross, 'y_pred': y_pred, 'metrics': metrics}
        }

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

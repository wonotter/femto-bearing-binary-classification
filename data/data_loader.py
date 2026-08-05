import os
import glob
import numpy as np
import pandas as pd

import sys
sys.path.append("..")

from config import (
    BASE_DIR, BEARING_DIR, N_EARLY_FILES, N_LATE_FILES,
    LABEL_NORMAL, LABEL_DEGRADED,
)
from features.feature_extractor import extract_all_features

def load_file_list(bearing_dir=BEARING_DIR):
    csv_files = sorted(glob.glob(os.path.join(bearing_dir, 'acc_*.csv')))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {bearing_dir}")
    
    return csv_files

def prepare_dataset(bearing_dir=BEARING_DIR):
    # csv 파일 목록 불러오기
    csv_files = load_file_list(bearing_dir)

    total_files = len(csv_files)

    early_files = csv_files[:N_EARLY_FILES]
    late_files = csv_files[-N_LATE_FILES:]

    print(f"총 파일 수: {total_files}")
    print(f"첫 {N_EARLY_FILES}개 파일: {early_files[0]} ~ {early_files[-1]}")
    print(f"마지막 {N_LATE_FILES}개 파일: {late_files[0]} ~ {late_files[-1]}")

    records = []

    print(f"\n 초기 {N_EARLY_FILES}개 파일에서 특징 추출")
    for i, fpath in enumerate(early_files):
        feats = extract_all_features(fpath)
        feats['label'] = LABEL_NORMAL
        records.append(feats)
        
        if (i + 1) % 25 == 0:
            print(f" 진행중:{i + 1}/{N_EARLY_FILES} 파일 처리 완료")
    
    print(f"\n 마지막 {N_LATE_FILES}개 파일에서 특징 추출")
    for i, fpath in enumerate(late_files):
        feats = extract_all_features(fpath)
        feats['label'] = LABEL_DEGRADED
        records.append(feats)
        
        if (i + 1) % 25 == 0:
            print(f" 진행중:{i + 1}/{N_LATE_FILES} 파일 처리 완료")
    
    # 200개짜리 딕셔너리를 포함한 리스트 -> 2차원 테이블 형태로 변환
    df = pd.DataFrame(records)
    feature_names = [c for c in df.columns if c != 'label']
    X = df[feature_names].values # label을 제외한 14개 열만 꺼냄
    y = df['label'].values # 정상(0) or 이상(1)의 정답 상태를 가지고 있는 정답지

    print(f"\n특징 추출 완료")
    print(f"샘플: {X.shape[0]}, 특징 수: {X.shape[1]}")
    print(f"정상(0): {np.sum(y == LABEL_NORMAL)}, 이상(1): {np.sum(y == LABEL_DEGRADED)}")

    return X, y, feature_names


def prepare_datasets(conditions, base_dir=BASE_DIR):
    """
    여러 조건 폴더를 순서대로 불러와 하나의 (X, y)로 합친다.
    conditions 예: ['Cond1_1', 'Cond1_2']
    """
    if not conditions:
        raise ValueError("conditions 리스트가 비어 있습니다.")

    X_list = []
    y_list = []
    feature_names = None

    print(f"조건 {len(conditions)}개 로드: {conditions}")

    for cond in conditions:
        print(f"\n{'─' * 40}")
        print(f"조건 로드: {cond}")
        print(f"{'─' * 40}")

        bearing_dir = os.path.join(base_dir, cond)
        X, y, names = prepare_dataset(bearing_dir)

        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                f"특징 이름이 조건마다 다릅니다. "
                f"기준={feature_names}, {cond}={names}"
            )

        X_list.append(X)
        y_list.append(y)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print(f"\n{'=' * 40}")
    print(f"다중 조건 병합 완료: {conditions}")
    print(f"샘플: {X.shape[0]}, 특징 수: {X.shape[1]}")
    print(f"정상(0): {np.sum(y == LABEL_NORMAL)}, 이상(1): {np.sum(y == LABEL_DEGRADED)}")
    print(f"{'=' * 40}")

    return X, y, feature_names

import os
import glob
import numpy as np
import pandas as pd

import sys
sys.path.append("..")

from config import (
    BEARING_DIR, N_EARLY_FILES, N_LATE_FILES, 
    LABEL_NORMAL, LABEL_DEGRADED
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
    X = df[feature_names].values # label을 제외한 26개 열만 꺼냄
    y = df['label'].values # 정상(0) or 열화(1)의 정답 상태를 가지고 있는 정답지

    print(f"\n특징 추출 완료")
    print(f"샘플: {X.shape[0]}, 특징 수: {X.shape[1]}")
    print(f"정상(0): {np.sum(y == LABEL_NORMAL)}, 열화(1): {np.sum(y == LABEL_DEGRADED)}")

    return X, y, feature_names

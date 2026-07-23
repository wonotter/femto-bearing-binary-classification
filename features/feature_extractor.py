import numpy as np
import pandas as pd
from scipy.signal import stft as scipy_stft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("..")
from config import SAMPLING_FREQ, STFT_NPERSEG, STFT_NOVERLAP, FREQ_BANDS

# 시간 영역 통계 특징 3개 추출
def extract_time_domain_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    mean_val = np.mean(signal)
    std_val = np.std(signal)

    return {
        'rms': rms,
        'crest_factor': peak / (rms + 1e-10),
        'kurtosis': np.mean(((signal - mean_val) / (std_val + 1e-10)) ** 4),
    }

# STFT 가번 주파수 영역 특징 4개 추출
def extract_frequency_domain_features(signal, fs=SAMPLING_FREQ):
    f, t, Zxx = scipy_stft(
        signal, fs=fs,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        window='hann'
    )
    power = np.abs(Zxx) ** 2
    features = {}

    # 대역별 에너지
    for name, (f_low, f_high) in FREQ_BANDS.items():
        mask = (f >= f_low) & (f <= f_high)
        features[f'{name}_energy'] = np.mean(power[mask, :])

    # 스펙트럴 중심 주파수 - 주파수별 에너지를 무게로 올려 놓았을 때, 무게중심이 어디에 있는지 계산
    freq_col = f.reshape(-1, 1) # (257, ) -> (257, 1)
    spectral_centroid = np.sum(freq_col * power, axis=0) / (np.sum(power, axis=0) + 1e-10)
    features['spectral_centroid'] = np.mean(spectral_centroid)

    return features

# 하나의 csv 파일에서 수평(h), 수직(v) 축 가속도 신호를 읽어와서 시간 및 주파수 영역 특징 추출
def extract_all_features(filepath):
    df = pd.read_csv(
        filepath, header=None,
        names=['hour', 'min', 'sec', 'usec', 'h_acc', 'v_acc']
    )

    all_features = {}

    for axis, col in [('h', 'h_acc'), ('v', 'v_acc')]:
        signal = df[col].values

        time_features = extract_time_domain_features(signal)
        for k, v in time_features.items():
            all_features[f'{axis}_{k}'] = v

        freq_features = extract_frequency_domain_features(signal)
        for k, v in freq_features.items():
            all_features[f'{axis}_{k}'] = v
        
    return all_features

# 학습 데이터만 받아서 PCA 적용 후 대표 특성 추출
def apply_pca(X_train, n_components):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)

    repr_features = pca.explained_variance_ratio_
    cumulative = np.cumsum(repr_features)
    print(f"PCA: {n_components}개 주성분으로 전체 분산의 {cumulative[-1]*100:.1f}% 대표 가능")

    return X_train_pca, pca, scaler

# 검증/테스트 데이터를 받아서 PCA 변환 시 사용
def transform_pca(X, pca, scaler):
    return pca.transform(scaler.transform(X))

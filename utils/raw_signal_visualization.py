"""
시간 영역 특성 시각화
- 베어링 수명 전체 파일의 RMS / Crest Factor / Kurtosis / Skewness 추이
- x축: acc_XXXXX.csv의 실제 파일 번호 (예: 1, 100, 2704, 2803)
- 정상(early 100개) / 이상(late 100개) 구간을 실제 번호로 표시
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BEARING_DIR,
    N_EARLY_FILES,
    N_LATE_FILES,
)
from data.data_loader import load_file_list

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

COLOR_H = 'steelblue'
COLOR_V = 'coral'
COLOR_EARLY = '#4CAF50'
COLOR_LATE = '#E53935'


def compute_rms(signal):
    """RMS: 신호의 평균 에너지 수준."""
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_crest_factor(signal):
    """Crest Factor = Peak / RMS. 충격성 결함 시 증가."""
    rms = compute_rms(signal)
    if rms == 0:
        return 0.0
    return float(np.max(np.abs(signal)) / rms)


def compute_kurtosis(signal):
    """
    첨도 (Fisher=False, Pearson).
    정규분포 ≈ 3, 결함 충격이 늘면 커진다.
    """
    return float(stats.kurtosis(signal, fisher=False))


def compute_skewness(signal):
    """왜도. 정상 ≈ 0, 비대칭 결함 시 변화."""
    return float(stats.skew(signal))


def _file_number_from_path(filepath):
    """acc_00001.csv → 1 처럼 파일명의 실제 번호를 반환."""
    name = os.path.basename(filepath)
    match = re.search(r'acc_(\d+)', name)
    if not match:
        raise ValueError(f"파일 번호를 파싱할 수 없습니다: {name}")
    return int(match.group(1))


def extract_time_domain_features(filepath):
    """한 CSV에서 수평/수직 시간 영역 특성 8개를 추출."""
    df = pd.read_csv(
        filepath,
        header=None,
        names=['hour', 'min', 'sec', 'usec', 'h_acc', 'v_acc'],
    )

    features = {'file_number': _file_number_from_path(filepath)}
    for axis, col in [('horizontal', 'h_acc'), ('vertical', 'v_acc')]:
        sig = df[col].values.astype(float)
        features[f'{axis}_rms'] = compute_rms(sig)
        features[f'{axis}_crest_factor'] = compute_crest_factor(sig)
        features[f'{axis}_kurtosis'] = compute_kurtosis(sig)
        features[f'{axis}_skewness'] = compute_skewness(sig)

    return features


def extract_bearing_time_features(bearing_dir):
    """베어링 폴더 전체 CSV의 시간 영역 특성 시퀀스를 추출."""
    csv_files = load_file_list(bearing_dir)
    records = []
    total = len(csv_files)

    for i, fpath in enumerate(csv_files):
        records.append(extract_time_domain_features(fpath))
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  특징 추출: {i + 1}/{total}")

    return pd.DataFrame(records)


def _shade_early_late(ax, early_start, early_end, late_start, late_end):
    """early/late 구간을 실제 파일 번호 범위로 음영 표시."""
    ax.axvspan(early_start, early_end, color=COLOR_EARLY, alpha=0.12, zorder=0)
    ax.axvspan(late_start, late_end, color=COLOR_LATE, alpha=0.12, zorder=0)
    ax.axvline(early_end, color=COLOR_EARLY, linestyle='--', linewidth=0.9, alpha=0.7)
    ax.axvline(late_start, color=COLOR_LATE, linestyle='--', linewidth=0.9, alpha=0.7)


def plot_time_domain_analysis(bearing_dir=BEARING_DIR, bearing_name=None):
    """
    베어링 수명 전체의 시간 영역 특성 추이를 시각화한다.

    x축: acc 파일의 실제 번호 (Cond1_1 예: 1 ~ 2803)
    음영: 정상 File 1~100, 이상 File (N-99)~N
    """
    if bearing_name is None:
        bearing_name = os.path.basename(os.path.normpath(bearing_dir))

    print(f"[{bearing_name}] 시간 영역 특성 추출 시작")
    df = extract_bearing_time_features(bearing_dir)

    file_numbers = df['file_number'].values
    n_files = len(df)

    early_start = int(file_numbers[0])
    early_end = int(file_numbers[N_EARLY_FILES - 1])
    late_start = int(file_numbers[n_files - N_LATE_FILES])
    late_end = int(file_numbers[-1])

    feature_pairs = [
        ('horizontal_rms', 'vertical_rms', 'RMS'),
        ('horizontal_crest_factor', 'vertical_crest_factor', 'Crest Factor'),
        ('horizontal_kurtosis', 'vertical_kurtosis', 'Kurtosis'),
        ('horizontal_skewness', 'vertical_skewness', 'Skewness'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 16), sharex=True)
    fig.suptitle(
        f'{bearing_name} - Time Domain Feature Trends',
        fontsize=15,
        fontweight='bold',
        y=0.995,
    )
    # 부제를 별도 영역에 두어 subplot 제목과 겹치지 않게 함
    fig.text(
        0.5, 0.975,
        f'전체 파일: {early_start} ~ {late_end} (총 {n_files}개)  |  '
        f'정상(early): {early_start}~{early_end} ({N_EARLY_FILES}개)  |  '
        f'이상(late): {late_start}~{late_end} ({N_LATE_FILES}개)',
        ha='center',
        va='top',
        fontsize=10,
    )

    xticks = sorted(set([early_start, early_end, late_start, late_end]))

    for row, (h_col, v_col, title) in enumerate(feature_pairs):
        ax_h = axes[row, 0]
        ax_v = axes[row, 1]

        _shade_early_late(ax_h, early_start, early_end, late_start, late_end)
        _shade_early_late(ax_v, early_start, early_end, late_start, late_end)

        ax_h.plot(file_numbers, df[h_col], color=COLOR_H, linewidth=0.6, alpha=0.85)
        ax_v.plot(file_numbers, df[v_col], color=COLOR_V, linewidth=0.6, alpha=0.85)

        ax_h.set_title(f'Horizontal - {title}', pad=8)
        ax_v.set_title(f'Vertical - {title}', pad=8)
        ax_h.set_ylabel(title, labelpad=6)
        ax_v.set_ylabel(title, labelpad=6)
        ax_h.grid(True, alpha=0.3)
        ax_v.grid(True, alpha=0.3)

    # 범례는 subplot 밖(부제 아래)에 공통으로 표시
    legend_handles = [
        Patch(facecolor=COLOR_EARLY, alpha=0.35, label=f'정상 File {early_start}~{early_end}'),
        Patch(facecolor=COLOR_LATE, alpha=0.35, label=f'이상 File {late_start}~{late_end}'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.948),
        ncol=2,
        fontsize=9,
        frameon=True,
        framealpha=0.95,
    )

    # 아래 행만 x축 라벨 표시 (위 행 tick이 아래 제목과 겹치는 것 방지)
    for ax in axes[:-1, :].ravel():
        ax.tick_params(labelbottom=False)

    for ax in axes[-1, :]:
        ax.set_xlabel('File Number (acc_XXXXX → 실제 번호)', labelpad=8)

    for ax in axes.ravel():
        ax.set_xticks(xticks)
        ax.set_xlim(early_start, late_end)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

    for ax in axes[-1, :]:
        ax.set_xticklabels(xticks, rotation=15, ha='right')

    # 상단(제목/부제), 하단(x라벨), 행 간격 확보
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.88,
        bottom=0.07,
        hspace=0.38,
        wspace=0.22,
    )
    plt.show()

    print(
        f"[INFO] {bearing_name} 시간 영역 분석 완료 — "
        f"총 {n_files}개 파일, "
        f"정상 {early_start}~{early_end}, "
        f"이상 {late_start}~{late_end}"
    )
    return df


if __name__ == '__main__':
    df = plot_time_domain_analysis(BEARING_DIR, bearing_name='Cond1_1')
    print("\n[샘플] 처음 5개 파일의 특성값:")
    print(df.head().to_string())
    print("\n[샘플] 마지막 5개 파일의 특성값:")
    print(df.tail().to_string())

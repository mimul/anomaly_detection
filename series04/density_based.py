"""
4편 - 밀도 기반 중 트리 기반(tree-based) 기법 예제, Isolation Forest

시계열을 슬라이딩 윈도우로 잘라 구간별 통계량(평균, 표준편차, 범위, 기울기)을
추출하고, 그 특징 공간에서 Isolation Forest로 이상 구간을 분리한다. 시간
순서를 모델에 직접 넣지 않고 구간을 독립적인 표본처럼 다룬다는 점이
예측 기반 기법과의 차이다.

실행:
    python series04/density_based.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

from common import generate_timeseries

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW = 20


def extract_window_features(series, window_len):
    """길이 window_len인 슬라이딩 윈도우마다 평균·표준편차·범위·기울기를 추출한다"""
    n_points = len(series)
    centers, feats = [], []
    for i in range(n_points - window_len + 1):
        window_vals = series[i:i + window_len]
        slope = np.polyfit(np.arange(window_len), window_vals, 1)[0]
        feats.append([window_vals.mean(), window_vals.std(), window_vals.max() - window_vals.min(), slope])
        centers.append(i + window_len // 2)
    return np.array(centers), np.array(feats)


def tree_based_anomaly_detection():
    """윈도우 특징에 Isolation Forest를 적합해 이상 구간의 중심 시점을 판정한다"""
    _, series, label, anomalies = generate_timeseries()
    centers, feats = extract_window_features(series, WINDOW)

    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    pred = clf.fit_predict(feats)  # -1: 이상, 1: 정상
    flagged_centers = centers[pred == -1]

    print(f"[트리 기반] window={WINDOW}, 이상으로 판정된 윈도우 수={len(flagged_centers)}"
          f" / 전체 {len(centers)}")
    for a in anomalies:
        hit = np.any((flagged_centers >= a["start"] - WINDOW // 2)
                      & (flagged_centers <= a["end"] + WINDOW // 2))
        print(f"  - {a['type']} 이상(구간 {a['start']}~{a['end']}) 탐지 여부: {hit}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(series, color="#999999", lw=1)
    for a in anomalies:
        ax.axvspan(a["start"] - 0.5, a["end"] + 0.5, color="#111111", alpha=0.15)
    ax.scatter(flagged_centers, series[flagged_centers], color="#111111", s=20, zorder=3,
               label="Isolation Forest flagged window center")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()

    path = os.path.join(OUTPUT_DIR, "tree-isolation-forest.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


if __name__ == "__main__":
    tree_based_anomaly_detection()

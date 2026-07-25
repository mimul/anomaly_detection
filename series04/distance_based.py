"""
4편 - 거리 기반 중 불일치 기반(discord-based) 기법 예제, Matrix Profile

라이브러리 없이 슬라이딩 윈도우 유클리드 거리로 Matrix Profile을 직접 계산해,
자기 자신과 가장 다른 부분열(discord)을 찾는 방식을 보여준다.

실행:
    python series04/distance_based.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from common import SEASONAL_PERIOD, generate_timeseries

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def matrix_profile(series, window):
    """길이 m인 모든 부분열끼리 z-정규화 유클리드 거리를 계산해 Matrix Profile을 구한다"""
    n_points = len(series)
    n_sub = n_points - window + 1
    windows = np.array([series[i:i + window] for i in range(n_sub)])
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True) + 1e-8
    windows_norm = (windows - means) / stds

    profile = np.full(n_sub, np.inf)
    exclusion = window // 2
    for i in range(n_sub):
        dists = np.sqrt(((windows_norm - windows_norm[i]) ** 2).sum(axis=1))
        exclude_start, exclude_end = max(0, i - exclusion), min(n_sub, i + exclusion + 1)
        dists[exclude_start:exclude_end] = np.inf
        profile[i] = dists.min()
    return profile


def discord_based_anomaly_detection():
    """Matrix Profile에서 값이 가장 큰 부분열(discord)을 이상으로 판정한다"""
    _, series, label, anomalies = generate_timeseries()
    window = SEASONAL_PERIOD
    profile = matrix_profile(series, window)

    top_k = 3
    order = np.argsort(profile)[::-1]
    discord_starts = []
    for i in order:
        if all(abs(i - s) >= window // 2 for s in discord_starts):
            discord_starts.append(i)
        if len(discord_starts) == top_k:
            break

    print(f"[불일치 기반] window={window}, 상위 {top_k}개 discord 시작 위치={discord_starts}, "
          f"거리={[round(profile[i], 3) for i in discord_starts]}")
    for a in anomalies:
        hit = any(s <= a["end"] and s + window >= a["start"] for s in discord_starts)
        print(f"  - {a['type']} 이상(구간 {a['start']}~{a['end']}) 탐지 여부: {hit}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax1.plot(series, color="#999999", lw=1)
    for a in anomalies:
        ax1.axvspan(a["start"] - 0.5, a["end"] + 0.5, color="#111111", alpha=0.25)
    ax1.set_ylabel("value")

    ax2.plot(range(window // 2, window // 2 + len(profile)), profile, color="#111111", lw=1)
    for s in discord_starts:
        ax2.axvspan(s, s + window, color="#111111", alpha=0.15)
    ax2.set_xlabel("time")
    ax2.set_ylabel("matrix profile")

    path = os.path.join(OUTPUT_DIR, "discord-matrix-profile.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


if __name__ == "__main__":
    discord_based_anomaly_detection()

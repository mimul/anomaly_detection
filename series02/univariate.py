"""
2편 - 확률분포 최우추정, 1변량 홀텔링 이론, 비정규분포

블로그 글에 실린 세 이미지(정규분포 최우추정 적합, 홀텔링 이론의 이상 점수와
임계값, 비정규분포 최우추정 적합)를 생성한다. 1편과 같은 5센서 생산 라인
데이터를 사용한다.

실행:
    python series02/univariate.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from common import generate_line_dataset

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_mle_fit(df_train):
    """정상 데이터(temp1)에 정규분포를 최우추정으로 적합한다"""
    df_normal = df_train[df_train["label"] == "normal"].dropna(subset=["temp1"])
    x = df_normal["temp1"].to_numpy()

    fitted_mean, fitted_std = stats.norm.fit(x)
    print(f"[정규분포 MLE] fitted_mean={fitted_mean:.3f}, fitted_std={fitted_std:.3f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(x, stat="density", color="#bbbbbb", ax=ax, label="temp1 histogram")
    x_line = np.linspace(x.min() - 5, x.max() + 5, 200)
    ax.plot(x_line, stats.norm.pdf(x_line, loc=fitted_mean, scale=fitted_std), color="#111111", lw=2,
            label="fitted normal")
    ax.legend()

    path = os.path.join(OUTPUT_DIR, "mle-fit.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"저장됨: {path}")


def plot_hotelling_threshold(df_train, df_inference):
    """홀텔링 이론(1변량, 샘플 크기가 큰 경우)으로 temp2의 이상 점수와 임계값을 구한다"""
    df_normal = df_train[df_train["label"] == "normal"].dropna(subset=["temp2"])
    x_train = df_normal["temp2"].to_numpy()

    fitted_mean = np.mean(x_train)
    fitted_std = np.std(x_train, ddof=0)

    TARGET_FP_RATE = 0.0027  # 정규분포 3시그마에 해당하는 목표 오탐률
    anomaly_threshold = stats.chi2.ppf(1 - TARGET_FP_RATE, df=1)
    print(f"[홀텔링 1변량] fitted_mean={fitted_mean:.3f}, fitted_std={fitted_std:.3f}, threshold={anomaly_threshold:.3f}")

    df_inf = df_inference.dropna(subset=["temp2"]).copy()
    x_inf = df_inf["temp2"].to_numpy()
    df_inf["anomaly_score"] = ((x_inf - fitted_mean) / fitted_std) ** 2
    pred = np.where(df_inf["anomaly_score"] > anomaly_threshold, "anomaly", "normal")
    print(f"[홀텔링 1변량] 추론 결과: {pd_value_counts(pred)}")

    fig, ax = plt.subplots(figsize=(5, 5))
    df_norm = df_inf[df_inf["label"] == "normal"]
    df_anom = df_inf[df_inf["label"] == "anomaly"]
    sns.histplot(df_norm["anomaly_score"], binwidth=1, color="#bbbbbb", ax=ax,
                 stat="density", label="normal")
    sns.histplot(df_anom["anomaly_score"], binwidth=1, color="#111111", ax=ax,
                 stat="density", label="anomaly")
    ax.axvline(anomaly_threshold, color="black", lw=2)
    ax.text(anomaly_threshold, ax.get_ylim()[1] * 0.9, f"threshold={anomaly_threshold:.2f}",
            ha="center", va="bottom")
    ax.legend()

    path = os.path.join(OUTPUT_DIR, "hotelling-threshold.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"저장됨: {path}")


def plot_nonnormal_fit():
    """정규분포를 따르지 않는 데이터(감마분포)에 감마분포를 최우추정으로 적합한다"""
    x_train = stats.gamma.rvs(a=3, scale=2, size=1000, random_state=42)
    params = stats.gamma.fit(x_train, floc=0)
    gamma_shape, gamma_scale = params[0], params[2]
    print(f"[감마분포 MLE] gamma_shape={gamma_shape:.3f}, gamma_scale={gamma_scale:.3f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(x_train, stat="density", color="#bbbbbb", ax=ax, label="sample histogram")
    x_line = np.linspace(0, x_train.max(), 200)
    ax.plot(x_line, stats.gamma.pdf(x_line, a=gamma_shape, scale=gamma_scale), color="#111111", lw=2,
            label="fitted gamma")
    ax.legend()

    path = os.path.join(OUTPUT_DIR, "nonnormal-fit.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"저장됨: {path}")


def pd_value_counts(arr):
    values, counts = np.unique(arr, return_counts=True)
    return dict(zip(values, counts))


if __name__ == "__main__":
    df_train, df_inference = generate_line_dataset()

    plot_mle_fit(df_train)
    plot_hotelling_threshold(df_train, df_inference)
    plot_nonnormal_fit()

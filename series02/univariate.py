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

    mu, sigma = stats.norm.fit(x)
    print(f"[정규분포 MLE] mu={mu:.3f}, sigma={sigma:.3f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(x, stat="density", color="#bbbbbb", ax=ax, label="temp1 histogram")
    xline = np.linspace(x.min() - 5, x.max() + 5, 200)
    ax.plot(xline, stats.norm.pdf(xline, loc=mu, scale=sigma), color="#111111", lw=2,
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

    mu = np.mean(x_train)
    sigma = np.std(x_train, ddof=0)

    TARGET_FP_RATE = 0.0027  # 정규분포 3시그마에 해당하는 목표 오탐률
    a_th = stats.chi2.ppf(1 - TARGET_FP_RATE, df=1)
    print(f"[홀텔링 1변량] mu={mu:.3f}, sigma={sigma:.3f}, threshold={a_th:.3f}")

    df_inf = df_inference.dropna(subset=["temp2"]).copy()
    x_inf = df_inf["temp2"].to_numpy()
    df_inf["anomaly_score"] = ((x_inf - mu) / sigma) ** 2
    pred = np.where(df_inf["anomaly_score"] > a_th, "anomaly", "normal")
    print(f"[홀텔링 1변량] 추론 결과: {pd_value_counts(pred)}")

    fig, ax = plt.subplots(figsize=(5, 5))
    df_norm = df_inf[df_inf["label"] == "normal"]
    df_anom = df_inf[df_inf["label"] == "anomaly"]
    sns.histplot(df_norm["anomaly_score"], binwidth=1, color="#bbbbbb", ax=ax,
                 stat="density", label="normal")
    sns.histplot(df_anom["anomaly_score"], binwidth=1, color="#111111", ax=ax,
                 stat="density", label="anomaly")
    ax.axvline(a_th, color="black", lw=2)
    ax.text(a_th, ax.get_ylim()[1] * 0.9, f"threshold={a_th:.2f}",
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
    k, theta = params[0], params[2]
    print(f"[감마분포 MLE] k={k:.3f}, theta={theta:.3f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(x_train, stat="density", color="#bbbbbb", ax=ax, label="sample histogram")
    xline = np.linspace(0, x_train.max(), 200)
    ax.plot(xline, stats.gamma.pdf(xline, a=k, scale=theta), color="#111111", lw=2,
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

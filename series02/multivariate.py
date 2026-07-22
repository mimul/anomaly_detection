"""
2편 - 다변량 홀텔링 이론과 마할라노비스-다구치법

블로그 글에 실린 두 이미지(다변량 홀텔링 이론의 결정 경계, 마할라노비스-다구치법의
변수별 SN비)를 생성한다. 1편과 같은 5센서 생산 라인 데이터에서 temp1, temp2
두 변수를 사용한다.

실행:
    python series02/multivariate.py
"""

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from common import generate_line_dataset

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_FP_RATE = 0.0027  # 정규분포 3시그마에 해당하는 목표 오탐률
FEATURES = ["temp2", "temp1"]


def fit_multivariate_normal(df_train):
    df_dropna = df_train.dropna(subset=FEATURES)
    df_normal = df_dropna[df_dropna["label"] == "normal"]
    X_train = df_normal[FEATURES].to_numpy()

    mu = np.mean(X_train, axis=0)
    Sigma = np.cov(X_train.T, ddof=0)
    n_features = X_train.shape[1]
    a_th = stats.chi2.ppf(1 - TARGET_FP_RATE, df=n_features)
    print(f"[다변량 정규분포] mu={mu}, a_th={a_th:.3f}")
    return mu, Sigma, a_th


def anomaly_scores(X, mu, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    X_dev = X - mu
    return np.sum((X_dev @ Sigma_inv) * X_dev, axis=1)


def plot_multivariate_hotelling(df_inference, mu, Sigma, a_th):
    df_inf = df_inference.dropna(subset=FEATURES).copy()
    X_inf = df_inf[FEATURES].to_numpy()
    scores = anomaly_scores(X_inf, mu, Sigma)
    df_inf["prediction"] = np.where(scores > a_th, "anomaly", "normal")
    print(f"[다변량 홀텔링] 이상 판정 수={sum(df_inf['prediction'] == 'anomaly')}")

    x1 = np.linspace(df_inf["temp2"].min() - 5, df_inf["temp2"].max() + 5, 200)
    x2 = np.linspace(df_inf["temp1"].min() - 5, df_inf["temp1"].max() + 5, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    scores_grid = anomaly_scores(X_grid, mu, Sigma)
    pred_grid = np.where(scores_grid > a_th, 0, 1).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(X1, X2, pred_grid, cmap=cm.gray, alpha=0.5)
    sns.scatterplot(data=df_inf, x="temp2", y="temp1", hue="label",
                     palette=["#999999", "#111111"], ax=ax)

    path = os.path.join(OUTPUT_DIR, "multivariate-hotelling-boundary.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"저장됨: {path}")

    return df_inf, scores


def plot_mahalanobis_taguchi_sn(df_inf, mu, Sigma, a_th):
    """이상 판정된 데이터에 대해 변수별 SN비를 구해, 어느 변수가 이상에 기여했는지 본다"""
    df_anomaly = df_inf[df_inf["prediction"] == "anomaly"].reset_index(drop=True)
    var = np.diag(Sigma)  # 각 변수의 분산 (분산공분산행렬의 대각성분)

    sn_records = []
    for _, row in df_anomaly.iterrows():
        for j, feature in enumerate(FEATURES):
            sn_ratio = 10 * np.log10((row[feature] - mu[j]) ** 2 / var[j])
            sn_records.append({"index": _, "feature": feature, "sn_ratio": sn_ratio})

    import pandas as pd
    df_sn = pd.DataFrame(sn_records)
    if df_sn.empty:
        print("[마할라노비스-다구치법] 이상 판정된 데이터가 없어 SN비를 계산하지 않았다")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(data=df_sn, x="index", y="sn_ratio", hue="feature",
                palette=["#999999", "#111111"], ax=ax)
    ax.set_xlabel("anomaly index")
    ax.set_ylabel("SN ratio (dB)")
    ax.legend(title="feature")

    path = os.path.join(OUTPUT_DIR, "mahalanobis-sn.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"저장됨: {path}")


if __name__ == "__main__":
    df_train, df_inference = generate_line_dataset()
    mu, Sigma, a_th = fit_multivariate_normal(df_train)
    df_inf, scores = plot_multivariate_hotelling(df_inference, mu, Sigma, a_th)
    plot_mahalanobis_taguchi_sn(df_inf, mu, Sigma, a_th)

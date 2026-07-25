"""
2편 - GMM과 kNN을 이용한 비모수적 이상탐지

블로그 글에 실린 두 이미지(GMM의 확률밀도와 결정 경계, k 값에 따른 kNN 결정 경계
비교)를 생성한다. 정상 군집이 두 개인 2차원 합성 데이터를 사용한다.

실행:
    python series02/nonparametric.py
"""

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import generate_bimodal_dataset

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_FP_RATE = 0.0027  # 정규분포 3시그마에 해당하는 목표 오탐률


def run_gmm(X_train, df_inference):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_train)

    train_scores = -gmm.score_samples(X_train)
    anomaly_threshold = np.quantile(train_scores, 1 - TARGET_FP_RATE)
    print(f"[GMM] 두 정상 군집의 평균={gmm.means_.tolist()}, threshold={anomaly_threshold:.3f}")

    X_inf = df_inference[["x1", "x2"]].to_numpy()
    inf_scores = -gmm.score_samples(X_inf)
    pred = np.where(inf_scores > anomaly_threshold, "anomaly", "normal")
    df_inference = df_inference.copy()
    df_inference["prediction"] = pred
    print(f"[GMM] 이상 판정 수={sum(pred == 'anomaly')}")

    x1 = np.linspace(df_inference["x1"].min() - 2, df_inference["x1"].max() + 2, 200)
    x2 = np.linspace(df_inference["x2"].min() - 1, df_inference["x2"].max() + 1, 200)
    mesh_x1, mesh_x2 = np.meshgrid(x1, x2)
    X_grid = np.c_[mesh_x1.ravel(), mesh_x2.ravel()]
    grid_scores = -gmm.score_samples(X_grid)
    pred_grid = np.where(grid_scores > anomaly_threshold, 0, 1).reshape(mesh_x1.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(mesh_x1, mesh_x2, pred_grid, cmap=cm.gray, alpha=0.5)
    sns.scatterplot(data=df_inference, x="x1", y="x2", hue="label",
                     palette=["#999999", "#111111"], ax=ax)

    path = os.path.join(OUTPUT_DIR, "gmm-boundary.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"저장됨: {path}")


def run_knn(X_train, df_inference):
    X_for_train, X_for_threshold = train_test_split(X_train, train_size=0.5, random_state=42)
    X_inf = df_inference[["x1", "x2"]].to_numpy()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
    axes = axes.flatten()
    axes[-1].axis("off")  # k값 3개 + 빈 칸 1개

    for i, k in enumerate([1, 5, 25]):
        knn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="minkowski")
        pipe = Pipeline([("scaler", StandardScaler()), ("knn", knn)])
        pipe.fit(X_for_train)

        X_th_std = pipe.named_steps["scaler"].transform(X_for_threshold)
        distances, _ = pipe.named_steps["knn"].kneighbors(X_th_std)
        anomaly_threshold = np.quantile(distances[:, -1], 1 - TARGET_FP_RATE)

        x1 = np.linspace(df_inference["x1"].min() - 2, df_inference["x1"].max() + 2, 150)
        x2 = np.linspace(df_inference["x2"].min() - 1, df_inference["x2"].max() + 1, 150)
        mesh_x1, mesh_x2 = np.meshgrid(x1, x2)
        X_grid = np.c_[mesh_x1.ravel(), mesh_x2.ravel()]
        X_grid_std = pipe.named_steps["scaler"].transform(X_grid)
        distances_grid, _ = pipe.named_steps["knn"].kneighbors(X_grid_std)
        pred_grid = np.where(distances_grid[:, -1] > anomaly_threshold, 0, 1).reshape(mesh_x1.shape)

        axes[i].contourf(mesh_x1, mesh_x2, pred_grid, cmap=cm.gray, alpha=0.5)
        sns.scatterplot(data=df_inference, x="x1", y="x2", hue="label",
                         palette=["#999999", "#111111"], ax=axes[i], legend=(i == 0),
                         s=12)
        axes[i].set_title(f"k={k}", fontsize=10)
        axes[i].tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "knn-k-comparison.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"저장됨: {path}")


if __name__ == "__main__":
    X_train, df_inference = generate_bimodal_dataset()
    run_gmm(X_train, df_inference)
    run_knn(X_train, df_inference)

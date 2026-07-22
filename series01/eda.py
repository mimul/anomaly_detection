"""
1편 - 탐색적 데이터 분석과 시각화

블로그 글에 실린 두 이미지(변수별 히스토그램, PCA 2차원 산점도)를 생성한다.
요약 통계량, 결측치 비율, 상관계수 행렬은 콘솔에 출력만 한다.

실행:
    python series01/eda.py
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from common import generate_dataset

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_summary(df_normal):
    df_normal_val = df_normal.drop("label", axis=1)

    print("=== 정상 데이터 요약 통계량 ===")
    print(df_normal_val.describe())

    print("\n=== 결측치 비율 ===")
    print((df_normal.isnull().sum() / len(df_normal)).round(3))

    print("\n=== 상관계수 행렬 ===")
    print(df_normal_val.corr(method="pearson").round(3))

    return df_normal_val


def plot_histogram(df, df_normal_val, df_anomaly_val):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))
    for i, colname in enumerate(df_normal_val.columns):
        sns.histplot(data=df_normal_val[colname], bins="sturges",
                     color="#bbbbbb", ax=axes[i], stat="density", label="정상")
        sns.histplot(data=df_anomaly_val[colname], bins="sturges",
                     color="#111111", ax=axes[i], stat="density", label="이상")
        axes[i].legend()
        axes[i].set_title(colname, fontsize=14)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "eda-histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장됨: {path}")


def plot_pca(df):
    df_dropna = df.dropna()
    df_normal_dropna = df_dropna[df_dropna["label"] == "normal"].drop("label", axis=1)
    df_anomaly_dropna = df_dropna[df_dropna["label"] == "anomaly"].drop("label", axis=1)

    X_normal = df_normal_dropna.to_numpy()
    X_anomaly = df_anomaly_dropna.to_numpy()

    # 정상 데이터로만 주성분을 학습한다
    pca = PCA(n_components=2)
    pca.fit(X_normal)
    X_normal_2d = pca.transform(X_normal)
    X_anomaly_2d = pca.transform(X_anomaly)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    sns.scatterplot(x=X_normal_2d[:, 0], y=X_normal_2d[:, 1],
                     color="#999999", ax=ax, label="정상")
    sns.scatterplot(x=X_anomaly_2d[:, 0], y=X_anomaly_2d[:, 1],
                     color="#111111", ax=ax, label="이상")
    ax.legend()

    path = os.path.join(OUTPUT_DIR, "eda-pca.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장됨: {path}")


if __name__ == "__main__":
    df_train, _ = generate_dataset()
    df_normal = df_train[df_train["label"] == "normal"]
    df_anomaly_val = df_train[df_train["label"] == "anomaly"].drop("label", axis=1)

    df_normal_val = print_summary(df_normal)
    plot_histogram(df_train, df_normal_val, df_anomaly_val)
    plot_pca(df_train)

"""
1편 - SVM과 로지스틱 회귀를 이용한 지도학습 기반 이상탐지

블로그 글에 실린 네 이미지(SVM 결정 경계, 감마 값 비교, 로지스틱 회귀 결정 경계,
로지스틱 회귀 클래스 확률)를 생성한다. 두 변수(temp1, temp2)만 사용해
결정 경계를 2차원으로 비교할 수 있게 한다. 감마 값별 학습·추론 탐지 건수와
두 모델의 추론 혼동행렬을 콘솔에 출력해 블로그 글의 수치를 재현한다.

실행:
    python series01/supervised.py
"""

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import generate_dataset

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_xy(df):
    df_dropna = df.dropna(subset=["temp2", "temp1"])
    X = df_dropna[["temp2", "temp1"]].to_numpy()
    y = df_dropna["label"].to_numpy()
    return df_dropna, X, y


def plot_decision_boundary(clf, df_dropna, X, path, title=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    DecisionBoundaryDisplay.from_estimator(
        clf, X, response_method="predict",
        xlabel="temp2", ylabel="temp1",
        cmap=cm.gray, alpha=0.5, ax=ax,
    )
    sns.scatterplot(data=df_dropna, x="temp2", y="temp1",
                     hue="label", palette=["#999999", "#111111"], ax=ax)
    if title:
        ax.set_title(title)
    plt.savefig(path, dpi=100)  # 이미지 폭을 600px 이내로 맞춘다 (5in x 100dpi = 500px)
    plt.close()
    print(f"저장됨: {path}")


def run_svm(df_dropna, X, y, X_inf, y_inf):
    # SVM은 거리 기반 알고리즘이므로 표준화와 파이프라인으로 묶는다
    svm = SVC(C=10, gamma=0.1)
    clf = Pipeline([("scaler", StandardScaler()), ("svm", svm)])
    clf.fit(X, y)
    plot_decision_boundary(clf, df_dropna, X,
                            os.path.join(OUTPUT_DIR, "svm-boundary.png"))

    # 감마 값에 따른 결정 경계와 과적합 경향 비교
    # 블로그 본문 이미지 폭을 600px 이내로 맞추기 위해 1행 4열 대신 2행 2열로 배치한다
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
    axes = axes.flatten()
    for i, gamma in enumerate([10, 1, 0.1, 0.01]):
        svm_i = SVC(C=10, gamma=gamma)
        clf_i = Pipeline([("scaler", StandardScaler()), ("svm", svm_i)])
        clf_i.fit(X, y)
        # 학습 데이터에서는 잘 잡아도 추론 데이터에서는 놓치는 과적합 경향을 수치로 확인한다
        n_train_anomaly = int(np.sum(y == "anomaly"))
        train_caught = int(np.sum((clf_i.predict(X) == "anomaly") & (y == "anomaly")))
        print(f"[SVM gamma={gamma}] 학습 데이터 이상 {train_caught}/{n_train_anomaly}건 탐지, "
              f"추론 데이터 혼동행렬(실제 x 예측):\n{pd.crosstab(y_inf, clf_i.predict(X_inf))}")
        DecisionBoundaryDisplay.from_estimator(
            clf_i, X, response_method="predict",
            xlabel="temp2", ylabel="temp1",
            cmap=cm.gray, alpha=0.5, ax=axes[i],
        )
        sns.scatterplot(data=df_dropna, x="temp2", y="temp1",
                         hue="label", palette=["#999999", "#111111"], ax=axes[i],
                         legend=(i == 0))
        axes[i].set_title(f"gamma={gamma}", fontsize=10)
        axes[i].tick_params(labelsize=7)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "svm-overfitting.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"저장됨: {path}")

    return clf


def run_logistic_regression(df_dropna, X, y):
    clf = LogisticRegression(penalty=None)
    clf.fit(X, y)
    plot_decision_boundary(clf, df_dropna, X,
                            os.path.join(OUTPUT_DIR, "logreg-boundary.png"))

    # 클래스 확률(이상 점수) 등고선
    temp2_axis = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    temp1_axis = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X_temp2, X_temp1 = np.meshgrid(temp2_axis, temp1_axis)
    X_grid = np.c_[X_temp2.ravel(), X_temp1.ravel()]
    anomaly_proba = clf.predict_proba(X_grid)[:, 0].reshape(X_temp2.shape)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    contour = ax.contourf(X_temp2, X_temp1, anomaly_proba, levels=50, cmap="Greys", vmin=0, vmax=1)
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.25, 0.5, 0.75, 1])
    colorbar.set_label("anomaly probability", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    # 이상 확률이 1에 가까운 검은 배경 위에서도 이상 데이터가 보이도록 흰 테두리를 두른다
    df_normal = df_dropna[df_dropna["label"] == "normal"]
    df_anomaly = df_dropna[df_dropna["label"] == "anomaly"]
    ax.scatter(df_normal["temp2"], df_normal["temp1"], s=22, c="#999999",
               edgecolors="white", linewidths=0.5, label="normal")
    ax.scatter(df_anomaly["temp2"], df_anomaly["temp1"], s=28, c="#111111",
               edgecolors="white", linewidths=1.2, label="anomaly")
    ax.set_xlabel("temp2")
    ax.set_ylabel("temp1")
    ax.legend(title="label", loc="upper left")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "logreg-probability.png")
    plt.savefig(path, dpi=100)  # 이미지 폭을 600px 이내로 맞춘다 (5in x 100dpi = 500px)
    plt.close()
    print(f"저장됨: {path}")

    return clf


if __name__ == "__main__":
    df_train, df_inference = generate_dataset()
    df_dropna, X, y = prepare_xy(df_train)

    df_inf_dropna, X_inf, y_inf = prepare_xy(df_inference)
    svm_clf = run_svm(df_dropna, X, y, X_inf, y_inf)
    logreg_clf = run_logistic_regression(df_dropna, X, y)

    for name, model in [("SVM", svm_clf), ("로지스틱 회귀", logreg_clf)]:
        print(f"\n=== {name} 추론 결과 (실제 x 예측) ===")
        print(pd.crosstab(y_inf, model.predict(X_inf)))

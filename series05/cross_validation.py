"""
5편 - 교차검증으로 평가를 신뢰할 수 있게 만들기

같은 데이터에 K-fold와 Stratified K-fold를 적용해, 이상 데이터처럼
클래스가 불균형한 데이터에서 두 분할법이 폴드마다 이상 데이터를 어떻게
배분하는지 비교한다. 이어서 비지도학습(홀텔링 이론)과 지도학습(로지스틱
회귀)의 F값·AU-ROC를 Stratified K-fold 교차검증으로 비교한다.

실행:
    python series05/cross_validation.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from common import generate_dataset

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FP_RATE = 0.0027  # 정규분포 3시그마에 해당하는 목표 오탐률
VARS = ["temp2", "temp1"]


def fold_balance_demo(X, y):
    """K-fold와 Stratified K-fold가 폴드마다 이상 데이터를 얼마나 고르게 배분하는지 비교한다"""
    kf = KFold(n_splits=5, shuffle=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf_counts = [int((y[test_idx] == "anomaly").sum()) for _, test_idx in kf.split(X)]
    skf_counts = [int((y[test_idx] == "anomaly").sum()) for _, test_idx in skf.split(X, y)]
    print(f"[폴드별 이상 데이터 수] K-fold(순서대로 분할)={kf_counts}")
    print(f"[폴드별 이상 데이터 수] Stratified K-fold={skf_counts}")
    return kf_counts, skf_counts


def _hotelling_metrics(X_train_normal, X_test, y_test):
    mean_vec = X_train_normal.mean(axis=0)
    cov_matrix = np.cov(X_train_normal.T, ddof=0)
    centered = X_test - mean_vec
    score = np.sum(centered @ np.linalg.inv(cov_matrix) * centered, axis=1)
    anomaly_threshold = stats.chi2.ppf(1 - TARGET_FP_RATE, df=X_train_normal.shape[1])
    pred = np.where(score > anomaly_threshold, "anomaly", "normal")
    f1 = f1_score(y_test, pred, pos_label="anomaly")
    auroc = roc_auc_score((y_test == "anomaly").astype(int), score)
    return f1, auroc


def _logistic_metrics(X_train, y_train, X_test, y_test):
    model = LogisticRegression(penalty=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label="anomaly")
    proba = model.predict_proba(X_test)[:, list(model.classes_).index("anomaly")]
    auroc = roc_auc_score((y_test == "anomaly").astype(int), proba)
    return f1, auroc


def compare_models_cv(X, y):
    """Stratified 5-fold로 홀텔링(비지도)과 로지스틱 회귀(지도)의 F값·AU-ROC를 비교한다"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hotelling_f1s, hotelling_aurocs = [], []
    logistic_f1s, logistic_aurocs = [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        f1, auroc = _hotelling_metrics(X_train[y_train == "normal"], X_test, y_test)
        hotelling_f1s.append(f1)
        hotelling_aurocs.append(auroc)

        f1, auroc = _logistic_metrics(X_train, y_train, X_test, y_test)
        logistic_f1s.append(f1)
        logistic_aurocs.append(auroc)

    print(f"[Stratified 5-fold] 홀텔링(비지도) F1={np.mean(hotelling_f1s):.4f}"
          f"±{np.std(hotelling_f1s):.4f}, AU-ROC={np.mean(hotelling_aurocs):.4f}"
          f"±{np.std(hotelling_aurocs):.4f}")
    print(f"[Stratified 5-fold] 로지스틱회귀(지도) F1={np.mean(logistic_f1s):.4f}"
          f"±{np.std(logistic_f1s):.4f}, AU-ROC={np.mean(logistic_aurocs):.4f}"
          f"±{np.std(logistic_aurocs):.4f}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.35
    x = np.arange(5)
    ax.bar(x - width / 2, hotelling_f1s, width, color="#999999", label="Hotelling (unsupervised)")
    ax.bar(x + width / 2, logistic_f1s, width, color="#111111", label="Logistic regression (supervised)")
    ax.set_xticks(x, [f"fold {i + 1}" for i in range(5)])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.legend()

    path = os.path.join(OUTPUT_DIR, "cv-model-comparison.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")
    return hotelling_f1s, logistic_f1s


if __name__ == "__main__":
    df_train, df_inference = generate_dataset()
    df = pd.concat([df_train, df_inference], ignore_index=True).dropna(subset=VARS)
    X_all = df[VARS].to_numpy()
    y_all = df["label"].to_numpy()

    fold_balance_demo(X_all, y_all)
    compare_models_cv(X_all, y_all)

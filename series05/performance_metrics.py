"""
5편 - 이상탐지 모델의 성능평가 (9.3)

1편 데이터셋에 다변량 홀텔링 모델을 적합해, 라벨의 예측 정확도(정확도,
적합률, 재현율, F값, AU-ROC)와 통계모델 선택의 타당성(로그가능도, AIC,
BIC)을 같은 모델에서 함께 구한다.

실행:
    python series05/performance_metrics.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from common import generate_dataset

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FP_RATE = 0.0027  # 정규분포 3시그마에 해당하는 목표 오탐률
VARS = ["temp2", "temp1"]


def _hotelling_fit(df_train_normal):
    mu = df_train_normal[VARS].mean().to_numpy()
    sigma = np.cov(df_train_normal[VARS].to_numpy().T, ddof=0)
    return mu, sigma


def _anomaly_score(df, mu, sigma):
    dev = df[VARS].to_numpy() - mu
    return np.sum(dev @ np.linalg.inv(sigma) * dev, axis=1)


def label_based_metrics(mu, sigma, df_inf):
    """라벨의 예측 정확도(정확도, 적합률, 재현율, F값, AU-ROC)를 계산한다"""
    score = _anomaly_score(df_inf, mu, sigma)
    a_th = stats.chi2.ppf(1 - TARGET_FP_RATE, df=len(VARS))
    y_true = df_inf["label"].to_numpy()
    y_pred = np.where(score > a_th, "anomaly", "normal")

    cm = confusion_matrix(y_true, y_pred, labels=["normal", "anomaly"])
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="anomaly")
    recall = recall_score(y_true, y_pred, pos_label="anomaly")
    f1 = f1_score(y_true, y_pred, pos_label="anomaly")

    fpr, tpr, _ = roc_curve(y_true, score, pos_label="anomaly")
    auroc = auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, score, pos_label="anomaly")

    print(f"[라벨의 예측 정확도] 혼동행렬(행:실제, 열:예측, 순서 normal/anomaly)\n{cm}")
    print(f"[라벨의 예측 정확도] accuracy={accuracy:.3f}, precision={precision:.3f}, "
          f"recall={recall:.3f}, f1={f1:.3f}, AU-ROC={auroc:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.3))
    ax1.plot(rec_curve, prec_curve, color="#111111")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_title("PR curve")
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)

    ax2.plot(fpr, tpr, color="#111111", label=f"AU-ROC={auroc:.3f}")
    ax2.plot([0, 1], [0, 1], color="#999999", linestyle="--")
    ax2.set_xlabel("false positive rate")
    ax2.set_ylabel("true positive rate")
    ax2.set_title("ROC curve")
    ax2.legend()

    path = os.path.join(OUTPUT_DIR, "performance-roc-pr.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")
    return accuracy, precision, recall, f1, auroc


def statistical_model_metrics(df_train_normal, mu, sigma):
    """통계모델 선택의 타당성(로그가능도, AIC, BIC)을 계산한다"""
    n, n_features = len(df_train_normal), len(VARS)
    score = _anomaly_score(df_train_normal, mu, sigma)
    log_likelihood = (
        -n * (n_features / 2 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(sigma)))
        - 0.5 * np.sum(score)
    )
    k = n_features + n_features * (n_features + 1) / 2
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood
    print(f"[통계모델 선택의 타당성] log_likelihood={log_likelihood:.2f}, "
          f"AIC={aic:.2f}, BIC={bic:.2f}")
    return log_likelihood, aic, bic


if __name__ == "__main__":
    df_train, df_inference = generate_dataset()
    df_train = df_train.dropna(subset=VARS)
    df_inference = df_inference.dropna(subset=VARS)
    df_train_normal = df_train[df_train["label"] == "normal"]

    mu_fit, sigma_fit = _hotelling_fit(df_train_normal)
    label_based_metrics(mu_fit, sigma_fit, df_inference)
    statistical_model_metrics(df_train_normal, mu_fit, sigma_fit)

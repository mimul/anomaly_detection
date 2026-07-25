"""
5편 - 이상탐지에서의 전처리 (9.2)

1편의 5센서 데이터셋에 리스트와이즈 삭제로 결측치를 제거하고, VIF로
다중공선성을 확인한 뒤, 변수를 2개 고르는 Forward 변수선택을 F값(라벨
예측 정확도)과 AIC(통계모델 선택의 타당성) 두 기준으로 각각 수행해
결과를 비교한다.

실행:
    python series05/preprocessing.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import COLUMNS, TARGET_FP_RATE, fit_hotelling, generate_dataset, hotelling_score


def listwise_deletion_demo():
    """1편 데이터셋을 합쳐 리스트와이즈 삭제 전후의 데이터 수를 비교한다"""
    df_train, df_inference = generate_dataset()
    df = pd.concat([df_train, df_inference], ignore_index=True)
    df_dropna = df.dropna()
    print(f"[리스트와이즈 삭제] 원본 {len(df)}건 → 결측 제거 후 {len(df_dropna)}건")
    return df_dropna


def vif_demo(df_dropna):
    """센서 5개 사이의 다중공선성을 VIF로 확인한다"""
    X = sm.add_constant(df_dropna[COLUMNS])
    vif = pd.Series(
        [variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])],
        index=X.columns,
    ).drop("const")
    print(f"[VIF]\n{vif.round(3)}")
    return vif


def _hotelling_f1_cv(X, y, cv):
    """M차원 정규분포 홀텔링 모델의 F값을 교차검증으로 계산한다"""
    f1s = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test, y_test = X[train_idx], X[test_idx], y[test_idx]
        n_features = X_train.shape[1]
        mean_vec, cov_matrix = fit_hotelling(X_train)
        score = hotelling_score(X_test, mean_vec, cov_matrix)
        anomaly_threshold = stats.chi2.ppf(1 - TARGET_FP_RATE, df=n_features)
        pred = np.where(score > anomaly_threshold, "anomaly", "normal")
        f1s.append(f1_score(y_test, pred, pos_label="anomaly"))
    return float(np.mean(f1s))


def _hotelling_aic(X_normal):
    """정상 데이터에만 적합한 M차원 정규분포 홀텔링 모델의 AIC를 계산한다"""
    n_samples, n_features = X_normal.shape
    mean_vec, cov_matrix = fit_hotelling(X_normal)
    score = hotelling_score(X_normal, mean_vec, cov_matrix)
    log_likelihood = (
        -n_samples * (n_features / 2 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(cov_matrix)))
        - 0.5 * np.sum(score)
    )
    n_params = n_features + n_features * (n_features + 1) / 2
    return 2 * n_params - 2 * log_likelihood


def stepwise_forward_demo(df_dropna, n_select=2):
    """F값 기준과 AIC 기준으로 Forward 변수선택을 각각 수행해 결과를 비교한다"""
    X_all = df_dropna[COLUMNS].to_numpy()
    y_all = df_dropna["label"].to_numpy()
    X_normal = df_dropna.loc[df_dropna["label"] == "normal", COLUMNS].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def forward(score_fn, larger_is_better):
        selected, remaining = [], list(COLUMNS)
        for _ in range(n_select):
            scored = [(score_fn(selected + [c]), c) for c in remaining]
            scored.sort(key=lambda t: t[0], reverse=larger_is_better)
            _, best_var = scored[0]
            selected.append(best_var)
            remaining.remove(best_var)
        return selected

    def f1_for(vars_):
        idx = [COLUMNS.index(v) for v in vars_]
        return _hotelling_f1_cv(X_all[:, idx], y_all, skf)

    def aic_for(vars_):
        idx = [COLUMNS.index(v) for v in vars_]
        return _hotelling_aic(X_normal[:, idx])

    f1_selected = forward(f1_for, larger_is_better=True)
    aic_selected = forward(aic_for, larger_is_better=False)
    print(f"[Forward 변수선택 · F값 기준] 선택된 변수: {f1_selected}, "
          f"F값: {f1_for(f1_selected):.4f}")
    print(f"[Forward 변수선택 · AIC 기준] 선택된 변수: {aic_selected}, "
          f"AIC: {aic_for(aic_selected):.2f}")
    print(f"[비교] F값 기준으로 고른 변수의 AIC: {aic_for(f1_selected):.2f}, "
          f"AIC 기준으로 고른 변수의 F값: {f1_for(aic_selected):.4f}")
    return f1_selected, aic_selected


if __name__ == "__main__":
    df_dropna_result = listwise_deletion_demo()
    vif_demo(df_dropna_result)
    stepwise_forward_demo(df_dropna_result)

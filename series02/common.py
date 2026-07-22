"""
2편에서 공통으로 쓰는 데이터 생성 함수

1편(series01)의 5센서 생산 라인 데이터와, GMM·kNN 예제에서 쓰는
두 봉우리(bimodal)를 가진 2차원 정규분포 혼합 데이터를 만든다.
"""

import numpy as np
import pandas as pd

MU_NORMAL = [400, 350, 300, 150, 100]
SIGMA = [
    [25, 15, 10, 0, 0],
    [15, 25, 12, 0, 0],
    [10, 12, 16, 0, 0],
    [0, 0, 0, 9, 0],
    [0, 0, 0, 0, 9],
]
COLUMNS = ["temp1", "temp2", "temp3", "temp4", "temp5"]


def generate_line_dataset(seed: int = 42):
    """1편과 같은 5센서 생산 라인 데이터셋 (series01/common.py와 동일)"""
    rng = np.random.default_rng(seed=seed)

    n_normal = 1500
    normal_dataset = rng.multivariate_normal(MU_NORMAL, SIGMA, n_normal)

    mu_anomaly = [410, 370, 310, 150, 100]
    n_anomaly = 30
    anomaly_dataset = rng.multivariate_normal(mu_anomaly, SIGMA, n_anomaly)

    df_normal = pd.DataFrame(normal_dataset, columns=COLUMNS)
    df_anomaly = pd.DataFrame(anomaly_dataset, columns=COLUMNS)

    temp1_missed = rng.choice([True, False], n_normal, p=[0.1, 0.9])
    df_normal.loc[temp1_missed, "temp1"] = np.nan
    temp5_missed = rng.choice([True, False], n_normal, p=[0.05, 0.95])
    df_normal.loc[temp5_missed, "temp5"] = np.nan

    df_normal["label"] = "normal"
    df_anomaly["label"] = "anomaly"

    df_norm_train, df_norm_inference = df_normal[:1000], df_normal[1000:]
    df_anom_train, df_anom_inference = df_anomaly[:20], df_anomaly[20:]

    df_train = pd.concat([df_norm_train, df_anom_train], axis=0).reset_index(drop=True)
    df_inference = pd.concat([df_norm_inference, df_anom_inference], axis=0).reset_index(drop=True)

    return df_train, df_inference


def generate_bimodal_dataset(seed: int = 42):
    """두 개의 정상 군집(봉우리)을 가진 2차원 데이터. GMM·kNN 예제에서 사용한다."""
    rng = np.random.default_rng(seed=seed)

    mu1, sigma1 = [0, 0], [[1, 0], [0, 1]]
    mu2, sigma2 = [4, 0], [[2, 1], [1, 1]]
    X_train_1 = rng.multivariate_normal(mu1, sigma1, 600)
    X_train_2 = rng.multivariate_normal(mu2, sigma2, 600)
    X_train = np.concatenate([X_train_1, X_train_2])

    rng_inf = np.random.default_rng(seed=seed)
    X_inf_norm_1 = rng_inf.multivariate_normal(mu1, sigma1, 100)
    X_inf_norm_2 = rng_inf.multivariate_normal(mu2, sigma2, 100)
    X_inf_norm = np.concatenate([X_inf_norm_1, X_inf_norm_2])
    X_inf_anom = rng_inf.multivariate_normal([3, 2], [[1, 0], [0, 1]], 20)

    df_inference = pd.DataFrame(
        np.concatenate([X_inf_norm, X_inf_anom]), columns=["x1", "x2"]
    )
    df_inference["label"] = ["normal"] * len(X_inf_norm) + ["anomaly"] * len(X_inf_anom)

    return X_train, df_inference

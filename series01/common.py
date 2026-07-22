"""
1편에서 공통으로 쓰는 샘플 데이터 생성 함수

5개 공정으로 이뤄진 생산 라인의 이상탐지를 흉내 낸 데이터셋을 만든다.
정상 데이터는 목표 온도를 중심으로 한 정규분포를 따르고, 공정 1~3은 온도가
서로 전달되어 상관관계를 가진다. 이상 데이터는 온도 조절 기능 고장으로
평균 온도가 일정량 어긋난 것으로 가정한다. 시드가 고정돼 있어 실행할 때마다
같은 데이터가 재현된다.
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


def generate_dataset(seed: int = 42):
    """정상 데이터 1500건, 이상 데이터 30건을 생성해 학습/추론용으로 나눠 반환한다."""
    rng = np.random.default_rng(seed=seed)

    n_normal = 1500
    normal_dataset = rng.multivariate_normal(MU_NORMAL, SIGMA, n_normal)

    mu_anomaly = [410, 370, 310, 150, 100]
    n_anomaly = 30
    anomaly_dataset = rng.multivariate_normal(mu_anomaly, SIGMA, n_anomaly)

    df_normal = pd.DataFrame(normal_dataset, columns=COLUMNS)
    df_anomaly = pd.DataFrame(anomaly_dataset, columns=COLUMNS)

    # 센서 1: 10%, 센서 5: 5% 확률로 결측 발생
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

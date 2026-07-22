"""
3편 - 회귀와 베이지안 모델링에서 공용으로 쓰는 데이터 로딩 함수

중고차 가격 데이터셋(연식·주행거리·차종·판매자별 가격)을 학습·추론용으로 읽어
원-핫 인코딩까지 마친 형태로 반환한다.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

TARGET_FP_RATE = 0.0027  # 정규분포 3시그마에 해당하는 목표 오탐률


def load_usedcar_data():
    """중고차 데이터셋을 읽어 학습(정상만)·추론 데이터를 원-핫 인코딩 형태로 반환한다"""
    df_train = pd.read_csv("series03/datasets/usedcar_train.csv")
    df_train_normal = df_train[df_train["label"] == "normal"]
    df_inference = pd.read_csv("series03/datasets/usedcar_inference.csv")

    enc = OneHotEncoder(drop="first")
    enc.fit(df_train_normal[["model_name"]].to_numpy())
    cat_names = [f"model_name_{c}" for c in enc.categories_[0][1:]]

    def _encode(df):
        """원-핫 인코딩 열을 덧붙이되, 표시용으로 model_name 원본 열은 그대로 남긴다"""
        cat = enc.transform(df[["model_name"]].to_numpy()).toarray()
        df_cat = pd.DataFrame(cat, columns=cat_names, index=df.index)
        return pd.concat([df, df_cat], axis=1)

    df_train_encoded = _encode(df_train_normal)
    df_inference_encoded = _encode(df_inference)
    return df_train_encoded, df_inference_encoded, cat_names, enc

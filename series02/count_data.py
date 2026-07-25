"""
2편 - 계수 데이터의 이상탐지: 이항분포, 포아송분포, 과분산 대처

블로그 글에 실린 세 이미지(이항분포 기반 경계, 포아송분포 기반 경계,
과분산을 반영한 음이항분포 기반 경계)를 생성한다.

이항분포 예제는 원서와 같은 NBA 3점슛 성공률 데이터
(ghmagazine/python_anomaly_detection_book 저장소의 공개 데이터)를 사용한다.

포아송분포·음이항분포 예제는 기상청이 제공하는 국내 지진 실측 데이터
(series02/datasets/earthquake_kma.csv, 2020.1~2026.7, 기상청 날씨누리
지진 조회 서비스에서 내려받음)를 사용한다. 시도별 면적은 위키백과
"대한민국의 행정 구역" 문서(2019.2 기준, 대구는 2023년 군위군 편입을
반영한 최신 면적)를 참고했다.

실행:
    python series02/count_data.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import nbinom

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = (
    "https://raw.githubusercontent.com/ghmagazine/python_anomaly_detection_book"
    "/refs/heads/main/notebooks/datasets"
)
TARGET_FP_RATE = 0.0027  # 정규분포 3시그마에 해당하는 목표 오탐률

# 시도별 면적(km²). 위키백과 "대한민국의 행정 구역" 문서 기준
AREA_KM2 = {
    "서울": 605.20, "부산": 769.89, "대구": 1499.51, "인천": 1062.60,
    "광주": 501.24, "대전": 539.35, "울산": 1060.79, "세종": 464.87,
    "경기": 10183.46, "강원": 16875.03, "충북": 7407.29, "충남": 8226.14,
    "전북": 8069.05, "전남": 12318.79, "경북": 18424.10, "경남": 10539.56,
    "제주": 1849.15,
}


def _plot_boundary_line(df_inf, x_col, y_col, x_line, lower, upper, filename,
                         xlabel, ylabel, ylim=None):
    """경계선 두 개(하한·상한)와 실제 데이터를 함께 그린다"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill_between(x_line, lower, upper, color="#dddddd", label="normal region")
    ax.plot(x_line, lower, color="#111111", lw=1)
    ax.plot(x_line, upper, color="#111111", lw=1)
    sns.scatterplot(data=df_inf, x=x_col, y=y_col, hue="prediction",
                     palette=["#999999", "#111111"], ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


def binomial_anomaly_detection():
    """3점슛 성공률(성공수/시도수)을 이항분포로 모델링해 이상 선수를 찾는다"""
    df_train = pd.read_csv(f"{BASE_URL}/player_stats_2022.csv")
    df_train = df_train[df_train["FG3A"] > 0]
    x_train, n_train = df_train["FG3M"].to_numpy(), df_train["FG3A"].to_numpy()

    success_rate = np.sum(x_train) / np.sum(n_train)  # 발생률(성공 확률) 파라미터
    print(f"[이항분포] 2022시즌 평균 3점슛 성공률={success_rate:.4f}")

    df_inf = pd.read_csv(f"{BASE_URL}/player_stats_2023.csv")
    df_inf = df_inf[df_inf["FG3A"] > 0].copy()
    n_inf, x_inf = df_inf["FG3A"].to_numpy(), df_inf["FG3M"].to_numpy()

    th_lower = stats.binom.ppf(TARGET_FP_RATE / 2, n=n_inf, p=success_rate)
    th_upper = stats.binom.ppf(1 - TARGET_FP_RATE / 2, n=n_inf, p=success_rate)
    df_inf["3P%"] = x_inf / n_inf
    df_inf["prediction"] = np.where((x_inf < th_lower) | (x_inf > th_upper), "anomaly", "normal")
    print(f"[이항분포] 2023시즌 이상 판정 선수 수={sum(df_inf['prediction'] == 'anomaly')}")

    n_line = np.arange(1, 751)
    lower = stats.binom.ppf(TARGET_FP_RATE / 2, n=n_line, p=success_rate) / n_line
    upper = stats.binom.ppf(1 - TARGET_FP_RATE / 2, n=n_line, p=success_rate) / n_line
    _plot_boundary_line(df_inf, "FG3A", "3P%", n_line, lower, upper,
                         "binomial-boundary.png", "FG3A (attempts)", "3P% (success rate)")


TRAIN_START, TRAIN_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01")
TEST_START, TEST_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-07-01")
TRAIN_YEARS = (TRAIN_END - TRAIN_START).days / 365.25  # 4.0
TEST_YEARS = (TEST_END - TEST_START).days / 365.25  # 약 2.5


def _load_earthquake_counts():
    """지진 실측 데이터를 읽어 학습 기간·추론 기간의 지역별 발생 건수를 센다"""
    df = pd.read_csv("series02/datasets/earthquake_kma.csv")
    df["dt"] = pd.to_datetime(df["시간"], format="%Y.%m.%d %H:%M")
    df["region"] = df["위치"].str.split().str[0]
    df = df[df["region"].isin(AREA_KM2)]  # 북한, 해외 등 국내 시도가 아닌 행 제외

    regions = list(AREA_KM2.keys())
    area = np.array([AREA_KM2[r] for r in regions])

    df_train = df[(df["dt"] >= TRAIN_START) & (df["dt"] < TRAIN_END)]
    df_test = df[(df["dt"] >= TEST_START) & (df["dt"] < TEST_END)]
    counts_train = df_train.groupby("region").size()
    counts_test = df_test.groupby("region").size()

    x_train = np.array([counts_train.get(r, 0) for r in regions])
    x_test = np.array([counts_test.get(r, 0) for r in regions])
    return regions, area, x_train, x_test


def poisson_anomaly_detection():
    """지역 면적당 지진 발생 건수를 포아송분포로 모델링해 이상 지역을 찾는다"""
    regions, area, x_train, x_test = _load_earthquake_counts()
    t_train = area * TRAIN_YEARS  # 노출량 = 면적 x 연수

    rate_per_exposure = np.sum(x_train) / np.sum(t_train)  # 면적·연 단위당 평균 발생 수
    print(f"[포아송분포] 2020~2023년 기준 면적당 연평균 지진 발생 수={rate_per_exposure:.6f} (건/km²/년)")

    t_test = area * TEST_YEARS
    th_lower = stats.poisson.ppf(TARGET_FP_RATE / 2, mu=rate_per_exposure * t_test)
    th_upper = stats.poisson.ppf(1 - TARGET_FP_RATE / 2, mu=rate_per_exposure * t_test)
    df_inf = pd.DataFrame({"region": regions, "area": area, "count": x_test})
    df_inf["density"] = df_inf["count"] / df_inf["area"]
    df_inf["prediction"] = np.where((x_test < th_lower) | (x_test > th_upper), "anomaly", "normal")
    print(f"[포아송분포] 2024~2026년 이상 판정 지역:\n"
          f"{df_inf[df_inf['prediction']=='anomaly'][['region', 'count']]}")

    area_line = np.linspace(400, 19000, 300)
    lower = stats.poisson.ppf(TARGET_FP_RATE / 2, mu=rate_per_exposure * area_line * TEST_YEARS) / area_line
    upper = stats.poisson.ppf(1 - TARGET_FP_RATE / 2, mu=rate_per_exposure * area_line * TEST_YEARS) / area_line
    _plot_boundary_line(df_inf, "area", "density", area_line, lower, upper,
                         "poisson-boundary.png", "area (km²)", "earthquakes / area")

    return regions, area, x_train


def negative_binomial_overdispersion(regions, area, x_train):
    """포아송분포의 과분산 여부를 확인하고, 음이항분포로 보정한다"""
    t_train = area * TRAIN_YEARS
    rate_per_exposure = np.sum(x_train) / np.sum(t_train)
    pearson_chi2 = np.sum((x_train - t_train * rate_per_exposure) ** 2 / (t_train * rate_per_exposure))
    degrees_of_freedom = len(x_train) - 1
    overdispersion_ratio = pearson_chi2 / degrees_of_freedom
    print(f"[과분산 점검] 피어슨 카이제곱 통계량={pearson_chi2:.2f}, 자유도={degrees_of_freedom}, "
          f"과분산비={overdispersion_ratio:.2f} (1보다 크면 과분산)")

    def neg_log_likelihood(params, x, t):
        nb_shape, nb_theta = params
        if nb_shape <= 0 or nb_theta <= 0:
            return np.inf
        nb_prob = 1.0 / (1.0 + nb_theta * t)
        return -nbinom.logpmf(x, n=nb_shape, p=nb_prob).sum()

    # 이 최적화는 초기값에 따라 다른 지역해로 수렴하기 쉬워, 여러 초기값을 시도해
    # 음의 로그가능도가 가장 작은(가장 잘 맞는) 결과를 채택한다
    initial_guesses = [(10.0, rate_per_exposure / 10.0), (1.0, rate_per_exposure),
                        (0.5, rate_per_exposure * 2), (50.0, rate_per_exposure / 50.0)]
    best_result = None
    for init_guess in initial_guesses:
        result = minimize(
            neg_log_likelihood, x0=init_guess, args=(x_train, t_train),
            method="L-BFGS-B", bounds=[(1e-5, None), (1e-5, None)],
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    nb_shape, nb_theta = best_result.x
    print(f"[음이항분포] k={nb_shape:.4f}, theta={nb_theta:.6f} (nll={best_result.fun:.3f})")

    _, _, _, x_test = _load_earthquake_counts()
    t_test = area * TEST_YEARS

    def th(rate, t):
        return nbinom.ppf(rate, n=nb_shape, p=1 / (1 + nb_theta * t))

    th_lower, th_upper = th(TARGET_FP_RATE / 2, t_test), th(1 - TARGET_FP_RATE / 2, t_test)
    df_inf = pd.DataFrame({"region": regions, "area": area, "count": x_test})
    df_inf["density"] = df_inf["count"] / df_inf["area"]
    df_inf["prediction"] = np.where((x_test < th_lower) | (x_test > th_upper), "anomaly", "normal")
    print(f"[음이항분포] 이상 판정 지역 수={sum(df_inf['prediction'] == 'anomaly')}"
          f" (포아송분포 대비 과탐지가 줄어드는지 비교)")

    area_line = np.linspace(400, 19000, 300)
    t_line = area_line * TEST_YEARS
    lower, upper = th(TARGET_FP_RATE / 2, t_line) / area_line, th(1 - TARGET_FP_RATE / 2, t_line) / area_line
    _plot_boundary_line(df_inf, "area", "density", area_line, lower, upper,
                         "negbinom-boundary.png", "area (km²)", "earthquakes / area")


if __name__ == "__main__":
    binomial_anomaly_detection()
    regions, area, x_train = poisson_anomaly_detection()
    negative_binomial_overdispersion(regions, area, x_train)

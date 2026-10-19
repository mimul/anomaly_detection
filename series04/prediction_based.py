"""
4편 - 예측 기반 중 예측 오차 기반(forecasting-based) 기법 예제, 지수평활

정상 구간에서 Holt-Winters 지수평활 모델을 적합해 평활 계수와 초기 상태를
얻은 뒤, 추론 구간은 한 걸음씩 직접 갱신하며 한 스텝 앞 예측과 실측값의
오차를 이상 점수로 쓴다. 예측이 그 시점의 추세·계절 맥락을 반영하므로,
값 자체는 정상 범위 안에 있는 맥락적 이상도 큰 오차로 드러난다.

실행:
    python series04/prediction_based.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common import SEASONAL_PERIOD, generate_timeseries

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FP_RATE = 0.0027  # 정규분포 3시그마에 해당하는 목표 오탐률


def forecasting_based_anomaly_detection():
    """정상 구간에서 적합한 지수평활 계수로, 추론 구간의 한 스텝 앞 예측 오차를 구한다"""
    train_y, series, label, anomalies = generate_timeseries()
    period = SEASONAL_PERIOD

    fit = ExponentialSmoothing(
        train_y, trend="add", seasonal="add", seasonal_periods=period,
        initialization_method="estimated",
    ).fit()
    alpha = fit.params["smoothing_level"]
    beta = fit.params["smoothing_trend"]
    gamma = fit.params["smoothing_seasonal"]
    resid_std = np.std(fit.resid)
    print(f"[예측 오차 기반] alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}, "
          f"학습 잔차 resid_std={resid_std:.3f}")

    level = fit.level[-1]
    trend = fit.trend[-1]
    season = list(fit.season[-period:])  # season[phase]는 t % period == phase 시점의 계절 성분

    n_points = len(series)
    forecast = np.zeros(n_points)
    for j in range(n_points):
        phase = j % period
        forecast[j] = level + trend + season[phase]
        resid = series[j] - forecast[j]
        new_level = alpha * (series[j] - season[phase]) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        season[phase] = gamma * (series[j] - level) + (1 - gamma) * season[phase]

    score = ((series - forecast) / resid_std) ** 2
    anomaly_threshold = stats.chi2.ppf(1 - TARGET_FP_RATE, df=1)
    pred = np.where(score > anomaly_threshold, "anomaly", "normal")

    print(f"[예측 오차 기반] threshold={anomaly_threshold:.3f}, 이상 판정 시점 수={sum(pred == 'anomaly')}"
          f" / 전체 {n_points}")
    for a in anomalies:
        hit = np.any(pred[a["start"]:a["end"] + 1] == "anomaly")
        print(f"  - {a['type']} 이상(구간 {a['start']}~{a['end']}) 탐지 여부: {hit}")

    # 학습 잔차로 정한 임계값이 추론 구간에서 목표 오탐률을 지키는지 확인한다
    # (블로그 글: 학습 잔차 표준편차가 추론 오차를 과소추정해 오탐률이 목표를 넘는다)
    normal = label == "normal"
    n_fp = int(np.sum((pred == "anomaly") & normal))
    rms_normal = np.sqrt(np.mean(score[normal]))
    print(f"[예측 오차 기반] 정상 시점 {int(normal.sum())}개 중 오탐={n_fp}건 (오탐률 {n_fp / normal.sum():.1%}, 목표 {TARGET_FP_RATE:.2%}), "
          f"정상 구간 정규화 오차의 제곱평균제곱근={rms_normal:.2f} (1보다 크면 학습 잔차가 추론 오차를 과소추정)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax1.plot(series, color="#999999", lw=1, label="observed")
    ax1.plot(forecast, color="#111111", lw=1, ls="--", label="1-step forecast")
    for a in anomalies:
        ax1.axvspan(a["start"] - 0.5, a["end"] + 0.5, color="#111111", alpha=0.15)
    ax1.set_ylabel("value")
    ax1.legend()

    ax2.plot(score, color="#111111", lw=1)
    ax2.axhline(anomaly_threshold, color="black", lw=1, ls=":")
    ax2.set_xlabel("time")
    ax2.set_ylabel("anomaly score")

    path = os.path.join(OUTPUT_DIR, "forecast-exponential-smoothing.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


if __name__ == "__main__":
    forecasting_based_anomaly_detection()

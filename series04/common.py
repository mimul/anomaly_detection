"""
4편 - 시계열 이상탐지 예제에서 공용으로 쓰는 데이터 생성 함수

계절성과 추세가 있는 합성 시계열에 점 이상, 맥락적 이상, 집단적 이상을
하나씩 주입해, 세 갈래(거리 기반·밀도 기반·예측 기반) 대표 기법이 같은
데이터에서 어떤 이상을 잡아내는지 비교할 수 있게 한다.
"""

import numpy as np

SEASONAL_PERIOD = 50


def generate_timeseries():
    """정상 학습용 구간과, 세 가지 이상 유형을 주입한 추론용 구간을 나눠 반환한다"""
    n_points = 700
    rng = np.random.default_rng(42)
    t = np.arange(n_points)
    seasonal = 5 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
    trend = 0.01 * t
    noise = rng.normal(0, 0.5, n_points)
    series = 20 + seasonal + trend + noise

    anomalies = []

    # 점 이상: 순간적으로 정상 범위를 크게 벗어나는 스파이크
    idx = 400
    series[idx] += 8
    anomalies.append({"type": "point", "start": idx, "end": idx})

    # 맥락적 이상: 값 자체는 전역 범위 안이지만, 봉우리여야 할 위상에 골 수준의
    # 값이 끼어들어 그 시점의 계절 맥락과 어긋난다
    idx = 520
    series[idx] -= 9
    anomalies.append({"type": "contextual", "start": idx, "end": idx})

    # 집단적 이상: 일정 구간 동안 계절 패턴이 사라지고 평탄한 잡음으로 바뀐다
    start, end = 620, 640
    series[start:end] = 20 + rng.normal(0, 0.3, end - start)
    anomalies.append({"type": "collective", "start": start, "end": end})

    label = np.array(["normal"] * n_points, dtype=object)
    for a in anomalies:
        label[a["start"]:a["end"] + 1] = "anomaly"

    train_y = series[:300]
    inf_y = series[300:]
    inf_label = label[300:]
    inf_anomalies = [
        {"type": a["type"], "start": a["start"] - 300, "end": a["end"] - 300}
        for a in anomalies
    ]

    return train_y, inf_y, inf_label, inf_anomalies

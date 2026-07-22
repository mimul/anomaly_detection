"""
3편 - 베이지안 모델링을 이용한 이상탐지: 베이지안 선형회귀, MCMC, 로지스틱/포아송 회귀

선형회귀는 켤레사전분포를 쓰면 MCMC 없이 닫힌 형태로 사후분포를 구할 수 있어
scikit-learn의 BayesianRidge를 그대로 사용한다. 반면 이항 로지스틱 회귀와
포아송 회귀처럼 우도와 사전분포가 켤레 관계가 아닌 모델은 사후분포를 해석적으로
구할 수 없어, 이 스크립트에서는 Metropolis-Hastings 알고리즘을 직접 구현해
사후분포의 MCMC 표본을 얻는다.

이항 로지스틱 회귀는 2편에서도 쓴 NBA 3점슛 성공률 데이터를, 포아송 회귀는
미국 지질조사국(USGS)의 지진 데이터와 스미스소니언 박물관의 화산 데이터를 결합한
국가별 데이터를 사용한다(원 데이터 출처는 책 저장소가 정리해 배포한 CSV를 그대로 사용).

실행:
    python series03/bayesian.py
"""

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import BayesianRidge

from common import TARGET_FP_RATE, load_usedcar_data
from regression import YEARS_TO_SHOW, _plot_multivariate_grid

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = (
    "https://raw.githubusercontent.com/ghmagazine/python_anomaly_detection_book"
    "/refs/heads/main/notebooks/datasets"
)


def metropolis_hastings(log_posterior, init, step, n_iter=8000, burn=2000, seed=42):
    """대칭 정규분포를 제안분포로 쓰는 랜덤워크 Metropolis-Hastings 샘플러"""
    rng = np.random.default_rng(seed)
    n_params = len(init)
    samples = np.zeros((n_iter, n_params))
    current = np.array(init, dtype=float)
    current_lp = log_posterior(*current)
    n_accept = 0
    for i in range(n_iter):
        proposal = current + rng.normal(0, step, size=n_params)
        proposal_lp = log_posterior(*proposal)
        if np.log(rng.uniform()) < proposal_lp - current_lp:
            current, current_lp = proposal, proposal_lp
            n_accept += 1
        samples[i] = current
    accept_rate = n_accept / n_iter
    print(f"[MCMC] 채택률={accept_rate:.3f} (0.2~0.6 사이면 적절한 스텝 크기)")
    return samples[burn:]


def bayesian_linear_regression():
    """켤레사전분포를 이용해 MCMC 없이 닫힌 형태로 구하는 베이지안 선형회귀"""
    df_train, df_inference, cat_names, enc = load_usedcar_data()
    feature_cols = ["year", "odometer"] + cat_names
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["price"].to_numpy()

    model = BayesianRidge(compute_score=True)
    model.fit(X_train, y_train)
    print(f"[베이지안 선형회귀] coef={np.round(model.coef_, 3)}, "
          f"intercept={model.intercept_:.3f}")
    print(f"[베이지안 선형회귀] 잡음 정밀도 alpha={model.alpha_:.6f}, "
          f"가중치 정밀도 lambda={model.lambda_:.6f}")

    mean_train, std_train = model.predict(X_train, return_std=True)
    score_train = ((y_train - mean_train) / std_train) ** 2
    a_th = np.quantile(score_train, 1 - TARGET_FP_RATE)
    print(f"[베이지안 선형회귀] a_th={a_th:.4f}")

    X_inf = df_inference[feature_cols].to_numpy()
    y_inf = df_inference["price"].to_numpy()
    mean_inf, std_inf = model.predict(X_inf, return_std=True)
    score_inf = ((y_inf - mean_inf) / std_inf) ** 2
    pred = np.where(score_inf > a_th, "anomaly", "normal")
    print(f"[베이지안 선형회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")

    def predict_fn(X_features, y_values):
        mean, std = model.predict(X_features, return_std=True)
        score = ((y_values - mean) / std) ** 2
        return np.where(score > a_th, 0, 1)

    model_cols = list(enc.categories_[0])
    dropped_name = enc.categories_[0][0]
    _plot_multivariate_grid(model_cols, cat_names, dropped_name, predict_fn,
                             df_inference, "bayesian-linear-boundary.png")


def _load_nba_data():
    df_train = pd.read_csv(f"{BASE_URL}/player_stats_2022.csv")
    df_train = df_train[df_train["FG3A"] > 0]
    df_inf = pd.read_csv(f"{BASE_URL}/player_stats_2023.csv")
    df_inf = df_inf[df_inf["FG3A"] > 0].copy()
    df_inf["3P%"] = df_inf["FG3M"] / df_inf["FG3A"]
    return df_train, df_inf


def bayesian_logistic_regression():
    """이항 로지스틱 회귀: 3점슛 성공률의 정상 범위를 나이에 따라 조정한다"""
    df_train, df_inf = _load_nba_data()
    x_train = df_train["AGE"].to_numpy(dtype=float)
    y_train = df_train["FG3M"].to_numpy(dtype=float)
    n_train = df_train["FG3A"].to_numpy(dtype=float)
    x_mean = x_train.mean()
    xc_train = x_train - x_mean

    def log_posterior(w0, w1):
        rho = np.clip(w0 + w1 * xc_train, -30, 30)
        mu = 1 / (1 + np.exp(-rho))
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        log_likelihood = np.sum(stats.binom.logpmf(y_train, n_train, mu))
        log_prior = stats.norm.logpdf(w0, 0, 100) + stats.norm.logpdf(w1, 0, 100)
        return log_likelihood + log_prior

    posterior = metropolis_hastings(log_posterior, init=[0.0, 0.0],
                                     step=[0.006, 0.0012])
    w0_samples, w1_samples = posterior[:, 0], posterior[:, 1]
    print(f"[이항 로지스틱 회귀] w0={w0_samples.mean():.4f}±{w0_samples.std():.4f}, "
          f"w1={w1_samples.mean():.4f}±{w1_samples.std():.4f} (사후평균±사후표준편차)")

    x_inf = df_inf["AGE"].to_numpy(dtype=float)
    n_inf = df_inf["FG3A"].to_numpy(dtype=int)
    y_inf = df_inf["FG3M"].to_numpy(dtype=float)
    rho_inf = w0_samples[:, None] + w1_samples[:, None] * (x_inf[None, :] - x_mean)
    mu_inf = 1 / (1 + np.exp(-rho_inf))
    rng_inf = np.random.default_rng(42)
    draws_inf = rng_inf.binomial(n_inf[None, :], mu_inf)
    th_lower_inf = np.quantile(draws_inf, TARGET_FP_RATE / 2, axis=0)
    th_upper_inf = np.quantile(draws_inf, 1 - TARGET_FP_RATE / 2, axis=0)
    pred_inf = np.where((y_inf < th_lower_inf) | (y_inf > th_upper_inf), "anomaly", "normal")
    df_inf["prediction"] = pred_inf
    print(f"[이항 로지스틱 회귀] 추론 데이터 이상 판정 수={sum(pred_inf == 'anomaly')} / {len(pred_inf)}")
    df_anomaly = df_inf[df_inf["prediction"] == "anomaly"][
        ["PLAYER_ID", "AGE", "3P%", "FG3A", "FG3M"]].sort_values("3P%", ascending=False)
    print(f"[이항 로지스틱 회귀] 이상 판정 상세:\n{df_anomaly.to_string(index=False)}")

    # MCMC 표본이 어떻게 수렴하는지 보여주는 트레이스 플롯 (8.3절 MCMC 개념)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 3.5))
    axes[0].plot(w0_samples, color="#333333", lw=0.5)
    axes[0].set_ylabel("w0 (intercept)")
    axes[1].plot(w1_samples, color="#333333", lw=0.5)
    axes[1].set_ylabel("w1 (slope)")
    axes[1].set_xlabel("MCMC sample (post burn-in)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "mcmc-trace.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")

    # 나이에 따른 성공률 평균과 신용구간(credible interval)
    x_grid = np.linspace(19, 43, 100)
    xc_grid = x_grid - x_mean
    rho_grid = w0_samples[:, None] + w1_samples[:, None] * xc_grid[None, :]
    mu_grid = 1 / (1 + np.exp(-rho_grid))
    mu_mean = mu_grid.mean(axis=0)
    mu_lower = np.quantile(mu_grid, TARGET_FP_RATE / 2, axis=0)
    mu_upper = np.quantile(mu_grid, 1 - TARGET_FP_RATE / 2, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill_between(x_grid, mu_lower, mu_upper, color="#dddddd", label="credible region")
    ax.plot(x_grid, mu_mean, color="#111111", lw=1.5)
    df_100 = df_train[df_train["FG3A"] > 100].copy()
    df_100["3P%"] = df_100["FG3M"] / df_100["FG3A"]
    sns.scatterplot(data=df_100, x="AGE", y="3P%", color="#555555", s=18, ax=ax)
    ax.set_xlabel("AGE")
    ax.set_ylabel("3P% (success rate)")
    path = os.path.join(OUTPUT_DIR, "bayesian-logistic-fit.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")

    # 두 연령대를 대표값으로 삼아, 시도 수(FG3A) 대비 성공률의 정상 범위를 비교
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3.2))
    n_line = np.arange(1, 751)
    for ax, age_rep, title in zip(axes, [25, 35], ["AGE=25", "AGE=35"]):
        rho_rep = w0_samples + w1_samples * (age_rep - x_mean)
        mu_rep = 1 / (1 + np.exp(-rho_rep))
        # n마다 이항분포의 예측분포 표본을 만들어 하한·상한 임계값 계산
        lower, upper = np.zeros_like(n_line, dtype=float), np.zeros_like(n_line, dtype=float)
        rng = np.random.default_rng(42)
        for i, n in enumerate(n_line):
            draws = rng.binomial(n, mu_rep)
            lower[i] = np.quantile(draws, TARGET_FP_RATE / 2)
            upper[i] = np.quantile(draws, 1 - TARGET_FP_RATE / 2)
        ax.fill_between(n_line, lower / n_line, upper / n_line, color="#dddddd")
        ax.plot(n_line, lower / n_line, color="#111111", lw=1)
        ax.plot(n_line, upper / n_line, color="#111111", lw=1)
        df_age = df_inf[(df_inf["AGE"] >= age_rep - 2) & (df_inf["AGE"] <= age_rep + 2)]
        preds = np.where((df_age["FG3M"] < np.interp(df_age["FG3A"], n_line, lower)) |
                          (df_age["FG3M"] > np.interp(df_age["FG3A"], n_line, upper)),
                          "anomaly", "normal")
        sns.scatterplot(x=df_age["FG3A"], y=df_age["3P%"], hue=preds,
                         hue_order=["normal", "anomaly"], palette=["#999999", "#111111"],
                         ax=ax, s=16, legend=False)
        ax.set_xlabel("FG3A", fontsize=8)
        ax.set_ylabel("3P%", fontsize=8)
        ax.set_title(f"{title} (±2)", fontsize=9)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bayesian-logistic-boundary.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


def bayesian_poisson_regression():
    """포아송 회귀: 화산 수(active_since_1800)에 따른 지진 발생 수의 정상 범위"""
    df_train = pd.read_csv("series03/datasets/earthquakes_volcanoes_train.csv")
    df_inf = pd.read_csv("series03/datasets/earthquakes_volcanoes_test.csv")
    x_train = df_train["active_since_1800"].to_numpy(dtype=float)
    y_train = df_train["earthquakes"].to_numpy(dtype=float)
    x_mean = x_train.mean()
    xc_train = x_train - x_mean

    def log_posterior(w0, w1):
        rho = np.clip(w0 + w1 * xc_train, -30, 30)
        mu = np.exp(rho)
        log_likelihood = np.sum(stats.poisson.logpmf(y_train, mu))
        log_prior = stats.norm.logpdf(w0, 0, 100) + stats.norm.logpdf(w1, 0, 100)
        return log_likelihood + log_prior

    posterior = metropolis_hastings(log_posterior, init=[0.0, 0.0],
                                     step=[0.05, 0.004])
    w0_samples, w1_samples = posterior[:, 0], posterior[:, 1]
    print(f"[포아송 회귀] w0={w0_samples.mean():.4f}±{w0_samples.std():.4f}, "
          f"w1={w1_samples.mean():.4f}±{w1_samples.std():.4f} (사후평균±사후표준편차)")

    x_inf = df_inf["active_since_1800"].to_numpy(dtype=float)
    y_inf = df_inf["earthquakes"].to_numpy(dtype=float)
    rho_inf = w0_samples[:, None] + w1_samples[:, None] * (x_inf[None, :] - x_mean)
    mu_inf = np.exp(rho_inf)
    rng = np.random.default_rng(42)
    draws_inf = rng.poisson(mu_inf)  # shape (n_samples, n_inference)
    th_lower = np.quantile(draws_inf, TARGET_FP_RATE / 2, axis=0)
    th_upper = np.quantile(draws_inf, 1 - TARGET_FP_RATE / 2, axis=0)
    pred = np.where((y_inf < th_lower) | (y_inf > th_upper), "anomaly", "normal")
    print(f"[포아송 회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")
    df_anomaly = df_inf[pred == "anomaly"][["country", "year", "earthquakes", "active_since_1800"]]
    print(f"[포아송 회귀] 이상 판정 상세:\n{df_anomaly.to_string(index=False)}")

    x_line = np.linspace(0, 75, 60)
    rho_line = w0_samples[:, None] + w1_samples[:, None] * (x_line[None, :] - x_mean)
    mu_line = np.exp(rho_line)
    draws_line = rng.poisson(mu_line)
    lower_line = np.quantile(draws_line, TARGET_FP_RATE / 2, axis=0)
    upper_line = np.quantile(draws_line, 1 - TARGET_FP_RATE / 2, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill_between(x_line, lower_line, upper_line, color="#dddddd", label="normal region")
    ax.plot(x_line, lower_line, color="#111111", lw=1)
    ax.plot(x_line, upper_line, color="#111111", lw=1)
    sns.scatterplot(x=x_inf, y=y_inf, hue=pred, hue_order=["anomaly", "normal"],
                     palette=["#111111", "#999999"], ax=ax)
    ax.set_xlabel("active_since_1800 (volcano count)")
    ax.set_ylabel("earthquakes")
    path = os.path.join(OUTPUT_DIR, "bayesian-poisson-boundary.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


def dealer_random_effect_illustration():
    """다변량 회귀의 잔차를 판매자별로 나눠, 설명변수로 못 잡아내는 개체차를 확인한다"""
    df_train, _, cat_names, _ = load_usedcar_data()
    feature_cols = ["year", "odometer"] + cat_names
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["price"].to_numpy()

    model = BayesianRidge()
    model.fit(X_train, y_train)
    residual = y_train - model.predict(X_train)
    df_resid = pd.DataFrame({"dealer": df_train["dealer"].to_numpy(), "residual": residual})
    dealer_order = (df_resid.groupby("dealer")["residual"].mean()
                     .sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(data=df_resid, x="dealer", y="residual", order=dealer_order,
                color="#999999", ax=ax, fliersize=2)
    ax.axhline(0, color="#111111", lw=1, ls="--")
    ax.set_xlabel("dealer")
    ax.set_ylabel("residual (actual - predicted price, 10k KRW)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "dealer-residual-boxplot.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


if __name__ == "__main__":
    bayesian_linear_regression()
    bayesian_logistic_regression()
    bayesian_poisson_regression()
    dealer_random_effect_illustration()

"""
3편 - 회귀 모델을 이용한 이상탐지: 선형회귀, 다변량 선형회귀, 리지회귀, 감마회귀

중고차 가격 데이터셋(연식·주행거리·차종·판매자로 생성한 합성 데이터)에서, 가격을
그대로 정상 범위로 보는 대신 연식·주행거리·차종에 따라 달라지는 "적정 가격대"를
회귀로 예측하고 그 예측값에서 크게 벗어난 매물을 이상으로 판정한다.

실행:
    python series03/regression.py
"""

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from common import TARGET_FP_RATE, load_usedcar_data

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS_TO_SHOW = [2, 8]  # 다변량 그림에서 대표로 보여줄 연식(경과년수)


def linear_regression_anomaly_detection():
    """연식 하나만으로 가격을 예측하는 1변수 선형회귀 이상탐지"""
    df_train, df_inference, _, _ = load_usedcar_data()
    x_train, y_train = df_train["year"].to_numpy(), df_train["price"].to_numpy()

    X_train_intercept = sm.add_constant(x_train)
    res = sm.OLS(y_train, X_train_intercept).fit()
    slope, intercept = res.params[1], res.params[0]
    residual_var = res.scale * res.df_resid / len(y_train)

    train_score = (y_train - slope * x_train - intercept) ** 2 / residual_var
    anomaly_threshold = np.quantile(train_score, 1 - TARGET_FP_RATE)
    print(f"[선형회귀] slope={slope:.3f}, intercept={intercept:.3f}, residual_var={residual_var:.2f}, anomaly_threshold={anomaly_threshold:.4f}")
    # 임계값이 카이제곱 기준(약 9)보다 낮은 이유: 학습 잔차의 꼬리가 정규분포보다 얇다
    standardized = res.resid / np.sqrt(residual_var)
    print(f"[선형회귀] 학습 잔차 왜도={stats.skew(res.resid):.3f}, 초과첨도={stats.kurtosis(res.resid):.3f}, "
          f"최대 |표준화 잔차|={np.abs(standardized).max():.2f}")

    x_inf, y_inf = df_inference["year"].to_numpy(), df_inference["price"].to_numpy()
    score_inf = (y_inf - slope * x_inf - intercept) ** 2 / residual_var
    pred = np.where(score_inf > anomaly_threshold, "anomaly", "normal")
    print(f"[선형회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")

    x_grid = np.arange(x_inf.min() - 1, x_inf.max() + 2)
    y_grid = np.linspace(y_inf.min() - 50, y_inf.max() + 50, 500)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    score_grid = (mesh_y - slope * mesh_x - intercept) ** 2 / residual_var
    pred_grid = np.where(score_grid > anomaly_threshold, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(mesh_x, mesh_y, pred_grid, levels=1, cmap=cm.gray, alpha=0.5)
    sns.scatterplot(x=x_inf, y=y_inf, hue=df_inference["label"],
                     hue_order=["normal", "anomaly"], palette=["#999999", "#111111"], ax=ax)
    ax.set_xlabel("year")
    ax.set_ylabel("price (10k KRW)")
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "linear-regression-boundary.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


def _plot_multivariate_grid(model_cols, model_cat_names, dropped_name,
                             predict_fn, df_inference, filename):
    """차종(열) x 연식(행) 그리드에 결정경계와 추론 데이터를 함께 그린다"""
    fig, axes = plt.subplots(nrows=len(YEARS_TO_SHOW), ncols=3, figsize=(6, 4))
    x_min, x_max = df_inference["odometer"].min(), df_inference["odometer"].max()
    y_min, y_max = df_inference["price"].min(), df_inference["price"].max()
    x_grid = np.linspace(x_min - 1, x_max + 2, 80)
    y_grid = np.linspace(y_min - 50, y_max + 50, 150)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)

    for col_idx, model_name in enumerate(model_cols):
        if model_name == dropped_name:
            onehot = np.zeros(len(model_cat_names))
        else:
            onehot = np.array([1.0 if c == f"model_name_{model_name}" else 0.0
                                for c in model_cat_names])
        df_model = df_inference[df_inference["model_name"] == model_name]
        for row_idx, year in enumerate(YEARS_TO_SHOW):
            ax = axes[row_idx][col_idx]
            n_grid = mesh_x.size
            X_grid_features = np.column_stack([
                np.full(n_grid, year), mesh_x.ravel(),
                np.tile(onehot, (n_grid, 1)) if len(onehot) else np.empty((n_grid, 0)),
            ])
            pred_grid = predict_fn(X_grid_features, mesh_y.ravel()).reshape(mesh_x.shape)
            ax.contourf(mesh_x, mesh_y, pred_grid, levels=1, cmap=cm.gray, alpha=0.5)

            df_year = df_model[df_model["year"] == year]
            sns.scatterplot(data=df_year, x="odometer", y="price", hue="label",
                             hue_order=["normal", "anomaly"], palette=["#999999", "#111111"],
                             ax=ax, legend=False, s=14)
            ax.set_title(f"{model_name}, year={year}", fontsize=8)
            ax.set_xlabel("odometer (10k km)", fontsize=7)
            ax.set_ylabel("price (10k KRW)", fontsize=7)
            ax.tick_params(labelsize=6)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


def multivariate_regression_anomaly_detection():
    """연식·주행거리·차종을 함께 쓰는 다변량 선형회귀 이상탐지"""
    df_train, df_inference, cat_names, encoder = load_usedcar_data()
    feature_cols = ["year", "odometer"] + cat_names
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["price"].to_numpy()

    X_train_intercept = sm.add_constant(X_train)
    res = sm.OLS(y_train, X_train_intercept).fit()
    coef, intercept = res.params[1:], res.params[0]
    residual_var = res.scale * res.df_resid / len(y_train)

    train_score = (y_train - X_train @ coef - intercept) ** 2 / residual_var
    anomaly_threshold = np.quantile(train_score, 1 - TARGET_FP_RATE)
    print(f"[다변량 선형회귀] coef={np.round(coef, 3)}, intercept={intercept:.3f}, "
          f"residual_var={residual_var:.2f}, anomaly_threshold={anomaly_threshold:.4f}")

    X_inf = df_inference[feature_cols].to_numpy()
    y_inf = df_inference["price"].to_numpy()
    score_inf = (y_inf - X_inf @ coef - intercept) ** 2 / residual_var
    pred = np.where(score_inf > anomaly_threshold, "anomaly", "normal")
    print(f"[다변량 선형회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")

    def predict_fn(X_features, y_values):
        score = (y_values - X_features @ coef - intercept) ** 2 / residual_var
        return np.where(score > anomaly_threshold, 0, 1)

    model_cols = list(encoder.categories_[0])
    dropped_name = encoder.categories_[0][0]
    _plot_multivariate_grid(model_cols, cat_names, dropped_name, predict_fn,
                             df_inference, "multivariate-regression-boundary.png")


def ridge_regression_anomaly_detection():
    """다중공선성에 대비한 L2 정규화 회귀(리지 회귀) 이상탐지"""
    df_train, df_inference, cat_names, encoder = load_usedcar_data()
    feature_cols = ["year", "odometer"] + cat_names
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["price"].to_numpy()

    X_train_intercept = sm.add_constant(X_train)
    mod = sm.OLS(y_train, X_train_intercept)
    LAMBDA = 2.0
    # statsmodels의 목적함수는 RSS/(2N) + alpha*||w||^2/2 이므로, 블로그 글의 식
    # RSS + λ||w||^2 과 맞추려면 alpha = λ/N 이다. 절편은 정규화 대상에서 제외한다.
    n_features = X_train_intercept.shape[1]
    alpha = np.full(n_features, LAMBDA / len(y_train))
    alpha[0] = 0.0
    res = mod.fit_regularized(alpha=alpha, L1_wt=0)
    ridge_coef = res.params

    design_matrix = np.concatenate([np.ones((len(y_train), 1)), X_train], axis=1)
    rss = np.sum((y_train - design_matrix @ ridge_coef) ** 2)
    residual_var = (rss + LAMBDA * np.sum(ridge_coef[1:] ** 2)) / len(y_train)

    penalty_matrix = LAMBDA * np.eye(n_features)
    penalty_matrix[0, 0] = 0.0  # 절편은 정규화하지 않는다
    gram_matrix = design_matrix.T @ design_matrix + penalty_matrix
    gram_matrix_inv = np.linalg.inv(gram_matrix)
    leverage_train = 1 + np.sum(design_matrix @ gram_matrix_inv * design_matrix, axis=1)

    log_term_train = np.log(2 * np.pi * residual_var * leverage_train) / 2
    sq_term_train = (y_train - design_matrix @ ridge_coef) ** 2 / (2 * residual_var * leverage_train)
    train_score = log_term_train + sq_term_train
    anomaly_threshold = np.quantile(train_score, 1 - TARGET_FP_RATE)
    print(f"[리지회귀] ridge_coef={np.round(ridge_coef, 3)}, residual_var={residual_var:.2f}, anomaly_threshold={anomaly_threshold:.4f}")

    X_inf = df_inference[feature_cols].to_numpy()
    y_inf = df_inference["price"].to_numpy()
    design_inf = np.concatenate([np.ones((len(y_inf), 1)), X_inf], axis=1)
    leverage_inf = 1 + np.sum(design_inf @ gram_matrix_inv * design_inf, axis=1)
    log_term_inf = np.log(2 * np.pi * residual_var * leverage_inf) / 2
    sq_term_inf = (y_inf - design_inf @ ridge_coef) ** 2 / (2 * residual_var * leverage_inf)
    score_inf = log_term_inf + sq_term_inf
    pred = np.where(score_inf > anomaly_threshold, "anomaly", "normal")
    print(f"[리지회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")

    def predict_fn(X_features, y_values):
        design = np.concatenate([np.ones((len(X_features), 1)), X_features], axis=1)
        leverage = 1 + np.sum(design @ gram_matrix_inv * design, axis=1)
        log_term = np.log(2 * np.pi * residual_var * leverage) / 2
        sq_term = (y_values - design @ ridge_coef) ** 2 / (2 * residual_var * leverage)
        score = log_term + sq_term
        return np.where(score > anomaly_threshold, 0, 1)

    model_cols = list(encoder.categories_[0])
    dropped_name = encoder.categories_[0][0]
    _plot_multivariate_grid(model_cols, cat_names, dropped_name, predict_fn,
                             df_inference, "ridge-regression-boundary.png")


def gamma_regression_anomaly_detection():
    """가격처럼 항상 양수이고 오른쪽 꼬리가 긴 데이터를 감마분포로 모델링하는 비정규 GLM"""
    df_train, df_inference, _, _ = load_usedcar_data()
    x_train, y_train = df_train["year"].to_numpy(), df_train["price"].to_numpy()

    X_train_intercept = sm.add_constant(x_train)
    family = sm.families.Gamma(sm.families.links.Log())
    res = sm.GLM(y_train, X_train_intercept, family=family).fit()

    train_dist = res.get_distribution(exog=X_train_intercept)
    train_score = -np.log(train_dist.pdf(y_train))
    anomaly_threshold = np.quantile(train_score, 1 - TARGET_FP_RATE)
    print(f"[감마회귀] anomaly_threshold={anomaly_threshold:.4f}")

    # 연식별 정상 범위의 폭. 감마분포는 분산이 평균의 제곱에 비례하므로 평균이 낮아질수록 폭이 줄고,
    # 평균 위쪽 폭이 아래쪽 폭보다 조금 넓다(오른쪽으로 치우친 분포).
    price_grid = np.linspace(500, 6500, 20001)
    for year in [2, 10]:
        exog = sm.add_constant(np.full_like(price_grid, year, dtype=float), has_constant="add")
        inside = price_grid[-np.log(res.get_distribution(exog=exog).pdf(price_grid)) <= anomaly_threshold]
        mean_price = float(np.exp(res.params[0] + res.params[1] * year))
        print(f"[감마회귀] year={year}: 평균={mean_price:.0f}, 정상 범위=[{inside.min():.0f}, {inside.max():.0f}], "
              f"아래 폭={mean_price - inside.min():.0f}, 위 폭={inside.max() - mean_price:.0f}")

    x_inf, y_inf = df_inference["year"].to_numpy(), df_inference["price"].to_numpy()
    X_inf_intercept = sm.add_constant(x_inf)
    inf_dist = res.get_distribution(exog=X_inf_intercept)
    score_inf = -np.log(inf_dist.pdf(y_inf))
    pred = np.where(score_inf > anomaly_threshold, "anomaly", "normal")
    print(f"[감마회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")

    x_grid = np.arange(x_inf.min() - 1, x_inf.max() + 2)
    y_grid = np.linspace(y_inf.min() - 50, y_inf.max() + 50, 500)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    X_grid_intercept = sm.add_constant(mesh_x.ravel())
    grid_dist = res.get_distribution(exog=X_grid_intercept)
    score_grid = (-np.log(grid_dist.pdf(mesh_y.ravel()))).reshape(mesh_x.shape)
    pred_grid = np.where(score_grid > anomaly_threshold, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(mesh_x, mesh_y, pred_grid, levels=1, cmap=cm.gray, alpha=0.5)
    sns.scatterplot(x=x_inf, y=y_inf, hue=df_inference["label"],
                     hue_order=["normal", "anomaly"], palette=["#999999", "#111111"], ax=ax)
    ax.set_xlabel("year")
    ax.set_ylabel("price (10k KRW)")
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "gamma-regression-boundary.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"저장됨: {path}")


if __name__ == "__main__":
    linear_regression_anomaly_detection()
    multivariate_regression_anomaly_detection()
    ridge_regression_anomaly_detection()
    gamma_regression_anomaly_detection()

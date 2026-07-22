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
    w_1, w_0 = res.params[1], res.params[0]
    sigma2 = res.scale * res.df_resid / len(y_train)

    train_score = (y_train - w_1 * x_train - w_0) ** 2 / sigma2
    a_th = np.quantile(train_score, 1 - TARGET_FP_RATE)
    print(f"[선형회귀] w_1={w_1:.3f}, w_0={w_0:.3f}, sigma2={sigma2:.2f}, a_th={a_th:.4f}")

    x_inf, y_inf = df_inference["year"].to_numpy(), df_inference["price"].to_numpy()
    score_inf = (y_inf - w_1 * x_inf - w_0) ** 2 / sigma2
    pred = np.where(score_inf > a_th, "anomaly", "normal")
    print(f"[선형회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")

    x_grid = np.arange(x_inf.min() - 1, x_inf.max() + 2)
    y_grid = np.linspace(y_inf.min() - 50, y_inf.max() + 50, 500)
    X, Y = np.meshgrid(x_grid, y_grid)
    score_grid = (Y - w_1 * X - w_0) ** 2 / sigma2
    pred_grid = np.where(score_grid > a_th, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(X, Y, pred_grid, levels=1, cmap=cm.gray, alpha=0.5)
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
    X, Y = np.meshgrid(x_grid, y_grid)

    for col_idx, model_name in enumerate(model_cols):
        if model_name == dropped_name:
            onehot = np.zeros(len(model_cat_names))
        else:
            onehot = np.array([1.0 if c == f"model_name_{model_name}" else 0.0
                                for c in model_cat_names])
        df_model = df_inference[df_inference["model_name"] == model_name]
        for row_idx, year in enumerate(YEARS_TO_SHOW):
            ax = axes[row_idx][col_idx]
            n_grid = X.size
            X_grid_features = np.column_stack([
                np.full(n_grid, year), X.ravel(),
                np.tile(onehot, (n_grid, 1)) if len(onehot) else np.empty((n_grid, 0)),
            ])
            pred_grid = predict_fn(X_grid_features, Y.ravel()).reshape(X.shape)
            ax.contourf(X, Y, pred_grid, levels=1, cmap=cm.gray, alpha=0.5)

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
    df_train, df_inference, cat_names, enc = load_usedcar_data()
    feature_cols = ["year", "odometer"] + cat_names
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["price"].to_numpy()

    X_train_intercept = sm.add_constant(X_train)
    res = sm.OLS(y_train, X_train_intercept).fit()
    w, w_0 = res.params[1:], res.params[0]
    sigma2 = res.scale * res.df_resid / len(y_train)

    train_score = (y_train - X_train @ w - w_0) ** 2 / sigma2
    a_th = np.quantile(train_score, 1 - TARGET_FP_RATE)
    print(f"[다변량 선형회귀] w={np.round(w, 3)}, w_0={w_0:.3f}, "
          f"sigma2={sigma2:.2f}, a_th={a_th:.4f}")

    X_inf = df_inference[feature_cols].to_numpy()
    y_inf = df_inference["price"].to_numpy()
    score_inf = (y_inf - X_inf @ w - w_0) ** 2 / sigma2
    pred = np.where(score_inf > a_th, "anomaly", "normal")
    print(f"[다변량 선형회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")

    def predict_fn(X_features, y_values):
        score = (y_values - X_features @ w - w_0) ** 2 / sigma2
        return np.where(score > a_th, 0, 1)

    model_cols = list(enc.categories_[0])
    dropped_name = enc.categories_[0][0]
    _plot_multivariate_grid(model_cols, cat_names, dropped_name, predict_fn,
                             df_inference, "multivariate-regression-boundary.png")


def ridge_regression_anomaly_detection():
    """다중공선성에 대비한 L2 정규화 회귀(리지 회귀) 이상탐지"""
    df_train, df_inference, cat_names, enc = load_usedcar_data()
    feature_cols = ["year", "odometer"] + cat_names
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["price"].to_numpy()

    X_train_intercept = sm.add_constant(X_train)
    mod = sm.OLS(y_train, X_train_intercept)
    LAMBDA = 2.0
    alpha = LAMBDA / 2 / len(y_train)
    res = mod.fit_regularized(alpha=alpha, L1_wt=0)
    u = res.params

    design_matrix = np.concatenate([np.ones((len(y_train), 1)), X_train], axis=1)
    rss = np.sum((y_train - design_matrix @ u) ** 2)
    sigma2 = (rss + LAMBDA * np.sum(u * u)) / len(y_train)

    n_features = X_train_intercept.shape[1]
    A = design_matrix.T @ design_matrix + LAMBDA * np.eye(n_features)
    A_inv = np.linalg.inv(A)
    sx_train = 1 + np.sum(design_matrix @ A_inv * design_matrix, axis=1)

    first_anom = np.log(2 * np.pi * sigma2 * sx_train) / 2
    second_anom = (y_train - design_matrix @ u) ** 2 / (2 * sigma2 * sx_train)
    train_score = first_anom + second_anom
    a_th = np.quantile(train_score, 1 - TARGET_FP_RATE)
    print(f"[리지회귀] u={np.round(u, 3)}, sigma2={sigma2:.2f}, a_th={a_th:.4f}")

    X_inf = df_inference[feature_cols].to_numpy()
    y_inf = df_inference["price"].to_numpy()
    design_inf = np.concatenate([np.ones((len(y_inf), 1)), X_inf], axis=1)
    sx_inf = 1 + np.sum(design_inf @ A_inv * design_inf, axis=1)
    first_inf = np.log(2 * np.pi * sigma2 * sx_inf) / 2
    second_inf = (y_inf - design_inf @ u) ** 2 / (2 * sigma2 * sx_inf)
    score_inf = first_inf + second_inf
    pred = np.where(score_inf > a_th, "anomaly", "normal")
    print(f"[리지회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")

    def predict_fn(X_features, y_values):
        design = np.concatenate([np.ones((len(X_features), 1)), X_features], axis=1)
        sx = 1 + np.sum(design @ A_inv * design, axis=1)
        first = np.log(2 * np.pi * sigma2 * sx) / 2
        second = (y_values - design @ u) ** 2 / (2 * sigma2 * sx)
        score = first + second
        return np.where(score > a_th, 0, 1)

    model_cols = list(enc.categories_[0])
    dropped_name = enc.categories_[0][0]
    _plot_multivariate_grid(model_cols, cat_names, dropped_name, predict_fn,
                             df_inference, "ridge-regression-boundary.png")


def gamma_regression_anomaly_detection():
    """가격처럼 항상 양수이고 오른쪽 꼬리가 긴 데이터를 감마분포로 모델링하는 비정규 GLM"""
    df_train, df_inference, _, _ = load_usedcar_data()
    x_train, y_train = df_train["year"].to_numpy(), df_train["price"].to_numpy()

    X_train_intercept = sm.add_constant(x_train)
    family = sm.families.Gamma(sm.families.links.Log())
    res = sm.GLM(y_train, X_train_intercept, family=family).fit()

    gen_train = res.get_distribution(exog=X_train_intercept)
    train_score = -np.log(gen_train.pdf(y_train))
    a_th = np.quantile(train_score, 1 - TARGET_FP_RATE)
    print(f"[감마회귀] a_th={a_th:.4f}")

    x_inf, y_inf = df_inference["year"].to_numpy(), df_inference["price"].to_numpy()
    X_inf_intercept = sm.add_constant(x_inf)
    gen_inf = res.get_distribution(exog=X_inf_intercept)
    score_inf = -np.log(gen_inf.pdf(y_inf))
    pred = np.where(score_inf > a_th, "anomaly", "normal")
    print(f"[감마회귀] 추론 데이터 이상 판정 수={sum(pred == 'anomaly')} / {len(pred)}")

    x_grid = np.arange(x_inf.min() - 1, x_inf.max() + 2)
    y_grid = np.linspace(y_inf.min() - 50, y_inf.max() + 50, 500)
    X, Y = np.meshgrid(x_grid, y_grid)
    X_grid_intercept = sm.add_constant(X.ravel())
    gen_grid = res.get_distribution(exog=X_grid_intercept)
    score_grid = (-np.log(gen_grid.pdf(Y.ravel()))).reshape(X.shape)
    pred_grid = np.where(score_grid > a_th, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(X, Y, pred_grid, levels=1, cmap=cm.gray, alpha=0.5)
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

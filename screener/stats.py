"""Statistical tests used to justify the strategy and each individual trade."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Performance summaries
# --------------------------------------------------------------------------- #
def annualized_return(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = r.dropna()
    return float((1.0 + r).prod() ** (periods / len(r)) - 1.0)


def annualized_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(r.dropna().std() * np.sqrt(periods))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    ex = (r - rf).dropna()
    return float(ex.mean() / ex.std() * np.sqrt(periods))


def max_drawdown(r: pd.Series) -> float:
    wealth = (1.0 + r.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def var_cvar(r: pd.Series, level: float = 0.95) -> dict[str, float]:
    """One-period Value-at-Risk and Conditional VaR (expected shortfall), as losses.

    Reports both the historical estimate and the Cornish-Fisher estimate, which
    adjusts the normal quantile for the skew and fat tails of the sample.
    """
    r = r.dropna()
    q = np.quantile(r, 1.0 - level)
    hist_var, hist_cvar = -q, -r[r <= q].mean()
    z = st.norm.ppf(1.0 - level)
    s, k = st.skew(r), st.kurtosis(r)  # excess kurtosis
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    cf_var = -(r.mean() + z_cf * r.std())
    return {"var_hist": float(hist_var), "cvar_hist": float(hist_cvar), "var_cf": float(cf_var)}


# --------------------------------------------------------------------------- #
# Hypothesis tests
# --------------------------------------------------------------------------- #
@dataclass
class RegressionResult:
    alpha_ann: float  # annualized intercept
    alpha_t: float
    alpha_p: float
    beta: float
    beta_t: float
    r2: float
    n: int


def capm_regression(
    r: pd.Series, mkt: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS, hac_lags: int | None = None
) -> RegressionResult:
    """OLS of excess returns on market excess returns with Newey-West (HAC) errors.

    HAC standard errors stay valid when returns are autocorrelated or
    heteroskedastic, which daily and monthly returns usually are.
    """
    df = pd.concat([r - rf, mkt - rf], axis=1, keys=["y", "x"]).dropna()
    lags = hac_lags if hac_lags is not None else int(4 * (len(df) / 100) ** (2 / 9))
    fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return RegressionResult(
        alpha_ann=float(fit.params["const"] * periods),
        alpha_t=float(fit.tvalues["const"]),
        alpha_p=float(fit.pvalues["const"]),
        beta=float(fit.params["x"]),
        beta_t=float(fit.tvalues["x"]),
        r2=float(fit.rsquared),
        n=int(fit.nobs),
    )


def mean_t_test(x: pd.Series, hac_lags: int = 0) -> tuple[float, float, float]:
    """Mean, t-stat and two-sided p-value of H0: mean = 0 (optionally Newey-West)."""
    x = x.dropna()
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return float(fit.params[0]), float(fit.tvalues[0]), float(fit.pvalues[0])


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0, rf: pd.Series | float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark) given sample length, skew and kurtosis.

    Bailey & Lopez de Prado (2012). Both Sharpe ratios are per-period (not annualized).
    """
    ex = (r - rf).dropna()
    n = len(ex)
    sr = ex.mean() / ex.std()
    g3, g4 = st.skew(ex), st.kurtosis(ex, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    return float(st.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, n_trials: int, sr_trials_std: float, rf: pd.Series | float = 0.0) -> float:
    """PSR against the Sharpe you'd expect from the best of `n_trials` lucky tries.

    Bailey & Lopez de Prado (2014). Guards against picking the best of many backtests.
    """
    gamma = 0.5772156649
    if n_trials <= 1:
        return probabilistic_sharpe(r, 0.0, rf)
    sr0 = sr_trials_std * (
        (1 - gamma) * st.norm.ppf(1 - 1 / n_trials) + gamma * st.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return probabilistic_sharpe(r, sr0, rf)


def stationary_bootstrap_pvalue(
    x: pd.Series, n_boot: int = 5000, mean_block: float = 6.0, seed: int = 7
) -> tuple[float, tuple[float, float]]:
    """One-sided p-value for H0: mean(x) <= 0 and a 90% CI for the mean.

    Politis & Romano (1994) stationary bootstrap: resamples random-length blocks
    so serial correlation in returns is preserved. No normality assumption.
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    p_new = 1.0 / mean_block
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        jumps = rng.random(n) < p_new
        starts = rng.integers(n, size=n)
        for i in range(1, n):
            idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
        means[b] = x[idx].mean()
    centered = means - means.mean()  # impose the null
    p = float((centered >= x.mean()).mean())
    ci = (float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95)))
    return p, ci


def bootstrap_annual_return_ci(r: pd.Series, n_boot: int = 2000, seed: int = 11) -> tuple[float, float]:
    """90% CI of the annualized mean return by i.i.d. bootstrap of daily returns."""
    x = np.asarray(r.dropna(), dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(n_boot, len(x)))].mean(axis=1) * TRADING_DAYS
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def information_coefficients(scores: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """Spearman rank correlation between each date's scores and the next period's returns."""
    ics = {}
    for d in scores.index:
        s, f = scores.loc[d], fwd_returns.loc[d]
        ok = s.notna() & f.notna()
        if ok.sum() >= 10:
            ics[d] = st.spearmanr(s[ok], f[ok]).statistic
    return pd.Series(ics, name="IC")

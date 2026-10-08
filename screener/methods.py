"""Portfolio-construction methods compared head to head.

Every method has the same signature: given the feature panel and a rebalance
date, return long-only weights (and, for ranking methods, the score used, so
its information coefficient can be measured). Each method may use only data
available at that date: Bayesian methods train on (features, realized next-month
return) pairs from strictly earlier months.

Signal methods share one construction rule so only the signal differs: take
the top N names by score (at most 3 per sector), weight by inverse volatility,
cap at 10%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from .factors import FACTOR_WEIGHTS
from .features import SIGNALS, Panel
from .universe import SECTOR_OF

TOP_N = 15
MAX_PER_SECTOR = 3
MAX_WEIGHT = 0.10
MIN_TRAIN_MONTHS = 24


@dataclass
class Context:
    panel: Panel
    prices: pd.DataFrame
    benchmark: str
    cash: str


MethodFn = Callable[[Context, pd.Timestamp], tuple[pd.Series, pd.Series | None]]


# --------------------------------------------------------------------------- #
# Shared construction helpers
# --------------------------------------------------------------------------- #
def cap_weights(w: pd.Series, cap: float = MAX_WEIGHT) -> pd.Series:
    w = w / w.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        w[under] += excess * w[under] / w[under].sum()
    return w


def top_n_inverse_vol(score: pd.Series, feats: pd.DataFrame, n: int = TOP_N) -> pd.Series:
    chosen, per_sector = [], {}
    for t in score.dropna().sort_values(ascending=False).index:
        sec = SECTOR_OF.get(t, "Other")
        if per_sector.get(sec, 0) >= MAX_PER_SECTOR:
            continue
        chosen.append(t)
        per_sector[sec] = per_sector.get(sec, 0) + 1
        if len(chosen) == n:
            break
    return cap_weights(1.0 / feats.loc[chosen, "vol"])


def signal_method(score_fn: Callable[[pd.DataFrame], pd.Series]) -> MethodFn:
    def run(ctx: Context, d: pd.Timestamp):
        f = ctx.panel.feats[d]
        s = score_fn(f)
        return top_n_inverse_vol(s, f), s

    return run


def training_set(ctx: Context, d: pd.Timestamp, cols: list[str]) -> tuple[np.ndarray, np.ndarray, list]:
    """Stacked (X, y) from months whose next-month return was known by date d."""
    xs, ys, months = [], [], []
    for m in ctx.panel.dates:
        if m >= d:
            break
        nxt_idx = ctx.panel.dates.index(m) + 1
        if ctx.panel.dates[nxt_idx] > d:  # outcome not yet realized at d
            continue
        f, y = ctx.panel.feats[m], ctx.panel.fwd[m]
        y = (y - y.mean()).reindex(f.index)  # cross-sectionally demeaned: pure selection
        ok = y.notna()
        xs.append(f.loc[ok, cols].values)
        ys.append(y[ok].values)
        months.append(m)
    return xs, ys, months


# --------------------------------------------------------------------------- #
# Bayesian methods
# --------------------------------------------------------------------------- #
ZCOLS = [f"z_{s}" for s in SIGNALS]


def bayes_regression(ctx: Context, d: pd.Timestamp, prior_sd: float = 0.004):
    """Conjugate Bayesian linear regression of next-month returns on all signals.

    Prior on each factor weight: N(0, prior_sd^2), a skeptical prior saying a
    1-sigma tilt in any signal is worth 0 +/- 0.4% a month. The data must
    overcome that skepticism. The posterior mean weights produce the score;
    `ctx.bayes_weights` keeps the posterior means and sds for reporting.
    """
    xs, ys, months = training_set(ctx, d, ZCOLS)
    f = ctx.panel.feats[d]
    if len(months) < MIN_TRAIN_MONTHS:
        return top_n_inverse_vol(-f["vol"], f), None
    X, y = np.vstack(xs), np.concatenate(ys)
    sigma2 = y.var()
    k = X.shape[1]
    prec = np.eye(k) / prior_sd**2 + X.T @ X / sigma2
    cov = np.linalg.inv(prec)
    mean = cov @ X.T @ y / sigma2
    record = getattr(ctx, "bayes_weights", {})
    record[d] = pd.DataFrame({"mean": mean, "sd": np.sqrt(np.diag(cov))}, index=SIGNALS)
    ctx.bayes_weights = record
    s = pd.Series(f[ZCOLS].values @ mean, index=f.index)
    return top_n_inverse_vol(s, f), s


def bayes_ic_ensemble(ctx: Context, d: pd.Timestamp, prior_sd: float = 0.02):
    """Combine signals with weights = posterior mean IC (normal-normal model).

    Each signal's monthly ICs are treated as draws around a true IC with a
    N(0, prior_sd^2) prior. Signals whose posterior mean IC is <= 0 get zero
    weight. If no signal has earned positive weight, hold the lowest-vol names.
    """
    ics = getattr(ctx, "_ic_hist", None)
    if ics is None:
        from scipy.stats import spearmanr

        rows = {}
        for m in ctx.panel.dates[:-1]:
            f, y = ctx.panel.feats[m], ctx.panel.fwd[m]
            rows[m] = {s: spearmanr(f[s], y.reindex(f.index), nan_policy="omit").statistic for s in SIGNALS}
        ics = pd.DataFrame(rows).T
        ctx._ic_hist = ics
    dates = ctx.panel.dates
    known = [m for m in ics.index if dates[dates.index(m) + 1] <= d]
    f = ctx.panel.feats[d]
    if len(known) < MIN_TRAIN_MONTHS:
        return top_n_inverse_vol(-f["vol"], f), None
    h = ics.loc[known]
    n, ybar, v = len(h), h.mean(), h.var()
    post_mean = (n * ybar / v) / (n / v + 1 / prior_sd**2)
    w = post_mean.clip(lower=0)
    rec = getattr(ctx, "ensemble_weights", {})
    rec[d] = post_mean
    ctx.ensemble_weights = rec
    if w.sum() <= 0:
        return top_n_inverse_vol(-f["vol"], f), None
    s = sum(w[sig] * f[f"z_{sig}"] for sig in SIGNALS) / w.sum()
    return top_n_inverse_vol(s, f), s


def black_litterman_posterior(
    f: pd.DataFrame, prices: pd.DataFrame, d: pd.Timestamp, delta: float = 2.5, tau: float = 0.05, ic: float = 0.03
) -> pd.DataFrame:
    """Black-Litterman prior (pi), views (q), posterior (mu) and raw mean-variance weights.

    Prior: implied returns pi = delta * Sigma * w_eq (equal-weight market proxy).
    Views: one absolute view per stock, q = pi + IC * vol * composite z (Grinold's
    "alpha = IC x volatility x score"), each as uncertain as tau * the stock's
    variance. Posterior: the precision-weighted blend of prior and views.
    """
    names = list(f.index)
    rets = prices[names].loc[:d].iloc[-253:].pct_change().iloc[1:].fillna(0)
    sigma = LedoitWolf().fit(rets.values).covariance_ * 252
    w_eq = np.full(len(names), 1 / len(names))
    pi = delta * sigma @ w_eq
    comp = sum(w * f[f"z_{k}"] for k, w in FACTOR_WEIGHTS.items()).values
    vol = np.sqrt(np.diag(sigma))
    q = pi + ic * vol * comp
    omega = np.diag(np.diag(tau * sigma))  # P = I
    ts_inv = np.linalg.inv(tau * sigma)
    om_inv = np.linalg.inv(omega)
    mu = np.linalg.solve(ts_inv + om_inv, ts_inv @ pi + om_inv @ q)
    raw = np.linalg.solve(delta * sigma, mu)
    return pd.DataFrame({"pi": pi, "view": q, "mu": mu, "mv_weight": raw}, index=names)


def black_litterman(ctx: Context, d: pd.Timestamp, delta: float = 2.5, tau: float = 0.05, ic: float = 0.03):
    """Black-Litterman with factor views, then a long-only mean-variance step (top N positive weights)."""
    f = ctx.panel.feats[d]
    bl = black_litterman_posterior(f, ctx.prices, d, delta, tau, ic)
    w = bl["mv_weight"]
    w = w[w > 0].sort_values(ascending=False).head(TOP_N)
    return cap_weights(w), bl["mu"] - bl["pi"]


# --------------------------------------------------------------------------- #
# Frequentist / risk-based methods
# --------------------------------------------------------------------------- #
def fama_macbeth(ctx: Context, d: pd.Timestamp, window: int = 60):
    """Rolling Fama-MacBeth: average monthly cross-sectional OLS slopes over 60 months."""
    xs, ys, months = training_set(ctx, d, ZCOLS)
    f = ctx.panel.feats[d]
    if len(months) < MIN_TRAIN_MONTHS:
        return top_n_inverse_vol(-f["vol"], f), None
    slopes = []
    for X, y in list(zip(xs, ys))[-window:]:
        Xc = np.column_stack([np.ones(len(X)), X])
        b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        slopes.append(b[1:])
    lam = np.mean(slopes, axis=0)
    s = pd.Series(f[ZCOLS].values @ lam, index=f.index)
    return top_n_inverse_vol(s, f), s


def min_variance(ctx: Context, d: pd.Timestamp):
    """Long-only minimum variance with Ledoit-Wolf shrinkage covariance."""
    f = ctx.panel.feats[d]
    names = list(f.index)
    rets = ctx.prices[names].loc[:d].iloc[-253:].pct_change().iloc[1:].fillna(0)
    sigma = LedoitWolf().fit(rets.values).covariance_
    active = np.ones(len(names), dtype=bool)
    for _ in range(len(names)):
        idx = np.where(active)[0]
        inv = np.linalg.solve(sigma[np.ix_(idx, idx)], np.ones(len(idx)))
        w = inv / inv.sum()
        if (w >= 0).all():
            break
        active[idx[w < 0]] = False
    out = pd.Series(0.0, index=names)
    out.iloc[np.where(active)[0]] = w
    return cap_weights(out[out > 1e-6]), None


def hmm_filter(returns: np.ndarray, n_iter: int = 15) -> float:
    """Two-state Gaussian hidden Markov model fitted by EM (Baum-Welch).

    Returns the filtered probability that the market is in the high-volatility
    state on the last day.
    """
    x = returns
    mu = np.array([x.mean(), x.mean()])
    sd = np.array([x.std() * 0.7, x.std() * 1.6])
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi0 = np.array([0.8, 0.2])
    T = len(x)
    for _ in range(n_iter):
        B = norm.pdf(x[:, None], mu, sd) + 1e-300
        alpha = np.empty((T, 2)); c = np.empty(T)
        alpha[0] = pi0 * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum(); alpha[t] /= c[t]
        beta = np.empty((T, 2)); beta[-1] = 1
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(0) / xi.sum((0, 2))[:, None]
        pi0 = gamma[0]
        mu = (gamma * x[:, None]).sum(0) / gamma.sum(0)
        sd = np.sqrt((gamma * (x[:, None] - mu) ** 2).sum(0) / gamma.sum(0))
    hi = int(np.argmax(sd))
    return float(alpha[-1, hi])


def hmm_regime_composite(ctx: Context, d: pd.Timestamp, lookback: int = 756):
    """Composite portfolio, but move half to T-bills when an HMM flags a high-vol regime."""
    f = ctx.panel.feats[d]
    s = sum(w * f[f"z_{k}"] for k, w in FACTOR_WEIGHTS.items())
    w = top_n_inverse_vol(s, f)
    spx = ctx.prices[ctx.benchmark].loc[:d].iloc[-lookback:].pct_change().dropna().values
    p_hi = hmm_filter(spx)
    rec = getattr(ctx, "hmm_prob", {})
    rec[d] = p_hi
    ctx.hmm_prob = rec
    if p_hi > 0.5:
        w = w * 0.5
        w[ctx.cash] = 0.5
    return w, s


# --------------------------------------------------------------------------- #
# The menu
# --------------------------------------------------------------------------- #
def _composite(f: pd.DataFrame) -> pd.Series:
    return sum(w * f[f"z_{k}"] for k, w in FACTOR_WEIGHTS.items())


METHODS: dict[str, tuple[str, MethodFn]] = {
    "composite": ("Equal-weight 4-factor composite (momentum, low vol, trend)", signal_method(_composite)),
    "momentum_12_1": ("12-1 month momentum", signal_method(lambda f: f["mom_12_1"])),
    "low_vol": ("Low volatility", signal_method(lambda f: f["low_vol"])),
    "reversal_1m": ("1-month reversal", signal_method(lambda f: f["rev_1m"])),
    "bollinger_mr": ("Bollinger-band mean reversion (20-day z)", signal_method(lambda f: f["boll_rev"])),
    "sharpe_momentum": ("Risk-adjusted momentum (return / vol)", signal_method(lambda f: f["sharpe_mom"])),
    "kalman_trend": ("Kalman-filter trend t-statistic", signal_method(lambda f: f["kalman_t"])),
    "eb_shrunk_drift": ("Empirical-Bayes (James-Stein) shrunk drift", signal_method(lambda f: f["eb_drift"])),
    "bayes_regression": ("Bayesian regression: posterior factor weights", bayes_regression),
    "bayes_ic_ensemble": ("Bayesian IC-weighted signal ensemble", bayes_ic_ensemble),
    "black_litterman": ("Black-Litterman with factor views", black_litterman),
    "fama_macbeth": ("Rolling Fama-MacBeth factor premia", fama_macbeth),
    "min_variance_lw": ("Minimum variance (Ledoit-Wolf covariance)", min_variance),
    "hmm_regime": ("Composite + HMM regime switch to T-bills", hmm_regime_composite),
}

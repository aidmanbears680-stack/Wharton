"""Month-end feature panel: every signal any method needs, computed point-in-time.

Each feature at date d uses only prices up to and including d, so nothing a
method sees at d leaks information from after d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import raw_factors, zscore

TRADING_DAYS = 252
SIGNALS = [
    "mom_12_1", "mom_6_1", "low_vol", "trend", "rev_1m", "sharpe_mom", "kalman_t", "boll_rev", "eb_drift",
]


def kalman_trend_t(logp: np.ndarray) -> np.ndarray:
    """Local-linear-trend Kalman filter run on each column of log prices.

    State = [level, slope]. Returns slope / sd(slope) at the last date: a
    t-statistic for "this stock is currently drifting upward", with the
    filter's own estimate of how uncertain that drift is.
    """
    t_len, k = logp.shape
    r = np.nanvar(np.diff(logp, axis=0), axis=0)  # daily return variance per stock
    q_level, q_slope, obs = r, r * 1e-4, r * 0.05
    lvl, slope = logp[0].copy(), np.zeros(k)
    p11, p12, p22 = r * 10, np.zeros(k), r * 1e-2
    for t in range(1, t_len):
        # predict
        lvl, slope = lvl + slope, slope
        p11, p12, p22 = p11 + 2 * p12 + p22 + q_level, p12 + p22, p22 + q_slope
        # update
        y = logp[t] - lvl
        s = p11 + obs
        k1, k2 = p11 / s, p12 / s
        lvl, slope = lvl + k1 * y, slope + k2 * y
        p11, p12, p22 = (1 - k1) * p11, (1 - k1) * p12, p22 - k2 * p12
    return slope / np.sqrt(np.maximum(p22, 1e-18))


def empirical_bayes_drift(daily: pd.DataFrame) -> pd.DataFrame:
    """James-Stein / empirical-Bayes shrinkage of each stock's mean daily return.

    Prior: true drifts mu_i ~ N(m0, tau^2) across stocks, with m0 and tau^2
    estimated from the cross-section (method of moments). Each stock's sample
    mean is pulled toward m0 in proportion to how noisy it is.
    """
    n = daily.count()
    m = daily.mean()
    se2 = daily.var() / n
    m0 = m.mean()
    tau2 = max(m.var() - se2.mean(), 1e-12)
    shrink = se2 / (se2 + tau2)
    post_mean = shrink * m0 + (1 - shrink) * m
    post_sd = np.sqrt(1.0 / (1.0 / se2 + 1.0 / tau2))
    from scipy.stats import norm

    return pd.DataFrame(
        {"eb_drift": post_mean, "eb_prob_pos": norm.cdf(post_mean / post_sd), "eb_shrink": shrink}
    )


def features_at(prices: pd.DataFrame, asof: pd.Timestamp, stocks: list[str], benchmark: str) -> pd.DataFrame:
    hist = prices.loc[:asof]
    raw = raw_factors(hist[stocks])
    names = list(raw.index)
    win = hist[names].iloc[-253:]
    daily = win.pct_change().iloc[1:]
    mkt = hist[benchmark].iloc[-253:].pct_change().iloc[1:]

    p, p_1m = win.iloc[-1], win.iloc[-22]
    vol_252 = daily.std() * np.sqrt(TRADING_DAYS)
    sma20, sd20 = win.iloc[-20:].mean(), win.iloc[-20:].std()

    f = raw.copy()
    f["rev_1m"] = -(p / p_1m - 1.0)
    f["sharpe_mom"] = f["mom_12_1"] / vol_252
    f["kalman_t"] = kalman_trend_t(np.log(win.values))
    f["boll_rev"] = -(p - sma20) / sd20
    f = f.join(empirical_bayes_drift(daily))
    f["vol"] = daily.iloc[-63:].std() * np.sqrt(TRADING_DAYS)
    f["beta"] = daily.apply(lambda c: c.cov(mkt)) / mkt.var()
    for s in SIGNALS:
        f[f"z_{s}"] = zscore(f[s])
    return f


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month in `index`."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


@dataclass
class Panel:
    dates: list[pd.Timestamp]          # month-end rebalance dates
    feats: dict[pd.Timestamp, pd.DataFrame]
    fwd: dict[pd.Timestamp, pd.Series]  # return from d to the next month-end (NaN for last date)


def build_panel(prices: pd.DataFrame, stocks: list[str], benchmark: str, start: str) -> Panel:
    stocks = [t for t in stocks if t in prices.columns]
    dates = [d for d in month_ends(prices.index) if d >= pd.Timestamp(start)]
    feats, fwd = {}, {}
    for i, d in enumerate(dates):
        f = features_at(prices, d, stocks, benchmark)
        feats[d] = f
        if i + 1 < len(dates):
            nxt = dates[i + 1]
            fwd[d] = prices.loc[nxt, f.index] / prices.loc[d, f.index] - 1.0
    return Panel(dates, feats, fwd)

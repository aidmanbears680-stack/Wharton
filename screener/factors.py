"""Cross-sectional factor scores.

All four factors are price-based so they can be computed point-in-time for any
historical date (free fundamental data is not point-in-time, so it would leak
future information into a backtest). Each factor has decades of published
evidence behind it, and the composite weights are fixed in advance (equal
weight) rather than tuned to the backtest, which keeps the test honest.

* mom_12_1  - 12-month return skipping the latest month (Jegadeesh & Titman 1993)
* mom_6_1   - 6-month return skipping the latest month
* low_vol   - negative 63-day realized volatility (Ang et al. 2006; Frazzini & Pedersen 2014)
* trend     - distance of price above its 200-day average (Faber 2007)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
FACTOR_WEIGHTS: dict[str, float] = {"mom_12_1": 0.25, "mom_6_1": 0.25, "low_vol": 0.25, "trend": 0.25}


def raw_factors(prices: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Raw factor values for every ticker as of `asof` (default: last row)."""
    hist = prices.loc[:asof] if asof is not None else prices
    if len(hist) < TRADING_DAYS + 1:
        raise ValueError("need at least 253 days of history")
    p = hist.iloc[-1]
    p_1m = hist.iloc[-22]
    p_6m = hist.iloc[-127]
    p_12m = hist.iloc[-253]
    rets = hist.iloc[-64:].pct_change().iloc[1:]
    sma200 = hist.iloc[-200:].mean()

    out = pd.DataFrame(
        {
            "mom_12_1": p_1m / p_12m - 1.0,
            "mom_6_1": p_1m / p_6m - 1.0,
            "low_vol": -rets.std() * np.sqrt(TRADING_DAYS),
            "trend": p / sma200 - 1.0,
        }
    )
    # A ticker needs a full year of clean history to be scored.
    full = hist.iloc[-253:].notna().all()
    return out[full].dropna()


def zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score, winsorized at +/-clip so one outlier cannot dominate."""
    z = (s - s.mean()) / s.std(ddof=0)
    z = z.clip(-clip, clip)
    return (z - z.mean()) / z.std(ddof=0)


def composite_scores(prices: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Factor z-scores, the weighted composite, its rank and percentile."""
    raw = raw_factors(prices, asof)
    z = raw.apply(zscore)
    z.columns = [f"z_{c}" for c in z.columns]
    score = sum(w * z[f"z_{f}"] for f, w in FACTOR_WEIGHTS.items())
    out = raw.join(z)
    out["composite"] = score
    out["rank"] = out["composite"].rank(ascending=False).astype(int)
    out["percentile"] = out["composite"].rank(pct=True)
    return out.sort_values("composite", ascending=False)

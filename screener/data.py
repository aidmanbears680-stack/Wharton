"""Price download with a local CSV cache (one file per request signature)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _cache_path(tickers: list[str], start: str, end: str | None) -> Path:
    key = ",".join(sorted(tickers)) + f"|{start}|{end}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"prices_{digest}.csv.gz"


def load_prices(
    tickers: list[str], start: str = "2010-01-01", end: str | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Daily dividend-adjusted closes, one column per ticker.

    Tickers with no data are dropped (with a printed warning) rather than failing
    the whole run.
    """
    path = _cache_path(tickers, start, end)
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        closes.columns = tickers
    closes = closes.sort_index()
    # Drop incomplete rows (e.g. today's, before the market closes).
    closes = closes[closes.notna().mean(axis=1) >= 0.5]
    missing = [t for t in tickers if t not in closes or closes[t].dropna().empty]
    if missing:
        print(f"warning: no price data for {', '.join(missing)}; dropped")
    closes = closes.drop(columns=missing, errors="ignore")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    closes.to_csv(path)
    return closes


def risk_free_daily(prices: pd.DataFrame, rf_col: str = "^IRX") -> pd.Series:
    """Daily risk-free return from the 13-week T-bill yield column (in percent)."""
    if rf_col not in prices:
        return pd.Series(0.0, index=prices.index)
    return (prices[rf_col].ffill().fillna(0.0) / 100.0 / 252.0).rename("rf")

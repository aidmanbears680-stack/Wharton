# Statistical stock screener and goal model: Wharton Global Youth 2026 (Laura Gao case)

A stock screener that tests 14 statistical methods (Bayesian and classical) head to head, corrects for
having tried that many, and documents every trade with the evidence behind it. It also includes a Monte
Carlo model of Laura Gao's goals: the operating reserve, the facility contribution and the range to quote
co-sponsors in 2031.

**Results:** [reports/REPORT.md](reports/REPORT.md) · **Per-trade justifications:** [reports/TRADING_NOTES.md](reports/TRADING_NOTES.md)

## Quick start

```bash
pip install -r requirements.txt
python -m screener report                      # everything -> reports/ (about 3 minutes)
python -m screener screen --capital 100000     # today's picks + trade notes only
python -m screener plan --sleeve-2031 180000   # plan summary + the range to quote in 2031
python -m pytest -q
```

Prices come from Yahoo Finance via `yfinance` and are cached in `data/cache/`. Use `--refresh` to re-download.

## Mapping to the competition deliverables

| Deliverable | Where it comes from |
|---|---|
| Trading Notes Analysis | `reports/TRADING_NOTES.md`: thesis, statistics, risk, decision gates and exit rule per stock |
| Investment Policy Statement | `reports/REPORT.md` §3-4: return assumptions, the definition of "high certainty", reserve design, rebalancing |
| Final Report | `reports/REPORT.md` §4-5: reserve size and composition, facility contribution, stress tests, 2031 co-sponsor range and draft fundraising text |

WInS trading gains are **not** fed into Laura's projections, as the case requires. The plan model starts
from $300k (2027) and $150k (2028) and assumes **zero alpha** from stock picking.

## Methods tested

All methods use the same universe, rebalance monthly, pay 10 bp costs, and see only data available at each date.

| Family | Methods |
|---|---|
| Classic factors | 12-1 momentum, low volatility, 1-month reversal, 4-factor composite |
| Statistical signals | Bollinger z-score mean reversion, risk-adjusted momentum, Kalman-filter trend t-stat |
| **Bayesian** | Empirical-Bayes (James-Stein) shrunk drift; conjugate Bayesian regression learning factor weights under a skeptical prior; Bayesian IC-weighted signal ensemble; **Black-Litterman** (equilibrium prior + factor views) |
| Regression / risk | Rolling Fama-MacBeth premia; minimum variance with Ledoit-Wolf shrinkage; 2-state hidden Markov model regime switch |

Each method is scored with:

- Newey-West alpha vs SPY **and** vs an equal-weight portfolio of the same universe (to neutralize survivorship bias)
- Holm and Benjamini-Hochberg multiple-testing corrections
- Bayesian posterior P(alpha > 0) and the Savage-Dickey Bayes factor under a skeptical prior
- Stationary-bootstrap p-values
- Probabilistic and deflated Sharpe ratios
- Spearman information coefficients

## Code map

| File | Purpose |
|---|---|
| `screener/universe.py` | 131 large caps across 11 sectors, plus benchmark tickers |
| `screener/data.py` | Cached price download |
| `screener/factors.py`, `screener/features.py` | Point-in-time signals: momentum, volatility, trend, reversal, Kalman, empirical Bayes |
| `screener/methods.py` | The 14 portfolio-construction methods |
| `screener/engine.py` | Walk-forward simulation, scorecard, multiple testing, Bayesian alpha |
| `screener/stats.py` | Statistical tests (HAC regression, PSR/DSR, bootstrap, VaR/CVaR, IC) |
| `screener/trade_notes.py` | Live screen and per-trade notes |
| `screener/laura_plan.py` | Monte Carlo of the 2027-2042 plan, reserve certainty, co-sponsor range, stress tests |
| `screener/report.py` | Charts and `REPORT.md` |

## Caveats

- The universe is today's large caps, so the backtest has survivorship bias. The equal-weight benchmark is the honest comparison.
- No method beats that benchmark after the multiple-testing correction. Black-Litterman's edge is likely (posterior 97%) but not proven.
- This is a backtest, not a guarantee, and not investment advice.

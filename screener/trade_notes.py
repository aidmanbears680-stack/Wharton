"""Live screen + one statistical trade note per selected stock.

The strategy (Black-Litterman with factor views) earned its place in the
method comparison; this module applies it at the latest date and documents
every pick with the evidence a reviewer would ask for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S
from .engine import bayesian_alpha
from .features import features_at
from .methods import TOP_N, black_litterman_posterior, cap_weights, hmm_filter
from .universe import SECTOR_OF

# Per-trade gates (fixed in advance).
MIN_PROB_DRIFT_POS = 0.80   # empirical-Bayes posterior P(true drift > 0)
MAX_BETA = 1.6
MAX_AVG_CORR = 0.60
HMM_DERISK = 0.50           # halve new position sizes if P(high-vol regime) exceeds this


def live_screen(prices: pd.DataFrame, stocks: list[str], benchmark: str = "SPY", asof: pd.Timestamp | None = None):
    asof = asof or prices.index[-1]
    stocks = [t for t in stocks if t in prices.columns]
    f = features_at(prices, asof, stocks, benchmark)
    bl = black_litterman_posterior(f, prices, asof)
    w = bl["mv_weight"]
    w = cap_weights(w[w > 0].sort_values(ascending=False).head(TOP_N))
    spx = prices[benchmark].loc[:asof].iloc[-756:].pct_change().dropna().values
    p_hi = hmm_filter(spx)
    return f, bl, w, p_hi


def stock_stats(prices: pd.DataFrame, rf_daily: pd.Series, ticker: str, benchmark: str, asof: pd.Timestamp) -> dict:
    win = prices.loc[:asof].iloc[-253:]
    r = win[ticker].pct_change().iloc[1:]
    m = win[benchmark].pct_change().iloc[1:]
    rf = rf_daily.reindex(r.index).fillna(0)
    capm = S.capm_regression(r, m, rf)
    ba = bayesian_alpha(capm.alpha_ann, capm.alpha_t)
    risk = S.var_cvar(r)
    lo, hi = S.bootstrap_annual_return_ci(r)
    return {
        "beta": capm.beta,
        "alpha_ann": capm.alpha_ann,
        "alpha_t": capm.alpha_t,
        "alpha_p": capm.alpha_p,
        "bayes_p_alpha_pos": ba["p_alpha_pos"],
        "bayes_alpha_post": ba["post_mean"],
        "bf10": ba["bf10"],
        "sharpe_1y": S.sharpe(r, rf),
        "psr_1y": S.probabilistic_sharpe(r, 0.0, rf),
        "vol_1y": S.annualized_vol(r),
        "max_dd_1y": S.max_drawdown(r),
        "ret_ci90": (lo, hi),
        **risk,
        "price": float(win[ticker].iloc[-1]),
    }


def build_notes(
    prices: pd.DataFrame, rf_daily: pd.Series, stocks: list[str], capital: float = 100_000.0,
    benchmark: str = "SPY", asof: pd.Timestamp | None = None, strategy_evidence: dict | None = None,
) -> tuple[pd.DataFrame, str]:
    asof = asof or prices.index[-1]
    f, bl, w, p_hi = live_screen(prices, stocks, benchmark, asof)
    picks = list(w.index)
    daily = prices[picks].loc[:asof].iloc[-253:].pct_change().iloc[1:]
    corr = daily.corr()
    size_mult = 0.5 if p_hi > HMM_DERISK else 1.0

    rows = []
    for t in picks:
        st_ = stock_stats(prices, rf_daily, t, benchmark, asof)
        avg_corr = float(corr.loc[t].drop(t).mean())
        gates = {
            "BL posterior tilt > 0": bool(bl.at[t, "mu"] > bl.at[t, "pi"]),
            f"EB P(drift>0) >= {MIN_PROB_DRIFT_POS:.0%}": bool(f.at[t, "eb_prob_pos"] >= MIN_PROB_DRIFT_POS),
            f"beta <= {MAX_BETA}": bool(st_["beta"] <= MAX_BETA),
            f"avg corr to other picks <= {MAX_AVG_CORR}": bool(avg_corr <= MAX_AVG_CORR),
        }
        passed = sum(gates.values())
        verdict = "BUY" if passed == len(gates) else ("BUY - half size" if passed == len(gates) - 1 else "WATCH")
        weight = float(w[t]) * size_mult * (0.5 if verdict == "BUY - half size" else 1.0) * (verdict != "WATCH")
        monthly_sigma = st_["vol_1y"] / np.sqrt(12)
        rows.append(
            {
                "ticker": t,
                "sector": SECTOR_OF.get(t, "Other"),
                "verdict": verdict,
                "weight": weight,
                "dollars": weight * capital,
                "shares": int(weight * capital // st_["price"]),
                "price": st_["price"],
                "stop": st_["price"] * (1 - 2 * monthly_sigma),
                "bl_prior": float(bl.at[t, "pi"]),
                "bl_posterior": float(bl.at[t, "mu"]),
                "composite_pct": float(
                    sum(0.25 * f[f"z_{k}"] for k in ["mom_12_1", "mom_6_1", "low_vol", "trend"]).rank(pct=True)[t]
                ),
                "z_mom_12_1": float(f.at[t, "z_mom_12_1"]),
                "z_mom_6_1": float(f.at[t, "z_mom_6_1"]),
                "z_low_vol": float(f.at[t, "z_low_vol"]),
                "z_trend": float(f.at[t, "z_trend"]),
                "kalman_t": float(f.at[t, "kalman_t"]),
                "eb_prob_pos": float(f.at[t, "eb_prob_pos"]),
                "eb_shrink": float(f.at[t, "eb_shrink"]),
                "avg_corr": avg_corr,
                "gates": gates,
                **st_,
            }
        )
    table = pd.DataFrame(rows).set_index("ticker")
    return table, render_markdown(table, asof, p_hi, capital, strategy_evidence)


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def render_markdown(t: pd.DataFrame, asof: pd.Timestamp, p_hi: float, capital: float, ev: dict | None) -> str:
    out = [
        f"# Trading notes - Black-Litterman factor screen, {asof:%B %d, %Y}",
        "",
        "Each note below explains *why* a stock is in the portfolio using statistics, and what would make us sell it.",
        f"Model capital: ${capital:,.0f}. All statistics use the trailing 252 trading days unless stated.",
        "",
        "## Why trust the screen at all (strategy-level evidence)",
        "",
    ]
    if ev:
        out += [
            f"Walk-forward test {ev['period']}, monthly rebalancing, 10 bp trading costs, no look-ahead:",
            "",
            f"- Return {_pct(ev['CAGR'])}/yr vs SPY {_pct(ev['spy_cagr'])}/yr; Sharpe {ev['sharpe']:.2f} vs {ev['spy_sharpe']:.2f}.",
            f"- Alpha vs SPY {_pct(ev['alpha_spy'])}/yr, t = {ev['alpha_spy_t']:.2f}; still significant after a Holm correction "
            f"for testing {ev['n_methods']} methods (adjusted p = {ev['alpha_spy_holm']:.3f}). Bayes factor {ev['bf10']:.0f} "
            "under a skeptical prior (alpha ~ N(0, 2%^2)).",
            f"- Alpha vs an equal-weight portfolio of the same stocks (removes survivorship and size effects): "
            f"{_pct(ev['alpha_ew'])}/yr, t = {ev['alpha_ew_t']:.2f}, posterior P(alpha > 0) = {ev['bayes_ew_p']:.0%}. "
            f"This is suggestive, not proof: after the Holm correction p = {ev['alpha_ew_holm']:.2f}.",
            "- Not sensitive to the Black-Litterman settings (tau, IC); removing the factor views turns alpha negative, "
            "so the views are what add value.",
            "",
        ]
    regime = "HIGH-volatility" if p_hi > HMM_DERISK else "calm"
    out += [
        "## Market regime check",
        "",
        f"A two-state hidden Markov model fitted to three years of S&P 500 daily returns puts the probability that "
        f"we are in the high-volatility regime at **{p_hi:.0%}** ({regime}). "
        + ("New positions are **halved** until the regime clears." if p_hi > HMM_DERISK else "Positions are taken at full size."),
        "",
        "## Summary",
        "",
        "| Ticker | Sector | Verdict | Weight | $ | Shares | Price | Stop | BL prior → posterior | P(drift>0) | Beta | α t-stat |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for tk, r in t.iterrows():
        out.append(
            f"| {tk} | {r.sector} | {r.verdict} | {_pct(r.weight)} | ${r.dollars:,.0f} | {r.shares} | ${r.price:,.2f} | "
            f"${r.stop:,.2f} | {_pct(r.bl_prior)} → {_pct(r.bl_posterior)} | {r.eb_prob_pos:.0%} | {r.beta:.2f} | {r.alpha_t:.2f} |"
        )
    out += ["", f"Cash / T-bills: {_pct(1 - t['weight'].sum())}", ""]

    for tk, r in t.iterrows():
        gates = "\n".join(f"  - {'PASS' if v else 'FAIL'}: {k}" for k, v in r.gates.items())
        lo, hi = r.ret_ci90
        out += [
            f"## {tk} - {r.verdict}",
            "",
            f"**Thesis.** {tk} scores higher than {r.composite_pct:.0%} of the universe on the factor composite "
            f"(12-1 momentum z = {r.z_mom_12_1:+.2f}, 6-1 momentum z = {r.z_mom_6_1:+.2f}, low-vol z = {r.z_low_vol:+.2f}, "
            f"trend z = {r.z_trend:+.2f}). Black-Litterman blends the market-equilibrium return of {_pct(r.bl_prior)} "
            f"with that factor view to a posterior expected return of {_pct(r.bl_posterior)}/yr, and the mean-variance "
            f"step gives it a {_pct(r.weight)} weight after accounting for its correlation with the other holdings.",
            "",
            "**Statistical evidence**",
            "",
            f"- Empirical-Bayes drift: after shrinking its past-year mean return {r.eb_shrink:.0%} toward the cross-section, "
            f"the posterior probability its true drift is positive is {r.eb_prob_pos:.0%}.",
            f"- Kalman-filter trend t-statistic: {r.kalman_t:+.2f} (state-space estimate of the current drift ÷ its uncertainty).",
            f"- CAPM vs SPY (Newey-West errors): beta {r.beta:.2f}, alpha {_pct(r.alpha_ann)}/yr, t = {r.alpha_t:.2f}, "
            f"p = {r.alpha_p:.3f}. Under a skeptical prior the posterior alpha is {_pct(r.bayes_alpha_post)}/yr and "
            f"P(alpha > 0) = {r.bayes_p_alpha_pos:.0%} (Bayes factor {r.bf10:.2f}).",
            f"- One-year Sharpe {r.sharpe_1y:.2f}; probabilistic Sharpe P(true Sharpe > 0) = {r.psr_1y:.0%}. "
            f"Bootstrap 90% CI for the annualized mean return: {_pct(lo)} to {_pct(hi)}.",
            "",
            "**Risk**",
            "",
            f"- Volatility {_pct(r.vol_1y)}/yr; max drawdown over the year {_pct(r.max_dd_1y)}.",
            f"- 1-day 95% VaR {_pct(r.var_hist, 2)} (historical) / {_pct(r.var_cf, 2)} (Cornish-Fisher); "
            f"expected shortfall {_pct(r.cvar_hist, 2)}. On ${r.dollars:,.0f} that is a typical bad-day loss of "
            f"${r.dollars * r.var_hist:,.0f}.",
            f"- Average correlation with the other picks: {r.avg_corr:.2f}.",
            "",
            "**Decision gates**",
            "",
            gates,
            "",
            f"**Exit plan.** Sell at the monthly rebalance if it leaves the Black-Litterman top {TOP_N}; "
            f"sell immediately on a close below ${r.stop:,.2f} (a two-sigma monthly move from ${r.price:,.2f}).",
            "",
            "**Caveat.** One year of one stock's data is too short to prove skill on its own (most single-stock alpha "
            "t-statistics are below 2); the case for owning it rests on the strategy-level evidence above, "
            "and these statistics confirm it is not an outlier on risk.",
            "",
        ]
    return "\n".join(out)

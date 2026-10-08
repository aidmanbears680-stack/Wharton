"""Walk-forward simulation of any method and a head-to-head statistical scorecard."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.stats.multitest import multipletests

from . import stats as S
from .methods import METHODS, Context

COST_BPS = 10.0
SKEPTICAL_ALPHA_SD = 0.02  # prior sd of annual alpha: most managers have |alpha| < 2%/yr


def simulate(ctx: Context, method, start: str) -> dict:
    """Hold each month's weights buy-and-hold to the next month-end, net of trading costs."""
    prices = ctx.prices
    rets = prices.pct_change()
    dates = [d for d in ctx.panel.dates if d >= pd.Timestamp(start)]
    daily = pd.Series(np.nan, index=prices.index)
    scores, holdings, turnover = {}, {}, {}
    prev = pd.Series(dtype=float)
    for d, nxt in zip(dates[:-1], dates[1:]):
        w, s = method(ctx, d)
        w = w[w > 0]
        holdings[d] = w
        if s is not None:
            scores[d] = s
        period = rets.loc[d:nxt].iloc[1:]
        growth = (1 + period[w.index].fillna(0)).cumprod()
        value = growth.mul(w, axis=1).sum(axis=1)
        pr = value.pct_change()
        pr.iloc[0] = value.iloc[0] - 1.0
        idx = w.index.union(prev.index)
        to = (w.reindex(idx, fill_value=0) - prev.reindex(idx, fill_value=0)).abs().sum()
        pr.iloc[0] -= to * COST_BPS / 1e4
        turnover[d] = to
        daily.loc[pr.index] = pr
        drift = growth.iloc[-1] * w
        prev = drift / drift.sum()
    return {"daily": daily.dropna(), "scores": scores, "holdings": holdings, "turnover": pd.Series(turnover)}


def equal_weight_universe(ctx: Context, start: str) -> pd.Series:
    def ew(c, d):
        names = c.panel.feats[d].index
        return pd.Series(1 / len(names), index=names), None

    return simulate(ctx, ew, start)["daily"]


def bayesian_alpha(alpha: float, t: float, prior_sd: float = SKEPTICAL_ALPHA_SD) -> dict:
    """Normal-normal posterior for annual alpha under a skeptical N(0, prior_sd^2) prior.

    Also returns the Savage-Dickey Bayes factor BF10 for "alpha != 0" vs "alpha = 0":
    prior density at zero divided by posterior density at zero. BF10 > 3 is
    conventionally "substantial" evidence, > 10 "strong", < 1/3 favours no alpha.
    """
    se = abs(alpha / t) if t != 0 else np.inf
    post_var = 1 / (1 / prior_sd**2 + 1 / se**2)
    post_mean = post_var * alpha / se**2
    post_sd = np.sqrt(post_var)
    bf10 = st.norm.pdf(0, 0, prior_sd) / st.norm.pdf(0, post_mean, post_sd)
    return {
        "post_mean": post_mean,
        "post_sd": post_sd,
        "p_alpha_pos": float(st.norm.cdf(post_mean / post_sd)),
        "bf10": float(bf10),
    }


def evaluate(name: str, desc: str, sim: dict, bench: pd.DataFrame) -> dict:
    """bench: daily DataFrame with columns SPY, EW universe, rf aligned to the strategy."""
    d = pd.concat([sim["daily"].rename("strategy"), bench], axis=1).dropna()
    m = (1 + d).groupby([d.index.year, d.index.month]).prod() - 1
    capm = S.capm_regression(m["strategy"], m["SPY"], m["rf"], periods=12, hac_lags=3)
    vs_ew = S.capm_regression(m["strategy"], m["EW universe"], m["rf"], periods=12, hac_lags=3)
    excess = m["strategy"] - m["EW universe"]
    boot_p, _ = S.stationary_bootstrap_pvalue(excess, n_boot=2000)
    row = {
        "method": name,
        "description": desc,
        "CAGR": S.annualized_return(d["strategy"]),
        "vol": S.annualized_vol(d["strategy"]),
        "sharpe": S.sharpe(d["strategy"], d["rf"]),
        "max_dd": S.max_drawdown(d["strategy"]),
        "cvar95_monthly": S.var_cvar(m["strategy"])["cvar_hist"],
        "beta": capm.beta,
        "alpha_spy": capm.alpha_ann,
        "alpha_spy_t": capm.alpha_t,
        "alpha_spy_p": capm.alpha_p,
        "alpha_ew": vs_ew.alpha_ann,
        "alpha_ew_t": vs_ew.alpha_t,
        "alpha_ew_p": vs_ew.alpha_p,
        "excess_ew_ann": float(excess.mean() * 12),
        "excess_ew_boot_p": boot_p,
        "psr_vs_spy": S.probabilistic_sharpe(
            m["strategy"], (m["SPY"] - m["rf"]).mean() / (m["SPY"] - m["rf"]).std(), m["rf"]
        ),
        "turnover": float(sim["turnover"].mean()),
    }
    ba = bayesian_alpha(capm.alpha_ann, capm.alpha_t)
    row.update({f"bayes_spy_{k}": v for k, v in ba.items()})
    be = bayesian_alpha(vs_ew.alpha_ann, vs_ew.alpha_t)
    row.update({f"bayes_ew_{k}": v for k, v in be.items()})

    if sim["scores"]:
        fw = {k: v for k, v in sim.get("fwd", {}).items()}
        ic = S.information_coefficients(pd.DataFrame(sim["scores"]).T, pd.DataFrame(fw).T)
        mean, t, p = S.mean_t_test(ic, hac_lags=3)
        row.update({"ic_mean": mean, "ic_t": t, "ic_p": p, "ic_hit": float((ic > 0).mean())})
        sim["ic"] = ic
    else:
        row.update({"ic_mean": np.nan, "ic_t": np.nan, "ic_p": np.nan, "ic_hit": np.nan})
    return row


def run_all(ctx: Context, start: str, methods: dict | None = None, verbose: bool = True):
    methods = methods or METHODS
    ew = equal_weight_universe(ctx, start)
    rets = ctx.prices.pct_change()
    from .data import risk_free_daily

    rf = risk_free_daily(ctx.prices)
    bench = pd.DataFrame({"SPY": rets[ctx.benchmark], "EW universe": ew, "rf": rf}).loc[ew.index]
    fwd = {d: ctx.panel.fwd[d] for d in ctx.panel.fwd}
    rows, sims = [], {}
    for name, (desc, fn) in methods.items():
        t0 = time.time()
        sim = simulate(ctx, fn, start)
        sim["fwd"] = fwd
        rows.append(evaluate(name, desc, sim, bench))
        sims[name] = sim
        if verbose:
            r = rows[-1]
            print(f"{name:18s} CAGR {r['CAGR']:6.1%}  Sharpe {r['sharpe']:.2f}  alpha/SPY t {r['alpha_spy_t']:5.2f}  "
                  f"alpha/EW t {r['alpha_ew_t']:5.2f}  IC t {r['ic_t']:5.2f}  ({time.time() - t0:.0f}s)")
    table = pd.DataFrame(rows).set_index("method")
    table = add_multiple_testing(table, sims, bench)
    return table, sims, bench


def add_multiple_testing(table: pd.DataFrame, sims: dict, bench: pd.DataFrame) -> pd.DataFrame:
    """Correct for having tried many methods.

    * Holm-Bonferroni controls the chance of even one false discovery (FWER).
    * Benjamini-Hochberg controls the expected share of false discoveries (FDR).
    * Deflated Sharpe ratio asks whether the Sharpe beats the best you'd expect
      from this many tries if none of them had real skill.
    """
    for col in ["alpha_spy_p", "alpha_ew_p"]:
        p = table[col].values
        table[col.replace("_p", "_holm")] = multipletests(p, method="holm")[1]
        table[col.replace("_p", "_bh")] = multipletests(p, method="fdr_bh")[1]
    # Applied to *active* returns (information ratio): every long-only equity
    # method has a positive raw Sharpe, so deflating the raw Sharpe says nothing
    # about selection skill.
    n = len(table)
    for bench_col, label in [("SPY", "spy"), ("EW universe", "ew")]:
        active = {}
        for name, sim in sims.items():
            d = pd.concat([sim["daily"].rename("s"), bench[bench_col]], axis=1).dropna()
            m = (1 + d).groupby([d.index.year, d.index.month]).prod() - 1
            active[name] = m["s"] - m[bench_col]
        irs = np.array([a.mean() / a.std() for a in active.values()])
        table[f"dsr_{label}"] = [S.deflated_sharpe(a, n, float(irs.std(ddof=1))) for a in active.values()]
    return table

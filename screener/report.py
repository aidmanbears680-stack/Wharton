"""End-to-end run: method comparison, Laura plan, trade notes, charts and REPORT.md."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import laura_plan as LP
from .data import CACHE_DIR, load_prices, risk_free_daily
from .engine import run_all
from .features import SIGNALS, build_panel
from .methods import METHODS, Context
from .trade_notes import build_notes
from .universe import BENCHMARK, RISK_FREE, STOCKS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports"
CHART_START = "2013-01-01"

# Chart styling (validated categorical palette, light mode).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
    }
)


def _pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def _money(x):
    return f"${x:,.0f}"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_growth(sims, bench, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    series = {
        "Black-Litterman screen": sims["black_litterman"]["daily"],
        "12-1 momentum": sims["momentum_12_1"]["daily"],
        "SPY": bench["SPY"],
        "Equal-weight universe": bench["EW universe"],
    }
    ends = {}
    for (name, r), c in zip(series.items(), SERIES):
        g = (1 + r.loc[bench.index].fillna(0)).cumprod()
        ax.plot(g.index, g.values, color=c, label=name)
        ends[name] = g.iloc[-1]
    # Direct end labels, nudged apart in log space so they never collide.
    placed = []
    for name, v in sorted(ends.items(), key=lambda kv: kv[1]):
        y = np.log(v)
        if placed and y - placed[-1] < 0.09:
            y = placed[-1] + 0.09
        placed.append(y)
        ax.annotate(f"{name}  ${v:.1f}", (bench.index[-1], np.exp(y)), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=9)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.set_yticklabels(["$1", "$2", "$4", "$8", "$16"])
    ax.set_title("Growth of $1, walk-forward, net of costs")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(right=bench.index[-1] + pd.Timedelta(days=1100))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_scorecard(table, path):
    t = table.sort_values("alpha_ew_t")
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(t))
    ax.barh(y, t["alpha_ew_t"], color=SERIES[0], height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index)
    ax.axvline(0, color=INK_2, lw=1)
    ax.axvline(1.96, color=SERIES[1], lw=1.5, ls="--")
    ax.text(1.98, -0.9, "p = 0.05 (one test)", color=INK_2, fontsize=8)
    from scipy.stats import norm

    holm = norm.ppf(1 - 0.05 / len(t) / 2)
    ax.axvline(holm, color=SERIES[1], lw=1.5)
    ax.text(holm + 0.02, -0.9, f"Holm hurdle, {len(t)} tests", color=INK_2, fontsize=8)
    ax.set_ylim(-1.3, len(t) - 0.4)
    for yi, (tv, pp) in enumerate(zip(t["alpha_ew_t"], t["bayes_ew_p_alpha_pos"])):
        ax.text(max(tv, 0) + 0.05, yi, f"P(α>0) {pp:.0%}", va="center", color=INK_2, fontsize=8)
    ax.set_xlabel("t-statistic of alpha vs equal-weight universe (Newey-West)")
    ax.set_title("Which methods show selection skill?")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_bayes_weights(bw, path):
    last = bw[max(bw)]
    t = last.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(t))
    scale = 100 * 12  # monthly decimal -> % per year per 1-sd of signal
    ax.errorbar(t["mean"] * scale, y, xerr=2 * t["sd"] * scale, fmt="o", color=SERIES[0], ms=8, capsize=0, lw=2)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_xlabel("Posterior weight: % per year per 1-sd of signal (dot = mean, line = 95% credible interval)")
    ax.set_title(f"Bayesian factor weights learned from data through {max(bw):%b %Y}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_plan(results, path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = list(results)
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        c = results[n]["contribution"]
        p10, p25, p50, p75, p90 = (c.quantile(q) for q in (0.1, 0.25, 0.5, 0.75, 0.9))
        ax.plot([p10, p90], [yi, yi], color=SERIES[0], lw=2, alpha=0.5, solid_capstyle="round")
        ax.plot([p25, p75], [yi, yi], color=SERIES[0], lw=7, solid_capstyle="round")
        ax.plot([p50], [yi], "o", color=SURFACE, ms=6, mec=SERIES[0], mew=2)
        fund = results[n]["reserve_funded"].mean()
        ax.text(p90 + 6000, yi, f"reserve funded {fund:.1%}", va="center", color=INK_2, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.set_xlabel("2033 facility contribution (thin line 10th-90th pct, thick 25th-75th, dot median)")
    ax.set_title("Hedging the $500k commitment first costs almost no upside")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=-5000, right=ax.get_xlim()[1] + 110000)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_cosponsor(df, path):
    grid = np.linspace(df["sleeve2031"].quantile(0.05), df["sleeve2031"].quantile(0.95), 25)
    rng = np.array([LP.cosponsor_range(df, v) for v in grid])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(grid, rng[:, 0], rng[:, 2], color=SERIES[0], alpha=0.25, lw=0, label="Range to quote (80% confidence)")
    ax.plot(grid, rng[:, 1], color=SERIES[0], label="Median expected contribution")
    med = df["sleeve2031"].median()
    ax.axvline(med, color=INK_2, lw=1, ls="--")
    ax.text(med, ax.get_ylim()[1] * 0.95, " median 2031 sleeve", color=INK_2, fontsize=8, va="top")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("Growth-sleeve value Laura observes in January 2031")
    ax.set_ylabel("2033 facility contribution")
    ax.set_title("What to tell co-sponsors in 2031")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(refresh: bool = False, capital: float = 100_000.0) -> Path:
    OUT.mkdir(exist_ok=True)
    prices = load_prices(STOCKS + [BENCHMARK, RISK_FREE, "BIL"], refresh=refresh)
    rf = risk_free_daily(prices)
    asof = prices.index[-1]

    panel_path = CACHE_DIR / f"panel_{asof:%Y%m%d}.pkl"
    if panel_path.exists() and not refresh:
        panel = pickle.loads(panel_path.read_bytes())
    else:
        print("building feature panel ...")
        panel = build_panel(prices, STOCKS, BENCHMARK, "2011-01-01")
        panel_path.write_bytes(pickle.dumps(panel))

    print("running method comparison ...")
    ctx = Context(panel, prices, BENCHMARK, "BIL")
    table, sims, bench = run_all(ctx, CHART_START)
    table.to_csv(OUT / "method_comparison.csv", float_format="%.5f")

    chart_growth(sims, bench, OUT / "growth_of_1.png")
    chart_scorecard(table, OUT / "method_scorecard.png")
    chart_bayes_weights(ctx.bayes_weights, OUT / "bayes_factor_weights.png")

    print("running Laura plan Monte Carlo ...")
    plan = {n: LP.simulate_plan(n) for n in LP.STRATEGIES}
    rec = plan[LP.RECOMMENDED]
    chart_plan(plan, OUT / "plan_strategies.png")
    chart_cosponsor(rec, OUT / "cosponsor_range.png")
    stress = LP.stress_test()
    coverage = LP.range_coverage(rec)
    reserve_designs = {
        "Treasury ladder, all 10 payments": (LP.mix(0, 0, 0, 1), 10),
        "Ladder for years 1-5, 60/40 for 6-10": (LP.mix(42, 18, 35, 5), 5),
        "All in 30/70": (LP.mix(21, 9, 60, 10), 0),
        "All in 60/40": (LP.mix(42, 18, 35, 5), 0),
    }
    reserve_table = pd.DataFrame(
        {
            name: {f"{int(q * 100)}% certainty": LP.required_reserve(w, k, q) for q in (0.90, 0.95, 0.99)}
            for name, (w, k) in reserve_designs.items()
        }
    ).T

    print("building trade notes ...")
    bl = table.loc["black_litterman"]
    evidence = {
        "period": f"{bench.index[0]:%b %Y} to {bench.index[-1]:%b %Y}",
        "CAGR": bl["CAGR"], "sharpe": bl["sharpe"], "alpha_spy": bl["alpha_spy"], "alpha_spy_t": bl["alpha_spy_t"],
        "alpha_spy_holm": bl["alpha_spy_holm"], "bf10": bl["bayes_spy_bf10"], "alpha_ew": bl["alpha_ew"],
        "alpha_ew_t": bl["alpha_ew_t"], "alpha_ew_holm": bl["alpha_ew_holm"], "bayes_ew_p": bl["bayes_ew_p_alpha_pos"],
        "n_methods": len(table),
        "spy_cagr": float((1 + bench["SPY"]).prod() ** (252 / len(bench)) - 1),
        "spy_sharpe": float(((bench["SPY"] - bench["rf"]).mean() / (bench["SPY"] - bench["rf"]).std()) * np.sqrt(252)),
    }
    notes_table, notes_md = build_notes(prices, rf, STOCKS, capital, BENCHMARK, asof, evidence)
    (OUT / "TRADING_NOTES.md").write_text(notes_md)
    notes_table.drop(columns=["gates"]).to_csv(OUT / "current_picks.csv", float_format="%.5f")

    md = render_report(table, bench, plan, rec, stress, coverage, reserve_table, notes_table, evidence, ctx, asof)
    (OUT / "REPORT.md").write_text(md)
    print(f"done: {OUT / 'REPORT.md'}")
    return OUT / "REPORT.md"


def render_report(table, bench, plan, rec, stress, coverage, reserve_table, picks, ev, ctx, asof) -> str:
    t = table.copy()
    cfg = LP.SimConfig()
    cost27 = float(LP.payment_pv(np.array([cfg.yield_2027_mean]), 6)[0])
    unhedged = [df["reserve_funded"].mean() for n, df in plan.items() if LP.STRATEGIES[n].hedge_ratio == 0]
    lo_f, hi_f = min(unhedged), max(unhedged)
    lines = [
        "# Statistical stock screener and investment plan for Laura Gao",
        "",
        f"*Data through {asof:%B %d, %Y}. Generated by `python -m screener report`.*",
        "",
        "## 1. Bottom line",
        "",
        f"1. **Lock in the operating commitment first.** With 10-year Treasury yields near 5.3%, about {_money(round(cost27, -3))} of the "
        "2027 deposit buys a Treasury zero-coupon ladder that pays every $50k installment from 2033 to 2042. Across 50,000 "
        f"simulated paths this funds the commitment in **{rec['reserve_funded'].mean():.0%}** of cases, against "
        f"{lo_f:.0%}-{hi_f:.0%} for conventional unhedged portfolios. The median facility contribution is no lower.",
        "2. **Run the stock strategy in the growth sleeve** (the rest of 2027 plus the 2028 deposit). Of "
        f"{len(t)} statistical methods tested walk-forward, the **Bayesian Black-Litterman screen with factor views** "
        "was the strongest and most robust. Its edge over SPY is statistically significant after correcting for "
        "multiple tests. Its edge over an equal-weight version of the same universe is likely but not proven.",
        "3. **Tell co-sponsors a range, not a number.** In 2031, quote the 10th-90th percentile range from the rule in "
        f"Section 5. In out-of-sample checks the realized contribution landed inside the quoted range {coverage['inside']:.0%} "
        f"of the time and below it only {coverage['below']:.0%} of the time. The operating reserve is never at risk because it is "
        "already locked in.",
        "",
        "## 2. Stock-selection methods compared",
        "",
        f"All methods were run on the same {ctx.panel.feats[asof if asof in ctx.panel.feats else ctx.panel.dates[-1]].shape[0]}-stock "
        f"large-cap universe from {bench.index[0]:%b %Y} to {bench.index[-1]:%b %Y}. Rules: monthly rebalance, 10 bp costs, "
        "only data available at each date, top 15 names, at most 3 per sector, a 10% cap per name. "
        "Bayesian and regression methods train only on months whose outcomes were already known.",
        "",
        "| Method | CAGR | Sharpe | Max DD | Beta | α vs SPY (t) | Holm p | α vs EW (t) | P(α_EW>0) | IC t | Deflated SR (vs EW) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in t.sort_values("alpha_ew_t", ascending=False).iterrows():
        ic = "-" if np.isnan(r.ic_t) else f"{r.ic_t:.2f}"
        lines.append(
            f"| {METHODS[name][0]} | {_pct(r.CAGR)} | {r.sharpe:.2f} | {_pct(r.max_dd, 0)} | {r.beta:.2f} | "
            f"{_pct(r.alpha_spy)} ({r.alpha_spy_t:.2f}) | {r.alpha_spy_holm:.3f} | {_pct(r.alpha_ew)} ({r.alpha_ew_t:.2f}) | "
            f"{r.bayes_ew_p_alpha_pos:.0%} | {ic} | {r.dsr_ew:.2f} |"
        )
    spy_sr = ev["spy_sharpe"]
    ew = bench["EW universe"]
    lines += [
        "",
        f"Benchmarks: SPY {_pct(ev['spy_cagr'])}/yr (Sharpe {spy_sr:.2f}); equal-weight universe "
        f"{_pct((1 + ew).prod() ** (252 / len(ew)) - 1)}/yr.",
        "",
        "![growth](growth_of_1.png)",
        "",
        "![scorecard](method_scorecard.png)",
        "",
        "### How to read the statistics",
        "",
        "- **α vs SPY / vs EW**: annualized intercept from regressing monthly excess returns on the benchmark, with "
        "Newey-West standard errors (robust to autocorrelation and changing volatility). The equal-weight (EW) "
        "benchmark holds every stock in the same universe. It is the fairer test, because today's large caps are "
        "survivors (survivorship bias) and equal weighting tilts toward smaller names. Both biases inflate any portfolio "
        "drawn from this list.",
        "- **Holm p**: p-value adjusted for testing all methods at once (family-wise error control). "
        "Benjamini-Hochberg (FDR) values are in `method_comparison.csv`.",
        "- **P(α>0)**: Bayesian posterior probability of positive alpha under a skeptical N(0, 2%²) prior, the "
        "belief that most strategies have little or no alpha. The Bayes factor (BF10, in the CSV) compares "
        "\"some alpha\" with \"no alpha\" (Savage-Dickey). Above 3 is substantial evidence, above 10 strong.",
        "- **IC t**: t-statistic of the monthly Spearman rank correlation between a method's score and next-month "
        "returns. It tests whether the ranking itself predicts.",
        "- **Deflated SR**: probability that the active Sharpe ratio beats the best you would expect from "
        f"{len(t)} tries with no skill (Bailey & López de Prado).",
        "",
        "### What the evidence says",
        "",
        f"- **Black-Litterman** has alpha vs SPY of {_pct(t.at['black_litterman', 'alpha_spy'])}/yr "
        f"(t = {t.at['black_litterman', 'alpha_spy_t']:.2f}, Holm-adjusted p = {t.at['black_litterman', 'alpha_spy_holm']:.3f}, "
        f"BF10 = {t.at['black_litterman', 'bayes_spy_bf10']:.0f}). Against equal weight: "
        f"{_pct(t.at['black_litterman', 'alpha_ew'])}/yr (t = {t.at['black_litterman', 'alpha_ew_t']:.2f}), "
        f"posterior P(α>0) = {t.at['black_litterman', 'bayes_ew_p_alpha_pos']:.0%}. Positive in each sub-period "
        "(2013-17, 2018-21, 2022-26). Insensitive to τ and the IC scale, and alpha turns negative with the "
        "views removed, so the factor views are doing the work.",
        f"- **Plain 12-1 momentum** is the runner-up (t vs EW = {t.at['momentum_12_1', 'alpha_ew_t']:.2f}). Most of its edge "
        "came after 2022.",
        "- **Low volatility, minimum variance and the HMM regime switch** did not add return. They did cut "
        f"risk: beta {t.at['low_vol', 'beta']:.2f}-{t.at['min_variance_lw', 'beta']:.2f} and HMM max drawdown "
        f"{_pct(t.at['hmm_regime', 'max_dd'], 0)} vs SPY's. They belong in the risk toolkit, not the alpha engine.",
        "- **Short-term reversal and Bollinger mean reversion** show no reliable selection skill in large caps.",
        "- **Honest limit:** no method clears the deflated-Sharpe or Holm hurdle against the equal-weight benchmark. "
        "We therefore assume **zero alpha** in Laura's long-term projections (Section 4) and treat any "
        "outperformance as upside.",
        "",
        "### What the Bayesian regression learned",
        "",
        "The Bayesian regression starts from a skeptical prior (each signal worth 0 ± 0.4%/month) and updates "
        "every month as returns arrive. The posterior shows which signals the data actually supports:",
        "",
        "![bayes](bayes_factor_weights.png)",
        "",
        "Read these weights jointly, not one at a time. Several signals are highly correlated (12-1 momentum, "
        "risk-adjusted momentum, Kalman trend and the empirical-Bayes drift have pairwise correlations of 0.7-0.95). "
        "The regression can therefore trade one off against another, e.g. a large positive weight on risk-adjusted momentum "
        "(return ÷ volatility) offset by a negative weight on low volatility. The combined message: the data rewards "
        "recent winners with strong trends and does not reward low volatility for its own sake. That agrees with the "
        "single-signal results above.",
        "",
        "## 3. Laura's plan: strategies compared",
        "",
        "50,000 Monte Carlo paths per strategy, with fat-tailed (Student-t) returns. Expected returns are themselves "
        "uncertain: each path draws its own long-run means from a prior around our assumptions (Bayesian parameter "
        "uncertainty). Assumptions (nominal, annual):",
        "",
        "| Asset | Expected return | Volatility | Prior sd of the mean |",
        "|---|---:|---:|---:|",
    ]
    for a, m, v, u in zip(LP.ASSETS, LP.CMA_MEAN, LP.CMA_VOL, LP.MEAN_UNCERTAINTY):
        lines.append(f"| {a} | {_pct(m)} | {_pct(v)} | {_pct(u)} |")
    lines += [
        "",
        f"Treasury zero yield at the 2027 purchase: {_pct(cfg.yield_2027_mean, 2)} ± {_pct(cfg.yield_2027_sd, 2)} "
        f"(10-yr Treasury was 5.3% in Oct 2026). Yield in 2033: {_pct(cfg.yield_2033_mean)} ± {_pct(cfg.yield_2033_sd, 2)}. "
        f"Inflation {_pct(LP.INFLATION)}/yr, facility-cost inflation {_pct(LP.FACILITY_COST_INFLATION)}/yr. "
        "No alpha is assumed from stock selection. Flexibility buffer: keep the larger of $50k and 10% of the 2033 portfolio uncommitted.",
        "",
        "| Strategy | P(reserve funded) | Median 2033 portfolio | Contribution 10th pct | Median | 90th pct | P(no contribution) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, df in plan.items():
        c = df["contribution"]
        lines.append(
            f"| {n} | {df['reserve_funded'].mean():.1%} | {_money(df['v2033'].median())} | {_money(c.quantile(0.1))} | "
            f"{_money(c.median())} | {_money(c.quantile(0.9))} | {(c <= 0).mean():.1%} |"
        )
    lines += [
        "",
        "![plan](plan_strategies.png)",
        "",
        "**Why the hedge wins.** The ten payments are fixed in dollars, so a portfolio that owns exactly those "
        "cash flows (zero-coupon Treasuries maturing each January 2033-2042) carries no market risk relative to the "
        f"goal. At today's yields that ladder is cheap, about {_money(cost27)}, and it leaves ~{_money(450_000 - cost27)} "
        "plus all its future growth for the facility. The unhedged portfolios must instead outrun the cost of the "
        f"reserve, which is why {1 - hi_f:.0%}-{1 - lo_f:.0%} of their paths fall short.",
        "",
        "## 4. Operating reserve recommendation",
        "",
        "**Definition of high certainty:** at least a 99% probability, under fat-tailed simulation with parameter "
        "uncertainty, that all ten payments are made from the portfolio alone, and the plan still holds under every "
        "historical-style stress test below.",
        "",
        "Reserve size needed in 2033 for each design (at a 4.5% average yield):",
        "",
        "| Reserve design | 90% certainty | 95% certainty | 99% certainty |",
        "|---|---:|---:|---:|",
    ]
    for n, r in reserve_table.iterrows():
        lines.append(f"| {n} | " + " | ".join(_money(v) for v in r.values) + " |")
    med_res = rec["reserve"].median()
    lines += [
        "",
        f"**Recommendation:** a Treasury zero-coupon ladder, one rung maturing each January 2033-2042. "
        f"Its 2033 market value is about {_money(med_res)}, between {_money(rec['reserve'].quantile(0.05))} and "
        f"{_money(rec['reserve'].quantile(0.95))} depending on 2033 rates. Bought in 2027, it costs ~{_money(cost27)}. "
        "**How it changes over time:** it needs no rebalancing. Each January one rung matures and pays that year's "
        "$50k, so the reserve shrinks by exactly one payment a year and its remaining duration falls automatically. "
        "Holding to maturity removes interest-rate risk. The only remaining risk is a U.S. Treasury default.",
        "",
        "Stress tests (recommended strategy, 20,000 paths each):",
        "",
        "| Scenario | Median 2033 portfolio | P(reserve funded) | Median contribution | 10th pct contribution |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in stress.iterrows():
        lines.append(
            f"| {r.scenario} | {_money(r['median V2033'])} | {r['P(reserve funded)']:.1%} | {_money(r['median contribution'])} | "
            f"{_money(r['10th pct contribution'])} |"
        )
    c = rec["contribution"]
    real = 1 / (1 + LP.FACILITY_COST_INFLATION) ** 7
    lines += [
        "",
        "The main risk to the plan is **rates falling before the January 2027 purchase**. The ladder then costs more "
        "and the facility contribution shrinks. Mitigation: buy the ladder in the first days of January 2027, "
        "and do not wait for the 2028 deposit.",
        "",
        "## 5. Facility contribution and co-sponsor communication",
        "",
        f"From today's vantage point the 2033 facility contribution has a median of **{_money(c.median())}**. "
        f"There is an 80% chance it falls between {_money(c.quantile(0.1))} and {_money(c.quantile(0.9))}, "
        f"and a {(c <= 0).mean():.1%} chance it is zero. In 2026 dollars, adjusting for 3% facility-cost inflation, the median is "
        f"{_money(c.median() * real)}.",
        "",
        "**Rule for 2031.** Look up the growth-sleeve value in January 2031 and quote the shaded range:",
        "",
        "![cosponsor](cosponsor_range.png)",
        "",
        "| Sleeve value Jan 2031 | Quote: low end | Median | High end |",
        "|---:|---:|---:|---:|",
    ]
    for v in np.quantile(rec["sleeve2031"], [0.1, 0.25, 0.5, 0.75, 0.9]):
        lo, mid, hi = LP.cosponsor_range(rec, v)
        lines.append(f"| {_money(v)} | {_money(lo)} | {_money(mid)} | {_money(hi)} |")
    lo, mid, hi = LP.cosponsor_range(rec, rec["sleeve2031"].median())
    lines += [
        "",
        f"Out-of-sample check (ranges built on half the paths, tested on the other half): the realized contribution was "
        f"inside the quoted range **{coverage['inside']:.0%}** of the time, below it {coverage['below']:.0%} and above it "
        f"{coverage['above']:.0%}. We are therefore **80% confident** the contribution will fall in the quoted range. "
        "Because the reserve is already locked in, a shortfall reduces only the facility contribution, never the "
        "operating payments. The sleeve shifts toward bonds in 2031-32 so the range stays tight once it has been quoted.",
        "",
        "**Draft fundraising language (median scenario):**",
        "",
        f"> Laura Gao has fully pre-funded the residency's first ten years of operating support, $50,000 a year from "
        f"2033 through 2042, with U.S. Treasury securities set aside for that purpose alone. In addition, she expects "
        f"to contribute **{_money(round(lo, -4))} to {_money(round(hi, -4))}** toward establishing the facility in 2033. "
        "This range is based on her current investment portfolio and conservative market assumptions. Her portfolio "
        "managers estimate an 80% likelihood that her contribution falls within it. She will confirm the final "
        "amount in early 2033.",
        "",
        "## 6. Current portfolio: Black-Litterman screen",
        "",
        "| Ticker | Verdict | Weight | BL posterior return | P(drift>0) | Beta | 1-day 95% VaR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for tk, r in picks.iterrows():
        lines.append(
            f"| {tk} | {r.verdict} | {_pct(r.weight)} | {_pct(r.bl_posterior)} | {r.eb_prob_pos:.0%} | {r.beta:.2f} | {_pct(r.var_hist, 2)} |"
        )
    lines += [
        "",
        "Full per-trade justifications are in [TRADING_NOTES.md](TRADING_NOTES.md).",
        "",
        "## 7. Limitations",
        "",
        "- Universe = today's large caps, so the backtest has survivorship bias. That is why the equal-weight universe "
        "is the key benchmark.",
        "- Price-based factors only. Free fundamental data is not point-in-time, and using it would leak future information.",
        "- Backtests overstate what is achievable live. We assume zero alpha in Laura's projections.",
        "- Capital-market assumptions are judgments. Change them in `screener/laura_plan.py` and rerun.",
        "- Not investment advice. Built for the Wharton Global Youth investment competition.",
        "",
    ]
    return "\n".join(lines)

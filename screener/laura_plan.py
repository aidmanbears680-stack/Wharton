"""Monte Carlo model of Laura Gao's goals (Wharton Global Youth 2026 case).

Cash flows (beginning of year, nominal):
  2027 (Yr 1)  +$300,000
  2028 (Yr 2)  +$150,000
  2033 (Yr 7)  operating reserve set aside for ten $50,000 payments, 2033-2042
               then the facility contribution from what remains

Returns are simulated, not taken from the WInS trading period (as the case requires).

Two layers of uncertainty are simulated:
1. Market noise: annual returns are multivariate Student-t (fat tails, df=5).
2. Parameter uncertainty (Bayesian): each path first draws its own "true"
   expected returns from a prior around our capital-market assumptions, so the
   spread of outcomes reflects that we don't know the long-run means exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ASSETS = ["US equity", "Intl equity", "Core bonds", "T-bills"]

# Capital-market assumptions: nominal, annual arithmetic means and volatilities.
# Equities are deliberately below their 2013-2026 history (starting valuations
# are high); bonds and bills reflect October 2026 yields (10-yr Treasury ~5.3%,
# 3-month bill ~4.0%) with some expected decline.
CMA_MEAN = np.array([0.065, 0.070, 0.048, 0.035])
CMA_VOL = np.array([0.160, 0.170, 0.060, 0.010])
CMA_CORR = np.array(
    [
        [1.00, 0.80, 0.10, 0.00],
        [0.80, 1.00, 0.15, 0.00],
        [0.10, 0.15, 1.00, 0.30],
        [0.00, 0.00, 0.30, 1.00],
    ]
)
MEAN_UNCERTAINTY = np.array([0.02, 0.02, 0.01, 0.005])  # prior sd of each true mean
INFLATION = 0.025
FACILITY_COST_INFLATION = 0.03

CONTRIB = {2027: 300_000.0, 2028: 150_000.0}
PAYMENT, N_PAYMENTS, START_YEAR = 50_000.0, 10, 2033


def mix(us: float, intl: float, bonds: float, bills: float) -> np.ndarray:
    w = np.array([us, intl, bonds, bills], dtype=float)
    return w / w.sum()


# Growth-sleeve allocations (weights for the year starting Jan 1).
SLEEVE_GROWTH = {y: mix(60, 20, 15, 5) for y in range(2027, 2031)}
SLEEVE_GROWTH.update({2031: mix(45, 15, 30, 10), 2032: mix(35, 10, 40, 15)})  # de-risk once the range is quoted

# Classic balanced glide path with no liability hedge (the comparison case).
GLIDE_PATH: dict[int, np.ndarray] = {
    2027: mix(50, 20, 25, 5),
    2028: mix(50, 20, 25, 5),
    2029: mix(48, 19, 28, 5),
    2030: mix(45, 18, 32, 5),
    2031: mix(38, 15, 37, 10),
    2032: mix(30, 12, 43, 15),
}


@dataclass
class Strategy:
    name: str
    hedge_ratio: float                # share of the 10 payments locked in with Treasury zeros in 2027-28
    sleeve: dict[int, np.ndarray]     # allocation of everything not in the ladder


STRATEGIES: dict[str, Strategy] = {
    "No hedge, conservative 40/60": Strategy("No hedge, conservative 40/60", 0.0, {y: mix(28, 12, 50, 10) for y in range(2027, 2033)}),
    "No hedge, balanced glide path": Strategy("No hedge, balanced glide path", 0.0, GLIDE_PATH),
    "No hedge, aggressive 80/20": Strategy("No hedge, aggressive 80/20", 0.0, {y: mix(56, 24, 17, 3) for y in range(2027, 2033)}),
    "Half hedged + growth sleeve": Strategy("Half hedged + growth sleeve", 0.5, SLEEVE_GROWTH),
    "Fully hedged + growth sleeve (recommended)": Strategy("Fully hedged + growth sleeve (recommended)", 1.0, SLEEVE_GROWTH),
}
RECOMMENDED = "Fully hedged + growth sleeve (recommended)"


@dataclass
class SimConfig:
    n_paths: int = 50_000
    seed: int = 2026
    df: float = 5.0
    yield_2027_mean: float = 0.0525  # zero-coupon Treasury yield when the ladder is bought
    yield_2027_sd: float = 0.0075
    yield_step_sd: float = 0.0075    # change in that yield from one year to the next
    yield_2033_mean: float = 0.045   # yield at which any unhedged reserve is bought in 2033
    yield_2033_sd: float = 0.0125
    flexibility_share: float = 0.10  # keep this share of the 2033 portfolio uncommitted
    flexibility_floor: float = 50_000.0
    shocks: dict = field(default_factory=dict)  # {year: array of asset returns} forced on every path


def simulate_returns(cfg: SimConfig, years: list[int]) -> np.ndarray:
    """Array (paths, years, assets) of annual returns."""
    rng = np.random.default_rng(cfg.seed)
    cov = np.outer(CMA_VOL, CMA_VOL) * CMA_CORR
    chol = np.linalg.cholesky(cov)
    n, t, k = cfg.n_paths, len(years), len(ASSETS)
    true_mean = CMA_MEAN + rng.standard_normal((n, 1, k)) * MEAN_UNCERTAINTY
    z = rng.standard_normal((n, t, k)) @ chol.T
    chi = rng.chisquare(cfg.df, size=(n, t, 1)) / cfg.df
    shocks = z / np.sqrt(chi) * np.sqrt((cfg.df - 2) / cfg.df)  # unit-variance t
    r = true_mean + shocks
    for i, y in enumerate(years):
        if y in cfg.shocks:
            r[:, i, :] = cfg.shocks[y]
    return np.maximum(r, -0.95)


def payment_pv(yields: np.ndarray, years_until_first: int) -> np.ndarray:
    """Cost of zero-coupon Treasuries paying $50k at the start of each of 2033-2042."""
    k = np.arange(N_PAYMENTS) + years_until_first
    return (PAYMENT / (1 + yields[:, None]) ** k).sum(axis=1)


def ladder_cost(yields: np.ndarray) -> np.ndarray:
    return payment_pv(yields, 0)


def simulate_plan(strategy: Strategy | str = RECOMMENDED, cfg: SimConfig | None = None) -> pd.DataFrame:
    """Simulate one strategy from 2027 to the 2033 set-aside.

    The ladder (if any) is bought with the 2027 money first, topped up in 2028
    if needed, and is held to maturity: it pays exactly its share of the ten
    payments whatever happens to rates or markets. Everything else is the
    sleeve. In 2033 the sleeve buys whatever part of the reserve is not yet
    hedged, keeps a flexibility buffer, and the rest is the facility contribution.
    """
    if isinstance(strategy, str):
        strategy = STRATEGIES[strategy]
    cfg = cfg or SimConfig()
    n = cfg.n_paths
    rng = np.random.default_rng(cfg.seed + 1)
    y27 = np.clip(rng.normal(cfg.yield_2027_mean, cfg.yield_2027_sd, n), 0.005, None)
    y28 = np.clip(y27 + rng.normal(0, cfg.yield_step_sd, n), 0.005, None)
    y33 = np.clip(rng.normal(cfg.yield_2033_mean, cfg.yield_2033_sd, n), 0.005, None)

    h = strategy.hedge_ratio
    # 2027: hedge as much of the target as the $300k allows; 2028: top up.
    want27 = h * payment_pv(y27, 6)
    buy27 = np.minimum(want27, CONTRIB[2027])
    hedged = buy27 / payment_pv(y27, 6)            # fraction of payments locked in
    need28 = np.clip(h - hedged, 0, None) * payment_pv(y28, 5)
    buy28 = np.minimum(need28, CONTRIB[2028])
    hedged = hedged + buy28 / payment_pv(y28, 5)

    years = list(range(2027, 2033))
    r = simulate_returns(cfg, years)
    sleeve = np.zeros(n)
    sleeve2031 = None
    for i, y in enumerate(years):
        if y == 2031:
            sleeve2031 = sleeve.copy()
        flow = {2027: CONTRIB[2027] - buy27, 2028: CONTRIB[2028] - buy28}.get(y, 0.0)
        sleeve = (sleeve + flow) * (1 + r[:, i, :] @ strategy.sleeve[y])

    full_reserve = ladder_cost(y33)                 # market cost of all ten payments in 2033
    ladder_mv = hedged * full_reserve               # what the hedge is worth in 2033
    residual = (1 - hedged) * full_reserve          # reserve still to buy from the sleeve
    total = sleeve + ladder_mv
    buffer_target = np.maximum(cfg.flexibility_share * total, cfg.flexibility_floor)
    after_reserve = sleeve - residual
    contribution = np.clip(after_reserve - buffer_target, 0, None)
    return pd.DataFrame(
        {
            "sleeve2031": sleeve2031,
            "sleeve2033": sleeve,
            "v2033": total,
            "hedged_fraction": hedged,
            "yield_2027": y27,
            "yield_2033": y33,
            "reserve": full_reserve,
            "residual_reserve": residual,
            "reserve_funded": after_reserve >= -1e-6,
            "buffer": np.minimum(buffer_target, np.clip(after_reserve, 0, None)),
            "contribution": contribution,
            "sleeve_growth_31_33": np.where(sleeve2031 > 0, sleeve / np.maximum(sleeve2031, 1e-9), 1.0),
        }
    )


def summarize_plan(df: pd.DataFrame) -> dict:
    q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    return {
        "v2033": df["v2033"].quantile(q).to_dict(),
        "reserve": df["reserve"].quantile(q).to_dict(),
        "contribution": df["contribution"].quantile(q).to_dict(),
        "p_reserve_funded": float(df["reserve_funded"].mean()),
        "p_contribution_zero": float((df["contribution"] <= 0).mean()),
        "contribution_mean": float(df["contribution"].mean()),
    }


# --------------------------------------------------------------------------- #
# Operating reserve: how much certainty does each reserve design buy?
# --------------------------------------------------------------------------- #
def reserve_success(
    reserve_amount: float, weights: np.ndarray, ladder_years: int, cfg: SimConfig | None = None
) -> float:
    """P(all ten $50k payments are met) for a reserve of `reserve_amount` in 2033.

    The first `ladder_years` payments are pre-funded with Treasury zeros (riskless
    if held to maturity); the rest of the reserve is invested in `weights` and
    pays the later payments as they come due.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed + 7)
    y = np.clip(rng.normal(cfg.yield_2033_mean, cfg.yield_2033_sd, cfg.n_paths), 0.005, None)
    k = np.arange(ladder_years)
    ladder = (PAYMENT / (1 + y[:, None]) ** k).sum(axis=1) if ladder_years else np.zeros(cfg.n_paths)
    invested = reserve_amount - ladder
    r = simulate_returns(SimConfig(n_paths=cfg.n_paths, seed=cfg.seed + 9, df=cfg.df), list(range(2033, 2043)))
    ok = invested >= 0
    v = invested.copy()
    for i in range(N_PAYMENTS):
        if i >= ladder_years:
            v = v - PAYMENT
            ok &= v >= -1e-6
        v = np.maximum(v, 0) * (1 + r[:, i, :] @ weights)
    return float(ok.mean())


def required_reserve(weights: np.ndarray, ladder_years: int, target: float, cfg: SimConfig | None = None) -> float:
    """Smallest reserve (to $1k) that meets every payment with probability >= target."""
    lo, hi = 0.0, 1_000_000.0
    while hi - lo > 1_000:
        mid = (lo + hi) / 2
        if reserve_success(mid, weights, ladder_years, cfg) >= target:
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------------------- #
# 2031 co-sponsor range
# --------------------------------------------------------------------------- #
def cosponsor_range(df: pd.DataFrame, sleeve2031: float, lo_q: float = 0.10, hi_q: float = 0.90, cfg: SimConfig | None = None):
    """Range Laura can quote in 2031 given the growth-sleeve value she observes then.

    Applies every simulated 2031->2033 sleeve growth factor, 2033 reserve cost
    and buffer rule to the observed 2031 value; the range is the [lo_q, hi_q]
    quantiles of the resulting contribution.
    """
    cfg = cfg or SimConfig()
    s33 = sleeve2031 * df["sleeve_growth_31_33"].values
    ladder_mv = df["hedged_fraction"].values * df["reserve"].values
    buffer = np.maximum(cfg.flexibility_share * (s33 + ladder_mv), cfg.flexibility_floor)
    c = np.clip(s33 - df["residual_reserve"].values - buffer, 0, None)
    return float(np.quantile(c, lo_q)), float(np.quantile(c, 0.5)), float(np.quantile(c, hi_q))


def range_coverage(df: pd.DataFrame, lo_q: float = 0.10, hi_q: float = 0.90, n_check: int = 2000, seed: int = 3) -> dict:
    """Out-of-sample check of the range rule: how often does the realized 2033
    contribution land inside / below / above the range quoted in 2031?"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), n_check, replace=False)
    half = len(df) // 2  # build ranges from one half of paths, test on the other
    ref = df.iloc[:half]
    inside = below = above = 0
    for i in idx:
        if i < half:
            i += half
        lo, _, hi = cosponsor_range(ref, df["sleeve2031"].iat[i], lo_q, hi_q)
        c = df["contribution"].iat[i]
        if c < lo - 1:
            below += 1
        elif c > hi + 1:
            above += 1
        else:
            inside += 1
    return {"inside": inside / n_check, "below": below / n_check, "above": above / n_check}


STRESS_SCENARIOS: dict[str, dict[int, np.ndarray]] = {
    "2008-style crash in 2032": {2032: np.array([-0.37, -0.43, 0.05, 0.015])},
    "2022-style stock-and-bond selloff in 2032": {2032: np.array([-0.19, -0.16, -0.13, 0.015])},
    "Lost half-decade (flat equities 2027-2031)": {y: np.array([0.0, 0.0, 0.03, 0.03]) for y in range(2027, 2032)},
}


def stress_test(strategy: str = RECOMMENDED, cfg: SimConfig | None = None) -> pd.DataFrame:
    base = cfg or SimConfig()
    scenarios = {"Base case": SimConfig(**{**base.__dict__, "n_paths": 20_000})}
    for name, shocks in STRESS_SCENARIOS.items():
        scenarios[name] = SimConfig(**{**base.__dict__, "shocks": shocks, "n_paths": 20_000})
    scenarios["Rates fall to 3.5% before the 2027 purchase"] = SimConfig(
        **{**base.__dict__, "yield_2027_mean": 0.035, "yield_2027_sd": 0.003, "n_paths": 20_000}
    )
    scenarios["Rates fall to 2% by 2033"] = SimConfig(
        **{**base.__dict__, "yield_2033_mean": 0.02, "yield_2033_sd": 0.003, "n_paths": 20_000}
    )
    rows = []
    for name, c in scenarios.items():
        df = simulate_plan(strategy, c)
        rows.append(
            {
                "scenario": name,
                "median V2033": df["v2033"].median(),
                "P(reserve funded)": df["reserve_funded"].mean(),
                "median contribution": df["contribution"].median(),
                "10th pct contribution": df["contribution"].quantile(0.10),
            }
        )
    return pd.DataFrame(rows)

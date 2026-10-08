import numpy as np
import pandas as pd
import pytest

from screener import laura_plan as LP
from screener import stats as S
from screener.engine import bayesian_alpha
from screener.features import Panel, empirical_bayes_drift, kalman_trend_t, month_ends
from screener.methods import Context, cap_weights, hmm_filter, training_set

rng = np.random.default_rng(0)


def test_capm_recovers_alpha_and_beta():
    n = 5000
    mkt = pd.Series(rng.normal(0.0004, 0.01, n))
    r = 0.0002 + 1.3 * mkt + pd.Series(rng.normal(0, 0.002, n))  # alpha SE ~0.7%/yr
    res = S.capm_regression(r, mkt)
    assert res.beta == pytest.approx(1.3, abs=0.03)
    assert res.alpha_ann == pytest.approx(0.0002 * 252, abs=0.02)
    assert res.alpha_t > 2


def test_max_drawdown_and_sharpe():
    r = pd.Series([0.1, -0.5, 0.2])
    assert S.max_drawdown(r) == pytest.approx(-0.5)
    assert S.sharpe(pd.Series(rng.normal(0.001, 0.01, 2000))) > 0


def test_probabilistic_sharpe_and_bootstrap():
    good = pd.Series(rng.normal(0.01, 0.02, 300))
    noise = pd.Series(rng.normal(0.0, 0.02, 300))
    assert S.probabilistic_sharpe(good) > 0.99
    p_good, ci = S.stationary_bootstrap_pvalue(good, n_boot=500)
    assert p_good < 0.01 and ci[0] > 0
    p_noise, _ = S.stationary_bootstrap_pvalue(noise - noise.mean(), n_boot=500)
    assert p_noise > 0.2


def test_bayesian_alpha_shrinks_noisy_estimates():
    weak = bayesian_alpha(0.10, 1.0)    # 10%/yr alpha but t = 1: very noisy
    strong = bayesian_alpha(0.03, 6.0)  # small but precise
    assert 0 < weak["post_mean"] < 0.10
    assert strong["post_mean"] == pytest.approx(0.03, abs=0.005)
    assert strong["bf10"] > weak["bf10"]


def test_kalman_detects_drift():
    t = 252
    up = np.cumsum(rng.normal(0.003, 0.01, t))
    flat = np.cumsum(rng.normal(0.0, 0.01, t))
    down = np.cumsum(rng.normal(-0.003, 0.01, t))
    stat = kalman_trend_t(np.column_stack([up, flat, down]))
    assert stat[0] > stat[1] > stat[2]


def test_empirical_bayes_shrinkage_is_between_prior_and_sample():
    daily = pd.DataFrame(rng.normal(0.0005, 0.02, (252, 30)))
    eb = empirical_bayes_drift(daily)
    m0, m = daily.mean().mean(), daily.mean()
    lo, hi = np.minimum(m0, m), np.maximum(m0, m)
    assert ((eb["eb_drift"] >= lo - 1e-12) & (eb["eb_drift"] <= hi + 1e-12)).all()
    assert eb["eb_shrink"].between(0, 1).all()


def test_hmm_flags_high_vol_regime():
    calm = rng.normal(0.0005, 0.006, 600)
    stormy = rng.normal(-0.001, 0.03, 60)
    assert hmm_filter(np.concatenate([calm, stormy])) > 0.9
    assert hmm_filter(np.concatenate([stormy, calm])) < 0.1


def test_cap_weights():
    w = cap_weights(pd.Series([10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    assert w.sum() == pytest.approx(1)
    assert w.max() <= 0.10 + 1e-9


def test_training_set_has_no_lookahead():
    dates = list(pd.date_range("2020-01-31", periods=6, freq="ME"))
    feats = {d: pd.DataFrame({"z_a": [1.0, -1.0]}, index=["X", "Y"]) for d in dates}
    fwd = {d: pd.Series([0.01, -0.01], index=["X", "Y"]) for d in dates[:-1]}
    ctx = Context(Panel(dates, feats, fwd), pd.DataFrame(), "SPY", "BIL")
    _, _, months = training_set(ctx, dates[3], ["z_a"])
    # At dates[3] only months whose next month-end is <= dates[3] are usable.
    assert months == dates[:3]


def test_month_ends():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    me = month_ends(idx)
    assert list(me.strftime("%m-%d")) == ["01-31", "02-29", "03-29"]


def test_ladder_cost():
    assert LP.ladder_cost(np.array([0.0]))[0] == pytest.approx(500_000)
    assert LP.ladder_cost(np.array([0.05]))[0] < 500_000


def test_fully_hedged_plan_always_funds_reserve():
    cfg = LP.SimConfig(n_paths=5000)
    df = LP.simulate_plan(LP.RECOMMENDED, cfg)
    assert df["reserve_funded"].all()
    assert df["hedged_fraction"].min() == pytest.approx(1.0)
    assert (df["contribution"] >= 0).all()


def test_cosponsor_range_increases_with_wealth():
    df = LP.simulate_plan(LP.RECOMMENDED, LP.SimConfig(n_paths=5000))
    lo1, mid1, hi1 = LP.cosponsor_range(df, 120_000)
    lo2, mid2, hi2 = LP.cosponsor_range(df, 220_000)
    assert lo1 <= mid1 <= hi1 and lo2 > lo1 and hi2 > hi1

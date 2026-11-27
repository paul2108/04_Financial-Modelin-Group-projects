"""Unit tests for the strategy core (app/strategy_core.py, generated from the notebook).

All tests run on synthetic data, so they need no internet connection.
Run with:  make test   (or:  .venv/bin/python -m pytest -m "not network")
"""

import numpy as np
import pandas as pd
import pytest

import strategy_core as sc

RISKY = ["AAA", "BBB", "CCC", "DDD"]
OTHERS = ["BND", "MKT"]
RF = "^IRX"


# --------------------------------------------------------------------------
# Synthetic market: business days 2005-2016, asset DDD starts one year later
# --------------------------------------------------------------------------
def make_raw_prices(seed: int = 7, end: str = "2016-12-30") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", end)
    n = len(idx)
    vols = np.array([0.18, 0.12, 0.06, 0.25, 0.05, 0.16]) / np.sqrt(252)
    drifts = np.array([0.07, 0.05, 0.03, 0.04, 0.03, 0.07]) / 252
    corr = np.full((6, 6), 0.3) + 0.7 * np.eye(6)
    chol = np.linalg.cholesky(corr)
    shocks = rng.standard_normal((n, 6)) @ chol.T
    rets = drifts + vols * shocks
    prices = 100 * np.cumprod(1 + rets, axis=0)
    df = pd.DataFrame(prices, index=idx, columns=RISKY + OTHERS)
    df.loc[:"2006-02-02", "DDD"] = np.nan  # late inception, like DBC
    df[RF] = 2.0 + 0.5 * np.sin(np.arange(n) / 300)  # T-bill yield in percent
    df.loc[df.index[::97], RF] = np.nan  # a few missing prints
    return df


@pytest.fixture(scope="module")
def data():
    return sc.build_market_data(make_raw_prices(), risky=RISKY, others=OTHERS, rf_ticker=RF)


def random_cov(rng, n=6):
    """Random positive-definite covariance matrix (sample covariance of 3n draws, rescaled)."""
    a = rng.standard_normal((n, 3 * n))
    corr = np.corrcoef(a)  # rows = variables, 3n observations -> full rank
    vols = rng.uniform(0.03, 0.30, n)
    return np.outer(vols, vols) * corr


# --------------------------------------------------------------------------
# Equal risk contribution (ERC) solved with fsolve
# --------------------------------------------------------------------------
def test_erc_diagonal_cov_equals_inverse_vol():
    vols = np.array([0.1, 0.2, 0.4])
    w, ok = sc.erc_weights(np.diag(vols**2))
    assert ok
    expected = (1 / vols) / (1 / vols).sum()
    np.testing.assert_allclose(w, expected, atol=1e-10)


def test_erc_equal_vols_equal_corr_is_one_over_n():
    n = 5
    cov = 0.04 * (np.full((n, n), 0.4) + 0.6 * np.eye(n))
    w, ok = sc.erc_weights(cov)
    assert ok
    np.testing.assert_allclose(w, np.ones(n) / n, atol=1e-10)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.7])
def test_erc_two_assets_closed_form(rho):
    s1, s2 = 0.08, 0.20
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w, ok = sc.erc_weights(cov)
    assert ok
    assert w[0] == pytest.approx(s2 / (s1 + s2), abs=1e-10)


def test_erc_scale_invariant():
    cov = random_cov(np.random.default_rng(1))
    w1, _ = sc.erc_weights(cov)
    w2, _ = sc.erc_weights(1000 * cov)
    np.testing.assert_allclose(w1, w2, atol=1e-10)


def test_erc_random_covariances_have_equal_risk_contributions():
    rng = np.random.default_rng(42)
    for _ in range(50):
        cov = random_cov(rng)
        w, ok = sc.erc_weights(cov)
        assert ok
        assert np.all(w > 0)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        rc = sc.risk_contributions(w, cov)
        np.testing.assert_allclose(rc, np.full(len(w), 1 / len(w)), atol=1e-8)


def test_inverse_vol_weights_sum_to_one():
    vols = np.array([0.1, 0.2, 0.4])
    w = sc.inverse_vol_weights(np.diag(vols**2))
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]


# --------------------------------------------------------------------------
# Data transformation
# --------------------------------------------------------------------------
def test_month_end_index_and_returns(data):
    assert data.prices_m.index.is_month_end.all()
    assert "^IRX" not in data.rets_m.columns
    # monthly simple return equals ratio of month-end prices
    p = data.prices_m["AAA"]
    np.testing.assert_allclose(data.rets_m["AAA"].iloc[1:], (p / p.shift(1) - 1).iloc[1:])


def test_rf_is_lagged_irx(data):
    # rf earned in month t+1 is the IRX yield observed at the end of month t, / 1200
    irx_m = make_raw_prices()[RF].ffill(limit=5).resample("ME").last()
    t = data.rf_m.index[10]
    prev = data.rf_m.index[9]
    assert data.rf_m.loc[t] == pytest.approx(irx_m.loc[prev] / 1200)


# --------------------------------------------------------------------------
# Trend signal and timing
# --------------------------------------------------------------------------
def test_trend_signal_uses_ten_month_sma_including_current_month():
    idx = pd.date_range("2010-01-31", periods=12, freq="ME")
    p = pd.DataFrame({"X": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 5, 30]}, index=idx, dtype=float)
    sig = sc.trend_signal(p, window=10)
    assert sig["X"].iloc[:9].isna().all()  # SMA not available yet
    assert bool(sig["X"].iloc[9]) is True  # 19 > mean(10..19)=14.5
    assert bool(sig["X"].iloc[10]) is False  # 5 < mean(11..19,5)
    assert bool(sig["X"].iloc[11]) is True


def test_first_decision_date_waits_for_late_asset(data):
    t0 = sc.first_decision_date(data, RISKY, cov_window=252, sma_window=10)
    # DDD starts 2006-02-03: the decision waits until 252 daily returns of DDD exist
    first_full_window = make_raw_prices()["DDD"].dropna().index[252]
    assert t0 == (first_full_window + pd.offsets.MonthEnd(0))
    assert t0 > pd.Timestamp("2006-11-30")  # later than the first 10-month SMA
    t0_without = sc.first_decision_date(data, ["AAA", "BBB", "CCC"], 252, 10)
    assert t0_without < t0


def test_target_weights_are_valid(data):
    dates = data.prices_m.loc["2007-02-28":].index
    W = sc.target_weights(data, RISKY, dates, cov_window=252, sma_window=10, use_trend=True)
    assert list(W.columns) == RISKY + ["CASH"]
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-12)
    assert (W >= -1e-12).all().all()
    assert W.attrs["fsolve_fallbacks"] == 0
    # trend off for an asset -> its weight is exactly zero
    sig = sc.trend_signal(data.prices_m[RISKY], 10).loc[dates]
    off = sig == 0.0
    assert off.to_numpy().any()  # the filter is actually switched off sometimes
    assert (W[RISKY].to_numpy()[off.to_numpy()] == 0.0).all()


def test_no_trend_means_fully_invested(data):
    dates = data.prices_m.loc["2007-02-28":].index
    W = sc.target_weights(data, RISKY, dates, 252, 10, use_trend=False)
    np.testing.assert_allclose(W["CASH"], 0.0, atol=1e-12)


def test_truncation_no_look_ahead():
    """Weights up to 2012-12 must not change when later data is added."""
    full = sc.build_market_data(make_raw_prices(end="2016-12-30"), RISKY, OTHERS, RF)
    short = sc.build_market_data(make_raw_prices(end="2012-12-31"), RISKY, OTHERS, RF)
    dates_s = short.prices_m.loc["2007-02-28":"2012-12-31"].index
    W_full = sc.target_weights(full, RISKY, dates_s, 252, 10, True)
    W_short = sc.target_weights(short, RISKY, dates_s, 252, 10, True)
    pd.testing.assert_frame_equal(W_full, W_short)


# --------------------------------------------------------------------------
# Backtest engine identities
# --------------------------------------------------------------------------
def test_constant_weights_without_costs_equal_weighted_sum(data):
    dates = data.prices_m.loc["2007-02-28":].index
    W = sc.constant_weights({"AAA": 0.6, "BND": 0.4}, dates)
    res = sc.backtest(W, data, cost_bps=0.0)
    expected = 0.6 * data.rets_m["AAA"] + 0.4 * data.rets_m["BND"]
    np.testing.assert_allclose(res["gross"], expected.loc[res.index], atol=1e-14)
    np.testing.assert_allclose(res["net"], res["gross"], atol=1e-14)
    assert res.index[0] == pd.Timestamp("2007-03-31")  # decide at t, earn t+1


def test_cash_earns_risk_free_rate(data):
    dates = data.prices_m.loc["2007-02-28":].index
    W = sc.constant_weights({"CASH": 1.0}, dates)
    res = sc.backtest(W, data, cost_bps=25.0)
    np.testing.assert_allclose(res["net"], data.rf_m.loc[res.index], atol=1e-14)
    assert res["turnover"].sum() == 0.0


def test_costs_and_nav_identity(data):
    dates = data.prices_m.loc["2007-02-28":].index
    W = sc.target_weights(data, RISKY, dates, 252, 10, True)
    res = sc.backtest(W, data, cost_bps=10.0)
    # first rebalance buys the whole risky book out of cash
    assert res["turnover"].iloc[0] == pytest.approx(W[RISKY].iloc[0].sum())
    np.testing.assert_allclose(res["cost"], 10.0 / 1e4 * res["turnover"])
    np.testing.assert_allclose(1 + res["net"], (1 - res["cost"]) * (1 + res["gross"]))
    nav = sc.nav_from_returns(res["net"])
    assert nav.iloc[0] == 1.0
    assert nav.index[0] == pd.Timestamp("2007-02-28")
    assert nav.iloc[-1] == pytest.approx(np.prod(1 + res["net"]))


def test_drifted_turnover_matches_hand_calculation():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"X": [100.0, 110.0, 110.0], "Y": [100.0, 100.0, 100.0]}, index=idx)
    md = sc.MarketData(
        prices_d=prices,
        rets_d=prices.pct_change(fill_method=None),
        prices_m=prices,
        rets_m=prices.pct_change(fill_method=None),
        rf_m=pd.Series(0.0, index=idx),
    )
    W = sc.constant_weights({"X": 0.5, "Y": 0.5}, idx)
    res = sc.backtest(W, md, cost_bps=0.0)
    # after +10% on X the drifted weights are 0.5238/0.4762 -> turnover 2*0.0238
    drift_x = 0.5 * 1.1 / (0.5 * 1.1 + 0.5)
    assert res["turnover"].iloc[1] == pytest.approx(2 * (drift_x - 0.5))


# --------------------------------------------------------------------------
# Performance measures
# --------------------------------------------------------------------------
def test_performance_summary_matches_hand_calculation():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    mkt = pd.Series(rng.normal(0.008, 0.045, 120), index=idx)
    rf = pd.Series(0.001, index=idx)
    r = 0.002 + 0.4 * mkt + pd.Series(rng.normal(0, 0.01, 120), index=idx)
    bench = 0.6 * mkt + 0.4 * 0.003
    s = sc.performance_summary(r, rf, benchmark=bench, market=mkt)
    ex, mex = r - rf, mkt - rf
    beta = np.cov(ex, mex, ddof=1)[0, 1] / np.var(mex, ddof=1)
    assert s["ann_mean"] == pytest.approx(12 * r.mean())
    assert s["ann_vol"] == pytest.approx(np.sqrt(12) * r.std(ddof=1))
    assert s["sharpe"] == pytest.approx(12 * ex.mean() / (np.sqrt(12) * ex.std(ddof=1)))
    assert s["beta_mkt"] == pytest.approx(beta)
    assert s["alpha_mkt"] == pytest.approx(12 * (ex.mean() - beta * mex.mean()))
    active = r - bench
    assert s["tracking_error"] == pytest.approx(np.sqrt(12) * active.std(ddof=1))
    assert s["info_ratio"] == pytest.approx(12 * active.mean() / (np.sqrt(12) * active.std(ddof=1)))
    assert s["cagr"] == pytest.approx(np.prod(1 + r) ** (12 / 120) - 1)
    assert np.isfinite(s["alpha_mkt_t"])


def test_max_drawdown_counts_from_initial_value():
    r = pd.Series([-0.10, 0.05, -0.20, 0.30], index=pd.date_range("2020-01-31", periods=4, freq="ME"))
    nav = np.array([1.0, 0.9, 0.945, 0.756, 0.9828])
    expected = (nav / np.maximum.accumulate(nav) - 1).min()
    assert sc.max_drawdown(r) == pytest.approx(expected)


def test_breakeven_cost_solves_root():
    # Sharpe difference that falls linearly with costs: root at 37.5 bps
    c_star, ok = sc.breakeven_cost(lambda c: 0.3 - 0.008 * c, x0=10.0)
    assert ok
    assert c_star == pytest.approx(37.5)


def test_breakeven_cost_reports_no_root_when_never_better():
    c_star, ok = sc.breakeven_cost(lambda c: -0.1 - 0.008 * c, x0=10.0)
    assert not ok
    assert np.isnan(c_star)


# --------------------------------------------------------------------------
# Convenience wrapper used by notebook and app
# --------------------------------------------------------------------------
def test_run_strategy_shares_sample_window(data):
    base = sc.run_strategy(data, assets=RISKY, benchmark_weights={"AAA": 0.6, "BND": 0.4})
    loo = sc.run_strategy(
        data,
        assets=["AAA", "BBB", "CCC"],
        start_decision=base["start_decision"],
        benchmark_weights={"AAA": 0.6, "BND": 0.4},
    )
    assert loo["returns"].index.equals(base["returns"].index)
    assert base["benchmark"].index.equals(base["returns"].index)

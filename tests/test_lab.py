"""Tests for the "Machine learning & robustness lab" building blocks (notebook Sections 9.6-9.8, exported to
app/strategy_core.py) and for the app logic that uses them.

Synthetic data only, no internet. Key properties:
* HRP gives inverse-variance weights when assets are uncorrelated and for two assets;
* the averaged trend signal lies in [0, 1] and equals the plain signal for a single window;
* target_weights_x and run_strategy reproduce the prospectus engine with default arguments;
* exante_vol recomputes the stored ex-ante volatility; leverage 1 reproduces the unlevered returns;
* the promotion gates and their explanation match a hand calculation.
"""

import numpy as np
import pandas as pd
import pytest

import strategy_core as sc
from test_strategy_core import OTHERS, RF, RISKY, make_raw_prices


@pytest.fixture(scope="module")
def data():
    return sc.build_market_data(make_raw_prices(), risky=RISKY, others=OTHERS, rf_ticker=RF)


@pytest.fixture(scope="module")
def dates(data):
    return data.prices_m.loc["2007-02-28":].index


# ---------------------------------------------------------------- Hierarchical Risk Parity
def test_hrp_two_assets_is_inverse_variance():
    cov = np.array([[0.04, 0.01], [0.01, 0.01]])
    w = sc.hrp_weights(cov)
    inv = 1.0 / np.diag(cov)
    np.testing.assert_allclose(w, inv / inv.sum(), atol=1e-12)


def test_hrp_uncorrelated_assets_is_inverse_variance():
    var = np.array([0.04, 0.01, 0.09, 0.0225, 0.0016])
    w = sc.hrp_weights(np.diag(var))
    np.testing.assert_allclose(w, (1 / var) / (1 / var).sum(), atol=1e-12)


def test_hrp_weights_are_positive_and_sum_to_one():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 30))
    cov = np.cov(a) + 1e-3 * np.eye(6)
    w = sc.hrp_weights(cov)
    assert np.all(w > 0) and w.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- averaged trend signal
def test_averaged_signal_with_one_window_equals_plain_signal(data):
    pm = data.prices_m[RISKY]
    pd.testing.assert_frame_equal(sc.trend_signal_avg(pm, (10,)), sc.trend_signal(pm, 10), check_dtype=False)


def test_averaged_signal_is_a_share(data):
    sig = sc.trend_signal_avg(data.prices_m[RISKY]).dropna()
    assert len(sig) > 0
    assert ((sig >= 0) & (sig <= 1)).all().all()
    allowed = {k / len(sc.AVG_WINDOWS) for k in range(len(sc.AVG_WINDOWS) + 1)}
    assert set(np.round(np.unique(sig.to_numpy()), 12)) <= {round(x, 12) for x in allowed}


# ---------------------------------------------------------------- covariance and weights
def test_ledoit_wolf_close_to_sample_for_long_samples():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, 4)) * [0.01, 0.02, 0.005, 0.015]
    np.testing.assert_allclose(sc.covariance(X, "ledoit_wolf"), sc.covariance(X), rtol=0.05, atol=1e-6)
    with pytest.raises(ValueError):
        sc.covariance(X, "unknown")


def test_target_weights_x_defaults_reproduce_target_weights(data, dates):
    W = sc.target_weights(data, RISKY, dates)
    Wx = sc.target_weights_x(data, RISKY, dates)
    np.testing.assert_allclose(Wx.to_numpy(), W.to_numpy(), atol=1e-10)
    assert Wx.attrs["fsolve_fallbacks"] == W.attrs["fsolve_fallbacks"]


@pytest.mark.parametrize(
    "kw",
    [
        {"weighting": "hrp"},
        {"trend_mode": "average"},
        {"cov_estimator": "ledoit_wolf"},
        {"weighting": "hrp", "use_trend": False},
    ],
)
def test_lab_weights_are_valid(data, dates, kw):
    W = sc.target_weights_x(data, RISKY, dates, **kw)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert (W.to_numpy() >= -1e-12).all()


def test_run_strategy_routes_lab_options(data, dates):
    base = sc.run_strategy(data, RISKY, start_decision=dates[0])
    plain = sc.run_strategy(data, RISKY, start_decision=dates[0], trend_mode="single", cov_estimator="sample")
    pd.testing.assert_series_equal(base["returns"], plain["returns"])
    hrp = sc.run_strategy(data, RISKY, weighting="hrp", start_decision=dates[0])
    Wx = sc.target_weights_x(data, RISKY, dates, weighting="hrp")
    np.testing.assert_allclose(hrp["weights"].to_numpy(), Wx.to_numpy(), atol=1e-12)


# ---------------------------------------------------------------- leverage
def test_exante_vol_matches_stored_value(data, dates):
    for est in ["sample", "ledoit_wolf"]:
        W = sc.target_weights_x(data, RISKY, dates, cov_estimator=est)
        np.testing.assert_allclose(sc.exante_vol(W, data, cov_estimator=est), W.attrs["exante_vol"], atol=1e-12)


def test_leverage_one_reproduces_unlevered_returns(data, dates):
    W = sc.target_weights_x(data, RISKY, dates)
    r = sc.backtest(W, data)["net"]
    target = 0.08
    r_L, L = sc.levered_returns(W, r, data.rf_m, target, vol=pd.Series(target, index=W.index))
    assert np.allclose(L, 1.0)
    np.testing.assert_allclose(r_L, r, atol=1e-14)


def test_leverage_is_capped_and_borrowing_costs_reduce_returns(data, dates):
    W = sc.target_weights_x(data, RISKY, dates)
    r = sc.backtest(W, data)["net"]
    r_free, L = sc.levered_returns(W, r, data.rf_m, 0.5, cap=2.0, spread=0.0)
    r_paid, _ = sc.levered_returns(W, r, data.rf_m, 0.5, cap=2.0, spread=0.02)
    assert L.max() <= 2.0 + 1e-12
    assert (r_paid <= r_free + 1e-15).all() and (r_paid < r_free).any()


# ---------------------------------------------------------------- promotion gates
def test_gate_verdict_hand_calculation():
    idx = pd.date_range("2007-01-31", "2023-12-31", freq="ME")
    rng = np.random.default_rng(1)
    r_ref = pd.Series(0.004 + 0.02 * rng.standard_normal(len(idx)), index=idx)
    rf = pd.Series(0.001, index=idx)
    periods = dict(sc.STRESS_PERIODS)
    better = r_ref + 0.002  # strictly better every month: passes everything
    g = sc.gate_verdict(better, r_ref, rf, periods, 2)
    assert g["PROMOTED"] and g["stress wins"] == "3 of 3"
    assert sc.gate_reason(g, 2) == "passes all gates"
    worse = r_ref * 1.5  # same sign, more risk: deeper drawdown
    g = sc.gate_verdict(worse, r_ref, rf, periods, 2)
    assert not g["MDD not deeper"] and not g["PROMOTED"]
    assert "deeper drawdown than A" in sc.gate_reason(g, 2)


def test_gate_verdict_without_stress_periods_is_sharpe_and_drawdown_only():
    idx = pd.date_range("2010-01-31", periods=60, freq="ME")
    r_ref = pd.Series(np.tile([0.01, -0.005], 30), index=idx)
    rf = pd.Series(0.0, index=idx)
    g = sc.gate_verdict(r_ref, r_ref, rf, {}, 0)
    assert g["PROMOTED"] and g["stress wins"] == "0 of 0"

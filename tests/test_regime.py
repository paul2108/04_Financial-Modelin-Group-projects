"""Tests for the machine-learning extension (regime detection) in app/strategy_core.py.

Synthetic data only, no internet. The key properties:
* the HMM finds a known turbulent block,
* filtered probabilities never use future data (truncation test),
* results are deterministic for fixed seeds,
* the simple volatility rule and the risk-off overlay behave as specified.
"""

import numpy as np
import pandas as pd
import pytest

import strategy_core as sc


def synthetic_index(seed=3, end="2014-12-31"):
    """Daily index 1995-2014 with two known turbulent blocks (2001-2002 and 2008-2009)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1995-01-02", "2020-12-31")
    vol = np.where(((idx >= "2001-01-01") & (idx <= "2002-12-31"))
                   | ((idx >= "2008-01-01") & (idx <= "2009-12-31")), 0.025, 0.006)
    mu = np.where(vol > 0.01, -0.0008, 0.0004)
    r = mu + vol * rng.standard_normal(len(idx))
    px = pd.Series(100 * np.cumprod(1 + r), index=idx, name="IDX")
    return px.loc[:end]


CFG = sc.HMMConfig(seeds=(0, 1), n_iter=100)


@pytest.fixture(scope="module")
def probs():
    px = synthetic_index()
    feats = sc.regime_features(px, CFG.vol_window)
    dates = pd.date_range("2000-01-31", "2014-12-31", freq="ME")
    return sc.hmm_regime_probs(feats, dates, CFG, progress=False)


def test_features_have_no_missing_values_and_annualised_vol():
    feats = sc.regime_features(synthetic_index(), 21)
    assert list(feats.columns) == ["ret", "vol"]
    assert not feats.isna().any().any()
    assert feats["vol"].median() == pytest.approx(0.006 * np.sqrt(252), rel=0.25)


def test_probabilities_are_valid(probs):
    assert probs.between(0, 1).all()
    assert probs.index.is_month_end.all()


def test_hmm_detects_known_turbulent_blocks(probs):
    turbulent = probs.loc["2008-03-31":"2009-11-30"].mean()
    calm = probs.loc["2004-01-31":"2006-12-31"].mean()
    assert turbulent > 0.8
    assert calm < 0.2


def test_hmm_has_no_look_ahead():
    px = synthetic_index(end="2014-12-31")
    dates = pd.date_range("2000-01-31", "2010-12-31", freq="ME")
    full = sc.hmm_regime_probs(sc.regime_features(px, 21), dates, CFG, progress=False)
    short = sc.hmm_regime_probs(sc.regime_features(px.loc[:"2010-12-31"], 21), dates, CFG, progress=False)
    pd.testing.assert_series_equal(full, short)


def test_hmm_is_deterministic(probs):
    px = synthetic_index()
    again = sc.hmm_regime_probs(sc.regime_features(px, 21), probs.index, CFG, progress=False)
    pd.testing.assert_series_equal(probs, again)


def test_forward_filter_matches_hmmlearn_posterior_at_the_last_observation():
    """Filtered P(state_t | obs_1..t) equals hmmlearn's smoothed posterior of the LAST observation."""
    feats = sc.regime_features(synthetic_index(end="2003-12-31"), 21)
    model, scaler, turb = sc.fit_hmm(feats.to_numpy(), CFG)
    filtered = sc.filtered_state_probs(model, scaler, feats.to_numpy())
    for n in [300, 1500, len(feats)]:
        z = scaler.transform(feats.to_numpy()[:n])
        smoothed_last = model.predict_proba(z)[-1]
        np.testing.assert_allclose(filtered[n - 1], smoothed_last, atol=1e-8)


def test_turbulent_state_is_the_high_volatility_state():
    feats = sc.regime_features(synthetic_index(end="2003-12-31"), 21)
    model, scaler, turb = sc.fit_hmm(feats.to_numpy(), CFG)
    vol_means = scaler.inverse_transform(model.means_)[:, 1]
    assert turb == int(np.argmax(vol_means))


def test_vol_rule_uses_only_the_past():
    px = synthetic_index()
    feats = sc.regime_features(px, 21)
    dates = pd.date_range("2000-01-31", "2010-12-31", freq="ME")
    full = sc.vol_rule_signal(feats, dates, quantile=0.8)
    short = sc.vol_rule_signal(sc.regime_features(px.loc[:"2010-12-31"], 21), dates, quantile=0.8)
    pd.testing.assert_series_equal(full, short)
    # early in a turbulent block volatility is far above its history; once the block lasts long enough it
    # lifts the 80th percentile itself, so only the first months are required to be risk-off
    assert full.loc["2008-02-29":"2008-12-31"].all()
    assert full.loc["2005-01-31":"2006-12-31"].mean() < 0.15   # rare false alarms in calm years


def test_risk_off_overlay_moves_growth_assets_into_cash():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    W = pd.DataFrame({"SPY": 0.2, "IEF": 0.5, "GLD": 0.1, "CASH": 0.2}, index=idx)
    risk_off = pd.Series([False, True, True], index=idx)
    out = sc.apply_risk_off(W, risk_off, scale=0.5, growth_assets=["SPY"])
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert out.loc[idx[0], "SPY"] == 0.2
    assert out.loc[idx[1], "SPY"] == pytest.approx(0.1)
    assert out.loc[idx[1], "CASH"] == pytest.approx(0.3)
    assert out.loc[idx[1], "IEF"] == 0.5  # defensive assets untouched


def test_parallel_walk_forward_matches_serial(probs):
    """n_jobs only changes the speed: yearly fits run in parallel (threads here, so the test runs anywhere)."""
    from joblib import parallel_config
    px = synthetic_index()
    feats = sc.regime_features(px, CFG.vol_window)
    dates = pd.date_range("2000-01-31", "2014-12-31", freq="ME")
    with parallel_config(backend="threading"):
        par = sc.hmm_regime_probs(feats, dates, CFG, progress=False, n_jobs=-1)
    assert par.index.equals(probs.index)
    np.testing.assert_allclose(par.to_numpy(), probs.to_numpy(), atol=1e-9)

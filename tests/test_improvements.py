"""Tests for the building blocks of the improvement tests (notebook Section 9.14, exported to app/strategy_core.py).

Synthetic data only, no internet. Key properties:
* the hysteresis signal equals the plain trend signal for band 0, keeps its state inside the band and uses no
  future prices;
* apply_trend reproduces target_weights with the trend filter;
* the no-trade band reproduces backtest() for band 0, lowers turnover and never borrows;
* the daily engine reproduces the monthly engine when it rebalances at month-ends;
* the deflated Sharpe ratio falls when more variants were tried.
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


# ---------------------------------------------------------------- hysteresis
def test_hysteresis_with_zero_band_equals_plain_trend_signal(data):
    pm = data.prices_m[RISKY]
    plain = sc.trend_signal(pm, 10)
    hyst = sc.trend_signal_hysteresis(pm, 10, 0.0)
    pd.testing.assert_frame_equal(hyst, plain, check_dtype=False)


def test_hysteresis_keeps_state_inside_the_band():
    idx = pd.date_range("2020-01-31", periods=14, freq="ME")
    # flat at 100 for 10 months (SMA = 100), then slightly below, far below, slightly above, far above
    px = pd.DataFrame({"X": [100.0] * 9 + [101.0, 99.5, 90.0, 100.5, 112.0]}, index=idx)
    sma = px["X"].rolling(10).mean()
    sig = sc.trend_signal_hysteresis(px, 10, 0.02)["X"]
    assert sig.iloc[:9].isna().all()
    assert sig.iloc[9] == 1.0  # first month: plain rule, 101 > SMA
    assert px["X"].iloc[10] < sma.iloc[10]  # plain rule would sell here ...
    assert sig.iloc[10] == 1.0  # ... hysteresis keeps the asset (inside the band)
    assert sig.iloc[11] == 0.0  # 90 is far below SMA x 0.98: sell
    assert px["X"].iloc[12] > sma.iloc[12]  # plain rule would buy back here ...
    assert sig.iloc[12] == 0.0  # ... hysteresis waits
    assert sig.iloc[13] == 1.0  # 112 is above SMA x 1.02: buy


def test_hysteresis_has_no_look_ahead(data):
    pm = data.prices_m[RISKY]
    full = sc.trend_signal_hysteresis(pm, 10, 0.02)
    cut = sc.trend_signal_hysteresis(pm.loc[:"2011-06-30"], 10, 0.02)
    pd.testing.assert_frame_equal(full.loc[:"2011-06-30"], cut)


def test_hysteresis_reduces_signal_changes(data):
    pm = data.prices_m[RISKY]
    flips = lambda s: int(s.diff().abs().sum().sum())
    assert flips(sc.trend_signal_hysteresis(pm, 10, 0.03)) <= flips(sc.trend_signal(pm, 10))


# ---------------------------------------------------------------- applying a signal to risk-parity weights
def test_apply_trend_reproduces_target_weights(data, dates):
    W_rp = sc.target_weights(data, RISKY, dates, 252, 10, use_trend=False)
    W = sc.target_weights(data, RISKY, dates, 252, 10, use_trend=True)
    W2 = sc.apply_trend(W_rp, sc.trend_signal(data.prices_m[RISKY], 10))
    np.testing.assert_allclose(W2.to_numpy(), W.to_numpy(), atol=1e-14)
    np.testing.assert_allclose(W2.sum(axis=1), 1.0)


def test_apply_trend_refuses_missing_signal(data, dates):
    W_rp = sc.target_weights(data, RISKY, dates, 252, 10, use_trend=False)
    sig = sc.trend_signal(data.prices_m[RISKY], 10).copy()
    sig.loc[dates[3], "AAA"] = np.nan
    with pytest.raises(ValueError):
        sc.apply_trend(W_rp, sig)


# ---------------------------------------------------------------- no-trade band
def test_band_zero_reproduces_backtest(data, dates):
    W = sc.target_weights(data, RISKY, dates, 252, 10, True)
    a = sc.backtest(W, data, 10.0)
    b = sc.backtest_band(W, data, 10.0, 0.0)
    pd.testing.assert_frame_equal(a, b)


def test_band_lowers_turnover_and_never_borrows(data, dates):
    W = sc.target_weights(data, RISKY, dates, 252, 10, True)
    a = sc.backtest(W, data, 10.0)
    b = sc.backtest_band(W, data, 10.0, 0.03)
    assert b["turnover"].sum() < a["turnover"].sum()
    ex = b.attrs["executed"]
    np.testing.assert_allclose(ex.sum(axis=1), 1.0)
    assert (ex.to_numpy() >= -1e-12).all()
    np.testing.assert_allclose(1 + b["net"], (1 - b["cost"]) * (1 + b["gross"]))


def test_band_always_executes_exits(data, dates):
    W = sc.target_weights(data, RISKY, dates, 252, 10, True)
    ex = sc.backtest_band(W, data, 10.0, 0.05).attrs["executed"]
    target = W.loc[ex.index, RISKY]
    exits = target.to_numpy() == 0.0
    assert exits.any()
    assert (ex[RISKY].to_numpy()[exits] == 0.0).all()  # sold whenever the trend filter says so


def test_band_hand_calculation():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"X": [100.0, 101.0, 101.0], "Y": [100.0, 100.0, 100.0]}, index=idx)
    d = sc.MarketData(
        prices_d=prices,
        rets_d=prices.pct_change(),
        prices_m=prices,
        rets_m=prices.pct_change(),
        rf_m=pd.Series(0.0, index=idx),
    )
    W = pd.DataFrame({"X": [0.5, 0.5], "Y": [0.5, 0.5], "CASH": [0.0, 0.0]}, index=idx[:2])
    res = sc.backtest_band(W, d, 0.0, 0.01)
    # after month 1, X has drifted to 50.5/100.5 = 0.5025: only 0.25 pp away -> not traded
    ex = res.attrs["executed"].iloc[1]
    assert ex["X"] == pytest.approx(50.5 / 100.5)
    assert res["turnover"].iloc[1] == pytest.approx(0.0)


# ---------------------------------------------------------------- daily engine
def test_daily_engine_reproduces_monthly_engine(data, dates):
    W = sc.target_weights(data, RISKY, dates, 252, 10, True)
    # monthly engine on month-end data sampled from the same (forward-filled) daily prices
    px = data.prices_d[RISKY].ffill(limit=5)
    days = pd.Series(px.index, index=px.index)
    picks = pd.DatetimeIndex(days.groupby(px.index.to_period("M")).last().to_numpy())
    prices_m = px.loc[picks]
    rf_m = (data.prices_d[RF].ffill(limit=5).loc[picks] / 1200).shift(1)
    dm = sc.MarketData(
        prices_d=data.prices_d,
        rets_d=data.rets_d,
        prices_m=prices_m,
        rets_m=prices_m.pct_change(fill_method=None),
        rf_m=rf_m,
    )
    Wd = W.set_axis(picks[picks.to_period("M").isin(W.index.to_period("M"))])
    monthly = sc.backtest(Wd, dm, 10.0)["net"]
    nav, to = sc.backtest_daily(Wd, px, rf_m.shift(-1), 10.0)
    sampled = nav.loc[Wd.index]
    daily = (sampled / sampled.shift(1) - 1.0).iloc[1:]
    np.testing.assert_allclose(daily.to_numpy(), monthly.to_numpy(), atol=1e-12)
    assert nav.iloc[0] == 1.0
    assert to.iloc[0] == pytest.approx(Wd[RISKY].iloc[0].sum())  # first trade buys the risky book


def test_daily_engine_cash_earns_period_rate():
    days = pd.bdate_range("2021-01-01", "2021-03-31")
    px = pd.DataFrame({"X": 100.0}, index=days)
    dec = pd.DatetimeIndex([days[0], days[20], days[40]])
    W = pd.DataFrame({"X": [0.0, 0.0, 0.0], "CASH": [1.0, 1.0, 1.0]}, index=dec)
    rf = pd.Series([0.01, 0.02, 0.03], index=dec)
    nav, _ = sc.backtest_daily(W, px, rf, 10.0)
    assert nav.loc[dec[1]] == pytest.approx(1.01)
    assert nav.loc[dec[2]] == pytest.approx(1.01 * 1.02)


def test_daily_engine_late_start_holds_cash():
    days = pd.bdate_range("2021-01-01", "2021-02-26")
    px = pd.DataFrame({"X": np.linspace(100, 120, len(days))}, index=days)
    dec = pd.DatetimeIndex([days[21]])
    W = pd.DataFrame({"X": [1.0], "CASH": [0.0]}, index=dec)
    nav, _ = sc.backtest_daily(W, px, pd.Series([0.0], index=dec), 0.0, start=days[0], rf_start=0.0021)
    assert nav.loc[days[21]] == pytest.approx(1.0021)  # 21 trading days in cash at the start rate
    assert nav.iloc[-1] == pytest.approx(1.0021 * px["X"].iloc[-1] / px["X"].iloc[21])


# ---------------------------------------------------------------- deflated Sharpe ratio
def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    r = pd.Series(0.005 + 0.02 * rng.standard_normal(240), index=idx)
    rf = pd.Series(0.001, index=idx)
    trials = rng.normal(0.15, 0.05, 20)
    d5, sr0_5, sr = sc.deflated_sharpe(r, rf, trials, 5)
    d50, sr0_50, _ = sc.deflated_sharpe(r, rf, trials, 50)
    assert 0.0 <= d50 < d5 <= 1.0
    assert sr0_50 > sr0_5 > 0.0
    ex = r - rf
    assert sr == pytest.approx(np.sqrt(12) * ex.mean() / ex.std(ddof=1))


def test_deflated_sharpe_without_trial_spread_is_probabilistic_sharpe():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    r = pd.Series(0.004 + 0.01 * rng.standard_normal(120), index=idx)
    rf = pd.Series(0.0, index=idx)
    dsr, sr0, _ = sc.deflated_sharpe(r, rf, [0.1, 0.1, 0.1], 10)
    assert sr0 == pytest.approx(0.0, abs=1e-12)
    from scipy.stats import norm

    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std(ddof=0)
    skew, kurt = (z**3).mean(), (z**4).mean()
    psr = norm.cdf(sr * np.sqrt(119) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2))
    assert dsr == pytest.approx(psr)

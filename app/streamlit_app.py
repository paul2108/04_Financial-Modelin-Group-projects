"""Streamlit app: Risk Parity + Trend (Group Z, Financial Modeling, Nova SBE).

Lets the user change the strategy's key assumptions and see the resulting performance against a
benchmark. All calculations come from strategy_core.py, which is generated from the notebook
(cells tagged "core"), so the default settings reproduce the prospectus exactly.

Run locally:  streamlit run app/streamlit_app.py
"""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import strategy_core as sc

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"   # precomputed notebook exports
COLORS = {"strategy": "#2a78d6", "benchmark": "#eb6834", "market": "#1baf7a", "reference": "#6b6a64"}
VARIANT_COLORS = ["#8a5cd1", "#c2410c", "#0e7490", "#a16207", "#be185d", "#4d7c0f", "#475569", "#b91c1c"]
ASSET_COLORS = {"SPY": "#2a78d6", "EFA": "#eb6834", "IEF": "#1baf7a", "VNQ": "#eda100",
                "DBC": "#e87ba4", "GLD": "#008300", "CASH": "#a8a79f"}
DEFAULTS = {"use_trend": True, "sma": sc.SMA_WINDOW, "cov": sc.COV_WINDOW, "weighting": sc.WEIGHTING,
            "cost": sc.COST_BPS, "bench": "60/40 (60% SPY, 40% AGG)", "start_year": 2007,
            "assets": list(sc.RISKY_ASSETS), "overlay": "none", "cov_est": "sample", "trend_mode": "single",
            "leverage": 0.0, "compare": []}
BENCHMARKS = ["60/40 (60% SPY, 40% AGG)", "S&P 500 total return", "Equal weight 1/N (no trend filter)"]
WEIGHTINGS = {"erc": "Risk parity (equal risk contribution)", "hrp": "Hierarchical Risk Parity (ML clustering)",
              "inverse_vol": "Inverse volatility", "equal": "Equal weights (1/N)"}
LABELS = {"use_trend": "trend filter", "sma": "moving-average length", "cov": "risk window",
          "weighting": "weighting scheme", "cost": "transaction cost", "bench": "benchmark",
          "start_year": "start year", "assets": "asset classes", "overlay": "regime overlay",
          "cov_est": "covariance estimator", "trend_mode": "trend signal", "leverage": "leverage"}
OVERLAYS = {"none": "None (prospectus strategy)", "hmm": "HMM regime filter (machine learning)",
            "vol": "Volatility rule (simple baseline)"}
COV_ESTIMATORS = {"sample": "Sample covariance (prospectus)", "ledoit_wolf": "Ledoit-Wolf shrinkage"}
TREND_MODES = {"single": "One moving average (prospectus)", "average": "Average of 6–12-month moving averages"}
LEVERAGE = {0.0: "None (prospectus)"} | {t: f"Lever to {t:.0%} volatility" for t in sc.LEV_TARGETS}

# Scenarios that the notebook tested with pre-registered gates (Sections 9.2, 9.7, 9.8): the settings that differ
# from the prospectus -> (name in the notebook, reference of its gates)
TESTED = {
    frozenset({("weighting", "hrp")}): ("E: HRP + trend (§9.7)", "A"),
    frozenset({("weighting", "hrp"), ("use_trend", False)}): ("HRP without trend filter (§9.7)", "ERC"),
    frozenset({("trend_mode", "average")}): ("H: averaged 6–12-month trend (§9.7)", "A"),
    frozenset({("cov_est", "ledoit_wolf")}): ("I: Ledoit-Wolf covariance + trend (§9.7)", "A"),
    frozenset({("overlay", "vol")}): ("B: volatility rule (§9.2)", "A"),
    frozenset({("overlay", "hmm"), ("use_trend", False)}): ("C: HMM replaces the trend filter (§9.2)", "A"),
    frozenset({("overlay", "hmm")}): ("D: trend filter + HMM (§9.2)", "A"),
} | {frozenset({("leverage", t)}): (f"A levered to {t:.0%} volatility (§9.8)", "60/40") for t in sc.LEV_TARGETS}


def reset_defaults():
    """Button callback: runs before the widgets are rebuilt, so the session state may be changed."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v


def effective(ss):
    """Settings as they act on the strategy: the trend signal only matters with the trend filter on."""
    p = {k: ss[k] for k in DEFAULTS}
    if not p["use_trend"]:
        p["trend_mode"] = "single"
    return p


def lab_active(p):
    """True if any switch of the "Machine learning & robustness lab" differs from the prospectus."""
    return (p["weighting"] == "hrp" or p["cov_est"] != "sample" or p["trend_mode"] != "single"
            or p["overlay"] != "none" or p["leverage"] != 0.0)


# --------------------------------------------------------------------------- data and calculations
@st.cache_data(ttl=24 * 3600, show_spinner="Loading 20 years of daily prices from Yahoo Finance ...")
def load_market_data(start: str, end: str):
    prices = sc.download_prices(sc.ALL_TICKERS, start, end)
    retrieved = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data = sc.build_market_data(prices, sc.RISKY_ASSETS, sc.OTHER_TICKERS, sc.RF_TICKER)
    return data, retrieved


@st.cache_data(ttl=24 * 3600, show_spinner="Training the hidden Markov model walk-forward (about 10-20 s) ...")
def regime_signals(start: str, end: str):
    """Risk-off signals of Section 9 of the notebook for all decision dates (computed once, then cached)."""
    data, _ = load_market_data(start, end)
    prices = sc.clean_prices(sc.download_prices([sc.MARKET], sc.HMM_CFG.start, end))[sc.MARKET].dropna()
    features = sc.regime_features(prices, sc.HMM_CFG.vol_window)
    t0 = sc.first_decision_date(data, sc.RISKY_ASSETS, sc.COV_WINDOW, sc.SMA_WINDOW)
    dates = data.prices_m.index[data.prices_m.index >= t0]
    p_turb = sc.hmm_regime_probs(features, dates, sc.HMM_CFG, progress=False)
    return {"hmm": p_turb > sc.HMM_CFG.prob_threshold,
            "vol": sc.vol_rule_signal(features, dates, sc.HMM_CFG.vol_quantile)}


def evaluate(data, assets, cov, sma, use_trend, weighting, cost, bench, start_year, overlay="none", signals=None,
             cov_est="sample", trend_mode="single", leverage=0.0):
    """Run strategy + benchmark exactly as in the notebook and return returns, weights and metrics."""
    base_t0 = sc.first_decision_date(data, sc.RISKY_ASSETS, sc.COV_WINDOW, sc.SMA_WINDOW)
    sma_needed = max(sc.AVG_WINDOWS) if trend_mode == "average" else sma
    feasible = sc.first_decision_date(data, assets, cov, sma_needed if use_trend else 1)
    start = max(base_t0, feasible, pd.Timestamp(f"{start_year - 1}-12-31"))
    run = sc.run_strategy(data, assets, cov, sma, use_trend, weighting, cost, start,
                          benchmark_weights=sc.BENCHMARK_6040, trend_mode=trend_mode, cov_estimator=cov_est)
    if overlay != "none":      # Section 9: halve the growth assets in risk-off months
        W = sc.apply_risk_off(run["weights"], signals[overlay], sc.HMM_CFG.risk_off_scale)
        res = sc.backtest(W, data, cost)
        run.update(weights=W, result=res, returns=res["net"].rename("Strategy"))
    run["leverage"] = None
    if leverage:               # Section 9.8: lever each month to an ex-ante volatility target
        vol = sc.exante_vol(run["weights"], data, cov, cov_est)
        r_lev, lev = sc.levered_returns(run["weights"], run["returns"], data.rf_m, leverage, sc.LEV_CAP,
                                        sc.LEV_SPREAD, cost, vol=vol)
        run.update(returns=r_lev.rename("Strategy"), leverage=lev)
    r = run["returns"]
    rf = data.rf_m.loc[r.index]
    market = data.rets_m[sc.MARKET].loc[r.index]
    if bench.startswith("60/40"):
        b = run["benchmark"]
    elif bench.startswith("S&P"):
        b = market
    else:
        b = sc.run_strategy(data, sc.RISKY_ASSETS, cov, sma, False, "equal", cost, start)["returns"]
    b = b.rename("Benchmark")
    summary = sc.performance_summary(r, rf, benchmark=b, market=market,
                                     turnover=run["result"]["turnover"], periods=sc.PERIODS,
                                     nw_lags=sc.NW_LAGS)
    b_turn = run["benchmark_result"]["turnover"] if bench.startswith("60/40") else None
    b_summary = sc.performance_summary(b, rf, market=market, turnover=b_turn, periods=sc.PERIODS,
                                       nw_lags=sc.NW_LAGS)
    return {"run": run, "returns": r, "benchmark": b, "market": market, "summary": summary,
            "bench_summary": b_summary, "start": start}


@st.cache_data(ttl=24 * 3600, show_spinner="Running the backtest ...")
def cached_evaluate(start, end, assets, cov, sma, use_trend, weighting, cost, bench, start_year, overlay,
                    cov_est="sample", trend_mode="single", leverage=0.0):
    data, _ = load_market_data(start, end)
    signals = regime_signals(start, end) if overlay != "none" else None
    return evaluate(data, list(assets), cov, sma, use_trend, weighting, cost, bench, start_year, overlay, signals,
                    cov_est, trend_mode, leverage)


def evaluate_params(p):
    """cached_evaluate() for a settings dict (see effective())."""
    return cached_evaluate(sc.START, sc.END, tuple(p["assets"]), p["cov"], p["sma"], p["use_trend"], p["weighting"],
                           float(p["cost"]), p["bench"], p["start_year"], p["overlay"], p["cov_est"],
                           p["trend_mode"], float(p["leverage"]))


def gate_panel(p, res, data):
    """Pre-registered promotion gates (notebook Section 9.2) of the current scenario against its reference."""
    diff = frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in p.items()
                     if k not in ("bench", "compare") and v != DEFAULTS[k])
    tested = TESTED.get(diff)
    r = res["returns"]
    if p["leverage"]:          # leverage is judged against 60/40 at similar risk (Section 9.8): no stress gate
        ref_name, r_ref, periods = "60/40", res["run"]["benchmark"], {}
    else:
        if tested and tested[1] == "A":   # the notebook compared this variant with the prospectus rule A
            ref_p = dict(DEFAULTS, bench=p["bench"])
        else:                             # otherwise: the same settings with every lab switch off
            ref_p = p | {"weighting": "erc" if p["weighting"] == "hrp" else p["weighting"], "cov_est": "sample",
                         "trend_mode": "single", "overlay": "none", "leverage": 0.0}
        if all(ref_p[k] == DEFAULTS[k] for k in DEFAULTS if k not in ("bench", "compare")):
            ref_name = "A, the prospectus rule"
        elif tested:
            ref_name = "risk parity without trend filter"
        else:
            ref_name = "your settings with the lab switches off"
        r_ref = evaluate_params(ref_p)["returns"]
        periods = {k: (a, b) for k, (a, b) in sc.STRESS_PERIODS.items() if pd.Timestamp(a) >= r.index[0]}
    need = min(2, len(periods))
    g = sc.gate_verdict(r, r_ref, data.rf_m, periods, need)

    st.subheader("Promotion gates")

    def tile(col, label, passed, value, note):
        mark = "" if passed is None else (" ✓" if passed else " ✗")
        col.metric(label + mark, value, note, delta_color="off", delta_arrow="off", border=True)

    c = st.columns(4)
    tile(c[0], "Sharpe ratio", g["Sharpe >= A"], f"{g['Sharpe']:.2f}", f"reference {g['Sharpe A']:.2f}")
    tile(c[1], "Max drawdown", g["MDD not deeper"], f"{g['MDD %']:.1f}%", f"reference {g['MDD A %']:.1f}%")
    if periods:
        wins = int(g["stress wins"].split()[0])
        tile(c[2], "Stress periods better", wins >= need, g["stress wins"], f"need {need}")
    else:
        tile(c[2], "Stress periods", None, "–", "not a gate for leverage")
    missed = (not g["Sharpe >= A"]) + (not g["MDD not deeper"]) + (bool(periods) and wins < need)
    tile(c[3], "Verdict", g["PROMOTED"], "passes" if g["PROMOTED"] else "fails",
         "all gates met" if g["PROMOTED"] else f"{missed} of {3 if periods else 2} gates missed")
    reason = "" if g["PROMOTED"] else f"Fails: {sc.gate_reason(g, need, 'the reference')}. "
    note = (reason + f"Reference: **{ref_name}**, same months and costs. Gates fixed before the first notebook run: Sharpe "
            "ratio not lower, maximum drawdown not deeper, better in at least 2 of 3 stress periods "
            "(financial crisis, Covid crash, 2022 rate shock); leverage is judged against 60/40.")
    if tested:
        st.caption(note + f" This scenario is variant **{tested[0]}** of the notebook.")
    else:
        st.caption(note)
        st.warning("This combination was **not pre-registered** in the notebook. Treat the verdict as exploratory: "
                   "trying many combinations and keeping the best one overstates how well it will work "
                   "(the notebook's deflated Sharpe ratio corrects for the 24 variants it tried).", icon="⚠️")


@st.cache_data(ttl=24 * 3600)
def load_variants():
    """Monthly returns and gate verdicts of the variants tested in notebook Sections 9.7-9.11 (None if missing)."""
    csv, meta = RESULTS_DIR / "variants_monthly.csv", RESULTS_DIR / "variants_gates.json"
    if not (csv.exists() and meta.exists()):
        return None
    rets = pd.read_csv(csv, index_col=0, parse_dates=True)
    return rets, json.loads(meta.read_text(encoding="utf-8"))


def run_default(data):
    """Prospectus base case (used by the consistency test against the notebook's results.json)."""
    d = DEFAULTS
    return evaluate(data, d["assets"], d["cov"], d["sma"], d["use_trend"], d["weighting"], d["cost"],
                    d["bench"], d["start_year"])


# --------------------------------------------------------------------------- charts
def _layout(fig, ytitle, height=380, log=False):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
                      yaxis_title=ytitle)
    if log:
        fig.update_yaxes(type="log")
    return fig


def chart_cumulative(res, bench_label, log):
    fig = go.Figure()
    for s, name, color, width in [(res["returns"], "Risk Parity + Trend (net)", COLORS["strategy"], 2.6),
                                  (res["benchmark"], bench_label, COLORS["benchmark"], 2.0)]:
        nav = sc.nav_from_returns(s)
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name=name, line=dict(color=color, width=width),
                                 hovertemplate="%{y:.2f}"))
    if not bench_label.startswith("S&P"):
        nav = sc.nav_from_returns(res["market"])
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name="S&P 500 total return",
                                 line=dict(color=COLORS["market"], width=1.4), hovertemplate="%{y:.2f}"))
    fig.add_hline(y=1, line_width=1, line_color="#8a8984")
    return _layout(fig, "Value of $1 invested" + (" (log scale)" if log else ""), log=log)


def chart_drawdowns(res, bench_label):
    fig = go.Figure()
    for s, name, color in [(res["returns"], "Risk Parity + Trend (net)", COLORS["strategy"]),
                           (res["benchmark"], bench_label, COLORS["benchmark"])]:
        dd = sc.drawdown_series(s)
        fig.add_trace(go.Scatter(x=dd.index, y=100 * dd, name=name, line=dict(color=color, width=2),
                                 fill="tozeroy" if "Risk" in name else None, hovertemplate="%{y:.1f}%"))
    return _layout(fig, "Drawdown from peak (%)", height=300)


def chart_weights(W):
    fig = go.Figure()
    for c in W.columns:
        fig.add_trace(go.Scatter(x=W.index, y=100 * W[c], name="T-bills (cash)" if c == "CASH" else c,
                                 stackgroup="w", line=dict(width=0, shape="hv"),
                                 fillcolor=ASSET_COLORS.get(c, "#888"), hovertemplate="%{y:.1f}%"))
    fig.update_yaxes(range=[0, 100])
    return _layout(fig, "Target weight (%)", height=320)


def variants_section(p, res, data, variants):
    """Classifier variants of notebook Section 9.11 (precomputed) next to the current scenario, from Jan 2010."""
    rets, meta = variants
    by_label = {v["label"]: v for v in meta["variants"]}
    chosen = [by_label[k] for k in p["compare"] if k in by_label]
    start = pd.Timestamp(chosen[0]["start"]) + pd.offsets.MonthEnd(0)
    months = rets.index[rets.index >= start]
    scen = res["returns"].reindex(months)
    st.subheader(f"Tested classifier variants, {months[0]:%b %Y} – {months[-1]:%b %Y}")
    st.caption("Precomputed in the notebook (Section 9.11) with the prospectus settings and "
               f"{meta['cost_bps']:.0f} bps costs: the classifiers need a 3-state HMM, macro data and the FinBERT "
               "tone of every FOMC statement, which is too heavy to recompute in the app. They start in 2010 "
               "because of the training burn-in. All lines start at $1 in the same month.")
    series = [("Your scenario", scen, COLORS["strategy"], 2.6, None),
              ("A: Risk parity + trend", rets["A: Trend (base)"].loc[months], COLORS["reference"], 1.8, "dot"),
              ("60/40", rets["60/40"].loc[months], COLORS["benchmark"], 1.8, None)]
    series += [(v["label"], rets[v["column"]].loc[months], VARIANT_COLORS[i % len(VARIANT_COLORS)], 1.6, "dash")
               for i, v in enumerate(chosen)]
    fig = go.Figure()
    rows = {}
    rf = data.rf_m
    for name, r, color, width, dash in series:
        if r.isna().any():      # scenario starts later than 2010: no like-for-like comparison
            continue
        nav = sc.nav_from_returns(r)
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name=name, line=dict(color=color, width=width, dash=dash),
                                 hovertemplate="%{y:.2f}"))
        s = sc.performance_summary(r, rf)
        v = next((x for x in chosen if x["label"] == name), None)
        rows[name] = {"CAGR": f"{s['cagr']:.1%}", "Volatility": f"{s['ann_vol']:.1%}",
                      "Sharpe ratio": f"{s['sharpe']:.2f}", "Max drawdown": f"{s['max_dd']:.1%}",
                      "Gates (vs A)": ("✓ passes" if v["promoted"] else "✗ " + v["reason"]) if v else ""}
    fig.add_hline(y=1, line_width=1, line_color="#8a8984")
    st.plotly_chart(_layout(fig, "Value of $1 invested (log scale)", log=True), width="stretch")
    if scen.isna().any():
        st.info("Your scenario starts after the classifiers' first month, so it is not shown here.")
    st.dataframe(pd.DataFrame(rows).T, width="stretch")


def all_variants_table(meta):
    """The verdicts of all variants of notebook Sections 9.7-9.11 (the data behind the summary chart)."""
    rows = []
    for v in meta["variants"]:
        rows.append({"Group": v["group"], "Variant": v["label"], "From": v["start"], "Sharpe ratio": f"{v['sharpe']:.2f}",
                     "Max drawdown": f"{v['max_dd']:.1%}", "Reference": v["reference"],
                     "Reference Sharpe": f"{v['ref_sharpe']:.2f}", "Reference max DD": f"{v['ref_max_dd']:.1%}",
                     "Verdict": "✓ passes" if v["promoted"] else "✗ " + v["reason"]})
    return pd.DataFrame(rows).set_index("Variant")


# --------------------------------------------------------------------------- table
def metrics_table(res, bench_label):
    s, b = res["summary"], res["bench_summary"]
    rows = [("Average return p.a.", "ann_mean", "pct"), ("CAGR", "cagr", "pct"), ("Volatility p.a.", "ann_vol", "pct"),
            ("Sharpe ratio", "sharpe", "num"), ("Maximum drawdown", "max_dd", "pct"),
            ("Worst 12 months", "worst_12m", "pct"), ("Beta vs S&P 500 (not annualised)", "beta_mkt", "num"),
            ("Alpha vs S&P 500 p.a.", "alpha_mkt", "pct"), ("  t-stat of alpha (Newey-West)", "alpha_mkt_t", "num"),
            ("Tracking error vs benchmark", "tracking_error", "pct"), ("Information ratio vs benchmark", "info_ratio", "num"),
            ("Turnover p.a.", "turnover", "pct"), ("Final value of $1", "final_value", "num")]

    def f(v, kind):
        if v is None or not np.isfinite(v):
            return "–"
        return f"{100 * v:.1f}%" if kind == "pct" else f"{v:.2f}"

    return pd.DataFrame({"Measure": [r[0] for r in rows],
                         "Risk Parity + Trend": [f(s.get(k), kind) for _, k, kind in rows],
                         bench_label: [f(b.get(k), kind) for _, k, kind in rows]}).set_index("Measure")


# --------------------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Risk Parity + Trend | Group Z", page_icon="📈", layout="wide")
    st.title("Risk Parity + Trend")
    st.caption("A lower-drawdown alternative to the 60/40 portfolio · Group Z · Financial Modeling, "
               f"Nova SBE · [Source code and notebook]({sc.REPO_URL})")

    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, list(v) if isinstance(v, list) else v)
    variants = load_variants()

    with st.sidebar:
        st.header("Assumptions")
        st.caption("The defaults reproduce Table 2 of the prospectus.")
        st.toggle("Trend filter (move assets below their moving average into T-bills)", key="use_trend")
        st.slider("Moving-average length (months)", 4, 12, key="sma",
                  disabled=not st.session_state.use_trend or st.session_state.trend_mode == "average")
        st.select_slider("Risk-estimation window (trading days)", options=[63, 126, 252, 504], key="cov")
        st.radio("Weighting scheme", list(WEIGHTINGS), format_func=WEIGHTINGS.get, key="weighting",
                 help="Hierarchical Risk Parity (Lopez de Prado, 2016) clusters the assets by correlation before "
                      "splitting the risk budget; notebook Section 9.7.")
        st.slider("Transaction cost (bps per unit of turnover)", 0.0, 50.0, step=1.0, key="cost")
        st.selectbox("Benchmark", BENCHMARKS, key="bench")
        st.slider("Start of the backtest (year)", 2007, 2024, key="start_year")
        st.multiselect("Asset classes held", sc.RISKY_ASSETS, key="assets",
                       format_func=lambda a: f"{a} – {sc.ASSET_NAMES[a]}")

        # constant `expanded`: a value that changes with the switches would re-create the box and close it
        with st.expander("Machine learning & robustness lab", expanded=False):
            st.caption("Switch the variants of notebook Section 9 on and off. Each one was tested against gates "
                       "fixed in advance, and none was adopted. The prospectus uses the first option everywhere.")
            st.radio("Covariance estimator", list(COV_ESTIMATORS), format_func=COV_ESTIMATORS.get, key="cov_est",
                     help="Ledoit-Wolf shrinks the sample covariance matrix towards a structured target "
                          "(variant I, Section 9.7).")
            st.radio("Trend signal", list(TREND_MODES), format_func=TREND_MODES.get, key="trend_mode",
                     disabled=not st.session_state.use_trend,
                     help="Averaged signal: the share of the 6- to 12-month moving averages the price is above, so "
                          "an asset is sold in steps (variant H, Section 9.7).")
            st.radio("Regime overlay (halves equities, REITs and commodities in risk-off months)",
                     list(OVERLAYS), format_func=OVERLAYS.get, key="overlay",
                     help="Section 9.2: walk-forward hidden Markov model vs a simple volatility rule. Combine with "
                          "the trend filter on or off to reproduce variants B-D.")
            st.radio("Leverage", list(LEVERAGE), format_func=LEVERAGE.get, key="leverage",
                     help=f"Section 9.8: each month the portfolio is levered to the target ex-ante volatility "
                          f"(at most {sc.LEV_CAP:.0f}x). Borrowing above own capital costs T-bills + "
                          f"{100 * sc.LEV_SPREAD:.1f}% p.a.")
            if variants is None:
                st.caption("Classifier comparison unavailable: run the notebook to export "
                           "results/variants_monthly.csv.")
            else:
                st.multiselect("Compare with tested classifier variants (precomputed, from 2010)",
                               [v["label"] for v in variants[1]["variants"] if v["group"] == "ML classifiers"],
                               key="compare",
                               help="Section 9.11: logistic regression and gradient boosting with HMM states, "
                                    "macro data and the FinBERT tone of FOMC statements as features. F replaces "
                                    "the trend filter; G decides whether to follow a trend exit.")
        st.button("Reset to prospectus defaults", on_click=reset_defaults, width="stretch")

    ss = st.session_state
    if len(ss.assets) < 2:
        st.warning("Select at least two asset classes.")
        st.stop()

    try:
        data, retrieved = load_market_data(sc.START, sc.END)
    except Exception as exc:  # Yahoo Finance unreachable or rate-limited
        st.error("Yahoo Finance did not respond, so the data could not be loaded. This usually passes "
                 f"within a few minutes. Please reload the page.\n\nDetails: {exc}")
        st.stop()

    p = effective(ss)
    try:
        res = evaluate_params(p)
    except Exception as exc:  # fallback: the overlay could not be computed -> show the strategy without it
        if p["overlay"] == "none":
            raise
        st.warning(f"The regime overlay could not be computed ({exc}). Showing the strategy without it.")
        p["overlay"] = "none"
        res = evaluate_params(p)
    s, b = res["summary"], res["bench_summary"]
    bench_label = ss.bench.split(" (")[0] if not ss.bench.startswith("Equal") else "1/N"
    changed = [k for k, v in DEFAULTS.items() if k != "compare" and p[k] != v]
    period = f"{res['returns'].index[0]:%b %Y} – {res['returns'].index[-1]:%b %Y}"
    if changed:
        st.info(f"Showing **your scenario** ({period}). Changed from the prospectus: "
                + ", ".join(LABELS[k] for k in changed) + ".")
    else:
        st.success(f"Showing the **prospectus base case** ({period}, net of {sc.COST_BPS:.0f} bps costs).")
    if lab_active(p):
        gate_panel(p, res, data)

    c = st.columns(6)
    c[0].metric("CAGR", f"{s['cagr']:.1%}", f"{100 * (s['cagr'] - b['cagr']):+.1f} pp vs {bench_label}")
    c[1].metric("Volatility", f"{s['ann_vol']:.1%}", f"{100 * (s['ann_vol'] - b['ann_vol']):+.1f} pp",
                delta_color="inverse")
    c[2].metric("Sharpe ratio", f"{s['sharpe']:.2f}", f"{s['sharpe'] - b['sharpe']:+.2f}")
    c[3].metric("Max drawdown", f"{s['max_dd']:.1%}", f"{100 * (s['max_dd'] - b['max_dd']):+.1f} pp")
    c[4].metric("Beta vs S&P 500", f"{s['beta_mkt']:.2f}")
    c[5].metric(f"Information ratio vs {bench_label}", f"{s['info_ratio']:.2f}")

    st.subheader("Growth of $1")
    log = st.toggle("Log scale", value=True)
    st.plotly_chart(chart_cumulative(res, ss.bench.split(" (")[0], log), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Drawdowns")
        st.plotly_chart(chart_drawdowns(res, ss.bench.split(" (")[0]), width="stretch")
    with right:
        st.subheader("Allocation over time")
        st.plotly_chart(chart_weights(res["run"]["weights"]), width="stretch")
        lev = res["run"]["leverage"]
        if lev is not None:
            st.caption(f"Weights before leverage. Leverage applied: {lev.mean():.2f}x on average, "
                       f"{lev.min():.2f}x to {lev.max():.2f}x.")

    st.subheader("Performance measures")
    st.dataframe(metrics_table(res, bench_label), width="stretch")
    if lev is not None:
        st.caption("Turnover is shown before leverage changes; their trading costs are included in the returns.")
    out = pd.DataFrame({"strategy_net": res["returns"], "benchmark": res["benchmark"],
                        "risk_free": data.rf_m.loc[res["returns"].index]})
    st.download_button("Download monthly returns (CSV)", out.to_csv().encode(), "risk_parity_trend_returns.csv",
                       "text/csv")

    if variants is not None and p["compare"]:
        variants_section(p, res, data, variants)
    if variants is not None:
        with st.expander("All variants tested in the notebook and their verdicts"):
            st.caption(f"Notebook Sections 9.7–9.11, prospectus settings, net of {variants[1]['cost_bps']:.0f} bps; "
                       f"data retrieved {variants[1]['retrieved_utc']}. A variant would only have replaced the "
                       "prospectus rule if it had passed every gate. None did.")
            st.dataframe(all_variants_table(variants[1]), width="stretch")

    with st.expander("How the numbers are computed"):
        st.markdown(f"""
* **Data:** daily adjusted close prices from Yahoo Finance via `yfinance` ({sc.START} to {sc.END}, end exclusive),
  retrieved {retrieved}. Month-end values, simple returns. Risk-free rate: 13-week T-bill yield (`^IRX`) of the
  previous month-end / 1200.
* **Rule (every month-end):** covariance of the last *n* daily returns → equal-risk-contribution weights
  solved with `scipy.optimize.fsolve` → assets below their moving average move into T-bills → the weights earn
  the **next** month's return (no look-ahead).
* **Costs:** cost × turnover, where turnover = Σ|target weight − drifted weight|. The 60/40 benchmark pays the same cost.
* **Measures:** average return ×12, volatility ×√12, Sharpe ratio on excess returns, beta (not annualised) and
  Jensen's alpha against the S&P 500 total return, information ratio = annualised active return / tracking error.
* **Machine learning & robustness lab (optional, notebook Section 9):**
  *Regime overlay:* a 2-state hidden Markov model on daily S&P 500 returns and volatility, re-estimated every
  December on data up to then; the filtered probability uses past data only. In risk-off months the growth assets
  are halved. *HRP:* hierarchical clustering of the correlation matrix, then recursive bisection of the risk budget.
  *Ledoit-Wolf:* shrinkage estimate of the covariance matrix. *Averaged trend:* each asset is held with the share
  of its 6- to 12-month moving averages that the price is above. *Leverage:* each month L = target / ex-ante
  volatility (at most {sc.LEV_CAP:.0f}x); only exposure above own capital is borrowed, at T-bills +
  {100 * sc.LEV_SPREAD:.1f}% p.a., and changes in leverage pay the transaction cost. *Classifier variants:*
  monthly returns precomputed by the notebook (they need FinBERT and other heavy models). *Gates:* the pre-registered
  rules of Section 9.2, computed live with the same function as in the notebook.
* The code is identical to the notebook's (`strategy_core.py` is generated from it): [{sc.REPO_URL}]({sc.REPO_URL})
""")


if __name__ == "__main__":
    main()

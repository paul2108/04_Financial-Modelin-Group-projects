"""Streamlit app: Risk Parity + Trend (Group Z, Financial Modeling, Nova SBE).

Four tabs: an overview that compares the strategy with the alternatives a client would consider, a tab to change
the strategy's assumptions, a research lab with the machine-learning variants of notebook Section 9, and the
method. All calculations come from strategy_core.py, which is generated from the notebook (cells tagged "core"),
so the default settings reproduce the prospectus exactly.

Run locally:  streamlit run app/streamlit_app.py
"""

import datetime as dt
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import strategy_core as sc

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"  # precomputed notebook exports
OURS = "Risk Parity + Trend"
COLORS = {"strategy": "#2a78d6", "benchmark": "#eb6834", "market": "#1baf7a", "reference": "#6b6a64"}
STRATEGY_COLORS = {
    OURS: "#2a78d6",
    "60/40 portfolio": "#eb6834",
    "S&P 500": "#1baf7a",
    "Equal weights (1/N)": "#a16207",
    "Risk parity without trend filter": "#8a5cd1",
}
VARIANT_COLORS = ["#8a5cd1", "#c2410c", "#0e7490", "#a16207", "#be185d", "#4d7c0f", "#475569", "#b91c1c"]
ASSET_COLORS = {
    "SPY": "#2a78d6",
    "EFA": "#eb6834",
    "IEF": "#1baf7a",
    "VNQ": "#eda100",
    "DBC": "#e87ba4",
    "GLD": "#008300",
    "CASH": "#a8a79f",
}
DEFAULTS = {
    "use_trend": True,
    "sma": sc.SMA_WINDOW,
    "cov": sc.COV_WINDOW,
    "weighting": sc.WEIGHTING,
    "cost": sc.COST_BPS,
    "bench": "60/40 (60% SPY, 40% AGG)",
    "start_year": 2007,
    "assets": list(sc.RISKY_ASSETS),
    "overlay": "none",
    "cov_est": "sample",
    "trend_mode": "single",
    "leverage": 0.0,
    "compare": [],
}
EXPLORE_KEYS = ["use_trend", "sma", "cov", "weighting", "cost", "bench", "start_year", "assets"]
BENCHMARKS = ["60/40 (60% SPY, 40% AGG)", "S&P 500 total return", "Equal weight 1/N (no trend filter)"]
WEIGHTINGS = {
    "erc": "Risk parity (equal risk contribution)",
    "inverse_vol": "Inverse volatility",
    "equal": "Equal weights (1/N)",
}
LABELS = {
    "use_trend": "trend filter",
    "sma": "moving-average length",
    "cov": "risk window",
    "weighting": "weighting scheme",
    "cost": "transaction cost",
    "bench": "benchmark",
    "start_year": "start year",
    "assets": "asset classes",
    "overlay": "regime overlay",
    "cov_est": "covariance estimator",
    "trend_mode": "trend signal",
    "leverage": "leverage",
}
# research lab: each switch changes one building block of the prospectus rule
LAB_WEIGHTING = {"erc": "Risk parity (prospectus)", "hrp": "Hierarchical Risk Parity (ML clustering)"}
LAB_COV = {"sample": "Sample covariance (prospectus)", "ledoit_wolf": "Ledoit-Wolf shrinkage"}
LAB_TREND = {
    "single": "10-month moving average (prospectus)",
    "average": "Average of 6–12-month moving averages",
    "off": "No trend filter",
}
LAB_OVERLAY = {
    "none": "None (prospectus)",
    "hmm": "Hidden Markov model (machine learning)",
    "vol": "Volatility rule (simple baseline)",
}
LAB_LEVERAGE = {0.0: "None (prospectus)"} | {t: f"Lever to {t:.0%} volatility" for t in sc.LEV_TARGETS}

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


def effective(p):
    """Settings as they act on the strategy: the trend signal only matters with the trend filter on."""
    p = dict(p)
    if not p["use_trend"]:
        p["trend_mode"] = "single"
    return p


def lab_active(p):
    """True if any switch of the research lab differs from the prospectus."""
    return (
        p["weighting"] == "hrp"
        or p["cov_est"] != "sample"
        or p["trend_mode"] != "single"
        or p["overlay"] != "none"
        or p["leverage"] != 0.0
        or not p["use_trend"]
    )


# --------------------------------------------------------------------------- data and calculations
@st.cache_data(ttl=24 * 3600, show_spinner="Loading 20 years of daily prices from Yahoo Finance ...")
def load_market_data(start: str, end: str):
    prices = sc.download_prices(sc.ALL_TICKERS, start, end)
    retrieved = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data = sc.build_market_data(prices, sc.RISKY_ASSETS, sc.OTHER_TICKERS, sc.RF_TICKER)
    return data, retrieved


def compute_regime_signals(data, end):
    """Risk-off signals of notebook Section 9 for all decision dates (pure calculation, no Streamlit calls)."""
    prices = sc.clean_prices(sc.download_prices([sc.MARKET], sc.HMM_CFG.start, end))[sc.MARKET].dropna()
    features = sc.regime_features(prices, sc.HMM_CFG.vol_window)
    t0 = sc.first_decision_date(data, sc.RISKY_ASSETS, sc.COV_WINDOW, sc.SMA_WINDOW)
    dates = data.prices_m.index[data.prices_m.index >= t0]
    p_turb = sc.hmm_regime_probs(features, dates, sc.HMM_CFG, progress=False)
    return {
        "hmm": p_turb > sc.HMM_CFG.prob_threshold,
        "vol": sc.vol_rule_signal(features, dates, sc.HMM_CFG.vol_quantile),
    }


@st.cache_resource(ttl=24 * 3600, show_spinner=False)
def _regime_job(start: str, end: str, _data):
    """Starts the walk-forward HMM (10-30 s) once per server in a background thread, so that it is usually
    ready before a visitor opens the research lab. The result is shared by all visitors."""
    job = {"done": threading.Event(), "result": None, "error": None}

    def work():
        try:
            job["result"] = compute_regime_signals(_data, end)
        except Exception as exc:  # reported when the lab needs the signals
            job["error"] = exc
        finally:
            job["done"].set()

    threading.Thread(target=work, daemon=True).start()
    return job


def regime_signals(data):
    job = _regime_job(sc.START, sc.END, data)
    if not job["done"].is_set():
        with st.spinner("Training the hidden Markov model year by year (only needed once, about 10–30 s) ..."):
            job["done"].wait()
    if job["error"] is not None:
        _regime_job.clear()  # allow a new attempt on the next run
        raise job["error"]
    return job["result"]


def evaluate(
    data,
    assets,
    cov,
    sma,
    use_trend,
    weighting,
    cost,
    bench,
    start_year,
    overlay="none",
    signals=None,
    cov_est="sample",
    trend_mode="single",
    leverage=0.0,
):
    """Run strategy + benchmark exactly as in the notebook and return returns, weights and metrics."""
    base_t0 = sc.first_decision_date(data, sc.RISKY_ASSETS, sc.COV_WINDOW, sc.SMA_WINDOW)
    sma_needed = max(sc.AVG_WINDOWS) if trend_mode == "average" else sma
    feasible = sc.first_decision_date(data, assets, cov, sma_needed if use_trend else 1)
    start = max(base_t0, feasible, pd.Timestamp(f"{start_year - 1}-12-31"))
    run = sc.run_strategy(
        data,
        assets,
        cov,
        sma,
        use_trend,
        weighting,
        cost,
        start,
        benchmark_weights=sc.BENCHMARK_6040,
        trend_mode=trend_mode,
        cov_estimator=cov_est,
    )
    if overlay != "none":  # Section 9: halve the growth assets in risk-off months
        W = sc.apply_risk_off(run["weights"], signals[overlay], sc.HMM_CFG.risk_off_scale)
        res = sc.backtest(W, data, cost)
        run.update(weights=W, result=res, returns=res["net"].rename("Strategy"))
    run["leverage"] = None
    if leverage:  # Section 9.8: lever each month to an ex-ante volatility target
        vol = sc.exante_vol(run["weights"], data, cov, cov_est)
        r_lev, lev = sc.levered_returns(
            run["weights"], run["returns"], data.rf_m, leverage, sc.LEV_CAP, sc.LEV_SPREAD, cost, vol=vol
        )
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
    summary = sc.performance_summary(
        r, rf, benchmark=b, market=market, turnover=run["result"]["turnover"], periods=sc.PERIODS, nw_lags=sc.NW_LAGS
    )
    b_turn = run["benchmark_result"]["turnover"] if bench.startswith("60/40") else None
    b_summary = sc.performance_summary(b, rf, market=market, turnover=b_turn, periods=sc.PERIODS, nw_lags=sc.NW_LAGS)
    return {
        "run": run,
        "returns": r,
        "benchmark": b,
        "market": market,
        "summary": summary,
        "bench_summary": b_summary,
        "start": start,
    }


@st.cache_data(ttl=24 * 3600, show_spinner="Running the backtest ...")
def cached_evaluate(
    start,
    end,
    assets,
    cov,
    sma,
    use_trend,
    weighting,
    cost,
    bench,
    start_year,
    overlay,
    cov_est="sample",
    trend_mode="single",
    leverage=0.0,
    _signals=None,
):
    data, _ = load_market_data(start, end)
    return evaluate(
        data,
        list(assets),
        cov,
        sma,
        use_trend,
        weighting,
        cost,
        bench,
        start_year,
        overlay,
        _signals,
        cov_est,
        trend_mode,
        leverage,
    )


def evaluate_params(p, data=None):
    """cached_evaluate() for a settings dict (see effective()); the regime signals are loaded only if needed."""
    signals = regime_signals(data) if p["overlay"] != "none" else None
    return cached_evaluate(
        sc.START,
        sc.END,
        tuple(p["assets"]),
        p["cov"],
        p["sma"],
        p["use_trend"],
        p["weighting"],
        float(p["cost"]),
        p["bench"],
        p["start_year"],
        p["overlay"],
        p["cov_est"],
        p["trend_mode"],
        float(p["leverage"]),
        _signals=signals,
    )


def run_default(data):
    """Prospectus base case (used by the consistency test against the notebook's results.json)."""
    d = DEFAULTS
    return evaluate(
        data, d["assets"], d["cov"], d["sma"], d["use_trend"], d["weighting"], d["cost"], d["bench"], d["start_year"]
    )


@st.cache_data(ttl=24 * 3600, show_spinner="Comparing the strategies ...")
def strategy_comparison(start: str, end: str):
    """Monthly net returns of the prospectus strategy and the alternatives of Table 2 on the same months."""
    data, _ = load_market_data(start, end)
    base = run_default(data)
    t0 = base["start"]
    alt = {
        name: sc.run_strategy(data, sc.RISKY_ASSETS, sc.COV_WINDOW, sc.SMA_WINDOW, False, w, sc.COST_BPS, t0)["returns"]
        for name, w in [("Equal weights (1/N)", "equal"), ("Risk parity without trend filter", "erc")]
    }
    rets = pd.DataFrame(
        {OURS: base["returns"], "60/40 portfolio": base["run"]["benchmark"], "S&P 500": base["market"]} | alt
    )
    return rets, data.rf_m.loc[rets.index]


@st.cache_data(ttl=24 * 3600)
def load_variants():
    """Monthly returns and gate verdicts of the variants tested in notebook Sections 9.7-9.11 (None if missing)."""
    csv, meta = RESULTS_DIR / "variants_monthly.csv", RESULTS_DIR / "variants_gates.json"
    if not (csv.exists() and meta.exists()):
        return None
    rets = pd.read_csv(csv, index_col=0, parse_dates=True)
    return rets, json.loads(meta.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- charts
def _layout(fig, ytitle, height=380, log=False):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        yaxis_title=ytitle,
    )
    if log:
        fig.update_yaxes(type="log")
    return fig


def chart_growth_all(rets, amount):
    fig = go.Figure()
    for name in rets.columns:
        nav = amount * sc.nav_from_returns(rets[name])
        ours = name == OURS
        fig.add_trace(
            go.Scatter(
                x=nav.index,
                y=nav,
                name=name,
                hovertemplate="$%{y:,.0f}",
                line=dict(
                    color=STRATEGY_COLORS[name],
                    width=3.2 if ours else 1.6,
                    dash=None if ours or name in ("60/40 portfolio", "S&P 500") else "dot",
                ),
            )
        )
    fig.add_hline(y=amount, line_width=1, line_color="#8a8984")
    fig = _layout(fig, "Value of the investment ($, log scale)", height=400, log=True)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def chart_drawdowns_all(rets):
    fig = go.Figure()
    for name in [c for c in rets.columns if c != OURS] + [OURS]:  # ours last = drawn on top
        dd = 100 * sc.drawdown_series(rets[name])
        ours = name == OURS
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd,
                name=name,
                hovertemplate="%{y:.1f}%",
                fill="tozeroy" if ours else None,
                line=dict(color=STRATEGY_COLORS[name], width=2.6 if ours else 1.3),
            )
        )
    return _layout(fig, "Loss from the previous peak (%)", height=340)


def chart_crises(rets, amount):
    fig = go.Figure()
    names = [f"{k}<br>{pd.Timestamp(a):%b %Y}–{pd.Timestamp(b):%b %Y}" for k, (a, b) in sc.STRESS_PERIODS.items()]
    for name in rets.columns:
        vals = [100 * sc.stress_return(rets[name], a, b) for a, b in sc.STRESS_PERIODS.values()]
        fig.add_trace(
            go.Bar(
                x=names,
                y=vals,
                name=name,
                marker_color=STRATEGY_COLORS[name],
                text=[f"{v:+.1f}%" for v in vals],
                textposition="outside",
                customdata=[amount * v / 100 for v in vals],
                hovertemplate="%{y:+.1f}% = $%{customdata:,.0f}",
            )
        )
    fig.add_hline(y=0, line_width=1, line_color="#8a8984")
    fig.update_layout(barmode="group", hovermode="closest")
    return _layout(fig, "Return during the crisis (%)", height=380)


def chart_cumulative(res, bench_label, log):
    fig = go.Figure()
    for s, name, color, width in [
        (res["returns"], "Your scenario (net)", COLORS["strategy"], 2.6),
        (res["benchmark"], bench_label, COLORS["benchmark"], 2.0),
    ]:
        nav = sc.nav_from_returns(s)
        fig.add_trace(
            go.Scatter(x=nav.index, y=nav, name=name, line=dict(color=color, width=width), hovertemplate="%{y:.2f}")
        )
    if not bench_label.startswith("S&P"):
        nav = sc.nav_from_returns(res["market"])
        fig.add_trace(
            go.Scatter(
                x=nav.index,
                y=nav,
                name="S&P 500 total return",
                line=dict(color=COLORS["market"], width=1.4),
                hovertemplate="%{y:.2f}",
            )
        )
    fig.add_hline(y=1, line_width=1, line_color="#8a8984")
    return _layout(fig, "Value of $1 invested" + (" (log scale)" if log else ""), log=log)


def chart_drawdowns(res, bench_label):
    fig = go.Figure()
    for s, name, color in [
        (res["returns"], "Your scenario (net)", COLORS["strategy"]),
        (res["benchmark"], bench_label, COLORS["benchmark"]),
    ]:
        dd = sc.drawdown_series(s)
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=100 * dd,
                name=name,
                line=dict(color=color, width=2),
                fill="tozeroy" if "scenario" in name else None,
                hovertemplate="%{y:.1f}%",
            )
        )
    return _layout(fig, "Drawdown from peak (%)", height=300)


def chart_weights(W):
    fig = go.Figure()
    for c in W.columns:
        fig.add_trace(
            go.Scatter(
                x=W.index,
                y=100 * W[c],
                name="T-bills (cash)" if c == "CASH" else c,
                stackgroup="w",
                line=dict(width=0, shape="hv"),
                fillcolor=ASSET_COLORS.get(c, "#888"),
                hovertemplate="%{y:.1f}%",
            )
        )
    fig.update_yaxes(range=[0, 100])
    return _layout(fig, "Target weight (%)", height=320)


# --------------------------------------------------------------------------- tables
def comparison_table(rets, rf, amount):
    """One row per strategy; the best value of every column is highlighted (higher is better, except volatility)."""
    rows = {}
    for name in rets.columns:
        r = rets[name]
        s = sc.performance_summary(r, rf)
        rows[name] = {
            "Return p.a.": s["cagr"],
            "Volatility p.a.": s["ann_vol"],
            "Sharpe ratio": s["sharpe"],
            "Largest loss": s["max_dd"],
            "Worst 12 months": s["worst_12m"],
        }
        rows[name] |= {k: sc.stress_return(r, a, b) for k, (a, b) in sc.STRESS_PERIODS.items()}
        rows[name][f"${amount:,.0f} grew to"] = amount * s["final_value"]
    df = pd.DataFrame(rows).T
    lower_is_better = {"Volatility p.a."}

    def best(col):
        target = col.min() if col.name in lower_is_better else col.max()
        return [
            "background-color: rgba(27, 175, 122, 0.22); font-weight: 600" if np.isclose(v, target) else "" for v in col
        ]

    fmt = {c: "{:.1%}" for c in df.columns} | {"Sharpe ratio": "{:.2f}", f"${amount:,.0f} grew to": "${:,.0f}"}
    return df.style.apply(best).format(fmt)


def metrics_table(res, bench_label):
    s, b = res["summary"], res["bench_summary"]
    rows = [
        ("Average return p.a.", "ann_mean", "pct"),
        ("CAGR", "cagr", "pct"),
        ("Volatility p.a.", "ann_vol", "pct"),
        ("Sharpe ratio", "sharpe", "num"),
        ("Maximum drawdown", "max_dd", "pct"),
        ("Worst 12 months", "worst_12m", "pct"),
        ("Beta vs S&P 500 (not annualised)", "beta_mkt", "num"),
        ("Alpha vs S&P 500 p.a.", "alpha_mkt", "pct"),
        ("  t-stat of alpha (Newey-West)", "alpha_mkt_t", "num"),
        ("Tracking error vs benchmark", "tracking_error", "pct"),
        ("Information ratio vs benchmark", "info_ratio", "num"),
        ("Turnover p.a.", "turnover", "pct"),
        ("Final value of $1", "final_value", "num"),
    ]

    def f(v, kind):
        if v is None or not np.isfinite(v):
            return "–"
        return f"{100 * v:.1f}%" if kind == "pct" else f"{v:.2f}"

    return pd.DataFrame(
        {
            "Measure": [r[0] for r in rows],
            "Your scenario": [f(s.get(k), kind) for _, k, kind in rows],
            bench_label: [f(b.get(k), kind) for _, k, kind in rows],
        }
    ).set_index("Measure")


def all_variants_table(meta):
    """The verdicts of all variants of notebook Sections 9.7-9.11."""
    rows = []
    for v in meta["variants"]:
        rows.append(
            {
                "Group": v["group"],
                "Variant": v["label"],
                "From": v["start"],
                "Sharpe ratio": f"{v['sharpe']:.2f}",
                "Max drawdown": f"{v['max_dd']:.1%}",
                "Reference": v["reference"],
                "Reference Sharpe": f"{v['ref_sharpe']:.2f}",
                "Reference max DD": f"{v['ref_max_dd']:.1%}",
                "Verdict": "✓ passes" if v["promoted"] else "✗ " + v["reason"],
            }
        )
    return pd.DataFrame(rows).set_index("Variant")


# --------------------------------------------------------------------------- tab 1: overview
@st.fragment
def overview_tab():
    rets, rf = strategy_comparison(sc.START, sc.END)
    period = f"{rets.index[0]:%b %Y} – {rets.index[-1]:%b %Y}"
    amount = st.number_input(
        "Example investment ($)",
        min_value=10_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        format="%d",
        help="Only scales the dollar figures; all percentages stay the same.",
    )
    s = {n: sc.performance_summary(rets[n], rf) for n in rets.columns}
    o, b = s[OURS], s["60/40 portfolio"]

    st.subheader("Against the 60/40 portfolio, the default choice for cautious investors")
    c = st.columns(4)
    c[0].metric(
        "Largest loss",
        f"{o['max_dd']:.1%}",
        f"{100 * (o['max_dd'] - b['max_dd']):+.1f} pp vs 60/40",
        help=f"Worst fall from a previous peak. On ${amount:,.0f}: ${-amount * o['max_dd']:,.0f} "
        f"against ${-amount * b['max_dd']:,.0f} for 60/40.",
        border=True,
    )
    c[1].metric(
        "Worst 12 months", f"{o['worst_12m']:.1%}", f"{100 * (o['worst_12m'] - b['worst_12m']):+.1f} pp", border=True
    )
    c[2].metric(
        "Volatility",
        f"{o['ann_vol']:.1%}",
        f"{100 * (o['ann_vol'] - b['ann_vol']):+.1f} pp",
        delta_color="inverse",
        border=True,
    )
    c[3].metric("Return p.a.", f"{o['cagr']:.1%}", f"{100 * (o['cagr'] - b['cagr']):+.1f} pp", border=True)
    st.info(
        f"**In short:** on ${amount:,.0f} the largest loss was **${-amount * o['max_dd']:,.0f}**, against "
        f"${-amount * b['max_dd']:,.0f} for 60/40 and ${-amount * s['S&P 500']['max_dd']:,.0f} for the S&P 500. "
        f"The return per unit of risk was the same as 60/40 (Sharpe ratio {o['sharpe']:.2f} vs "
        f"{b['sharpe']:.2f}). The price: a lower return in long bull markets ({o['cagr']:.1%} vs "
        f"{b['cagr']:.1%} a year). The strategy is built for clients who would sell a 60/40 portfolio in a "
        "crash, not for clients who want to beat it."
    )

    st.subheader("All strategies side by side")
    st.caption(f"{period}, monthly, net of {sc.COST_BPS:.0f} bps trading costs. Green = best value in each column.")
    st.dataframe(comparison_table(rets, rf, amount), width="stretch")

    st.subheader("How each strategy did in the three crises")
    st.plotly_chart(chart_crises(rets, amount), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Growth of the investment")
        st.plotly_chart(chart_growth_all(rets, amount), width="stretch")
    with right:
        st.subheader("Losses from the previous peak")
        st.plotly_chart(chart_drawdowns_all(rets), width="stretch")


# --------------------------------------------------------------------------- tab 2: explore
def reset_explore():
    for k in EXPLORE_KEYS:
        st.session_state[k] = list(DEFAULTS[k]) if isinstance(DEFAULTS[k], list) else DEFAULTS[k]


@st.fragment
def explore_tab(data, retrieved):
    ss = st.session_state
    controls, results = st.columns([1, 3], gap="large")
    with controls:
        st.markdown("**Assumptions**")
        st.caption("The defaults are the prospectus strategy (Table 2).")
        st.toggle("Trend filter", key="use_trend", help="Move asset classes below their moving average into T-bills.")
        st.slider("Moving-average length (months)", 4, 12, key="sma", disabled=not ss.use_trend)
        st.select_slider(
            "Risk window (trading days)",
            options=[63, 126, 252, 504],
            key="cov",
            help="How many past days are used to estimate volatility and correlation.",
        )
        st.radio("Weighting", list(WEIGHTINGS), format_func=WEIGHTINGS.get, key="weighting")
        st.slider("Trading cost (bps per unit of turnover)", 0.0, 50.0, step=1.0, key="cost")
        st.selectbox("Benchmark", BENCHMARKS, key="bench")
        st.slider("Start year", 2007, 2024, key="start_year")
        st.multiselect(
            "Asset classes held", sc.RISKY_ASSETS, key="assets", format_func=lambda a: f"{a} – {sc.ASSET_NAMES[a]}"
        )
        st.button("Reset to prospectus", on_click=reset_explore, width="stretch")

    with results:
        if len(ss.assets) < 2:
            st.warning("Select at least two asset classes.")
            return
        p = effective(DEFAULTS | {k: ss[k] for k in EXPLORE_KEYS})
        res = evaluate_params(p, data)
        s, b = res["summary"], res["bench_summary"]
        bench_label = ss.bench.split(" (")[0] if not ss.bench.startswith("Equal") else "1/N"
        changed = [k for k in EXPLORE_KEYS if p[k] != DEFAULTS[k]]
        period = f"{res['returns'].index[0]:%b %Y} – {res['returns'].index[-1]:%b %Y}"
        if changed:
            st.info(
                f"**Your scenario** ({period}). Changed from the prospectus: "
                + ", ".join(LABELS[k] for k in changed)
                + "."
            )
        else:
            st.success(f"**Prospectus strategy** ({period}, net of {sc.COST_BPS:.0f} bps costs).")

        c = st.columns(4)
        c[0].metric("Return p.a.", f"{s['cagr']:.1%}", f"{100 * (s['cagr'] - b['cagr']):+.1f} pp vs {bench_label}")
        c[1].metric(
            "Volatility", f"{s['ann_vol']:.1%}", f"{100 * (s['ann_vol'] - b['ann_vol']):+.1f} pp", delta_color="inverse"
        )
        c[2].metric("Sharpe ratio", f"{s['sharpe']:.2f}", f"{s['sharpe'] - b['sharpe']:+.2f}")
        c[3].metric("Largest loss", f"{s['max_dd']:.1%}", f"{100 * (s['max_dd'] - b['max_dd']):+.1f} pp")

        log = st.toggle("Log scale", value=True)
        st.plotly_chart(chart_cumulative(res, bench_label, log), width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("**Drawdowns**")
            st.plotly_chart(chart_drawdowns(res, bench_label), width="stretch")
        with right:
            st.markdown("**Allocation over time**")
            st.plotly_chart(chart_weights(res["run"]["weights"]), width="stretch")
        with st.expander("All performance measures"):
            st.dataframe(metrics_table(res, bench_label), width="stretch")
        out = pd.DataFrame(
            {
                "strategy_net": res["returns"],
                "benchmark": res["benchmark"],
                "risk_free": data.rf_m.loc[res["returns"].index],
            }
        )
        st.download_button(
            "Download monthly returns (CSV)", out.to_csv().encode(), "risk_parity_trend_returns.csv", "text/csv"
        )


# --------------------------------------------------------------------------- tab 3: research lab
def gate_panel(p, res, data):
    """Pre-registered promotion gates (notebook Section 9.2) of the current scenario against its reference."""
    diff = frozenset(
        (k, tuple(v) if isinstance(v, list) else v)
        for k, v in p.items()
        if k not in ("bench", "compare") and v != DEFAULTS[k]
    )
    tested = TESTED.get(diff)
    r = res["returns"]
    if p["leverage"]:  # leverage is judged against 60/40 at similar risk (Section 9.8): no stress gate
        ref_name, r_ref, periods = "60/40", res["run"]["benchmark"], {}
    else:
        if tested and tested[1] == "A":  # the notebook compared this variant with the prospectus rule A
            ref_p = dict(DEFAULTS, bench=p["bench"])
        else:  # otherwise: the same settings with every lab switch off
            ref_p = p | {
                "weighting": "erc" if p["weighting"] == "hrp" else p["weighting"],
                "cov_est": "sample",
                "trend_mode": "single",
                "overlay": "none",
                "leverage": 0.0,
            }
        if all(ref_p[k] == DEFAULTS[k] for k in DEFAULTS if k not in ("bench", "compare")):
            ref_name = "the prospectus strategy"
        elif tested:
            ref_name = "risk parity without trend filter"
        else:
            ref_name = "the same settings with the lab switches off"
        r_ref = evaluate_params(ref_p, data)["returns"]
        periods = {k: (a, b) for k, (a, b) in sc.STRESS_PERIODS.items() if pd.Timestamp(a) >= r.index[0]}
    need = min(2, len(periods))
    g = sc.gate_verdict(r, r_ref, data.rf_m, periods, need)

    def tile(col, label, passed, value, note):
        mark = "" if passed is None else (" ✓" if passed else " ✗")
        col.metric(label + mark, value, note, delta_color="off", delta_arrow="off", border=True)

    c = st.columns(4)
    tile(c[0], "Sharpe ratio", g["Sharpe >= A"], f"{g['Sharpe']:.2f}", f"reference {g['Sharpe A']:.2f}")
    tile(c[1], "Largest loss", g["MDD not deeper"], f"{g['MDD %']:.1f}%", f"reference {g['MDD A %']:.1f}%")
    if periods:
        wins = int(g["stress wins"].split()[0])
        tile(c[2], "Crises better", wins >= need, g["stress wins"], f"need {need}")
    else:
        tile(c[2], "Crises", None, "–", "not a test for leverage")
    missed = (not g["Sharpe >= A"]) + (not g["MDD not deeper"]) + (bool(periods) and wins < need)
    tile(
        c[3],
        "Verdict",
        g["PROMOTED"],
        "would replace" if g["PROMOTED"] else "rejected",
        "all tests met" if g["PROMOTED"] else f"{missed} of {3 if periods else 2} tests missed",
    )
    reason = "" if g["PROMOTED"] else f"Rejected because of: {sc.gate_reason(g, need, 'the reference')}. "
    note = reason + f"Reference: **{ref_name}**, same months and costs."
    if tested:
        st.caption(note + f" This is variant **{tested[0]}** of the notebook.")
    else:
        st.caption(note)
        st.warning(
            "This combination was **not tested in advance** in the notebook. Treat the verdict as exploratory: "
            "trying many combinations and keeping the best one overstates how well it will work.",
            icon="⚠️",
        )


def variants_chart(rets, meta, chosen_labels, scenario=None):
    """Precomputed classifier variants (notebook Section 9.11) next to the prospectus strategy, from Jan 2010."""
    by_label = {v["label"]: v for v in meta["variants"]}
    chosen = [by_label[k] for k in chosen_labels if k in by_label]
    start = pd.Timestamp(chosen[0]["start"]) + pd.offsets.MonthEnd(0)
    months = rets.index[rets.index >= start]
    series = [
        ("Prospectus strategy (A)", rets["A: Trend (base)"].loc[months], COLORS["strategy"], 2.6, None),
        ("60/40", rets["60/40"].loc[months], COLORS["benchmark"], 1.8, None),
    ]
    series += [
        (v["label"], rets[v["column"]].loc[months], VARIANT_COLORS[i % len(VARIANT_COLORS)], 1.6, "dash")
        for i, v in enumerate(chosen)
    ]
    fig, rows = go.Figure(), {}
    rf = load_market_data(sc.START, sc.END)[0].rf_m
    for name, r, color, width, dash in series:
        nav = sc.nav_from_returns(r)
        fig.add_trace(
            go.Scatter(
                x=nav.index, y=nav, name=name, line=dict(color=color, width=width, dash=dash), hovertemplate="%{y:.2f}"
            )
        )
        s = sc.performance_summary(r, rf)
        v = next((x for x in chosen if x["label"] == name), None)
        rows[name] = {
            "Return p.a.": f"{s['cagr']:.1%}",
            "Volatility": f"{s['ann_vol']:.1%}",
            "Sharpe ratio": f"{s['sharpe']:.2f}",
            "Largest loss": f"{s['max_dd']:.1%}",
            "Verdict": ("✓ passes" if v["promoted"] else "✗ " + v["reason"]) if v else "",
        }
    fig.add_hline(y=1, line_width=1, line_color="#8a8984")
    st.caption(
        f"{months[0]:%b %Y} – {months[-1]:%b %Y} (the classifiers need a training period, so they start "
        "in 2010). Precomputed in the notebook: they use FinBERT and other heavy models."
    )
    st.plotly_chart(_layout(fig, "Value of $1 invested (log scale)", log=True), width="stretch")
    st.dataframe(pd.DataFrame(rows).T, width="stretch")


def reset_lab():
    for k, v in {
        "lab_weighting": "erc",
        "lab_cov": "sample",
        "lab_trend": "single",
        "lab_overlay": "none",
        "lab_leverage": 0.0,
    }.items():
        st.session_state[k] = v


@st.fragment
def lab_tab(data):
    variants = load_variants()
    st.markdown(
        "**What is this?** Before we settled on the prospectus rule, we tested whether machine learning or other "
        "changes would make it better. To avoid fooling ourselves, the tests were fixed **before** the first run: "
        "an alternative would only replace the prospectus strategy if it had (1) a Sharpe ratio at least as high, "
        "(2) a largest loss no deeper and (3) a better result in at least two of the three crises. "
        "**None of the 24 alternatives passed**, so the prospectus strategy stays as it is. "
        "Here you can check this yourself."
    )

    st.subheader("1 · Switch an alternative on")
    st.caption("Each switch replaces one building block of the prospectus strategy. The tests below are computed live.")
    c = st.columns(5)
    c[0].radio(
        "Weighting",
        list(LAB_WEIGHTING),
        format_func=LAB_WEIGHTING.get,
        key="lab_weighting",
        help="Hierarchical Risk Parity clusters the assets by correlation first (variant E).",
    )
    c[1].radio(
        "Risk estimate",
        list(LAB_COV),
        format_func=LAB_COV.get,
        key="lab_cov",
        help="Ledoit-Wolf shrinkage gives a less noisy covariance matrix (variant I).",
    )
    c[2].radio(
        "Trend filter",
        list(LAB_TREND),
        format_func=LAB_TREND.get,
        key="lab_trend",
        help="The averaged signal sells an asset in steps (variant H).",
    )
    c[3].radio(
        "Crash warning",
        list(LAB_OVERLAY),
        format_func=LAB_OVERLAY.get,
        key="lab_overlay",
        help="Halves equities, REITs and commodities in months flagged as risky (variants B–D). The hidden "
        "Markov model is trained once in the background; the first use can take up to 30 s.",
    )
    c[4].radio(
        "Leverage",
        list(LAB_LEVERAGE),
        format_func=LAB_LEVERAGE.get,
        key="lab_leverage",
        help=f"Borrows to reach the target volatility (at most {sc.LEV_CAP:.0f}x, at T-bills + "
        f"{100 * sc.LEV_SPREAD:.1f}%); judged against 60/40.",
    )
    st.button("Switch all off", on_click=reset_lab)

    ss = st.session_state
    p = effective(
        DEFAULTS
        | {
            "weighting": ss.lab_weighting,
            "cov_est": ss.lab_cov,
            "use_trend": ss.lab_trend != "off",
            "trend_mode": "single" if ss.lab_trend == "off" else ss.lab_trend,
            "overlay": ss.lab_overlay,
            "leverage": ss.lab_leverage,
        }
    )
    if not lab_active(p):
        st.success("All switches are off: this is the prospectus strategy. Switch an alternative on to test it.")
    else:
        try:
            res = evaluate_params(p, data)
        except Exception as exc:  # e.g. the S&P 500 history for the HMM could not be downloaded
            st.error(f"This alternative could not be computed right now ({exc}). Please try again in a minute.")
            return
        gate_panel(p, res, data)
        base = evaluate_params(dict(DEFAULTS), data)
        fig = go.Figure()
        for r, name, color, width in [
            (base["returns"], "Prospectus strategy", COLORS["strategy"], 2.6),
            (res["returns"], "Alternative", "#c2410c", 2.0),
            (res["run"]["benchmark"], "60/40", COLORS["reference"], 1.4),
        ]:
            nav = sc.nav_from_returns(r)
            fig.add_trace(
                go.Scatter(x=nav.index, y=nav, name=name, line=dict(color=color, width=width), hovertemplate="%{y:.2f}")
            )
        st.plotly_chart(_layout(fig, "Value of $1 invested (log scale)", height=340, log=True), width="stretch")

    if variants is not None:
        rets, meta = variants
        st.subheader("2 · Machine-learning classifiers (precomputed)")
        labels = [v["label"] for v in meta["variants"] if v["group"] == "ML classifiers"]
        chosen = st.multiselect(
            "Classifiers to compare",
            labels,
            default=labels[:2],
            key="compare",
            help="Logistic regression and gradient boosting with HMM states, macro data and the "
            "FinBERT tone of FOMC statements. F replaces the trend filter; G decides "
            "whether to follow a trend exit.",
        )
        if chosen:
            variants_chart(rets, meta, chosen)
        st.subheader("3 · All tested alternatives and their verdicts")
        st.caption(
            f"Notebook Sections 9.7–9.11, prospectus settings, net of {meta['cost_bps']:.0f} bps; data "
            f"retrieved {meta['retrieved_utc']}."
        )
        st.dataframe(all_variants_table(meta), width="stretch")


# --------------------------------------------------------------------------- tab 4: method
def method_tab(retrieved):
    st.markdown(f"""
* **Data:** daily adjusted close prices from Yahoo Finance via `yfinance` ({sc.START} to {sc.END}, end exclusive),
  retrieved {retrieved}. Month-end values, simple returns. Risk-free rate: 13-week T-bill yield (`^IRX`) of the
  previous month-end / 1200.
* **Rule (every month-end):** covariance of the last 252 daily returns → equal-risk-contribution weights
  solved with `scipy.optimize.fsolve` → assets below their 10-month moving average move into T-bills → the weights
  earn the **next** month's return (no look-ahead).
* **Costs:** cost × turnover, where turnover = Σ|target weight − drifted weight|. All strategies pay the same cost.
* **Alternatives in the overview:** 60/40 = 60% SPY and 40% AGG, rebalanced monthly; S&P 500 = total return index;
  1/N = equal weights in the six asset classes without trend filter; risk parity without trend filter = step 1 of
  the rule only.
* **Measures:** return p.a. = compound annual growth rate; volatility = standard deviation × √12; Sharpe ratio on
  excess returns over T-bills; largest loss = maximum drawdown from a previous peak; crises = total return over
  {", ".join(f"{k} ({pd.Timestamp(a):%b %Y}–{pd.Timestamp(b):%b %Y})" for k, (a, b) in sc.STRESS_PERIODS.items())}.
* **Research lab:** *Hidden Markov model:* a 2-state model on daily S&P 500 returns and volatility, re-estimated
  every December on data up to then; only past data are used. *HRP:* hierarchical clustering of the correlation
  matrix, then recursive bisection of the risk budget. *Ledoit-Wolf:* shrinkage estimate of the covariance matrix.
  *Averaged trend:* each asset is held with the share of its 6- to 12-month moving averages that the price is above.
  *Leverage:* each month L = target / ex-ante volatility (at most {sc.LEV_CAP:.0f}x), borrowing at T-bills +
  {100 * sc.LEV_SPREAD:.1f}% p.a. *Classifiers:* monthly returns precomputed by the notebook.
* The code is identical to the notebook's (`strategy_core.py` is generated from it): [{sc.REPO_URL}]({sc.REPO_URL})

*Past performance, in particular simulated past performance, is not a reliable indicator of future results.
University project, not investment advice.*
""")


# --------------------------------------------------------------------------- page
def main():
    st.set_page_config(
        page_title="Risk Parity + Trend | Group Z", page_icon="📈", layout="wide", initial_sidebar_state="collapsed"
    )
    st.title("Risk Parity + Trend")
    st.caption(
        "A lower-drawdown alternative to the 60/40 portfolio · Group Z · Financial Modeling, "
        f"Nova SBE · [Source code and notebook]({sc.REPO_URL})"
    )

    for k, v in DEFAULTS.items():
        if k in EXPLORE_KEYS:
            st.session_state.setdefault(k, list(v) if isinstance(v, list) else v)
    for k, v in {
        "lab_weighting": "erc",
        "lab_cov": "sample",
        "lab_trend": "single",
        "lab_overlay": "none",
        "lab_leverage": 0.0,
    }.items():
        st.session_state.setdefault(k, v)

    try:
        data, retrieved = load_market_data(sc.START, sc.END)
    except Exception as exc:  # Yahoo Finance unreachable or rate-limited
        st.error(
            "Yahoo Finance did not respond, so the data could not be loaded. This usually passes "
            f"within a few minutes. Please reload the page.\n\nDetails: {exc}"
        )
        st.stop()
    _regime_job(sc.START, sc.END, data)  # start the slow HMM in the background right away

    tabs = st.tabs(["Overview", "Explore the strategy", "Research lab: tested alternatives", "How it works"])
    with tabs[0]:
        overview_tab()
    with tabs[1]:
        explore_tab(data, retrieved)
    with tabs[2]:
        lab_tab(data)
    with tabs[3]:
        method_tab(retrieved)


if __name__ == "__main__":
    main()

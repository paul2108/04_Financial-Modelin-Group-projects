"""Streamlit app: Risk Parity + Trend (Group Z, Financial Modeling, Nova SBE).

Lets the user change the strategy's key assumptions and see the resulting performance against a
benchmark. All calculations come from strategy_core.py, which is generated from the notebook
(cells tagged "core"), so the default settings reproduce the prospectus exactly.

Run locally:  streamlit run app/streamlit_app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import strategy_core as sc

COLORS = {"strategy": "#0D366B", "benchmark": "#5B6676", "market": "#AEB7C2"}   # same palette as the prospectus
ASSET_COLORS = {"SPY": "#0D366B", "EFA": "#3F78BF", "IEF": "#9DC0EA", "VNQ": "#4D5866",
                "DBC": "#A7B0BC", "GLD": "#B07D1E", "CASH": "#D3D9E1"}
DEFAULTS = {"use_trend": True, "sma": sc.SMA_WINDOW, "cov": sc.COV_WINDOW, "weighting": sc.WEIGHTING,
            "cost": sc.COST_BPS, "bench": "60/40 (60% SPY, 40% AGG)", "start_year": 2007,
            "assets": list(sc.RISKY_ASSETS), "overlay": "none"}
BENCHMARKS = ["60/40 (60% SPY, 40% AGG)", "S&P 500 total return", "Equal weight 1/N (no trend filter)"]
WEIGHTINGS = {"erc": "Risk parity (equal risk contribution)", "inverse_vol": "Inverse volatility",
              "equal": "Equal weights (1/N)"}
LABELS = {"use_trend": "trend filter", "sma": "moving-average length", "cov": "risk window",
          "weighting": "weighting scheme", "cost": "transaction cost", "bench": "benchmark",
          "start_year": "start year", "assets": "asset classes", "overlay": "machine-learning overlay"}
OVERLAYS = {"none": "None (prospectus strategy)", "hmm": "HMM regime filter (machine learning)",
            "vol": "Volatility rule (simple baseline)"}


def reset_defaults():
    """Button callback: runs before the widgets are rebuilt, so the session state may be changed."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


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


def evaluate(data, assets, cov, sma, use_trend, weighting, cost, bench, start_year, overlay="none", signals=None):
    """Run strategy + benchmark exactly as in the notebook and return returns, weights and metrics."""
    base_t0 = sc.first_decision_date(data, sc.RISKY_ASSETS, sc.COV_WINDOW, sc.SMA_WINDOW)
    feasible = sc.first_decision_date(data, assets, cov, sma if use_trend else 1)
    start = max(base_t0, feasible, pd.Timestamp(f"{start_year - 1}-12-31"))
    run = sc.run_strategy(data, assets, cov, sma, use_trend, weighting, cost, start,
                          benchmark_weights=sc.BENCHMARK_6040)
    if overlay != "none":      # Section 9: halve the growth assets in risk-off months
        W = sc.apply_risk_off(run["weights"], signals[overlay], sc.HMM_CFG.risk_off_scale)
        res = sc.backtest(W, data, cost)
        run.update(weights=W, result=res, returns=res["net"].rename("Strategy"))
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
def cached_evaluate(start, end, assets, cov, sma, use_trend, weighting, cost, bench, start_year, overlay):
    data, _ = load_market_data(start, end)
    signals = regime_signals(start, end) if overlay != "none" else None
    return evaluate(data, list(assets), cov, sma, use_trend, weighting, cost, bench, start_year, overlay, signals)


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
    fig.add_hline(y=1, line_width=1, line_color="#8A94A3")
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
        st.session_state.setdefault(k, v)

    with st.sidebar:
        st.header("Assumptions")
        st.caption("The defaults reproduce Table 2 of the prospectus.")
        st.toggle("Trend filter (move assets below their moving average into T-bills)", key="use_trend")
        st.slider("Moving-average length (months)", 4, 12, key="sma", disabled=not st.session_state.use_trend)
        st.select_slider("Risk-estimation window (trading days)", options=[63, 126, 252, 504], key="cov")
        st.radio("Weighting scheme", list(WEIGHTINGS), format_func=WEIGHTINGS.get, key="weighting")
        st.slider("Transaction cost (bps per unit of turnover)", 0.0, 50.0, step=1.0, key="cost")
        st.selectbox("Benchmark", BENCHMARKS, key="bench")
        st.slider("Start of the backtest (year)", 2007, 2024, key="start_year")
        st.multiselect("Asset classes held", sc.RISKY_ASSETS, key="assets",
                       format_func=lambda a: f"{a} – {sc.ASSET_NAMES[a]}")
        st.radio("Machine-learning overlay (halves equities, REITs and commodities in risk-off months)",
                 list(OVERLAYS), format_func=OVERLAYS.get, key="overlay",
                 help="Section 9 of the notebook: walk-forward hidden Markov model vs a simple volatility rule. "
                      "Combine with the trend filter on or off to reproduce variants A-D.")
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

    try:
        res = cached_evaluate(sc.START, sc.END, tuple(ss.assets), ss.cov, ss.sma, ss.use_trend, ss.weighting,
                              float(ss.cost), ss.bench, ss.start_year, ss.overlay)
    except Exception as exc:  # fallback: the overlay could not be computed -> show the strategy without it
        st.warning(f"The machine-learning overlay could not be computed ({exc}). Showing the strategy without it.")
        res = cached_evaluate(sc.START, sc.END, tuple(ss.assets), ss.cov, ss.sma, ss.use_trend, ss.weighting,
                              float(ss.cost), ss.bench, ss.start_year, "none")
    s, b = res["summary"], res["bench_summary"]
    bench_label = ss.bench.split(" (")[0] if not ss.bench.startswith("Equal") else "1/N"
    changed = [k for k, v in DEFAULTS.items() if ss[k] != v]
    period = f"{res['returns'].index[0]:%b %Y} – {res['returns'].index[-1]:%b %Y}"
    if changed:
        st.info(f"Showing **your scenario** ({period}). Changed from the prospectus: "
                + ", ".join(LABELS[k] for k in changed) + ".")
    else:
        st.success(f"Showing the **prospectus base case** ({period}, net of {sc.COST_BPS:.0f} bps costs).")

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

    st.subheader("Performance measures")
    st.dataframe(metrics_table(res, bench_label), width="stretch")
    out = pd.DataFrame({"strategy_net": res["returns"], "benchmark": res["benchmark"],
                        "risk_free": data.rf_m.loc[res["returns"].index]})
    st.download_button("Download monthly returns (CSV)", out.to_csv().encode(), "risk_parity_trend_returns.csv",
                       "text/csv")

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
* **Machine-learning overlay (optional):** a 2-state hidden Markov model on daily S&P 500 returns and volatility,
  re-estimated every December on data up to then; the filtered probability uses past data only. In risk-off months
  the growth assets are halved. Same gates and comparison as Section 9 of the notebook.
* The code is identical to the notebook's (`strategy_core.py` is generated from it): [{sc.REPO_URL}]({sc.REPO_URL})
""")


if __name__ == "__main__":
    main()

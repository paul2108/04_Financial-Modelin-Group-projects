"""Consistency between notebook, generated core and app.

* test_core_is_up_to_date: app/strategy_core.py equals the notebook's "core" cells (no network).
* test_app_defaults_reproduce_notebook: the app's default run reproduces the headline numbers the
  notebook exported to results/results.json (needs Yahoo Finance, marked "network").
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_core_is_up_to_date():
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "export_core.py"), "--check"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.network
def test_app_defaults_reproduce_notebook():
    import strategy_core as sc
    import streamlit_app as app

    results = json.loads((ROOT / "results" / "results.json").read_text())
    data, _ = app.load_market_data(sc.START, sc.END)
    run = app.run_default(data)
    for key in ["ann_mean", "ann_vol", "sharpe", "beta_mkt", "alpha_mkt", "info_ratio", "max_dd"]:
        # a fresh download can differ from the notebook's in the last digits (Yahoo re-computes adjusted
        # prices), so we allow 1e-4; any difference in the calculation itself would be far larger
        assert run["summary"][key] == pytest.approx(results["strategy"][key], abs=1e-4), key


def test_variant_exports_are_consistent():
    """results/variants_monthly.csv and variants_gates.json (notebook Section 9.13) describe the same variants."""
    import pandas as pd

    rets = pd.read_csv(ROOT / "results" / "variants_monthly.csv", index_col=0, parse_dates=True)
    meta = json.loads((ROOT / "results" / "variants_gates.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "results" / "results.json").read_text())
    assert len(meta["variants"]) == 14
    assert {"A: Trend (base)", "60/40", "ERC (no trend)"} <= set(rets.columns)
    for v in meta["variants"]:
        assert v["column"] in rets.columns, v["label"]
        r = rets[v["column"]].dropna()
        assert f"{r.index[0]:%Y-%m}" == v["start"], v["label"]
        assert not v["promoted"], f"{v['label']} passed its gates: the prospectus says no variant did"
    assert len(rets) == results["strategy"]["months"]


@pytest.mark.network
def test_app_lab_reproduces_notebook_variants():
    """The app's live lab switches reproduce the notebook's variants E, H, I and the levered strategy."""
    import pandas as pd
    import strategy_core as sc
    import streamlit_app as app

    rets = pd.read_csv(ROOT / "results" / "variants_monthly.csv", index_col=0, parse_dates=True)
    data, _ = app.load_market_data(sc.START, sc.END)
    d = app.DEFAULTS
    cases = {"E: HRP + trend": {"weighting": "hrp"}, "H: ERC + averaged trend": {"trend_mode": "average"},
             "I: ERC (Ledoit-Wolf) + trend": {"cov_est": "ledoit_wolf"}, "A levered to 10% vol": {"leverage": 0.10},
             "HRP (no trend)": {"weighting": "hrp", "use_trend": False}}
    for col, change in cases.items():
        p = d | change
        res = app.evaluate(data, p["assets"], p["cov"], p["sma"], p["use_trend"], p["weighting"], p["cost"],
                           p["bench"], p["start_year"], cov_est=p["cov_est"], trend_mode=p["trend_mode"],
                           leverage=p["leverage"])
        live, nb = res["returns"], rets[col]
        assert live.index.equals(nb.index), col
        assert live.to_numpy() == pytest.approx(nb.to_numpy(), abs=1e-5), col

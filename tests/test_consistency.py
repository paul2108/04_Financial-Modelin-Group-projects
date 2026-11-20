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
    data = app.load_market_data(sc.START, sc.END)
    run = app.run_default(data)
    for key in ["ann_mean", "ann_vol", "sharpe", "beta_mkt", "alpha_mkt", "info_ratio", "max_dd"]:
        assert run["summary"][key] == pytest.approx(results["strategy"][key], abs=1e-6), key

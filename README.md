# Risk Parity + Trend — Group Z (Financial Modeling, Nova SBE)

An investor prospectus, a backtest notebook and an interactive app for a simple, rules-based model portfolio:
six asset-class ETFs (SPY, EFA, IEF, VNQ, DBC, GLD), **equal risk contribution** weights solved with
`scipy.optimize.fsolve`, and a **10-month trend filter** that moves weak asset classes into T-bills.
Rebalanced monthly and compared with a 60/40 portfolio, net of transaction costs.
A machine-learning extension (notebook Section 9) tests a walk-forward hidden Markov model as regime filter against
the trend filter and a simple volatility rule, with promotion gates fixed before the first run. Sections 9.6–9.13 add
Hierarchical Risk Parity, walk-forward classifiers (logistic regression, gradient boosting) with HMM states, macro data
and the FinBERT tone of FOMC statements as features, leverage to a volatility target, and robustness checks (block
bootstrap, rebalancing day, subsamples). Sections 9.14–9.21 test changes to the rule itself: hysteresis and no-trade
bands, tranche rebalancing (daily engine), an out-of-sample test on 1992–2007 proxy data, TIP, leverage financing, a EUR
investor view and the deflated Sharpe ratio. A full run takes about 2–3 minutes (with the local cache in `notebook/cache/`)
and needs internet access to Yahoo Finance, FRED (euro interest rate), federalreserve.gov and Hugging Face. FinBERT uses
the Apple-silicon GPU (MPS) or a CUDA GPU when available. Background: `docs/handover/`.

| | |
|---|---|
| Interactive app | https://04financial-modelin-group-projects-u7jx8pavhcsfkqmkps5b5d.streamlit.app |
| Notebook in Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paul2108/04_Financial-Modelin-Group-projects/blob/main/notebook/GroupZ_RiskParity_Trend.ipynb) |
| Prospectus (PDF) | [`prospectus/GroupZ_Prospectus.pdf`](prospectus/GroupZ_Prospectus.pdf) |
| Executed notebook as PDF (read without running) | [`notebook/GroupZ_RiskParity_Trend.pdf`](notebook/GroupZ_RiskParity_Trend.pdf) |

> **Disclaimer:** This is a university group project (Financial Modeling, Nova School of Business and Economics)
> and **not investment advice**. Past performance, in particular simulated past performance, is not a reliable
> indicator of future results.

**Group Z:** Gabriel Depraz, Marwan El-shazly, Bruno Francisco Barth, Tim Jonas Trompa, Paul Kuehl, Felix Schoenstein.

## Repository layout

```
notebook/GroupZ_RiskParity_Trend.ipynb   the backtest (ETL -> strategy -> results -> checks -> export); single source of truth
app/strategy_core.py                     AUTO-GENERATED from the notebook cells tagged "core" (make core)
app/streamlit_app.py                     Streamlit app using exactly that core
results/variants_*.{csv,json}            notebook export of the tested ML variants (classifiers shown in the app)
prospectus/main.tex, sections/*.tex      LaTeX prospectus; every number comes from results_macros.tex
prospectus/resultref.sty                 \res{key}: value + link to the "Sources of figures" appendix
tests/                                   unit tests (synthetic data) and notebook-vs-app consistency test
tools/export_core.py, check_numbers.py   core export and provenance check (both support --dry-run)
```

## Run it

**Colab:** open the badge above and choose *Runtime -> Run all*. No data files are needed: all prices come from
Yahoo Finance through `yfinance`.

**Locally** (Python 3.12, [uv](https://docs.astral.sh/uv/), GNU make):

```bash
make venv        # .venv with all dependencies
make test        # unit tests (no internet needed)
make notebook    # run the notebook, refresh figures/tables/results_macros.tex and save a PDF copy of it
make pdf         # build prospectus/GroupZ_Prospectus.pdf with tectonic
make check       # verify that every number in the PDF comes from the notebook register
make app         # start the Streamlit app on http://localhost:8501
```

**LaTeX in VS Code:** install the *LaTeX Workshop* extension and [tectonic](https://tectonic-typesetting.github.io)
(`brew install tectonic`). Open `prospectus/main.tex` and save it, or run *LaTeX Workshop: Build LaTeX project*. The recipe
in `.vscode/settings.json` uses tectonic. A latexmk recipe for MacTeX/TeX Live is included as an alternative.

## Deploying the app (Streamlit Community Cloud)

1. Go to <https://share.streamlit.io>, sign in with GitHub and choose *Create app*, then *Deploy a public app from GitHub*.
2. Repository `paul2108/04_Financial-Modelin-Group-projects`, branch `main`, main file `app/streamlit_app.py`.
3. *Advanced settings*: Python 3.12. App URL: `04financial-modelin-group-projects-u7jx8pavhcsfkqmkps5b5d` (assigned by Streamlit).
4. Dependencies come from `requirements.txt` in the repository root.

## Consistency between notebook, app and prospectus

* The notebook is the only place where calculations are written. `make core` copies its `core` cells to `app/strategy_core.py`,
  and `tests/test_consistency.py` fails if the copy is out of date.
* With default settings the app reproduces Table 1 of the prospectus (`pytest -m network` compares against `results/results.json`).
* The app's *Machine learning & robustness lab* switches HRP, Ledoit-Wolf, the averaged trend signal, the HMM overlay and
  leverage on and off with the notebook's own `core` functions, and shows the pre-registered gate verdict live. The
  classifier/FinBERT variants are too heavy to recompute, so the app shows the monthly returns the notebook exports to
  `results/variants_monthly.csv`. `pytest -m network` checks that the live switches reproduce those exports.
* The prospectus prints results only through `\res{key}`. The appendix table lists, for every value, the page where it is cited,
  the notebook section that computes it, the data source and the retrieval time stamp.

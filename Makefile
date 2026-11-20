# Group Z - Risk Parity + Trend. Run `make help` for an overview.
PY      := .venv/bin/python
NB      := notebook/GroupZ_RiskParity_Trend.ipynb
export MPLCONFIGDIR := $(CURDIR)/.venv/.mplconfig

.PHONY: help venv core test test-all notebook app pdf check all clean

help:            ## list the available commands
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

venv:            ## create .venv (Python 3.12) and install all dependencies
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements-dev.txt

core:            ## export the notebook's "core" cells to app/strategy_core.py
	$(PY) tools/export_core.py

test:            ## unit tests without internet access
	$(PY) -m pytest -m "not network"

test-all:        ## all tests incl. the app-vs-notebook check (needs Yahoo Finance)
	$(PY) -m pytest

notebook: core   ## execute the notebook top to bottom and refresh the prospectus exports
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=python3 $(NB)

app:             ## start the Streamlit app locally
	$(PY) -m streamlit run app/streamlit_app.py

pdf:             ## build the prospectus PDF with tectonic
	cd prospectus && tectonic -X compile --keep-logs main.tex && cp main.pdf GroupZ_Prospectus.pdf

check:           ## check that every number in the prospectus comes from the notebook register
	$(PY) tools/check_numbers.py

all: test notebook pdf check   ## full rebuild and verification

clean:           ## remove LaTeX and Python build files
	rm -rf prospectus/*.aux prospectus/*.log prospectus/*.out prospectus/*.synctex.gz .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

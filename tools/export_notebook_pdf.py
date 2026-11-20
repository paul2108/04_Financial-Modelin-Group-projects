"""Save a PDF copy of the executed notebook (for readers who cannot or do not want to re-run it).

The notebook must already contain its outputs (`make notebook` runs it first). The script
1. converts every HTML table in the outputs (pandas DataFrames and Stylers) into a LaTeX table,
2. removes elements that only make sense online (Colab badge, HTML anchors),
3. converts the notebook to LaTeX with nbconvert and compiles it with tectonic,
4. writes notebook/GroupZ_RiskParity_Trend.pdf.

Usage:
    python tools/export_notebook_pdf.py            # write the PDF
    python tools/export_notebook_pdf.py --dry-run  # only report what would be done
Environment: TECTONIC_BUNDLE (optional) = bundle URL or path passed to tectonic --bundle.
"""

import argparse
import copy
import os
import re
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import nbformat
import pandas as pd
from nbconvert import LatexExporter

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook" / "GroupZ_RiskParity_Trend.ipynb"
TARGET = ROOT / "notebook" / "GroupZ_RiskParity_Trend.pdf"


def html_table_to_latex(html: str):
    """LaTeX version of the first HTML table in an output, scaled to the page width (None if not possible)."""
    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, ImportError):
        return None
    if not tables:
        return None
    df = tables[0].fillna("")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(x) for x in col if not str(x).startswith("Unnamed")).strip() for col in df.columns]
    df.columns = ["" if str(c).startswith("Unnamed") else str(c) for c in df.columns]
    tex = df.to_latex(index=False, escape=True)
    return "\\begin{center}\\resizebox{\\linewidth}{!}{%\n" + tex + "}\\end{center}\n"


def prepare(nb):
    nb = copy.deepcopy(nb)
    n_tables = 0
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            src = re.sub(r"<a id=\"[^\"]*\"></a>\n?", "", cell.source)
            src = re.sub(r"\[!\[Open In Colab\][^\n]*\n?", "", src)
            cell.source = src
        elif cell.cell_type == "code":
            for out in cell.get("outputs", []):
                data = out.get("data", {})
                if "text/html" in data and "<table" in data["text/html"]:
                    tex = html_table_to_latex(data["text/html"])
                    if tex:
                        out["data"] = {"text/latex": tex}
                        n_tables += 1
    return nb, n_tables


def ensure_pandoc():
    """nbconvert needs pandoc for markdown cells; use the binary shipped with pypandoc_binary if none is installed."""
    if shutil.which("pandoc"):
        return
    import pypandoc  # noqa: PLC0415  (pypandoc_binary bundles a pandoc executable)
    os.environ["PATH"] = str(Path(pypandoc.get_pandoc_path()).parent) + os.pathsep + os.environ["PATH"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    nb = nbformat.read(NOTEBOOK, as_version=4)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    n_run = sum(c.cell_type == "code" and c.get("execution_count") for c in nb.cells)
    if n_run < n_code - 1:          # the Colab install cell may have no execution count
        print(f"Notebook has only {n_run} of {n_code} code cells executed - run `make notebook` first.")
        return 1
    nb, n_tables = prepare(nb)
    if args.dry_run:
        print(f"would convert {n_tables} HTML tables and write {TARGET.relative_to(ROOT)} (--dry-run: nothing written)")
        return 0

    ensure_pandoc()
    exporter = LatexExporter()
    body, resources = exporter.from_notebook_node(
        nb, resources={"metadata": {"name": "Risk Parity + Trend: backtest notebook (Group Z)"}})
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # recent pandoc versions write \def\LTcaptype{none} for markdown tables without caption
        body = body.replace("\\begin{document}",
                            "\\setcounter{secnumdepth}{0}\n\\makeatletter\\@ifundefined{c@none}{\\newcounter{none}}{}\\makeatother\n\\begin{document}", 1)
        (tmp / "notebook.tex").write_text(body, encoding="utf-8")
        for name, content in resources.get("outputs", {}).items():
            (tmp / name).write_bytes(content)
        cmd = ["tectonic", "-X", "compile", "notebook.tex"]
        if os.environ.get("TECTONIC_BUNDLE"):
            cmd[3:3] = ["--bundle", os.environ["TECTONIC_BUNDLE"]]
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
            return proc.returncode
        shutil.copy(tmp / "notebook.pdf", TARGET)
    print(f"wrote {TARGET.relative_to(ROOT)} ({TARGET.stat().st_size / 1e6:.1f} MB, {n_tables} tables converted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

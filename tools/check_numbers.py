"""Check that every result in the prospectus comes from the notebook's number register.

Checks:
  1. every \\res{key} / \\resmark{key} used in the .tex files is defined in results_macros.tex
  2. the prose contains no hand-typed result numbers (percentages or decimals outside \\res{})
  3. every register row has a notebook section, a link, a source and a retrieval date
  4. (if main.aux exists) every cited key has at least one page recorded, i.e. it appears in the appendix
Read-only: the script never changes files, so --dry-run only prints what would be checked.

Usage: python tools/check_numbers.py [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROSP = ROOT / "prospectus"

# numbers allowed in prose: years, section/figure refs, assumptions stated as parameters, list numbering
ALLOWED = {"60/40", "1/N", "7-10", "10-month", "252", "S&P 500", "13-week", "equities fall by", "AGG (US"}
NUM_PATTERN = re.compile(r"(?<![\w\\{=])[-$]*\d+[.,]\d+(?!\d|cm|pt|em|mm|in)\s*\\?%?|(?<![\w\\{])\d+\s*\\%")


def tex_files():
    files = [PROSP / "main.tex"] + sorted((PROSP / "sections").glob("*.tex"))
    return [f for f in files if f.exists()]


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list the files that would be checked and exit")
    args = ap.parse_args()
    files = tex_files()
    if args.dry_run:
        print("Would check:", *[f.relative_to(ROOT) for f in files], PROSP / "results_macros.tex", sep="\n  ")
        return 0

    macros = (PROSP / "results_macros.tex").read_text(encoding="utf-8")
    defs = {}
    for line in macros.splitlines():
        m = re.match(r"\\defres\{([^}]*)\}(.*)$", line)
        if m:
            defs[m.group(1)] = m.group(2)
    problems = []

    used = set()
    for f in files:
        text = strip_comments(f.read_text(encoding="utf-8"))
        for key in re.findall(r"\\res(?:mark)?\{([^}]*)\}", text):
            used.add(key)
            if key not in defs:
                problems.append(f"{f.name}: \\res{{{key}}} is not in the register")
        prose = re.sub(r"\\res(?:mark)?\{[^}]*\}", "", text)
        prose = re.sub(r"\\(cite[pt]?|ref|label|input|includegraphics|url|href|hyperlink)(\[[^]]*\])?\{[^}]*\}", "", prose)
        prose = re.sub(r"\$[^$]*\$", "", prose)                     # maths (formulas, not results)
        for m in NUM_PATTERN.finditer(prose):
            snippet = prose[max(0, m.start() - 30): m.end() + 10].replace("\n", " ")
            if not any(a in snippet for a in ALLOWED):
                problems.append(f"{f.name}: hand-typed number '{m.group(0).strip()}' in: ...{snippet}...")

    for key, rest in defs.items():
        fields = re.findall(r"\{((?:[^{}]|\{[^{}]*\})*)\}", rest)
        if len(fields) != 6 or not all(fields[i].strip() for i in (2, 3, 4, 5)):
            problems.append(f"register row '{key}' is incomplete: {fields}")

    aux = PROSP / "main.aux"
    if aux.exists():
        cited = set(re.findall(r"\\res@used\{([^}]*)\}", aux.read_text(encoding="utf-8")))
        for key in sorted(used - cited):
            problems.append(f"\\res{{{key}}} has no page in main.aux (rebuild the PDF)")

    print(f"{len(used)} keys cited, {len(defs)} keys in the register, {len(files)} .tex files checked")
    if problems:
        print("\n".join("PROBLEM: " + p for p in problems))
        return 1
    print("OK: all results in the prospectus come from the notebook register")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build the paper figures as Excel workbooks with native, editable charts; nothing trains.

Writes one .xlsx per figure plus tables.xlsx to ``--out-dir`` (default ``excel/``).
Reads ``--results-dir`` (default ``artifacts/``) and never writes under it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.excel_figures import make_all_workbooks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--out-dir", default=str(ROOT / "excel"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir == results_dir or results_dir in out_dir.parents:
        raise SystemExit(f"--out-dir {out_dir} is inside --results-dir {results_dir}; refusing to write there.")
    for path in make_all_workbooks(results_dir, out_dir, repo=ROOT):
        print(path.relative_to(ROOT) if ROOT in path.parents else path)


if __name__ == "__main__":
    main()

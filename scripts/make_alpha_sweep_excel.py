"""Build the alpha-sweep workbook (RMSE / skill vs alpha, descriptors, statistics); nothing trains.

Writes ``--out`` (default ``excel/alpha_sweep.xlsx``) from ``--results-dir`` (default
``artifacts/``: the sweep under ``alpha_sweep/``, draw 1 of alpha = 0.1 and 0.3 under
``relabel/``) and never writes under it. Runs not yet measured are listed in the notes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.alpha_sweep import make_workbook  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--out", default=str(ROOT / "excel" / "alpha_sweep.xlsx"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()
    out = Path(args.out).resolve()
    if results_dir in out.parents:
        raise SystemExit(f"--out {out} is inside --results-dir {results_dir}; refusing to write there.")
    print(make_workbook(results_dir, out, ROOT))


if __name__ == "__main__":
    main()

"""Relabel step 3: re-score saved S5 classification predictions under identities.

No training. Reads artifacts/s5 (read-only) and writes artifacts/relabel/rescore/:
local-only natural K=2, the cross-session control redefined by identity links
(configs/identity_links.yaml), the reverse-direction check, and the supplementary
reverse-direction check over every run of the other source (inference only).

  python scripts/run_relabel_rescore.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.identities import DEFAULT_LINKS_PATH  # noqa: E402
from redo_by_sara.relabel_rescore import DEFAULT_OUT, DEFAULT_S5_ROOT, SUMMARY_FILENAME, run_rescore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s5-root", default=str(DEFAULT_S5_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--links", default=str(DEFAULT_LINKS_PATH))
    args = parser.parse_args()
    run_rescore(Path(args.s5_root), Path(args.out), Path(args.links))
    print(f"Wrote {Path(args.out) / SUMMARY_FILENAME}", flush=True)


if __name__ == "__main__":
    main()

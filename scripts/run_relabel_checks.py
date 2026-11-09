"""Relabel step 2: prove what the identity relabel leaves unchanged (no training).

  python scripts/run_relabel_checks.py artifact      # relabelled vs canonical classification artifact
  python scripts/run_relabel_checks.py partitions    # K=2 / K=3 assignments vs S5; Dirichlet counts
  python scripts/run_relabel_checks.py compare --s5 artifacts/s5/fl/k2_natural/seed4601/regression \\
      --rerun <scratch>/k2_natural_seed4601_regression --name regression_bitwise_k2_natural_seed4601

The regression rerun itself is the ordinary runner, into scratch:

  python scripts/run_combined_iid_flower.py --config configs/relabel_k2_natural.yaml \\
      --task regression --output-root <scratch>/k2_natural_seed4601_regression

Each check writes artifacts/relabel/checks/<name>.json and exits 1 if it fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.relabel_checks import (  # noqa: E402
    CANONICAL_CLASSIFICATION_ARTIFACT,
    DIRICHLET_CONDITIONS,
    KEPT_ASSIGNMENT_CONDITIONS,
    REGRESSION_ARTIFACT,
    RELABEL_CLASSIFICATION_ARTIFACT,
    S5_ROOT,
    artifact_equality,
    compare_result_dirs,
    conditions_configs,
    identity_window_counts,
    load_artifact,
    partition_checks,
    write_check,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="check", required=True)
    sub.add_parser("artifact")
    sub.add_parser("partitions")
    compare = sub.add_parser("compare")
    compare.add_argument("--s5", required=True, help="Committed S5 job folder (read-only).")
    compare.add_argument("--rerun", required=True, help="Rerun output root (as passed to --output-root).")
    compare.add_argument("--name", required=True, help="Check name; writes checks/<name>.json.")
    args = parser.parse_args()

    if args.check == "artifact":
        relabelled = load_artifact(RELABEL_CLASSIFICATION_ARTIFACT)
        result = artifact_equality(relabelled, load_artifact(CANONICAL_CLASSIFICATION_ARTIFACT))
        result["identity_walking_windows_by_split"] = identity_window_counts(relabelled)
        result["passed"] = True  # artifact_equality raises on any other difference
        path = write_check("artifact_equality", result)
    elif args.check == "partitions":
        configs = conditions_configs(KEPT_ASSIGNMENT_CONDITIONS + DIRICHLET_CONDITIONS)
        result = partition_checks(load_artifact(RELABEL_CLASSIFICATION_ARTIFACT), load_artifact(REGRESSION_ARTIFACT),
                                  configs, S5_ROOT)
        path = write_check("partitions", result)
    else:
        result = compare_result_dirs(Path(args.s5), Path(args.rerun))
        path = write_check(args.name, result)
    print(json.dumps({"check": args.check, "passed": result["passed"], "written": str(path)}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

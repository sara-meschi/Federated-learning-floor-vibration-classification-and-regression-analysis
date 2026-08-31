from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.combined_iid_flower import (
    load_combined_iid_flower_config,
    prepare_combined_iid_flower_experiment,
    run_combined_iid_flower_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the leakage-safe, three-client Flower IID experiments on the "
            "combined Test_2 and 20251124_Testing artifacts."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--task", choices=("classification", "regression", "both"), default="both"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Testing override. Requires --output-root so production results are protected.",
    )
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config = load_combined_iid_flower_config(args.config)
    if args.verify_only:
        prepared = prepare_combined_iid_flower_experiment(config, args.output_root)
        public_setup = {
            key: value
            for key, value in prepared["setup"].items()
            if key != "partition"
        }
        public_setup["partition"] = {
            "num_training_runs": prepared["setup"]["partition"]["num_training_runs"],
            "num_test_runs": prepared["setup"]["partition"]["num_test_runs"],
            "clients": prepared["setup"]["partition"]["clients"],
            "audits": prepared["setup"]["partition"]["audits"],
        }
        print(json.dumps(public_setup, indent=2), flush=True)
        return

    tasks = ("classification", "regression") if args.task == "both" else (args.task,)
    summaries = {}
    for task in tasks:
        summaries[task] = run_combined_iid_flower_experiment(
            config,
            task,
            num_rounds_override=args.rounds,
            output_root=args.output_root,
        )
        print(json.dumps({task: summaries[task]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

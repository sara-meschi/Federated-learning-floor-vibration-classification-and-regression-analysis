from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.combined_residual_run_regression import (
    audit_residual_run_setup,
    load_residual_run_config,
    train_and_evaluate_residual_run_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the combined-data source/subject-residual CNN using whole-run "
            "aggregation and equal run weighting."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    config = load_residual_run_config(args.config)
    if not config["artifact_path"].exists():
        raise FileNotFoundError(config["artifact_path"])
    if not config["baseline_summary_path"].exists():
        raise FileNotFoundError(config["baseline_summary_path"])
    artifact = torch.load(
        config["artifact_path"], map_location="cpu", weights_only=False
    )
    setup_summary = audit_residual_run_setup(artifact)
    print(json.dumps(setup_summary, indent=2), flush=True)
    if args.verify_only:
        return

    summary = train_and_evaluate_residual_run_model(artifact, config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

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

from redo_by_sara.combined_centralized_regression import (
    build_regression_artifact,
    load_regression_config,
    save_regression_artifact,
    train_and_evaluate_regression,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated combined-data centralized walking-speed regression experiment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--rebuild-artifact", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    config = load_regression_config(args.config)
    output_dir: Path = config["output_dir"]
    artifact_path = output_dir / config["artifact_name"]
    artifact_summary_path = output_dir / "artifact_summary.json"
    manifest_path = output_dir / "run_manifest.csv"

    if args.rebuild_artifact or not artifact_path.exists():
        artifact = build_regression_artifact(config)
        save_regression_artifact(
            artifact, artifact_path, artifact_summary_path, manifest_path
        )
        print(json.dumps(artifact["summary"], indent=2), flush=True)
        print(f"Saved regression artifact to {artifact_path}", flush=True)
    else:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        print(f"Loaded existing regression artifact from {artifact_path}", flush=True)

    if args.preprocess_only:
        return
    summary = train_and_evaluate_regression(artifact, config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

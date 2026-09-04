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

from redo_by_sara.combined_centralized_classification import (
    build_combined_artifact,
    load_combined_config,
    save_combined_artifact,
    train_and_evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated combined-data centralized classification experiment."
    )
    parser.add_argument("--config", required=True, help="Path to the combined experiment YAML file.")
    parser.add_argument(
        "--rebuild-artifact",
        action="store_true",
        help="Rebuild the combined artifact even if it already exists.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Build and audit the artifact without starting training.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Redirect training outputs so a test run cannot overwrite live artifacts. "
            "The input artifact is still read from its canonical location, because it is "
            "an input rather than a result."
        ),
    )
    args = parser.parse_args()

    config = load_combined_config(args.config)
    # The artifact lives under the canonical output_dir and is an INPUT here. Redirecting
    # it along with the results would silently trigger a full rebuild from the 7.6 GB
    # TestData tree, so only the results move.
    canonical_dir: Path = config["output_dir"]
    artifact_path = canonical_dir / config["artifact_name"]
    artifact_summary_path = canonical_dir / "artifact_summary.json"
    manifest_path = canonical_dir / "run_manifest.csv"
    if args.output_root is not None:
        if args.rebuild_artifact:
            raise SystemExit(
                "--rebuild-artifact writes the canonical artifact and cannot be combined "
                "with --output-root. Rebuild first, then re-run with --output-root."
            )
        config["output_dir"] = Path(args.output_root).resolve()
    output_dir: Path = config["output_dir"]

    if args.rebuild_artifact or not artifact_path.exists():
        artifact = build_combined_artifact(config)
        save_combined_artifact(
            artifact=artifact,
            artifact_path=artifact_path,
            summary_path=artifact_summary_path,
            manifest_path=manifest_path,
        )
        print(json.dumps(artifact["summary"], indent=2), flush=True)
        print(f"Saved combined artifact to {artifact_path}", flush=True)
    else:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        print(f"Loaded existing combined artifact from {artifact_path}", flush=True)

    if args.preprocess_only:
        return

    summary = train_and_evaluate(artifact, config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

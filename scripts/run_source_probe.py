"""Source-classification probe (review plan §2 R4 item 2): the subject classifier's recipe with
label = source. Results go to --output-root only."""

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

from redo_by_sara.combined_centralized_classification import load_combined_config  # noqa: E402
from redo_by_sara.source_probe import run_source_probe  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/centralized_combined_9ch_no006_nowalking_80_20_60e.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Testing override of the pinned 60 epochs; the summary records it.",
    )
    args = parser.parse_args()

    config = load_combined_config(args.config)
    artifact_path = config["output_dir"] / config["artifact_name"]
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    if Path(args.output_root).resolve() == config["output_dir"].resolve():
        raise SystemExit("--output-root must not be the canonical classification directory.")
    if args.epochs is not None:
        config["epochs"] = args.epochs
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    summary = run_source_probe(config, artifact, args.output_root)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

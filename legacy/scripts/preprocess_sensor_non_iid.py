from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/scripts/preprocess_sensor_non_iid.py"
raise ImportError(
    "Quarantined legacy module: legacy/scripts/preprocess_sensor_non_iid.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.config import load_config
from redo_by_sara.sensor_non_iid import build_sensor_non_iid_artifact, save_sensor_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train/test raw-window artifact for sensor-owned non-IID FL."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    artifact = build_sensor_non_iid_artifact(config)
    save_sensor_artifact(artifact, config.artifact_path)
    print(json.dumps(artifact["summary"], indent=2))
    print(f"Saved artifact to {config.artifact_path}")


if __name__ == "__main__":
    main()

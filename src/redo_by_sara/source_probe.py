"""Source-classification probe (review plan §2 R4 item 2).

Same architecture, same windows, same seed-4601 run split, same train-only
normalization and the same centralized training recipe as the subject classifier; the
only change is the label: **source** (collection campaign) instead of subject. It
measures how separable the two campaigns are on their own. The two sources are the same
corridor 19.6 months apart, so anything that separates them is session nuisance: DAQ
trigger configuration, true sampling rate, and 19 months of drift.

The relabeled artifact exists only in memory; nothing canonical is written. Run-level
results aggregate **all** of a run's windows (no_walking included: ambient floor
vibration carries the session signature too), and window accuracy is also split by
activity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .combined_centralized_classification import train_and_evaluate
from .run_level_classification import PREDICTIONS_FILENAME, read_prediction_rows, run_level_for_result

SUMMARY_FILENAME = "source_probe_summary.json"
CHECKPOINT_NAME = "combined_centralized_cnn_final.pt"


def source_relabeled_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """A shallow copy of the classification artifact whose label is the source ID."""
    sources = sorted({item["source_id"] for item in artifact["metadata"]})
    class_to_index = {source: index for index, source in enumerate(sources)}
    targets = torch.tensor([class_to_index[item["source_id"]] for item in artifact["metadata"]], dtype=torch.long)
    return {
        **artifact,
        "classification_targets": targets,
        "class_to_index": class_to_index,
        "index_to_class": sources,
        "summary": {
            **artifact["summary"],
            "experiment": artifact["summary"]["experiment"] + "_source_probe",
            "class_names": sources,
            "label": "source_id (collection campaign), not subject",
        },
    }


def run_source_probe(config: dict[str, Any], artifact: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    probe = source_relabeled_artifact(artifact)
    training = train_and_evaluate(probe, {**config, "output_dir": output_dir})

    rows = read_prediction_rows(output_dir / PREDICTIONS_FILENAME)
    by_activity: dict[str, Any] = {}
    for activity in ("no_walking", "walking"):
        correct = [
            row["true_class"] == row["predicted_class"]
            for row in rows
            if (row["activity"] == "no_walking") == (activity == "no_walking")
        ]
        by_activity[activity] = {"num_windows": len(correct), "accuracy": float(np.mean(correct))}
    run_level = run_level_for_result(
        output_dir, CHECKPOINT_NAME, probe["samples"], output_dir / "run_level",
        exclude_true=(), batch_size=config["batch_size"],
    )
    summary = {
        "experiment": probe["summary"]["experiment"],
        "classes": probe["index_to_class"],
        "split_seed": config["split_seed"],
        "init_seed": config["init_seed"],
        "epochs": config["epochs"],
        "chance_accuracy_majority_class_test_windows": max(
            float(np.mean([row["true_class"] == source for row in rows])) for source in probe["index_to_class"]
        ),
        "test_window_accuracy": training["test_accuracy"],
        "test_window_balanced_accuracy": training["test_balanced_accuracy"],
        "test_window_accuracy_by_activity": by_activity,
        "run_level": {key: run_level[key] for key in ("majority_vote", "mean_probability", "per_subject_source")},
        "final_train_accuracy": training["final_train_accuracy"],
        "training_summary_path": str(output_dir / "training_summary.json"),
    }
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    return summary

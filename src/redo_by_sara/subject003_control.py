"""Subject-003 cross-session control (review plan §2 R4 item 3).

Leave-one-source-out: train a subject classifier on one source only and test it on the
same person recorded in the other source, 19.6 months later, after a DAQ configuration
change. If identity transfers, the model is reading gait; if not, it is reading session.

**This is a separate evaluation split under the CLAUDE.md carve-out.** It builds an
in-memory derived artifact and never touches ``assign_run_splits``, the seed-4601
split, or any saved artifact. The split, derived from the data (no subject ID is
hard-coded):

- **train**: walking windows of the training source's canonical *train* runs; the
  classes are the subjects found there;
- **in-session test**: walking windows of the training source's canonical *test* runs;
- **cross-session test**: every walking window, from any canonical split, of a training
  subject recorded in another source. The model never sees that source, so canonical
  train runs are valid test runs here;
- **negative control**: every walking window of the other source's remaining subjects,
  who are not among the classes. Their predicted-label distribution shows what the
  model does with a person it cannot know. If it labels them the way it labels the
  cross-session subject, the cross-session "hit" reflects the session, not the person.

Normalization statistics are recomputed from the control's train windows only. The
canonical statistics include the other source's windows, which would leak the
cross-session test session into training.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .combined_centralized_classification import train_and_evaluate
from .run_level_classification import (
    BACKGROUND_CLASSES,
    PREDICTIONS_FILENAME,
    aggregate_runs,
    assert_reproduces_predictions,
    load_classifier,
    read_prediction_rows,
    summarize_runs,
    window_probabilities,
)

SUMMARY_FILENAME = "subject003_control_summary.json"
CHECKPOINT_NAME = "combined_centralized_cnn_final.pt"
SETS = ("in_session", "cross_session", "negative_control")


def train_channel_statistics(samples: np.ndarray, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over train windows and time, as the artifact builder does."""
    train = samples[list(indices)]
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def cross_session_control_split(
    artifact: dict[str, Any], train_source: str, background: Sequence[str] = BACKGROUND_CLASSES
) -> dict[str, Any]:
    metadata = artifact["metadata"]
    names = artifact["index_to_class"]
    targets = artifact["classification_targets"].tolist()
    walking = [i for i in range(len(metadata)) if names[targets[i]] not in background]
    if train_source not in {metadata[i]["source_id"] for i in walking}:
        raise ValueError(f"Unknown train source {train_source!r}.")

    train = [i for i in walking if metadata[i]["source_id"] == train_source and metadata[i]["split"] == "train"]
    in_session = [i for i in walking if metadata[i]["source_id"] == train_source and metadata[i]["split"] == "test"]
    train_subjects = sorted({metadata[i]["subject_id"] for i in train})
    other = [i for i in walking if metadata[i]["source_id"] != train_source]
    cross = [i for i in other if metadata[i]["subject_id"] in train_subjects]
    negative = [i for i in other if metadata[i]["subject_id"] not in train_subjects]
    classes = sorted({names[targets[i]] for i in train})

    if not cross:
        raise AssertionError(f"No training subject of {train_source} appears in another source.")
    if {names[targets[i]] for i in in_session + cross} - set(classes):
        raise AssertionError("A test window's true class is not among the control's classes.")
    train_runs = {metadata[i]["run_uid"] for i in train}
    for label, indices in (("in_session", in_session), ("cross_session", cross), ("negative_control", negative)):
        shared = train_runs & {metadata[i]["run_uid"] for i in indices}
        if shared:
            raise AssertionError(f"{label} shares runs with the control's train set: {sorted(shared)[:3]}")
    if len(set(train) | set(in_session) | set(cross) | set(negative)) != len(train) + len(in_session) + len(cross) + len(negative):
        raise AssertionError("Control sets overlap.")
    return {
        "train_source": train_source,
        "classes": classes,
        "cross_session_subjects": sorted({metadata[i]["subject_id"] for i in cross}),
        "train": train,
        "in_session": in_session,
        "cross_session": cross,
        "negative_control": negative,
    }


def build_control_artifact(artifact: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    """In-memory artifact: train = control train, test = in-session + cross-session."""
    metadata = artifact["metadata"]
    names = artifact["index_to_class"]
    targets = artifact["classification_targets"].tolist()
    class_to_index = {name: index for index, name in enumerate(split["classes"])}
    members = [("train", "train", i) for i in split["train"]]
    members += [("test", label, i) for label in ("in_session", "cross_session") for i in split[label]]
    canonical = [i for _, _, i in members]
    samples = artifact["samples"][torch.as_tensor(canonical)]
    mean, std = train_channel_statistics(samples.numpy(), range(len(split["train"])))
    derived_metadata = [
        {**metadata[i], "split": role, "control_set": label, "canonical_split": metadata[i]["split"], "canonical_artifact_index": i}
        for role, label, i in members
    ]
    return {
        "samples": samples,
        "classification_targets": torch.tensor([class_to_index[names[targets[i]]] for i in canonical], dtype=torch.long),
        "train_indices": torch.arange(len(split["train"]), dtype=torch.long),
        "test_indices": torch.arange(len(split["train"]), len(canonical), dtype=torch.long),
        "channel_mean": torch.from_numpy(mean),
        "channel_std": torch.from_numpy(std),
        "class_to_index": class_to_index,
        "index_to_class": list(split["classes"]),
        "metadata": derived_metadata,
        "summary": {
            **artifact["summary"],
            "experiment": artifact["summary"]["experiment"] + f"_cross_session_control_train_{split['train_source']}",
            "class_names": list(split["classes"]),
        },
    }


def _distribution_rows(records: Sequence[dict[str, Any]], set_of_run: dict[str, str], classes: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (set_of_run[record["run_uid"]], record["source_id"], record["subject_id"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (label, source, subject), members in sorted(groups.items(), key=lambda item: (SETS.index(item[0][0]), item[0][1], item[0][2])):
        windows: Counter[str] = Counter()
        for record in members:
            windows.update(json.loads(record["window_label_counts"]))
        total = sum(windows.values())
        majority = Counter(record["majority_vote_label"] for record in members)
        mean_probability = Counter(record["mean_probability_label"] for record in members)
        row: dict[str, Any] = {"control_set": label, "source_id": source, "subject_id": subject,
                               "num_runs": len(members), "num_windows": total}
        for name in classes:
            row[f"window_fraction_{name}"] = windows[name] / total
        for name in classes:
            row[f"runs_majority_{name}"] = majority[name]
        for name in classes:
            row[f"runs_mean_probability_{name}"] = mean_probability[name]
        rows.append(row)
    return rows


def run_cross_session_control(
    config: dict[str, Any], artifact: dict[str, Any], output_dir: str | Path, train_source: str
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    split = cross_session_control_split(artifact, train_source)
    derived = build_control_artifact(artifact, split)
    training = train_and_evaluate(derived, {**config, "output_dir": output_dir})

    classifier = load_classifier(output_dir / CHECKPOINT_NAME)
    classes = classifier["class_names"]
    if classes != split["classes"]:
        raise AssertionError("Checkpoint classes differ from the control's classes.")
    rows = read_prediction_rows(output_dir / PREDICTIONS_FILENAME)
    probabilities = window_probabilities(classifier, derived["samples"], [int(r["artifact_index"]) for r in rows], config["batch_size"])
    assert_reproduces_predictions(rows, probabilities, classes, "cross-session control")
    records = aggregate_runs(rows, probabilities, classes, exclude_true=())

    metadata = artifact["metadata"]
    names = artifact["index_to_class"]
    targets = artifact["classification_targets"].tolist()
    negative_rows = [
        {"artifact_index": str(i), "run_uid": metadata[i]["run_uid"], "source_id": metadata[i]["source_id"],
         "subject_id": metadata[i]["subject_id"], "true_class": names[targets[i]]}
        for i in split["negative_control"]
    ]
    negative_probabilities = window_probabilities(classifier, artifact["samples"], split["negative_control"], config["batch_size"])
    negative_records = aggregate_runs(negative_rows, negative_probabilities, classes, exclude_true=())

    set_of_run = {item["run_uid"]: item["control_set"] for item in derived["metadata"]}
    set_of_run.update({record["run_uid"]: "negative_control" for record in negative_records})
    all_records = [{"control_set": set_of_run[r["run_uid"]], **r} for r in records + negative_records]
    distribution = _distribution_rows(records + negative_records, set_of_run, classes)

    def runs_of(label: str) -> list[dict[str, Any]]:
        return [r for r in records if set_of_run[r["run_uid"]] == label]

    def count_runs(indices: Sequence[int]) -> int:
        return len({metadata[i]["run_uid"] for i in indices})

    summary = {
        "experiment": derived["summary"]["experiment"],
        "carve_out": "separate evaluation split (CLAUDE.md working agreement); canonical split and artifacts untouched",
        "train_source": train_source,
        "classes": classes,
        "cross_session_subjects": split["cross_session_subjects"],
        "split_seed": config["split_seed"],
        "init_seed": config["init_seed"],
        "epochs": config["epochs"],
        "normalization": "recomputed from the control's train windows only",
        "set_sizes": {
            label: {"num_runs": count_runs(split[label]), "num_windows": len(split[label])}
            for label in ("train", *SETS)
        },
        "in_session": summarize_runs(runs_of("in_session")),
        "cross_session": summarize_runs(runs_of("cross_session")),
        "predicted_label_distribution": distribution,
        "final_train_accuracy": training["final_train_accuracy"],
        "pooled_test_window_accuracy_note": (
            "training_summary.json's test_accuracy pools in-session and cross-session windows; "
            "read the per-set results here instead."
        ),
        "training_summary_path": str(output_dir / "training_summary.json"),
    }
    with (output_dir / "run_level_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    with (output_dir / "predicted_label_distribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distribution[0]))
        writer.writeheader()
        writer.writerows(distribution)
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    return summary

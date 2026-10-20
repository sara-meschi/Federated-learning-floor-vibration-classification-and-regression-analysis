"""Run-level classification metrics (review plan §2 R4 item 1).

The held-out test set is 28 runs, not 307 independent windows: with 5 s windows at a
1 s stride, adjacent windows share 4 s of signal. So the primary classification result
aggregates each run's window predictions into one run prediction, two ways:

- **majority vote** over the window argmax labels. A tie goes to the tied label with the
  higher mean probability over the run's windows, then to the lower class index;
- **mean probability**: argmax of the run's mean softmax vector.

Each is reported as run-level accuracy with a **Wilson 95% interval**, plus macro-F1
over the labels that occur as true run labels, per source, and per subject x source.

Window probabilities are not stored by the training pipelines. They are recomputed
afterwards from each run's saved final checkpoint, on the same windows, in the same
order and batch size, with the normalization stored in that checkpoint, and every
window's recomputed argmax is asserted to equal the ``predicted_class`` the pipeline
wrote. A checkpoint that does not reproduce its own predictions is an error, not a
warning. No training code path is touched.

Windows whose true class is a background class (``no_walking``) carry no subject and
are excluded from subject aggregation. Their predictions still count if they land in a
subject run's windows: a run whose vote is won by ``no_walking`` is simply wrong. No
subject or source ID is hard-coded here; groups come from the prediction rows.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .combined_centralized_classification import CombinedArtifactDataset
from .models import SimpleCNN1D

#: Two-sided 95% normal quantile.
Z95 = 1.959963984540054
METHODS = ("majority_vote", "mean_probability")
BACKGROUND_CLASSES = ("no_walking",)
PREDICTIONS_FILENAME = "test_predictions.csv"
RUN_PREDICTIONS_FILENAME = "run_level_predictions.csv"
SUBJECT_SOURCE_FILENAME = "run_level_subject_source.csv"
METRICS_FILENAME = "run_level_metrics.json"


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when n is 0."""
    if n < 0 or not 0 <= successes <= n:
        raise ValueError(f"Need 0 <= successes <= n, got {successes}/{n}.")
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_classifier(checkpoint_path: str | Path) -> dict[str, Any]:
    """A saved SimpleCNN1D classifier with the class names and normalization it used."""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_to_index: dict[str, int] = saved["class_to_index"]
    class_names = [name for name, _ in sorted(class_to_index.items(), key=lambda item: item[1])]
    if [class_to_index[name] for name in class_names] != list(range(len(class_names))):
        raise AssertionError(f"{checkpoint_path}: class indices are not 0..C-1.")
    model = SimpleCNN1D(in_channels=int(saved["channel_mean"].shape[1]), output_dim=len(class_names))
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "class_names": class_names,
        "channel_mean": saved["channel_mean"],
        "channel_std": saved["channel_std"],
    }


def window_probabilities(
    classifier: dict[str, Any],
    samples: torch.Tensor,
    indices: Sequence[int],
    batch_size: int = 32,
) -> np.ndarray:
    """Softmax over classes for ``samples[indices]``, normalized as the model was trained.

    Goes through ``CombinedArtifactDataset`` so the normalization is the pipeline's own
    code. Evaluation is always unmasked: S5 evaluates on the global 9-channel test set.
    """
    view = {
        "samples": samples,
        "classification_targets": torch.zeros(len(samples), dtype=torch.long),
        "channel_mean": classifier["channel_mean"],
        "channel_std": classifier["channel_std"],
    }
    loader = DataLoader(
        CombinedArtifactDataset(view, list(indices)), batch_size=batch_size, shuffle=False, num_workers=0
    )
    batches = []
    with torch.no_grad():
        for batch, _ in loader:
            batches.append(torch.softmax(classifier["model"](batch), dim=1).double())
    return torch.cat(batches).numpy() if batches else np.zeros((0, len(classifier["class_names"])))


def read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def assert_reproduces_predictions(
    rows: Sequence[dict[str, str]], probabilities: np.ndarray, class_names: Sequence[str], label: str
) -> None:
    recomputed = [class_names[int(i)] for i in probabilities.argmax(axis=1)]
    mismatches = [
        (row["artifact_index"], row["predicted_class"], new)
        for row, new in zip(rows, recomputed)
        if row["predicted_class"] != new
    ]
    if len(rows) != len(recomputed) or mismatches:
        raise AssertionError(
            f"{label}: the saved checkpoint does not reproduce the saved window predictions "
            f"({len(mismatches)} of {len(rows)} differ; first: {mismatches[:3]})."
        )


def aggregate_runs(
    rows: Sequence[dict[str, str]],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    true_field: str = "true_class",
    exclude_true: Sequence[str] = BACKGROUND_CLASSES,
) -> list[dict[str, Any]]:
    """One record per run, from that run's windows whose true label is not excluded.

    ``rows`` are window prediction rows (``run_uid``, ``source_id``, ``subject_id`` and
    ``true_field``), aligned with ``probabilities``. A true label outside
    ``class_names`` is allowed (a negative control): the run still gets predictions,
    and its ``*_correct`` fields are ``None``.
    """
    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities must align one-to-one.")
    index_of = {name: i for i, name in enumerate(class_names)}
    by_run: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        if row[true_field] not in exclude_true:
            by_run[row["run_uid"]].append(position)

    records = []
    for run_uid, positions in by_run.items():
        first = rows[positions[0]]
        true_labels = {rows[p][true_field] for p in positions}
        if len(true_labels) != 1:
            raise AssertionError(f"{run_uid}: windows carry more than one true label {sorted(true_labels)}.")
        true_label = true_labels.pop()
        run_probabilities = probabilities[positions]
        mean_probability = run_probabilities.mean(axis=0)
        window_labels = [class_names[int(i)] for i in run_probabilities.argmax(axis=1)]
        votes = Counter(window_labels)
        top = max(votes.values())
        tied = [label for label, count in votes.items() if count == top]
        majority = min(tied, key=lambda label: (-mean_probability[index_of[label]], index_of[label]))
        mean_label = class_names[int(np.argmax(mean_probability))]
        known = true_label in index_of
        records.append(
            {
                "run_uid": run_uid,
                "source_id": first["source_id"],
                "subject_id": first["subject_id"],
                "true_label": true_label,
                "num_windows": len(positions),
                "window_correct": sum(label == true_label for label in window_labels),
                "majority_vote_label": majority,
                "majority_vote_count": top,
                "majority_vote_tied": len(tied) > 1,
                "mean_probability_label": mean_label,
                "mean_probability_max": float(mean_probability.max()),
                "mean_probability_of_true": float(mean_probability[index_of[true_label]]) if known else None,
                "window_label_counts": json.dumps(dict(sorted(votes.items()))),
                "majority_vote_correct": (majority == true_label) if known else None,
                "mean_probability_correct": (mean_label == true_label) if known else None,
            }
        )
    return records


def _macro_f1(true: Sequence[str], predicted: Sequence[str]) -> float:
    scores = []
    for label in sorted(set(true)):
        tp = sum(t == label and p == label for t, p in zip(true, predicted))
        fp = sum(t != label and p == label for t, p in zip(true, predicted))
        fn = sum(t == label and p != label for t, p in zip(true, predicted))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


def _accuracy_block(correct: Sequence[bool]) -> dict[str, Any]:
    n, k = len(correct), int(sum(correct))
    low, high = wilson_interval(k, n)
    return {"num_runs": n, "num_correct": k, "accuracy": k / n if n else float("nan"), "wilson95": [low, high]}


def summarize_runs(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Run-level accuracy (Wilson 95%), macro-F1, per source, per subject x source."""
    scored = [r for r in records if r["majority_vote_correct"] is not None]
    summary: dict[str, Any] = {
        "unit": "held-out run; windows with a background true class excluded",
        "interval": "Wilson score, 95%",
        "tie_rule": "majority-vote ties go to the higher run-mean probability, then the lower class index",
    }
    for method in METHODS:
        label_key, correct_key = f"{method}_label", f"{method}_correct"
        by_source: dict[str, list[bool]] = defaultdict(list)
        for r in scored:
            by_source[r["source_id"]].append(bool(r[correct_key]))
        summary[method] = {
            **_accuracy_block([bool(r[correct_key]) for r in scored]),
            "macro_f1": _macro_f1([r["true_label"] for r in scored], [r[label_key] for r in scored]),
            "per_source": {s: _accuracy_block(v) for s, v in sorted(by_source.items())},
        }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r["source_id"], r["subject_id"])].append(r)
    summary["per_subject_source"] = [subject_source_row(source, subject, rs) for (source, subject), rs in sorted(groups.items())]
    return summary


def subject_source_row(source: str, subject: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    windows = sum(r["num_windows"] for r in records)
    row: dict[str, Any] = {
        "source_id": source,
        "subject_id": subject,
        "num_runs": len(records),
        "num_windows": windows,
        "window_accuracy": sum(r["window_correct"] for r in records) / windows if windows else float("nan"),
        "majority_vote_labels": json.dumps(dict(sorted(Counter(r["majority_vote_label"] for r in records).items()))),
        "mean_probability_labels": json.dumps(dict(sorted(Counter(r["mean_probability_label"] for r in records).items()))),
    }
    for method in METHODS:
        values = [r[f"{method}_correct"] for r in records]
        row[f"{method}_correct"] = None if None in values else int(sum(values))
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_level_for_result(
    result_dir: str | Path,
    checkpoint_name: str,
    samples: torch.Tensor,
    out_dir: str | Path,
    *,
    true_field: str = "true_class",
    exclude_true: Sequence[str] = BACKGROUND_CLASSES,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run-level metrics for one finished classification result folder.

    Reads ``test_predictions.csv`` and the checkpoint from ``result_dir`` (read-only)
    and writes the three run-level files into ``out_dir``.
    """
    result_dir, out_dir = Path(result_dir), Path(out_dir)
    rows = read_prediction_rows(result_dir / PREDICTIONS_FILENAME)
    classifier = load_classifier(result_dir / checkpoint_name)
    probabilities = window_probabilities(classifier, samples, [int(r["artifact_index"]) for r in rows], batch_size)
    assert_reproduces_predictions(rows, probabilities, classifier["class_names"], str(result_dir))
    records = aggregate_runs(rows, probabilities, classifier["class_names"], true_field=true_field, exclude_true=exclude_true)
    summary = summarize_runs(records)
    summary.update(
        {
            "source_result_dir": str(result_dir),
            "checkpoint": checkpoint_name,
            "num_windows_scored": len(rows),
            "checkpoint_reproduces_saved_window_predictions": True,
            "window_accuracy_all_test_windows": float(np.mean([r["true_class"] == r["predicted_class"] for r in rows])),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / RUN_PREDICTIONS_FILENAME, records)
    _write_csv(out_dir / SUBJECT_SOURCE_FILENAME, summary["per_subject_source"])
    (out_dir / METRICS_FILENAME).write_text(json.dumps(summary, indent=2))
    return summary

"""Plots for the combined-data, three-client IID Flower experiments.

The functions in this module deliberately accept ordinary dictionaries/lists or
CSV/JSON paths.  Keeping plotting separate from Flower makes the result files
reproducible without starting another simulation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PathLike = str | Path
RowsInput = Sequence[Mapping[str, Any]] | PathLike
MappingInput = Mapping[str, Any] | PathLike
MatrixInput = Sequence[Sequence[int | float]] | np.ndarray | PathLike


CENTRAL_COLOR = "#dd6b20"
FL_COLOR = "#2b6cb0"
BASELINE_COLORS = ["#a0aec0", "#718096", "#4a5568"]


def _output_path(path: PathLike) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _load_rows(value: RowsInput) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        with path.open(newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        rows = [dict(row) for row in value]
    if not rows:
        raise ValueError("Plot input contains no rows.")
    return rows


def _load_mapping(value: MappingInput) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        payload = json.loads(Path(value).read_text())
    else:
        payload = dict(value)
    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object or mapping.")
    return payload


def _number(row: Mapping[str, Any], names: Sequence[str], context: str) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            value = float(row[name])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite {context} in column '{name}'.")
            return value
    raise KeyError(f"Missing {context}; expected one of {list(names)}.")


def _series(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str], context: str
) -> np.ndarray:
    return np.asarray([_number(row, names, context) for row in rows], dtype=np.float64)


def _save_close(
    figure: plt.Figure, output_path: PathLike, *, tight_layout: bool = True
) -> Path:
    path = _output_path(output_path)
    if tight_layout:
        figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def _display_class_name(name: str) -> str:
    if name == "no_walking":
        return "No walking"
    return name.replace("subject_", "S").replace("_walking", "")


def _load_matrix(
    value: MatrixInput, class_names: Sequence[str] | None
) -> tuple[np.ndarray, list[str]]:
    labels: list[str] | None = None
    if isinstance(value, (str, Path)):
        with Path(value).open(newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2 or len(rows[0]) < 2:
            raise ValueError(f"Invalid confusion-matrix CSV: {value}")
        labels = [str(item) for item in rows[0][1:]]
        row_labels = [str(row[0]) for row in rows[1:]]
        if row_labels != labels:
            raise ValueError(
                "Confusion-matrix CSV row labels do not match its column labels."
            )
        matrix = np.asarray(
            [[float(item) for item in row[1:]] for row in rows[1:]],
            dtype=np.float64,
        )
    else:
        matrix = np.asarray(value, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Confusion matrix must be square; got {matrix.shape}.")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Confusion matrix contains invalid counts.")
    if class_names is not None:
        supplied = [str(item) for item in class_names]
        if labels is not None and supplied != labels:
            raise ValueError("Supplied class names disagree with the matrix CSV labels.")
        labels = supplied
    if labels is None:
        labels = [str(index) for index in range(matrix.shape[0])]
    if len(labels) != matrix.shape[0]:
        raise ValueError("Number of class names does not match confusion-matrix size.")
    return matrix, labels


def _draw_confusion(
    axis: plt.Axes,
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    *,
    vmax: float,
) -> Any:
    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=max(vmax, 1.0),
    )
    display = [_display_class_name(item) for item in labels]
    axis.set_xticks(range(len(display)), display, rotation=45, ha="right")
    axis.set_yticks(range(len(display)), display)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    threshold = vmax / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = str(int(round(value))) if np.isclose(value, round(value)) else f"{value:.2f}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    return image


def plot_classification_training_comparison(
    fl_history: RowsInput,
    centralized_history: RowsInput,
    output_path: PathLike,
) -> Path:
    """Compare FL post-round global-train metrics with centralized training.

    FL metrics are expected to be a global-model evaluation over the union of
    the three client training partitions after aggregation, not local minibatch
    metrics.  The legend and title make this semantic difference explicit.
    """

    fl_rows = _load_rows(fl_history)
    central_rows = _load_rows(centralized_history)
    fl_x = _series(fl_rows, ["round", "server_round"], "FL round")
    central_x = _series(central_rows, ["epoch"], "centralized epoch")
    fl_loss = _series(
        fl_rows,
        ["train_loss", "global_train_loss", "global_train_cross_entropy"],
        "FL train loss",
    )
    fl_accuracy = _series(
        fl_rows,
        ["train_accuracy", "global_train_accuracy", "accuracy"],
        "FL train accuracy",
    )
    central_loss = _series(central_rows, ["train_loss"], "centralized train loss")
    central_accuracy = _series(
        central_rows, ["train_accuracy"], "centralized train accuracy"
    )

    figure, axes = plt.subplots(2, 1, figsize=(9.5, 8), sharex=False)
    axes[0].plot(
        central_x,
        central_loss,
        color=CENTRAL_COLOR,
        linewidth=2,
        label="Centralized (within-epoch train metric)",
    )
    axes[0].plot(
        fl_x,
        fl_loss,
        color=FL_COLOR,
        linewidth=2,
        label="Flower FL (post-round global model on client-train data)",
    )
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Classification Training: Centralized Epochs vs FL Rounds")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        central_x,
        100.0 * central_accuracy,
        color=CENTRAL_COLOR,
        linewidth=2,
        label="Centralized (within-epoch train metric)",
    )
    axes[1].plot(
        fl_x,
        100.0 * fl_accuracy,
        color=FL_COLOR,
        linewidth=2,
        label="Flower FL (post-round global model on client-train data)",
    )
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Optimization epoch / federated round")
    axes[1].set_ylim(0.0, 101.0)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    return _save_close(figure, output_path)


def plot_classification_confusion(
    matrix: MatrixInput,
    class_names: Sequence[str] | None,
    output_path: PathLike,
    *,
    title: str = "Flower IID Global Model: Final Held-out Test Confusion Matrix",
) -> Path:
    values, labels = _load_matrix(matrix, class_names)
    figure, axis = plt.subplots(figsize=(9, 7.5))
    image = _draw_confusion(axis, values, labels, title, vmax=float(values.max()))
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save_close(figure, output_path)


def plot_classification_confusion_comparison(
    centralized_matrix: MatrixInput,
    fl_matrix: MatrixInput,
    class_names: Sequence[str] | None,
    output_path: PathLike,
) -> Path:
    """Plot the two test matrices with identical labels and color limits."""

    central, central_labels = _load_matrix(centralized_matrix, class_names)
    fl, fl_labels = _load_matrix(fl_matrix, class_names)
    if central_labels != fl_labels or central.shape != fl.shape:
        raise ValueError("Centralized and FL confusion matrices are not label-compatible.")
    vmax = float(max(central.max(), fl.max()))
    figure, axes = plt.subplots(1, 2, figsize=(17, 7.5), sharex=True, sharey=True)
    image = _draw_confusion(
        axes[0], central, central_labels, "Centralized CNN", vmax=vmax
    )
    _draw_confusion(axes[1], fl, fl_labels, "Flower IID global CNN", vmax=vmax)
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.025, label="Test windows")
    figure.suptitle("Final Models on the Same Held-out Test Windows (Common Scale)")
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.16, top=0.88, wspace=0.18)
    return _save_close(figure, output_path, tight_layout=False)


def _classification_test_metrics(summary: MappingInput) -> dict[str, float]:
    payload = _load_mapping(summary)
    candidates = [payload]
    for key in ("test_metrics", "final_test_metrics", "classification_test_metrics"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            candidates.insert(0, nested)

    aliases = {
        "Accuracy": ("accuracy", "test_accuracy"),
        "Balanced accuracy": ("balanced_accuracy", "test_balanced_accuracy"),
        "Macro F1": ("macro_f1", "test_macro_f1"),
    }
    result: dict[str, float] = {}
    for label, names in aliases.items():
        found: float | None = None
        for candidate in candidates:
            for name in names:
                if name in candidate:
                    found = float(candidate[name])
                    break
            if found is not None:
                break
        if found is None or not np.isfinite(found) or not 0.0 <= found <= 1.0:
            raise ValueError(f"Missing or invalid classification test metric: {label}.")
        result[label] = found
    return result


def plot_classification_test_metric_comparison(
    centralized_summary: MappingInput,
    fl_summary: MappingInput,
    output_path: PathLike,
) -> Path:
    central = _classification_test_metrics(centralized_summary)
    fl = _classification_test_metrics(fl_summary)
    labels = list(central)
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.5))
    central_bars = axis.bar(
        x - width / 2,
        [100.0 * central[label] for label in labels],
        width,
        color=CENTRAL_COLOR,
        label="Centralized CNN",
    )
    fl_bars = axis.bar(
        x + width / 2,
        [100.0 * fl[label] for label in labels],
        width,
        color=FL_COLOR,
        label="Flower IID global CNN",
    )
    for bars in (central_bars, fl_bars):
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Held-out test score (%)")
    axis.set_ylim(0.0, 105.0)
    axis.set_title("Classification on the Same Held-out Test Windows")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    return _save_close(figure, output_path)


def plot_regression_training_comparison(
    fl_history: RowsInput,
    centralized_history: RowsInput,
    output_path: PathLike,
) -> Path:
    """Compare physical run metrics after each FL round and central epoch."""

    fl_rows = _load_rows(fl_history)
    central_rows = _load_rows(centralized_history)
    fl_x = _series(fl_rows, ["round", "server_round"], "FL round")
    central_x = _series(central_rows, ["epoch"], "centralized epoch")
    columns = [
        (
            [
                "global_train_mse_mps2",
                "train_run_loss_mse",
                "global_train_run_mse",
                "train_run_mse",
                "mse",
            ],
            ["train_run_loss_mse"],
            "Run MSE (m²/s²)",
        ),
        (
            [
                "global_train_rmse_mps",
                "train_run_rmse_mps",
                "global_train_run_rmse_mps",
                "rmse_mps",
            ],
            ["train_run_rmse_mps"],
            "Run RMSE (m/s)",
        ),
        (
            ["global_train_r2", "train_run_r2", "global_train_run_r2", "r2"],
            ["train_run_r2"],
            "Run R²",
        ),
        (
            [
                "train_run_within_0_10_mps_pct",
                "global_train_within_0_10_mps_pct",
                "global_train_run_within_0_10_mps_pct",
                "within_0_10_mps_pct",
            ],
            ["train_run_within_0_10_mps_pct"],
            "Within ±0.10 m/s (%)",
        ),
    ]
    figure, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)
    for axis, (fl_names, central_names, ylabel) in zip(axes, columns):
        axis.plot(
            central_x,
            _series(central_rows, central_names, f"centralized {ylabel}"),
            color=CENTRAL_COLOR,
            linewidth=2,
            label="Central residual CNN (post-epoch train eval)",
        )
        axis.plot(
            fl_x,
            _series(fl_rows, fl_names, f"FL {ylabel}"),
            color=FL_COLOR,
            linewidth=2,
            label="Flower IID global model (post-round client-train eval)",
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].set_title("Run-balanced Residual Regression: Centralized vs Flower IID")
    axes[0].legend(fontsize=8)
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[-1].set_xlabel("Optimization epoch / federated round")
    return _save_close(figure, output_path)


def _prediction_rows(value: RowsInput, label: str) -> list[dict[str, Any]]:
    rows = _load_rows(value)
    seen: set[str] = set()
    for row in rows:
        if "run_uid" not in row or not str(row["run_uid"]):
            raise KeyError(f"{label} prediction row is missing run_uid.")
        run_uid = str(row["run_uid"])
        if run_uid in seen:
            raise ValueError(f"{label} contains duplicate run_uid {run_uid!r}.")
        seen.add(run_uid)
        _number(row, ["actual_speed_mps"], f"{label} actual speed")
        _number(row, ["predicted_speed_mps"], f"{label} predicted speed")
    return rows


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Actual and predicted speeds must be nonempty and aligned.")
    residual = predicted - actual
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mse": mse,
        "rmse_mps": float(np.sqrt(mse)),
        "r2": float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 1e-12
        else 0.0,
        "within_0_10_mps_pct": float(100.0 * np.mean(np.abs(residual) <= 0.10)),
    }


def _joined_run_predictions(
    centralized_predictions: RowsInput,
    fl_predictions: RowsInput,
    *,
    actual_tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    central_rows = _prediction_rows(centralized_predictions, "Centralized")
    fl_rows = _prediction_rows(fl_predictions, "FL")
    central_by_uid = {str(row["run_uid"]): row for row in central_rows}
    fl_by_uid = {str(row["run_uid"]): row for row in fl_rows}
    if set(central_by_uid) != set(fl_by_uid):
        central_only = sorted(set(central_by_uid) - set(fl_by_uid))
        fl_only = sorted(set(fl_by_uid) - set(central_by_uid))
        raise AssertionError(
            "Centralized and FL results do not contain the same held-out runs: "
            f"centralized_only={central_only}, fl_only={fl_only}."
        )

    joined: list[dict[str, Any]] = []
    for run_uid in sorted(central_by_uid):
        central = central_by_uid[run_uid]
        fl = fl_by_uid[run_uid]
        central_actual = _number(
            central, ["actual_speed_mps"], "centralized actual speed"
        )
        fl_actual = _number(fl, ["actual_speed_mps"], "FL actual speed")
        if not np.isclose(
            central_actual, fl_actual, rtol=0.0, atol=float(actual_tolerance)
        ):
            raise AssertionError(
                f"Actual speed differs for {run_uid}: "
                f"centralized={central_actual}, FL={fl_actual}."
            )
        for field in ("source_id", "subject_id", "direction"):
            if field in central and field in fl and str(central[field]) != str(fl[field]):
                raise AssertionError(f"{field} differs between methods for {run_uid}.")
        joined.append(
            {
                "run_uid": run_uid,
                "source_id": str(fl.get("source_id", central.get("source_id", "unknown"))),
                "subject_id": str(fl.get("subject_id", central.get("subject_id", "unknown"))),
                "actual_speed_mps": central_actual,
                "centralized_predicted_speed_mps": _number(
                    central, ["predicted_speed_mps"], "centralized predicted speed"
                ),
                "fl_predicted_speed_mps": _number(
                    fl, ["predicted_speed_mps"], "FL predicted speed"
                ),
            }
        )
    return joined


def plot_regression_actual_vs_predicted(
    fl_predictions: RowsInput,
    output_path: PathLike,
) -> Path:
    rows = _prediction_rows(fl_predictions, "FL")
    actual = _series(rows, ["actual_speed_mps"], "actual speed")
    predicted = _series(rows, ["predicted_speed_mps"], "predicted speed")
    metrics = _regression_metrics(actual, predicted)
    subjects = sorted({str(row.get("subject_id", "unknown")) for row in rows})
    sources = sorted({str(row.get("source_id", "unknown")) for row in rows})
    colors = plt.get_cmap("tab10")
    marker_options = ["o", "^", "s", "D", "P", "X"]
    markers = {
        source: marker_options[index % len(marker_options)]
        for index, source in enumerate(sources)
    }

    figure, axis = plt.subplots(figsize=(8, 7))
    for subject_index, subject in enumerate(subjects):
        for source in sources:
            selected = [
                row
                for row in rows
                if str(row.get("subject_id", "unknown")) == subject
                and str(row.get("source_id", "unknown")) == source
            ]
            if not selected:
                continue
            axis.scatter(
                [float(row["actual_speed_mps"]) for row in selected],
                [float(row["predicted_speed_mps"]) for row in selected],
                s=60,
                alpha=0.85,
                color=colors(subject_index),
                marker=markers[source],
                edgecolors="white",
                linewidths=0.5,
            )
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    padding = max(0.03, 0.08 * (high - low))
    bounds = [low - padding, high + padding]
    axis.plot(bounds, bounds, "k--", linewidth=1.2, label="Ideal prediction")
    axis.set_xlim(bounds)
    axis.set_ylim(bounds)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Actual run speed (m/s)")
    axis.set_ylabel("Predicted run speed (m/s)")
    axis.set_title(
        "Flower IID Residual CNN: Final Held-out Runs\n"
        f"RMSE={metrics['rmse_mps']:.4f} m/s, R²={metrics['r2']:.3f}"
    )
    axis.grid(alpha=0.25)
    subject_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors(index),
            markeredgecolor="white",
            label=f"Subject {subject}",
            markersize=7,
        )
        for index, subject in enumerate(subjects)
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[source],
            linestyle="none",
            color="black",
            label=source,
            markersize=7,
        )
        for source in sources
    ]
    subject_legend = axis.legend(handles=subject_handles, loc="upper left", fontsize=8)
    axis.add_artist(subject_legend)
    axis.legend(handles=source_handles, loc="lower right", fontsize=8, title="Dataset")
    return _save_close(figure, output_path)


def _nested_metric(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> float:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            result = float(value)
            if np.isfinite(result):
                return result
    raise KeyError(f"Could not find metric at any of: {list(paths)}")


def plot_regression_rmse_comparison(
    baseline_summary: MappingInput,
    centralized_residual_summary: MappingInput,
    fl_summary: MappingInput,
    output_path: PathLike,
) -> Path:
    """Add the FL result to the existing same-28-run RMSE comparison."""

    baseline = _load_mapping(baseline_summary)
    central = _load_mapping(centralized_residual_summary)
    fl = _load_mapping(fl_summary)
    baseline_root = baseline["train_run_mean_baselines"]["run"]
    labels = [
        "Overall\nmean",
        "Subject\nmean",
        "Source + subject\nmean",
        "Original CNN\nwindow loss",
        "Central residual CNN\nrun loss",
        "FL residual CNN\nrun loss",
    ]
    values = [
        float(baseline_root["overall_train_run_mean"]["rmse_mps"]),
        float(baseline_root["subject_train_run_mean"]["rmse_mps"]),
        float(baseline_root["source_subject_train_run_mean"]["rmse_mps"]),
        _nested_metric(baseline, [["test_run_metrics", "rmse_mps"]]),
        _nested_metric(central, [["test_run_metrics", "rmse_mps"], ["rmse_mps"]]),
        _nested_metric(
            fl,
            [
                ["test_run_metrics", "rmse_mps"],
                ["final_test_metrics", "rmse_mps"],
                ["rmse_mps"],
            ],
        ),
    ]
    if not np.isfinite(values).all() or np.any(np.asarray(values) < 0):
        raise ValueError("RMSE comparison contains invalid values.")
    colors = [*BASELINE_COLORS, "#805ad5", CENTRAL_COLOR, FL_COLOR]
    figure, axis = plt.subplots(figsize=(12, 6))
    bars = axis.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_ylim(0.0, max(values) * 1.16)
    axis.set_ylabel("Held-out run RMSE (m/s); lower is better")
    axis.set_title("Walking-Speed Regression on the Same 28 Held-out Runs")
    axis.grid(axis="y", alpha=0.25)
    return _save_close(figure, output_path)


def plot_regression_subject_rmse_comparison(
    centralized_predictions: RowsInput,
    fl_predictions: RowsInput,
    output_path: PathLike,
    *,
    actual_tolerance: float = 1e-6,
) -> Path:
    joined = _joined_run_predictions(
        centralized_predictions,
        fl_predictions,
        actual_tolerance=actual_tolerance,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[str(row["subject_id"])].append(row)
    subjects = sorted(grouped)
    central_values: list[float] = []
    fl_values: list[float] = []
    for subject in subjects:
        rows = grouped[subject]
        actual = np.asarray([row["actual_speed_mps"] for row in rows])
        central_values.append(
            _regression_metrics(
                actual,
                np.asarray([row["centralized_predicted_speed_mps"] for row in rows]),
            )["rmse_mps"]
        )
        fl_values.append(
            _regression_metrics(
                actual,
                np.asarray([row["fl_predicted_speed_mps"] for row in rows]),
            )["rmse_mps"]
        )

    x = np.arange(len(subjects))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 5.8))
    central_bars = axis.bar(
        x - width / 2,
        central_values,
        width,
        color=CENTRAL_COLOR,
        label="Central residual CNN",
    )
    fl_bars = axis.bar(
        x + width / 2,
        fl_values,
        width,
        color=FL_COLOR,
        label="Flower IID residual CNN",
    )
    for bars in (central_bars, fl_bars):
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=45)
    axis.set_xticks(x, [f"S{subject}" for subject in subjects])
    axis.set_ylabel("Held-out run RMSE (m/s); lower is better")
    axis.set_title("Per-subject RMSE on Identical Held-out Runs")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    return _save_close(figure, output_path)


def plot_regression_paired_run_absolute_errors(
    centralized_predictions: RowsInput,
    fl_predictions: RowsInput,
    output_path: PathLike,
    *,
    actual_tolerance: float = 1e-6,
) -> Path:
    """Dumbbell plot of paired absolute errors after strict run/test matching."""

    joined = _joined_run_predictions(
        centralized_predictions,
        fl_predictions,
        actual_tolerance=actual_tolerance,
    )
    joined.sort(key=lambda row: (str(row["subject_id"]), str(row["run_uid"])))
    central_error = np.asarray(
        [
            abs(row["centralized_predicted_speed_mps"] - row["actual_speed_mps"])
            for row in joined
        ]
    )
    fl_error = np.asarray(
        [abs(row["fl_predicted_speed_mps"] - row["actual_speed_mps"]) for row in joined]
    )
    x = np.arange(1, len(joined) + 1)
    figure, axis = plt.subplots(figsize=(13, 6))
    for index in range(len(joined)):
        better_color = "#38a169" if fl_error[index] < central_error[index] else "#a0aec0"
        axis.plot(
            [x[index], x[index]],
            [central_error[index], fl_error[index]],
            color=better_color,
            linewidth=1.1,
            alpha=0.75,
            zorder=1,
        )
    axis.scatter(
        x,
        central_error,
        color=CENTRAL_COLOR,
        marker="o",
        s=32,
        label="Central residual CNN",
        zorder=2,
    )
    axis.scatter(
        x,
        fl_error,
        color=FL_COLOR,
        marker="^",
        s=35,
        label="Flower IID residual CNN",
        zorder=3,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"S{row['subject_id']}\n{row['run_uid'].split(':')[-1]}" for row in joined],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    improved = int(np.sum(fl_error < central_error))
    tied = int(np.sum(np.isclose(fl_error, central_error, atol=1e-12, rtol=0.0)))
    axis.set_xlabel("Held-out run (subject and run index)")
    axis.set_ylabel("Absolute speed error (m/s); lower is better")
    axis.set_title(
        f"Paired Errors on the Same {len(joined)} Runs: FL Lower on {improved}"
        + (f", Tied on {tied}" if tied else "")
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    return _save_close(figure, output_path)


def _client_summaries(summary: MappingInput) -> list[dict[str, Any]]:
    payload = _load_mapping(summary)
    candidates: Any = payload.get("clients", payload.get("client_summaries"))
    if candidates is None and "partitions" in payload:
        candidates = payload["partitions"]
    if isinstance(candidates, Mapping):
        rows = []
        for client_id, values in sorted(candidates.items()):
            if not isinstance(values, Mapping):
                raise TypeError("Each client summary must be a mapping.")
            rows.append({"client_id": client_id, **dict(values)})
    elif isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        rows = [dict(row) for row in candidates]
    else:
        raise KeyError(
            "Partition summary must contain 'clients', 'client_summaries', or 'partitions'."
        )
    if not rows:
        raise ValueError("Partition summary contains no clients.")
    return rows


def _count(
    row: Mapping[str, Any], aliases: Sequence[str], *, default: float | None = None
) -> float:
    for name in aliases:
        if name in row and row[name] not in (None, ""):
            value = row[name]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return float(len(value))
            return float(value)
    if default is not None:
        return default
    raise KeyError(f"Client summary is missing all count fields {list(aliases)}.")


def _counts_mapping(row: Mapping[str, Any], aliases: Sequence[str]) -> dict[str, float]:
    for name in aliases:
        value = row.get(name)
        if isinstance(value, Mapping):
            return {str(key): float(count) for key, count in value.items()}
    return {}


def plot_partition_balance(summary: MappingInput, output_path: PathLike) -> Path:
    """Plot per-client runs/windows plus optional source/subject composition."""

    rows = _client_summaries(summary)
    client_ids = [str(row.get("client_id", f"client_{index}")) for index, row in enumerate(rows)]
    runs = np.asarray(
        [
            _count(row, ["num_runs", "train_runs", "run_count", "run_uids"])
            for row in rows
        ]
    )
    classification = np.asarray(
        [
            _count(
                row,
                [
                    "num_classification_windows",
                    "classification_windows",
                    "classification_count",
                    "num_classification_examples",
                    "classification_indices",
                ],
            )
            for row in rows
        ]
    )
    regression = np.asarray(
        [
            _count(
                row,
                [
                    "num_regression_windows",
                    "regression_windows",
                    "regression_count",
                    "num_regression_examples",
                    "regression_indices",
                ],
            )
            for row in rows
        ]
    )
    subject_counts = [
        _counts_mapping(
            row,
            ["runs_by_subject", "subject_run_counts", "run_counts_by_subject"],
        )
        for row in rows
    ]
    source_counts = [
        _counts_mapping(
            row, ["runs_by_source", "source_run_counts", "run_counts_by_source"]
        )
        for row in rows
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(rows))
    run_bars = axes[0, 0].bar(x, runs, color="#4a5568")
    axes[0, 0].bar_label(run_bars, fmt="%.0f", padding=2)
    axes[0, 0].set_xticks(x, client_ids)
    axes[0, 0].set_ylabel("Unique training runs")
    axes[0, 0].set_title("Atomic Run Allocation")

    width = 0.36
    class_bars = axes[0, 1].bar(
        x - width / 2,
        classification,
        width,
        color="#805ad5",
        label="Classification (walking + no walking)",
    )
    reg_bars = axes[0, 1].bar(
        x + width / 2,
        regression,
        width,
        color="#2f855a",
        label="Regression (walking only)",
    )
    axes[0, 1].bar_label(class_bars, fmt="%.0f", padding=2, fontsize=8)
    axes[0, 1].bar_label(reg_bars, fmt="%.0f", padding=2, fontsize=8)
    axes[0, 1].set_xticks(x, client_ids)
    axes[0, 1].set_ylabel("Training windows")
    axes[0, 1].set_title("Task-specific Window Counts")
    axes[0, 1].legend(fontsize=8)

    subjects = sorted({key for counts in subject_counts for key in counts})
    if subjects:
        values = np.asarray(
            [[counts.get(subject, 0.0) for subject in subjects] for counts in subject_counts]
        )
        image = axes[1, 0].imshow(values, cmap="Blues", aspect="auto", vmin=0)
        axes[1, 0].set_xticks(
            range(len(subjects)), [f"S{subject}" for subject in subjects], rotation=45
        )
        axes[1, 0].set_yticks(range(len(client_ids)), client_ids)
        axes[1, 0].set_title("Unique Runs by Subject")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                axes[1, 0].text(
                    column_index,
                    row_index,
                    f"{values[row_index, column_index]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(
            0.5,
            0.5,
            "Subject composition not included\nin partition summary",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )

    sources = sorted({key for counts in source_counts for key in counts})
    if sources:
        bottom = np.zeros(len(rows), dtype=np.float64)
        source_colors = plt.get_cmap("Set2")
        for source_index, source in enumerate(sources):
            values = np.asarray([counts.get(source, 0.0) for counts in source_counts])
            axes[1, 1].bar(
                x,
                values,
                bottom=bottom,
                color=source_colors(source_index),
                label=source,
            )
            bottom += values
        axes[1, 1].set_xticks(x, client_ids)
        axes[1, 1].set_ylabel("Unique training runs")
        axes[1, 1].set_title("Dataset-source Composition")
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "Source composition not included\nin partition summary",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    for axis in axes.flat:
        if axis.axison:
            axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Three-client Approximate-IID Training Partition Audit")
    return _save_close(figure, output_path)


__all__ = [
    "plot_classification_confusion",
    "plot_classification_confusion_comparison",
    "plot_classification_test_metric_comparison",
    "plot_classification_training_comparison",
    "plot_partition_balance",
    "plot_regression_actual_vs_predicted",
    "plot_regression_paired_run_absolute_errors",
    "plot_regression_rmse_comparison",
    "plot_regression_subject_rmse_comparison",
    "plot_regression_training_comparison",
]

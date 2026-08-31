from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .combined_centralized_regression import (
    load_regression_config,
    regression_metrics,
)
from .models import SimpleCNN1D


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_residual_run_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    payload: dict[str, Any] = yaml.safe_load(config_path.read_text())
    base_config_path = _resolve_path(project_root, payload["base_experiment_config"])
    base_config = load_regression_config(base_config_path)
    training = payload["training"]

    if int(payload["seed"]) != base_config["seed"]:
        raise ValueError("Residual and baseline experiments must use the same seed.")
    if int(training["epochs"]) != base_config["epochs"]:
        raise ValueError("Residual and baseline experiments must use the same epoch count.")
    if int(training["epochs"]) != 60:
        raise ValueError("This comparison is fixed to 60 optimization epochs.")

    return {
        "config_path": config_path,
        "project_root": project_root,
        "base_config_path": base_config_path,
        "artifact_path": _resolve_path(project_root, payload["artifact_path"]),
        "baseline_summary_path": _resolve_path(
            project_root, payload["baseline_summary_path"]
        ),
        "output_dir": _resolve_path(project_root, payload["output_dir"]),
        "seed": int(payload["seed"]),
        "run_batch_size": int(training["run_batch_size"]),
        "epochs": int(training["epochs"]),
        "learning_rate": float(training["learning_rate"]),
        "minimum_learning_rate": float(training["minimum_learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "num_workers": int(training.get("num_workers", 0)),
        "selected_channels": base_config["selected_channels"],
    }


def validate_regression_artifact(artifact: dict[str, object]) -> None:
    summary: dict[str, Any] = artifact["summary"]
    expected = {
        "num_usable_runs": 140,
        "num_train_runs": 112,
        "num_test_runs": 28,
        "num_examples": 1204,
        "num_train": 966,
        "num_test": 238,
        "sample_shape": [9, 2000],
        "train_test_run_overlap": 0,
    }
    for key, expected_value in expected.items():
        if summary[key] != expected_value:
            raise AssertionError(
                f"Unexpected source regression artifact {key}: "
                f"{summary[key]} != {expected_value}."
            )

    samples: torch.Tensor = artifact["samples"]
    targets: torch.Tensor = artifact["regression_targets"]
    if tuple(samples.shape) != (1204, 9, 2000):
        raise AssertionError(f"Unexpected sample tensor shape: {tuple(samples.shape)}")
    if tuple(targets.shape) != (1204,):
        raise AssertionError(f"Unexpected target tensor shape: {tuple(targets.shape)}")
    if not torch.isfinite(samples).all() or not torch.isfinite(targets).all():
        raise ValueError("Regression artifact contains non-finite samples or targets.")
    if not torch.isfinite(artifact["channel_mean"]).all():
        raise ValueError("Regression artifact contains a non-finite channel mean.")
    if not torch.isfinite(artifact["channel_std"]).all():
        raise ValueError("Regression artifact contains a non-finite channel standard deviation.")
    if not torch.all(artifact["channel_std"] > 0):
        raise ValueError("Regression artifact channel standard deviations must be positive.")


def _group_run_records(
    artifact: dict[str, object], indices: Sequence[int]
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = artifact["metadata"]
    grouped: dict[str, dict[str, Any]] = {}
    for artifact_index in indices:
        item = metadata[int(artifact_index)]
        run_uid = str(item["run_uid"])
        if run_uid not in grouped:
            grouped[run_uid] = {
                "run_uid": run_uid,
                "source_id": str(item["source_id"]),
                "subject_id": str(item["subject_id"]),
                "run_index": int(item["run_index"]),
                "direction": str(item["direction"]),
                "split": str(item["split"]),
                "actual_speed_mps": float(item["speed_mps"]),
                "window_indices": [],
            }
        run = grouped[run_uid]
        if not np.isclose(run["actual_speed_mps"], float(item["speed_mps"])):
            raise AssertionError(f"Run {run_uid} has inconsistent window targets.")
        if run["split"] != str(item["split"]):
            raise AssertionError(f"Run {run_uid} appears in multiple splits.")
        run["window_indices"].append(int(artifact_index))
    return [grouped[run_uid] for run_uid in sorted(grouped)]


def compute_train_source_subject_means(
    artifact: dict[str, object],
) -> tuple[dict[tuple[str, str], float], float, list[dict[str, Any]]]:
    train_indices: list[int] = artifact["train_indices"].tolist()
    test_indices: list[int] = artifact["test_indices"].tolist()
    train_runs = _group_run_records(artifact, train_indices)
    test_runs = _group_run_records(artifact, test_indices)
    values_by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in train_runs:
        key = (run["source_id"], run["subject_id"])
        values_by_group[key].append(float(run["actual_speed_mps"]))

    means = {key: float(np.mean(values)) for key, values in values_by_group.items()}
    test_groups = {(run["source_id"], run["subject_id"]) for run in test_runs}
    if not test_groups <= set(means):
        raise AssertionError(
            f"Test source/subject groups missing training means: {sorted(test_groups - set(means))}"
        )

    all_residuals: list[float] = []
    rows: list[dict[str, Any]] = []
    for (source_id, subject_id), values in sorted(values_by_group.items()):
        residuals = np.asarray(values) - means[(source_id, subject_id)]
        all_residuals.extend(float(value) for value in residuals)
        rows.append(
            {
                "source_id": source_id,
                "subject_id": subject_id,
                "num_train_runs": len(values),
                "train_mean_speed_mps": means[(source_id, subject_id)],
                "train_residual_mean_mps": float(residuals.mean()),
                "train_residual_std_mps": float(residuals.std()),
                "train_min_speed_mps": float(np.min(values)),
                "train_max_speed_mps": float(np.max(values)),
            }
        )
    residual_scale_mps = float(np.sqrt(np.mean(np.square(all_residuals))))
    if not np.isfinite(residual_scale_mps) or residual_scale_mps <= 0:
        raise ValueError(f"Invalid train residual scale: {residual_scale_mps}")
    return means, residual_scale_mps, rows


class RunGroupedResidualDataset(Dataset):
    def __init__(
        self,
        artifact: dict[str, object],
        indices: Sequence[int],
        source_subject_means: dict[tuple[str, str], float],
        residual_scale_mps: float,
    ) -> None:
        self.samples: torch.Tensor = artifact["samples"]
        self.channel_mean: torch.Tensor = artifact["channel_mean"]
        self.channel_std: torch.Tensor = artifact["channel_std"]
        self.runs = _group_run_records(artifact, indices)
        self.source_subject_means = source_subject_means
        self.residual_scale_mps = float(residual_scale_mps)

    def __len__(self) -> int:
        return len(self.runs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        run = self.runs[item]
        window_indices = torch.tensor(run["window_indices"], dtype=torch.long)
        samples = self.samples[window_indices]
        samples = (samples - self.channel_mean) / self.channel_std
        group_key = (run["source_id"], run["subject_id"])
        baseline = self.source_subject_means[group_key]
        actual = float(run["actual_speed_mps"])
        return {
            "samples": samples.float(),
            "window_indices": window_indices,
            "run_uid": run["run_uid"],
            "source_id": run["source_id"],
            "subject_id": run["subject_id"],
            "run_index": run["run_index"],
            "direction": run["direction"],
            "baseline_speed_mps": torch.tensor(baseline, dtype=torch.float32),
            "actual_speed_mps": torch.tensor(actual, dtype=torch.float32),
            "residual_scale_mps": torch.tensor(
                self.residual_scale_mps, dtype=torch.float32
            ),
            "target_residual_standardized": torch.tensor(
                (actual - baseline) / self.residual_scale_mps, dtype=torch.float32
            ),
        }


def collate_whole_runs(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty run batch.")
    window_counts = torch.tensor(
        [int(item["samples"].shape[0]) for item in items], dtype=torch.long
    )
    if torch.any(window_counts <= 0):
        raise ValueError("Every run batch item must contain at least one window.")
    run_assignment = torch.repeat_interleave(
        torch.arange(len(items), dtype=torch.long), window_counts
    )
    return {
        "samples": torch.cat([item["samples"] for item in items], dim=0),
        "window_indices": torch.cat([item["window_indices"] for item in items]),
        "window_counts": window_counts,
        "run_assignment": run_assignment,
        "run_uid": [str(item["run_uid"]) for item in items],
        "source_id": [str(item["source_id"]) for item in items],
        "subject_id": [str(item["subject_id"]) for item in items],
        "run_index": [int(item["run_index"]) for item in items],
        "direction": [str(item["direction"]) for item in items],
        "baseline_speed_mps": torch.stack(
            [item["baseline_speed_mps"] for item in items]
        ),
        "actual_speed_mps": torch.stack([item["actual_speed_mps"] for item in items]),
        "residual_scale_mps": torch.stack(
            [item["residual_scale_mps"] for item in items]
        ),
        "target_residual_standardized": torch.stack(
            [item["target_residual_standardized"] for item in items]
        ),
    }


def aggregate_windows_by_run(
    window_predictions: torch.Tensor,
    run_assignment: torch.Tensor,
    num_runs: int,
) -> torch.Tensor:
    if window_predictions.ndim != 1:
        raise ValueError("Window predictions must be a one-dimensional tensor.")
    if run_assignment.shape != window_predictions.shape:
        raise ValueError("Run assignments and window predictions must have matching shapes.")
    sums = window_predictions.new_zeros(num_runs)
    sums.index_add_(0, run_assignment, window_predictions)
    counts = torch.bincount(run_assignment, minlength=num_runs).to(
        device=window_predictions.device, dtype=window_predictions.dtype
    )
    if torch.any(counts == 0):
        raise ValueError("Every run must receive at least one window prediction.")
    return sums / counts


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_zero_initialized_residual_model() -> SimpleCNN1D:
    model = SimpleCNN1D(in_channels=9, output_dim=1)
    final_layer = model.head[-1]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError("Expected the SimpleCNN1D head to end with a linear layer.")
    nn.init.zeros_(final_layer.weight)
    nn.init.zeros_(final_layer.bias)
    return model


def _run_group_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    collect_windows: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_runs = 0
    run_actual: list[torch.Tensor] = []
    run_predicted: list[torch.Tensor] = []
    run_actual_residual: list[torch.Tensor] = []
    run_predicted_residual: list[torch.Tensor] = []
    run_uids: list[str] = []
    window_indices: list[torch.Tensor] = []
    window_predicted: list[torch.Tensor] = []
    window_predicted_residual: list[torch.Tensor] = []

    for batch in loader:
        samples = batch["samples"].to(device)
        assignment = batch["run_assignment"].to(device)
        target_residual_standardized = batch["target_residual_standardized"].to(device)
        residual_scale = batch["residual_scale_mps"].to(device)
        baseline = batch["baseline_speed_mps"].to(device)
        actual = batch["actual_speed_mps"].to(device)
        num_runs = int(actual.shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predicted_window_residual_standardized = model(samples).squeeze(-1)
            predicted_run_residual_standardized = aggregate_windows_by_run(
                predicted_window_residual_standardized, assignment, num_runs
            )
            loss = criterion(
                predicted_run_residual_standardized, target_residual_standardized
            )
            if training:
                loss.backward()
                optimizer.step()

        predicted_run_residual = residual_scale * predicted_run_residual_standardized
        actual_run_residual = residual_scale * target_residual_standardized
        predicted_run_speed = baseline + predicted_run_residual
        total_loss += float(loss.item()) * num_runs
        total_runs += num_runs
        run_actual.append(actual.detach().cpu())
        run_predicted.append(predicted_run_speed.detach().cpu())
        run_actual_residual.append(actual_run_residual.detach().cpu())
        run_predicted_residual.append(predicted_run_residual.detach().cpu())
        run_uids.extend(batch["run_uid"])

        if collect_windows:
            predicted_window_residual = (
                residual_scale[assignment] * predicted_window_residual_standardized
            )
            predicted_window_speed = baseline[assignment] + predicted_window_residual
            window_indices.append(batch["window_indices"].detach().cpu())
            window_predicted.append(predicted_window_speed.detach().cpu())
            window_predicted_residual.append(predicted_window_residual.detach().cpu())

    result: dict[str, Any] = {
        "loss": total_loss / max(total_runs, 1),
        "run_actual": torch.cat(run_actual).numpy(),
        "run_predicted": torch.cat(run_predicted).numpy(),
        "run_actual_residual": torch.cat(run_actual_residual).numpy(),
        "run_predicted_residual": torch.cat(run_predicted_residual).numpy(),
        "run_uids": run_uids,
    }
    if collect_windows:
        result.update(
            {
                "window_indices": torch.cat(window_indices).numpy(),
                "window_predicted": torch.cat(window_predicted).numpy(),
                "window_predicted_residual": torch.cat(
                    window_predicted_residual
                ).numpy(),
            }
        )
    return result


def _write_rows(rows: Sequence[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _extended_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    metrics = regression_metrics(actual, predicted)
    residual = np.asarray(predicted, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    metrics["median_absolute_error_mps"] = float(np.median(np.abs(residual)))
    return metrics


def _metrics_by_field(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        key: _extended_metrics(
            np.asarray([float(row["actual_speed_mps"]) for row in group_rows]),
            np.asarray([float(row["predicted_speed_mps"]) for row in group_rows]),
        )
        for key, group_rows in sorted(grouped.items())
    }


def _plot_epoch_metrics(history: Sequence[dict[str, Any]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    axes[0].plot(
        epochs, [row["train_run_loss_mse"] for row in history], color="#c43c39", linewidth=2
    )
    axes[0].set_ylabel("Run MSE loss")
    axes[0].set_title("Residual, Run-Balanced 1D CNN Training")
    axes[1].plot(
        epochs, [row["train_run_rmse_mps"] for row in history], color="#2b6cb0", linewidth=2
    )
    axes[1].set_ylabel("Run RMSE (m/s)")
    axes[2].plot(
        epochs, [row["train_run_r2"] for row in history], color="#2f855a", linewidth=2
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Run R²")
    axes[3].plot(
        epochs,
        [row["train_run_within_0_10_mps_pct"] for row in history],
        color="#805ad5",
        linewidth=2,
    )
    axes[3].set_ylabel("Within ±0.10 m/s (%)")
    axes[3].set_xlabel("Optimization epoch (0 = subject/session mean)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_actual_vs_predicted(
    run_rows: Sequence[dict[str, Any]], metrics: dict[str, float], output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    subjects = sorted({str(row["subject_id"]) for row in run_rows})
    sources = sorted({str(row["source_id"]) for row in run_rows})
    colors = plt.get_cmap("tab10")
    marker_options = ["o", "^"]
    markers = {
        source_id: marker_options[index]
        for index, source_id in enumerate(sources)
    }
    all_actual: list[float] = []
    all_predicted: list[float] = []
    for color_index, subject_id in enumerate(subjects):
        for source_id in sources:
            selected = [
                row
                for row in run_rows
                if str(row["subject_id"]) == subject_id
                and str(row["source_id"]) == source_id
            ]
            if not selected:
                continue
            actual = [float(row["actual_speed_mps"]) for row in selected]
            predicted = [float(row["predicted_speed_mps"]) for row in selected]
            all_actual.extend(actual)
            all_predicted.extend(predicted)
            axis.scatter(
                actual,
                predicted,
                s=60,
                alpha=0.85,
                color=colors(color_index),
                marker=markers[source_id],
                edgecolors="white",
                linewidths=0.5,
            )

    low = min(all_actual + all_predicted)
    high = max(all_actual + all_predicted)
    padding = max(0.03, 0.08 * (high - low))
    axis.plot(
        [low - padding, high + padding],
        [low - padding, high + padding],
        "k--",
        linewidth=1.2,
    )
    axis.set_xlim(low - padding, high + padding)
    axis.set_ylim(low - padding, high + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Actual run speed (m/s)")
    axis.set_ylabel("Predicted run speed (m/s)")
    axis.set_title(
        f"Residual Run Model: RMSE={metrics['rmse_mps']:.3f} m/s, "
        f"R²={metrics['r2']:.3f}"
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
            label=f"Subject {subject_id}",
            markersize=7,
        )
        for index, subject_id in enumerate(subjects)
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[source_id],
            linestyle="none",
            color="black",
            label=source_id,
            markersize=7,
        )
        for source_id in sources
    ]
    subject_legend = axis.legend(handles=subject_handles, loc="upper left", fontsize=8)
    axis.add_artist(subject_legend)
    axis.legend(handles=source_handles, loc="lower right", fontsize=8, title="Dataset")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_rmse_comparison(
    existing_summary: dict[str, Any],
    residual_metrics: dict[str, float],
    output_path: Path,
) -> None:
    run_baselines = existing_summary["train_run_mean_baselines"]["run"]
    labels = [
        "Overall\nmean",
        "Subject\nmean",
        "Source + subject\nmean",
        "Original CNN\nwindow loss",
        "Residual CNN\nrun loss",
    ]
    values = [
        run_baselines["overall_train_run_mean"]["rmse_mps"],
        run_baselines["subject_train_run_mean"]["rmse_mps"],
        run_baselines["source_subject_train_run_mean"]["rmse_mps"],
        existing_summary["test_run_metrics"]["rmse_mps"],
        residual_metrics["rmse_mps"],
    ]
    colors = ["#a0aec0", "#718096", "#4a5568", "#dd6b20", "#2b6cb0"]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_ylabel("Held-out run RMSE (m/s); lower is better")
    axis.set_title("Walking-Speed Regression Comparison on the Same 28 Test Runs")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def audit_residual_run_setup(artifact: dict[str, object]) -> dict[str, Any]:
    validate_regression_artifact(artifact)
    means, residual_scale_mps, mean_rows = compute_train_source_subject_means(artifact)
    train_indices: list[int] = artifact["train_indices"].tolist()
    test_indices: list[int] = artifact["test_indices"].tolist()
    train_runs = _group_run_records(artifact, train_indices)
    test_runs = _group_run_records(artifact, test_indices)
    train_uids = {run["run_uid"] for run in train_runs}
    test_uids = {run["run_uid"] for run in test_runs}
    train_residuals = np.asarray(
        [
            run["actual_speed_mps"] - means[(run["source_id"], run["subject_id"])]
            for run in train_runs
        ]
    )
    group_residual_means = {
        f"{row['source_id']}:{row['subject_id']}": row["train_residual_mean_mps"]
        for row in mean_rows
    }
    summary = {
        "num_source_subject_strata": len(means),
        "num_train_runs": len(train_runs),
        "num_test_runs": len(test_runs),
        "num_train_windows": sum(len(run["window_indices"]) for run in train_runs),
        "num_test_windows": sum(len(run["window_indices"]) for run in test_runs),
        "train_test_run_overlap": len(train_uids & test_uids),
        "minimum_windows_per_train_run": min(
            len(run["window_indices"]) for run in train_runs
        ),
        "maximum_windows_per_train_run": max(
            len(run["window_indices"]) for run in train_runs
        ),
        "train_residual_mean_mps": float(train_residuals.mean()),
        "train_residual_std_mps": float(train_residuals.std()),
        "train_residual_scale_mps": residual_scale_mps,
        "maximum_absolute_group_residual_mean_mps": float(
            max(abs(value) for value in group_residual_means.values())
        ),
        "all_test_strata_have_train_means": True,
        "loss_weighting": "One MSE term per run after averaging its window predictions.",
        "group_residual_means_mps": group_residual_means,
    }
    expected = {
        "num_source_subject_strata": 8,
        "num_train_runs": 112,
        "num_test_runs": 28,
        "num_train_windows": 966,
        "num_test_windows": 238,
        "train_test_run_overlap": 0,
    }
    for key, expected_value in expected.items():
        if summary[key] != expected_value:
            raise AssertionError(
                f"Unexpected residual setup {key}: {summary[key]} != {expected_value}."
            )
    if summary["maximum_absolute_group_residual_mean_mps"] > 1e-12:
        raise AssertionError("Train residuals are not centered within each source/subject stratum.")
    return summary


def _comparison(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {
        "candidate_rmse_mps": candidate["rmse_mps"],
        "reference_rmse_mps": reference["rmse_mps"],
        "rmse_improvement_mps": reference["rmse_mps"] - candidate["rmse_mps"],
        "rmse_improvement_pct": float(
            100.0 * (reference["rmse_mps"] - candidate["rmse_mps"])
            / reference["rmse_mps"]
        ),
        "candidate_r2": candidate["r2"],
        "reference_r2": reference["r2"],
        "r2_improvement": candidate["r2"] - reference["r2"],
    }


def train_and_evaluate_residual_run_model(
    artifact: dict[str, object], config: dict[str, Any]
) -> dict[str, Any]:
    setup_summary = audit_residual_run_setup(artifact)
    source_subject_means, residual_scale_mps, mean_rows = (
        compute_train_source_subject_means(artifact)
    )
    train_indices: list[int] = artifact["train_indices"].tolist()
    test_indices: list[int] = artifact["test_indices"].tolist()
    train_dataset = RunGroupedResidualDataset(
        artifact, train_indices, source_subject_means, residual_scale_mps
    )
    test_dataset = RunGroupedResidualDataset(
        artifact, test_indices, source_subject_means, residual_scale_mps
    )

    _set_random_seeds(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["run_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        generator=generator,
        collate_fn=collate_whole_runs,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config["run_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_whole_runs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["run_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_whole_runs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_zero_initialized_residual_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["minimum_learning_rate"],
    )

    def history_row(epoch: int, learning_rate: float, result: dict[str, Any]) -> dict[str, Any]:
        metrics = _extended_metrics(result["run_actual"], result["run_predicted"])
        return {
            "epoch": epoch,
            "optimization_loss_standardized_mse": result["loss"],
            "train_run_loss_mse": metrics["mse"],
            "train_run_rmse_mps": metrics["rmse_mps"],
            "train_run_mae_mps": metrics["mae_mps"],
            "train_run_bias_mps": metrics["bias_mps"],
            "train_run_r2": metrics["r2"],
            "train_run_within_0_10_mps_pct": metrics["within_0_10_mps_pct"],
            "learning_rate": learning_rate,
        }

    initial_result = _run_group_epoch(
        model, train_eval_loader, criterion, device, optimizer=None
    )
    history: list[dict[str, Any]] = [
        history_row(0, config["learning_rate"], initial_result)
    ]
    print(
        f"Epoch 000/{config['epochs']}: source/session mean baseline, "
        f"rmse={history[0]['train_run_rmse_mps']:.4f} m/s, "
        f"r2={history[0]['train_run_r2']:.4f}",
        flush=True,
    )

    for epoch in range(1, config["epochs"] + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        _run_group_epoch(model, train_loader, criterion, device, optimizer)
        train_result = _run_group_epoch(
            model, train_eval_loader, criterion, device, optimizer=None
        )
        row = history_row(epoch, learning_rate, train_result)
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{config['epochs']}: "
            f"run_loss={row['train_run_loss_mse']:.6f}, "
            f"rmse={row['train_run_rmse_mps']:.4f} m/s, "
            f"r2={row['train_run_r2']:.4f}, "
            f"within_0.10={row['train_run_within_0_10_mps_pct']:.1f}%, "
            f"lr={learning_rate:.8f}",
            flush=True,
        )
        scheduler.step()

    test_result = _run_group_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        collect_windows=True,
    )
    run_metrics = _extended_metrics(
        test_result["run_actual"], test_result["run_predicted"]
    )

    test_run_by_uid = {run["run_uid"]: run for run in test_dataset.runs}
    run_rows: list[dict[str, Any]] = []
    for run_uid, actual, predicted, actual_residual, predicted_residual in zip(
        test_result["run_uids"],
        test_result["run_actual"],
        test_result["run_predicted"],
        test_result["run_actual_residual"],
        test_result["run_predicted_residual"],
    ):
        run = test_run_by_uid[run_uid]
        baseline = source_subject_means[(run["source_id"], run["subject_id"])]
        run_rows.append(
            {
                "source_id": run["source_id"],
                "subject_id": run["subject_id"],
                "run_index": run["run_index"],
                "run_uid": run_uid,
                "direction": run["direction"],
                "num_windows": len(run["window_indices"]),
                "source_subject_train_mean_mps": baseline,
                "actual_residual_mps": float(actual_residual),
                "predicted_residual_mps": float(predicted_residual),
                "actual_speed_mps": float(actual),
                "predicted_speed_mps": float(predicted),
                "error_mps": float(predicted - actual),
                "absolute_error_mps": float(abs(predicted - actual)),
            }
        )

    metadata: list[dict[str, Any]] = artifact["metadata"]
    targets: torch.Tensor = artifact["regression_targets"]
    window_rows: list[dict[str, Any]] = []
    for artifact_index, predicted, predicted_residual in zip(
        test_result["window_indices"],
        test_result["window_predicted"],
        test_result["window_predicted_residual"],
    ):
        index = int(artifact_index)
        item = metadata[index]
        baseline = source_subject_means[(item["source_id"], item["subject_id"])]
        actual = float(targets[index])
        window_rows.append(
            {
                "artifact_index": index,
                "source_id": item["source_id"],
                "subject_id": item["subject_id"],
                "run_index": item["run_index"],
                "run_uid": item["run_uid"],
                "direction": item["direction"],
                "start_time": item["start_time"],
                "source_subject_train_mean_mps": baseline,
                "predicted_residual_mps": float(predicted_residual),
                "actual_speed_mps": actual,
                "predicted_speed_mps": float(predicted),
                "error_mps": float(predicted - actual),
                "absolute_error_mps": float(abs(predicted - actual)),
            }
        )
    window_rows.sort(key=lambda row: int(row["artifact_index"]))
    window_metrics = _extended_metrics(
        np.asarray([row["actual_speed_mps"] for row in window_rows]),
        np.asarray([row["predicted_speed_mps"] for row in window_rows]),
    )
    run_baseline_metrics = _extended_metrics(
        np.asarray([row["actual_speed_mps"] for row in run_rows]),
        np.asarray([row["source_subject_train_mean_mps"] for row in run_rows]),
    )
    window_baseline_metrics = _extended_metrics(
        np.asarray([row["actual_speed_mps"] for row in window_rows]),
        np.asarray(
            [row["source_subject_train_mean_mps"] for row in window_rows]
        ),
    )

    existing_summary = json.loads(config["baseline_summary_path"].read_text())
    existing_cnn_metrics = existing_summary["test_run_metrics"]
    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "residual_run_balanced_1d_cnn_final.pt"
    summary_path = output_dir / "training_summary.json"
    history_path = output_dir / "training_history.csv"
    mean_path = output_dir / "train_source_subject_means.csv"
    run_predictions_path = output_dir / "test_run_predictions.csv"
    window_predictions_path = output_dir / "test_window_predictions.csv"
    epoch_plot_path = output_dir / "run_loss_rmse_r2_accuracy_vs_epoch.png"
    prediction_plot_path = output_dir / "actual_vs_predicted_speed.png"
    comparison_plot_path = output_dir / "heldout_run_rmse_comparison.png"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "SimpleCNN1D(9 input channels, 1 residual output)",
            "selected_channels_one_based": config["selected_channels"],
            "channel_mean": artifact["channel_mean"],
            "channel_std": artifact["channel_std"],
            "source_subject_train_means": mean_rows,
            "train_residual_scale_mps": residual_scale_mps,
            "aggregation": "Arithmetic mean of window residual predictions per run",
            "target": (
                "(APDM gait speed minus train source/subject run mean) divided by "
                "the train-only residual scale"
            ),
            "seed": config["seed"],
            "epochs": config["epochs"],
        },
        model_path,
    )
    _write_rows(history, history_path)
    _write_rows(mean_rows, mean_path)
    _write_rows(run_rows, run_predictions_path)
    _write_rows(window_rows, window_predictions_path)
    _plot_epoch_metrics(history, epoch_plot_path)
    _plot_actual_vs_predicted(run_rows, run_metrics, prediction_plot_path)
    _plot_rmse_comparison(existing_summary, run_metrics, comparison_plot_path)

    summary = {
        "experiment": "combined_residual_run_level_run_balanced_1d_cnn",
        "method": {
            "architecture": "Unchanged SimpleCNN1D with a scalar residual output",
            "baseline": "Mean speed of unique training runs in the same source/subject stratum",
            "prediction": "source_subject_train_mean + predicted_residual",
            "train_residual_scale_mps": residual_scale_mps,
            "aggregation": "Mean residual prediction over every walking window in a run",
            "optimization_unit": "run",
            "run_balance": "Every training run appears exactly once per epoch and contributes one MSE term",
            "final_layer_initialization": "zeros, so epoch 0 exactly reproduces the source/subject mean",
        },
        "setup_audit": setup_summary,
        "device": str(device),
        "epochs": config["epochs"],
        "run_batch_size": config["run_batch_size"],
        "initial_learning_rate": config["learning_rate"],
        "minimum_learning_rate": config["minimum_learning_rate"],
        "weight_decay": config["weight_decay"],
        "test_evaluation_policy": (
            "Same held-out runs as the original CNN; this residual model evaluated them once "
            "after epoch 60. No validation set. Because prior test results motivated "
            "this design, the comparison is exploratory rather than a pristine final test."
        ),
        "epoch_0_train_source_subject_baseline_metrics": {
            "mse": history[0]["train_run_loss_mse"],
            "rmse_mps": history[0]["train_run_rmse_mps"],
            "mae_mps": history[0]["train_run_mae_mps"],
            "bias_mps": history[0]["train_run_bias_mps"],
            "r2": history[0]["train_run_r2"],
            "within_0_10_mps_pct": history[0]["train_run_within_0_10_mps_pct"],
        },
        "final_train_run_metrics": {
            "mse": history[-1]["train_run_loss_mse"],
            "rmse_mps": history[-1]["train_run_rmse_mps"],
            "mae_mps": history[-1]["train_run_mae_mps"],
            "bias_mps": history[-1]["train_run_bias_mps"],
            "r2": history[-1]["train_run_r2"],
            "within_0_10_mps_pct": history[-1]["train_run_within_0_10_mps_pct"],
        },
        "test_run_metrics": run_metrics,
        "test_window_metrics": window_metrics,
        "test_source_subject_mean_baseline_run_metrics": run_baseline_metrics,
        "test_source_subject_mean_baseline_window_metrics": window_baseline_metrics,
        "original_cnn_test_run_metrics": existing_cnn_metrics,
        "comparison_vs_original_cnn": _comparison(run_metrics, existing_cnn_metrics),
        "comparison_vs_source_subject_mean": _comparison(
            run_metrics, run_baseline_metrics
        ),
        "test_run_metrics_by_subject": _metrics_by_field(run_rows, "subject_id"),
        "test_run_metrics_by_source": _metrics_by_field(run_rows, "source_id"),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "train_source_subject_means_path": str(mean_path),
        "test_run_predictions_path": str(run_predictions_path),
        "test_window_predictions_path": str(window_predictions_path),
        "epoch_metrics_plot": str(epoch_plot_path),
        "actual_vs_predicted_plot": str(prediction_plot_path),
        "rmse_comparison_plot": str(comparison_plot_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary

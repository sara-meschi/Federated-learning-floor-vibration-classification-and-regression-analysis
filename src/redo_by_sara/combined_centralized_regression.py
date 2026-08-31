from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .combined_centralized_classification import (
    RunRecord,
    _resample_run,
    assign_run_splits,
    discover_run_records,
    load_combined_config,
    window_start_indices,
)
from .models import SimpleCNN1D


SPEED_LEFT_COLUMN = "Gait - Lower Limb - Gait Speed L (m/s) [mean]"
SPEED_RIGHT_COLUMN = "Gait - Lower Limb - Gait Speed R (m/s) [mean]"


class CombinedRegressionDataset(Dataset):
    def __init__(self, artifact: dict[str, object], indices: Sequence[int]) -> None:
        self.samples = artifact["samples"]
        self.targets = artifact["regression_targets"]
        self.mean = artifact["channel_mean"].squeeze(0)
        self.std = artifact["channel_std"].squeeze(0)
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        sample = (self.samples[index] - self.mean) / self.std
        return sample.float(), self.targets[index].float()


def load_regression_config(path: str | Path) -> dict[str, Any]:
    config = load_combined_config(path)
    payload: dict[str, Any] = yaml.safe_load(Path(path).resolve().read_text())
    speed_globs: dict[str, str] = {}
    for source in payload["data"]["sources"]:
        source_id = str(source["id"])
        speed_glob = source.get("speed_label_glob")
        if not speed_glob:
            raise ValueError(f"Source {source_id} is missing speed_label_glob.")
        speed_globs[source_id] = str(speed_glob)
    config["speed_label_globs"] = speed_globs
    return config


def _read_speed_rows(path: Path) -> list[dict[str, Any] | None]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"APDM CSV has no header: {path}")
        missing_columns = {
            SPEED_LEFT_COLUMN,
            SPEED_RIGHT_COLUMN,
        } - set(reader.fieldnames)
        if missing_columns:
            raise KeyError(f"APDM CSV {path} is missing columns: {sorted(missing_columns)}")
        speeds: list[dict[str, Any] | None] = []
        for row in reader:
            left = (row.get(SPEED_LEFT_COLUMN) or "").strip()
            right = (row.get(SPEED_RIGHT_COLUMN) or "").strip()
            if not left or not right:
                speeds.append(None)
            else:
                left_speed = float(left)
                right_speed = float(right)
                speed = (left_speed + right_speed) / 2.0
                if not np.isfinite(speed) or speed <= 0:
                    raise ValueError(f"Invalid gait speed {speed} in {path}.")
                speeds.append(
                    {
                        "speed_left_mps": left_speed,
                        "speed_right_mps": right_speed,
                        "speed_mps": speed,
                        "apdm_record_datetime": row.get("Record Date/Time", ""),
                        "apdm_file_name": row.get("File Name", ""),
                        "speed_label_path": str(path),
                    }
                )
    return speeds


def load_speed_labels(
    config: dict[str, Any],
    records: Sequence[RunRecord],
    split_by_uid: dict[str, str],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    sources_by_id = {source.source_id: source for source in config["sources"]}
    records_by_source_subject: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        if split_by_uid[record.run_uid] != "excluded":
            records_by_source_subject[(record.source_id, record.subject_id)].append(record)

    speed_by_uid: dict[str, float] = {}
    label_details_by_uid: dict[str, dict[str, Any]] = {}
    for (source_id, subject_id), subject_records in sorted(records_by_source_subject.items()):
        source = sources_by_id[source_id]
        pattern = config["speed_label_globs"][source_id].format(subject_id=subject_id)
        matches = sorted(source.root.glob(pattern))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one aggregate APDM speed CSV for {source_id} subject "
                f"{subject_id} using '{pattern}', found {matches}."
            )
        label_path = matches[0]
        speeds = _read_speed_rows(label_path)
        expected_count = max(record.run_index for record in subject_records) + 1
        if len(speeds) != expected_count:
            raise ValueError(
                f"APDM/HDF5 run-count mismatch for {source_id} subject {subject_id}: "
                f"{len(speeds)} labels versus {expected_count} runs."
            )

        for record in subject_records:
            speed_row = speeds[record.run_index]
            status = split_by_uid[record.run_uid]
            if speed_row is None:
                if status in {"train", "test"}:
                    raise ValueError(
                        f"Usable run {record.run_uid} has no APDM gait-speed label."
                    )
                continue
            speed_by_uid[record.run_uid] = float(speed_row["speed_mps"])
            label_details_by_uid[record.run_uid] = speed_row

    usable_uids = {
        record.run_uid
        for record in records
        if split_by_uid[record.run_uid] in {"train", "test"}
    }
    if not usable_uids <= set(speed_by_uid):
        missing = sorted(usable_uids - set(speed_by_uid))
        raise AssertionError(f"Usable runs missing speed labels: {missing}")
    return speed_by_uid, label_details_by_uid


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "minimum_mps": float(values.min()),
        "maximum_mps": float(values.max()),
        "mean_mps": float(values.mean()),
        "std_mps": float(values.std()),
    }


def build_regression_artifact(config: dict[str, Any]) -> dict[str, object]:
    records = discover_run_records(config)
    split_by_uid = assign_run_splits(records, config["seed"], config["test_ratio"])
    speed_by_uid, label_details_by_uid = load_speed_labels(config, records, split_by_uid)

    channel_indices = [channel - 1 for channel in config["selected_channels"]]
    target_rate = config["target_sample_rate"]
    window_samples = int(round(config["window_seconds"] * target_rate))
    samples: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []
    windows_by_run: Counter[str] = Counter()

    records_by_hdf5: dict[Path, list[RunRecord]] = defaultdict(list)
    for record in records:
        if split_by_uid[record.run_uid] in {"train", "test"}:
            records_by_hdf5[record.hdf5_path].append(record)

    for hdf5_path, path_records in sorted(records_by_hdf5.items(), key=lambda item: str(item[0])):
        with h5py.File(hdf5_path, "r") as handle:
            dataset = handle["experiment/data"]
            for record in sorted(path_records, key=lambda item: item.run_index):
                signal = _resample_run(
                    raw_signal=dataset[record.run_index, channel_indices, :],
                    original_rate=record.effective_sample_rate,
                    target_rate=target_rate,
                )
                available_duration = signal.shape[-1] / target_rate
                walking_end = record.data_end if record.data_end > 0 else available_duration
                starts = window_start_indices(
                    start_seconds=record.data_start,
                    end_seconds=walking_end,
                    available_samples=signal.shape[-1],
                    sample_rate=target_rate,
                    window_seconds=config["window_seconds"],
                    step_seconds=config["step_seconds"],
                )
                if not starts:
                    raise ValueError(f"Walking interval for {record.run_uid} produced no windows.")
                speed = speed_by_uid[record.run_uid]
                for start_index in starts:
                    sample = signal[:, start_index : start_index + window_samples]
                    if sample.shape != (9, 2000):
                        raise AssertionError(
                            f"Unexpected regression window shape for {record.run_uid}: {sample.shape}."
                        )
                    if not np.isfinite(sample).all():
                        raise ValueError(f"Non-finite signal values in {record.run_uid}.")
                    samples.append(sample)
                    targets.append(speed)
                    metadata.append(
                        {
                            "source_id": record.source_id,
                            "subject_id": record.subject_id,
                            "run_index": record.run_index,
                            "run_uid": record.run_uid,
                            "direction": record.direction,
                            "start_time": start_index / target_rate,
                            "split": split_by_uid[record.run_uid],
                            "speed_mps": speed,
                        }
                    )
                    windows_by_run[record.run_uid] += 1

    sample_array = np.stack(samples).astype(np.float32, copy=False)
    target_array = np.asarray(targets, dtype=np.float32)
    train_indices = [index for index, item in enumerate(metadata) if item["split"] == "train"]
    test_indices = [index for index, item in enumerate(metadata) if item["split"] == "test"]
    train_run_uids = {metadata[index]["run_uid"] for index in train_indices}
    test_run_uids = {metadata[index]["run_uid"] for index in test_indices}
    if train_run_uids & test_run_uids:
        raise AssertionError("Regression run leakage detected between train and test.")

    train_samples = sample_array[train_indices]
    channel_mean = train_samples.mean(axis=(0, 2), keepdims=True)
    channel_std = train_samples.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std < 1e-8, 1.0, channel_std)

    run_manifest: list[dict[str, Any]] = []
    for record in records:
        row = asdict(record)
        row["hdf5_path"] = str(record.hdf5_path)
        row["split"] = split_by_uid[record.run_uid]
        label_details = label_details_by_uid.get(record.run_uid, {})
        row["speed_left_mps"] = label_details.get("speed_left_mps")
        row["speed_right_mps"] = label_details.get("speed_right_mps")
        row["speed_mps"] = label_details.get("speed_mps")
        row["apdm_record_datetime"] = label_details.get("apdm_record_datetime")
        row["apdm_file_name"] = label_details.get("apdm_file_name")
        row["speed_label_path"] = label_details.get("speed_label_path")
        row["walking_windows"] = windows_by_run[record.run_uid]
        run_manifest.append(row)

    included_subjects = ["001", "002", "003", "004", "005", "007", "008"]
    speed_by_subject: dict[str, dict[str, float | int]] = {}
    for subject_id in included_subjects:
        values = np.asarray(
            [
                speed_by_uid[record.run_uid]
                for record in records
                if record.subject_id == subject_id
                and split_by_uid[record.run_uid] in {"train", "test"}
            ],
            dtype=np.float64,
        )
        speed_by_subject[subject_id] = _distribution(values)

    status_counts = Counter(split_by_uid.values())
    summary = {
        "experiment": "combined_centralized_regression_9ch_no006_walking_only_80_20",
        "target": "Mean APDM left/right gait speed",
        "target_units": "m/s",
        "num_recorded_runs": len(records),
        "num_raw_skip_flags": int(sum(record.skip_run for record in records)),
        "run_status_counts": dict(sorted(status_counts.items())),
        "num_usable_runs": int(status_counts["train"] + status_counts["test"]),
        "num_train_runs": int(status_counts["train"]),
        "num_test_runs": int(status_counts["test"]),
        "num_examples": len(sample_array),
        "num_train": len(train_indices),
        "num_test": len(test_indices),
        "sample_shape": list(sample_array.shape[1:]),
        "selected_channels_one_based": config["selected_channels"],
        "selected_channels_zero_based": channel_indices,
        "subjects": included_subjects,
        "excluded_subjects": sorted(config["excluded_subjects"]),
        "target_sample_rate": target_rate,
        "window_seconds": config["window_seconds"],
        "step_seconds": config["step_seconds"],
        "train_ratio": config["train_ratio"],
        "test_ratio": config["test_ratio"],
        "split_seed": config["seed"],
        "timing_assumption": config["timing_assumption"],
        "speed_distribution_all_runs": _distribution(
            np.asarray([speed_by_uid[uid] for uid in sorted(train_run_uids | test_run_uids)])
        ),
        "speed_distribution_train_runs": _distribution(
            np.asarray([speed_by_uid[uid] for uid in sorted(train_run_uids)])
        ),
        "speed_distribution_test_runs": _distribution(
            np.asarray([speed_by_uid[uid] for uid in sorted(test_run_uids)])
        ),
        "speed_distribution_by_subject": speed_by_subject,
        "train_test_run_overlap": 0,
    }

    expected = {
        "num_usable_runs": 140,
        "num_train_runs": 112,
        "num_test_runs": 28,
        "num_examples": 1204,
        "num_train": 966,
        "num_test": 238,
        "sample_shape": [9, 2000],
    }
    for key, expected_value in expected.items():
        if summary[key] != expected_value:
            raise AssertionError(
                f"Unexpected regression artifact {key}: {summary[key]} != {expected_value}."
            )

    return {
        "samples": torch.from_numpy(sample_array),
        "regression_targets": torch.from_numpy(target_array),
        "train_indices": torch.tensor(train_indices, dtype=torch.long),
        "test_indices": torch.tensor(test_indices, dtype=torch.long),
        "channel_mean": torch.from_numpy(channel_mean.astype(np.float32)),
        "channel_std": torch.from_numpy(channel_std.astype(np.float32)),
        "metadata": metadata,
        "run_manifest": run_manifest,
        "summary": summary,
    }


def save_regression_artifact(
    artifact: dict[str, object], artifact_path: Path, summary_path: Path, manifest_path: Path
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    summary_path.write_text(json.dumps(artifact["summary"], indent=2))
    manifest = artifact["run_manifest"]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    residual = predicted - actual
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 1e-12 else 0.0
    within_0_10 = float(100.0 * np.mean(np.abs(residual) <= 0.10))
    return {
        "mse": mse,
        "rmse_mps": float(np.sqrt(mse)),
        "mae_mps": mae,
        "bias_mps": bias,
        "r2": r2,
        "within_0_10_mps_pct": within_0_10,
    }


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for samples, batch_targets in loader:
        samples = samples.to(device)
        batch_targets = batch_targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(samples).squeeze(-1)
            loss = criterion(outputs, batch_targets)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = samples.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        predictions.append(outputs.detach().cpu())
        targets.append(batch_targets.detach().cpu())
    return (
        total_loss / max(total_examples, 1),
        torch.cat(predictions).numpy(),
        torch.cat(targets).numpy(),
    )


def _aggregate_run_predictions(
    artifact: dict[str, object], test_indices: Sequence[int], predicted: np.ndarray
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = artifact["metadata"]
    for artifact_index, prediction in zip(test_indices, predicted):
        item = metadata[int(artifact_index)]
        group = grouped.setdefault(
            item["run_uid"],
            {
                "source_id": item["source_id"],
                "subject_id": item["subject_id"],
                "run_index": item["run_index"],
                "run_uid": item["run_uid"],
                "direction": item["direction"],
                "actual_speed_mps": float(item["speed_mps"]),
                "window_predictions": [],
            },
        )
        group["window_predictions"].append(float(prediction))

    rows: list[dict[str, Any]] = []
    for run_uid, group in sorted(grouped.items()):
        run_prediction = float(np.mean(group.pop("window_predictions")))
        rows.append(
            {
                **group,
                "predicted_speed_mps": run_prediction,
                "error_mps": run_prediction - float(group["actual_speed_mps"]),
                "absolute_error_mps": abs(run_prediction - float(group["actual_speed_mps"])),
            }
        )
    return rows


def _per_subject_metrics(
    subjects: Sequence[str], actual: np.ndarray, predicted: np.ndarray
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    subject_array = np.asarray(subjects)
    for subject_id in sorted(set(subjects)):
        mask = subject_array == subject_id
        result[subject_id] = regression_metrics(actual[mask], predicted[mask])
    return result


def plot_epoch_metrics(history: Sequence[dict[str, float]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], color="#c43c39", linewidth=2)
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Combined Centralized Speed Regression Training")
    axes[1].plot(epochs, [row["train_rmse_mps"] for row in history], color="#2b6cb0", linewidth=2)
    axes[1].set_ylabel("RMSE (m/s)")
    axes[2].plot(epochs, [row["train_r2"] for row in history], color="#2f855a", linewidth=2)
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("R²")
    axes[3].plot(
        epochs,
        [row["train_within_0_10_mps_pct"] for row in history],
        color="#805ad5",
        linewidth=2,
    )
    axes[3].set_ylabel("Within ±0.10 m/s (%)")
    axes[3].set_xlabel("Epoch")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_actual_vs_predicted(
    run_rows: Sequence[dict[str, Any]], metrics: dict[str, float], output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    subjects = sorted({str(row["subject_id"]) for row in run_rows})
    sources = sorted({str(row["source_id"]) for row in run_rows})
    colors = plt.get_cmap("tab10")
    markers = {source_id: marker for source_id, marker in zip(sources, ["o", "^"])}
    all_actual: list[float] = []
    all_predicted: list[float] = []
    for color_index, subject_id in enumerate(subjects):
        for source_id in sources:
            selected_rows = [
                row
                for row in run_rows
                if str(row["subject_id"]) == subject_id
                and str(row["source_id"]) == source_id
            ]
            if not selected_rows:
                continue
            actual = [float(row["actual_speed_mps"]) for row in selected_rows]
            predicted = [float(row["predicted_speed_mps"]) for row in selected_rows]
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
    axis.plot([low - padding, high + padding], [low - padding, high + padding], "k--", linewidth=1.2)
    axis.set_xlim(low - padding, high + padding)
    axis.set_ylim(low - padding, high + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Actual run speed (m/s)")
    axis.set_ylabel("Predicted run speed (m/s)")
    axis.set_title(
        f"Held-out Run Speeds: RMSE={metrics['rmse_mps']:.3f} m/s, R²={metrics['r2']:.3f}"
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


def _write_rows(rows: Sequence[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _add_train_run_mean_baselines(
    artifact: dict[str, object],
    train_indices: Sequence[int],
    window_rows: Sequence[dict[str, Any]],
    run_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    metadata: list[dict[str, Any]] = artifact["metadata"]
    train_runs: dict[str, dict[str, Any]] = {}
    for artifact_index in train_indices:
        item = metadata[int(artifact_index)]
        train_runs.setdefault(item["run_uid"], item)

    all_values: list[float] = []
    by_subject: dict[str, list[float]] = defaultdict(list)
    by_source_subject: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in train_runs.values():
        speed = float(item["speed_mps"])
        subject_id = str(item["subject_id"])
        source_id = str(item["source_id"])
        all_values.append(speed)
        by_subject[subject_id].append(speed)
        by_source_subject[(source_id, subject_id)].append(speed)

    overall_mean = float(np.mean(all_values))
    subject_means = {key: float(np.mean(values)) for key, values in by_subject.items()}
    source_subject_means = {
        key: float(np.mean(values)) for key, values in by_source_subject.items()
    }

    for row in [*window_rows, *run_rows]:
        subject_id = str(row["subject_id"])
        source_id = str(row["source_id"])
        row["overall_train_run_mean_mps"] = overall_mean
        row["subject_train_run_mean_mps"] = subject_means[subject_id]
        row["source_subject_train_run_mean_mps"] = source_subject_means[
            (source_id, subject_id)
        ]

    def metrics_for(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
        actual = np.asarray([float(row["actual_speed_mps"]) for row in rows])
        return {
            "overall_train_run_mean": regression_metrics(
                actual,
                np.asarray([float(row["overall_train_run_mean_mps"]) for row in rows]),
            ),
            "subject_train_run_mean": regression_metrics(
                actual,
                np.asarray([float(row["subject_train_run_mean_mps"]) for row in rows]),
            ),
            "source_subject_train_run_mean": regression_metrics(
                actual,
                np.asarray(
                    [float(row["source_subject_train_run_mean_mps"]) for row in rows]
                ),
            ),
        }

    return {"window": metrics_for(window_rows), "run": metrics_for(run_rows)}


def train_and_evaluate_regression(
    artifact: dict[str, object], config: dict[str, Any]
) -> dict[str, Any]:
    _set_random_seeds(config["seed"])
    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_indices = artifact["train_indices"].tolist()
    test_indices = artifact["test_indices"].tolist()
    train_dataset = CombinedRegressionDataset(artifact, train_indices)
    test_dataset = CombinedRegressionDataset(artifact, test_indices)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        generator=generator,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN1D(in_channels=9, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["minimum_learning_rate"]
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, config["epochs"] + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimization_loss, _, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        train_loss, predicted, actual = _run_epoch(
            model, train_eval_loader, criterion, device, optimizer=None
        )
        metrics = regression_metrics(actual, predicted)
        row = {
            "epoch": float(epoch),
            "optimization_loss": optimization_loss,
            "train_loss": train_loss,
            "train_rmse_mps": metrics["rmse_mps"],
            "train_mae_mps": metrics["mae_mps"],
            "train_r2": metrics["r2"],
            "train_within_0_10_mps_pct": metrics["within_0_10_mps_pct"],
            "learning_rate": learning_rate,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{config['epochs']}: loss={train_loss:.6f}, "
            f"rmse={metrics['rmse_mps']:.4f} m/s, r2={metrics['r2']:.4f}, "
            f"within_0.10={metrics['within_0_10_mps_pct']:.1f}%, "
            f"lr={learning_rate:.8f}",
            flush=True,
        )
        scheduler.step()

    test_loss, window_predicted, window_actual = _run_epoch(
        model, test_loader, criterion, device, optimizer=None
    )
    window_metrics = regression_metrics(window_actual, window_predicted)
    run_rows = _aggregate_run_predictions(artifact, test_indices, window_predicted)
    run_actual = np.asarray([row["actual_speed_mps"] for row in run_rows])
    run_predicted = np.asarray([row["predicted_speed_mps"] for row in run_rows])
    run_subjects = [str(row["subject_id"]) for row in run_rows]
    run_metrics = regression_metrics(run_actual, run_predicted)

    metadata: list[dict[str, Any]] = artifact["metadata"]
    window_rows: list[dict[str, Any]] = []
    for artifact_index, actual, predicted in zip(test_indices, window_actual, window_predicted):
        item = metadata[int(artifact_index)]
        window_rows.append(
            {
                "artifact_index": int(artifact_index),
                "source_id": item["source_id"],
                "subject_id": item["subject_id"],
                "run_index": item["run_index"],
                "run_uid": item["run_uid"],
                "direction": item["direction"],
                "start_time": item["start_time"],
                "actual_speed_mps": float(actual),
                "predicted_speed_mps": float(predicted),
                "error_mps": float(predicted - actual),
                "absolute_error_mps": float(abs(predicted - actual)),
            }
        )

    model_path = output_dir / "combined_centralized_speed_regression_final.pt"
    history_path = output_dir / "training_history.csv"
    summary_path = output_dir / "training_summary.json"
    epoch_plot_path = output_dir / "loss_rmse_r2_accuracy_vs_epoch.png"
    actual_plot_path = output_dir / "actual_vs_predicted_speed.png"
    window_predictions_path = output_dir / "test_window_predictions.csv"
    run_predictions_path = output_dir / "test_run_predictions.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selected_channels_one_based": artifact["summary"]["selected_channels_one_based"],
            "channel_mean": artifact["channel_mean"],
            "channel_std": artifact["channel_std"],
            "sample_shape": artifact["summary"]["sample_shape"],
            "target": artifact["summary"]["target"],
            "target_units": "m/s",
            "seed": config["seed"],
            "epochs": config["epochs"],
        },
        model_path,
    )
    baseline_metrics = _add_train_run_mean_baselines(
        artifact, train_indices, window_rows, run_rows
    )
    _write_rows(history, history_path)
    _write_rows(window_rows, window_predictions_path)
    _write_rows(run_rows, run_predictions_path)
    plot_epoch_metrics(history, epoch_plot_path)
    plot_actual_vs_predicted(run_rows, run_metrics, actual_plot_path)

    window_subjects = [str(metadata[index]["subject_id"]) for index in test_indices]
    run_sources = [str(row["source_id"]) for row in run_rows]
    window_sources = [str(metadata[index]["source_id"]) for index in test_indices]
    source_subject_baseline_mse = baseline_metrics["run"][
        "source_subject_train_run_mean"
    ]["mse"]
    summary = {
        "experiment": artifact["summary"]["experiment"],
        "device": str(device),
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "initial_learning_rate": config["learning_rate"],
        "minimum_learning_rate": config["minimum_learning_rate"],
        "weight_decay": config["weight_decay"],
        "test_evaluation_policy": "Test set evaluated once after the final epoch; no validation set.",
        "final_train": {
            "mse": history[-1]["train_loss"],
            "rmse_mps": history[-1]["train_rmse_mps"],
            "mae_mps": history[-1]["train_mae_mps"],
            "r2": history[-1]["train_r2"],
            "within_0_10_mps_pct": history[-1]["train_within_0_10_mps_pct"],
        },
        "test_window_metrics": window_metrics,
        "test_run_metrics": run_metrics,
        "train_run_mean_baselines": baseline_metrics,
        "test_run_mse_skill_vs_source_subject_baseline": float(
            1.0 - run_metrics["mse"] / source_subject_baseline_mse
        ),
        "test_window_metrics_by_subject": _per_subject_metrics(
            window_subjects, window_actual, window_predicted
        ),
        "test_run_metrics_by_subject": _per_subject_metrics(
            run_subjects, run_actual, run_predicted
        ),
        "test_window_metrics_by_source": _per_subject_metrics(
            window_sources, window_actual, window_predicted
        ),
        "test_run_metrics_by_source": _per_subject_metrics(
            run_sources, run_actual, run_predicted
        ),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "epoch_metrics_plot": str(epoch_plot_path),
        "actual_vs_predicted_plot": str(actual_plot_path),
        "test_window_predictions_path": str(window_predictions_path),
        "test_run_predictions_path": str(run_predictions_path),
        "test_loss_mse": test_loss,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary

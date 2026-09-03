from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/preprocessing.py"
raise ImportError(
    "Quarantined legacy module: legacy/preprocessing.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.signal import resample_poly

from .config import ExperimentConfig


@dataclass
class Example:
    sample: np.ndarray
    speed: float
    subject_id: str
    run_index: int
    start_time: float


def _get_general_parameter(perams: np.ndarray, name: str) -> float | None:
    mask = perams["parameter"] == name.encode()
    matches = perams[mask]
    if len(matches) == 0:
        return None
    value = matches["value"][0]
    if isinstance(value, bytes):
        return float(value.decode("utf-8"))
    return float(value)


def _subject_id_from_path(path: Path) -> str:
    return path.stem.split("_")[-1]


def _find_apdm_csv(dataset_root: Path, subject_id: str) -> Path | None:
    matches = sorted(dataset_root.glob(f"APDM_Data/*_{subject_id}_*/*.csv"))
    return matches[0] if matches else None


def _parse_speed_labels(csv_path: Path) -> list[float]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"Empty APDM CSV: {csv_path}")

    old_left = "Gait - Lower Limb - Gait Speed L (m/s) [mean]"
    old_right = "Gait - Lower Limb - Gait Speed R (m/s) [mean]"
    if old_left in rows[0]:
        header = rows[0]
        speeds: list[float] = []
        for raw_row in rows[1:]:
            row = dict(zip(header, raw_row))
            left = (row.get(old_left) or "").strip()
            right = (row.get(old_right) or "").strip()
            if not left or not right:
                speeds.append(0.0)
            else:
                speeds.append((float(left) + float(right)) / 2.0)
        return speeds

    header = None
    left_row = None
    right_row = None
    for row in rows:
        if not row:
            continue
        if row[0] == "Measure":
            header = row
        elif row[0] == "Gait - Lower Limb - Gait Speed L (m/s)":
            left_row = row
        elif row[0] == "Gait - Lower Limb - Gait Speed R (m/s)":
            right_row = row

    if header is None or left_row is None or right_row is None:
        raise KeyError(f"Unsupported APDM CSV format: {csv_path}")

    speeds = []
    for idx in range(5, min(len(header), len(left_row), len(right_row))):
        if not header[idx].strip():
            continue
        left = left_row[idx].strip()
        right = right_row[idx].strip()
        if not left or not right:
            speeds.append(0.0)
        else:
            speeds.append((float(left) + float(right)) / 2.0)
    return speeds


def _load_run_segments(csv_path: Path) -> dict[tuple[str, int], dict[str, float | int]]:
    rows: dict[tuple[str, int], dict[str, float | int]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = f"{int(row['Subject']):03d}"
            run_index = int(row["Run"])
            rows[(subject_id, run_index)] = {
                "no_step_start": float(row["No Step Start (s)"]),
                "no_step_end": float(row["No Step End (s)"]),
                "data_start": float(row["Data Start (s)"]),
                "data_end": float(row["Data End (s)"] or 0.0),
                "skip_run": int(row["Skip Run"]),
            }
    return rows


def _resolve_channel_selection(config: ExperimentConfig) -> tuple[list[int], list[str]]:
    selected_channels: list[int] = []
    channel_labels: list[str] = []
    for sensor_name in config.data.selected_sensors:
        for channel in config.data.sensor_channel_map[sensor_name]:
            selected_channels.append(channel)
            channel_labels.append(f"{sensor_name}:ch{channel}")
    return selected_channels, channel_labels


def _resample_trials(data: np.ndarray, original_rate: float, target_rate: float) -> np.ndarray:
    if np.isclose(original_rate, target_rate):
        return data.astype(np.float32, copy=False)
    ratio = Fraction(target_rate / original_rate).limit_denominator(1000)
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32, copy=False)


def _window_run(
    trial: np.ndarray,
    speed: float,
    subject_id: str,
    run_index: int,
    start_sec: float,
    end_sec: float,
    sample_rate: float,
    window_seconds: float,
    step_seconds: float,
) -> list[Example]:
    window_size = int(round(window_seconds * sample_rate))
    step_size = int(round(step_seconds * sample_rate))
    start_idx = int(round(start_sec * sample_rate))
    end_idx = int(round(end_sec * sample_rate))
    end_idx = min(end_idx, trial.shape[-1])

    examples: list[Example] = []
    for idx in range(start_idx, end_idx - window_size + 1, step_size):
        window = trial[:, idx : idx + window_size].astype(np.float32, copy=False)
        examples.append(
            Example(
                sample=window,
                speed=float(speed),
                subject_id=subject_id,
                run_index=run_index,
                start_time=idx / sample_rate,
            )
        )
    return examples


def _assign_subject_run_splits(
    metadata: list[dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict[str, list[tuple[str, int]]]]:
    subject_runs: dict[str, list[int]] = defaultdict(list)
    for item in metadata:
        subject_id = str(item["subject_id"])
        run_index = int(item["run_index"])
        if run_index not in subject_runs[subject_id]:
            subject_runs[subject_id].append(run_index)

    rng = random.Random(seed)
    split_groups = {"train": [], "val": [], "test": []}

    for subject_id, runs in sorted(subject_runs.items()):
        shuffled = sorted(runs)
        rng.shuffle(shuffled)
        n_runs = len(shuffled)
        if n_runs < 3:
            raise ValueError(
                f"Subject {subject_id} has only {n_runs} runs. "
                "A true train/val/test split needs at least 3 runs per subject."
            )

        n_test = max(1, round(n_runs * test_ratio))
        n_val = max(1, round(n_runs * val_ratio))
        if n_test + n_val >= n_runs:
            overflow = n_test + n_val - (n_runs - 1)
            while overflow > 0 and n_val > 1:
                n_val -= 1
                overflow -= 1
            while overflow > 0 and n_test > 1:
                n_test -= 1
                overflow -= 1
            if n_test + n_val >= n_runs:
                raise ValueError(
                    f"Unable to allocate non-overlapping train/val/test runs for subject {subject_id}."
                )

        test_runs = shuffled[:n_test]
        val_runs = shuffled[n_test : n_test + n_val]
        train_runs = shuffled[n_test + n_val :]
        if not train_runs:
            raise ValueError(f"Subject {subject_id} ended up with no training runs.")

        split_groups["test"].extend((subject_id, run_index) for run_index in test_runs)
        split_groups["val"].extend((subject_id, run_index) for run_index in val_runs)
        split_groups["train"].extend((subject_id, run_index) for run_index in train_runs)

    train_group_set = set(split_groups["train"])
    val_group_set = set(split_groups["val"])
    test_group_set = set(split_groups["test"])
    if train_group_set & val_group_set or train_group_set & test_group_set or val_group_set & test_group_set:
        raise ValueError("Detected overlap between train/val/test run groups.")

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for idx, item in enumerate(metadata):
        group = (str(item["subject_id"]), int(item["run_index"]))
        if group in train_group_set:
            train_indices.append(idx)
        elif group in val_group_set:
            val_indices.append(idx)
        elif group in test_group_set:
            test_indices.append(idx)
        else:
            raise ValueError(f"Run group {group} was not assigned to any split.")

    return train_indices, val_indices, test_indices, split_groups


def build_artifact(config: ExperimentConfig) -> dict[str, object]:
    dataset_root = config.data.dataset_root
    run_segments = _load_run_segments(dataset_root / "runPeramiters.csv")
    hdf5_files = sorted((dataset_root / "vib" / "Data").glob("*.hdf5"))

    if config.data.subject_limit is not None:
        hdf5_files = hdf5_files[: config.data.subject_limit]

    selected_channels, channel_labels = _resolve_channel_selection(config)
    zero_based_channels = [channel - 1 for channel in selected_channels]
    sorted_positions = sorted(range(len(zero_based_channels)), key=lambda idx: zero_based_channels[idx])
    sorted_channels = [zero_based_channels[idx] for idx in sorted_positions]
    restore_order = np.argsort(sorted_positions)

    examples: list[Example] = []
    skipped_subjects: list[str] = []

    for hdf5_path in hdf5_files:
        subject_id = _subject_id_from_path(hdf5_path)
        apdm_csv = _find_apdm_csv(dataset_root, subject_id)
        if apdm_csv is None:
            skipped_subjects.append(subject_id)
            continue

        speeds = _parse_speed_labels(apdm_csv)
        with h5py.File(hdf5_path, "r") as handle:
            data = handle["experiment/data"][:, sorted_channels, :]
            data = data[:, restore_order, :]
            general_parameters = handle["experiment/general_parameters"][:]
            original_rate = _get_general_parameter(general_parameters, "fs")
            if original_rate is None:
                raise ValueError(f"Missing sample rate in {hdf5_path}")

        data = _resample_trials(data, original_rate, config.data.target_sample_rate)
        run_count = min(len(data), len(speeds))

        for run_index in range(run_count):
            segment = run_segments.get((subject_id, run_index))
            if segment is None or int(segment["skip_run"]) == 1:
                continue

            end_sec = float(segment["data_end"])
            if end_sec <= 0:
                end_sec = data.shape[-1] / config.data.target_sample_rate

            examples.extend(
                _window_run(
                    trial=data[run_index],
                    speed=speeds[run_index],
                    subject_id=subject_id,
                    run_index=run_index,
                    start_sec=float(segment["data_start"]),
                    end_sec=end_sec,
                    sample_rate=config.data.target_sample_rate,
                    window_seconds=config.data.window_seconds,
                    step_seconds=config.data.step_seconds,
                )
            )

    if not examples:
        raise RuntimeError("No examples were created. Check paths and label parsing.")

    subject_ids = sorted({example.subject_id for example in examples})
    subject_to_class = {subject_id: idx for idx, subject_id in enumerate(subject_ids)}

    samples = np.stack([example.sample for example in examples]).astype(np.float32, copy=False)
    regression_targets = np.array([example.speed for example in examples], dtype=np.float32)
    classification_targets = np.array([subject_to_class[example.subject_id] for example in examples], dtype=np.int64)
    metadata = [
        {
            "subject_id": example.subject_id,
            "run_index": example.run_index,
            "start_time": example.start_time,
        }
        for example in examples
    ]

    train_indices, val_indices, test_indices, split_groups = _assign_subject_run_splits(
        metadata=metadata,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.seed,
    )

    train_samples = samples[train_indices]
    channel_mean = train_samples.mean(axis=(0, 2), keepdims=True)
    channel_std = train_samples.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)

    summary = {
        "num_examples": int(len(samples)),
        "num_train": int(len(train_indices)),
        "num_val": int(len(val_indices)),
        "num_test": int(len(test_indices)),
        "sample_shape": list(samples.shape[1:]),
        "subjects": subject_ids,
        "skipped_subjects": skipped_subjects,
        "selected_sensors": config.data.selected_sensors,
        "selected_channels": selected_channels,
        "channel_labels": channel_labels,
        "split_group_counts": {split_name: len(groups) for split_name, groups in split_groups.items()},
        "split_overlap": {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        },
    }

    return {
        "samples": torch.from_numpy(samples),
        "regression_targets": torch.from_numpy(regression_targets),
        "classification_targets": torch.from_numpy(classification_targets),
        "train_indices": torch.tensor(train_indices, dtype=torch.long),
        "val_indices": torch.tensor(val_indices, dtype=torch.long),
        "test_indices": torch.tensor(test_indices, dtype=torch.long),
        "channel_mean": torch.from_numpy(channel_mean.astype(np.float32)),
        "channel_std": torch.from_numpy(channel_std.astype(np.float32)),
        "subject_to_class": subject_to_class,
        "metadata": metadata,
        "summary": summary,
        "config": {
            "seed": config.seed,
            "dataset_root": str(config.data.dataset_root),
            "sensor_channel_map": config.data.sensor_channel_map,
            "selected_sensors": config.data.selected_sensors,
            "target_sample_rate": config.data.target_sample_rate,
            "window_seconds": config.data.window_seconds,
            "step_seconds": config.data.step_seconds,
            "train_ratio": config.data.train_ratio,
            "val_ratio": config.data.val_ratio,
            "test_ratio": config.data.test_ratio,
        },
    }


def save_artifact(artifact: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(artifact["summary"], indent=2))

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.signal import resample_poly
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .guardrails import (
    assert_channel_index_conversion,
    assert_determinism_flags,
    assert_split_integrity,
    seeding_record,
    set_random_seeds,
)
from .models import SimpleCNN1D


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    root: Path
    hdf5_glob: str
    run_parameters: Path
    sample_rate_override: float | None
    annotation_time_scale: float
    direction_mode: str


@dataclass(frozen=True)
class RunRecord:
    source_id: str
    subject_id: str
    run_index: int
    run_uid: str
    hdf5_path: Path
    metadata_sample_rate: float
    effective_sample_rate: float
    raw_num_samples: int
    no_step_start: float
    no_step_end: float
    data_start: float
    data_end: float
    skip_run: bool
    excluded: bool
    direction: str


class CombinedArtifactDataset(Dataset):
    def __init__(self, artifact: dict[str, object], indices: Sequence[int]) -> None:
        self.samples = artifact["samples"]
        self.targets = artifact["classification_targets"]
        self.mean = artifact["channel_mean"].squeeze(0)
        self.std = artifact["channel_std"].squeeze(0)
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        sample = (self.samples[index] - self.mean) / self.std
        return sample.float(), self.targets[index]


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_combined_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    payload: dict[str, Any] = yaml.safe_load(config_path.read_text())

    data_config = payload["data"]
    selected_channels = [int(channel) for channel in data_config["selected_channels"]]
    if selected_channels != [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        raise ValueError(
            "This experiment must use one-based channels [1, 2, 3, 4, 5, 6, 7, 8, 10]."
        )
    if len(selected_channels) != len(set(selected_channels)):
        raise ValueError("Selected channels contain duplicates.")

    train_ratio = float(data_config["train_ratio"])
    test_ratio = float(data_config["test_ratio"])
    if not math.isclose(train_ratio + test_ratio, 1.0, abs_tol=1e-9):
        raise ValueError("train_ratio + test_ratio must equal 1.0.")
    if not math.isclose(train_ratio, 0.8) or not math.isclose(test_ratio, 0.2):
        raise ValueError("This experiment is fixed to an 80/20 train/test split.")

    sources: list[SourceSpec] = []
    for raw_source in data_config["sources"]:
        source_root = _resolve_path(project_root, raw_source["root"])
        sources.append(
            SourceSpec(
                source_id=str(raw_source["id"]),
                root=source_root,
                hdf5_glob=str(raw_source["hdf5_glob"]),
                run_parameters=source_root / str(raw_source["run_parameters"]),
                sample_rate_override=(
                    None
                    if raw_source.get("sample_rate_override") is None
                    else float(raw_source["sample_rate_override"])
                ),
                annotation_time_scale=float(raw_source.get("annotation_time_scale", 1.0)),
                direction_mode=str(raw_source.get("direction_mode", "unknown")),
            )
        )

    training_config = payload["training"]
    if int(training_config["epochs"]) != 60:
        raise ValueError("This experiment is fixed to 60 epochs.")

    return {
        "config_path": config_path,
        "project_root": project_root,
        "seed": int(payload["seed"]),
        "output_dir": _resolve_path(project_root, payload["output_dir"]),
        "artifact_name": str(payload["artifact_name"]),
        "sources": sources,
        "selected_channels": selected_channels,
        "excluded_subjects": {
            f"{int(subject_id):03d}" for subject_id in data_config.get("excluded_subjects", [])
        },
        "target_sample_rate": float(data_config["target_sample_rate"]),
        "window_seconds": float(data_config["window_seconds"]),
        "step_seconds": float(data_config["step_seconds"]),
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "batch_size": int(training_config["batch_size"]),
        "epochs": int(training_config["epochs"]),
        "learning_rate": float(training_config["learning_rate"]),
        "minimum_learning_rate": float(training_config["minimum_learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
        "num_workers": int(training_config.get("num_workers", 0)),
        "timing_assumption": str(data_config["timing_assumption"]),
    }


def _clean_hdf_value(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _general_parameter(handle: h5py.File, name: str) -> float:
    parameters = handle["experiment/general_parameters"][:]
    for row in parameters:
        parameter_name = str(_clean_hdf_value(row["parameter"]))
        if parameter_name == name:
            return float(_clean_hdf_value(row["value"]))
    raise KeyError(f"Missing HDF5 general parameter '{name}' in {handle.filename}.")


def _subject_id_from_hdf5(path: Path) -> str:
    suffix = path.stem.split("_")[-1]
    try:
        return f"{int(suffix):03d}"
    except ValueError as exc:
        raise ValueError(f"Cannot extract a subject id from HDF5 filename: {path}") from exc


def _direction_for_run(mode: str, run_index: int) -> str:
    if mode == "alternating_north_south":
        return "N_to_S" if run_index % 2 == 0 else "S_to_N"
    if mode == "single_direction_unknown":
        return "single_direction_unknown"
    return "unknown"


def discover_run_records(config: dict[str, Any]) -> list[RunRecord]:
    excluded_subjects: set[str] = config["excluded_subjects"]
    records: list[RunRecord] = []
    seen_run_uids: set[str] = set()

    for source in config["sources"]:
        if not source.run_parameters.exists():
            raise FileNotFoundError(source.run_parameters)

        hdf5_paths = sorted(source.root.glob(source.hdf5_glob))
        if not hdf5_paths:
            raise FileNotFoundError(
                f"No HDF5 files matched {source.root / source.hdf5_glob}."
            )
        subject_files: dict[str, Path] = {}
        hdf5_shapes: dict[str, tuple[int, int, int]] = {}
        metadata_rates: dict[str, float] = {}
        effective_rates: dict[str, float] = {}
        for hdf5_path in hdf5_paths:
            subject_id = _subject_id_from_hdf5(hdf5_path)
            if subject_id in subject_files:
                raise ValueError(
                    f"Source {source.source_id} has multiple primary HDF5 files for subject {subject_id}."
                )
            with h5py.File(hdf5_path, "r") as handle:
                shape = tuple(int(size) for size in handle["experiment/data"].shape)
                if len(shape) != 3:
                    raise ValueError(f"Unexpected signal shape in {hdf5_path}: {shape}")
                if shape[1] < max(config["selected_channels"]):
                    raise ValueError(
                        f"{hdf5_path} has only {shape[1]} channels, but channel "
                        f"{max(config['selected_channels'])} was requested."
                    )
                metadata_rate = _general_parameter(handle, "fs")
            subject_files[subject_id] = hdf5_path
            hdf5_shapes[subject_id] = shape
            metadata_rates[subject_id] = metadata_rate
            effective_rates[subject_id] = (
                metadata_rate
                if source.sample_rate_override is None
                else source.sample_rate_override
            )

        rows_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
        with source.run_parameters.open(newline="") as handle:
            for row in csv.DictReader(handle):
                subject_id = f"{int(row['Subject']):03d}"
                rows_by_subject[subject_id].append(row)

        unknown_subjects = sorted(set(rows_by_subject) - set(subject_files))
        if unknown_subjects:
            raise ValueError(
                f"Run table {source.run_parameters} contains subjects without primary HDF5 files: "
                f"{unknown_subjects}"
            )

        for subject_id, hdf5_path in sorted(subject_files.items()):
            rows = rows_by_subject.get(subject_id, [])
            expected_run_indices = set(range(hdf5_shapes[subject_id][0]))
            actual_run_indices = {int(row["Run"]) for row in rows}
            if actual_run_indices != expected_run_indices:
                missing = sorted(expected_run_indices - actual_run_indices)
                extra = sorted(actual_run_indices - expected_run_indices)
                raise ValueError(
                    f"HDF5/run-table mismatch for {source.source_id} subject {subject_id}: "
                    f"missing={missing}, extra={extra}."
                )

            for row in sorted(rows, key=lambda item: int(item["Run"])):
                run_index = int(row["Run"])
                run_uid = f"{source.source_id}:{subject_id}:{run_index:03d}"
                if run_uid in seen_run_uids:
                    raise ValueError(f"Duplicate run UID: {run_uid}")
                seen_run_uids.add(run_uid)
                time_scale = source.annotation_time_scale
                raw_data_end = float(row["Data End (s)"] or 0.0)
                records.append(
                    RunRecord(
                        source_id=source.source_id,
                        subject_id=subject_id,
                        run_index=run_index,
                        run_uid=run_uid,
                        hdf5_path=hdf5_path,
                        metadata_sample_rate=metadata_rates[subject_id],
                        effective_sample_rate=effective_rates[subject_id],
                        raw_num_samples=hdf5_shapes[subject_id][2],
                        no_step_start=float(row["No Step Start (s)"]) * time_scale,
                        no_step_end=float(row["No Step End (s)"]) * time_scale,
                        data_start=float(row["Data Start (s)"]) * time_scale,
                        data_end=(raw_data_end * time_scale if raw_data_end > 0 else 0.0),
                        skip_run=bool(int(row["Skip Run"])),
                        excluded=subject_id in excluded_subjects,
                        direction=_direction_for_run(source.direction_mode, run_index),
                    )
                )

    if len(records) != len(seen_run_uids):
        raise AssertionError("Run discovery produced duplicate UIDs.")
    return records


def _direction_balanced_test_records(
    records: Sequence[RunRecord],
    test_count: int,
    test_ratio: float,
    seed: int,
    stratum_key: tuple[str, str],
) -> list[RunRecord]:
    by_direction: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_direction[record.direction].append(record)

    if len(by_direction) == 1:
        shuffled = sorted(records, key=lambda item: item.run_index)
        random.Random(f"{seed}:{stratum_key}:split").shuffle(shuffled)
        return shuffled[:test_count]

    allocations: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for direction, direction_records in sorted(by_direction.items()):
        exact = len(direction_records) * test_ratio
        allocation = int(math.floor(exact))
        if test_count >= len(by_direction):
            allocation = max(1, allocation)
        allocation = min(allocation, len(direction_records) - 1)
        allocations[direction] = allocation
        fractions.append((exact - math.floor(exact), direction))

    remaining = test_count - sum(allocations.values())
    for _, direction in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        capacity = len(by_direction[direction]) - 1 - allocations[direction]
        if capacity > 0:
            allocations[direction] += 1
            remaining -= 1
    if remaining != 0:
        raise ValueError(
            f"Could not allocate {test_count} direction-balanced test runs for {stratum_key}."
        )

    selected: list[RunRecord] = []
    for direction, direction_records in sorted(by_direction.items()):
        shuffled = sorted(direction_records, key=lambda item: item.run_index)
        random.Random(f"{seed}:{stratum_key}:{direction}:split").shuffle(shuffled)
        selected.extend(shuffled[: allocations[direction]])
    return selected


def assign_run_splits(
    records: Sequence[RunRecord], seed: int, test_ratio: float
) -> dict[str, str]:
    split_by_uid: dict[str, str] = {}
    usable_by_stratum: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)

    for record in records:
        if record.excluded:
            split_by_uid[record.run_uid] = "excluded"
        elif record.skip_run:
            split_by_uid[record.run_uid] = "skipped"
        else:
            usable_by_stratum[(record.source_id, record.subject_id)].append(record)

    for stratum_key, stratum_records in sorted(usable_by_stratum.items()):
        test_count = max(1, round(len(stratum_records) * test_ratio))
        if test_count >= len(stratum_records):
            raise ValueError(f"Stratum {stratum_key} has too few runs for 80/20 splitting.")
        test_records = _direction_balanced_test_records(
            records=stratum_records,
            test_count=test_count,
            test_ratio=test_ratio,
            seed=seed,
            stratum_key=stratum_key,
        )
        test_uids = {record.run_uid for record in test_records}
        for record in stratum_records:
            split_by_uid[record.run_uid] = (
                "test" if record.run_uid in test_uids else "train"
            )

    if set(split_by_uid) != {record.run_uid for record in records}:
        raise AssertionError("Some discovered runs were not assigned a status.")
    return split_by_uid


def resample_ratio(original_rate: float, target_rate: float) -> tuple[int, int]:
    ratio = Fraction(target_rate / original_rate).limit_denominator(1000)
    return ratio.numerator, ratio.denominator


def _resample_run(
    raw_signal: np.ndarray, original_rate: float, target_rate: float
) -> np.ndarray:
    up, down = resample_ratio(original_rate, target_rate)
    return resample_poly(raw_signal, up, down, axis=-1).astype(np.float32, copy=False)


def window_start_indices(
    start_seconds: float,
    end_seconds: float,
    available_samples: int,
    sample_rate: float,
    window_seconds: float,
    step_seconds: float,
) -> list[int]:
    window_samples = int(round(window_seconds * sample_rate))
    step_samples = int(round(step_seconds * sample_rate))
    start_index = max(0, int(round(start_seconds * sample_rate)))
    end_index = min(available_samples, int(round(end_seconds * sample_rate)))
    if end_index - start_index < window_samples:
        return []
    return list(range(start_index, end_index - window_samples + 1, step_samples))


def _serializable_run_record(
    record: RunRecord,
    split: str,
    no_walking_windows: int,
    walking_windows: int,
) -> dict[str, Any]:
    row = asdict(record)
    row["hdf5_path"] = str(record.hdf5_path)
    row["split"] = split
    row["no_walking_windows"] = no_walking_windows
    row["walking_windows"] = walking_windows
    return row


def build_combined_artifact(config: dict[str, Any]) -> dict[str, object]:
    records = discover_run_records(config)
    split_by_uid = assign_run_splits(
        records=records,
        seed=config["seed"],
        test_ratio=config["test_ratio"],
    )

    included_subjects = sorted(
        {
            record.subject_id
            for record in records
            if split_by_uid[record.run_uid] in {"train", "test"}
        }
    )
    expected_subjects = ["001", "002", "003", "004", "005", "007", "008"]
    if included_subjects != expected_subjects:
        raise ValueError(
            f"Unexpected included subjects: {included_subjects}; expected {expected_subjects}."
        )

    class_names = ["no_walking"] + [
        f"subject_{subject_id}_walking" for subject_id in included_subjects
    ]
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    # Config channels are 1-indexed; numpy needs 0-indexed. CLAUDE.md requires this
    # conversion to be asserted, not merely correct: a silent off-by-one shifts every
    # channel, keeps the array shape, still trains, and is undetectable downstream.
    channel_indices = [channel - 1 for channel in config["selected_channels"]]
    assert_channel_index_conversion(config["selected_channels"], channel_indices)
    target_rate = config["target_sample_rate"]
    window_samples = int(round(config["window_seconds"] * target_rate))

    samples: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []
    windows_by_run_activity: Counter[tuple[str, str]] = Counter()

    records_by_hdf5: dict[Path, list[RunRecord]] = defaultdict(list)
    for record in records:
        if split_by_uid[record.run_uid] in {"train", "test"}:
            records_by_hdf5[record.hdf5_path].append(record)

    for hdf5_path, path_records in sorted(records_by_hdf5.items(), key=lambda item: str(item[0])):
        with h5py.File(hdf5_path, "r") as handle:
            dataset = handle["experiment/data"]
            for record in sorted(path_records, key=lambda item: item.run_index):
                raw_signal = dataset[record.run_index, channel_indices, :]
                signal = _resample_run(
                    raw_signal=raw_signal,
                    original_rate=record.effective_sample_rate,
                    target_rate=target_rate,
                )
                available_duration = signal.shape[-1] / target_rate
                walking_end = record.data_end if record.data_end > 0 else available_duration
                intervals = [
                    (
                        "no_walking",
                        record.no_step_start,
                        record.no_step_end,
                    ),
                    (
                        f"subject_{record.subject_id}_walking",
                        record.data_start,
                        walking_end,
                    ),
                ]

                if record.no_step_end > record.data_start + 1e-9:
                    raise ValueError(
                        f"No-walking interval overlaps walking interval for {record.run_uid}: "
                        f"{record.no_step_end} > {record.data_start}."
                    )

                for class_name, start_seconds, end_seconds in intervals:
                    starts = window_start_indices(
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        available_samples=signal.shape[-1],
                        sample_rate=target_rate,
                        window_seconds=config["window_seconds"],
                        step_seconds=config["step_seconds"],
                    )
                    if not starts:
                        raise ValueError(
                            f"Interval {class_name} for {record.run_uid} produced no windows "
                            f"({start_seconds:.3f}-{end_seconds:.3f}s, available={available_duration:.3f}s)."
                        )
                    for start_index in starts:
                        sample = signal[:, start_index : start_index + window_samples]
                        if sample.shape != (len(channel_indices), window_samples):
                            raise AssertionError(
                                f"Unexpected window shape for {record.run_uid}: {sample.shape}."
                            )
                        if not np.isfinite(sample).all():
                            raise ValueError(f"Non-finite signal values in {record.run_uid}.")
                        samples.append(sample)
                        targets.append(class_to_index[class_name])
                        metadata.append(
                            {
                                "source_id": record.source_id,
                                "subject_id": record.subject_id,
                                "run_index": record.run_index,
                                "run_uid": record.run_uid,
                                "direction": record.direction,
                                "activity": class_name,
                                "start_time": start_index / target_rate,
                                "split": split_by_uid[record.run_uid],
                            }
                        )
                        windows_by_run_activity[(record.run_uid, class_name)] += 1

    sample_array = np.stack(samples).astype(np.float32, copy=False)
    target_array = np.asarray(targets, dtype=np.int64)
    train_indices = [index for index, item in enumerate(metadata) if item["split"] == "train"]
    test_indices = [index for index, item in enumerate(metadata) if item["split"] == "test"]
    if not train_indices or not test_indices:
        raise RuntimeError("The combined artifact has an empty train or test split.")

    split_audit = assert_split_integrity(
        {"train_indices": train_indices, "test_indices": test_indices, "metadata": metadata},
        "classification artifact (build)",
    )

    train_samples = sample_array[train_indices]
    channel_mean = train_samples.mean(axis=(0, 2), keepdims=True)
    channel_std = train_samples.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std < 1e-8, 1.0, channel_std)

    run_manifest = []
    for record in records:
        no_walking_count = windows_by_run_activity[(record.run_uid, "no_walking")]
        walking_count = windows_by_run_activity[
            (record.run_uid, f"subject_{record.subject_id}_walking")
        ]
        run_manifest.append(
            _serializable_run_record(
                record=record,
                split=split_by_uid[record.run_uid],
                no_walking_windows=no_walking_count,
                walking_windows=walking_count,
            )
        )

    class_window_counts: dict[str, dict[str, int]] = {}
    for class_name, class_index in class_to_index.items():
        class_window_counts[class_name] = {
            "all": int(np.sum(target_array == class_index)),
            "train": int(np.sum(target_array[train_indices] == class_index)),
            "test": int(np.sum(target_array[test_indices] == class_index)),
        }

    status_counts = Counter(split_by_uid.values())
    raw_skip_count = sum(record.skip_run for record in records)
    summary = {
        "experiment": "combined_centralized_9ch_no006_nowalking_80_20",
        "num_recorded_runs": len(records),
        "num_raw_skip_flags": int(raw_skip_count),
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
        "class_names": class_names,
        "class_window_counts": class_window_counts,
        "target_sample_rate": target_rate,
        "window_seconds": config["window_seconds"],
        "step_seconds": config["step_seconds"],
        "train_ratio": config["train_ratio"],
        "test_ratio": config["test_ratio"],
        "split_seed": config["seed"],
        "timing_assumption": config["timing_assumption"],
        "train_test_run_overlap": split_audit["train_test_run_overlap"],
    }

    if summary["num_usable_runs"] != 140:
        raise AssertionError(
            f"Expected 140 usable included runs, found {summary['num_usable_runs']}."
        )
    if summary["num_train_runs"] != 112 or summary["num_test_runs"] != 28:
        raise AssertionError(
            "Expected an exact 112/28 run split, found "
            f"{summary['num_train_runs']}/{summary['num_test_runs']}."
        )
    if summary["sample_shape"] != [9, 2000]:
        raise AssertionError(f"Unexpected artifact sample shape: {summary['sample_shape']}")

    return {
        "samples": torch.from_numpy(sample_array),
        "classification_targets": torch.from_numpy(target_array),
        "train_indices": torch.tensor(train_indices, dtype=torch.long),
        "test_indices": torch.tensor(test_indices, dtype=torch.long),
        "channel_mean": torch.from_numpy(channel_mean.astype(np.float32)),
        "channel_std": torch.from_numpy(channel_std.astype(np.float32)),
        "class_to_index": class_to_index,
        "index_to_class": class_names,
        "metadata": metadata,
        "run_manifest": run_manifest,
        "summary": summary,
    }


def save_combined_artifact(
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


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    collect_outputs: bool,
) -> tuple[float, float, torch.Tensor | None, torch.Tensor | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    output_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    for samples, targets in loader:
        samples = samples.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(samples)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = samples.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((torch.argmax(outputs, dim=1) == targets).sum().item())
        total_examples += batch_size
        if collect_outputs:
            output_batches.append(outputs.detach().cpu())
            target_batches.append(targets.detach().cpu())

    outputs_tensor = torch.cat(output_batches) if output_batches else None
    targets_tensor = torch.cat(target_batches) if target_batches else None
    return (
        total_loss / max(total_examples, 1),
        total_correct / max(total_examples, 1),
        outputs_tensor,
        targets_tensor,
    )


def _confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def _classification_metrics(matrix: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[class_name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
    }


def plot_loss_accuracy(history: Sequence[dict[str, float]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    losses = [float(row["train_loss"]) for row in history]
    accuracies = [100.0 * float(row["train_accuracy"]) for row in history]

    fig, loss_axis = plt.subplots(figsize=(9, 5.5))
    accuracy_axis = loss_axis.twinx()
    loss_line = loss_axis.plot(epochs, losses, color="#c43c39", linewidth=2, label="Training loss")
    accuracy_line = accuracy_axis.plot(
        epochs,
        accuracies,
        color="#2b6cb0",
        linewidth=2,
        label="Training accuracy",
    )
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss", color="#c43c39")
    accuracy_axis.set_ylabel("Accuracy (%)", color="#2b6cb0")
    loss_axis.tick_params(axis="y", labelcolor="#c43c39")
    accuracy_axis.tick_params(axis="y", labelcolor="#2b6cb0")
    accuracy_axis.set_ylim(0, 100)
    loss_axis.grid(alpha=0.25)
    lines = loss_line + accuracy_line
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    loss_axis.set_title("Combined Centralized CNN Training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray, class_names: Sequence[str], output_path: Path
) -> None:
    display_names = [
        "No walking" if name == "no_walking" else name.replace("subject_", "S").replace("_walking", "")
        for name in class_names
    ]
    fig, axis = plt.subplots(figsize=(9, 7.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(display_names)), display_names, rotation=45, ha="right")
    axis.set_yticks(range(len(display_names)), display_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Combined CNN Test Confusion Matrix")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_history(history: Sequence[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _write_confusion_matrix(
    matrix: np.ndarray, class_names: Sequence[str], path: Path
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *[int(value) for value in row]])


def _write_test_predictions(
    artifact: dict[str, object], test_indices: Sequence[int], predictions: np.ndarray, path: Path
) -> None:
    class_names: list[str] = artifact["index_to_class"]
    metadata: list[dict[str, Any]] = artifact["metadata"]
    targets = artifact["classification_targets"]
    fieldnames = [
        "artifact_index",
        "source_id",
        "subject_id",
        "run_index",
        "run_uid",
        "direction",
        "activity",
        "start_time",
        "true_class",
        "predicted_class",
        "correct",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for artifact_index, prediction in zip(test_indices, predictions):
            item = metadata[int(artifact_index)]
            target = int(targets[int(artifact_index)])
            writer.writerow(
                {
                    "artifact_index": int(artifact_index),
                    "source_id": item["source_id"],
                    "subject_id": item["subject_id"],
                    "run_index": item["run_index"],
                    "run_uid": item["run_uid"],
                    "direction": item["direction"],
                    "activity": item["activity"],
                    "start_time": item["start_time"],
                    "true_class": class_names[target],
                    "predicted_class": class_names[int(prediction)],
                    "correct": int(target == int(prediction)),
                }
            )


def train_and_evaluate(
    artifact: dict[str, object], config: dict[str, Any]
) -> dict[str, Any]:
    seeding = set_random_seeds(config["seed"])
    assert_determinism_flags()
    assert_split_integrity(artifact, "classification artifact (load)")
    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_indices = artifact["train_indices"].tolist()
    test_indices = artifact["test_indices"].tolist()
    train_dataset = CombinedArtifactDataset(artifact, train_indices)
    test_dataset = CombinedArtifactDataset(artifact, test_indices)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN1D(
        in_channels=int(artifact["samples"].shape[1]),
        output_dim=len(artifact["index_to_class"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
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

    history: list[dict[str, float]] = []
    for epoch in range(1, config["epochs"] + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss, train_accuracy, _, _ = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            collect_outputs=False,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"Epoch {epoch:03d}/{config['epochs']}: "
            f"loss={train_loss:.6f}, accuracy={train_accuracy:.4f}, lr={learning_rate:.8f}",
            flush=True,
        )
        scheduler.step()

    test_loss, test_accuracy, test_outputs, test_targets = _run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        collect_outputs=True,
    )
    if test_outputs is None or test_targets is None:
        raise RuntimeError("Test evaluation did not return predictions.")
    predictions = torch.argmax(test_outputs, dim=1).numpy()
    targets = test_targets.numpy()
    class_names: list[str] = artifact["index_to_class"]
    matrix = _confusion_matrix(targets, predictions, len(class_names))
    metrics = _classification_metrics(matrix, class_names)

    model_path = output_dir / "combined_centralized_cnn_final.pt"
    history_path = output_dir / "training_history.csv"
    summary_path = output_dir / "training_summary.json"
    curve_path = output_dir / "loss_accuracy_vs_epoch.png"
    confusion_plot_path = output_dir / "test_confusion_matrix.png"
    confusion_csv_path = output_dir / "test_confusion_matrix.csv"
    predictions_path = output_dir / "test_predictions.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_index": artifact["class_to_index"],
            "selected_channels_one_based": artifact["summary"]["selected_channels_one_based"],
            "channel_mean": artifact["channel_mean"],
            "channel_std": artifact["channel_std"],
            "sample_shape": artifact["summary"]["sample_shape"],
            "seed": config["seed"],
            "epochs": config["epochs"],
        },
        model_path,
    )
    _write_history(history, history_path)
    plot_loss_accuracy(history, curve_path)
    plot_confusion_matrix(matrix, class_names, confusion_plot_path)
    _write_confusion_matrix(matrix, class_names, confusion_csv_path)
    _write_test_predictions(artifact, test_indices, predictions, predictions_path)

    summary = {
        "experiment": artifact["summary"]["experiment"],
        "device": str(device),
        "seeding": seeding,
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "initial_learning_rate": config["learning_rate"],
        "minimum_learning_rate": config["minimum_learning_rate"],
        "weight_decay": config["weight_decay"],
        "test_evaluation_policy": "Test set evaluated once after the final epoch; no validation set.",
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "model_path": str(model_path),
        "history_path": str(history_path),
        "loss_accuracy_plot": str(curve_path),
        "confusion_matrix_plot": str(confusion_plot_path),
        "confusion_matrix_csv": str(confusion_csv_path),
        "test_predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary

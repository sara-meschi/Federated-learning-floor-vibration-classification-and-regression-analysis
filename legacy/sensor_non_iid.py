from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/sensor_non_iid.py"
raise ImportError(
    "Quarantined legacy module: legacy/sensor_non_iid.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from .config import ExperimentConfig


@dataclass(frozen=True)
class SensorClientPartition:
    client_id: str
    sensor_names: list[str]
    channel_indices: list[int]
    train_indices: list[int]


@dataclass(frozen=True)
class SensorWindowExample:
    sample: np.ndarray
    speed: float
    subject_id: str
    run_index: int
    start_time: float
    end_time: float
    split: str


def _as_index_list(indices: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(indices, torch.Tensor):
        return [int(index) for index in indices.tolist()]
    return [int(index) for index in indices]


def _load_sensor_non_iid_settings(config: ExperimentConfig) -> dict[str, object]:
    import yaml

    payload = yaml.safe_load(config.config_path.read_text()) or {}
    settings = payload.get("sensor_non_iid", {}) or {}
    if not isinstance(settings, dict):
        raise ValueError("sensor_non_iid must be a mapping when provided.")
    return settings


def resolve_sensor_client_map(config: ExperimentConfig) -> dict[str, list[str]]:
    if config.federated is None:
        raise ValueError("Sensor non-IID partitioning requires a federated config.")

    settings = _load_sensor_non_iid_settings(config)
    raw_client_sensors = settings.get("client_sensors")
    if raw_client_sensors is None:
        selected_sensors = list(config.data.selected_sensors)
        num_clients = config.federated.num_clients
        if len(selected_sensors) % num_clients != 0:
            raise ValueError(
                f"Cannot split {len(selected_sensors)} sensors evenly across {num_clients} clients."
            )
        sensors_per_client = len(selected_sensors) // num_clients
        client_sensors = {
            str(client_id): selected_sensors[
                client_id * sensors_per_client : (client_id + 1) * sensors_per_client
            ]
            for client_id in range(num_clients)
        }
    else:
        if not isinstance(raw_client_sensors, dict):
            raise ValueError("sensor_non_iid.client_sensors must be a mapping of client id to sensors.")
        client_sensors = {
            str(client_id): [str(sensor_name) for sensor_name in sensor_names]
            for client_id, sensor_names in raw_client_sensors.items()
        }

    sensors_per_client = settings.get("sensors_per_client")
    expected_group_size = None if sensors_per_client is None else int(sensors_per_client)
    return _validate_sensor_client_map(
        client_sensors=client_sensors,
        selected_sensors=config.data.selected_sensors,
        num_clients=config.federated.num_clients,
        expected_group_size=expected_group_size,
    )


def _validate_sensor_client_map(
    client_sensors: dict[str, list[str]],
    selected_sensors: Sequence[str],
    num_clients: int,
    expected_group_size: int | None,
) -> dict[str, list[str]]:
    expected_client_ids = {str(client_id) for client_id in range(num_clients)}
    if set(client_sensors) != expected_client_ids:
        raise ValueError(
            f"Sensor client ids must match {sorted(expected_client_ids)}, got {sorted(client_sensors)}."
        )

    selected_sensor_set = set(selected_sensors)
    seen_sensors: list[str] = []
    for client_id in sorted(client_sensors, key=int):
        sensor_names = client_sensors[client_id]
        if not sensor_names:
            raise ValueError(f"Client {client_id} has no sensors.")
        if len(sensor_names) != len(set(sensor_names)):
            raise ValueError(f"Client {client_id} repeats at least one sensor: {sensor_names}.")
        if expected_group_size is not None and len(sensor_names) != expected_group_size:
            raise ValueError(
                f"Client {client_id} has {len(sensor_names)} sensors, expected {expected_group_size}."
            )
        unknown = sorted(set(sensor_names) - selected_sensor_set)
        if unknown:
            raise ValueError(f"Client {client_id} uses sensors not in selected_sensors: {unknown}.")
        seen_sensors.extend(sensor_names)

    duplicated = sorted({sensor_name for sensor_name in seen_sensors if seen_sensors.count(sensor_name) > 1})
    if duplicated:
        raise ValueError(f"Sensors assigned to more than one client: {duplicated}.")

    missing = sorted(selected_sensor_set - set(seen_sensors))
    if missing:
        raise ValueError(f"Selected sensors missing from client assignment: {missing}.")

    return {client_id: list(client_sensors[client_id]) for client_id in sorted(client_sensors, key=int)}


def _sensor_channel_positions(config: ExperimentConfig) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    next_position = 0
    for sensor_name in config.data.selected_sensors:
        channels = config.data.sensor_channel_map[sensor_name]
        positions[sensor_name] = list(range(next_position, next_position + len(channels)))
        next_position += len(channels)
    return positions


def _load_selected_channels(hdf5_path: Path, config: ExperimentConfig) -> tuple[np.ndarray, float]:
    import h5py

    from .preprocessing import _get_general_parameter, _resolve_channel_selection

    selected_channels, _ = _resolve_channel_selection(config)
    zero_based_channels = [channel - 1 for channel in selected_channels]
    sorted_positions = sorted(range(len(zero_based_channels)), key=lambda idx: zero_based_channels[idx])
    sorted_channels = [zero_based_channels[idx] for idx in sorted_positions]
    restore_order = np.argsort(sorted_positions)

    with h5py.File(hdf5_path, "r") as handle:
        data = handle["experiment/data"][:, sorted_channels, :]
        data = data[:, restore_order, :]
        general_parameters = handle["experiment/general_parameters"][:]
        original_rate = _get_general_parameter(general_parameters, "fs")
        if original_rate is None:
            raise ValueError(f"Missing sample rate in {hdf5_path}")
    return data, float(original_rate)


def _split_raw_run_indices(
    start_sec: float,
    end_sec: float,
    original_rate: float,
    trial_length: int,
    train_ratio: float,
    purge_gap_seconds: float,
    window_seconds: float,
) -> dict[str, tuple[int, int]]:
    raw_start = max(0, int(round(start_sec * original_rate)))
    raw_end = min(int(round(end_sec * original_rate)), trial_length)
    if raw_end <= raw_start:
        return {}

    minimum_split_samples = int(np.ceil(window_seconds * original_rate))
    purge_samples = max(0, int(round(purge_gap_seconds * original_rate)))
    left_gap = purge_samples // 2
    right_gap = purge_samples - left_gap

    lower_split = raw_start + minimum_split_samples + left_gap
    upper_split = raw_end - minimum_split_samples - right_gap
    if lower_split > upper_split:
        return {}

    desired_split = raw_start + int(round((raw_end - raw_start) * train_ratio))
    split_idx = min(max(desired_split, lower_split), upper_split)

    train_end = max(raw_start, split_idx - left_gap)
    test_start = min(raw_end, split_idx + right_gap)
    return {
        "train": (raw_start, train_end),
        "test": (test_start, raw_end),
    }


def _window_resampled_segment(
    segment: np.ndarray,
    speed: float,
    subject_id: str,
    run_index: int,
    split: str,
    segment_start_time: float,
    segment_end_time: float,
    sample_rate: float,
    window_seconds: float,
    step_seconds: float,
) -> list[SensorWindowExample]:
    window_size = int(round(window_seconds * sample_rate))
    step_size = int(round(step_seconds * sample_rate))
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_seconds and step_seconds must produce positive sample counts.")

    examples: list[SensorWindowExample] = []
    for local_start in range(0, segment.shape[-1] - window_size + 1, step_size):
        local_end = local_start + window_size
        start_time = segment_start_time + local_start / sample_rate
        end_time = start_time + window_size / sample_rate
        if end_time > segment_end_time + 1e-9:
            continue
        examples.append(
            SensorWindowExample(
                sample=segment[:, local_start:local_end].astype(np.float32, copy=False),
                speed=float(speed),
                subject_id=subject_id,
                run_index=run_index,
                start_time=float(start_time),
                end_time=float(end_time),
                split=split,
            )
        )
    return examples


def count_train_test_window_overlaps(metadata: Sequence[dict[str, object]]) -> int:
    grouped: dict[tuple[str, int], dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {"train": [], "test": []}
    )
    for item in metadata:
        group = (str(item["subject_id"]), int(item["run_index"]))
        split = str(item["split"])
        if split not in grouped[group]:
            continue
        grouped[group][split].append((float(item["start_time"]), float(item["end_time"])))

    overlaps = 0
    eps = 1e-9
    for split_intervals in grouped.values():
        train_intervals = sorted(split_intervals["train"])
        test_intervals = sorted(split_intervals["test"])
        test_index = 0
        for train_start, train_end in train_intervals:
            while test_index < len(test_intervals) and test_intervals[test_index][1] <= train_start + eps:
                test_index += 1
            probe_index = test_index
            while probe_index < len(test_intervals) and test_intervals[probe_index][0] < train_end - eps:
                test_start, test_end = test_intervals[probe_index]
                if train_start < test_end - eps and test_start < train_end - eps:
                    overlaps += 1
                probe_index += 1
    return overlaps


def _load_purge_gap_seconds(config: ExperimentConfig) -> float:
    settings = _load_sensor_non_iid_settings(config)
    return float(settings.get("purge_gap_seconds", 0.0))


def _load_excluded_subjects(config: ExperimentConfig) -> set[str]:
    settings = _load_sensor_non_iid_settings(config)
    raw_subjects = settings.get("exclude_subjects", []) or []
    excluded: set[str] = set()
    for subject_id in raw_subjects:
        subject_text = str(subject_id).strip()
        if subject_text.isdigit():
            subject_text = f"{int(subject_text):03d}"
        excluded.add(subject_text)
    return excluded


def build_sensor_non_iid_artifact(config: ExperimentConfig) -> dict[str, object]:
    from .preprocessing import (
        _find_apdm_csv,
        _load_run_segments,
        _parse_speed_labels,
        _resolve_channel_selection,
        _resample_trials,
        _subject_id_from_path,
    )

    if config.federated is None:
        raise ValueError("Sensor non-IID preprocessing requires a federated config.")
    if abs(config.data.val_ratio) > 1e-9:
        raise ValueError("Sensor non-IID preprocessing has no validation split; set data.val_ratio to 0.0.")
    ratio_sum = config.data.train_ratio + config.data.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"data.train_ratio + data.test_ratio must equal 1.0, got {ratio_sum:.6f}.")

    client_sensors = resolve_sensor_client_map(config)
    sensor_channel_positions = _sensor_channel_positions(config)
    selected_channels, channel_labels = _resolve_channel_selection(config)
    purge_gap_seconds = _load_purge_gap_seconds(config)
    excluded_subjects = _load_excluded_subjects(config)

    dataset_root = config.data.dataset_root
    run_segments = _load_run_segments(dataset_root / "runPeramiters.csv")
    hdf5_files = sorted((dataset_root / "vib" / "Data").glob("*.hdf5"))
    if config.data.subject_limit is not None:
        hdf5_files = hdf5_files[: config.data.subject_limit]

    examples: list[SensorWindowExample] = []
    skipped_subjects: list[str] = []
    excluded_subjects_seen: list[str] = []
    skipped_short_splits: list[dict[str, object]] = []

    for hdf5_path in hdf5_files:
        subject_id = _subject_id_from_path(hdf5_path)
        if subject_id in excluded_subjects:
            excluded_subjects_seen.append(subject_id)
            continue
        apdm_csv = _find_apdm_csv(dataset_root, subject_id)
        if apdm_csv is None:
            skipped_subjects.append(subject_id)
            continue

        speeds = _parse_speed_labels(apdm_csv)
        data, original_rate = _load_selected_channels(hdf5_path, config)
        run_count = min(len(data), len(speeds))

        for run_index in range(run_count):
            segment = run_segments.get((subject_id, run_index))
            if segment is None or int(segment["skip_run"]) == 1:
                continue

            end_sec = float(segment["data_end"])
            if end_sec <= 0:
                end_sec = data.shape[-1] / original_rate

            raw_ranges = _split_raw_run_indices(
                start_sec=float(segment["data_start"]),
                end_sec=end_sec,
                original_rate=original_rate,
                trial_length=data.shape[-1],
                train_ratio=config.data.train_ratio,
                purge_gap_seconds=purge_gap_seconds,
                window_seconds=config.data.window_seconds,
            )
            for split_name, (raw_start, raw_end) in raw_ranges.items():
                raw_segment = data[run_index, :, raw_start:raw_end]
                resampled_segment = _resample_trials(
                    raw_segment,
                    original_rate,
                    config.data.target_sample_rate,
                )
                split_examples = _window_resampled_segment(
                    segment=resampled_segment,
                    speed=speeds[run_index],
                    subject_id=subject_id,
                    run_index=run_index,
                    split=split_name,
                    segment_start_time=raw_start / original_rate,
                    segment_end_time=raw_end / original_rate,
                    sample_rate=config.data.target_sample_rate,
                    window_seconds=config.data.window_seconds,
                    step_seconds=config.data.step_seconds,
                )
                if not split_examples:
                    skipped_short_splits.append(
                        {
                            "subject_id": subject_id,
                            "run_index": run_index,
                            "split": split_name,
                            "duration_seconds": (raw_end - raw_start) / original_rate,
                        }
                    )
                examples.extend(split_examples)

    if not examples:
        raise RuntimeError("No examples were created. Check paths, split lengths, and label parsing.")

    samples = np.stack([example.sample for example in examples]).astype(np.float32, copy=False)
    subject_ids = sorted({example.subject_id for example in examples})
    subject_to_class = {subject_id: idx for idx, subject_id in enumerate(subject_ids)}
    regression_targets = np.array([example.speed for example in examples], dtype=np.float32)
    classification_targets = np.array(
        [subject_to_class[example.subject_id] for example in examples],
        dtype=np.int64,
    )
    metadata = [
        {
            "subject_id": example.subject_id,
            "run_index": example.run_index,
            "start_time": example.start_time,
            "end_time": example.end_time,
            "split": example.split,
        }
        for example in examples
    ]

    train_indices = [index for index, item in enumerate(metadata) if item["split"] == "train"]
    test_indices = [index for index, item in enumerate(metadata) if item["split"] == "test"]
    val_indices: list[int] = []
    if not train_indices:
        raise RuntimeError("No training windows were created.")
    if not test_indices:
        raise RuntimeError("No testing windows were created.")

    train_test_overlap = len(set(train_indices) & set(test_indices))
    window_overlap_count = count_train_test_window_overlaps(metadata)
    if train_test_overlap or window_overlap_count:
        raise ValueError(
            "Detected train/test leakage in sensor non-IID artifact: "
            f"index_overlap={train_test_overlap}, window_overlap={window_overlap_count}."
        )

    train_samples = samples[train_indices]
    channel_mean = train_samples.mean(axis=(0, 2), keepdims=True)
    channel_std = train_samples.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)

    split_run_groups = defaultdict(set)
    for item in metadata:
        split_run_groups[str(item["split"])].add((str(item["subject_id"]), int(item["run_index"])))

    client_channel_indices = {
        client_id: sorted(
            chain.from_iterable(sensor_channel_positions[sensor_name] for sensor_name in sensor_names)
        )
        for client_id, sensor_names in client_sensors.items()
    }
    summary = {
        "partition_mode": "sensor_owned_feature_non_iid",
        "num_examples": int(len(samples)),
        "num_train": int(len(train_indices)),
        "num_val": 0,
        "num_test": int(len(test_indices)),
        "sample_shape": list(samples.shape[1:]),
        "subjects": subject_ids,
        "skipped_subjects": skipped_subjects,
        "excluded_subjects": sorted(excluded_subjects_seen),
        "skipped_short_splits": skipped_short_splits,
        "selected_sensors": config.data.selected_sensors,
        "selected_channels": selected_channels,
        "channel_labels": channel_labels,
        "sensor_channel_positions": sensor_channel_positions,
        "sensor_client_map": client_sensors,
        "client_channel_indices": client_channel_indices,
        "split_group_counts": {
            split_name: len(groups)
            for split_name, groups in sorted(split_run_groups.items())
        },
        "split_overlap": {
            "train_val": 0,
            "train_test": train_test_overlap,
            "val_test": 0,
            "overlapping_train_test_windows": window_overlap_count,
        },
        "split_rule": (
            "Each subject/run raw interval is split into train/test before resampling and "
            "before windowing. The split boundary is clamped when needed so both sides "
            "can contain at least one full window. Windows are generated only inside "
            "one split, and the final server test set is evaluated only after federated "
            "training."
        ),
        "train_ratio": config.data.train_ratio,
        "val_ratio": 0.0,
        "test_ratio": config.data.test_ratio,
        "purge_gap_seconds": purge_gap_seconds,
        "split_before_resampling": True,
        "split_before_windowing": True,
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
        "sensor_channel_positions": sensor_channel_positions,
        "sensor_client_map": client_sensors,
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
            "val_ratio": 0.0,
            "test_ratio": config.data.test_ratio,
            "purge_gap_seconds": purge_gap_seconds,
            "exclude_subjects": sorted(excluded_subjects),
            "sensor_client_map": client_sensors,
        },
    }


class SensorMaskedIndexedArtifactDataset(Dataset):
    def __init__(
        self,
        artifact: dict[str, object],
        indices: Sequence[int] | torch.Tensor,
        task: str,
        channel_indices: Sequence[int] | None = None,
        include_channel_mask: bool = False,
    ) -> None:
        self.indices = _as_index_list(indices)
        self.samples = artifact["samples"]
        self.mean = artifact["channel_mean"]
        self.std = artifact["channel_std"]
        if task == "regression":
            self.targets = artifact["regression_targets"]
        elif task == "classification":
            self.targets = artifact["classification_targets"]
        else:
            raise ValueError(f"Unsupported task: {task}")
        self.task = task
        self.include_channel_mask = include_channel_mask

        num_channels = int(self.samples.shape[1])
        self.channel_mask: torch.Tensor | None = None
        if channel_indices is not None:
            channel_indices = [int(index) for index in channel_indices]
            unknown = [index for index in channel_indices if index < 0 or index >= num_channels]
            if unknown:
                raise ValueError(f"Channel mask indices outside sample shape: {unknown}.")
            self.channel_mask = torch.zeros(num_channels, dtype=torch.bool)
            self.channel_mask[channel_indices] = True

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_index = self.indices[index]
        x = self.samples[item_index]
        x = (x - self.mean.squeeze(0)) / self.std.squeeze(0)
        active_mask = self.channel_mask
        if active_mask is not None:
            x = x.clone()
            x[~active_mask] = 0.0
        elif self.include_channel_mask:
            active_mask = torch.ones(x.shape[0], dtype=torch.bool)

        if self.include_channel_mask:
            assert active_mask is not None
            mask_channels = active_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1).expand_as(x)
            x = torch.cat((x, mask_channels), dim=0)

        y = self.targets[item_index]
        if self.task == "regression":
            y = y.unsqueeze(0)
        return x.float(), y


def build_sensor_masked_loader(
    artifact: dict[str, object],
    indices: Sequence[int] | torch.Tensor,
    task: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    channel_indices: Sequence[int] | None = None,
    include_channel_mask: bool = False,
) -> DataLoader:
    dataset = SensorMaskedIndexedArtifactDataset(
        artifact=artifact,
        indices=indices,
        task=task,
        channel_indices=channel_indices,
        include_channel_mask=include_channel_mask,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def build_sensor_client_partitions(
    artifact: dict[str, object],
    num_clients: int,
    client_sensors: dict[str, list[str]],
) -> list[SensorClientPartition]:
    train_indices = _as_index_list(artifact["train_indices"])
    val_indices = set(_as_index_list(artifact["val_indices"]))
    test_indices = set(_as_index_list(artifact["test_indices"]))
    if set(train_indices) & val_indices:
        raise ValueError("Global train indices overlap validation indices.")
    if set(train_indices) & test_indices:
        raise ValueError("Global train indices overlap server test indices.")

    selected_sensors = [str(sensor_name) for sensor_name in artifact["summary"]["selected_sensors"]]
    sensor_channel_positions = {
        str(sensor_name): [int(index) for index in channel_indices]
        for sensor_name, channel_indices in artifact["sensor_channel_positions"].items()
    }
    resolved_client_sensors = _validate_sensor_client_map(
        client_sensors=client_sensors,
        selected_sensors=selected_sensors,
        num_clients=num_clients,
        expected_group_size=None,
    )

    partitions: list[SensorClientPartition] = []
    for client_id, sensor_names in resolved_client_sensors.items():
        channel_indices = sorted(
            chain.from_iterable(sensor_channel_positions[sensor_name] for sensor_name in sensor_names)
        )
        if not channel_indices:
            raise ValueError(f"Client {client_id} has no channel indices.")
        partitions.append(
            SensorClientPartition(
                client_id=client_id,
                sensor_names=list(sensor_names),
                channel_indices=channel_indices,
                train_indices=list(train_indices),
            )
        )
    return partitions


def create_sensor_partition_summary(
    artifact: dict[str, object],
    client_partitions: Sequence[SensorClientPartition],
) -> dict[str, object]:
    train_indices = _as_index_list(artifact["train_indices"])
    val_indices = _as_index_list(artifact["val_indices"])
    test_indices = _as_index_list(artifact["test_indices"])
    clients = [
        {
            "client_id": partition.client_id,
            "sensors": partition.sensor_names,
            "channel_indices": partition.channel_indices,
            "num_train_windows": len(partition.train_indices),
        }
        for partition in client_partitions
    ]
    return {
        "partition_mode": "sensor_owned_feature_non_iid",
        "clients": clients,
        "num_global_train": len(train_indices),
        "num_global_val": len(val_indices),
        "num_global_test": len(test_indices),
        "total_client_train_window_views": sum(client["num_train_windows"] for client in clients),
        "client_training_window_indices_are_shared": True,
        "client_sensor_channels_are_disjoint": True,
        "split_overlap": artifact["summary"].get("split_overlap", {}),
        "split_rule": (
            "All clients share the same global training window ids, but each client loader "
            "receives only its owned sensor channels; all other channels are zeroed after "
            "normalization. The held-out server test windows come from raw time intervals "
            "that were split before resampling/windowing."
        ),
    }


def save_sensor_artifact(artifact: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(artifact["summary"], indent=2))

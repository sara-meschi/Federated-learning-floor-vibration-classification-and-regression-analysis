#!/usr/bin/env python
"""
Lightweight checks for the sensor-owned non-IID partitioning path.

This uses a synthetic artifact, so it does not require the HDF5 dataset or Flower.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.federated import create_federated_model
from redo_by_sara.sensor_non_iid import (
    build_sensor_client_partitions,
    build_sensor_masked_loader,
    count_train_test_window_overlaps,
    create_sensor_partition_summary,
)


def _synthetic_artifact() -> dict[str, object]:
    selected_sensors = [str(index) for index in range(1, 21)]
    samples = torch.arange(10 * 20 * 4, dtype=torch.float32).reshape(10, 20, 4)
    metadata = []
    for index in range(10):
        split = "train" if index < 8 else "test"
        metadata.append(
            {
                "subject_id": "001",
                "run_index": 0,
                "start_time": float(index * 5),
                "end_time": float(index * 5 + 5),
                "split": split,
            }
        )
    return {
        "samples": samples,
        "regression_targets": torch.arange(10, dtype=torch.float32),
        "classification_targets": torch.zeros(10, dtype=torch.long),
        "train_indices": torch.arange(0, 8, dtype=torch.long),
        "val_indices": torch.tensor([], dtype=torch.long),
        "test_indices": torch.arange(8, 10, dtype=torch.long),
        "channel_mean": torch.zeros(1, 20, 1),
        "channel_std": torch.ones(1, 20, 1),
        "subject_to_class": {"001": 0},
        "metadata": metadata,
        "sensor_channel_positions": {
            sensor_name: [sensor_index]
            for sensor_index, sensor_name in enumerate(selected_sensors)
        },
        "summary": {
            "selected_sensors": selected_sensors,
            "num_train": 8,
            "num_val": 0,
            "num_test": 2,
            "sample_shape": [20, 4],
            "split_overlap": {
                "train_val": 0,
                "train_test": 0,
                "val_test": 0,
                "overlapping_train_test_windows": 0,
            },
        },
    }


def test_sensor_partitions() -> None:
    artifact = _synthetic_artifact()
    client_sensors = {
        "0": ["1", "2", "3", "4", "5"],
        "1": ["6", "7", "8", "9", "10"],
        "2": ["11", "12", "13", "14", "15"],
        "3": ["16", "17", "18", "19", "20"],
    }
    partitions = build_sensor_client_partitions(
        artifact=artifact,
        num_clients=4,
        client_sensors=client_sensors,
    )
    assert len(partitions) == 4
    assert [partition.channel_indices for partition in partitions] == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ]
    for partition in partitions:
        assert partition.train_indices == list(range(8))

    summary = create_sensor_partition_summary(artifact, partitions)
    assert summary["num_global_val"] == 0
    assert summary["num_global_test"] == 2
    assert summary["total_client_train_window_views"] == 32
    print("✓ Sensor client partition summary is valid")


def test_masked_loader() -> None:
    artifact = _synthetic_artifact()
    loader = build_sensor_masked_loader(
        artifact=artifact,
        indices=[1],
        task="classification",
        batch_size=1,
        num_workers=0,
        shuffle=False,
        seed=4601,
        channel_indices=[0, 1, 2, 3, 4],
    )
    x, y = next(iter(loader))
    assert y.item() == 0
    assert torch.count_nonzero(x[:, :5, :]).item() > 0
    assert torch.count_nonzero(x[:, 5:, :]).item() == 0
    print("✓ Masked loader exposes only the owned sensor channels")


def test_mask_aware_loader_and_model() -> None:
    artifact = _synthetic_artifact()
    loader = build_sensor_masked_loader(
        artifact=artifact,
        indices=[1],
        task="classification",
        batch_size=1,
        num_workers=0,
        shuffle=False,
        seed=4601,
        channel_indices=[0, 1, 2, 3, 4],
        include_channel_mask=True,
    )
    x, y = next(iter(loader))
    signal = x[:, :20, :]
    mask = x[:, 20:, :]

    assert y.item() == 0
    assert x.shape == (1, 40, 4)
    assert torch.count_nonzero(signal[:, :5, :]).item() > 0
    assert torch.count_nonzero(signal[:, 5:, :]).item() == 0
    assert torch.count_nonzero(mask[:, :5, :]).item() == 5 * 4
    assert torch.count_nonzero(mask[:, 5:, :]).item() == 0

    model = create_federated_model(artifact, "classification", mask_aware=True)
    with torch.no_grad():
        outputs = model(x)
    assert outputs.shape == (1, 1)
    print("✓ Mask-aware loader and model use signal-plus-mask channels")


def test_overlap_counter() -> None:
    artifact = _synthetic_artifact()
    assert count_train_test_window_overlaps(artifact["metadata"]) == 0
    leaky_metadata = list(artifact["metadata"])
    leaky_metadata[-1] = {
        "subject_id": "001",
        "run_index": 0,
        "start_time": 35.0,
        "end_time": 40.0,
        "split": "test",
    }
    assert count_train_test_window_overlaps(leaky_metadata) == 1
    print("✓ Train/test window-overlap counter catches boundary leakage")


if __name__ == "__main__":
    try:
        test_sensor_partitions()
        test_masked_loader()
        test_mask_aware_loader_and_model()
        test_overlap_counter()
    except Exception as exc:
        print("\n✗ Test FAILED with error:")
        print(f"  {type(exc).__name__}: {exc}")
        sys.exit(1)
    print("\n✓ Sensor non-IID partitioning tests PASSED!")

from __future__ import annotations

import numpy as np

from redo_by_sara.combined_centralized_classification import (
    assign_run_splits,
    discover_run_records,
)
from redo_by_sara.combined_centralized_regression import (
    load_regression_config,
    load_speed_labels,
    regression_metrics,
)


CONFIG_PATH = "configs/centralized_combined_regression_9ch_no006_walking_80_20_60e.yaml"


def test_all_usable_runs_have_speed_labels() -> None:
    config = load_regression_config(CONFIG_PATH)
    records = discover_run_records(config)
    split_by_uid = assign_run_splits(records, config["seed"], config["test_ratio"])
    speeds, _ = load_speed_labels(config, records, split_by_uid)
    usable = {
        record.run_uid
        for record in records
        if split_by_uid[record.run_uid] in {"train", "test"}
    }
    assert len(usable) == 140
    assert usable <= set(speeds)
    assert all(speeds[uid] > 0 for uid in usable)


def test_regression_metrics() -> None:
    actual = np.asarray([1.0, 2.0, 3.0])
    predicted = np.asarray([1.0, 2.0, 3.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics["mse"] == 0.0
    assert metrics["rmse_mps"] == 0.0
    assert metrics["mae_mps"] == 0.0
    assert metrics["bias_mps"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["within_0_10_mps_pct"] == 100.0

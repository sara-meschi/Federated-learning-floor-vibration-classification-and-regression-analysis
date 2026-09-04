from __future__ import annotations

import numpy as np
import pytest
import torch

from redo_by_sara.combined_residual_run_regression import (
    RunGroupedResidualDataset,
    aggregate_windows_by_run,
    audit_residual_run_setup,
    build_zero_initialized_residual_model,
    collate_whole_runs,
    compute_train_source_subject_means,
    load_residual_run_config,
)


CONFIG_PATH = "configs/centralized_combined_residual_runbalanced_9ch_no006_60e.yaml"


def _load() -> tuple[dict[str, object], dict[str, object]]:
    config = load_residual_run_config(CONFIG_PATH)
    artifact = torch.load(
        config["artifact_path"], map_location="cpu", weights_only=False
    )
    return config, artifact


@pytest.mark.requires_data
def test_residual_setup_exact_counts_and_means() -> None:
    _, artifact = _load()
    setup = audit_residual_run_setup(artifact)
    assert setup["num_source_subject_strata"] == 8
    assert setup["num_train_runs"] == 112
    assert setup["num_test_runs"] == 28
    assert setup["num_train_windows"] == 966
    assert setup["num_test_windows"] == 238
    assert setup["train_test_run_overlap"] == 0
    assert setup["minimum_windows_per_train_run"] == 6
    assert setup["maximum_windows_per_train_run"] == 11
    assert np.isclose(setup["train_residual_scale_mps"], 0.03161902033)

    means, scale, rows = compute_train_source_subject_means(artifact)
    assert len(means) == 8
    assert np.isclose(scale, 0.03161902033)
    assert np.isclose(means[("test_2", "001")], 1.137647058824)
    assert np.isclose(means[("testing_20251124", "007")], 1.325909090909)
    assert sum(int(row["num_train_runs"]) for row in rows) == 112
    assert max(abs(float(row["train_residual_mean_mps"])) for row in rows) < 1e-12


@pytest.mark.requires_data
def test_run_grouping_covers_each_window_once() -> None:
    _, artifact = _load()
    means, scale, _ = compute_train_source_subject_means(artifact)
    train_indices = artifact["train_indices"].tolist()
    dataset = RunGroupedResidualDataset(artifact, train_indices, means, scale)
    assert len(dataset) == 112
    grouped_indices = [index for run in dataset.runs for index in run["window_indices"]]
    assert len(grouped_indices) == 966
    assert len(set(grouped_indices)) == 966
    assert set(grouped_indices) == set(train_indices)

    batch = collate_whole_runs([dataset[0], dataset[1]])
    assert int(batch["window_counts"].sum()) == int(batch["samples"].shape[0])
    assert set(batch["run_assignment"].tolist()) == {0, 1}


def test_run_aggregation_and_equal_run_loss() -> None:
    predictions = torch.tensor([1.0, 3.0, 9.0, 9.0, 9.0], requires_grad=True)
    assignments = torch.tensor([0, 0, 1, 1, 1])
    aggregated = aggregate_windows_by_run(predictions, assignments, num_runs=2)
    assert torch.allclose(aggregated, torch.tensor([2.0, 9.0]))

    targets = torch.tensor([0.0, 10.0])
    loss = torch.mean((aggregated - targets) ** 2)
    assert torch.isclose(loss, torch.tensor(2.5))
    loss.backward()
    assert torch.allclose(
        predictions.grad, torch.tensor([1.0, 1.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])
    )


def test_zero_initialized_head_reproduces_zero_residual() -> None:
    model = build_zero_initialized_residual_model()
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(3, 9, 2000)).squeeze(-1)
    assert torch.equal(output, torch.zeros(3))

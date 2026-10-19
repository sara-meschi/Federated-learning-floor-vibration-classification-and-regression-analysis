from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from redo_by_sara.combined_iid_flower import (
    FEDADAM_HYPERPARAMETERS,
    _adam_step_count,
    _prepare_client_optimizer,
    _save_client_optimizer,
    aggregate_confusion_matrices,
    aggregate_regression_sufficient_metrics,
    cosine_round_learning_rate,
    gradient_step_budget,
    load_combined_iid_flower_config,
    local_epoch_learning_rate,
    round_learning_rate,
)

ROOT = Path(__file__).resolve().parent
IID_K3_CONFIG = ROOT / "configs/combined_iid_flower_3clients_60r.yaml"
NATURAL_K2_CONFIG = ROOT / "configs/combined_natural_flower_2clients_60r.yaml"
from redo_by_sara.combined_residual_run_regression import (
    build_zero_initialized_residual_model,
)
from redo_by_sara.parameters import get_parameters, set_parameters


def test_cosine_round_learning_rate_matches_declared_schedule() -> None:
    assert cosine_round_learning_rate(1, 1e-3, 1e-5, 60) == 1e-3
    assert math.isclose(
        cosine_round_learning_rate(60, 1e-3, 1e-5, 60),
        1.0678380296485953e-5,
        rel_tol=0.0,
        abs_tol=1e-16,
    )


def test_regression_metrics_are_aggregated_from_global_sufficient_statistics() -> None:
    metrics = aggregate_regression_sufficient_metrics(
        [
            {
                "n": 2,
                "sse": 0,
                "sae": 0,
                "error_sum": 0,
                "within_count": 2,
                "sum_y": 4,
                "sum_y2": 10,
            },
            {
                "n": 1,
                "sse": 4,
                "sae": 2,
                "error_sum": -2,
                "within_count": 0,
                "sum_y": 5,
                "sum_y2": 25,
            },
        ]
    )
    assert math.isclose(metrics["mse"], 4 / 3)
    assert math.isclose(metrics["rmse_mps"], math.sqrt(4 / 3))
    assert math.isclose(metrics["mae_mps"], 2 / 3)
    assert math.isclose(metrics["bias_mps"], -2 / 3)
    assert math.isclose(metrics["within_0_10_mps_pct"], 200 / 3)
    assert math.isclose(metrics["r2"], 0.5)


def test_classification_macro_metrics_are_computed_after_confusion_sum() -> None:
    first = np.asarray([[4, 0], [1, 1]])
    second = np.asarray([[0, 2], [0, 4]])
    matrix, metrics = aggregate_confusion_matrices([first, second], ["a", "b"])
    np.testing.assert_array_equal(matrix, np.asarray([[4, 2], [1, 5]]))
    assert math.isclose(metrics["accuracy"], 0.75)
    assert math.isclose(metrics["balanced_accuracy"], 0.75)
    expected_macro_f1 = ((2 * 0.8 * (4 / 6) / (0.8 + 4 / 6)) + (2 * (5 / 7) * (5 / 6) / (5 / 7 + 5 / 6))) / 2
    assert math.isclose(metrics["macro_f1"], expected_macro_f1)


def test_residual_model_zero_head_and_parameter_roundtrip() -> None:
    model = build_zero_initialized_residual_model()
    initial = get_parameters(model)
    assert np.count_nonzero(initial[-2]) == 0
    assert np.count_nonzero(initial[-1]) == 0
    restored = build_zero_initialized_residual_model()
    set_parameters(restored, initial)
    for expected, actual in zip(initial, get_parameters(restored)):
        np.testing.assert_array_equal(actual, expected)


# --- S3: config validation -------------------------------------------------------------


def test_config_defaults_are_the_pre_s3_behaviour() -> None:
    config = load_combined_iid_flower_config(IID_K3_CONFIG)
    assert config["aggregation"] == "FedAvg"
    assert config["client_optimizer_state"] == "fresh"
    assert config["lr_schedule"] == "cosine_over_rounds"
    assert config["server_optimizer"] is None
    assert config["client_num_cpus"] == 3.0


def test_config_accepts_the_s3_knobs() -> None:
    config = load_combined_iid_flower_config(
        IID_K3_CONFIG,
        {
            "federated.aggregation": "FedAdam",
            "federated.client_optimizer_state": "persistent",
            "federated.local_epochs": 5,
            "training.lr_schedule": "cosine_restart_per_round",
        },
    )
    assert config["aggregation"] == "FedAdam"
    assert config["local_epochs"] == 5
    assert config["client_optimizer_state"] == "persistent"
    assert config["lr_schedule"] == "cosine_restart_per_round"
    # Pre-registered in docs/s3_winner_rule.md §1; written out, not imported.
    assert config["server_optimizer"] == {"eta": 1e-2, "beta_1": 0.9, "beta_2": 0.99, "tau": 1e-3}
    assert FEDADAM_HYPERPARAMETERS == config["server_optimizer"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"federated.aggregation": "FedProx"},
        {"federated.client_optimizer_state": "warm"},
        {"training.lr_schedule": "step"},
        {"federated.local_epochs": 0},
        {"federated.num_clients": 4},
        {"federated.server_optimizer": {"eta": 0.1}},  # FedAvg has no server optimizer
        {"federated.aggregation": "FedAdam", "federated.server_optimizer": {"lr": 0.1}},
    ],
)
def test_config_rejects_invalid_s3_settings(overrides: dict) -> None:
    with pytest.raises(ValueError):
        load_combined_iid_flower_config(IID_K3_CONFIG, overrides)


def test_override_of_a_missing_section_fails_loudly() -> None:
    with pytest.raises(KeyError):
        load_combined_iid_flower_config(IID_K3_CONFIG, {"federaetd.local_epochs": 5})


# --- S3: learning-rate schedules -------------------------------------------------------


def test_cosine_over_rounds_is_unchanged() -> None:
    for server_round in (1, 17, 60):
        assert round_learning_rate("cosine_over_rounds", server_round, 1e-3, 1e-5, 60) == (
            cosine_round_learning_rate(server_round, 1e-3, 1e-5, 60)
        )
        assert local_epoch_learning_rate("cosine_over_rounds", 4e-4, 1e-5, 3, 5) == 4e-4


def test_constant_schedule() -> None:
    assert round_learning_rate("constant", 1, 1e-3, 1e-5, 60) == 1e-3
    assert round_learning_rate("constant", 60, 1e-3, 1e-5, 60) == 1e-3
    assert local_epoch_learning_rate("constant", 1e-3, 1e-5, 4, 5) == 1e-3


def test_cosine_restart_per_round_schedule() -> None:
    assert round_learning_rate("cosine_restart_per_round", 37, 1e-3, 1e-5, 60) == 1e-3
    lrs = [local_epoch_learning_rate("cosine_restart_per_round", 1e-3, 1e-5, e, 5) for e in range(5)]
    assert lrs[0] == 1e-3
    assert all(a > b for a, b in zip(lrs, lrs[1:]))
    assert lrs[-1] > 1e-5
    # One local epoch: the restart schedule is constant at the round LR.
    assert local_epoch_learning_rate("cosine_restart_per_round", 1e-3, 1e-5, 0, 1) == 1e-3
    with pytest.raises(ValueError):
        local_epoch_learning_rate("cosine_restart_per_round", 1e-3, 1e-5, 5, 5)


# --- S3: matched-budget step counts ----------------------------------------------------


def test_gradient_step_budget_iid_k3_and_natural_k2() -> None:
    # Regression, batch 4 runs. IID K=3 holds (37, 38, 37) runs; natural K=2 (52, 60).
    iid = gradient_step_budget({"client_0": 37, "client_1": 38, "client_2": 37}, 4, 1, 60)
    assert iid["steps_per_local_epoch_by_client"] == {"client_0": 10, "client_1": 10, "client_2": 10}
    assert iid["total_steps_per_round"] == 30
    assert iid["total_gradient_steps"] == 1800
    natural = gradient_step_budget({"client_0": 52, "client_1": 60}, 4, 5, 60)
    assert natural["consecutive_steps_per_round_by_client"] == {"client_0": 65, "client_1": 75}
    assert natural["total_gradient_steps"] == 140 * 60


# --- S3: client optimizer persistence --------------------------------------------------


def _adam_after_steps(steps: int) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad()
        model(torch.randn(4, 3)).pow(2).mean().backward()
        optimizer.step()
    return model, optimizer


def _experiment(state: str, state_dir: Path) -> dict:
    return {"client_optimizer_state": state, "optimizer_state_dir": str(state_dir)}


def test_persistent_optimizer_state_round_trips_exactly(tmp_path: Path) -> None:
    experiment = _experiment("persistent", tmp_path)
    model, trained = _adam_after_steps(7)
    assert _prepare_client_optimizer(trained, experiment, "client_0", 1) == 0.0
    _save_client_optimizer(trained, experiment, "client_0")

    fresh = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    assert _prepare_client_optimizer(fresh, experiment, "client_0", 2) == 7.0
    assert _adam_step_count(fresh) == 7.0
    for saved, loaded in zip(trained.state.values(), fresh.state.values()):
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(saved[key], loaded[key])


def test_persistent_mode_fails_loudly_on_missing_or_stale_state(tmp_path: Path) -> None:
    experiment = _experiment("persistent", tmp_path)
    _, optimizer = _adam_after_steps(1)
    # Missing at round 2 would silently turn persistent into fresh: the R1 regime.
    with pytest.raises(AssertionError, match="missing"):
        _prepare_client_optimizer(optimizer, experiment, "client_1", 2)
    _save_client_optimizer(optimizer, experiment, "client_1")
    # Present before round 1 means state leaked in from an earlier run.
    with pytest.raises(AssertionError, match="before round 1"):
        _prepare_client_optimizer(optimizer, experiment, "client_1", 1)


def test_fresh_mode_never_touches_disk(tmp_path: Path) -> None:
    experiment = _experiment("fresh", tmp_path / "state")
    _, optimizer = _adam_after_steps(3)
    assert _prepare_client_optimizer(optimizer, experiment, "client_0", 5) == 0.0
    _save_client_optimizer(optimizer, experiment, "client_0")
    assert not (tmp_path / "state").exists()


# --- S3: real-data client and determinism tests ----------------------------------------


def _real_client_setup(tmp_path: Path, state: str, local_epochs: int):
    from redo_by_sara.combined_iid_fl_partitioning import build_combined_partitions
    from redo_by_sara.combined_iid_flower import _federated_residual_baseline

    config = load_combined_iid_flower_config(IID_K3_CONFIG)
    classification = torch.load(config["classification_artifact"], map_location="cpu", weights_only=False)
    regression = torch.load(config["regression_artifact"], map_location="cpu", weights_only=False)
    partitions, _ = build_combined_partitions(classification, regression, scheme="iid", num_clients=3, seed=4601)
    means, scale, _ = _federated_residual_baseline(regression, partitions)
    experiment = {
        "seed": 4601,
        "torch_threads": 3,
        "classification_batch_size": 32,
        "regression_run_batch_size": 4,
        "local_epochs": local_epochs,
        "weight_decay": 1e-4,
        "minimum_learning_rate": 1e-5,
        "lr_schedule": "cosine_over_rounds",
        "client_optimizer_state": state,
        "optimizer_state_dir": str(tmp_path),
    }
    return config, partitions, means, scale, experiment


@pytest.mark.requires_data
def test_regression_client_trains_more_than_one_local_epoch(tmp_path: Path) -> None:
    """Pre-S3 the run-ownership check accumulated across epochs and raised at E > 1."""
    from redo_by_sara.combined_iid_flower import FlowerResidualClient
    from redo_by_sara.combined_residual_run_regression import build_zero_initialized_residual_model
    from redo_by_sara.parameters import get_parameters

    config, partitions, means, scale, experiment = _real_client_setup(tmp_path, "persistent", 2)
    partition = partitions[0]
    client = FlowerResidualClient(
        config["regression_artifact"], partition.client_id, partition.regression_indices,
        partition.run_uids, means, scale, experiment,
    )
    initial = get_parameters(build_zero_initialized_residual_model())
    arrays, num_examples, metrics = client.fit(initial, {"server_round": 1, "learning_rate": 1e-3})
    assert num_examples == len(partition.run_uids) == 37
    assert metrics["local_steps"] == 2 * math.ceil(37 / 4) == 20
    assert metrics["adam_steps_before_round"] == 0.0
    assert metrics["adam_steps_after_round"] == 20.0
    assert metrics["local_end_standardized_mse"] < metrics["local_start_standardized_mse"]

    # Round 2 picks up the Adam state where round 1 left it.
    _, _, round_two = client.fit(arrays, {"server_round": 2, "learning_rate": 1e-3})
    assert round_two["adam_steps_before_round"] == 20.0
    assert round_two["adam_steps_after_round"] == 40.0


@pytest.mark.requires_data
def test_persistent_optimizer_runs_are_bitwise_reproducible(tmp_path: Path) -> None:
    """docs/s3_winner_rule.md §6: the sweep's precondition.

    Optimizer-state files written and read between rounds are a new way for
    non-reproducibility to come in, so the same persistent config run twice must give
    bitwise-identical final parameters and optimizer state.
    """
    from redo_by_sara.combined_iid_flower import run_combined_iid_flower_experiment

    config = load_combined_iid_flower_config(
        NATURAL_K2_CONFIG,
        {"federated.local_epochs": 2, "federated.client_optimizer_state": "persistent"},
    )
    outputs = []
    for name in ("first", "second"):
        run_combined_iid_flower_experiment(
            config, "regression", num_rounds_override=2,
            output_root=tmp_path / name, allow_degenerate=True,
        )
        outputs.append(tmp_path / name / "regression")

    first, second = outputs
    final_first = torch.load(first / "round_checkpoints/round_002.pt", weights_only=True)
    final_second = torch.load(second / "round_checkpoints/round_002.pt", weights_only=True)
    assert final_first.keys() == final_second.keys()
    for key in final_first:
        assert torch.equal(final_first[key], final_second[key]), key

    state_files = sorted(p.name for p in (first / "client_optimizer_state").glob("*.pt"))
    assert state_files == ["client_0.pt", "client_1.pt"]
    for name in state_files:
        a = torch.load(first / "client_optimizer_state" / name, weights_only=True)
        b = torch.load(second / "client_optimizer_state" / name, weights_only=True)
        for index in a["state"]:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.equal(a["state"][index][key], b["state"][index][key]), (name, index, key)
    assert (first / "local_client_fit_history.csv").read_bytes() == (
        second / "local_client_fit_history.csv"
    ).read_bytes()

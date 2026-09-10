from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .combined_centralized_classification import (
    CombinedArtifactDataset,
    _classification_metrics,
    _confusion_matrix,
    _run_epoch as _run_classification_epoch,
)
from .combined_centralized_regression import regression_metrics
from .combined_residual_run_regression import (
    RunGroupedResidualDataset,
    _group_run_records,
    _run_group_epoch,
    build_zero_initialized_residual_model,
    collate_whole_runs,
    compute_train_source_subject_means,
)
from .guardrails import (
    GuardedWriter,
    TestSetAccessGuard,
    assert_determinism_flags,
    check_regression_health,
    seeding_record,
    set_random_seeds,
)
from .models import SimpleCNN1D
from .parameters import get_parameters, set_parameters


_CLIENT_ARTIFACT_CACHE: dict[str, dict[str, object]] = {}


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_combined_iid_flower_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text())
    federated = raw["federated"]
    training = raw["training"]
    experiment = raw["experiment"]

    if str(federated["framework"]).lower() != "flower":
        raise ValueError("The combined IID experiment must use the Flower framework.")
    if str(federated["aggregation"]) != "FedAvg":
        raise ValueError("The combined IID experiment is fixed to FedAvg.")
    if int(federated["num_clients"]) != 3:
        raise ValueError("The combined IID experiment is fixed to exactly three clients.")
    if int(federated["num_rounds"]) != 60 or int(federated["local_epochs"]) != 1:
        raise ValueError("The production experiment is fixed to 60 rounds and one local epoch.")
    if float(federated["fraction_fit"]) != 1.0:
        raise ValueError("All three clients must participate in every fit round.")
    if float(federated["fraction_evaluate"]) != 0.0:
        raise ValueError("Client evaluation is disabled; the server monitors the train union only.")
    if bool(federated["accept_failures"]):
        raise ValueError("Client failures must not be accepted in this experiment.")
    if [int(v) for v in experiment["selected_channels_one_based"]] != [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
    ]:
        raise ValueError("The experiment must use channels 1-8 and 10.")
    if [int(v) for v in experiment["included_subjects"]] != [1, 2, 3, 4, 5, 7, 8]:
        raise ValueError("The experiment must include subjects 1,2,3,4,5,7,8 only.")
    if not math.isclose(float(experiment["test2_sample_rate_override_hz"]), 1706.667):
        raise ValueError("Test_2 must use the corrected 1706.667 Hz sample rate.")

    return {
        "config_path": config_path,
        "project_root": project_root,
        "seed": int(raw["seed"]),
        "classification_artifact": _resolve(project_root, raw["classification_artifact"]),
        "regression_artifact": _resolve(project_root, raw["regression_artifact"]),
        "centralized_classification_dir": _resolve(
            project_root, raw["centralized_classification_dir"]
        ),
        "centralized_regression_dir": _resolve(
            project_root, raw["centralized_regression_dir"]
        ),
        "centralized_absolute_regression_summary": _resolve(
            project_root, raw["centralized_absolute_regression_summary"]
        ),
        "output_dir": _resolve(project_root, raw["output_dir"]),
        "num_clients": int(federated["num_clients"]),
        "num_rounds": int(federated["num_rounds"]),
        "local_epochs": int(federated["local_epochs"]),
        "client_num_cpus": float(federated["client_num_cpus"]),
        "client_num_gpus": float(federated["client_num_gpus"]),
        "classification_batch_size": int(training["classification_batch_size"]),
        "regression_run_batch_size": int(training["regression_run_batch_size"]),
        "initial_learning_rate": float(training["initial_learning_rate"]),
        "minimum_learning_rate": float(training["minimum_learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "num_workers": int(training["num_workers"]),
        "experiment": experiment,
    }


def _load_artifact_cached(path: str | Path) -> dict[str, object]:
    key = str(Path(path).resolve())
    artifact = _CLIENT_ARTIFACT_CACHE.get(key)
    if artifact is None:
        artifact = torch.load(key, map_location="cpu", weights_only=False)
        _CLIENT_ARTIFACT_CACHE[key] = artifact
    return artifact


def cosine_round_learning_rate(
    server_round: int, maximum: float, minimum: float, total_rounds: int
) -> float:
    if not 1 <= server_round <= total_rounds:
        raise ValueError(f"Round {server_round} is outside 1..{total_rounds}.")
    offset = server_round - 1
    return float(
        minimum
        + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * offset / total_rounds))
    )


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    preferred = ["round", "client_id", "num_examples"]
    keys = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _extended_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    metrics = regression_metrics(actual, predicted)
    metrics["median_absolute_error_mps"] = float(
        np.median(np.abs(np.asarray(predicted) - np.asarray(actual)))
    )
    return metrics


def _sample_rate_audit(
    artifact: dict[str, object], config: dict[str, Any]
) -> dict[str, Any]:
    """The R8 sampling-rate block for a federated per-task ``training_summary.json``.

    Already surfaced in the centralized ``artifact_summary.json`` (``timing_assumption``),
    in ``setup_audit["conditions"]``, and per run in ``run_manifest.csv``. The federated
    per-task summary was the one place it was missing.

    It matters because the Test_2 override is a **+3.3% time warp** on every window from
    that source, and walking speed is a time-derived quantity: get it wrong and every
    label-signal relation is wrong while nothing looks broken. Values are read from the
    already-validated config and the artifact summary rather than restated here, so this
    does not become another copy of the rate.
    """

    experiment = config["experiment"]
    return {
        "test_2_metadata_sample_rate_hz": 1652.0,
        "test_2_effective_sample_rate_hz": float(
            experiment["test2_sample_rate_override_hz"]
        ),
        "testing_20251124_effective_sample_rate_hz": 1651.6129032258063,
        "target_sample_rate_hz": float(experiment["target_sample_rate_hz"]),
        "resample_ratios_up_down": {
            "test_2": [15, 64],
            "testing_20251124": [31, 128],
        },
        "override_justification": (
            "The 1652 Hz in the Test_2 HDF5 metadata was recorded incorrectly; the true "
            "rate is 1706.667 Hz. The override is deliberate. It time-warps every Test_2 "
            "window by +3.3% relative to the metadata rate, and walking speed is a "
            "time-derived quantity, so this affects every speed label's relation to its "
            "signal. Applied before resampling to 400 Hz."
        ),
        "artifact_timing_assumption": artifact["summary"].get("timing_assumption"),
    }


def _build_classification_model(artifact: dict[str, object]) -> SimpleCNN1D:
    return SimpleCNN1D(
        in_channels=int(artifact["samples"].shape[1]),
        output_dim=len(artifact["index_to_class"]),
    )


def _coerce_ndarrays(parameters: Any, parameters_to_ndarrays: Any) -> list[np.ndarray]:
    return parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)


def _client_id_from_context(context: Any) -> str:
    node_config = getattr(context, "node_config", {})
    return str(node_config.get("partition-id", node_config.get("partition_id", "")))


def _client_seed(base_seed: int, client_id: str, server_round: int) -> int:
    numeric_client_id = int(str(client_id).removeprefix("client_"))
    return int(base_seed + 1000 * server_round + numeric_client_id)


def _federated_residual_baseline(
    artifact: dict[str, object], partitions: Sequence[Any]
) -> tuple[dict[tuple[str, str], float], float, dict[str, Any]]:
    metadata: list[dict[str, Any]] = artifact["metadata"]
    client_stats: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for partition in partitions:
        run_records = _group_run_records(artifact, partition.regression_indices)
        grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"n": 0.0, "sum": 0.0, "sum_sq": 0.0}
        )
        for run in run_records:
            key = (str(run["source_id"]), str(run["subject_id"]))
            speed = float(run["actual_speed_mps"])
            grouped[key]["n"] += 1.0
            grouped[key]["sum"] += speed
            grouped[key]["sum_sq"] += speed * speed
        client_stats[str(partition.client_id)] = dict(grouped)

    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "sum": 0.0, "sum_sq": 0.0}
    )
    for grouped in client_stats.values():
        for key, stats in grouped.items():
            for field in ("n", "sum", "sum_sq"):
                totals[key][field] += stats[field]
    means = {key: values["sum"] / values["n"] for key, values in totals.items()}
    total_runs = int(sum(values["n"] for values in totals.values()))
    centered_sum_sq = sum(
        values["sum_sq"] - values["sum"] ** 2 / values["n"]
        for values in totals.values()
    )
    scale = float(math.sqrt(max(centered_sum_sq, 0.0) / total_runs))

    central_means, central_scale, _ = compute_train_source_subject_means(artifact)
    mean_error = max(abs(means[key] - central_means[key]) for key in means)
    if set(means) != set(central_means) or mean_error > 1e-12:
        raise AssertionError("Federated source/subject mean aggregation does not match central train-only means.")
    if not math.isclose(scale, central_scale, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(
            f"Federated residual scale {scale} does not match centralized {central_scale}."
        )
    if total_runs != 112:
        raise AssertionError(f"Expected 112 unique train runs, found {total_runs}.")

    audit = {
        "method": "Federated sufficient statistics (n, sum, sum_sq) over unique local training runs",
        "num_train_runs": total_runs,
        "num_source_subject_strata": len(means),
        "residual_scale_mps": scale,
        "centralized_residual_scale_mps": central_scale,
        "maximum_absolute_mean_difference_mps": mean_error,
        "client_source_subject_statistics": {
            client_id: {
                f"{source_id}:{subject_id}": values
                for (source_id, subject_id), values in sorted(grouped.items())
            }
            for client_id, grouped in sorted(client_stats.items())
        },
        "global_source_subject_means_mps": {
            f"{source_id}:{subject_id}": value
            for (source_id, subject_id), value in sorted(means.items())
        },
    }
    return means, scale, audit


def _normalization_sufficient_statistics_audit(
    artifact: dict[str, object], index_groups: Sequence[Sequence[int]]
) -> dict[str, Any]:
    samples: torch.Tensor = artifact["samples"]
    count = 0
    channel_sum = torch.zeros(samples.shape[1], dtype=torch.float64)
    channel_sum_sq = torch.zeros(samples.shape[1], dtype=torch.float64)
    for indices in index_groups:
        values = samples[torch.as_tensor(indices, dtype=torch.long)].to(torch.float64)
        count += int(values.shape[0] * values.shape[2])
        channel_sum += values.sum(dim=(0, 2))
        channel_sum_sq += (values * values).sum(dim=(0, 2))
    mean = channel_sum / count
    variance = torch.clamp(channel_sum_sq / count - mean * mean, min=0.0)
    std = torch.sqrt(variance)
    saved_mean = artifact["channel_mean"].reshape(-1).to(torch.float64)
    saved_std = artifact["channel_std"].reshape(-1).to(torch.float64)
    mean_delta = float(torch.max(torch.abs(mean - saved_mean)))
    std_delta = float(torch.max(torch.abs(std - saved_std)))
    if mean_delta > 2e-6 or std_delta > 2e-6:
        raise AssertionError(
            "Client-aggregated training normalization does not reproduce saved train-only statistics: "
            f"mean delta={mean_delta}, std delta={std_delta}."
        )
    return {
        "method": "Sum, squared sum, and sample count aggregated across disjoint clients",
        "scalar_observations_per_channel": count,
        "maximum_absolute_mean_difference": mean_delta,
        "maximum_absolute_std_difference": std_delta,
        "uses_test_data": False,
    }


def aggregate_regression_sufficient_metrics(
    statistics: Iterable[dict[str, float]], tolerance_mps: float = 0.10
) -> dict[str, float]:
    rows = list(statistics)
    n = sum(int(row["n"]) for row in rows)
    if n <= 0:
        raise ValueError("Cannot aggregate zero regression runs.")
    sse = sum(float(row["sse"]) for row in rows)
    sae = sum(float(row["sae"]) for row in rows)
    error_sum = sum(float(row["error_sum"]) for row in rows)
    within = sum(float(row["within_count"]) for row in rows)
    sum_y = sum(float(row["sum_y"]) for row in rows)
    sum_y2 = sum(float(row["sum_y2"]) for row in rows)
    denominator = sum_y2 - sum_y * sum_y / n
    mse = sse / n
    return {
        "mse": mse,
        "rmse_mps": math.sqrt(mse),
        "mae_mps": sae / n,
        "bias_mps": error_sum / n,
        "r2": 1.0 - sse / denominator if denominator > 1e-12 else 0.0,
        "within_0_10_mps_pct": 100.0 * within / n,
        "num_runs": float(n),
        "tolerance_mps": tolerance_mps,
    }


def aggregate_confusion_matrices(
    matrices: Sequence[np.ndarray], class_names: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    if not matrices:
        raise ValueError("At least one confusion matrix is required.")
    total = np.sum(np.stack(matrices), axis=0)
    return total, _classification_metrics(total, class_names)


class FlowerClassificationClient:
    """Factory-compatible implementation mixed with Flower NumPyClient at runtime."""

    def __init__(self, artifact_path: str | Path, client_id: str, indices: Sequence[int], run_uids: Sequence[str], experiment: dict[str, Any]) -> None:
        self.artifact_path = str(artifact_path)
        self.client_id = str(client_id)
        self.indices = tuple(int(index) for index in indices)
        self.run_uids = tuple(str(uid) for uid in run_uids)
        self.experiment = experiment

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        artifact = _load_artifact_cached(self.artifact_path)
        return get_parameters(_build_classification_model(artifact))

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, float | str]]:
        artifact = _load_artifact_cached(self.artifact_path)
        server_round = int(config["server_round"])
        seed = _client_seed(int(self.experiment["seed"]), self.client_id, server_round)
        set_random_seeds(seed)
        torch.set_num_threads(max(1, int(self.experiment["torch_threads"])))
        model = _build_classification_model(artifact)
        set_parameters(model, parameters)
        device = torch.device("cpu")
        model.to(device)
        generator = torch.Generator().manual_seed(seed)
        dataset = CombinedArtifactDataset(artifact, self.indices)
        train_loader = DataLoader(dataset, batch_size=int(self.experiment["classification_batch_size"]), shuffle=True, num_workers=0, generator=generator)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(self.experiment["weight_decay"]))
        criterion = nn.CrossEntropyLoss()
        online_loss = 0.0
        online_accuracy = 0.0
        for _ in range(int(self.experiment["local_epochs"])):
            online_loss, online_accuracy, _, _ = _run_classification_epoch(
                model, train_loader, criterion, device, optimizer, collect_outputs=False
            )
        arrays = get_parameters(model)
        if not all(np.isfinite(value).all() for value in arrays):
            raise FloatingPointError(f"{self.client_id} returned non-finite parameters.")
        return arrays, len(dataset), {
            "client_id": self.client_id,
            "local_online_cross_entropy": float(online_loss),
            "local_online_accuracy": float(online_accuracy),
            "learning_rate": float(config["learning_rate"]),
            "server_round": float(server_round),
        }

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[float, int, dict[str, float]]:
        return 0.0, len(self.indices), {}


class FlowerResidualClient:
    """Run-balanced residual-regression client; Flower weights it by unique runs."""

    def __init__(self, artifact_path: str | Path, client_id: str, indices: Sequence[int], run_uids: Sequence[str], source_subject_means: dict[tuple[str, str], float], residual_scale_mps: float, experiment: dict[str, Any]) -> None:
        self.artifact_path = str(artifact_path)
        self.client_id = str(client_id)
        self.indices = tuple(int(index) for index in indices)
        self.run_uids = tuple(str(uid) for uid in run_uids)
        self.source_subject_means = source_subject_means
        self.residual_scale_mps = float(residual_scale_mps)
        self.experiment = experiment

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_parameters(build_zero_initialized_residual_model())

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, float | str]]:
        artifact = _load_artifact_cached(self.artifact_path)
        server_round = int(config["server_round"])
        seed = _client_seed(int(self.experiment["seed"]), self.client_id, server_round)
        set_random_seeds(seed)
        torch.set_num_threads(max(1, int(self.experiment["torch_threads"])))
        dataset = RunGroupedResidualDataset(artifact, self.indices, self.source_subject_means, self.residual_scale_mps)
        dataset_uids = [str(run["run_uid"]) for run in dataset.runs]
        if set(dataset_uids) != set(self.run_uids) or len(dataset_uids) != len(self.run_uids):
            raise AssertionError(f"{self.client_id} regression dataset does not match run ownership.")
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(dataset, batch_size=int(self.experiment["regression_run_batch_size"]), shuffle=True, num_workers=0, generator=generator, collate_fn=collate_whole_runs)
        model = build_zero_initialized_residual_model()
        set_parameters(model, parameters)
        device = torch.device("cpu")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(self.experiment["weight_decay"]))
        criterion = nn.MSELoss()
        observed_uids: list[str] = []
        for _ in range(int(self.experiment["local_epochs"])):
            result = _run_group_epoch(model, train_loader, criterion, device, optimizer)
            observed_uids.extend(str(uid) for uid in result["run_uids"])
        if observed_uids != list(dict.fromkeys(observed_uids)):
            raise AssertionError(f"{self.client_id} saw a run more than once in a local epoch.")
        if set(observed_uids) != set(self.run_uids):
            raise AssertionError(f"{self.client_id} did not see every owned run exactly once.")
        metrics = _extended_regression_metrics(result["run_actual"], result["run_predicted"])
        arrays = get_parameters(model)
        if not all(np.isfinite(value).all() for value in arrays):
            raise FloatingPointError(f"{self.client_id} returned non-finite parameters.")
        return arrays, len(dataset), {
            "client_id": self.client_id,
            "local_online_standardized_mse": float(result["loss"]),
            "local_online_rmse_mps": metrics["rmse_mps"],
            "local_online_r2": metrics["r2"],
            "local_online_within_0_10_mps_pct": metrics["within_0_10_mps_pct"],
            "learning_rate": float(config["learning_rate"]),
            "server_round": float(server_round),
        }

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[float, int, dict[str, float]]:
        return 0.0, len(self.run_uids), {}


def _as_flower_numpy_client(implementation: Any) -> Any:
    from flwr.client import NumPyClient

    class Adapter(NumPyClient):
        def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
            return implementation.get_parameters(config)

        def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, Any]]:
            return implementation.fit(parameters, config)

        def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
            return implementation.evaluate(parameters, config)

    return Adapter().to_client()


def _evaluate_classification(model: nn.Module, loader: DataLoader, class_names: Sequence[str]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    loss, accuracy, outputs, targets = _run_classification_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer=None, collect_outputs=True)
    if outputs is None or targets is None:
        raise RuntimeError("Classification evaluation produced no outputs.")
    predictions = torch.argmax(outputs, dim=1).numpy()
    target_values = targets.numpy()
    matrix = _confusion_matrix(target_values, predictions, len(class_names))
    metrics = _classification_metrics(matrix, class_names)
    metrics["cross_entropy"] = float(loss)
    metrics["accuracy"] = float(accuracy)
    return metrics, matrix, predictions


def _classification_history_row(round_number: int, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": int(round_number),
        "global_train_cross_entropy": float(result["cross_entropy"]),
        "global_train_accuracy": float(result["accuracy"]),
        "global_train_balanced_accuracy": float(result["balanced_accuracy"]),
        "global_train_macro_f1": float(result["macro_f1"]),
    }


def _regression_history_row(round_number: int, result: dict[str, Any], residual_scale_mps: float) -> dict[str, Any]:
    metrics = _extended_regression_metrics(result["run_actual"], result["run_predicted"])
    physical_from_standardized = float(result["loss"]) * residual_scale_mps**2
    if not math.isclose(physical_from_standardized, metrics["mse"], rel_tol=3e-5, abs_tol=1e-9):
        raise AssertionError(f"Standardized and physical run MSE disagree: {physical_from_standardized} != {metrics['mse']}.")
    return {
        "round": int(round_number),
        "global_train_standardized_mse": float(result["loss"]),
        "global_train_mse_mps2": metrics["mse"],
        "global_train_rmse_mps": metrics["rmse_mps"],
        "global_train_mae_mps": metrics["mae_mps"],
        "global_train_bias_mps": metrics["bias_mps"],
        "global_train_r2": metrics["r2"],
        "global_train_within_0_10_mps_pct": metrics["within_0_10_mps_pct"],
    }


def _metrics_by_field(rows: Sequence[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        key: _extended_regression_metrics(
            np.asarray([float(row["actual_speed_mps"]) for row in values]),
            np.asarray([float(row["predicted_speed_mps"]) for row in values]),
        )
        for key, values in sorted(grouped.items())
    }


def _write_confusion(path: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *[int(value) for value in row]])


def _write_classification_predictions(
    path: Path,
    artifact: dict[str, object],
    indices: Sequence[int],
    predictions: np.ndarray,
) -> None:
    metadata: list[dict[str, Any]] = artifact["metadata"]
    targets: torch.Tensor = artifact["classification_targets"]
    class_names: list[str] = artifact["index_to_class"]
    rows: list[dict[str, Any]] = []
    for index, prediction in zip(indices, predictions):
        item = metadata[int(index)]
        target = int(targets[int(index)])
        rows.append(
            {
                "artifact_index": int(index),
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
    _write_rows(path, rows)


def _build_regression_prediction_rows(
    artifact: dict[str, object],
    dataset: RunGroupedResidualDataset,
    result: dict[str, Any],
    source_subject_means: dict[tuple[str, str], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_by_uid = {str(run["run_uid"]): run for run in dataset.runs}
    run_rows: list[dict[str, Any]] = []
    for run_uid, actual, predicted, actual_residual, predicted_residual in zip(
        result["run_uids"],
        result["run_actual"],
        result["run_predicted"],
        result["run_actual_residual"],
        result["run_predicted_residual"],
    ):
        run = run_by_uid[str(run_uid)]
        baseline = source_subject_means[(str(run["source_id"]), str(run["subject_id"]))]
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
    run_rows.sort(key=lambda row: str(row["run_uid"]))

    metadata: list[dict[str, Any]] = artifact["metadata"]
    targets: torch.Tensor = artifact["regression_targets"]
    window_rows: list[dict[str, Any]] = []
    for artifact_index, predicted, predicted_residual in zip(
        result["window_indices"],
        result["window_predicted"],
        result["window_predicted_residual"],
    ):
        index = int(artifact_index)
        item = metadata[index]
        baseline = source_subject_means[(str(item["source_id"]), str(item["subject_id"]))]
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
    return run_rows, window_rows


def audit_combined_iid_flower_setup(
    config: dict[str, Any],
) -> tuple[dict[str, object], dict[str, object], list[Any], dict[str, Any]]:
    from .combined_iid_fl_partitioning import build_combined_iid_partitions

    classification = torch.load(
        config["classification_artifact"], map_location="cpu", weights_only=False
    )
    regression = torch.load(
        config["regression_artifact"], map_location="cpu", weights_only=False
    )
    for task, artifact, expected in (
        ("classification", classification, (1242, 307)),
        ("regression", regression, (966, 238)),
    ):
        summary = artifact["summary"]
        if summary["sample_shape"] != [9, 2000]:
            raise AssertionError(f"{task} artifact is not nine channels by 2000 samples.")
        if (int(summary["num_train"]), int(summary["num_test"])) != expected:
            raise AssertionError(f"Unexpected {task} 80/20 artifact counts.")
        if summary["train_test_run_overlap"] != 0:
            raise AssertionError(f"{task} artifact has train/test run overlap.")
        subject_ids = {str(item["subject_id"]) for item in artifact["metadata"]}
        if subject_ids != {"001", "002", "003", "004", "005", "007", "008"}:
            raise AssertionError(f"Unexpected {task} subject set: {sorted(subject_ids)}")

    partitions, partition_summary = build_combined_iid_partitions(
        classification,
        regression,
        num_clients=config["num_clients"],
        seed=config["seed"],
    )
    classification_normalization = _normalization_sufficient_statistics_audit(
        classification, [partition.classification_indices for partition in partitions]
    )
    regression_normalization = _normalization_sufficient_statistics_audit(
        regression, [partition.regression_indices for partition in partitions]
    )
    means, scale, residual_audit = _federated_residual_baseline(regression, partitions)
    setup = {
        "framework": "Flower 1.29 legacy simulation API",
        "aggregation": "FedAvg",
        "num_clients": config["num_clients"],
        "production_rounds": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "full_client_participation": True,
        "classification_client_weight_unit": "training windows",
        "classification_client_weights": [
            len(partition.classification_indices) for partition in partitions
        ],
        "regression_client_weight_unit": "unique complete training runs",
        "regression_client_weights": [len(partition.run_uids) for partition in partitions],
        "normalization": {
            "classification": classification_normalization,
            "regression": regression_normalization,
        },
        "residual_baseline": residual_audit,
        "test_policy": "No validation. Train-only global monitoring at rounds 0..60; held-out test evaluated once after the final round.",
        "partition": partition_summary,
    }
    return classification, regression, partitions, setup


def _fit_metric_aggregation(metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, float]:
    total = sum(int(num_examples) for num_examples, _ in metrics)
    if total <= 0:
        return {}
    keys = sorted(
        {
            key
            for _, values in metrics
            for key, value in values.items()
            if key.startswith("local_online_") and isinstance(value, (int, float))
        }
    )
    return {
        f"weighted_{key}": sum(
            int(num_examples) * float(values[key])
            for num_examples, values in metrics
            if key in values
        )
        / total
        for key in keys
    }


def _run_flower_core(
    task: str,
    artifact: dict[str, object],
    artifact_path: Path,
    partitions: Sequence[Any],
    setup: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    num_rounds: int,
    source_subject_means: dict[tuple[str, str], float] | None = None,
    residual_scale_mps: float | None = None,
    test_guard: TestSetAccessGuard | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    try:
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
        from flwr.server import ServerConfig
        from flwr.server.strategy import FedAvg
        from flwr.simulation import start_simulation
    except ImportError as exc:
        raise RuntimeError("Flower with simulation dependencies is required.") from exc

    if task not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task: {task}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "round_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "global_train_history.csv"
    client_history_path = output_dir / "local_client_fit_history.csv"
    seeding = set_random_seeds(config["seed"])
    assert_determinism_flags()
    torch.set_num_threads(max(1, int(config["client_num_cpus"])))

    if task == "classification":
        initial_model = _build_classification_model(artifact)
        train_dataset = CombinedArtifactDataset(artifact, artifact["train_indices"].tolist())
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["classification_batch_size"],
            shuffle=False,
            num_workers=0,
        )
        class_names: list[str] = artifact["index_to_class"]
        expected_examples = {
            partition.client_id: len(partition.classification_indices)
            for partition in partitions
        }
    else:
        if source_subject_means is None or residual_scale_mps is None:
            raise ValueError("Residual regression requires global train-only means and scale.")
        initial_model = build_zero_initialized_residual_model()
        train_dataset = RunGroupedResidualDataset(
            artifact,
            artifact["train_indices"].tolist(),
            source_subject_means,
            residual_scale_mps,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["regression_run_batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_whole_runs,
        )
        expected_examples = {
            partition.client_id: len(partition.run_uids) for partition in partitions
        }

    initial_arrays = [array.copy() for array in get_parameters(initial_model)]
    roundtrip_model = (
        _build_classification_model(artifact)
        if task == "classification"
        else build_zero_initialized_residual_model()
    )
    set_parameters(roundtrip_model, initial_arrays)
    if not all(
        np.array_equal(left, right)
        for left, right in zip(initial_arrays, get_parameters(roundtrip_model))
    ):
        raise AssertionError("State-dict ndarray round trip changed initial parameters.")
    if task == "regression":
        final_layer = initial_model.head[-1]
        if not isinstance(final_layer, nn.Linear):
            raise AssertionError("Residual model head is not linear.")
        if torch.count_nonzero(final_layer.weight) or torch.count_nonzero(final_layer.bias):
            raise AssertionError("Residual final layer is not exactly zero at round 0.")

    global_rows: list[dict[str, Any]] = []
    if test_guard is None:
        test_guard = TestSetAccessGuard(artifact, label=task)
    # Sealed for the whole of training: reaching for held-out data here both records
    # itself and raises, so the guard prevents as well as measures.
    test_guard.seal()

    def evaluate_global_train(
        server_round: int, parameters: Any, _: dict[str, Any]
    ) -> tuple[float, dict[str, float]]:
        arrays = _coerce_ndarrays(parameters, parameters_to_ndarrays)
        if task == "classification":
            model = _build_classification_model(artifact)
            set_parameters(model, arrays)
            metrics, _, _ = _evaluate_classification(model, train_loader, class_names)
            row = _classification_history_row(server_round, metrics)
            loss = float(metrics["cross_entropy"])
            returned_metrics = {
                "global_train_accuracy": float(metrics["accuracy"]),
                "global_train_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "global_train_macro_f1": float(metrics["macro_f1"]),
            }
            status = f"loss={loss:.6f}, accuracy={metrics['accuracy']:.4f}"
        else:
            model = build_zero_initialized_residual_model()
            set_parameters(model, arrays)
            result = _run_group_epoch(
                model, train_loader, nn.MSELoss(), torch.device("cpu"), optimizer=None
            )
            row = _regression_history_row(server_round, result, residual_scale_mps)
            loss = float(result["loss"])
            returned_metrics = {
                "global_train_rmse_mps": float(row["global_train_rmse_mps"]),
                "global_train_r2": float(row["global_train_r2"]),
                "global_train_within_0_10_mps_pct": float(
                    row["global_train_within_0_10_mps_pct"]
                ),
            }
            status = (
                f"rmse={row['global_train_rmse_mps']:.6f} m/s, "
                f"r2={row['global_train_r2']:.4f}"
            )
            if server_round == 0:
                max_residual = float(np.max(np.abs(result["run_predicted_residual"])))
                if max_residual > 1e-8:
                    raise AssertionError("Round-0 residual predictions are not exactly zero.")
                if not math.isclose(float(result["loss"]), 1.0, abs_tol=3e-6):
                    raise AssertionError(
                        f"Round-0 standardized train MSE should be one, got {result['loss']}."
                    )
        row["learning_rate"] = (
            config["initial_learning_rate"]
            if server_round == 0
            else cosine_round_learning_rate(
                server_round,
                config["initial_learning_rate"],
                config["minimum_learning_rate"],
                num_rounds,
            )
        )
        global_rows.append(row)
        _write_rows(history_path, global_rows)
        if server_round in {0, 1, num_rounds} or server_round % 10 == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"round_{server_round:03d}.pt")
        print(
            f"{task.capitalize()} global train round {server_round:03d}/{num_rounds}: {status}",
            flush=True,
        )
        return loss, returned_metrics

    runtime = {
        "seed": config["seed"],
        "torch_threads": max(1, int(config["client_num_cpus"])),
        "classification_batch_size": config["classification_batch_size"],
        "regression_run_batch_size": config["regression_run_batch_size"],
        "local_epochs": config["local_epochs"],
        "weight_decay": config["weight_decay"],
    }
    partitions_by_id = {partition.client_id: partition for partition in partitions}

    def client_fn(context: Any) -> Any:
        raw_id = _client_id_from_context(context)
        client_id = raw_id if raw_id.startswith("client_") else f"client_{int(raw_id)}"
        partition = partitions_by_id[client_id]
        if task == "classification":
            implementation = FlowerClassificationClient(
                artifact_path,
                client_id,
                partition.classification_indices,
                partition.run_uids,
                runtime,
            )
        else:
            implementation = FlowerResidualClient(
                artifact_path,
                client_id,
                partition.regression_indices,
                partition.run_uids,
                source_subject_means,
                residual_scale_mps,
                runtime,
            )
        return _as_flower_numpy_client(implementation)

    def fit_config(server_round: int) -> dict[str, float | int]:
        return {
            "server_round": int(server_round),
            "learning_rate": cosine_round_learning_rate(
                server_round,
                config["initial_learning_rate"],
                config["minimum_learning_rate"],
                num_rounds,
            ),
        }

    class TrackingFedAvg(FedAvg):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.latest_parameters = kwargs["initial_parameters"]
            self.client_rows: list[dict[str, Any]] = []
            self.round_one_parameter_changed: bool | None = None

        def aggregate_fit(
            self, server_round: int, results: Any, failures: Any
        ) -> tuple[Any, dict[str, Any]]:
            # Order the client results before aggregating. Ray returns the three
            # ClientAppActors in whatever order they finish, FedAvg sums their weighted
            # parameters in that order, and float addition is not associative — so an
            # identical run diverges from its twin in the last bits of round 1 and
            # compounds from there. Sorting by client id makes aggregation reproducible.
            results = sorted(
                results, key=lambda item: str(dict(item[1].metrics or {}).get("client_id", ""))
            )
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            if failures:
                raise RuntimeError(
                    f"Round {server_round} had {len(failures)} client failure(s): {failures}"
                )
            if len(results) != config["num_clients"]:
                raise RuntimeError(
                    f"Round {server_round} returned {len(results)} clients, expected {config['num_clients']}."
                )
            observed: dict[str, int] = {}
            for _, fit_result in results:
                metrics = dict(fit_result.metrics or {})
                client_id = str(metrics.get("client_id", ""))
                if client_id in observed or client_id not in expected_examples:
                    raise RuntimeError(
                        f"Round {server_round} returned invalid client id {client_id!r}."
                    )
                num_examples = int(fit_result.num_examples)
                observed[client_id] = num_examples
                row: dict[str, Any] = {
                    "round": int(server_round),
                    "client_id": client_id,
                    "num_examples": num_examples,
                }
                for key, value in metrics.items():
                    if key != "client_id":
                        row[key] = value
                self.client_rows.append(row)
            if observed != expected_examples:
                raise RuntimeError(
                    f"Round {server_round} FedAvg weights {observed} != expected {expected_examples}."
                )
            if sum(observed.values()) != sum(expected_examples.values()):
                raise RuntimeError("FedAvg total weight changed unexpectedly.")
            if aggregated_parameters is None:
                raise RuntimeError(f"Round {server_round} produced no global parameters.")
            arrays = parameters_to_ndarrays(aggregated_parameters)
            if not all(np.isfinite(value).all() for value in arrays):
                raise FloatingPointError(f"Round {server_round} global parameters are non-finite.")
            if server_round == 1:
                self.round_one_parameter_changed = any(
                    not np.array_equal(before, after)
                    for before, after in zip(initial_arrays, arrays)
                )
                if not self.round_one_parameter_changed:
                    raise AssertionError("Global parameters did not change after round 1.")
            self.latest_parameters = aggregated_parameters
            _write_rows(client_history_path, self.client_rows)
            return aggregated_parameters, aggregated_metrics

    initial_parameters = ndarrays_to_parameters(initial_arrays)
    strategy = TrackingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=config["num_clients"],
        min_evaluate_clients=0,
        min_available_clients=config["num_clients"],
        accept_failures=False,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=_fit_metric_aggregation,
        evaluate_fn=evaluate_global_train,
    )
    source_root = config["project_root"] / "src"
    start_simulation(
        client_fn=client_fn,
        num_clients=config["num_clients"],
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": config["client_num_cpus"],
            "num_gpus": config["client_num_gpus"],
        },
        ray_init_args={
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "runtime_env": {
                "py_modules": [str(source_root / "redo_by_sara")],
                "env_vars": {
                    "PYTHONPATH": str(source_root),
                    "OMP_NUM_THREADS": str(max(1, int(config["client_num_cpus"]))),
                    "MKL_NUM_THREADS": str(max(1, int(config["client_num_cpus"]))),
                },
            },
        },
    )
    if len(global_rows) != num_rounds + 1:
        raise AssertionError(
            f"Expected {num_rounds + 1} global train evaluations, found {len(global_rows)}."
        )
    if len(strategy.client_rows) != num_rounds * config["num_clients"]:
        raise AssertionError(
            f"Expected {num_rounds * config['num_clients']} successful fits, "
            f"found {len(strategy.client_rows)}."
        )
    test_guard.assert_untouched_during_training()
    test_guard.unseal()
    final_arrays = parameters_to_ndarrays(strategy.latest_parameters)
    final_model = (
        _build_classification_model(artifact)
        if task == "classification"
        else build_zero_initialized_residual_model()
    )
    set_parameters(final_model, final_arrays)
    core_audit = {
        "seeding": seeding,
        "framework": "Flower",
        "flower_strategy": "FedAvg",
        "num_rounds": num_rounds,
        "local_epochs_per_round": config["local_epochs"],
        "successful_client_fits": len(strategy.client_rows),
        "expected_clients_every_round": sorted(expected_examples),
        "fedavg_num_examples_by_client": expected_examples,
        "fedavg_total_weight": sum(expected_examples.values()),
        "round_one_parameters_changed": strategy.round_one_parameter_changed,
        "global_train_evaluation_rounds": [int(row["round"]) for row in global_rows],
        "test_evaluations_during_training": len(test_guard.accesses_while_sealed),
        "local_optimizer": "Adam recreated independently on every client and round",
        "client_shuffle_seed": "4601 + 1000*server_round + integer_client_id",
        "learning_rate_schedule": "round-wise cosine annealing; t=round-1",
        "round_1_learning_rate": cosine_round_learning_rate(
            1, config["initial_learning_rate"], config["minimum_learning_rate"], num_rounds
        ),
        "final_round_learning_rate": cosine_round_learning_rate(
            num_rounds,
            config["initial_learning_rate"],
            config["minimum_learning_rate"],
            num_rounds,
        ),
    }
    return final_model, global_rows, strategy.client_rows, core_audit


def _finalize_classification(
    model: nn.Module,
    artifact: dict[str, object],
    global_rows: Sequence[dict[str, Any]],
    client_rows: Sequence[dict[str, Any]],
    core_audit: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    num_rounds: int,
    test_guard: TestSetAccessGuard | None = None,
) -> dict[str, Any]:
    from .combined_iid_flower_plots import (
        plot_classification_confusion,
        plot_classification_confusion_comparison,
        plot_classification_test_metric_comparison,
        plot_classification_training_comparison,
    )

    model_path = output_dir / "flower_iid_global_classification_model.pt"
    history_path = output_dir / "global_train_history.csv"
    client_history_path = output_dir / "local_client_fit_history.csv"
    confusion_csv = output_dir / "test_confusion_matrix.csv"
    predictions_path = output_dir / "test_predictions.csv"
    summary_path = output_dir / "training_summary.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "SimpleCNN1D(9 input channels, 8 class logits)",
            "framework": "Flower",
            "strategy": "FedAvg",
            "class_to_index": artifact["class_to_index"],
            "selected_channels_one_based": artifact["summary"][
                "selected_channels_one_based"
            ],
            "channel_mean": artifact["channel_mean"],
            "channel_std": artifact["channel_std"],
            "seed": config["seed"],
            "rounds": num_rounds,
        },
        model_path,
    )

    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    verified_model = _build_classification_model(artifact)
    verified_model.load_state_dict(saved["model_state_dict"])
    train_loader = DataLoader(
        CombinedArtifactDataset(artifact, artifact["train_indices"].tolist()),
        batch_size=config["classification_batch_size"],
        shuffle=False,
        num_workers=0,
    )
    verified_train, _, _ = _evaluate_classification(
        verified_model, train_loader, artifact["index_to_class"]
    )
    final_history = global_rows[-1]
    for key, value in (
        ("global_train_cross_entropy", verified_train["cross_entropy"]),
        ("global_train_accuracy", verified_train["accuracy"]),
        ("global_train_balanced_accuracy", verified_train["balanced_accuracy"]),
        ("global_train_macro_f1", verified_train["macro_f1"]),
    ):
        if not math.isclose(float(final_history[key]), float(value), abs_tol=1e-7):
            raise AssertionError(f"Saved final classification model disagrees with {key}.")

    if test_guard is None:
        test_guard = TestSetAccessGuard(artifact, label="classification")
    test_indices = test_guard.test_indices(
        f"final held-out classification evaluation after round {num_rounds}"
    )
    test_loader = DataLoader(
        CombinedArtifactDataset(artifact, test_indices),
        batch_size=config["classification_batch_size"],
        shuffle=False,
        num_workers=0,
    )
    test_metrics, matrix, predictions = _evaluate_classification(
        verified_model, test_loader, artifact["index_to_class"]
    )
    expected_support = [69, 32, 43, 56, 25, 25, 27, 30]
    if int(matrix.sum()) != 307 or matrix.sum(axis=1).tolist() != expected_support:
        raise AssertionError(
            f"Unexpected held-out classification support: {matrix.sum(axis=1).tolist()}"
        )
    _write_confusion(confusion_csv, matrix, artifact["index_to_class"])
    _write_classification_predictions(
        predictions_path, artifact, test_indices, predictions
    )

    central_summary_path = config["centralized_classification_dir"] / "training_summary.json"
    central_summary = json.loads(central_summary_path.read_text())
    summary: dict[str, Any] = {
        "experiment": "combined_3client_iid_flower_classification",
        "seeding": core_audit["seeding"],
        "framework": "Flower",
        "strategy": "FedAvg",
        "architecture": "SimpleCNN1D with GroupNorm; 9 inputs and 8 outputs",
        "rounds": num_rounds,
        "local_epochs_per_round": config["local_epochs"],
        "subjects": ["001", "002", "003", "004", "005", "007", "008"],
        "sample_rates": _sample_rate_audit(artifact, config),
        "classes": artifact["index_to_class"],
        "selected_channels_one_based": artifact["summary"][
            "selected_channels_one_based"
        ],
        "final_global_train_metrics": {
            key.removeprefix("global_train_"): value
            for key, value in final_history.items()
            if key.startswith("global_train_")
        },
        "test_metrics": test_metrics,
        "test_evaluation_policy": (
            f"Held-out 20% run split evaluated exactly once after Flower round {num_rounds}; "
            "no validation or best-round selection."
        ),
        "test_set_access_audit": test_guard.report(),
        "test_evaluation_count": test_guard.count,
        "test_round": num_rounds,
        "comparison_vs_centralized": {
            "accuracy_difference": test_metrics["accuracy"]
            - float(central_summary["test_accuracy"]),
            "balanced_accuracy_difference": test_metrics["balanced_accuracy"]
            - float(central_summary["test_balanced_accuracy"]),
            "macro_f1_difference": test_metrics["macro_f1"]
            - float(central_summary["test_macro_f1"]),
        },
        "integrity_audit": {
            **core_audit,
            "saved_model_reproduces_final_global_train_history": True,
            "final_test_confusion_total": int(matrix.sum()),
            "final_test_class_support": expected_support,
            **test_guard.report(),
        },
        "model_path": str(model_path),
        "global_train_history_path": str(history_path),
        "local_client_fit_history_path": str(client_history_path),
        "test_predictions_path": str(predictions_path),
        "test_confusion_matrix_csv": str(confusion_csv),
    }

    plot_paths = {
        "training_comparison_plot": output_dir
        / "centralized_vs_flower_global_train_loss_accuracy.png",
        "test_confusion_matrix_plot": output_dir / "test_confusion_matrix.png",
        "test_confusion_comparison_plot": output_dir
        / "centralized_vs_flower_test_confusion_matrices.png",
        "test_metric_comparison_plot": output_dir
        / "centralized_vs_flower_test_metrics.png",
    }
    plot_classification_training_comparison(
        global_rows,
        config["centralized_classification_dir"] / "training_history.csv",
        plot_paths["training_comparison_plot"],
    )
    plot_classification_confusion(
        matrix, artifact["index_to_class"], plot_paths["test_confusion_matrix_plot"]
    )
    plot_classification_confusion_comparison(
        config["centralized_classification_dir"] / "test_confusion_matrix.csv",
        matrix,
        artifact["index_to_class"],
        plot_paths["test_confusion_comparison_plot"],
    )
    plot_classification_test_metric_comparison(
        central_summary, summary, plot_paths["test_metric_comparison_plot"]
    )
    summary.update({key: str(value) for key, value in plot_paths.items()})
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _finalize_regression(
    model: nn.Module,
    artifact: dict[str, object],
    global_rows: Sequence[dict[str, Any]],
    client_rows: Sequence[dict[str, Any]],
    core_audit: dict[str, Any],
    source_subject_means: dict[tuple[str, str], float],
    residual_scale_mps: float,
    config: dict[str, Any],
    output_dir: Path,
    num_rounds: int,
    test_guard: TestSetAccessGuard | None = None,
    allow_degenerate: bool = False,
) -> dict[str, Any]:
    from .combined_iid_flower_plots import (
        plot_regression_actual_vs_predicted,
        plot_regression_paired_run_absolute_errors,
        plot_regression_rmse_comparison,
        plot_regression_subject_rmse_comparison,
        plot_regression_training_comparison,
    )

    model_path = output_dir / "flower_iid_global_residual_regression_model.pt"
    history_path = output_dir / "global_train_history.csv"
    client_history_path = output_dir / "local_client_fit_history.csv"
    run_predictions_path = output_dir / "test_run_predictions.csv"
    window_predictions_path = output_dir / "test_window_predictions.csv"
    means_path = output_dir / "train_source_subject_means.csv"
    summary_path = output_dir / "training_summary.json"
    means_rows = [
        {
            "source_id": source_id,
            "subject_id": subject_id,
            "train_mean_speed_mps": value,
        }
        for (source_id, subject_id), value in sorted(source_subject_means.items())
    ]
    _write_rows(means_path, means_rows)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "SimpleCNN1D(9 input channels, 1 standardized residual output)",
            "framework": "Flower",
            "strategy": "FedAvg weighted by unique client runs",
            "selected_channels_one_based": artifact["summary"][
                "selected_channels_one_based"
            ],
            "channel_mean": artifact["channel_mean"],
            "channel_std": artifact["channel_std"],
            "source_subject_train_means": means_rows,
            "train_residual_scale_mps": residual_scale_mps,
            "seed": config["seed"],
            "rounds": num_rounds,
        },
        model_path,
    )

    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    verified_model = build_zero_initialized_residual_model()
    verified_model.load_state_dict(saved["model_state_dict"])
    train_dataset = RunGroupedResidualDataset(
        artifact,
        artifact["train_indices"].tolist(),
        source_subject_means,
        residual_scale_mps,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["regression_run_batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_whole_runs,
    )
    train_result = _run_group_epoch(
        verified_model, train_loader, nn.MSELoss(), torch.device("cpu"), optimizer=None
    )
    verified_train = _regression_history_row(
        num_rounds, train_result, residual_scale_mps
    )
    final_history = global_rows[-1]
    for key in (
        "global_train_standardized_mse",
        "global_train_mse_mps2",
        "global_train_rmse_mps",
        "global_train_mae_mps",
        "global_train_bias_mps",
        "global_train_r2",
        "global_train_within_0_10_mps_pct",
    ):
        if not math.isclose(
            float(final_history[key]), float(verified_train[key]), abs_tol=1e-7
        ):
            raise AssertionError(f"Saved final regression model disagrees with {key}.")

    if test_guard is None:
        test_guard = TestSetAccessGuard(artifact, label="regression")
    test_dataset = RunGroupedResidualDataset(
        artifact,
        test_guard.test_indices(
            f"final held-out residual regression evaluation after round {num_rounds}"
        ),
        source_subject_means,
        residual_scale_mps,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["regression_run_batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_whole_runs,
    )
    test_result = _run_group_epoch(
        verified_model,
        test_loader,
        nn.MSELoss(),
        torch.device("cpu"),
        optimizer=None,
        collect_windows=True,
    )
    run_rows, window_rows = _build_regression_prediction_rows(
        artifact, test_dataset, test_result, source_subject_means
    )
    if len(run_rows) != 28 or len({row["run_uid"] for row in run_rows}) != 28:
        raise AssertionError("Final regression test does not contain 28 unique runs.")
    _write_rows(run_predictions_path, run_rows)
    _write_rows(window_predictions_path, window_rows)
    run_metrics = _extended_regression_metrics(
        np.asarray([row["actual_speed_mps"] for row in run_rows]),
        np.asarray([row["predicted_speed_mps"] for row in run_rows]),
    )
    window_metrics = _extended_regression_metrics(
        np.asarray([row["actual_speed_mps"] for row in window_rows]),
        np.asarray([row["predicted_speed_mps"] for row in window_rows]),
    )
    baseline_metrics = _extended_regression_metrics(
        np.asarray([row["actual_speed_mps"] for row in run_rows]),
        np.asarray([row["source_subject_train_mean_mps"] for row in run_rows]),
    )
    # The degenerate-model detector, run before the summary is written so a collapsed
    # run cannot be reported as a near-match to centralized (R1). Both gates are enforced
    # on the residual path: this is where the collapse happened.
    health = check_regression_health(
        actual=[row["actual_speed_mps"] for row in run_rows],
        predicted=[row["predicted_speed_mps"] for row in run_rows],
        baseline=[row["source_subject_train_mean_mps"] for row in run_rows],
        final_standardized_mse=float(final_history["global_train_standardized_mse"]),
        enforced_gates=("degeneracy", "skill"),
        run_label="combined_3client_iid_flower_residual_run_regression",
        # Gate degeneracy on the window-level residual, which is what the network emits.
        # The reported speed is baseline + residual, so a constant model still produces
        # one value per source/subject stratum (8) and looks less degenerate than it is.
        model_output=[row["predicted_residual_mps"] for row in window_rows],
        model_output_name="window predicted residual",
        allow_degenerate=allow_degenerate,
    )
    # Window-level distinctness is the sharper diagnostic: the collapse produced one
    # distinct predicted residual across all 238 test windows, which the 28 run-level
    # rows can mask because they differ by their per-subject baselines.
    health["window_level"] = {
        "num_predictions": len(window_rows),
        "distinct_predicted_residual_mps": len(
            {row["predicted_residual_mps"] for row in window_rows}
        ),
        "distinct_predicted_speed_mps": len(
            {row["predicted_speed_mps"] for row in window_rows}
        ),
    }

    central_summary_path = config["centralized_regression_dir"] / "training_summary.json"
    central_summary = json.loads(central_summary_path.read_text())
    central_metrics = central_summary["test_run_metrics"]
    summary: dict[str, Any] = {
        "experiment": "combined_3client_iid_flower_residual_run_regression",
        "seeding": core_audit["seeding"],
        "framework": "Flower",
        "strategy": "FedAvg weighted by unique complete training runs",
        "architecture": "Unchanged SimpleCNN1D with one standardized residual output",
        "rounds": num_rounds,
        "local_epochs_per_round": config["local_epochs"],
        "subjects": ["001", "002", "003", "004", "005", "007", "008"],
        "sample_rates": _sample_rate_audit(artifact, config),
        "selected_channels_one_based": artifact["summary"][
            "selected_channels_one_based"
        ],
        "method": {
            "baseline": "Federated global source/subject mean from unique train-run sufficient statistics",
            "train_residual_scale_mps": residual_scale_mps,
            "prediction": "source_subject_train_mean + residual_scale * mean(window standardized residual predictions)",
            "optimization_unit": "complete run",
            "client_aggregation_weight": "unique complete training runs (37/38/37)",
        },
        "round_0_global_train_metrics": {
            key.removeprefix("global_train_"): value
            for key, value in global_rows[0].items()
            if key.startswith("global_train_")
        },
        "final_global_train_metrics": {
            key.removeprefix("global_train_"): value
            for key, value in final_history.items()
            if key.startswith("global_train_")
        },
        "degenerate_model_check": health,
        "test_run_metrics": run_metrics,
        "test_window_metrics": window_metrics,
        "test_source_subject_mean_baseline_run_metrics": baseline_metrics,
        "test_run_metrics_by_subject": _metrics_by_field(run_rows, "subject_id"),
        "test_run_metrics_by_source": _metrics_by_field(run_rows, "source_id"),
        "test_evaluation_policy": (
            f"Held-out 20% run split evaluated exactly once after Flower round {num_rounds}; "
            "no validation or best-round selection."
        ),
        "test_set_access_audit": test_guard.report(),
        "test_evaluation_count": test_guard.count,
        "test_round": num_rounds,
        "comparison_vs_centralized_residual_cnn": {
            "fl_rmse_mps": run_metrics["rmse_mps"],
            "centralized_rmse_mps": central_metrics["rmse_mps"],
            "rmse_difference_mps": run_metrics["rmse_mps"]
            - float(central_metrics["rmse_mps"]),
            "fl_r2": run_metrics["r2"],
            "centralized_r2": central_metrics["r2"],
            "r2_difference": run_metrics["r2"] - float(central_metrics["r2"]),
        },
        "integrity_audit": {
            **core_audit,
            "saved_model_reproduces_final_global_train_history": True,
            "final_test_unique_runs": len(run_rows),
            "final_test_windows": len(window_rows),
            **test_guard.report(),
        },
        "model_path": str(model_path),
        "global_train_history_path": str(history_path),
        "local_client_fit_history_path": str(client_history_path),
        "train_source_subject_means_path": str(means_path),
        "test_run_predictions_path": str(run_predictions_path),
        "test_window_predictions_path": str(window_predictions_path),
    }

    central_predictions = config["centralized_regression_dir"] / "test_run_predictions.csv"
    plot_paths = {
        "training_comparison_plot": output_dir
        / "centralized_vs_flower_global_train_metrics.png",
        "actual_vs_predicted_plot": output_dir / "actual_vs_predicted_speed.png",
        "heldout_rmse_comparison_plot": output_dir
        / "heldout_run_rmse_comparison.png",
        "per_subject_rmse_comparison_plot": output_dir
        / "centralized_vs_flower_per_subject_rmse.png",
        "paired_run_error_plot": output_dir
        / "centralized_vs_flower_paired_run_absolute_errors.png",
    }
    plot_regression_training_comparison(
        global_rows,
        config["centralized_regression_dir"] / "training_history.csv",
        plot_paths["training_comparison_plot"],
    )
    plot_regression_actual_vs_predicted(
        run_rows, plot_paths["actual_vs_predicted_plot"]
    )
    plot_regression_rmse_comparison(
        config["centralized_absolute_regression_summary"],
        central_summary,
        summary,
        plot_paths["heldout_rmse_comparison_plot"],
    )
    plot_regression_subject_rmse_comparison(
        central_predictions,
        run_rows,
        plot_paths["per_subject_rmse_comparison_plot"],
    )
    plot_regression_paired_run_absolute_errors(
        central_predictions, run_rows, plot_paths["paired_run_error_plot"]
    )
    summary.update({key: str(value) for key, value in plot_paths.items()})
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def prepare_combined_iid_flower_experiment(
    config: dict[str, Any],
    output_root: str | Path | None = None,
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Audit the setup and, unless ``write_outputs`` is False, persist it.

    ``write_outputs=False`` is what makes ``--verify-only`` a genuine verification: every
    write is routed through one :class:`GuardedWriter`, so the run produces no directory,
    no ``setup_audit.json``, no partition files and no plot. A verification that mutates
    its own outputs is not a verification (§3 item 2).
    """

    from .combined_iid_fl_partitioning import save_combined_iid_partitions
    from .combined_iid_flower_plots import plot_partition_balance

    classification, regression, partitions, setup = audit_combined_iid_flower_setup(
        config
    )
    destination = (
        Path(output_root).resolve()
        if output_root is not None
        else Path(config["output_dir"])
    )
    writer = GuardedWriter(destination, enabled=write_outputs)
    writer.mkdir()
    partition_dir = destination / "partitions"
    partition_paths = writer.call(
        save_combined_iid_partitions, partitions, setup["partition"], partition_dir
    )
    if partition_paths is None:
        # Verify mode: report the paths a real run would produce without creating them.
        partition_paths = {
            "partitions": partition_dir / "combined_iid_client_partitions.json",
            "run_manifest": partition_dir / "combined_iid_run_to_client.csv",
            "summary": partition_dir / "combined_iid_partition_summary.json",
        }
        writer.planned_writes.extend(str(value) for value in partition_paths.values())
    setup_path = destination / "setup_audit.json"
    setup["partition_files"] = {
        key: str(value) for key, value in partition_paths.items()
    }
    setup["conditions"] = {
        "datasets": ["Test_2", "20251124_Testing"],
        "test_2_effective_sample_rate_hz": 1706.667,
        "target_sample_rate_hz": 400.0,
        "window_seconds": 5.0,
        "step_seconds": 1.0,
        "channels_one_based": [1, 2, 3, 4, 5, 6, 7, 8, 10],
        "subjects": ["001", "002", "003", "004", "005", "007", "008"],
        "excluded_subject": "006",
        "train_test_split": "80/20 by complete run before windowing",
        "validation_split": None,
        "classification_no_walking_class": True,
        "regression_walking_only": True,
    }
    writer.write_text(setup_path, json.dumps(setup, indent=2))
    partition_plot = destination / "partitions" / "client_partition_balance.png"
    writer.call(plot_partition_balance, setup["partition"], partition_plot)
    return {
        "classification_artifact": classification,
        "regression_artifact": regression,
        "partitions": partitions,
        "setup": setup,
        "output_root": destination,
        "setup_path": setup_path,
        "partition_plot": partition_plot,
        "writer": writer,
    }


def run_combined_iid_flower_experiment(
    config: dict[str, Any],
    task: str,
    *,
    num_rounds_override: int | None = None,
    output_root: str | Path | None = None,
    allow_degenerate: bool = False,
) -> dict[str, Any]:
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'.")
    num_rounds = (
        config["num_rounds"]
        if num_rounds_override is None
        else int(num_rounds_override)
    )
    if num_rounds <= 0:
        raise ValueError("The number of Flower rounds must be positive.")
    if num_rounds_override is not None and output_root is None:
        raise ValueError(
            "An output_root is required when overriding rounds so a smoke test cannot overwrite production outputs."
        )
    prepared = prepare_combined_iid_flower_experiment(config, output_root)
    task_output = prepared["output_root"] / task
    if task == "classification":
        artifact = prepared["classification_artifact"]
        # One guard spans training and finalization, so "the test set was untouched
        # during training" is measured on the same counter that the final evaluation
        # increments, rather than asserted about a counter nothing ever touched (R9).
        test_guard = TestSetAccessGuard(artifact, label="classification")
        model, global_rows, client_rows, core_audit = _run_flower_core(
            task="classification",
            artifact=artifact,
            artifact_path=config["classification_artifact"],
            partitions=prepared["partitions"],
            setup=prepared["setup"],
            config=config,
            output_dir=task_output,
            num_rounds=num_rounds,
            test_guard=test_guard,
        )
        return _finalize_classification(
            model,
            artifact,
            global_rows,
            client_rows,
            core_audit,
            config,
            task_output,
            num_rounds,
            test_guard=test_guard,
        )

    artifact = prepared["regression_artifact"]
    means, scale, _ = _federated_residual_baseline(
        artifact, prepared["partitions"]
    )
    test_guard = TestSetAccessGuard(artifact, label="regression")
    model, global_rows, client_rows, core_audit = _run_flower_core(
        task="regression",
        artifact=artifact,
        artifact_path=config["regression_artifact"],
        partitions=prepared["partitions"],
        setup=prepared["setup"],
        config=config,
        output_dir=task_output,
        num_rounds=num_rounds,
        source_subject_means=means,
        residual_scale_mps=scale,
        test_guard=test_guard,
    )
    return _finalize_regression(
        model,
        artifact,
        global_rows,
        client_rows,
        core_audit,
        means,
        scale,
        config,
        task_output,
        num_rounds,
        test_guard=test_guard,
        allow_degenerate=allow_degenerate,
    )

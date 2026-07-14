from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.config import ExperimentConfig, load_config
from redo_by_sara.federated import (
    create_federated_model,
    evaluate_model,
    get_parameters,
    regression_r2_score,
    save_partition_summary,
    save_round_history,
    set_parameters,
    train_local_model,
)
from redo_by_sara.sensor_non_iid import (
    SensorClientPartition,
    build_sensor_client_partitions,
    build_sensor_masked_loader,
    create_sensor_partition_summary,
    resolve_sensor_client_map,
)


class FlowerDependencyError(RuntimeError):
    pass


def _build_wandb_config(
    config: ExperimentConfig,
    artifact: dict[str, object],
    client_sensors: dict[str, list[str]],
) -> dict[str, Any]:
    federated = config.federated
    if federated is None:
        raise ValueError("Missing federated config.")
    return {
        "seed": config.seed,
        "task": config.training.task,
        "partition_mode": "sensor_owned_feature_non_iid",
        "model_name": _model_name_for_config(config),
        "mask_as_input": _uses_mask_aware_classifier(config),
        "artifact_name": config.artifact_name,
        "dataset_root": str(config.data.dataset_root),
        "selected_sensors": config.data.selected_sensors,
        "selected_channels": artifact["summary"]["selected_channels"],
        "sensor_client_map": client_sensors,
        "num_train": artifact["summary"]["num_train"],
        "num_val": artifact["summary"]["num_val"],
        "num_test": artifact["summary"]["num_test"],
        "sample_shape": artifact["summary"]["sample_shape"],
        "num_clients": federated.num_clients,
        "num_rounds": federated.num_rounds,
        "local_epochs": federated.local_epochs,
        "result_name": federated.result_name,
        "split_before_resampling": artifact["summary"].get("split_before_resampling"),
        "split_before_windowing": artifact["summary"].get("split_before_windowing"),
        "purge_gap_seconds": artifact["summary"].get("purge_gap_seconds"),
    }


def _build_run_name(config: ExperimentConfig) -> str:
    federated = config.federated
    if federated is None:
        raise ValueError("Missing federated config.")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return (
        f"{config.training.task}-flower-sensor-non-iid-"
        f"{federated.num_clients}c-{federated.num_rounds}r-"
        f"{federated.result_name + '-' if federated.result_name else ''}{timestamp}"
    )


def _init_wandb(
    config: ExperimentConfig,
    artifact: dict[str, object],
    client_sensors: dict[str, list[str]],
) -> wandb.sdk.wandb_run.Run | None:
    if not config.wandb.enabled:
        return None

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        tags=(config.wandb.tags or []) + ["flower", "federated-learning", "sensor-non-iid"],
        config=_build_wandb_config(config, artifact, client_sensors),
        name=_build_run_name(config),
        job_type="federated-sensor-non-iid-train",
    )
    wandb.log(
        {
            "dataset/num_train": artifact["summary"]["num_train"],
            "dataset/num_val": artifact["summary"]["num_val"],
            "dataset/num_test": artifact["summary"]["num_test"],
        },
        step=0,
    )
    return run


def _aggregate_fit_metrics(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    for key in ("train_loss", "train_score", "train_r2"):
        weighted_sum = 0.0
        contributed = False
        for num_examples, client_metrics in metrics:
            if key not in client_metrics:
                continue
            weighted_sum += float(client_metrics[key]) * num_examples
            contributed = True
        if contributed:
            aggregated[key] = weighted_sum / total_examples
    return aggregated


def _coerce_ndarrays(parameters: Any, parameters_to_ndarrays: Any) -> list[Any]:
    if isinstance(parameters, list):
        return parameters
    return parameters_to_ndarrays(parameters)


def _client_id_from_context(client_context: Any) -> str:
    node_config = getattr(client_context, "node_config", None)
    if node_config is None:
        return str(client_context)
    return str(
        node_config.get(
            "partition-id",
            node_config.get("partition_id", node_config.get("cid", str(client_context))),
        )
    )


def _result_stem(config: ExperimentConfig) -> str:
    federated = config.federated
    if federated is None:
        raise ValueError("Missing federated config.")
    suffix = f"_{federated.result_name}" if federated.result_name else ""
    return f"{config.training.task}_sensor_non_iid{suffix}"


def _uses_mask_aware_classifier(config: ExperimentConfig) -> bool:
    model_name = _model_name_for_config(config)
    if model_name == "mask_aware_simple_cnn1d":
        if config.training.task != "classification":
            raise ValueError("mask_aware_simple_cnn1d is currently supported for classification only.")
        return True
    if model_name == "simple_cnn1d":
        return False
    raise ValueError(f"Unsupported model_name for sensor non-IID FL: {model_name}")


def _model_name_for_config(config: ExperimentConfig) -> str:
    return config.training.model_name


def _score_row_for_result(
    task: str,
    loss: float,
    score: float,
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | None]:
    row: dict[str, float | None] = {
        "loss": float(loss),
        "score": float(score),
        "r2": None,
    }
    if task == "regression":
        row["r2"] = float(regression_r2_score(outputs, targets))
    return row


def _class_names(artifact: dict[str, object]) -> list[str]:
    subject_to_class = artifact["subject_to_class"]
    return [subject for subject, _ in sorted(subject_to_class.items(), key=lambda item: item[1])]


def _confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_id, pred_id in zip(y_true, y_pred, strict=True):
        matrix[int(true_id)][int(pred_id)] += 1
    return matrix


def _format_cell(value: int, row_total: int) -> str:
    if row_total == 0:
        return str(value)
    return f"{value}\n{value / row_total:.0%}"


def _plot_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted subject")
    ax.set_ylabel("True subject")
    ax.set_title(title)

    max_value = max(max(row) for row in matrix) if matrix else 0
    for row_idx, row in enumerate(matrix):
        row_total = sum(row)
        for col_idx, value in enumerate(row):
            color = "white" if max_value and value > max_value / 2 else "black"
            ax.text(
                col_idx,
                row_idx,
                _format_cell(value, row_total),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_classification_confusion_outputs(
    artifact: dict[str, object],
    outputs: torch.Tensor,
    targets: torch.Tensor,
    accuracy: float,
    model_path: Path,
    config_path: Path,
    output_path: Path,
) -> dict[str, object]:
    class_names = _class_names(artifact)
    y_true = [int(item) for item in targets.reshape(-1).tolist()]
    y_pred = [int(item) for item in torch.argmax(outputs, dim=1).reshape(-1).tolist()]
    matrix = _confusion_matrix(y_true, y_pred, len(class_names))
    title = f"Sensor Non-IID Test Confusion Matrix Accuracy {accuracy:.3f}"
    _plot_confusion_matrix(matrix, class_names, title, output_path)

    payload = {
        "output_path": str(output_path),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "split": "test",
        "accuracy": float(accuracy),
        "num_examples": len(y_true),
        "class_names": class_names,
        "matrix": matrix,
    }
    json_output = output_path.with_suffix(".json")
    json_output.write_text(json.dumps(payload, indent=2))
    return {**payload, "json_output": str(json_output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sensor-owned non-IID Flower simulation.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.federated is None:
        raise ValueError("This config does not contain a federated section.")
    if abs(config.data.val_ratio) > 1e-9:
        raise ValueError("Sensor non-IID runs use no validation split; set data.val_ratio to 0.0.")

    try:
        from flwr.client import NumPyClient
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
        from flwr.server import ServerConfig
        from flwr.server.strategy import FedAvg
        from flwr.simulation import start_simulation
    except ImportError as exc:
        raise FlowerDependencyError(
            "Flower is not installed in this environment. Install `flwr[simulation]` first."
        ) from exc

    artifact = torch.load(config.artifact_path, map_location="cpu", weights_only=False)
    client_sensors = resolve_sensor_client_map(config)
    client_partitions = build_sensor_client_partitions(
        artifact=artifact,
        num_clients=config.federated.num_clients,
        client_sensors=client_sensors,
    )
    partitions_by_id = {partition.client_id: partition for partition in client_partitions}
    partition_summary = create_sensor_partition_summary(
        artifact=artifact,
        client_partitions=client_partitions,
    )

    result_stem = _result_stem(config)
    partition_summary_path = config.output_dir / f"{result_stem}_federated_partitions.json"
    history_path = config.output_dir / f"{result_stem}_federated_history.csv"
    client_history_path = config.output_dir / f"{result_stem}_federated_client_history.csv"
    summary_path = config.output_dir / f"{result_stem}_federated_summary.json"
    model_path = config.output_dir / f"{result_stem}_flower_model.pt"
    confusion_path = config.output_dir / "confusion_comparisons" / f"{result_stem}_test_confusion.png"
    save_partition_summary(partition_summary_path, partition_summary)

    run = _init_wandb(config, artifact, client_sensors)

    class FlowerSensorClient(NumPyClient):
        def __init__(
            self,
            artifact_path: Path,
            partition: SensorClientPartition,
            experiment: ExperimentConfig,
        ) -> None:
            self.partition = partition
            self.experiment = experiment
            self.task = experiment.training.task
            self.mask_as_input = _uses_mask_aware_classifier(experiment)
            self.local_epochs = experiment.federated.local_epochs if experiment.federated else 1
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                and experiment.federated is not None
                and experiment.federated.client_num_gpus > 0
                else "cpu"
            )
            self.artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
            self.train_loader = build_sensor_masked_loader(
                artifact=self.artifact,
                indices=self.partition.train_indices,
                task=self.task,
                batch_size=experiment.training.batch_size,
                num_workers=experiment.training.num_workers,
                shuffle=True,
                seed=experiment.seed + int(partition.client_id),
                channel_indices=self.partition.channel_indices,
                include_channel_mask=self.mask_as_input,
            )

        def get_parameters(self, config: dict[str, Any]) -> list[Any]:
            model = create_federated_model(self.artifact, self.task, mask_aware=self.mask_as_input)
            return get_parameters(model)

        def fit(
            self,
            parameters: list[Any],
            config: dict[str, Any],
        ) -> tuple[list[Any], int, dict[str, float]]:
            model = create_federated_model(self.artifact, self.task, mask_aware=self.mask_as_input).to(self.device)
            set_parameters(model, parameters)
            train_result = train_local_model(
                model=model,
                loader=self.train_loader,
                task=self.task,
                device=self.device,
                epochs=self.local_epochs,
                learning_rate=self.experiment.training.learning_rate,
                weight_decay=self.experiment.training.weight_decay,
            )
            metrics = {
                "client_id": self.partition.client_id,
                "train_loss": float(train_result.loss),
                "train_score": float(train_result.score),
            }
            if self.task == "regression":
                metrics["train_r2"] = float(regression_r2_score(train_result.outputs, train_result.targets))
            return get_parameters(model), len(self.partition.train_indices), metrics

        def evaluate(
            self,
            parameters: list[Any],
            config: dict[str, Any],
        ) -> tuple[float, int, dict[str, float]]:
            return 0.0, len(self.partition.train_indices), {}

    class TrackingFedAvg(FedAvg):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.latest_parameters = kwargs.get("initial_parameters")
            self.fit_rows: list[dict[str, float]] = []
            self.client_rows: list[dict[str, float | str]] = []

        def aggregate_fit(self, server_round: int, results: Any, failures: Any) -> tuple[Any, dict[str, float]]:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                self.latest_parameters = aggregated_parameters

            for _, fit_result in results:
                metrics = getattr(fit_result, "metrics", {}) or {}
                client_id = str(metrics.get("client_id", "unknown"))
                client_row: dict[str, float | str] = {
                    "round": float(server_round),
                    "client_id": client_id,
                    "num_examples": float(getattr(fit_result, "num_examples", 0)),
                }
                if "train_loss" in metrics:
                    client_row["train_loss"] = float(metrics["train_loss"])
                if "train_score" in metrics:
                    client_row["train_score"] = float(metrics["train_score"])
                if "train_r2" in metrics:
                    client_row["train_r2"] = float(metrics["train_r2"])
                self.client_rows.append(client_row)

                if run is not None:
                    client_payload = {
                        f"clients/{client_id}/train_loss": client_row.get("train_loss"),
                        f"clients/{client_id}/train_score": client_row.get("train_score"),
                        f"clients/{client_id}/num_examples": client_row["num_examples"],
                    }
                    if "train_r2" in client_row:
                        client_payload[f"clients/{client_id}/train_r2"] = client_row["train_r2"]
                    wandb.log(client_payload, step=server_round)

            row = {"round": float(server_round)}
            if "train_loss" in aggregated_metrics:
                row["train_loss"] = float(aggregated_metrics["train_loss"])
            if "train_score" in aggregated_metrics:
                row["train_score"] = float(aggregated_metrics["train_score"])
            if "train_r2" in aggregated_metrics:
                row["train_r2"] = float(aggregated_metrics["train_r2"])
            self.fit_rows.append(row)

            if run is not None and len(row) > 1:
                fit_payload = {
                    "federated/train_loss": row.get("train_loss"),
                    "federated/train_score": row.get("train_score"),
                }
                if "train_r2" in row:
                    fit_payload["federated/train_r2"] = row["train_r2"]
                wandb.log(fit_payload, step=server_round)
            return aggregated_parameters, aggregated_metrics

    server_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_as_input = _uses_mask_aware_classifier(config)
    model_name = _model_name_for_config(config)
    initial_model = create_federated_model(artifact, config.training.task, mask_aware=mask_as_input)
    initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))
    strategy = TrackingFedAvg(
        fraction_fit=config.federated.fraction_fit,
        fraction_evaluate=config.federated.fraction_evaluate,
        min_fit_clients=max(1, math.ceil(config.federated.num_clients * config.federated.fraction_fit)),
        min_evaluate_clients=(
            0
            if config.federated.fraction_evaluate <= 0
            else max(1, math.ceil(config.federated.num_clients * config.federated.fraction_evaluate))
        ),
        min_available_clients=config.federated.num_clients,
        fit_metrics_aggregation_fn=_aggregate_fit_metrics,
        initial_parameters=initial_parameters,
    )

    def client_fn(client_context: Any) -> Any:
        client_id = _client_id_from_context(client_context)
        return FlowerSensorClient(
            artifact_path=config.artifact_path,
            partition=partitions_by_id[client_id],
            experiment=config,
        ).to_client()

    try:
        start_simulation(
            client_fn=client_fn,
            num_clients=config.federated.num_clients,
            config=ServerConfig(num_rounds=config.federated.num_rounds),
            strategy=strategy,
            client_resources={
                "num_cpus": config.federated.client_num_cpus,
                "num_gpus": config.federated.client_num_gpus,
            },
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "runtime_env": {
                    "py_modules": [str(SRC / "redo_by_sara")],
                },
            },
        )

        final_parameters = parameters_to_ndarrays(strategy.latest_parameters)
        final_model = create_federated_model(artifact, config.training.task, mask_aware=mask_as_input).to(server_device)
        set_parameters(final_model, _coerce_ndarrays(final_parameters, parameters_to_ndarrays))

        test_loader = build_sensor_masked_loader(
            artifact=artifact,
            indices=artifact["test_indices"],
            task=config.training.task,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            shuffle=False,
            seed=config.seed,
            channel_indices=None,
            include_channel_mask=mask_as_input,
        )
        test_result = evaluate_model(
            model=final_model,
            loader=test_loader,
            task=config.training.task,
            device=server_device,
        )
        test_metrics = _score_row_for_result(
            task=config.training.task,
            loss=test_result.loss,
            score=test_result.score,
            outputs=test_result.outputs,
            targets=test_result.targets,
        )

        masked_client_test_metrics = []
        for partition in client_partitions:
            masked_test_loader = build_sensor_masked_loader(
                artifact=artifact,
                indices=artifact["test_indices"],
                task=config.training.task,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
                shuffle=False,
                seed=config.seed,
                channel_indices=partition.channel_indices,
                include_channel_mask=mask_as_input,
            )
            masked_result = evaluate_model(
                model=final_model,
                loader=masked_test_loader,
                task=config.training.task,
                device=server_device,
            )
            masked_row = _score_row_for_result(
                task=config.training.task,
                loss=masked_result.loss,
                score=masked_result.score,
                outputs=masked_result.outputs,
                targets=masked_result.targets,
            )
            masked_row.update(
                {
                    "client_id": partition.client_id,
                    "sensors": partition.sensor_names,
                }
            )
            masked_client_test_metrics.append(masked_row)

        save_round_history(history_path, strategy.fit_rows)
        save_round_history(client_history_path, strategy.client_rows)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), model_path)

        confusion_payload = None
        if config.training.task == "classification":
            confusion_payload = _save_classification_confusion_outputs(
                artifact=artifact,
                outputs=test_result.outputs,
                targets=test_result.targets,
                accuracy=float(test_result.score),
                model_path=model_path,
                config_path=config.config_path,
                output_path=confusion_path,
            )

        summary = {
            "task": config.training.task,
            "partition_mode": "sensor_owned_feature_non_iid",
            "model_name": model_name,
            "mask_as_input": mask_as_input,
            "result_name": config.federated.result_name,
            "model_path": str(model_path),
            "history_path": str(history_path),
            "client_history_path": str(client_history_path),
            "partition_summary_path": str(partition_summary_path),
            "num_clients": config.federated.num_clients,
            "num_rounds": config.federated.num_rounds,
            "local_epochs": config.federated.local_epochs,
            "test_model": "final_round",
            "test_evaluation": "full_server_sensor_set_after_training",
            "test_loss": test_metrics["loss"],
            "test_score": test_metrics["score"],
            "test_r2": test_metrics["r2"],
            "masked_client_test_metrics": masked_client_test_metrics,
            "confusion_matrix_path": None if confusion_payload is None else confusion_payload["output_path"],
            "confusion_matrix_json_path": None if confusion_payload is None else confusion_payload["json_output"],
            "split_overlap": artifact["summary"].get("split_overlap", {}),
        }
        summary_path.write_text(json.dumps(summary, indent=2))

        if run is not None:
            log_payload = {
                "test_loss": summary["test_loss"],
                "test_score": summary["test_score"],
                "test_r2": summary["test_r2"],
            }
            if confusion_payload is not None:
                log_payload["test_confusion_matrix"] = wandb.Image(str(confusion_payload["output_path"]))
            wandb.log(log_payload, step=config.federated.num_rounds)
            run.summary.update(summary)

        print(json.dumps(summary, indent=2))
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()

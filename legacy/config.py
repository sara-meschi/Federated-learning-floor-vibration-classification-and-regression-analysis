from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/config.py"
raise ImportError(
    "Quarantined legacy module: legacy/config.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_root: Path
    sensor_channel_map: dict[str, list[int]]
    selected_sensors: list[str]
    target_sample_rate: float
    window_seconds: float
    step_seconds: float
    train_ratio: float
    val_ratio: float
    test_ratio: float
    subject_limit: int | None = None


@dataclass
class TrainingConfig:
    task: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int = 0
    model_name: str = "simple_cnn1d"


@dataclass
class WandBConfig:
    enabled: bool
    project: str
    entity: str | None = None
    tags: list[str] | None = None
    watch: bool = True


@dataclass
class FederatedConfig:
    num_clients: int
    num_rounds: int
    local_epochs: int
    fraction_fit: float = 1.0
    fraction_evaluate: float = 0.0
    client_num_cpus: float = 1.0
    client_num_gpus: float = 0.0
    client_subjects: dict[str, list[str]] | None = None
    result_name: str | None = None


@dataclass
class ExperimentConfig:
    seed: int
    output_dir: Path
    artifact_name: str
    data: DataConfig
    training: TrainingConfig
    wandb: WandBConfig
    config_path: Path
    federated: FederatedConfig | None = None

    @property
    def artifact_path(self) -> Path:
        return self.output_dir / self.artifact_name


def _resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def _load_sensor_channel_map(raw_map: dict[str, Any]) -> dict[str, list[int]]:
    sensor_map: dict[str, list[int]] = {}
    for sensor_name, channels in raw_map.items():
        if not isinstance(channels, list) or not channels:
            raise ValueError(f"Sensor '{sensor_name}' must map to a non-empty list of channels.")
        sensor_map[str(sensor_name)] = [int(channel) for channel in channels]
    return sensor_map


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    payload: dict[str, Any] = yaml.safe_load(config_path.read_text())

    data_cfg = payload["data"]
    train_cfg = payload["training"]
    wandb_cfg = payload.get("wandb", {})
    federated_cfg = payload.get("federated")

    output_dir = _resolve_path(project_root, payload.get("output_dir", "artifacts"))
    dataset_root = _resolve_path(project_root, data_cfg["dataset_root"])
    sensor_channel_map = _load_sensor_channel_map(data_cfg["sensor_channel_map"])
    selected_sensors = [str(sensor_name) for sensor_name in data_cfg["selected_sensors"]]

    unknown_sensors = [sensor_name for sensor_name in selected_sensors if sensor_name not in sensor_channel_map]
    if unknown_sensors:
        raise ValueError(f"Selected sensors missing from sensor_channel_map: {unknown_sensors}")

    train_ratio = float(data_cfg["train_ratio"])
    val_ratio = float(data_cfg["val_ratio"])
    test_ratio = float(data_cfg["test_ratio"])
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {ratio_sum:.6f}"
        )

    federated: FederatedConfig | None = None
    if federated_cfg is not None:
        client_subjects_raw = federated_cfg.get("client_subjects")
        client_subjects = None
        if client_subjects_raw is not None:
            client_subjects = {
                str(client_id): [str(subject_id) for subject_id in subject_ids]
                for client_id, subject_ids in client_subjects_raw.items()
            }

        federated = FederatedConfig(
            num_clients=int(federated_cfg["num_clients"]),
            num_rounds=int(federated_cfg["num_rounds"]),
            local_epochs=int(federated_cfg["local_epochs"]),
            fraction_fit=float(federated_cfg.get("fraction_fit", 1.0)),
            fraction_evaluate=float(federated_cfg.get("fraction_evaluate", 0.0)),
            client_num_cpus=float(federated_cfg.get("client_num_cpus", 1.0)),
            client_num_gpus=float(federated_cfg.get("client_num_gpus", 0.0)),
            client_subjects=client_subjects,
            result_name=(
                None
                if federated_cfg.get("result_name") in (None, "")
                else str(federated_cfg.get("result_name"))
            ),
        )

        if federated.num_clients < 1:
            raise ValueError("federated.num_clients must be at least 1.")
        if federated.num_rounds < 1:
            raise ValueError("federated.num_rounds must be at least 1.")
        if federated.local_epochs < 1:
            raise ValueError("federated.local_epochs must be at least 1.")

    return ExperimentConfig(
        seed=int(payload["seed"]),
        output_dir=output_dir,
        artifact_name=str(payload.get("artifact_name", "raw_windows.pt")),
        data=DataConfig(
            dataset_root=dataset_root,
            sensor_channel_map=sensor_channel_map,
            selected_sensors=selected_sensors,
            target_sample_rate=float(data_cfg["target_sample_rate"]),
            window_seconds=float(data_cfg["window_seconds"]),
            step_seconds=float(data_cfg["step_seconds"]),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            subject_limit=(None if data_cfg.get("subject_limit") is None else int(data_cfg["subject_limit"])),
        ),
        training=TrainingConfig(
            task=str(train_cfg["task"]),
            batch_size=int(train_cfg["batch_size"]),
            epochs=int(train_cfg["epochs"]),
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
            num_workers=int(train_cfg.get("num_workers", 0)),
            model_name=str(train_cfg.get("model_name", "simple_cnn1d")),
        ),
        wandb=WandBConfig(
            enabled=bool(wandb_cfg.get("enabled", False)),
            project=str(wandb_cfg.get("project", "ENGR859-final-project")),
            entity=(None if wandb_cfg.get("entity") in (None, "") else str(wandb_cfg.get("entity"))),
            tags=None if wandb_cfg.get("tags") is None else [str(tag) for tag in wandb_cfg.get("tags")],
            watch=bool(wandb_cfg.get("watch", True)),
        ),
        config_path=config_path,
        federated=federated,
    )

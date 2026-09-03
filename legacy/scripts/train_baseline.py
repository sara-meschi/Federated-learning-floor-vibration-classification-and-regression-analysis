from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/scripts/train_baseline.py"
raise ImportError(
    "Quarantined legacy module: legacy/scripts/train_baseline.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.config import ExperimentConfig, load_config
from redo_by_sara.training import EvalResult, fit


def _build_wandb_config(config: ExperimentConfig, artifact: dict[str, object]) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "task": config.training.task,
        "artifact_name": config.artifact_name,
        "dataset_root": str(config.data.dataset_root),
        "selected_sensors": config.data.selected_sensors,
        "selected_channels": artifact["summary"]["selected_channels"],
        "target_sample_rate": config.data.target_sample_rate,
        "window_seconds": config.data.window_seconds,
        "step_seconds": config.data.step_seconds,
        "train_ratio": config.data.train_ratio,
        "val_ratio": config.data.val_ratio,
        "test_ratio": config.data.test_ratio,
        "batch_size": config.training.batch_size,
        "epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "num_train": artifact["summary"]["num_train"],
        "num_val": artifact["summary"]["num_val"],
        "num_test": artifact["summary"]["num_test"],
        "sample_shape": artifact["summary"]["sample_shape"],
        "num_subjects": len(artifact["summary"]["subjects"]),
    }


def _build_run_name(config: ExperimentConfig, artifact: dict[str, object]) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    num_channels = len(artifact["summary"]["selected_channels"])
    return f"{config.training.task}-{num_channels}ch-{config.training.epochs}ep-{timestamp}"



def _init_wandb(config: ExperimentConfig, artifact: dict[str, object]) -> wandb.sdk.wandb_run.Run | None:
    if not config.wandb.enabled:
        return None

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        tags=config.wandb.tags,
        config=_build_wandb_config(config, artifact),
        name=_build_run_name(config, artifact),
        job_type="train",
    )

    dataset_summary = artifact["summary"]
    wandb.log(
        {
            "dataset/num_train": dataset_summary["num_train"],
            "dataset/num_val": dataset_summary["num_val"],
            "dataset/num_test": dataset_summary["num_test"],
            "dataset/num_examples": dataset_summary["num_examples"],
        },
        step=0,
    )
    return run



def _make_metric_logger(run: wandb.sdk.wandb_run.Run | None) -> Any:
    if run is None:
        return None

    def _log(row: dict[str, float]) -> None:
        wandb.log(row, step=int(row["epoch"]))

    return _log



def _class_names(artifact: dict[str, object]) -> list[str]:
    subject_to_class = artifact["subject_to_class"]
    return [subject for subject, _ in sorted(subject_to_class.items(), key=lambda item: item[1])]



def _log_regression_plot(test_result: EvalResult) -> None:
    targets = test_result.targets.numpy().reshape(-1)
    predictions = test_result.outputs.numpy().reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, predictions, alpha=0.75, s=18)
    diag_min = float(min(targets.min(), predictions.min()))
    diag_max = float(max(targets.max(), predictions.max()))
    ax.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", linewidth=1)
    ax.set_xlabel("True Speed (m/s)")
    ax.set_ylabel("Predicted Speed (m/s)")
    ax.set_title("Test Predictions vs Targets")
    ax.grid(alpha=0.25)
    wandb.log({"plots/test_prediction_scatter": wandb.Image(fig)})
    plt.close(fig)



def _log_classification_plot(test_result: EvalResult, class_names: list[str]) -> None:
    predictions = torch.argmax(test_result.outputs, dim=1).numpy()
    targets = test_result.targets.numpy()
    n_classes = len(class_names)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_class, pred_class in zip(targets, predictions):
        matrix[int(true_class), int(pred_class)] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(n_classes), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    wandb.log(
        {
            "plots/test_confusion_matrix": wandb.Image(fig),
            "plots/test_confusion_matrix_interactive": wandb.plot.confusion_matrix(
                probs=None,
                y_true=targets.tolist(),
                preds=predictions.tolist(),
                class_names=class_names,
            ),
        }
    )
    plt.close(fig)



def _log_test_visuals(config: ExperimentConfig, artifact: dict[str, object], test_result: EvalResult) -> None:
    if config.training.task == "regression":
        _log_regression_plot(test_result)
    else:
        _log_classification_plot(test_result, _class_names(artifact))



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple raw-window CNN baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    artifact = torch.load(config.artifact_path, map_location="cpu", weights_only=False)
    run = _init_wandb(config, artifact)
    metric_logger = _make_metric_logger(run)

    try:
        model, history, test_result = fit(
            artifact=artifact,
            task=config.training.task,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            num_workers=config.training.num_workers,
            seed=config.seed,
            metric_logger=metric_logger,
            run_logger=(run if config.wandb.enabled and config.wandb.watch else None),
        )

        config.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = config.output_dir / f"{config.training.task}_baseline.pt"
        history_path = config.output_dir / f"{config.training.task}_history.csv"
        summary_path = config.output_dir / f"{config.training.task}_summary.json"

        torch.save(model.state_dict(), model_path)
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        best_row = min(history, key=lambda row: row["val_score"]) if config.training.task == "regression" else max(history, key=lambda row: row["val_score"])
        summary = {
            "task": config.training.task,
            "model_path": str(model_path),
            "history_path": str(history_path),
            "best_epoch": int(best_row["epoch"]),
            "best_val_loss": float(best_row["val_loss"]),
            "best_val_score": float(best_row["val_score"]),
            "test_loss": float(test_result.loss),
            "test_score": float(test_result.score),
        }
        summary_path.write_text(json.dumps(summary, indent=2))

        if run is not None:
            wandb.log(
                {
                    "best_epoch": summary["best_epoch"],
                    "best_val_loss": summary["best_val_loss"],
                    "best_val_score": summary["best_val_score"],
                    "test_loss": summary["test_loss"],
                    "test_score": summary["test_score"],
                },
                step=int(summary["best_epoch"]),
            )
            _log_test_visuals(config, artifact, test_result)
            run.summary.update(summary)

        print(json.dumps(summary, indent=2))
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()

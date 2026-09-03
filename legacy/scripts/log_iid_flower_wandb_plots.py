from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/scripts/log_iid_flower_wandb_plots.py"
raise ImportError(
    "Quarantined legacy module: legacy/scripts/log_iid_flower_wandb_plots.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import wandb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.config import load_config
from redo_by_sara.federated import build_loader, create_federated_model


def _read_history(path: Path) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float | None] = {}
            for key, value in row.items():
                parsed[key] = None if value == "" else float(value)
            rows.append(parsed)
    return rows


def _read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _plot_metric(
    rows: list[dict[str, float | None]],
    test_value: float,
    task: str,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    rounds = [int(row["round"]) for row in rows if row["round"] is not None]
    train_key = f"train_{metric}"
    val_key = f"val_{metric}"

    fig, ax = plt.subplots(figsize=(8, 5))
    train_points = [
        (int(row["round"]), row[train_key])
        for row in rows
        if row.get(train_key) is not None and row.get("round") is not None
    ]
    val_points = [
        (int(row["round"]), row[val_key])
        for row in rows
        if row.get(val_key) is not None and row.get("round") is not None
    ]

    if train_points:
        ax.plot(
            [point[0] for point in train_points],
            [float(point[1]) for point in train_points],
            marker="o",
            label="Train",
        )
    if val_points:
        ax.plot(
            [point[0] for point in val_points],
            [float(point[1]) for point in val_points],
            marker="o",
            label="Validation",
        )
    if rounds:
        ax.hlines(
            y=test_value,
            xmin=min(rounds),
            xmax=max(rounds),
            linestyles="dashed",
            colors="black",
            label="Test",
        )

    ax.set_title(f"{task.title()} IID Flower {ylabel}")
    ax.set_xlabel("Federated round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _format_count(value: int, total: int) -> str:
    if total == 0:
        return str(value)
    return f"{value}\n{value / total:.0%}"


def _plot_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted subject")
    ax.set_ylabel("True subject")
    ax.set_title("Classification IID Flower Test Confusion Matrix")

    for row_idx, row in enumerate(matrix):
        row_total = sum(row)
        for col_idx, value in enumerate(row):
            color = "white" if value > max(max(row) for row in matrix) / 2 else "black"
            ax.text(
                col_idx,
                row_idx,
                _format_count(value, row_total),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _evaluate_classification_predictions(
    artifact_path: Path,
    model_path: Path,
    config_path: Path,
) -> tuple[list[int], list[int], list[str]]:
    config = load_config(config_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    model = create_federated_model(artifact, "classification")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = build_loader(
        artifact=artifact,
        indices=artifact["test_indices"],
        task="classification",
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        seed=config.seed,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(int(item) for item in predictions)
            y_true.extend(int(item) for item in y.tolist())

    class_to_subject = {
        int(class_id): str(subject_id)
        for subject_id, class_id in artifact["subject_to_class"].items()
    }
    class_names = [class_to_subject[index] for index in sorted(class_to_subject)]
    return y_true, y_pred, class_names


def _confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_id, pred_id in zip(y_true, y_pred, strict=True):
        matrix[int(true_id)][int(pred_id)] += 1
    return matrix


def _log_task_plots(
    task: str,
    history_path: Path,
    summary_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    rows = _read_history(history_path)
    summary = _read_summary(summary_path)

    loss_path = output_dir / f"{task}_iid_loss_train_val_test.png"
    score_path = output_dir / f"{task}_iid_score_train_val_test.png"

    score_label = "RMSE" if task == "regression" else "Accuracy"
    _plot_metric(rows, float(summary["test_loss"]), task, "loss", "Loss", loss_path)
    _plot_metric(rows, float(summary["test_score"]), task, "score", score_label, score_path)

    table = wandb.Table(columns=["round", "train_loss", "val_loss", "test_loss", "train_score", "val_score", "test_score"])
    for row in rows:
        table.add_data(
            row.get("round"),
            row.get("train_loss"),
            row.get("val_loss"),
            float(summary["test_loss"]),
            row.get("train_score"),
            row.get("val_score"),
            float(summary["test_score"]),
        )
    wandb.log(
        {
            f"{task}/metrics_table": table,
            f"{task}/loss_train_val_test": wandb.Image(str(loss_path)),
            f"{task}/score_train_val_test": wandb.Image(str(score_path)),
        }
    )
    return {"loss": loss_path, "score": score_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Log IID Flower result plots to W&B.")
    parser.add_argument("--artifact", default="artifacts/raw_windows.pt")
    parser.add_argument("--output-dir", default="artifacts/wandb_iid_plots")
    parser.add_argument("--project", default="ENGR859-final-project")
    parser.add_argument("--run-name", default="iid-flower-result-plots")
    parser.add_argument("--classification-config", default="configs/fl_iid_classification.yaml")
    parser.add_argument("--regression-config", default="configs/fl_iid_regression.yaml")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run = wandb.init(
        project=args.project,
        name=args.run_name,
        job_type="plot-results",
        tags=["flower", "iid", "plots", "confusion-matrix"],
    )

    try:
        plot_paths = {}
        plot_paths["regression"] = _log_task_plots(
            task="regression",
            history_path=Path("artifacts/regression_iid_federated_history.csv"),
            summary_path=Path("artifacts/regression_iid_federated_summary.json"),
            output_dir=output_dir,
        )
        plot_paths["classification"] = _log_task_plots(
            task="classification",
            history_path=Path("artifacts/classification_iid_federated_history.csv"),
            summary_path=Path("artifacts/classification_iid_federated_summary.json"),
            output_dir=output_dir,
        )

        y_true, y_pred, class_names = _evaluate_classification_predictions(
            artifact_path=Path(args.artifact),
            model_path=Path("artifacts/classification_iid_flower_model.pt"),
            config_path=Path(args.classification_config),
        )
        matrix = _confusion_matrix(y_true, y_pred, len(class_names))
        confusion_path = output_dir / "classification_iid_test_confusion_matrix.png"
        _plot_confusion_matrix(matrix, class_names, confusion_path)

        wandb.log(
            {
                "classification/test_confusion_matrix": wandb.Image(str(confusion_path)),
                "classification/test_confusion_matrix_wandb": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names,
                ),
            }
        )

        summary_path = output_dir / "plot_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "regression_loss_plot": str(plot_paths["regression"]["loss"]),
                    "regression_score_plot": str(plot_paths["regression"]["score"]),
                    "classification_loss_plot": str(plot_paths["classification"]["loss"]),
                    "classification_score_plot": str(plot_paths["classification"]["score"]),
                    "classification_confusion_matrix": str(confusion_path),
                    "wandb_run_name": args.run_name,
                },
                indent=2,
            )
        )
        print(summary_path.read_text())
    finally:
        run.finish()


if __name__ == "__main__":
    main()

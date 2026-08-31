from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "artifacts/combined_iid_flower_3clients_60r"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"History is empty: {path}")
    return rows


def _plot_classification(history_path: Path, output_path: Path) -> None:
    rows = _read_rows(history_path)
    rounds = np.asarray([int(row["round"]) for row in rows])
    loss = np.asarray([float(row["global_train_cross_entropy"]) for row in rows])
    accuracy = 100.0 * np.asarray(
        [float(row["global_train_accuracy"]) for row in rows]
    )

    figure, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(rounds, loss, color="#c43c39", linewidth=2)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_title("Flower IID Classification: Global Training Metrics")
    axes[1].plot(rounds, accuracy, color="#2b6cb0", linewidth=2)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Federated-learning round (0 = initial global model)")
    axes[1].set_ylim(0.0, 101.0)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_regression(history_path: Path, output_path: Path) -> None:
    rows = _read_rows(history_path)
    rounds = np.asarray([int(row["round"]) for row in rows])
    loss = np.asarray([float(row["global_train_standardized_mse"]) for row in rows])
    tolerance_accuracy = np.asarray(
        [float(row["global_train_within_0_10_mps_pct"]) for row in rows]
    )

    figure, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(rounds, loss, color="#c43c39", linewidth=2)
    axes[0].axhline(1.0, color="#718096", linestyle="--", linewidth=1.1)
    axes[0].set_ylabel("Standardized run MSE loss")
    axes[0].set_ylim(0.0, max(1.05, 1.05 * float(loss.max())))
    axes[0].set_title("Flower IID Residual Regression: Global Training Metrics")
    axes[1].plot(rounds, tolerance_accuracy, color="#2b6cb0", linewidth=2)
    axes[1].set_ylabel("Runs within ±0.10 m/s (%)")
    axes[1].set_xlabel("Federated-learning round (0 = source/subject mean)")
    axes[1].set_ylim(0.0, 101.0)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _read_confusion(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    class_names = rows[0][1:]
    if [row[0] for row in rows[1:]] != class_names:
        raise ValueError("Confusion-matrix row and column labels differ.")
    matrix = np.asarray([[int(value) for value in row[1:]] for row in rows[1:]])
    return class_names, matrix


def _display_name(name: str) -> str:
    if name == "no_walking":
        return "No walking"
    return name.replace("subject_", "S").replace("_walking", "")


def _plot_confusion(confusion_path: Path, output_path: Path) -> None:
    class_names, matrix = _read_confusion(confusion_path)
    display_names = [_display_name(name) for name in class_names]
    figure, axis = plt.subplots(figsize=(9, 7.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(display_names)), display_names, rotation=45, ha="right")
    axis.set_yticks(range(len(display_names)), display_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    accuracy = float(np.trace(matrix) / matrix.sum())
    axis.set_title(
        f"Flower IID Classification: Final Test Confusion Matrix\n"
        f"Accuracy = {100.0 * accuracy:.2f}% ({int(np.trace(matrix))}/{int(matrix.sum())})"
    )
    threshold = float(matrix.max()) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=9,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create dedicated FL-round metric plots and the final confusion matrix."
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = results_dir / "requested_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    classification_plot = output_dir / "classification_loss_accuracy_vs_fl_round.png"
    regression_plot = output_dir / "regression_loss_accuracy_vs_fl_round.png"
    confusion_plot = output_dir / "classification_final_test_confusion_matrix.png"
    _plot_classification(
        results_dir / "classification/global_train_history.csv", classification_plot
    )
    _plot_regression(
        results_dir / "regression/global_train_history.csv", regression_plot
    )
    _plot_confusion(
        results_dir / "classification/test_confusion_matrix.csv", confusion_plot
    )
    for path in (classification_plot, regression_plot, confusion_plot):
        print(path)


if __name__ == "__main__":
    main()

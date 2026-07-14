from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.config import load_config
from redo_by_sara.federated import build_loader, create_federated_model
from redo_by_sara.sensor_non_iid import build_sensor_masked_loader


def _class_names(artifact: dict[str, object]) -> list[str]:
    subject_to_class = artifact["subject_to_class"]
    return [subject for subject, _ in sorted(subject_to_class.items(), key=lambda item: item[1])]


def _confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_id, pred_id in zip(y_true, y_pred, strict=True):
        matrix[int(true_id)][int(pred_id)] += 1
    return matrix


def _accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(true_id == pred_id) for true_id, pred_id in zip(y_true, y_pred, strict=True))
    return correct / len(y_true)


def _format_cell(value: int, row_total: int) -> str:
    if row_total == 0:
        return str(value)
    return f"{value}\n{value / row_total:.0%}"


def _plot_matrix(matrix: list[list[int]], class_names: list[str], title: str, output_path: Path) -> None:
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


def _uses_mask_aware_classifier(config: object) -> bool:
    return (
        getattr(config.training, "task", None) == "classification"
        and getattr(config.training, "model_name", "simple_cnn1d") == "mask_aware_simple_cnn1d"
    )


def _evaluate(
    artifact: dict[str, object],
    model_path: Path,
    config_path: Path,
    split: str,
) -> tuple[list[list[int]], float, int]:
    config = load_config(config_path)
    indices_key = f"{split}_indices"
    if indices_key not in artifact:
        raise KeyError(f"Artifact does not contain {indices_key}.")

    mask_as_input = _uses_mask_aware_classifier(config)
    model = create_federated_model(artifact, "classification", mask_aware=mask_as_input)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if mask_as_input:
        loader = build_sensor_masked_loader(
            artifact=artifact,
            indices=artifact[indices_key],
            task="classification",
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            shuffle=False,
            seed=config.seed,
            channel_indices=None,
            include_channel_mask=True,
        )
    else:
        loader = build_loader(
            artifact=artifact,
            indices=artifact[indices_key],
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
            outputs = model(x.to(device))
            predictions = torch.argmax(outputs, dim=1).cpu().tolist()
            y_pred.extend(int(item) for item in predictions)
            y_true.extend(int(item) for item in y.tolist())

    return _confusion_matrix(y_true, y_pred, len(_class_names(artifact))), _accuracy(y_true, y_pred), len(y_true)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a classification confusion matrix for a saved model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/raw_windows.pt"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    class_names = _class_names(artifact)
    matrix, accuracy, num_examples = _evaluate(
        artifact=artifact,
        model_path=args.model,
        config_path=Path(args.config),
        split=args.split,
    )

    title = args.title or f"{args.split.title()} Confusion Matrix Accuracy {accuracy:.3f}"
    _plot_matrix(matrix, class_names, title, args.output)

    payload = {
        "output_path": str(args.output),
        "model_path": str(args.model),
        "config_path": str(Path(args.config)),
        "split": args.split,
        "accuracy": accuracy,
        "num_examples": num_examples,
        "class_names": class_names,
        "matrix": matrix,
    }
    json_output = args.json_output or args.output.with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({**payload, "json_output": str(json_output)}, indent=2))


if __name__ == "__main__":
    main()

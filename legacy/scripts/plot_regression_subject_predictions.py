from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/scripts/plot_regression_subject_predictions.py"
raise ImportError(
    "Quarantined legacy module: legacy/scripts/plot_regression_subject_predictions.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import argparse
import csv
import json
import sys
from collections import defaultdict
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
from redo_by_sara.federated import build_loader, create_federated_model, regression_r2_score


def _default_summary(result_name: str | None) -> Path:
    suffix = f"_{result_name}" if result_name else ""
    return Path(f"artifacts/regression_iid{suffix}_federated_summary.json")


def _model_path(summary: dict[str, object], model_kind: str) -> Path:
    if model_kind == "best-val":
        return Path(str(summary["best_model_path"]))
    return Path(str(summary["model_path"]))


def _split_indices(artifact: dict[str, object], split: str) -> list[int]:
    indices = artifact[f"{split}_indices"]
    if isinstance(indices, torch.Tensor):
        return [int(index) for index in indices.tolist()]
    return [int(index) for index in indices]


def _evaluate_predictions(
    artifact: dict[str, object],
    config_path: Path,
    model_path: Path,
    split: str,
) -> tuple[list[int], list[float], list[float], float]:
    config = load_config(config_path)
    model = create_federated_model(artifact, "regression")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    split_indices = _split_indices(artifact, split)
    loader = build_loader(
        artifact=artifact,
        indices=split_indices,
        task="regression",
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        seed=config.seed,
    )

    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x.to(device)).detach().cpu().reshape(-1)
            predictions.extend(float(item) for item in outputs.tolist())
            targets.extend(float(item) for item in y.reshape(-1).tolist())

    outputs_tensor = torch.tensor(predictions)
    targets_tensor = torch.tensor(targets)
    return split_indices, targets, predictions, regression_r2_score(outputs_tensor, targets_tensor)


def _subject_rows(
    artifact: dict[str, object],
    split_indices: list[int],
    targets: list[float],
    predictions: list[float],
) -> list[dict[str, float | str | int]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"true": [], "pred": []})
    metadata = artifact["metadata"]
    for artifact_index, target, prediction in zip(split_indices, targets, predictions, strict=True):
        subject_id = str(metadata[artifact_index]["subject_id"])
        grouped[subject_id]["true"].append(float(target))
        grouped[subject_id]["pred"].append(float(prediction))

    rows: list[dict[str, float | str | int]] = []
    for subject_id in sorted(grouped):
        true_values = grouped[subject_id]["true"]
        pred_values = grouped[subject_id]["pred"]
        true_avg = sum(true_values) / len(true_values)
        pred_avg = sum(pred_values) / len(pred_values)
        rows.append(
            {
                "subject_id": subject_id,
                "num_windows": len(true_values),
                "avg_true_speed": true_avg,
                "avg_predicted_speed": pred_avg,
                "avg_error": pred_avg - true_avg,
                "avg_abs_error": abs(pred_avg - true_avg),
            }
        )
    return rows


def _save_rows(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_subject_averages(
    rows: list[dict[str, float | str | int]],
    split: str,
    model_kind: str,
    r2: float,
    output_path: Path,
) -> None:
    true_values = [float(row["avg_true_speed"]) for row in rows]
    pred_values = [float(row["avg_predicted_speed"]) for row in rows]
    subjects = [str(row["subject_id"]) for row in rows]
    lower = min(true_values + pred_values)
    upper = max(true_values + pred_values)
    padding = max((upper - lower) * 0.08, 0.02)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_values, pred_values, s=90)
    ax.plot([lower - padding, upper + padding], [lower - padding, upper + padding], linestyle="--", color="black")
    for subject_id, true_value, pred_value in zip(subjects, true_values, pred_values, strict=True):
        ax.annotate(subject_id, (true_value, pred_value), xytext=(5, 5), textcoords="offset points")

    ax.set_title(f"Regression {split.title()} Subject Average Speeds ({model_kind}, R^2={r2:.3f})")
    ax.set_xlabel("Average true speed (m/s)")
    ax.set_ylabel("Average predicted speed (m/s)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot subject-average true vs predicted speed for regression.")
    parser.add_argument("--config", type=Path, default=Path("configs/fl_iid_regression_r2_5r_1e.yaml"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/raw_windows.pt"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--result-name", default="r2_5r_1e")
    parser.add_argument("--model-kind", choices=["best-val", "final"], default="best-val")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/regression_subject_predictions"))
    args = parser.parse_args()

    summary_path = args.summary or _default_summary(args.result_name)
    summary = json.loads(summary_path.read_text())
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    model_path = _model_path(summary, args.model_kind)
    split_indices, targets, predictions, r2 = _evaluate_predictions(
        artifact=artifact,
        config_path=args.config,
        model_path=model_path,
        split=args.split,
    )
    rows = _subject_rows(artifact, split_indices, targets, predictions)

    prefix = f"regression_iid_{args.result_name}_{args.model_kind}_{args.split}"
    csv_path = args.output_dir / f"{prefix}_subject_average_predictions.csv"
    plot_path = args.output_dir / f"{prefix}_subject_average_predictions.png"
    _save_rows(csv_path, rows)
    _plot_subject_averages(rows, args.split, args.model_kind, r2, plot_path)

    print(
        json.dumps(
            {
                "plot_path": str(plot_path),
                "csv_path": str(csv_path),
                "split": args.split,
                "model_kind": args.model_kind,
                "r2": r2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

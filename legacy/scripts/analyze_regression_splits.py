from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/scripts/analyze_regression_splits.py"
raise ImportError(
    "Quarantined legacy module: legacy/scripts/analyze_regression_splits.py. It contradicts CLAUDE.md (see "
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


def _as_indices(artifact: dict[str, object], split: str) -> list[int]:
    indices = artifact[f"{split}_indices"]
    if isinstance(indices, torch.Tensor):
        return [int(index) for index in indices.tolist()]
    return [int(index) for index in indices]


def _rmse(predictions: list[float], targets: list[float]) -> float:
    pred = torch.tensor(predictions, dtype=torch.float32)
    true = torch.tensor(targets, dtype=torch.float32)
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()


def _r2(predictions: list[float], targets: list[float]) -> float:
    return regression_r2_score(torch.tensor(predictions), torch.tensor(targets))


def _metric_row(name: str, split: str, level: str, predictions: list[float], targets: list[float]) -> dict[str, object]:
    return {
        "model": name,
        "split": split,
        "level": level,
        "num_points": len(targets),
        "rmse": _rmse(predictions, targets),
        "r2": _r2(predictions, targets),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_model_predictions(
    artifact: dict[str, object],
    config_path: Path,
    model_path: Path,
    split: str,
) -> tuple[list[int], list[float], list[float]]:
    config = load_config(config_path)
    indices = _as_indices(artifact, split)
    model = create_federated_model(artifact, "regression")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = build_loader(
        artifact=artifact,
        indices=indices,
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
    return indices, targets, predictions


def _run_rows_for_split(artifact: dict[str, object], split: str) -> list[dict[str, object]]:
    metadata = artifact["metadata"]
    targets = artifact["regression_targets"]
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in _as_indices(artifact, split):
        item = metadata[index]
        grouped[(str(item["subject_id"]), int(item["run_index"]))].append(index)

    rows: list[dict[str, object]] = []
    for (subject_id, run_index), indices in sorted(grouped.items()):
        speeds = [float(targets[index]) for index in indices]
        rows.append(
            {
                "split": split,
                "subject_id": subject_id,
                "run_index": run_index,
                "num_windows": len(indices),
                "true_speed": sum(speeds) / len(speeds),
            }
        )
    return rows


def _run_level_from_window_predictions(
    artifact: dict[str, object],
    indices: list[int],
    targets: list[float],
    predictions: list[float],
    split: str,
) -> list[dict[str, object]]:
    metadata = artifact["metadata"]
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: {"true": [], "pred": []})
    for index, target, prediction in zip(indices, targets, predictions, strict=True):
        item = metadata[index]
        key = (str(item["subject_id"]), int(item["run_index"]))
        grouped[key]["true"].append(float(target))
        grouped[key]["pred"].append(float(prediction))

    rows: list[dict[str, object]] = []
    for (subject_id, run_index), values in sorted(grouped.items()):
        true_speed = sum(values["true"]) / len(values["true"])
        predicted_speed = sum(values["pred"]) / len(values["pred"])
        rows.append(
            {
                "split": split,
                "subject_id": subject_id,
                "run_index": run_index,
                "num_windows": len(values["true"]),
                "true_speed": true_speed,
                "predicted_speed": predicted_speed,
                "error": predicted_speed - true_speed,
                "abs_error": abs(predicted_speed - true_speed),
            }
        )
    return rows


def _plot_speed_distribution(rows: list[dict[str, object]], output_path: Path) -> None:
    subjects = sorted({str(row["subject_id"]) for row in rows})
    split_offsets = {"train": -0.22, "val": 0.0, "test": 0.22}
    split_colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for split, offset in split_offsets.items():
        split_rows = [row for row in rows if row["split"] == split]
        ax.scatter(
            [subjects.index(str(row["subject_id"])) + offset for row in split_rows],
            [float(row["true_speed"]) for row in split_rows],
            label=split,
            color=split_colors[split],
            alpha=0.78,
            s=45,
        )

    ax.set_title("Run-Level True Speed Distribution by Subject and Split")
    ax.set_xlabel("Subject")
    ax.set_ylabel("True speed (m/s)")
    ax.set_xticks(range(len(subjects)), subjects)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_run_predictions(rows: list[dict[str, object]], split: str, r2: float, output_path: Path) -> None:
    true_values = [float(row["true_speed"]) for row in rows if row["split"] == split]
    pred_values = [float(row["predicted_speed"]) for row in rows if row["split"] == split]
    labels = [
        f"{row['subject_id']}-r{row['run_index']}"
        for row in rows
        if row["split"] == split
    ]
    lower = min(true_values + pred_values)
    upper = max(true_values + pred_values)
    padding = max((upper - lower) * 0.08, 0.02)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_values, pred_values, s=70)
    ax.plot([lower - padding, upper + padding], [lower - padding, upper + padding], linestyle="--", color="black")
    for label, true_value, pred_value in zip(labels, true_values, pred_values, strict=True):
        ax.annotate(label, (true_value, pred_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title(f"Regression {split.title()} Run-Level Predictions (R^2={r2:.3f})")
    ax.set_xlabel("Run true speed (m/s)")
    ax.set_ylabel("Run predicted speed (m/s)")
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _default_summary(result_name: str) -> Path:
    return Path(f"artifacts/regression_iid_{result_name}_federated_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze regression splits, baseline, and run-level predictions.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/raw_windows.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/fl_iid_regression_r2_5r_1e.yaml"))
    parser.add_argument("--result-name", default="r2_5r_1e")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--model-kind", choices=["best-val", "final"], default="best-val")
    parser.add_argument("--subject", default="006")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/regression_diagnostics"))
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    summary_path = args.summary or _default_summary(args.result_name)
    summary = json.loads(summary_path.read_text())
    model_path = Path(str(summary["best_model_path" if args.model_kind == "best-val" else "model_path"]))

    run_rows = []
    for split in ("train", "val", "test"):
        run_rows.extend(_run_rows_for_split(artifact, split))

    run_distribution_csv = args.output_dir / f"regression_{args.result_name}_run_speed_distribution.csv"
    _write_csv(run_distribution_csv, run_rows)
    speed_plot = args.output_dir / f"regression_{args.result_name}_run_speed_distribution.png"
    _plot_speed_distribution(run_rows, speed_plot)

    subject_rows = [row for row in run_rows if str(row["subject_id"]) == args.subject]
    subject_csv = args.output_dir / f"regression_{args.result_name}_subject_{args.subject}_speeds.csv"
    _write_csv(subject_csv, subject_rows)

    train_targets = [
        float(artifact["regression_targets"][index])
        for index in _as_indices(artifact, "train")
    ]
    train_mean_speed = sum(train_targets) / len(train_targets)

    metric_rows: list[dict[str, object]] = []
    all_run_prediction_rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        indices, targets, predictions = _load_model_predictions(artifact, args.config, model_path, split)
        baseline_predictions = [train_mean_speed for _ in targets]
        metric_rows.append(_metric_row("model", split, "window", predictions, targets))
        metric_rows.append(_metric_row("train_mean_baseline", split, "window", baseline_predictions, targets))

        run_prediction_rows = _run_level_from_window_predictions(artifact, indices, targets, predictions, split)
        baseline_run_rows = [
            {**row, "predicted_speed": train_mean_speed, "error": train_mean_speed - float(row["true_speed"]), "abs_error": abs(train_mean_speed - float(row["true_speed"]))}
            for row in run_prediction_rows
        ]
        all_run_prediction_rows.extend(run_prediction_rows)

        run_targets = [float(row["true_speed"]) for row in run_prediction_rows]
        run_predictions = [float(row["predicted_speed"]) for row in run_prediction_rows]
        baseline_run_predictions = [float(row["predicted_speed"]) for row in baseline_run_rows]
        metric_rows.append(_metric_row("model", split, "run", run_predictions, run_targets))
        metric_rows.append(_metric_row("train_mean_baseline", split, "run", baseline_run_predictions, run_targets))

    metrics_csv = args.output_dir / f"regression_{args.result_name}_baseline_model_metrics.csv"
    _write_csv(metrics_csv, metric_rows)
    run_predictions_csv = args.output_dir / f"regression_{args.result_name}_{args.model_kind}_run_predictions.csv"
    _write_csv(run_predictions_csv, all_run_prediction_rows)

    test_run_targets = [
        float(row["true_speed"])
        for row in all_run_prediction_rows
        if row["split"] == "test"
    ]
    test_run_predictions = [
        float(row["predicted_speed"])
        for row in all_run_prediction_rows
        if row["split"] == "test"
    ]
    test_run_r2 = _r2(test_run_predictions, test_run_targets)
    run_prediction_plot = args.output_dir / f"regression_{args.result_name}_{args.model_kind}_test_run_predictions.png"
    _plot_run_predictions(all_run_prediction_rows, "test", test_run_r2, run_prediction_plot)

    summary_output = {
        "train_mean_speed": train_mean_speed,
        "speed_distribution_plot": str(speed_plot),
        "run_distribution_csv": str(run_distribution_csv),
        "subject_speed_csv": str(subject_csv),
        "metrics_csv": str(metrics_csv),
        "run_predictions_csv": str(run_predictions_csv),
        "test_run_prediction_plot": str(run_prediction_plot),
        "subject": args.subject,
    }
    summary_json = args.output_dir / f"regression_{args.result_name}_diagnostic_summary.json"
    summary_json.write_text(json.dumps(summary_output, indent=2))
    print(json.dumps(summary_output, indent=2))


if __name__ == "__main__":
    main()

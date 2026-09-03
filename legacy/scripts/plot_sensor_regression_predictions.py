from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/scripts/plot_sensor_regression_predictions.py"
raise ImportError(
    "Quarantined legacy module: legacy/scripts/plot_sensor_regression_predictions.py. It contradicts CLAUDE.md (see "
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
from redo_by_sara.federated import create_federated_model, regression_r2_score, set_parameters
from redo_by_sara.sensor_non_iid import build_sensor_masked_loader


def _as_indices(artifact: dict[str, object], split: str) -> list[int]:
    indices = artifact[f"{split}_indices"]
    if isinstance(indices, torch.Tensor):
        return [int(index) for index in indices.tolist()]
    return [int(index) for index in indices]


def _default_paths(result_name: str) -> tuple[Path, Path, Path]:
    stem = f"regression_sensor_non_iid_{result_name}"
    return (
        Path("artifacts/raw_windows_sensor_non_iid_4c_3ch_no006.pt"),
        Path(f"artifacts/{stem}_federated_summary.json"),
        Path(f"artifacts/{stem}_flower_model.pt"),
    )


def _load_predictions(
    artifact: dict[str, object],
    config_path: Path,
    model_path: Path,
    split: str,
) -> tuple[list[dict[str, object]], float, float]:
    config = load_config(config_path)
    model = create_federated_model(artifact, "regression")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    split_indices = _as_indices(artifact, split)
    loader = build_sensor_masked_loader(
        artifact=artifact,
        indices=split_indices,
        task="regression",
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        seed=config.seed,
        channel_indices=None,
    )

    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x.to(device)).detach().cpu().reshape(-1)
            predictions.extend(float(item) for item in outputs.tolist())
            targets.extend(float(item) for item in y.reshape(-1).tolist())

    metadata = artifact["metadata"]
    rows: list[dict[str, object]] = []
    for artifact_index, target, prediction in zip(split_indices, targets, predictions, strict=True):
        item = metadata[artifact_index]
        rows.append(
            {
                "artifact_index": artifact_index,
                "subject_id": str(item["subject_id"]),
                "run_index": int(item["run_index"]),
                "start_time": float(item["start_time"]),
                "end_time": float(item["end_time"]),
                "true_speed": float(target),
                "predicted_speed": float(prediction),
                "error": float(prediction - target),
                "abs_error": float(abs(prediction - target)),
            }
        )

    pred_tensor = torch.tensor(predictions, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    rmse = torch.sqrt(torch.mean((pred_tensor - target_tensor) ** 2)).item()
    r2 = regression_r2_score(pred_tensor, target_tensor)
    return rows, rmse, r2


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _group_by_subject(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    for subject_rows in grouped.values():
        subject_rows.sort(key=lambda row: (int(row["run_index"]), float(row["start_time"])))
    return dict(sorted(grouped.items()))


def _plot_subject_traces(
    rows: list[dict[str, object]],
    split: str,
    rmse: float,
    r2: float,
    output_path: Path,
) -> None:
    grouped = _group_by_subject(rows)
    fig, axes = plt.subplots(len(grouped), 1, figsize=(10, max(2.2 * len(grouped), 4)), sharex=False)
    if len(grouped) == 1:
        axes = [axes]

    for ax, (subject_id, subject_rows) in zip(axes, grouped.items(), strict=True):
        x_values = list(range(1, len(subject_rows) + 1))
        true_values = [float(row["true_speed"]) for row in subject_rows]
        pred_values = [float(row["predicted_speed"]) for row in subject_rows]
        ax.plot(x_values, true_values, marker="o", linewidth=1.6, label="Actual")
        ax.plot(x_values, pred_values, marker="s", linewidth=1.4, label="Predicted")
        ax.set_title(f"Subject {subject_id}")
        ax.set_ylabel("Velocity (m/s)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel(f"{split.title()} window order within subject")
    fig.suptitle(f"Regression {split.title()} Velocity Predictions (RMSE={rmse:.3f}, R^2={r2:.3f})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_subject_scatter(
    rows: list[dict[str, object]],
    split: str,
    rmse: float,
    r2: float,
    output_path: Path,
) -> None:
    grouped = _group_by_subject(rows)
    subjects = list(grouped)
    n_cols = min(3, len(subjects))
    n_rows = (len(subjects) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    all_true = [float(row["true_speed"]) for row in rows]
    all_pred = [float(row["predicted_speed"]) for row in rows]
    lower = min(all_true + all_pred)
    upper = max(all_true + all_pred)
    padding = max((upper - lower) * 0.08, 0.02)
    limits = (lower - padding, upper + padding)

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, subject_id in zip(axes.ravel(), subjects, strict=False):
        ax.set_visible(True)
        subject_rows = grouped[subject_id]
        true_values = [float(row["true_speed"]) for row in subject_rows]
        pred_values = [float(row["predicted_speed"]) for row in subject_rows]
        ax.scatter(true_values, pred_values, s=45, alpha=0.82)
        ax.plot(limits, limits, linestyle="--", color="black", linewidth=1)
        ax.set_title(f"Subject {subject_id}")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_xlabel("Actual velocity (m/s)")
        ax.set_ylabel("Predicted velocity (m/s)")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Regression {split.title()} Actual vs Predicted Velocity (RMSE={rmse:.3f}, R^2={r2:.3f})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sensor non-IID regression actual vs predicted velocity.")
    parser.add_argument("--config", type=Path, default=Path("configs/fl_sensor_non_iid_regression_4c_3ch_no006.yaml"))
    parser.add_argument("--result-name", default="4c_3ch_no006_15r_3e")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/regression_subject_predictions"))
    args = parser.parse_args()

    default_artifact, default_summary, default_model = _default_paths(args.result_name)
    artifact_path = args.artifact or default_artifact
    summary_path = args.summary or default_summary
    summary = json.loads(summary_path.read_text())
    model_path = args.model or Path(str(summary.get("model_path", default_model)))

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    rows, rmse, r2 = _load_predictions(
        artifact=artifact,
        config_path=args.config,
        model_path=model_path,
        split=args.split,
    )

    prefix = f"regression_sensor_non_iid_{args.result_name}_{args.split}"
    csv_path = args.output_dir / f"{prefix}_window_predictions.csv"
    trace_path = args.output_dir / f"{prefix}_subject_velocity_traces.png"
    scatter_path = args.output_dir / f"{prefix}_subject_actual_vs_predicted.png"
    _write_csv(csv_path, rows)
    _plot_subject_traces(rows, args.split, rmse, r2, trace_path)
    _plot_subject_scatter(rows, args.split, rmse, r2, scatter_path)

    print(
        json.dumps(
            {
                "csv_path": str(csv_path),
                "trace_plot_path": str(trace_path),
                "scatter_plot_path": str(scatter_path),
                "split": args.split,
                "num_windows": len(rows),
                "rmse": rmse,
                "r2": r2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

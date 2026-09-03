from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/scripts/log_fl_comparison_wandb.py"
raise ImportError(
    "Quarantined legacy module: legacy/scripts/log_fl_comparison_wandb.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from redo_by_sara.federated import build_loader, create_federated_model, evaluate_model


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    config_path: Path
    history_path: Path
    summary_path: Path
    color: str
    linestyle: str = "-"


def _read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text())


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _round_from_checkpoint(path: Path) -> int | None:
    match = re.search(r"_round_(\d+)\.pt$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _existing_checkpoint_dirs(summary: dict[str, Any], history_path: Path) -> list[Path]:
    dirs: list[Path] = []
    raw_checkpoint_dir = summary.get("checkpoint_dir")
    if raw_checkpoint_dir:
        checkpoint_dir = Path(str(raw_checkpoint_dir))
        dirs.append(checkpoint_dir)
        dirs.append(ROOT / checkpoint_dir)
        dirs.append(ROOT / "artifacts" / "round_checkpoints" / checkpoint_dir.name)

    stem = history_path.name.replace("_federated_history.csv", "")
    dirs.append(ROOT / "artifacts" / "round_checkpoints" / stem)
    dirs.append(ROOT / "artifacts" / "round_checkpoints")

    seen: set[Path] = set()
    existing: list[Path] = []
    for item in dirs:
        resolved = item if item.is_absolute() else ROOT / item
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            existing.append(resolved)
    return existing


def _checkpoint_map(summary: dict[str, Any], history_path: Path) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    stem = history_path.name.replace("_federated_history.csv", "")
    for checkpoint_dir in _existing_checkpoint_dirs(summary, history_path):
        for checkpoint_path in checkpoint_dir.glob(f"{stem}_round_*.pt"):
            round_id = _round_from_checkpoint(checkpoint_path)
            if round_id is not None:
                checkpoints[round_id] = checkpoint_path
    return checkpoints


def _evaluate_checkpoint(
    artifact: dict[str, Any],
    checkpoint_path: Path,
    config_path: Path,
    split: str,
) -> float:
    config = load_config(config_path)
    model = create_federated_model(artifact, config.training.task)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = build_loader(
        artifact=artifact,
        indices=artifact[f"{split}_indices"],
        task=config.training.task,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        seed=config.seed,
    )
    result = evaluate_model(model=model, loader=loader, task=config.training.task, device=device)
    return float(result.score)


def _series_from_history(
    spec: ExperimentSpec,
    artifact: dict[str, Any],
    split: str,
    fallback_to_val: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_history(spec.history_path)
    summary = _read_summary(spec.summary_path)
    checkpoints = _checkpoint_map(summary, spec.history_path)
    points: list[dict[str, Any]] = []

    for row in rows:
        round_value = _float(row, "round")
        if round_value is None:
            continue
        round_id = int(round_value)
        checkpoint_path = checkpoints.get(round_id)
        if split == "test" and checkpoint_path is not None:
            score = _evaluate_checkpoint(artifact, checkpoint_path, spec.config_path, "test")
            source = "test_checkpoint"
        elif split == "val":
            score = _float(row, "val_score")
            source = "validation_history"
        elif fallback_to_val:
            score = _float(row, "val_score")
            source = "validation_fallback"
        else:
            score = None
            source = "missing_checkpoint"

        if score is None:
            continue
        points.append(
            {
                "experiment": spec.label,
                "round": round_id,
                "accuracy": float(score),
                "source": source,
                "checkpoint": "" if checkpoint_path is None else str(checkpoint_path),
            }
        )

    return points, summary


def _plot_comparison(
    series: dict[str, list[dict[str, Any]]],
    specs: list[ExperimentSpec],
    summaries: dict[str, dict[str, Any]],
    output_path: Path,
    title: str,
    split: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for spec in specs:
        points = series.get(spec.label, [])
        if not points:
            continue
        has_fallback = any(point["source"] == "validation_fallback" for point in points)
        label = f"{spec.label} (val fallback)" if split == "test" and has_fallback else spec.label
        ax.plot(
            [point["round"] for point in points],
            [point["accuracy"] for point in points],
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=1.4,
            label=label,
        )

        summary = summaries.get(spec.label, {})
        if split == "test" and "test_score" in summary:
            test_round = int(summary.get("test_round", summary.get("final_round", points[-1]["round"])))
            ax.scatter(
                [test_round],
                [float(summary["test_score"])],
                color=spec.color,
                marker="x",
                s=55,
                linewidths=1.5,
                zorder=4,
            )

    ax.set_title(title)
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("Test accuracy" if split == "test" else "Validation accuracy")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_xlim(left=0)

    all_scores = [point["accuracy"] for points in series.values() for point in points]
    if all_scores:
        lower = max(0.0, min(all_scores) - 0.04)
        upper = min(1.0, max(all_scores) + 0.04)
        if upper - lower < 0.1:
            lower = max(0.0, upper - 0.1)
        ax.set_ylim(lower, upper)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _default_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            label="IID, B=3 E=3",
            config_path=Path("configs/fl_iid_classification_30r_3e.yaml"),
            history_path=Path("artifacts/classification_iid_30r_3e_federated_history.csv"),
            summary_path=Path("artifacts/classification_iid_30r_3e_federated_summary.json"),
            color="tab:blue",
        ),
        ExperimentSpec(
            label="Non-IID, B=2 E=3",
            config_path=Path("configs/fl_semi_non_iid_classification_15r_3e.yaml"),
            history_path=Path("artifacts/classification_semi_non_iid_15r_3e_federated_history.csv"),
            summary_path=Path("artifacts/classification_semi_non_iid_15r_3e_federated_summary.json"),
            color="tab:orange",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a paper-style IID vs non-IID FL accuracy plot and log it to W&B."
    )
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/raw_windows.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/wandb_fl_comparison"))
    parser.add_argument("--project", default="ENGR859-final-project")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", default="fl-iid-noniid-comparison")
    parser.add_argument("--title", default="(a) Floor vibration classification")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument(
        "--no-val-fallback",
        action="store_true",
        help="For --split test, skip runs that do not have per-round checkpoints instead of plotting val_score.",
    )
    parser.add_argument("--offline", action="store_true", help="Use W&B offline mode.")
    args = parser.parse_args()

    if args.offline:
        wandb_mode = "offline"
    else:
        wandb_mode = None

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    specs = _default_specs()
    series: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []

    for spec in specs:
        points, summary = _series_from_history(
            spec=spec,
            artifact=artifact,
            split=args.split,
            fallback_to_val=not args.no_val_fallback,
        )
        series[spec.label] = points
        summaries[spec.label] = summary
        table_rows.extend(points)

    output_path = args.output_dir / f"classification_iid_vs_non_iid_{args.split}_accuracy.png"
    _plot_comparison(
        series=series,
        specs=specs,
        summaries=summaries,
        output_path=output_path,
        title=args.title,
        split=args.split,
    )

    table = wandb.Table(columns=["experiment", "round", "accuracy", "source", "checkpoint"])
    for row in table_rows:
        table.add_data(row["experiment"], row["round"], row["accuracy"], row["source"], row["checkpoint"])

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        job_type="plot-results",
        tags=["flower", "federated-learning", "iid", "non-iid", "comparison", args.split],
        mode=wandb_mode,
    )
    try:
        wandb.log(
            {
                "classification/iid_vs_non_iid_accuracy": wandb.Image(str(output_path)),
                "classification/iid_vs_non_iid_accuracy_table": table,
            }
        )
        run.summary.update(
            {
                "plot_path": str(output_path),
                "split": args.split,
                "num_points": len(table_rows),
                "used_validation_fallback": any(
                    row["source"] == "validation_fallback" for row in table_rows
                ),
            }
        )
    finally:
        run.finish()

    summary_path = args.output_dir / "classification_iid_vs_non_iid_plot_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "plot_path": str(output_path),
                "split": args.split,
                "num_points": len(table_rows),
                "used_validation_fallback": any(
                    row["source"] == "validation_fallback" for row in table_rows
                ),
            },
            indent=2,
        )
    )
    print(summary_path.read_text())


if __name__ == "__main__":
    main()

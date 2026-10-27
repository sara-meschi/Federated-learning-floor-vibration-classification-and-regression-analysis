"""Paper figure and table suite (S6, review plan §4).

Everything here is built from saved results; nothing trains. ``make_all_figures``
reads ``results_dir`` (the repo's ``artifacts/``):

- ``s5/results_index.csv`` and ``s5/derived/*.csv`` (the S5 record), plus each job's
  saved histories, predictions, partitions and local-only summaries;
- ``s3_r1_sweep/sweep_summary.json`` and the sweep runs' saved predictions;
- the canonical centralized folders (manifests, artifact summaries, the 300-epoch run).

Seeds, clients and subjects are discovered from those files, never listed here. Outcome
labels (``trained`` / ``did not train`` / ``collapsed``) are the S5 ones as recorded in
the index and the local-only summaries; for runs the S5 queue never scored (the S3 sweep,
the 300-epoch reference) the S5 detector is re-applied to the saved predictions with
``s5_queue.reapply_detector``.

Style (IEEE two-column): 3.5 in / 7.16 in widths, 8 pt base and 7 pt ticks in a
Times-compatible serif (STIX, bundled with matplotlib), Type 42 fonts, no titles inside
figures, Okabe–Ito-derived colours always paired with a marker or line style, a vector
PDF plus a 300 dpi PNG per figure. Metadata timestamps are stripped so a rebuild is
byte-identical.

Outputs: ``out_dir/figures/fig{1..5}_*.{pdf,png}`` and ``captions.md``;
``out_dir/tables/t{1..5}_*.{tex,csv}``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .guardrails import training_outcome  # noqa: E402
from .regression_metrics import report_from_run_rows  # noqa: E402
from .run_level_classification import wilson_interval  # noqa: E402
from .s5_queue import reapply_detector  # noqa: E402

PathLike = str | Path

# --------------------------------------------------------------------------------------
# layout of the saved results
# --------------------------------------------------------------------------------------

CENTRALIZED_CLASSIFICATION_DIR = "centralized_combined_9ch_no006_nowalking_80_20_60e"
CENTRALIZED_REGRESSION_DIR = "centralized_combined_regression_9ch_no006_walking_80_20_60e"
CENTRALIZED_RESIDUAL_300E_DIR = "centralized_combined_residual_runbalanced_9ch_no006_300e"

TERMINAL_STATUSES = {
    "completed",
    "completed_with_failed_clients",
    "failed_detector",
    "dropped_fallback",
    "reused_canonical",
}

#: Display order and labels of the partition conditions. These are condition *names*
#: (configs), not results; ``load_results`` asserts the index holds no condition missing
#: from this list, so nothing is dropped silently.
CONDITION_LABELS = {
    "k2_iid": "K=2 IID",
    "k2_natural": "K=2 natural",
    "k3_iid": r"K=3 $\rho$=1.0 (IID)",
    "k3_rho0.375": r"K=3 $\rho$=0.375",
    "k3_rho0": r"K=3 $\rho$=0",
    "k3_dirichlet0.1": r"K=3 Dirichlet $\alpha$=0.1",
}
CONDITION_TEXT = {
    "k2_iid": "K=2 IID",
    "k2_natural": "K=2 natural",
    "k3_iid": "K=3 rho=1.0 (IID)",
    "k3_rho0.375": "K=3 rho=0.375",
    "k3_rho0": "K=3 rho=0",
    "k3_dirichlet0.1": "K=3 Dirichlet alpha=0.1",
}
RHO_CONDITIONS = {"k3_iid": 1.0, "k3_rho0.375": 0.375, "k3_rho0": 0.0}

SOURCE_LABELS = {"test_2": "Test_2", "testing_20251124": "20251124"}

# --------------------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------------------

SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.16
PNG_DPI = 300

IEEE_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.3,
    "figure.dpi": 100,
    "savefig.dpi": PNG_DPI,
    "svg.hashsalt": "s6-paper-figures",
}

PDF_METADATA = {"Creator": None, "Producer": None, "CreationDate": None}
PNG_METADATA = {"Software": None}

INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

#: Family encoding: colour + marker + fill. Centralized is ink (a reference, not a series).
FL_COLOR = "#0072B2"
LOCAL_COLOR = "#D55E00"
CENTRAL_COLOR = INK
FAMILY_STYLE = {
    "centralized": {"color": CENTRAL_COLOR, "marker": "D", "fill": True, "label": "Centralized"},
    "fl": {"color": FL_COLOR, "marker": "o", "fill": True, "label": "FL (FedAvg)"},
    "local_only": {"color": LOCAL_COLOR, "marker": "s", "fill": False, "label": "Local-only (per client)"},
}
#: Outcome encoding for regression models that did not train: marker + text label.
OUTCOME_MARKER = {"did not train": "^", "collapsed": "X"}

#: Per-subject categorical palette, assigned to the sorted subject IDs in this order.
#: Okabe–Ito with black replaced by wine (#882255); this order passes the dataviz
#: palette validator (lightness band, chroma floor, adjacent CVD ΔE ≥ 9.6, normal-vision
#: ΔE ≥ 20). Every subject also gets its own marker, so identity is never colour alone.
SUBJECT_COLORS = ("#0072B2", "#D55E00", "#009E73", "#882255", "#E69F00", "#56B4E9", "#CC79A7")
SUBJECT_MARKERS = ("o", "s", "^", "D", "v", "P", "X")

#: Fig 2 regression axis: linear, clipped here; cut-off values get arrowheads and print.
SKILL_CLIP = (-5.0, 1.0)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, format="pdf", metadata=PDF_METADATA)
    fig.savefig(png, format="png", dpi=PNG_DPI, metadata=PNG_METADATA)
    plt.close(fig)
    return [pdf, png]


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=INK)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def _read_csv(path: PathLike) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: PathLike) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _seed_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return int(row["seed"]), str(row.get("client") or "")


@dataclass
class Results:
    """The saved results, loaded once. ``root`` is the ``artifacts/`` folder."""

    root: Path
    index: list[dict[str, str]]
    run_level: list[dict[str, str]]
    seed_aggregates: list[dict[str, str]]

    def rows(self, family: str, condition: str | None = None, task: str | None = None) -> list[dict[str, str]]:
        found = [
            row for row in self.index
            if row["family"] == family
            and (condition is None or row["condition"] == condition)
            and (task is None or row["task"] == task)
            and row["status"] != "dropped_fallback"
        ]
        return sorted(found, key=_seed_sort_key)

    def rebase(self, absolute: str) -> Path:
        """Saved paths are absolute on the machine that ran them; re-root them here."""
        marker = "/artifacts/"
        position = absolute.rfind(marker)
        if position < 0:
            raise ValueError(f"Saved path is not under an artifacts/ folder: {absolute}")
        return self.root / absolute[position + len(marker):]

    def result_dir(self, row: Mapping[str, str]) -> Path:
        return self.rebase(row["result_dir"])

    def job_dir(self, row: Mapping[str, str]) -> Path:
        return self.root / "s5" / row["job_id"]

    def conditions(self, family: str) -> list[str]:
        present = {row["condition"] for row in self.index if row["family"] == family}
        return [c for c in CONDITION_LABELS if c in present]


def load_results(results_dir: PathLike | Results) -> Results:
    if isinstance(results_dir, Results):
        return results_dir
    root = Path(results_dir)
    s5 = root / "s5"
    index = _read_csv(s5 / "results_index.csv")
    bad = [(r["job_id"], r["status"]) for r in index if r["status"] not in TERMINAL_STATUSES]
    if bad:
        raise AssertionError(f"S5 jobs not in a terminal state: {bad}")
    unknown = {
        r["condition"] for r in index if r["family"] in ("fl", "local_only")
    } - set(CONDITION_LABELS)
    if unknown:
        raise AssertionError(f"Conditions in the index without a display label: {sorted(unknown)}")
    return Results(
        root=root,
        index=index,
        run_level=_read_csv(s5 / "derived" / "run_level_classification.csv"),
        seed_aggregates=_read_csv(s5 / "derived" / "seed_aggregates.csv"),
    )


# --------------------------------------------------------------------------------------
# one record per evaluated model
# --------------------------------------------------------------------------------------


def _classification_record(results: Results, row: Mapping[str, str], client: str = "") -> dict[str, Any]:
    matches = [
        r for r in results.run_level
        if r["job_id"] == row["job_id"] and r["client"] == client and r["method"] == "majority_vote"
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one run-level row for {row['job_id']} {client!r}, got {len(matches)}")
    m = matches[0]
    correct, runs = int(m["num_correct"]), int(m["num_runs"])
    low, high = wilson_interval(correct, runs)
    if not (math.isclose(low, float(m["wilson95_low"]), abs_tol=1e-12)
            and math.isclose(high, float(m["wilson95_high"]), abs_tol=1e-12)):
        raise AssertionError(f"Wilson interval mismatch for {row['job_id']} {client!r}")
    return {
        "family": row["family"], "condition": row["condition"], "task": "classification",
        "seed": int(row["seed"]), "client": client, "correct": correct, "runs": runs,
        "accuracy": correct / runs, "ci_low": low, "ci_high": high,
        "accuracy_test_2": _float(m["accuracy_test_2"]),
        "accuracy_testing_20251124": _float(m["accuracy_testing_20251124"]),
        "macro_f1": float(m["macro_f1"]),
        "window_accuracy": _float(row.get("window_accuracy")) if not client else None,
        "flat": row["flat"] == "True",
    }


def classification_records(results: Results, family: str, condition: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in results.rows(family, condition, "classification"):
        if family == "local_only":
            clients = sorted({r["client"] for r in results.run_level if r["job_id"] == row["job_id"] and r["client"]})
            out.extend(_classification_record(results, row, c) for c in clients)
        else:
            out.append(_classification_record(results, row))
    return out


def regression_records(results: Results, family: str, condition: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    task_rows = results.rows(family, condition, "regression")
    for row in task_rows:
        if family == "local_only":
            # The job record's per-client metrics are the S5 re-derived ones (current
            # gates, own-data trained basis) for every client, failed clients included.
            per_client = _read_json(results.job_dir(row) / "job.json")["per_client"]
            outcomes = json.loads(row["client_outcomes"])
            if outcomes != {c: m["training_outcome"] for c, m in per_client.items()}:
                raise AssertionError(f"{row['job_id']}: job.json client outcomes disagree with the index")
            for client, m in sorted(per_client.items()):
                out.append({
                    "family": family, "condition": row["condition"], "task": "regression",
                    "seed": int(row["seed"]), "client": client,
                    "skill": float(m["skill"]), "ci_low": float(m["skill_ci_low"]),
                    "ci_high": float(m["skill_ci_high"]),
                    "rmse": float(m["model_rmse_mps"]), "baseline_rmse": float(m["baseline_rmse_mps"]),
                    "outcome": m["training_outcome"],
                    "gated_mse": float(m["trained_gate_mse"]),
                    "result_dir": results.result_dir(row) / client,
                })
        else:
            out.append({
                "family": family, "condition": row["condition"], "task": "regression",
                "seed": int(row["seed"]), "client": "",
                "skill": float(row["skill"]), "ci_low": float(row["skill_ci_low"]),
                "ci_high": float(row["skill_ci_high"]),
                "rmse": float(row["model_rmse_mps"]), "baseline_rmse": float(row["baseline_rmse_mps"]),
                "outcome": row["training_outcome"],
                "gated_mse": _float(row["trained_gate_mse"]),
                "result_dir": results.result_dir(row),
            })
    return out


def _centralized_regression(results: Results, condition: str) -> list[dict[str, Any]]:
    """``condition`` is ``absolute`` or ``residual`` (the centralized index conditions)."""
    return regression_records(results, "centralized", condition)


def _one_value(values: Sequence[float], what: str, tolerance: float = 1e-12) -> float:
    """A quantity that must be the same everywhere (up to float summation order)."""
    if max(values) - min(values) > tolerance:
        raise AssertionError(f"{what} differs: {sorted(set(values))}")
    return float(values[0])


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample std (ddof=1), as in ``derived/seed_aggregates.csv``."""
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


# --------------------------------------------------------------------------------------
# shared drawing helpers
# --------------------------------------------------------------------------------------


def _family_marker(family: str, outcome: str | None) -> dict[str, Any]:
    style = FAMILY_STYLE[family]
    marker = OUTCOME_MARKER.get(outcome or "", style["marker"])
    face = style["color"] if style["fill"] else "white"
    if outcome in OUTCOME_MARKER:
        face = "white"
    return {
        "marker": marker, "markeredgecolor": style["color"], "markerfacecolor": face,
        "markeredgewidth": 0.9, "markersize": 4.5 if marker != "X" else 5.5,
    }


def _fmt_signed(value: float, digits: int = 1, plus: bool = False) -> str:
    text = f"{value:+.{digits}f}" if plus else f"{value:.{digits}f}"
    return text.replace("-", "−")


def _clipped_interval_h(ax: plt.Axes, y: float, value: float, low: float, high: float,
                        family: str, outcome: str | None, clip: tuple[float, float]) -> None:
    """Horizontal point + whisker on a clipped axis. Cut-off whisker ends get arrowheads;
    a cut-off point is drawn as an arrowhead at the edge; hidden values are printed."""
    color = FAMILY_STYLE[family]["color"]
    lo_edge, hi_edge = clip
    a, b = max(low, lo_edge), min(high, hi_edge)
    if b > a:
        ax.plot([a, b], [y, y], color=color, lw=0.8, solid_capstyle="butt", zorder=2)
    for end, edge, marker in ((low, lo_edge, "<"), (high, hi_edge, ">")):
        if (end < edge) if marker == "<" else (end > edge):
            ax.plot([edge], [y], marker=marker, color=color, markersize=3.2, zorder=3, clip_on=False)
    hidden: list[str] = []
    if value < lo_edge or value > hi_edge:
        edge = lo_edge if value < lo_edge else hi_edge
        ax.plot([edge], [y], marker="<" if value < lo_edge else ">", markersize=5.5,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.9,
                zorder=4, clip_on=False)
        hidden.append(f"{_fmt_signed(value, 2)} [{_fmt_signed(low)}, {_fmt_signed(high, 2)}]")
    else:
        ax.plot([value], [y], linestyle="none", zorder=4, **_family_marker(family, outcome))
        if low < lo_edge:
            hidden.append(f"CI from {_fmt_signed(low)}")
        if high > hi_edge:
            hidden.append(f"CI to {_fmt_signed(high)}")
    if hidden:
        ax.text(lo_edge + 0.08, y - 0.05, "; ".join(hidden), fontsize=5.5, color=MUTED,
                ha="left", va="top", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9})


def _offsets(n: int, half_width: float) -> np.ndarray:
    if n <= 1:
        return np.zeros(max(n, 0))
    return np.linspace(-half_width, half_width, n)


# --------------------------------------------------------------------------------------
# Figure 1: convergence
# --------------------------------------------------------------------------------------


def _history(results: Results, row: Mapping[str, str]) -> list[dict[str, str]]:
    return _read_csv(results.result_dir(row) / "global_train_history.csv")


def _not_trained(row: Mapping[str, str], task: str) -> str | None:
    """The label to show a seed separately, or None if it belongs in the band."""
    if task == "regression":
        return None if row["training_outcome"] == "trained" else row["training_outcome"]
    return "flat" if row["flat"] == "True" else None


def gradient_steps(results: Results) -> dict[str, int]:
    """Step budgets for the Fig 1 caption, read from the saved summaries."""
    def budget(row: Mapping[str, str]) -> dict[str, Any]:
        folder = results.result_dir(row)
        # A run that failed the detector at run time keeps its record under this name.
        name = "training_summary.json" if (folder / "training_summary.json").exists() else "degenerate_model_check_FAILED.json"
        return _read_json(folder / name)["integrity_audit"]["gradient_step_budget"]

    out: dict[str, int] = {}
    for task in ("classification", "regression"):
        for condition in ("k2_iid", "k2_natural"):
            values = {int(budget(r)["measured_total_gradient_steps"]) for r in results.rows("fl", condition, task)}
            if len(values) != 1:
                raise AssertionError(f"FL {condition} {task}: step counts differ across seeds: {values}")
            out[f"fl_{condition}_{task}"] = values.pop()
        reference = budget(results.rows("fl", "k2_iid", task)[0])["centralized_reference"]
        out[f"centralized_60e_{task}"] = int(reference["total_gradient_steps"])
        out[f"centralized_steps_per_epoch_{task}"] = int(reference["steps_per_epoch"])
    epochs_300 = int(_read_json(results.root / CENTRALIZED_RESIDUAL_300E_DIR / "training_summary.json")["epochs"])
    out["centralized_300e_regression"] = epochs_300 * out["centralized_steps_per_epoch_regression"]
    return out


def convergence_data(results: Results) -> dict[str, Any]:
    """Everything Fig 1 draws, as plain numbers."""
    data: dict[str, Any] = {"curves": {}, "references": {}}
    metric = {"classification": "global_train_accuracy", "regression": "global_train_standardized_mse"}
    for task in ("classification", "regression"):
        for condition in ("k2_iid", "k2_natural"):
            seeds = []
            for row in results.rows("fl", condition, task):
                history = _history(results, row)
                seeds.append({
                    "seed": int(row["seed"]),
                    "label": _not_trained(row, task),
                    "round": [int(h["round"]) for h in history],
                    "value": [float(h[metric[task]]) for h in history],
                })
            data["curves"][(task, condition)] = seeds

    central_cls = []
    for row in results.rows("centralized", "classification", "classification"):
        summary = _read_json(results.result_dir(row) / "training_summary.json")
        central_cls.append(float(summary["final_train_accuracy"]))
    counts = _read_json(results.root / CENTRALIZED_CLASSIFICATION_DIR / "artifact_summary.json")["class_window_counts"]
    train_counts = {name: c["train"] for name, c in counts.items()}
    data["references"]["classification"] = {
        "centralized_final_train_accuracy": central_cls,
        "majority_class_train_rate": max(train_counts.values()) / sum(train_counts.values()),
    }
    residual = [r["gated_mse"] for r in _centralized_regression(results, "residual")]
    summary_300 = _read_json(results.root / CENTRALIZED_RESIDUAL_300E_DIR / "training_summary.json")
    data["references"]["regression"] = {
        "lookup": 1.0,
        "centralized_60e_final_standardized_mse": residual,
        "centralized_300e_final_standardized_mse": float(
            summary_300["degenerate_model_check"]["final_standardized_mse"]
        ),
    }
    return data


def plot_convergence(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    data = convergence_data(results)
    styles = {
        "k2_iid": {"color": FL_COLOR, "ls": "-", "marker": "o"},
        "k2_natural": {"color": LOCAL_COLOR, "ls": "--", "marker": "s"},
    }
    with plt.rc_context(IEEE_RC):
        fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COLUMN_IN, 4.3), sharex=True,
                                 layout="constrained")
        for ax, task, letter in zip(axes, ("classification", "regression"), "ab"):
            handles: list[Any] = []
            for condition in ("k2_iid", "k2_natural"):
                st = styles[condition]
                seeds = data["curves"][(task, condition)]
                band = [s for s in seeds if s["label"] is None]
                if band:
                    rounds = np.asarray(band[0]["round"])
                    values = np.asarray([s["value"] for s in band])
                    ax.fill_between(rounds, values.min(axis=0), values.max(axis=0),
                                    color=st["color"], alpha=0.18, linewidth=0, zorder=1)
                    for s in band:
                        ax.plot(s["round"], s["value"], color=st["color"], ls=st["ls"], lw=0.7, zorder=2)
                    handles.append(Line2D([], [], color=st["color"], ls=st["ls"], lw=1.0,
                                          label=f"{CONDITION_LABELS[condition]} ({len(band)} "
                                                f"{'seed' if len(band) == 1 else 'seeds'}"
                                                f"{' trained' if task == 'regression' else ''})"))
                for s in seeds:
                    if s["label"] is None:
                        continue
                    ax.plot(s["round"], s["value"], color=st["color"], ls=":", lw=1.1,
                            marker="X", markevery=6, markersize=4.5, markerfacecolor="white",
                            markeredgewidth=0.8, zorder=4)
                    handles.append(Line2D([], [], color=st["color"], ls=":", lw=1.1, marker="X",
                                          markersize=4.5, markerfacecolor="white",
                                          label=f"{CONDITION_LABELS[condition]}, seed {s['seed']} ({s['label']})"))
            refs = data["references"][task]
            if task == "classification":
                central = refs["centralized_final_train_accuracy"]
                ax.axhline(np.mean(central), color=CENTRAL_COLOR, ls=(0, (5, 2)), lw=0.8, zorder=0)
                ax.axhline(refs["majority_class_train_rate"], color=MUTED, ls=(0, (1, 1.5)), lw=0.8, zorder=0)
                handles.append(Line2D([], [], color=CENTRAL_COLOR, ls=(0, (5, 2)), lw=0.8,
                                      label=f"centralized, final train ({len(central)} seeds)"))
                handles.append(Line2D([], [], color=MUTED, ls=(0, (1, 1.5)), lw=0.8,
                                      label="majority class"))
                ax.set_ylim(0, 1.04)
                ax.set_ylabel("Global-train accuracy")
                ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=6.5)
            else:
                central = refs["centralized_60e_final_standardized_mse"]
                ax.set_yscale("log")
                ax.axhspan(min(central), max(central), color=CENTRAL_COLOR, alpha=0.10, lw=0, zorder=0)
                ax.axhline(np.mean(central), color=CENTRAL_COLOR, ls=(0, (5, 2)), lw=0.8, zorder=0)
                ax.axhline(refs["centralized_300e_final_standardized_mse"], color=CENTRAL_COLOR,
                           ls=(0, (5, 1.5, 1, 1.5)), lw=0.8, zorder=0)
                ax.axhline(refs["lookup"], color=MUTED, ls="-", lw=0.6, zorder=0)
                handles.append(Line2D([], [], color=CENTRAL_COLOR, ls=(0, (5, 2)), lw=0.8,
                                      label=f"centralized, 60 epochs ({len(central)} seeds)"))
                handles.append(Line2D([], [], color=CENTRAL_COLOR, ls=(0, (5, 1.5, 1, 1.5)), lw=0.8,
                                      label="centralized, matched steps"))
                handles.append(Line2D([], [], color=MUTED, ls="-", lw=0.6, label="lookup (= 1)"))
                ax.set_ylim(1e-6, 3.0)
                ax.set_ylabel("Global-train standardized MSE")
                ax.legend(handles=handles, loc="lower left", fontsize=6.0, ncol=1)
            ax.grid(axis="y", color=GRID, lw=0.4, which="major")
            ax.set_axisbelow(True)
            _panel_letter(ax, letter)
        axes[-1].set_xlabel("Communication round")
        axes[-1].set_xlim(0, max(max(s["round"]) for v in data["curves"].values() for s in v))
        return _save(fig, Path(out_dir), "fig1_convergence")


# --------------------------------------------------------------------------------------
# Figure 2: centralized vs FL vs local-only, every condition
# --------------------------------------------------------------------------------------


def comparison_rows(results: Results) -> list[tuple[str, str]]:
    """(row key, label), top to bottom: centralized, then every FL condition present."""
    return [("centralized", "Centralized")] + [
        (c, CONDITION_LABELS[c]) for c in results.conditions("fl")
    ]


def comparison_data(results: Results) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """``{task: {row key: [records]}}``; FL and local-only records share a row."""
    data: dict[str, dict[str, list[dict[str, Any]]]] = {"classification": {}, "regression": {}}
    data["classification"]["centralized"] = classification_records(results, "centralized", "classification")
    data["regression"]["centralized"] = _centralized_regression(results, "residual")
    for condition in results.conditions("fl"):
        data["classification"][condition] = (
            classification_records(results, "fl", condition)
            + classification_records(results, "local_only", condition)
        )
        data["regression"][condition] = (
            regression_records(results, "fl", condition)
            + regression_records(results, "local_only", condition)
        )
    return data


def plot_comparison(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    data = comparison_data(results)
    rows = comparison_rows(results)
    sub_offset = {"centralized": 0.0, "fl": 0.20, "local_only": -0.20}
    with plt.rc_context(IEEE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 3.0), sharey=True,
                                 layout="constrained")
        for ax, task, letter in zip(axes, ("classification", "regression"), "ab"):
            for position, (key, _) in enumerate(rows):
                y0 = len(rows) - 1 - position
                records = data[task][key]
                for family in ("centralized", "fl", "local_only"):
                    members = sorted((r for r in records if r["family"] == family), key=_seed_sort_key)
                    for offset, record in zip(_offsets(len(members), 0.13)[::-1], members):
                        y = y0 + sub_offset[family] + offset
                        if task == "classification":
                            st = _family_marker(family, None)
                            ax.plot([record["ci_low"], record["ci_high"]], [y, y],
                                    color=FAMILY_STYLE[family]["color"], lw=0.8, zorder=2)
                            ax.plot([record["accuracy"]], [y], linestyle="none", zorder=4, **st)
                        else:
                            _clipped_interval_h(ax, y, record["skill"], record["ci_low"],
                                                record["ci_high"], family, record["outcome"], SKILL_CLIP)
                            if record["outcome"] != "trained":
                                who = record["client"].replace("_", " ") if record["client"] else (
                                    "FL" if family == "fl" else family)
                                ax.text(1.01, y, f"{who}: {record['outcome']}", transform=ax.get_yaxis_transform(),
                                        fontsize=5.8, color=INK, ha="left", va="center", clip_on=False)
                if position < len(rows) - 1:
                    ax.axhline(y0 - 0.5, color=GRID, lw=0.4, zorder=0)
            ax.set_ylim(-0.55, len(rows) - 0.45)
            if task == "classification":
                ax.set_xlim(-0.02, 1.02)
                ax.set_xlabel("Run accuracy (majority vote), Wilson 95% CI")
            else:
                ax.axvline(0.0, color=MUTED, lw=0.7, ls="-", zorder=1)
                ax.set_xlim(*SKILL_CLIP)
                ax.set_xlabel("Skill vs lookup, run-level bootstrap 95% CI")
            ax.tick_params(axis="y", length=0)
            _panel_letter(ax, letter)
        axes[0].set_yticks(range(len(rows)))
        axes[0].set_yticklabels([label for _, label in reversed(rows)])
        handles = [
            Line2D([], [], linestyle="none", label=FAMILY_STYLE[f]["label"], **_family_marker(f, None))
            for f in ("centralized", "fl", "local_only")
        ] + [
            Line2D([], [], linestyle="none", color=INK, label=f"regression: {o}",
                   **{**_family_marker("fl", o), "markeredgecolor": INK})
            for o in ("did not train", "collapsed")
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=5, fontsize=6.5)
        return _save(fig, Path(out_dir), "fig2_comparison")


# --------------------------------------------------------------------------------------
# Figure 3: sensor axis
# --------------------------------------------------------------------------------------


def sensor_axis_data(results: Results) -> dict[str, Any]:
    data: dict[str, Any] = {"rho": [], "classification": {}, "regression": {}}
    for condition, rho in sorted(RHO_CONDITIONS.items(), key=lambda kv: -kv[1]):
        if condition not in results.conditions("fl"):
            continue
        data["rho"].append(rho)
        data["classification"][rho] = {
            "fl": classification_records(results, "fl", condition),
            "local_only": classification_records(results, "local_only", condition),
        }
        data["regression"][rho] = {
            "fl": regression_records(results, "fl", condition),
            "local_only": regression_records(results, "local_only", condition),
        }
    data["centralized_run_accuracy"] = [r["accuracy"] for r in classification_records(results, "centralized", "classification")]
    data["centralized_skill"] = [r["skill"] for r in _centralized_regression(results, "residual")]
    return data


def plot_sensor_axis(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    data = sensor_axis_data(results)
    rhos = data["rho"]
    dx = 0.035
    with plt.rc_context(IEEE_RC):
        fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COLUMN_IN, 4.0), sharex=True,
                                 layout="constrained")
        for ax, task, letter in zip(axes, ("classification", "regression"), "ab"):
            value_key = "accuracy" if task == "classification" else "skill"
            fl_x, fl_y = [], []
            lo_x, lo_mean, lo_std = [], [], []
            for rho in rhos:
                fl = data[task][rho]["fl"]
                for record in fl:
                    x = rho + dx
                    ax.plot([x, x], [record["ci_low"], record["ci_high"]], color=FL_COLOR, lw=0.8, zorder=2)
                    ax.plot([x], [record[value_key]], linestyle="none", zorder=4,
                            **_family_marker("fl", record.get("outcome")))
                    fl_x.append(x)
                    fl_y.append(record[value_key])
                    if task == "regression":
                        ax.annotate(record["outcome"], (x, record[value_key]), xytext=(4, 4),
                                    textcoords="offset points", fontsize=5.8, color=INK,
                                    ha="left", va="bottom")
                local = data[task][rho]["local_only"]
                for record in local:
                    ax.plot([rho - dx], [record[value_key]], linestyle="none", alpha=0.45, zorder=3,
                            **_family_marker("local_only", record.get("outcome")))
                counted = [r for r in local if task == "classification" or r["outcome"] == "trained"]
                if counted:
                    mean, std = _mean_std([r[value_key] for r in counted])
                    ax.errorbar([rho - dx], [mean], yerr=[[std], [std]], color=LOCAL_COLOR, lw=0.8,
                                capsize=2, capthick=0.6, linestyle="none", zorder=4,
                                marker="s", markersize=5, markerfacecolor=LOCAL_COLOR,
                                markeredgecolor=LOCAL_COLOR)
                    lo_x.append(rho - dx)
                    lo_mean.append(mean)
                    lo_std.append(std)
                    if task == "regression":
                        ax.annotate(f"{len(counted)}/{len(local)} trained", (rho - dx, mean),
                                    xytext=(-4, -6), textcoords="offset points", fontsize=5.8,
                                    color=INK, ha="right", va="top")
            ax.plot(fl_x, fl_y, color=FL_COLOR, lw=0.8, ls="-", zorder=1)
            ax.plot(lo_x, lo_mean, color=LOCAL_COLOR, lw=0.8, ls="--", zorder=1)
            if task == "classification":
                ax.axhline(np.mean(data["centralized_run_accuracy"]), color=CENTRAL_COLOR,
                           ls=(0, (5, 2)), lw=0.8, zorder=0)
                ax.set_ylabel("Run accuracy")
                ax.set_ylim(0.85, 1.015)
            else:
                ax.axhline(0.0, color=MUTED, lw=0.6, zorder=0)
                ax.axhline(np.mean(data["centralized_skill"]), color=CENTRAL_COLOR,
                           ls=(0, (5, 2)), lw=0.8, zorder=0)
                ax.set_ylabel("Skill vs lookup")
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_axisbelow(True)
            _panel_letter(ax, letter)
        axes[-1].set_xlim(1.0 + 0.12, -0.12)
        axes[-1].set_xticks(rhos)
        axes[-1].set_xticklabels(["1.0" if r == 1.0 else f"{r:g}" for r in rhos])
        axes[-1].set_xlabel(r"Position overlap $\rho$ (K=3)")
        handles = [
            Line2D([], [], color=FL_COLOR, ls="-", label="FL, 95% CI", **_family_marker("fl", None)),
            Line2D([], [], color=LOCAL_COLOR, ls="--", marker="s", markersize=5,
                   markerfacecolor=LOCAL_COLOR, label="Local-only, mean ± std"),
            Line2D([], [], linestyle="none", alpha=0.45, label="Local-only client",
                   **_family_marker("local_only", None)),
            Line2D([], [], color=CENTRAL_COLOR, ls=(0, (5, 2)), lw=0.8, label="Centralized"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=6.5)
        return _save(fig, Path(out_dir), "fig3_sensor_axis")


# --------------------------------------------------------------------------------------
# Figure 4: partitions
# --------------------------------------------------------------------------------------


def _run_to_client(results: Results, row: Mapping[str, str]) -> dict[str, str]:
    rows = _read_csv(results.job_dir(row) / "partitions" / "combined_iid_run_to_client.csv")
    return {r["run_uid"]: r["client_id"] for r in rows}


def partition_schemes(results: Results) -> list[dict[str, Any]]:
    """One entry per distinct run assignment. Conditions that share an assignment (the
    K=3 ρ layouts mask channels on the IID K=3 runs) are merged into one panel; the
    assignment is asserted identical across tasks and init seeds of every condition."""
    by_condition: dict[str, dict[str, str]] = {}
    for condition in results.conditions("fl"):
        rows = results.rows("fl", condition) + results.rows("local_only", condition)
        assignments = [_run_to_client(results, row) for row in rows]
        if any(a != assignments[0] for a in assignments[1:]):
            raise AssertionError(f"{condition}: run assignment differs across tasks or seeds")
        by_condition[condition] = assignments[0]
    schemes: list[dict[str, Any]] = []
    for condition, assignment in by_condition.items():
        for scheme in schemes:
            if scheme["assignment"] == assignment:
                scheme["conditions"].append(condition)
                break
        else:
            schemes.append({"conditions": [condition], "assignment": assignment})
    return schemes


def _split_uid(uid: str) -> tuple[str, str]:
    source, subject, _ = uid.split(":")
    return source, subject


def partition_matrix(assignment: Mapping[str, str]) -> tuple[list[str], list[tuple[str, str]], np.ndarray]:
    clients = sorted(set(assignment.values()))
    groups = sorted({_split_uid(uid) for uid in assignment})
    matrix = np.zeros((len(clients), len(groups)), dtype=int)
    for uid, client in assignment.items():
        matrix[clients.index(client), groups.index(_split_uid(uid))] += 1
    return clients, groups, matrix


def _scheme_label(conditions: Sequence[str]) -> str:
    if len(conditions) == 1:
        return CONDITION_LABELS[conditions[0]]
    rhos = [RHO_CONDITIONS[c] for c in conditions if c in RHO_CONDITIONS]
    if len(rhos) == len(conditions):
        return "K=3 IID (" + ", ".join(rf"$\rho$={r:g}" for r in sorted(rhos, reverse=True)) + ")"
    return " / ".join(CONDITION_LABELS[c] for c in conditions)


def plot_partitions(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    schemes = partition_schemes(results)
    cmap = plt.get_cmap("Blues")
    vmax = max(partition_matrix(s["assignment"])[2].max() for s in schemes)
    ncols = 2
    nrows = math.ceil(len(schemes) / ncols)
    with plt.rc_context(IEEE_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COLUMN_IN, 1.45 * nrows + 0.3),
                                 layout="constrained", squeeze=False)
        for ax, scheme, letter in zip(axes.flat, schemes, "abcdefgh"):
            clients, groups, matrix = partition_matrix(scheme["assignment"])
            ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax * 1.15, aspect="auto")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=6.5,
                            color="white" if value > 0.6 * vmax else (INK if value else MUTED))
                ax.text(matrix.shape[1] - 0.35, i, f"  {matrix[i].sum()}", ha="left", va="center",
                        fontsize=6.5, color=INK)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([subject for _, subject in groups])
            ax.set_yticks(range(len(clients)))
            ax.set_yticklabels([c.replace("client_", "client ") for c in clients])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            sources = [source for source, _ in groups]
            for source in dict.fromkeys(sources):
                cols = [j for j, s in enumerate(sources) if s == source]
                ax.annotate(SOURCE_LABELS.get(source, source), xy=(np.mean(cols), 1.0),
                            xycoords=("data", "axes fraction"), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=6.5,
                            color=MUTED)
            ax.annotate("runs", xy=(matrix.shape[1] - 0.35, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points", ha="left", va="bottom", fontsize=6.5,
                        color=MUTED)
            boundaries = [j for j in range(1, len(sources)) if sources[j] != sources[j - 1]]
            for b in boundaries:
                ax.axvline(b - 0.5, color="white", lw=2.0)
            ax.set_xlabel(f"Subject — {_scheme_label(scheme['conditions'])}", fontsize=7)
            _panel_letter(ax, letter)
        for ax in list(axes.flat)[len(schemes):]:
            ax.set_visible(False)
        return _save(fig, Path(out_dir), "fig4_partitions")


# --------------------------------------------------------------------------------------
# Figure 5: run-level scatter
# --------------------------------------------------------------------------------------

SCATTER_SEED = 4601


def scatter_panels(results: Results) -> list[dict[str, Any]]:
    """Centralized residual and FL K=2 IID at seed 4601, then each local-only K=2 IID
    client (local-only runs at seed 4601 only)."""
    panels: list[dict[str, Any]] = []
    for label, record in (
        ("Centralized", _one(_centralized_regression(results, "residual"), SCATTER_SEED)),
        ("FL, K=2 IID", _one(regression_records(results, "fl", "k2_iid"), SCATTER_SEED)),
    ):
        panels.append({"label": label, "record": record})
    for record in regression_records(results, "local_only", "k2_iid"):
        if record["seed"] == SCATTER_SEED:
            panels.append({"label": f"Local-only, K=2 IID, {record['client'].replace('_', ' ')}",
                           "record": record})
    for panel in panels:
        panel["runs"] = _read_csv(panel["record"]["result_dir"] / "test_run_predictions.csv")
    return panels


def _one(records: Sequence[dict[str, Any]], seed: int) -> dict[str, Any]:
    found = [r for r in records if r["seed"] == seed and not r["client"]]
    if len(found) != 1:
        raise AssertionError(f"Expected one record at seed {seed}, got {len(found)}")
    return found[0]


def plot_run_scatter(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    panels = scatter_panels(results)
    subjects = sorted({r["subject_id"] for p in panels for r in p["runs"]})
    if len(subjects) > len(SUBJECT_COLORS):
        raise AssertionError("More subjects than palette slots; fold or facet instead.")
    style = {s: (SUBJECT_COLORS[i], SUBJECT_MARKERS[i]) for i, s in enumerate(subjects)}
    values = [float(r[k]) for p in panels for r in p["runs"] for k in ("actual_speed_mps", "predicted_speed_mps")]
    lo, hi = math.floor(min(values) * 20) / 20, math.ceil(max(values) * 20) / 20
    with plt.rc_context(IEEE_RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE_COLUMN_IN, 2.35), sharex=True,
                                 sharey=True, layout="constrained", squeeze=False)
        for ax, panel, letter in zip(axes.flat, panels, "abcdefgh"):
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, zorder=1)
            for run in sorted(panel["runs"], key=lambda r: r["run_uid"]):
                color, marker = style[run["subject_id"]]
                filled = run["source_id"] == "test_2"
                ax.plot([float(run["actual_speed_mps"])], [float(run["predicted_speed_mps"])],
                        linestyle="none", marker=marker, markersize=3.6, markeredgewidth=0.7,
                        markeredgecolor=color, markerfacecolor=color if filled else "white", zorder=3)
            record = panel["record"]
            text = (f"RMSE {record['rmse']:.4f} m/s\nlookup {record['baseline_rmse']:.4f} m/s\n"
                    f"skill {_fmt_signed(record['skill'], 3, plus=True)}\n[{_fmt_signed(record['ci_low'], 2, plus=True)}, "
                    f"{_fmt_signed(record['ci_high'], 2, plus=True)}]")
            ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=6.0,
                    color=INK, linespacing=1.15)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Actual speed (m/s)")
            ax.text(0.97, 0.03, panel["label"], transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.0, color=MUTED)
            _panel_letter(ax, letter)
        axes[0, 0].set_ylabel("Predicted speed (m/s)")
        handles = [
            Line2D([], [], linestyle="none", marker=style[s][1], markersize=3.8,
                   markeredgecolor=style[s][0], markerfacecolor=style[s][0], label=s)
            for s in subjects
        ] + [
            Line2D([], [], linestyle="none", marker="o", markersize=3.8, markeredgecolor=INK,
                   markerfacecolor=INK, label="Test_2 (filled)"),
            Line2D([], [], linestyle="none", marker="o", markersize=3.8, markeredgecolor=INK,
                   markerfacecolor="white", label="20251124 (hollow)"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), fontsize=6.5,
                   handletextpad=0.2, columnspacing=0.9)
        return _save(fig, Path(out_dir), "fig5_run_scatter")


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def _tex(text: Any) -> str:
    s = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    s = s.replace("−", "$-$").replace("±", r"$\pm$").replace("→", r"$\rightarrow$")
    return s.replace("rho=", r"$\rho$=").replace("alpha=", r"$\alpha$=")


def _num(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if value != 0 and abs(value) < 10 ** (-digits):
        text = f"{value:.1e}"
    else:
        text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "−")


def _ci(low: float, high: float, digits: int = 3, signed: bool = False) -> str:
    return f"[{_num(low, digits, signed)}, {_num(high, digits, signed)}]"


def _runs(correct: int, runs: int, low: float, high: float) -> str:
    return f"{correct}/{runs} {_ci(low, high)}"


def _write_table(out_dir: Path, stem: str, csv_rows: Sequence[Mapping[str, Any]],
                 header: Sequence[str], body: Sequence[Sequence[str] | str], align: str,
                 notes: Sequence[str] = ()) -> list[Path]:
    """CSV (machine-readable, full precision) and a booktabs ``tabular`` (formatted).
    A ``str`` entry in ``body`` is emitted verbatim (e.g. ``\\midrule``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    fields: list[str] = []
    for row in csv_rows:
        fields.extend(k for k in row if k not in fields)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    lines = [f"% generated by redo_by_sara.paper_figures; do not edit by hand",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(_tex(h) for h in header) + r" \\", r"\midrule"]
    for entry in body:
        if isinstance(entry, str):
            lines.append(entry)
        else:
            lines.append(" & ".join(_tex(c) for c in entry) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    for note in notes:
        lines.append(f"% note: {note}")
    tex_path = out_dir / f"{stem}.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    return [tex_path, csv_path]


# ---- T1 dataset ----------------------------------------------------------------------


def dataset_table_data(results: Results) -> dict[str, Any]:
    cls_summary = _read_json(results.root / CENTRALIZED_CLASSIFICATION_DIR / "artifact_summary.json")
    reg_summary = _read_json(results.root / CENTRALIZED_REGRESSION_DIR / "artifact_summary.json")
    cls_manifest = _read_csv(results.root / CENTRALIZED_CLASSIFICATION_DIR / "run_manifest.csv")
    reg_manifest = {r["run_uid"]: r for r in _read_csv(results.root / CENTRALIZED_REGRESSION_DIR / "run_manifest.csv")}
    if set(reg_manifest) != {r["run_uid"] for r in cls_manifest}:
        raise AssertionError("Classification and regression manifests list different runs.")
    status = Counter(r["split"] for r in cls_manifest)
    if dict(status) != cls_summary["run_status_counts"] or dict(status) != reg_summary["run_status_counts"]:
        raise AssertionError("Manifest split counts disagree with the artifact summaries.")

    strata: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {
        "runs_train": 0, "runs_test": 0, "cls_train": 0, "cls_test": 0, "reg_train": 0,
        "reg_test": 0, "speeds": []})
    by_source = defaultdict(Counter)
    for run in cls_manifest:
        by_source[run["source_id"]][run["split"]] += 1
        if run["split"] not in ("train", "test"):
            continue
        key = (run["source_id"], run["subject_id"])
        split = run["split"]
        reg = reg_manifest[run["run_uid"]]
        if reg["split"] != split:
            raise AssertionError(f"{run['run_uid']}: split differs between tasks")
        strata[key][f"runs_{split}"] += 1
        strata[key][f"cls_{split}"] += int(run["no_walking_windows"]) + int(run["walking_windows"])
        strata[key][f"reg_{split}"] += int(reg["walking_windows"])
        strata[key]["speeds"].append(float(reg["speed_mps"]))
    totals = {k: sum(s[k] for s in strata.values()) for k in
              ("runs_train", "runs_test", "cls_train", "cls_test", "reg_train", "reg_test")}
    expected = {"runs_train": cls_summary["num_train_runs"], "runs_test": cls_summary["num_test_runs"],
                "cls_train": cls_summary["num_train"], "cls_test": cls_summary["num_test"],
                "reg_train": reg_summary["num_train"], "reg_test": reg_summary["num_test"]}
    if totals != expected:
        raise AssertionError(f"T1 totals {totals} disagree with the artifact summaries {expected}")
    return {"strata": dict(sorted(strata.items())), "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
            "status": dict(status), "excluded_subjects": cls_summary["excluded_subjects"],
            "num_recorded_runs": cls_summary["num_recorded_runs"]}


def _within_std(groups: Sequence[Sequence[float]]) -> float:
    """Pooled within-group std: sqrt(Σ_g Σ (x − x̄_g)² / (N − G))."""
    ss = sum(float(np.sum((np.asarray(g) - np.mean(g)) ** 2)) for g in groups)
    dof = sum(len(g) for g in groups) - len(groups)
    return math.sqrt(ss / dof)


def table_dataset(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    data = dataset_table_data(results)
    csv_rows, body = [], []

    def emit(source: str, subject: str, members: list[dict[str, Any]]) -> None:
        speeds = [x for m in members for x in m["speeds"]]
        within = _within_std([m["speeds"] for m in members]) if len(members) > 1 else float(np.std(speeds, ddof=1))
        values = {k: sum(m[k] for m in members) for k in
                  ("runs_train", "runs_test", "cls_train", "cls_test", "reg_train", "reg_test")}
        csv_rows.append({"source": source, "subject": subject,
                         "usable_runs": values["runs_train"] + values["runs_test"], **values,
                         "speed_mean_mps": float(np.mean(speeds)), "speed_std_mps": float(np.std(speeds, ddof=1)),
                         "speed_min_mps": float(np.min(speeds)), "speed_max_mps": float(np.max(speeds)),
                         "within_subject_std_mps": within})
        body.append([source, subject, str(values["runs_train"] + values["runs_test"]),
                     f"{values['runs_train']}/{values['runs_test']}",
                     f"{values['cls_train']}/{values['cls_test']}",
                     f"{values['reg_train']}/{values['reg_test']}",
                     f"{np.mean(speeds):.3f}", f"{np.std(speeds, ddof=1):.3f}",
                     f"{np.min(speeds):.3f}–{np.max(speeds):.3f}", f"{within:.3f}"])

    for source in sorted({s for s, _ in data["strata"]}):
        keys = [k for k in data["strata"] if k[0] == source]
        for key in keys:
            emit(SOURCE_LABELS.get(source, source), key[1], [data["strata"][key]])
        emit(SOURCE_LABELS.get(source, source), "all", [data["strata"][k] for k in keys])
        body.append(r"\midrule")
    emit("All", "all", list(data["strata"].values()))

    funnel = data["status"]
    usable = funnel["train"] + funnel["test"]
    csv_rows.append({"source": "run funnel", "subject": "",
                     "recorded_runs": data["num_recorded_runs"], "excluded_runs": funnel["excluded"],
                     "skipped_runs": funnel["skipped"], "usable_runs": usable,
                     "excluded_subjects": " ".join(data["excluded_subjects"])})
    for source, counts in data["by_source"].items():
        csv_rows.append({"source": f"run funnel {SOURCE_LABELS.get(source, source)}", "subject": "",
                         "recorded_runs": sum(counts.values()), "excluded_runs": counts.get("excluded", 0),
                         "skipped_runs": counts.get("skipped", 0),
                         "usable_runs": counts.get("train", 0) + counts.get("test", 0)})
    notes = [f"{data['num_recorded_runs']} recorded runs -> {funnel['excluded']} excluded "
             f"(subject {', '.join(data['excluded_subjects'])}) + {funnel['skipped']} skipped -> {usable} usable",
             "speed statistics are over runs (one APDM mean speed per run); within-subject std is "
             "the pooled std within (source, subject) strata",
             "window counts: classification includes no_walking windows; regression is walking windows only"]
    return _write_table(Path(out_dir), "t1_dataset", csv_rows,
                        ["Source", "Subject", "Runs", "Train/test runs", "Cls. windows", "Reg. windows",
                         "Speed mean", "Speed std", "Range (m/s)", "Within-subj. std"],
                        body, "llrrrrrrrr", notes)


# ---- T2 centralized reference --------------------------------------------------------


def residual_300e_record(results: Results) -> dict[str, Any]:
    folder = results.root / CENTRALIZED_RESIDUAL_300E_DIR
    summary = _read_json(folder / "training_summary.json")
    runs = _read_csv(folder / "test_run_predictions.csv")
    report = report_from_run_rows(runs, "source_subject_train_mean_mps")
    final_mse = float(summary["degenerate_model_check"]["final_standardized_mse"])
    reapplied = reapply_detector(folder, absolute=False, final_standardized_mse=final_mse)
    overall = report["overall"]
    return {"skill": overall["skill"], "ci_low": overall["skill_ci95"][0], "ci_high": overall["skill_ci95"][1],
            "rmse": overall["model_rmse_mps"], "baseline_rmse": overall["baseline_rmse_mps"],
            "gated_mse": final_mse, "outcome": training_outcome(reapplied), "epochs": int(summary["epochs"]),
            "seed": int(summary["seeding"]["seed"]), "result_dir": folder}


def table_centralized(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    steps = gradient_steps(results)
    csv_rows: list[dict[str, Any]] = []
    body: list[Any] = []
    lookup_rmse = _one_value([r["baseline_rmse"] for r in _centralized_regression(results, "residual")],
                             "lookup RMSE across centralized runs")

    cls = classification_records(results, "centralized", "classification")
    for r in cls:
        csv_rows.append({"model": "classification", "epochs": 60, "steps": steps["centralized_60e_classification"],
                         "seed": r["seed"], "run_correct": r["correct"], "runs": r["runs"],
                         "run_accuracy": r["accuracy"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "window_accuracy": r["window_accuracy"]})
        body.append(["Classification", "60", f"{steps['centralized_60e_classification']:,}", str(r["seed"]),
                     _runs(r["correct"], r["runs"], r["ci_low"], r["ci_high"]), _num(r["window_accuracy"], 4),
                     "", ""])
    mean, std = _mean_std([r["window_accuracy"] for r in cls])
    body.append(["", "", "", "mean ± std", "", f"{_num(mean, 4)} ± {_num(std, 4)}", "", ""])
    body.append(r"\midrule")
    csv_rows.append({"model": "lookup baseline", "skill": 0.0, "rmse_mps": lookup_rmse})
    body.append(["Lookup (source/subject train mean)", "", "", "", "0 (by definition)",
                 _num(lookup_rmse, 4), "", ""])
    for condition, label in (("absolute", "Absolute speed"), ("residual", "Residual")):
        records = _centralized_regression(results, condition)
        for r in records:
            csv_rows.append({"model": f"regression {condition}", "epochs": 60,
                             "steps": steps["centralized_60e_regression"], "seed": r["seed"],
                             "skill": r["skill"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                             "rmse_mps": r["rmse"], "final_train_standardized_mse": r["gated_mse"],
                             "outcome": r["outcome"]})
            body.append([label, "60", f"{steps['centralized_60e_regression']:,}", str(r["seed"]),
                         f"{_num(r['skill'], 3, True)} {_ci(r['ci_low'], r['ci_high'], signed=True)}", _num(r["rmse"], 4),
                         _num(r["gated_mse"], 3), r["outcome"]])
        mean, std = _mean_std([r["skill"] for r in records])
        body.append(["", "", "", "mean ± std", f"{_num(mean, 3, True)} ± {_num(std, 3)}", "", "",
                     f"{sum(r['outcome'] == 'trained' for r in records)} of {len(records)} trained"])
    r300 = residual_300e_record(results)
    csv_rows.append({"model": "regression residual (matched steps)", "epochs": r300["epochs"],
                     "steps": steps["centralized_300e_regression"], "seed": r300["seed"], "skill": r300["skill"],
                     "ci_low": r300["ci_low"], "ci_high": r300["ci_high"], "rmse_mps": r300["rmse"],
                     "final_train_standardized_mse": r300["gated_mse"], "outcome": r300["outcome"]})
    body.append(["Residual, matched steps", str(r300["epochs"]), f"{steps['centralized_300e_regression']:,}",
                 str(r300["seed"]), f"{_num(r300['skill'], 3, True)} {_ci(r300['ci_low'], r300['ci_high'], signed=True)}",
                 _num(r300["rmse"], 4), _num(r300["gated_mse"], 3), r300["outcome"]])
    notes = ["classification: run accuracy (majority vote) k/28 [Wilson 95%]; regression: skill = 1 - SSE/SSE_lookup "
             "[run-level bootstrap 95%, 10,000 resamples]; window accuracy for classification, RMSE (m/s) for regression",
             "absolute path has no standardized MSE and no trained gate; it is labelled on degeneracy alone"]
    return _write_table(Path(out_dir), "t2_centralized", csv_rows,
                        ["Model", "Epochs", "Steps", "Seed", "Run acc. / skill [95% CI]", "Win. acc. / RMSE",
                         "Train std. MSE", "Outcome"], body, "lrrrlrrl", notes)


# ---- T3 the S3 sweep ------------------------------------------------------------------


def sweep_records(results: Results) -> list[dict[str, Any]]:
    sweep = _read_json(results.root / "s3_r1_sweep" / "sweep_summary.json")
    entries = [("K=3 IID", e) for e in sweep["results"].values()]
    entries += [("K=2 natural", a["regression"]) for a in sweep["confirmation"]["attempts"]]
    out = []
    for partition, entry in entries:
        folder = results.rebase(entry["output_dir"])
        runs = _read_csv(folder / "test_run_predictions.csv")
        report = report_from_run_rows(runs, "source_subject_train_mean_mps")
        final_mse = float(entry["final_global_train_standardized_mse"])
        reapplied = reapply_detector(folder, absolute=False, final_standardized_mse=final_mse)
        detector = entry["detector"]
        recorded_skill = float(detector["skill_score_vs_subject_mean"])
        if not math.isclose(report["overall"]["skill"], recorded_skill, rel_tol=0, abs_tol=1e-9):
            raise AssertionError(f"{entry['name']}: recomputed skill {report['overall']['skill']} "
                                 f"!= recorded {recorded_skill}")
        failed = [g for g, v in detector["gates"].items() if v["enforced"] and not v["passed"]]
        out.append({
            "partition": partition, "config": entry["name"], "stage": entry["stage"],
            "aggregation": entry["aggregation"], "local_epochs": int(entry["local_epochs"]),
            "optimizer_state": entry["client_optimizer_state"], "steps": int(entry["total_gradient_steps"]),
            "final_standardized_mse": final_mse,
            "distinct_residuals": int(reapplied["distinct_predictions"]),
            "relative_output_spread": float(reapplied["relative_output_spread"]),
            "skill": report["overall"]["skill"], "ci_low": report["overall"]["skill_ci95"][0],
            "ci_high": report["overall"]["skill_ci95"][1], "rmse": report["overall"]["model_rmse_mps"],
            "s3_acceptance": "passed" if detector["passed"] else "failed: " + ", ".join(failed),
            "s3_status": entry["status"], "outcome_s5": training_outcome(reapplied),
        })
    return out


def table_sweep(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    records = sweep_records(results)
    body: list[Any] = []
    previous = None
    for r in records:
        if previous is not None and r["partition"] != previous:
            body.append(r"\midrule")
        previous = r["partition"]
        body.append([r["partition"], r["aggregation"], str(r["local_epochs"]), r["optimizer_state"],
                     f"{r['steps']:,}", _num(r["final_standardized_mse"], 4), str(r["distinct_residuals"]),
                     _num(r["relative_output_spread"], 3),
                     f"{_num(r['skill'], 3, True)} {_ci(r['ci_low'], r['ci_high'], signed=True)}",
                     r["s3_acceptance"], r["outcome_s5"]])
    notes = ["S3 acceptance: final standardized MSE < 0.95, > 10 distinct residuals, skill > 0 (as recorded in S3)",
             "outcome re-applied with S5 definitions to the saved predictions: collapsed = output spread "
             "std(out)/std(target) < 1e-3; did not train = has spread, standardized MSE >= 0.95; trained = both pass",
             "every row is reported, failures included"]
    return _write_table(Path(out_dir), "t3_s3_sweep", records,
                        ["Partition", "Server", "E", "Client optimizer", "Steps", "Std. MSE", "Distinct",
                         "Spread", "Skill [95% CI]", "S3 acceptance",
                         "Outcome (re-applied with S5 definitions)"], body, "llrlrrrrlll", notes)


# ---- T4 the main matrix ---------------------------------------------------------------


def _classification_cell(records: Sequence[dict[str, Any]]) -> str:
    keys = {(r["correct"], r["runs"]) for r in records}
    if len(keys) == 1:
        r = records[0]
        text = _runs(r["correct"], r["runs"], r["ci_low"], r["ci_high"])
        return text + (f", every seed ({len(records)})" if len(records) > 1 else "")
    return "; ".join(f"{r['seed']}: {_runs(r['correct'], r['runs'], r['ci_low'], r['ci_high'])}" for r in records)


def _local_classification_cell(records: Sequence[dict[str, Any]]) -> str:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_seed[r["seed"]].append(r)
    signatures = {tuple((r["client"], r["correct"]) for r in sorted(v, key=_seed_sort_key)) for v in by_seed.values()}
    first = sorted(next(iter(by_seed.values())), key=_seed_sort_key)
    mean, std = _mean_std([r["accuracy"] for r in first])
    clients = ", ".join(f"{r['correct']}/{r['runs']}" for r in first)
    text = f"{_num(mean)} ± {_num(std)} ({clients})"
    if len(by_seed) > 1:
        if len(signatures) != 1:
            return "; ".join(
                f"{seed}: " + ", ".join(f"{r['correct']}/{r['runs']}" for r in sorted(v, key=_seed_sort_key))
                for seed, v in sorted(by_seed.items()))
        text += f", every seed ({len(by_seed)})"
    return text


def _fl_regression_cell(records: Sequence[dict[str, Any]]) -> str:
    if len(records) == 1:
        r = records[0]
        return f"{_num(r['skill'], 3, True)} {_ci(r['ci_low'], r['ci_high'], signed=True)}, {r['outcome']}"
    trained = [r for r in records if r["outcome"] == "trained"]
    per_seed = "; ".join(f"{r['seed']}: {_num(r['skill'], 3, True)} {_ci(r['ci_low'], r['ci_high'], signed=True)}"
                         + ("" if r["outcome"] == "trained" else f" {r['outcome']}") for r in records)
    if len(trained) == len(records):
        mean, std = _mean_std([r["skill"] for r in records])
        return f"{_num(mean, 3, True)} ± {_num(std, 3)}, {len(records)} of {len(records)} trained ({per_seed})"
    labels = Counter(r["outcome"] for r in records if r["outcome"] != "trained")
    status = ", ".join(f"{n} of {len(records)} seeds {label}" for label, n in sorted(labels.items()))
    return f"per seed; {status} ({per_seed})"


def _local_regression_cell(records: Sequence[dict[str, Any]]) -> str:
    trained = [r for r in records if r["outcome"] == "trained"]
    failed = [r for r in records if r["outcome"] != "trained"]
    mean, std = _mean_std([r["skill"] for r in trained])
    text = f"{_num(mean, 3, True)} ± {_num(std, 3)} over {len(trained)}"
    text += f"; {len(failed)} of {len(records)} clients failed"
    if failed:
        text += " (" + ", ".join(f"{r['client']} {r['outcome']}" for r in failed) + ")"
    return text


def main_matrix(results: Results) -> list[dict[str, Any]]:
    rows = []
    central_cls = classification_records(results, "centralized", "classification")
    central_reg = _centralized_regression(results, "residual")
    rows.append({"condition": "centralized", "label": "Centralized",
                 "cls_fl": _classification_cell(central_cls), "cls_local": "—",
                 "reg_fl": _fl_regression_cell(central_reg), "reg_local": "—"})
    for condition in results.conditions("fl"):
        rows.append({
            "condition": condition, "label": CONDITION_TEXT[condition],
            "cls_fl": _classification_cell(classification_records(results, "fl", condition)),
            "cls_local": _local_classification_cell(classification_records(results, "local_only", condition)),
            "reg_fl": _fl_regression_cell(regression_records(results, "fl", condition)),
            "reg_local": _local_regression_cell(regression_records(results, "local_only", condition)),
        })
    return rows


def main_matrix_long(results: Results) -> list[dict[str, Any]]:
    """One row per evaluated model, every condition (the CSV of T4)."""
    out = []
    for family, conditions in (("centralized", ["classification"]), ("fl", results.conditions("fl")),
                               ("local_only", results.conditions("local_only"))):
        for condition in conditions:
            for r in classification_records(results, family, condition):
                out.append({"family": family, "condition": condition, "task": "classification", "seed": r["seed"],
                            "client": r["client"], "metric": "run_accuracy_majority_vote", "value": r["accuracy"],
                            "ci_low": r["ci_low"], "ci_high": r["ci_high"], "correct": r["correct"],
                            "runs": r["runs"], "outcome": ""})
    for family, conditions in (("centralized", ["residual"]), ("fl", results.conditions("fl")),
                               ("local_only", results.conditions("local_only"))):
        for condition in conditions:
            for r in regression_records(results, family, condition):
                out.append({"family": family, "condition": condition, "task": "regression", "seed": r["seed"],
                            "client": r["client"], "metric": "skill", "value": r["skill"], "ci_low": r["ci_low"],
                            "ci_high": r["ci_high"], "rmse_mps": r["rmse"], "outcome": r["outcome"]})
    return out


def table_main_matrix(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    rows = main_matrix(results)
    lookup = _one_value([r["baseline_rmse"] for r in regression_records(results, "fl")],
                        "lookup RMSE across FL runs")
    body: list[Any] = [[r["label"], r["cls_fl"], r["cls_local"], r["reg_fl"], r["reg_local"]] for r in rows]
    body.append(r"\midrule")
    body.append(["Lookup baseline", "", "", f"skill 0 by definition; RMSE {_num(lookup, 4)} m/s", ""])
    paths = _write_table(Path(out_dir), "t4_main_matrix", main_matrix_long(results),
                         ["Condition", "Cls.: FL (centralized row 1)", "Cls.: local-only, mean ± std over clients",
                          "Reg.: FL (centralized row 1)", "Reg.: local-only, mean ± std over trained clients"],
                         body, "lp{3.2cm}p{3.2cm}p{4.2cm}p{3.2cm}",
                         ["classification: run accuracy (majority vote) k/28 [Wilson 95%]; regression: skill "
                          "[run-level bootstrap 95%]; centralized regression is the residual model",
                          "a multi-seed group with any seed that is not trained is reported per seed"])
    summary = Path(out_dir) / "t4_main_matrix_summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return paths + [summary]


# ---- T5 controls ----------------------------------------------------------------------


def _label_fractions(labels: Sequence[str]) -> str:
    counts = Counter(labels)
    total = sum(counts.values())
    return ", ".join(f"{_short(k)} {v / total:.3f} ({v})" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _short(label: str) -> str:
    return label.replace("subject_", "").replace("_walking", "")


def reverse_direction_records(results: Results) -> list[dict[str, Any]]:
    """Unplanned: local-only K=2 natural, the client trained on 20251124 only, on the
    Test_2 canonical-test runs (walking windows). The client is identified from its
    partition, not by name."""
    rows = results.rows("local_only", "k2_natural", "classification")
    if len(rows) != 1:
        raise AssertionError("Expected one local-only K=2 natural classification job")
    row = rows[0]
    assignment = _run_to_client(results, row)
    sources_by_client: dict[str, set[str]] = defaultdict(set)
    for uid, client in assignment.items():
        sources_by_client[client].add(_split_uid(uid)[0])
    trained_on = {c: s for c, s in sources_by_client.items() if len(s) == 1}
    client = next(c for c, s in sorted(trained_on.items()) if s == {"testing_20251124"})
    seen = {_split_uid(uid)[1] for uid, c in assignment.items() if c == client}
    folder = results.result_dir(row) / client
    windows = [w for w in _read_csv(folder / "test_predictions.csv")
               if w["source_id"] == "test_2" and w["true_class"] != "no_walking"]
    runs = [r for r in _read_csv(folder / "run_level" / "run_level_predictions.csv") if r["source_id"] == "test_2"]
    out = []
    for subject in sorted({w["subject_id"] for w in windows}):
        sw = [w for w in windows if w["subject_id"] == subject]
        sr = [r for r in runs if r["subject_id"] == subject]
        vote = sum(r["majority_vote_label"] == r["true_label"] for r in sr)
        meanprob = sum(r["mean_probability_label"] == r["true_label"] for r in sr)
        low, high = wilson_interval(vote, len(sr))
        out.append({"control": "reverse direction (unplanned)", "set": f"{client}: trained on 20251124 only",
                    "source": "test_2", "subject": subject, "seen_in_training": subject in seen,
                    "runs": len(sr), "windows": len(sw),
                    "run_vote_correct": vote if subject in seen else None,
                    "run_meanprob_correct": meanprob if subject in seen else None,
                    "ci_low": low if subject in seen else None, "ci_high": high if subject in seen else None,
                    "run_vote_labels": _label_fractions([r["majority_vote_label"] for r in sr]),
                    "window_labels": _label_fractions([w["predicted_class"] for w in sw])})
    return out


def controls_records(results: Results) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    probe_row = results.rows("source_probe")[0]
    probe = _read_json(results.job_dir(probe_row) / "source_probe_summary.json")
    vote = probe["run_level"]["majority_vote"]
    out.append({"control": "source probe", "set": "canonical test", "source": "both", "subject": "all",
                "runs": vote["num_runs"], "windows": None, "run_vote_correct": vote["num_correct"],
                "ci_low": vote["wilson95"][0], "ci_high": vote["wilson95"][1],
                "window_accuracy": probe["test_window_accuracy"],
                "window_accuracy_no_walking": probe["test_window_accuracy_by_activity"]["no_walking"]["accuracy"],
                "window_accuracy_walking": probe["test_window_accuracy_by_activity"]["walking"]["accuracy"],
                "majority_class_rate": probe["chance_accuracy_majority_class_test_windows"]})
    for source, block in sorted(vote["per_source"].items()):
        out.append({"control": "source probe", "set": "canonical test", "source": source, "subject": "all",
                    "runs": block["num_runs"], "run_vote_correct": block["num_correct"],
                    "ci_low": block["wilson95"][0], "ci_high": block["wilson95"][1]})
    control_row = results.rows("subject003_control")[0]
    control = _read_json(results.job_dir(control_row) / "subject003_control_summary.json")
    distribution = control["predicted_label_distribution"]
    classes = control["classes"]
    for entry in distribution:
        fractions = ", ".join(f"{_short(c)} {entry[f'window_fraction_{c}']:.3f}" for c in classes)
        runs_vote = ", ".join(f"{_short(c)} {entry[f'runs_majority_{c}']}" for c in classes if entry[f"runs_majority_{c}"])
        record = {"control": "003 cross-session control" if entry["control_set"] != "negative_control"
                  else "negative control", "set": entry["control_set"], "source": entry["source_id"],
                  "subject": entry["subject_id"], "runs": entry["num_runs"], "windows": entry["num_windows"],
                  "run_vote_labels": runs_vote, "window_labels": fractions}
        if entry["control_set"] in ("in_session", "cross_session"):
            target = f"subject_{entry['subject_id']}_walking"
            correct = entry[f"runs_majority_{target}"]
            low, high = wilson_interval(correct, entry["num_runs"])
            record.update({"run_vote_correct": correct, "ci_low": low, "ci_high": high,
                           "run_meanprob_correct": entry[f"runs_mean_probability_{target}"]})
        out.append(record)
    out.extend(reverse_direction_records(results))
    return out


CONTROL_SETS = ("in_session", "cross_session", "negative_control")


def table_controls(results: PathLike | Results, out_dir: PathLike) -> list[Path]:
    results = load_results(results)
    records = controls_records(results)
    body: list[Any] = []
    previous = None
    for r in records:
        if previous is not None and r["control"] != previous:
            body.append(r"\midrule")
        previous = r["control"]
        if r.get("run_vote_correct") is not None:
            result = _runs(r["run_vote_correct"], r["runs"], r["ci_low"], r["ci_high"])
            if r.get("run_meanprob_correct") is not None:
                result += f"; mean prob. {r['run_meanprob_correct']}/{r['runs']}"
        else:
            result = "not in the classes" if r["control"] == "negative control" else (
                "not a training subject" if r.get("seen_in_training") is False else "")
        if r["control"] == "source probe" and r["source"] == "both":
            detail = (f"window acc. {r['window_accuracy']:.3f} (no_walking {r['window_accuracy_no_walking']:.3f}, "
                      f"walking {r['window_accuracy_walking']:.3f}); majority class {r['majority_class_rate']:.3f}")
        else:
            detail = r.get("window_labels", "")
        set_label = r["set"].replace("_", "-") if r["set"] in CONTROL_SETS else r["set"]
        body.append([r["control"], set_label, SOURCE_LABELS.get(r["source"], r["source"]),
                     r["subject"], str(r["runs"]), "" if r.get("windows") is None else str(r["windows"]),
                     result, detail])
    notes = ["run results: majority vote k/n [Wilson 95%]; window labels: predicted-label fractions over walking windows",
             "the reverse-direction rows are an unplanned observation from the existing local-only K=2 natural run, "
             "not a planned control"]
    return _write_table(Path(out_dir), "t5_controls", records,
                        ["Control", "Set", "Source", "Subject", "Runs", "Windows", "Run result",
                         "Window-level predicted labels"], body, "lllrrrlp{5cm}", notes)


# --------------------------------------------------------------------------------------
# captions and the entry point
# --------------------------------------------------------------------------------------


def write_captions(results: Results, out_dir: Path) -> Path:
    steps = gradient_steps(results)
    seeds = sorted({int(r["seed"]) for r in results.rows("fl", "k2_iid")})
    text = f"""% Draft captions, generated from the saved results. Numbers are read from the artifacts.

Fig. 1 (fig1_convergence). Global-train metric versus communication round for FedAvg
(E=5, persistent client optimizer) at K=2, IID and natural cross-session split; (a)
classification accuracy, (b) residual-regression standardized MSE (log scale; 1 = the
lookup). Bands span the trained init seeds ({', '.join(map(str, seeds))}); a seed that did not train or
collapsed is drawn as its own dotted line. FL takes {steps['fl_k2_iid_regression']:,} gradient steps
(regression) and {steps['fl_k2_iid_classification']:,} (classification); the centralized references take
{steps['centralized_60e_regression']:,} / {steps['centralized_60e_classification']:,} (60 epochs), and the
"centralized, matched steps" residual run {steps['centralized_300e_regression']:,} (300 epochs).

Fig. 2 (fig2_comparison). Centralized, FL and local-only models on the 28 held-out
runs, every partition condition. (a) Run accuracy by majority vote with Wilson 95%
intervals; (b) skill against the source/subject lookup with run-level bootstrap 95%
intervals (10,000 resamples), zero = the lookup. One point per init seed (FL,
centralized) or per client (local-only). The skill axis is clipped at {_fmt_signed(SKILL_CLIP[0], 0)}; arrowheads
mark cut-off intervals and the hidden values are printed. Regression models that did not
train or collapsed are marked and labelled.

Fig. 3 (fig3_sensor_axis). FL and local-only performance against position overlap rho at
K=3 (contiguous position blocks, masked inputs, global 9-channel evaluation). Each FL
regression point is labelled with its training outcome; local-only shows mean ± std over
trained clients with each client faint behind it.

Fig. 4 (fig4_partitions). Training runs per client and (source, subject) for each
partition scheme at split seed 4601; row totals at right. The three rho layouts share
the IID K=3 run assignment and differ only in which positions each client holds.

Fig. 5 (fig5_run_scatter). Predicted versus actual walking speed per held-out run,
identity line; colour and marker by subject, filled = Test_2, hollow = 20251124. Init seed
{SCATTER_SEED} for centralized (residual) and FL K=2 IID; local-only K=2 IID, one panel per client.
"""
    path = out_dir / "captions.md"
    path.write_text(text)
    return path


FIGURES = (plot_convergence, plot_comparison, plot_sensor_axis, plot_partitions, plot_run_scatter)
TABLES = (table_dataset, table_centralized, table_sweep, table_main_matrix, table_controls)


def make_all_figures(results_dir: PathLike | Results, out_dir: PathLike) -> list[Path]:
    """Every figure (``out_dir/figures``) and table (``out_dir/tables``), in paper order.
    A failure in an earlier figure stops the build; nothing is skipped silently."""
    results = load_results(results_dir)
    out = Path(out_dir)
    written: list[Path] = []
    for figure in FIGURES:
        written += figure(results, out / "figures")
    written.append(write_captions(results, out / "figures"))
    for table in TABLES:
        written += table(results, out / "tables")
    return written

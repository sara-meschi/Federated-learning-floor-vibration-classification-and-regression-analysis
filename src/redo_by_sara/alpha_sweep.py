"""Alpha sweep (2026-11-18): FL K=3 Dirichlet regression across alpha and partition draws.

``scripts/run_alpha_sweep.py`` drives the queue; ``scripts/make_alpha_sweep_excel.py``
builds the workbook. The design and the analysis below were fixed and recorded in
CHANGES.md ("Alpha sweep — design and pre-registered analysis") before any job ran.

Design:
- alpha in {0.05, 0.1, 0.3, 1, 10}; K=3; FedAvg, E=5, persistent client optimizer,
  60 rounds, client_num_cpus 3, 2-identity floor; split seed 4601, init seed 4601.
- Only the Dirichlet **partition seed** varies: 4601 / 5601 / 6601 = draws 1 / 2 / 3.
  Draw 1 is the split-seed partition every earlier Dirichlet run used, so the relabel
  alpha = 0.1 and alpha = 0.3 regression runs (init 4601) are draw 1 of those alphas and
  are read from ``artifacts/relabel`` (read-only), not rerun. 13 new confirmatory jobs.
- Two initialization controls, outside the confirmatory set: alpha = 0.1 and 0.3, draw 1,
  init seed 4602, to separate partition effects from training-trajectory effects.

Pre-registered analysis (all measured confirmatory runs; gate outcomes are labels, never
exclusions):
1. Per alpha: mean ± sample std of RMSE and skill over the draws, every draw plotted.
2. Primary (confirmatory): Spearman rho(alpha, RMSE) over the 15 runs, two-sided
   permutation p-value. rho(alpha, skill) alongside.
3. Support rule for "RMSE follows how far the model deviates from the lookup, not
   alpha": |rho(spread, RMSE)| > |rho(alpha, RMSE)| and p(alpha, RMSE) >= 0.05; with
   rho(alpha, spread) reported. Spread = std(predicted residual) / std(actual residual)
   over the 28 test runs; the saved window-level spread is recorded next to it.
4. Between-alpha spread (sample std of the five per-alpha mean RMSEs) against
   within-alpha spread (mean of the five per-alpha sample stds): within >= between means
   the differences are dominated by the particular draw, not alpha.
5. Exploratory: per-run partition descriptors and the split of model SSE - lookup SSE
   into testing_20251124:007 and the other identities, each against RMSE. Post hoc (from
   the step-1 diagnosis of these two runs): identities whose training runs all sit on one
   client that is not the largest. Never part of the confirmatory test.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from . import s5_queue
from .regression_metrics import report_from_run_rows
from .s5_queue import Job, MEASURED_OUTCOMES, read_job_record

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "artifacts/alpha_sweep"
RELABEL_ROOT = ROOT / "artifacts/relabel"
SPLIT_SEED = 4601
INIT_SEED = 4601
CONTROL_INIT_SEED = 4602
ALPHAS = (0.05, 0.1, 0.3, 1.0, 10.0)
#: Dirichlet partition seeds of draws 1, 2, 3. 4601 = the split seed = the relabel partition.
PARTITION_SEEDS = (4601, 5601, 6601)
#: Draw 1 at init 4601 of these alphas is the relabel run; their init-4602 runs are the controls.
RELABEL_ALPHAS = (0.1, 0.3)
FOCUS_IDENTITY = "testing_20251124:007"
NUM_TRAINING_RUNS = 112
NUM_TEST_RUNS = 28
#: First-run estimate: the longest relabel FL regression wall time (31.8 min), rounded up.
STATIC_MINUTES = 32.0
PERMUTATIONS = 100_000
PERMUTATION_SEED = 4601
SIGNIFICANCE = 0.05
RUN_TO_CLIENT = "combined_iid_run_to_client.csv"


def condition_name(alpha: float) -> str:
    return f"k3_dirichlet{alpha:g}"


def config_path(alpha: float) -> Path:
    return ROOT / f"configs/alpha_sweep_{condition_name(alpha)}.yaml"


def draw_of(partition_seed: int) -> int:
    return PARTITION_SEEDS.index(partition_seed) + 1


# --------------------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepJob(Job):
    """An S5 ``Job`` (``seed`` = init seed) plus alpha, partition seed and role."""

    alpha: float = 0.0
    partition_seed: int = SPLIT_SEED
    role: str = "confirmatory"  # confirmatory | init_control


def sweep_job_id(alpha: float, partition_seed: int, init_seed: int) -> str:
    return f"fl/{condition_name(alpha)}/partition{partition_seed}/init{init_seed}/regression"


def build_jobs() -> list[SweepJob]:
    """The 15 new jobs: (1) draw 1 of the new alphas, (2) draw 2 and (3) draw 3 of every
    alpha, then (4) the two initialization controls. A partial queue still covers every
    alpha before it adds draws."""
    jobs: list[SweepJob] = []

    def add(group: int, alpha: float, partition_seed: int, init_seed: int, role: str) -> None:
        jobs.append(SweepJob(sweep_job_id(alpha, partition_seed, init_seed), group, "fl", condition_name(alpha),
                             init_seed, "regression", alpha=alpha, partition_seed=partition_seed, role=role))

    for alpha in ALPHAS:
        if alpha not in RELABEL_ALPHAS:
            add(1, alpha, PARTITION_SEEDS[0], INIT_SEED, "confirmatory")
    for group, partition_seed in ((2, PARTITION_SEEDS[1]), (3, PARTITION_SEEDS[2])):
        for alpha in ALPHAS:
            add(group, alpha, partition_seed, INIT_SEED, "confirmatory")
    for alpha in RELABEL_ALPHAS:
        add(4, alpha, PARTITION_SEEDS[0], CONTROL_INIT_SEED, "init_control")
    return jobs


def command(job: SweepJob, root: Path, python: str) -> list[str]:
    return [python, str(ROOT / "scripts/run_combined_iid_flower.py"),
            "--config", str(config_path(job.alpha)), "--task", job.task,
            "--output-root", str(job.directory(root)),
            "--set", f"init_seed={job.seed}", "--set", f"partition_seed={job.partition_seed}"]


def estimate_minutes(job: SweepJob, root: Path, jobs: Sequence[SweepJob]) -> float:
    """Longest measured wall time of a finished sweep job, else the relabel maximum."""
    measured = [float(r["wall_seconds"]) / 60.0 for other in jobs
                if (r := read_job_record(other, root)) and r.get("outcome") in MEASURED_OUTCOMES]
    return max(measured) if measured else STATIC_MINUTES


# --------------------------------------------------------------------------------------
# the runs the analysis reads (13 + 2 confirmatory reused from the relabel, 2 controls)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepRun:
    alpha: float
    partition_seed: int
    init_seed: int
    role: str
    job_dir: Path  # holds job.json and partitions/
    reused: bool

    @property
    def draw(self) -> int:
        return draw_of(self.partition_seed)

    @property
    def result_dir(self) -> Path:
        return self.job_dir / "regression"


def sweep_runs(root: Path = DEFAULT_RESULTS_ROOT, relabel_root: Path = RELABEL_ROOT) -> list[SweepRun]:
    """Confirmatory runs by draw then alpha, then the controls (the workbook row order)."""
    runs = []
    for partition_seed in PARTITION_SEEDS:
        for alpha in ALPHAS:
            if partition_seed == PARTITION_SEEDS[0] and alpha in RELABEL_ALPHAS:
                job_dir = Path(relabel_root) / f"fl/{condition_name(alpha)}/seed{INIT_SEED}/regression"
                runs.append(SweepRun(alpha, partition_seed, INIT_SEED, "confirmatory", job_dir, True))
            else:
                job_dir = Path(root) / sweep_job_id(alpha, partition_seed, INIT_SEED)
                runs.append(SweepRun(alpha, partition_seed, INIT_SEED, "confirmatory", job_dir, False))
    for alpha in RELABEL_ALPHAS:
        job_dir = Path(root) / sweep_job_id(alpha, PARTITION_SEEDS[0], CONTROL_INIT_SEED)
        runs.append(SweepRun(alpha, PARTITION_SEEDS[0], CONTROL_INIT_SEED, "init_control", job_dir, False))
    return runs


# --------------------------------------------------------------------------------------
# per-run descriptors
# --------------------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def identity_of_run(run_uid: str) -> str:
    """``test_2:001:004`` → ``test_2:001``: the (source, subject) identity of a run."""
    source, subject, _ = run_uid.split(":")
    return f"{source}:{subject}"


def prediction_descriptors(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """RMSE, lookup RMSE, skill (bootstrap CI), the matched RMSE CI, the 28-run spread and
    the split of model SSE - lookup SSE into the focus identity and the rest."""
    from .excel_figures import matched_bootstrap

    if len(rows) != NUM_TEST_RUNS:
        raise AssertionError(f"expected {NUM_TEST_RUNS} test runs, got {len(rows)}")
    overall = report_from_run_rows(rows, "source_subject_train_mean_mps")["overall"]
    actual = np.asarray([float(r["actual_speed_mps"]) for r in rows])
    predicted = np.asarray([float(r["predicted_speed_mps"]) for r in rows])
    baseline = np.asarray([float(r["source_subject_train_mean_mps"]) for r in rows])
    predicted_residual = np.asarray([float(r["predicted_residual_mps"]) for r in rows])
    actual_residual = np.asarray([float(r["actual_residual_mps"]) for r in rows])
    focus = np.asarray([f"{r['source_id']}:{r['subject_id']}" == FOCUS_IDENTITY for r in rows])
    if not focus.any():
        raise AssertionError(f"{FOCUS_IDENTITY} has no test run")
    excess = (predicted - actual) ** 2 - (baseline - actual) ** 2
    boot = matched_bootstrap(actual, predicted, baseline)
    return {
        "rmse": overall["model_rmse_mps"], "lookup_rmse": overall["baseline_rmse_mps"],
        "rmse_ci_low": boot["rmse_ci_low"], "rmse_ci_high": boot["rmse_ci_high"],
        "skill": overall["skill"], "skill_ci_low": overall["skill_ci95"][0], "skill_ci_high": overall["skill_ci95"][1],
        # np.std (ddof 0) on both sides, as guardrails.check_regression_health computes the saved spread
        "spread_runs": float(np.std(predicted_residual) / np.std(actual_residual)),
        "dsse_focus": float(excess[focus].sum()), "dsse_other": float(excess[~focus].sum()),
    }


def partition_descriptors(run_to_client: Mapping[str, str]) -> dict[str, Any]:
    """Client run-count imbalance, each identity's share of training runs on the largest
    client, where the focus identity sits, and (post hoc) the identities held entirely by
    one client that is not the largest. The largest client is the one with the most
    training runs (ties: the lowest client ID)."""
    if len(run_to_client) != NUM_TRAINING_RUNS:
        raise AssertionError(f"expected {NUM_TRAINING_RUNS} training runs, got {len(run_to_client)}")
    totals = Counter(run_to_client.values())
    clients = sorted(totals)
    largest = max(clients, key=lambda c: totals[c])
    counts: dict[str, Counter[str]] = {}
    for run_uid, client in run_to_client.items():
        counts.setdefault(identity_of_run(run_uid), Counter())[client] += 1
    share = {identity: held[largest] / sum(held.values()) for identity, held in sorted(counts.items())}
    isolated = sorted(i for i, held in counts.items() if len(held) == 1 and largest not in held)
    sizes = np.asarray([totals[c] for c in clients], dtype=float)
    focus = counts.get(FOCUS_IDENTITY, Counter())
    return {
        "client_runs": "/".join(str(totals[c]) for c in clients),
        "largest_client": largest,
        "largest_client_share": totals[largest] / NUM_TRAINING_RUNS,
        "client_run_cv": float(sizes.std() / sizes.mean()),
        "mean_identity_share_on_largest": float(np.mean(list(share.values()))),
        **{f"share_on_largest:{identity}": value for identity, value in share.items()},
        "focus_share_on_largest": share.get(FOCUS_IDENTITY, math.nan),
        "focus_num_clients": len(focus),
        "focus_clients": json.dumps(dict(sorted(focus.items()))),
        "posthoc_isolated_identities": len(isolated),
        "posthoc_isolated_list": " ".join(isolated),
        "identity_client_runs": json.dumps({i: dict(sorted(h.items())) for i, h in sorted(counts.items())}),
    }


def _saved_spread(result_dir: Path) -> float | None:
    for name in ("training_summary.json", "degenerate_model_check_FAILED.json"):
        path = result_dir / name
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("degenerate_model_check", data).get("relative_output_spread")
    return None


def run_row(run: SweepRun) -> dict[str, Any]:
    """One analysis row, or a row with only the status when the run is not measured."""
    record = _job_record(run.job_dir)
    base = {"role": run.role, "alpha": run.alpha, "draw": run.draw, "partition_seed": run.partition_seed,
            "init_seed": run.init_seed, "reused_from_relabel": run.reused,
            "status": (record or {}).get("outcome", "pending")}
    if not record or record.get("outcome") not in MEASURED_OUTCOMES:
        return base
    predictions = run.result_dir / "test_run_predictions.csv"
    partition = run.job_dir / "partitions" / RUN_TO_CLIENT
    assignment = {r["run_uid"]: r["client_id"] for r in _read_csv(partition)}
    rows = _read_csv(predictions)
    if set(assignment) & {r["run_uid"] for r in rows}:
        raise AssertionError(f"{partition}: a test run is assigned to a client")
    return {**base, "outcome": (record.get("metrics") or {}).get("training_outcome", ""),
            **prediction_descriptors(rows), "spread_saved_window": _saved_spread(run.result_dir),
            **partition_descriptors(assignment),
            "predictions_file": _relative(predictions), "partition_file": _relative(partition)}


def _job_record(job_dir: Path) -> dict[str, Any] | None:
    path = Path(job_dir) / "job.json"
    return json.loads(path.read_text()) if path.exists() else None


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------
# pre-registered statistics
# --------------------------------------------------------------------------------------

#: (column, label, kind). kind: model (from the predictions), partition, posthoc.
DESCRIPTORS = (
    ("spread_runs", "Prediction spread (28 runs)", "model"),
    ("dsse_focus", f"Model − lookup SSE, {FOCUS_IDENTITY}", "model"),
    ("dsse_other", "Model − lookup SSE, other identities", "model"),
    ("largest_client_share", "Largest client's share of training runs", "partition"),
    ("client_run_cv", "CV of client run counts", "partition"),
    ("mean_identity_share_on_largest", "Mean identity share on the largest client", "partition"),
    ("focus_share_on_largest", f"{FOCUS_IDENTITY}: share on the largest client", "partition"),
    ("focus_num_clients", f"{FOCUS_IDENTITY}: clients holding it", "partition"),
    ("posthoc_isolated_identities", "Identities all on one non-largest client (post hoc)", "posthoc"),
)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation of average ranks; nan when either side is constant."""
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p(x: Sequence[float], y: Sequence[float], permutations: int = PERMUTATIONS,
                  seed: int = PERMUTATION_SEED) -> float:
    """Two-sided Monte Carlo permutation p-value of Spearman rho: y's ranks shuffled
    ``permutations`` times (generator seed ``seed``), p = (1 + #|rho*| >= |rho|) / (1 + N)."""
    observed = spearman(x, y)
    if math.isnan(observed):
        return math.nan
    rx, ry = rankdata(x), rankdata(y)
    shuffled = np.random.default_rng(seed).permuted(np.tile(ry, (permutations, 1)), axis=1)
    cx = rx - rx.mean()
    cy = shuffled - shuffled.mean(axis=1, keepdims=True)
    rho = (cy @ cx) / (np.sqrt((cy ** 2).sum(axis=1)) * np.sqrt((cx ** 2).sum()))
    hits = int(np.sum(np.abs(rho) >= abs(observed) - 1e-12))
    return (1 + hits) / (1 + permutations)


def _correlation(rows: Sequence[Mapping[str, Any]], x: str, y: str) -> dict[str, Any]:
    pairs = [(r[x], r[y]) for r in rows if _number(r.get(x)) and _number(r.get(y))]
    if len(pairs) < 3:
        return {"n": len(pairs), "rho": None, "p": None}
    xs, ys = zip(*pairs)
    rho = spearman(xs, ys)
    return {"n": len(pairs), "rho": None if math.isnan(rho) else rho,
            "p": None if math.isnan(rho) else permutation_p(xs, ys)}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


def per_alpha_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for alpha in ALPHAS:
        members = [r for r in rows if r["role"] == "confirmatory" and r["alpha"] == alpha and "rmse" in r]
        entry: dict[str, Any] = {"alpha": alpha, "n": len(members)}
        for key in ("rmse", "skill"):
            values = np.asarray([r[key] for r in members], dtype=float)
            entry[f"{key}_mean"] = float(values.mean()) if values.size else None
            entry[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else None
        out.append(entry)
    return out


def _core_statistics(confirmatory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Items 1-4 of the pre-registered analysis on the given confirmatory runs."""
    primary = _correlation(confirmatory, "alpha", "rmse")
    spread = _correlation(confirmatory, "spread_runs", "rmse")
    support = None
    if primary["rho"] is not None and spread["rho"] is not None:
        support = bool(abs(spread["rho"]) > abs(primary["rho"]) and primary["p"] >= SIGNIFICANCE)
    summary = per_alpha_summary(confirmatory)
    means = [s["rmse_mean"] for s in summary if s["rmse_mean"] is not None]
    stds = [s["rmse_std"] for s in summary if s["rmse_std"] is not None]
    variance_rule: dict[str, Any] = {"between_alpha_std_of_means": None, "within_alpha_mean_of_stds": None,
                                     "draw_dominates": None}
    if len(stds) == len(ALPHAS):
        between, within = float(np.std(means, ddof=1)), float(np.mean(stds))
        variance_rule = {"between_alpha_std_of_means": between, "within_alpha_mean_of_stds": within,
                         "draw_dominates": bool(within >= between)}
    return {"per_alpha": summary, "primary_alpha_vs_rmse": primary,
            "alpha_vs_skill": _correlation(confirmatory, "alpha", "skill"),
            "alpha_vs_spread": _correlation(confirmatory, "alpha", "spread_runs"),
            "spread_vs_rmse": spread, "spread_rule_supported": support, "variance_rule": variance_rule}


def analyse(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every pre-registered quantity, computed on whatever is measured. Conclusions are
    marked final only when all 15 confirmatory runs are measured.

    ``sensitivity_excluding_collapsed`` is NOT pre-registered: items 1-4 recomputed without
    the confirmatory runs whose gate outcome is collapsed, reported next to the primary
    result. The pre-registered statistics keep every measured run."""
    confirmatory = [r for r in rows if r["role"] == "confirmatory" and "rmse" in r]
    complete = len(confirmatory) == len(ALPHAS) * len(PARTITION_SEEDS)
    core = _core_statistics(confirmatory)
    collapsed = [r for r in confirmatory if r.get("outcome") == "collapsed"]
    kept = [r for r in confirmatory if r.get("outcome") != "collapsed"]
    sensitivity = {
        "pre_registered": False,
        "label": "NOT PRE-REGISTERED sensitivity check: items 1-4 without the collapsed confirmatory runs",
        "excluded": [f"alpha={r['alpha']:g} draw {r['draw']}" for r in collapsed],
        "n": len(kept), **_core_statistics(kept)}
    controls = []
    for control in (r for r in rows if r["role"] == "init_control"):
        partner = next((r for r in confirmatory if r["alpha"] == control["alpha"] and r["draw"] == control["draw"]), None)
        entry = {"alpha": control["alpha"], "draw": control["draw"], "status_init4602": control["status"]}
        for key in ("spread_runs", "spread_saved_window", "rmse", "skill", "outcome"):
            entry[f"{key}_init4601"] = None if partner is None else partner.get(key)
            entry[f"{key}_init4602"] = control.get(key)
        controls.append(entry)
    return {
        "complete": complete,
        "confirmatory_runs_measured": len(confirmatory),
        "confirmatory_runs_planned": len(ALPHAS) * len(PARTITION_SEEDS),
        "permutations": PERMUTATIONS, "permutation_seed": PERMUTATION_SEED, "significance": SIGNIFICANCE,
        **core,
        "sensitivity_excluding_collapsed": sensitivity,
        "descriptors_vs_rmse": {
            column: {"label": label, "kind": kind, **_correlation(confirmatory, column, "rmse")}
            for column, label, kind in DESCRIPTORS},
        "identity_share_vs_rmse": {
            column: _correlation(confirmatory, column, "rmse")
            for column in sorted({k for r in confirmatory for k in r if k.startswith("share_on_largest:")})},
        "init_controls": controls,
    }


# --------------------------------------------------------------------------------------
# index and derived tables
# --------------------------------------------------------------------------------------


def rebuild_index(jobs: Sequence[SweepJob], root: Path) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        record = read_job_record(job, root) or {}
        metrics = dict(record.get("metrics") or {})
        rows.append({"job_id": job.job_id, "group": job.group, "role": job.role, "alpha": job.alpha,
                     "partition_seed": job.partition_seed, "draw": draw_of(job.partition_seed), "init_seed": job.seed,
                     "status": record.get("outcome", "pending"), "flat": "FLAT" if record.get("flat") else "",
                     "training_outcome": metrics.pop("training_outcome", ""),
                     "wall_minutes": "" if record.get("wall_seconds") is None else round(record["wall_seconds"] / 60, 1),
                     "result_dir": str(job.result_dir(root)), **metrics})
    s5_queue._write_rows(root / "results_index.csv", rows,
                         ["job_id", "group", "role", "alpha", "partition_seed", "draw", "init_seed", "status", "flat",
                          "training_outcome", "wall_minutes", "result_dir"])
    return rows


def write_derived(root: Path, relabel_root: Path = RELABEL_ROOT) -> dict[str, Any]:
    rows = [run_row(run) for run in sweep_runs(root, relabel_root)]
    analysis = analyse(rows)
    s5_queue._write_rows(root / "derived/alpha_sweep_runs.csv", rows,
                         ["role", "alpha", "draw", "partition_seed", "init_seed", "status", "outcome", "rmse",
                          "lookup_rmse", "skill", "spread_runs", "spread_saved_window"])
    (root / "derived/alpha_sweep_analysis.json").write_text(json.dumps(analysis, indent=2))
    return analysis


#: Text columns of ``derived/alpha_sweep_runs.csv``; INT_COLUMNS are integers, every other
#: non-empty cell is a float. Empty cells are left out, as in ``run_row``.
TEXT_COLUMNS = {"role", "status", "outcome", "client_runs", "largest_client", "focus_clients",
                "posthoc_isolated_list", "identity_client_runs", "predictions_file", "partition_file"}
INT_COLUMNS = {"draw", "partition_seed", "init_seed", "focus_num_clients", "posthoc_isolated_identities"}


def load_derived(root: Path = DEFAULT_RESULTS_ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rows and the analysis exactly as ``write_derived`` saved them, typed back."""
    rows = []
    for raw in _read_csv(Path(root) / "derived/alpha_sweep_runs.csv"):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value == "":
                continue
            if key in TEXT_COLUMNS:
                row[key] = value
            elif key == "reused_from_relabel":
                row[key] = {"True": True, "False": False}[value]
            elif key in INT_COLUMNS:
                row[key] = int(value)
            else:
                row[key] = float(value)
        rows.append(row)
    analysis = json.loads((Path(root) / "derived/alpha_sweep_analysis.json").read_text())
    return rows, analysis


# --------------------------------------------------------------------------------------
# workbook
# --------------------------------------------------------------------------------------


DERIVED_FILES = ("artifacts/alpha_sweep/derived/alpha_sweep_runs.csv",
                 "artifacts/alpha_sweep/derived/alpha_sweep_analysis.json")


def centralized_reference(artifacts_root: Path) -> list[dict[str, Any]]:
    """The centralized residual regression model at its three init seeds, read from the
    relabel / S5 results exactly as Fig 2 reads it: the reference lines of the workbook."""
    from .excel_figures import load_relabel_results, regression_points

    points = sorted((p for p in regression_points(load_relabel_results(artifacts_root))
                     if p["family"] == "centralized"), key=lambda p: int(p["seed"]))
    if [int(p["seed"]) for p in points] != [4601, 4602, 4603]:
        raise AssertionError(f"expected the centralized model at seeds 4601-4603, got {[p['seed'] for p in points]}")
    return [{"seed": int(p["seed"]), "rmse": p["rmse"], "skill": p["skill"], "outcome": p["outcome"],
             "source_file": p["predictions_file"]} for p in points]


def build_workbook(rows: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any], commit: str,
                   centralized: Sequence[Mapping[str, Any]] | None = None):
    """``excel/alpha_sweep.xlsx``: the excel/ conventions (data / chart / stats / notes).

    Rows are ordered confirmatory first; within them, runs that did not train or collapsed
    come last so each draw's trained runs and each non-trained outcome form one block (one
    chart series each). That is display only: the per-α formulas, the rank formulas and
    the stats sheet cover every confirmatory row, the collapsed run included."""
    from openpyxl.chart.marker import DataPoint

    from . import excel_figures as ef

    roles = {"confirmatory": 0, "init_control": 1}
    measured = sorted((dict(r) for r in rows if "rmse" in r),
                      key=lambda r: (roles[r["role"]], r["outcome"] != "trained", r["outcome"], r["draw"], r["alpha"]))
    identities = sorted({k.split(":", 1)[1] for r in measured for k in r if k.startswith("share_on_largest:")})
    wb = ef._new_workbook()
    data_rows = []
    for r in measured:
        data_rows.append({**r, "value": r["rmse"], "ci_lower": r["rmse_ci_low"], "ci_upper": r["rmse_ci_high"],
                          "skill_ci_lower": r["skill_ci_low"], "skill_ci_upper": r["skill_ci_high"],
                          "reused_from_relabel": "yes" if r["reused_from_relabel"] else "no"})
    columns = ["role", "alpha", "draw", "partition_seed", "init_seed", "status", "outcome", "value", "ci_lower",
               "ci_upper", "err_plus", "err_minus", "lookup_rmse", "skill", "skill_ci_lower", "skill_ci_upper", "skill_err_plus",
               "skill_err_minus", "spread_runs", "spread_saved_window", "dsse_focus", "dsse_other", "client_runs",
               "largest_client_share", "client_run_cv", "mean_identity_share_on_largest",
               *[f"share_on_largest:{i}" for i in identities], "focus_share_on_largest", "focus_num_clients",
               "posthoc_isolated_identities", "posthoc_isolated_list", "rank_alpha", "rank_rmse", "rank_spread",
               "reused_from_relabel", "predictions_file", "partition_file"]
    confirmatory = [i for i, r in enumerate(data_rows) if r["role"] == "confirmatory"]
    if confirmatory != list(range(len(confirmatory))):
        raise AssertionError("confirmatory rows must come first")
    n_conf = len(confirmatory)

    probe = ef.Table(wb["data"], columns, 1, len(data_rows))

    def rank(source: str) -> str:
        """Average rank among the confirmatory rows (ties averaged, as Spearman needs).
        SUMPRODUCT compares the numbers exactly; COUNTIF's "<"&x criterion would round x
        to 15 significant digits as text and can miscount full-precision values."""
        col = probe.letter(source)
        block = f"${col}$2:${col}${1 + n_conf}"
        x = "{" + source + "}"
        return (f'=IF({{role}}="confirmatory",SUMPRODUCT(--({block}<{x}))'
                f'+(SUMPRODUCT(--({block}={x}))+1)/2,"")')

    formulas = {"skill_err_plus": "={skill_ci_upper}-{skill}", "skill_err_minus": "={skill}-{skill_ci_lower}"}
    for key, source in (("rank_alpha", "alpha"), ("rank_rmse", "value"), ("rank_spread", "spread_runs")):
        formulas[key] = rank(source)
    table = ef.write_table(wb["data"], data_rows, columns, formulas=formulas)
    rng = {k: table.ref(k) for k in ("role", "alpha", "value", "skill", "lookup_rmse")}

    # per-alpha summary block, right of the table: formulas over the confirmatory rows
    s_columns = ["summary_alpha", "summary_n", "summary_rmse_mean", "summary_rmse_std", "summary_skill_mean",
                 "summary_skill_std"]
    present = [s["alpha"] for s in analysis["per_alpha"] if s["n"] > 0]
    s_table = ef.Table(wb["data"], s_columns, 1, len(present), len(columns) + 3)
    summary = []
    for i, alpha in enumerate(present):
        cell = {k: s_table.cell(k, i, absolute=False) for k in s_columns}
        where = f'{rng["role"]},"confirmatory",{rng["alpha"]},{cell["summary_alpha"]}'
        mask = f'({rng["role"]}="confirmatory")*({rng["alpha"]}={cell["summary_alpha"]})'
        row: dict[str, Any] = {"summary_alpha": alpha, "summary_n": f"=COUNTIFS({where})"}
        for key, column in (("rmse", "value"), ("skill", "skill")):
            mean = cell[f"summary_{key}_mean"]
            row[f"summary_{key}_mean"] = f'=AVERAGEIFS({rng[column]},{where})'
            row[f"summary_{key}_std"] = (f'=IF({cell["summary_n"]}>1,SQRT(SUMPRODUCT({mask}*({rng[column]}-{mean})^2)'
                                         f'/({cell["summary_n"]}-1)),0)')
        summary.append(row)
    ef.write_table(wb["data"], summary, s_columns, first_col=s_table.first_col)
    # reference lines across the alpha axis: lookup, zero, and the centralized residual
    # model (mean, min and max over its 3 init seeds, formulas over the block below them)
    ref_first = s_table.first_col
    ref_top = len(present) + 4
    ws = wb["data"]
    line_values: dict[str, Any] = {"lookup_rmse_line": f"=AVERAGE({rng['lookup_rmse']})", "zero_line": 0.0}
    if centralized:
        c_table = ef.write_table(
            ws, [{"centralized_seed": c["seed"], "centralized_rmse": c["rmse"], "centralized_skill": c["skill"],
                  "centralized_outcome": c["outcome"], "centralized_source_file": c["source_file"]}
                 for c in centralized],
            ["centralized_seed", "centralized_rmse", "centralized_skill", "centralized_outcome",
             "centralized_source_file"], header_row=ref_top + 4, first_col=ref_first)
        for metric in ("rmse", "skill"):
            for stat, function in (("mean", "AVERAGE"), ("min", "MIN"), ("max", "MAX")):
                line_values[f"centralized_{metric}_{stat}"] = f"={function}({c_table.ref(f'centralized_{metric}')})"
    line_keys = ["line_alpha", *line_values]
    for j, key in enumerate(line_keys):
        ws.cell(row=ref_top, column=ref_first + j, value=key).font = ef._font(bold=True)
    for k, x in enumerate((0.01, 100.0)):
        r = ref_top + 1 + k
        ws.cell(row=r, column=ref_first, value=x).font = ef._font()
        for j, key in enumerate(line_values, start=1):
            ws.cell(row=r, column=ref_first + j, value=line_values[key]).font = ef._font()
    sheet = ef.quote_sheetname(ws.title)
    lines_x = f"{sheet}!${ef.get_column_letter(ref_first)}${ref_top + 1}:${ef.get_column_letter(ref_first)}${ref_top + 2}"

    def line_ref(key: str) -> str:
        col = ef.get_column_letter(ref_first + line_keys.index(key))
        return f"{sheet}!${col}${ref_top + 1}:${col}${ref_top + 2}"

    def block(**match: Any) -> tuple[int, int] | None:
        """The contiguous data rows matching ``match``, or None when there are none."""
        if not any(all(r.get(k) == v for k, v in match.items()) for r in data_rows):
            return None
        return table.rows_where(data_rows, **match)

    def collapsed_points(stop: int) -> list[Any]:
        """Per-point × markers for the non-trained rows among data rows 0 .. stop - 1."""
        return [DataPoint(idx=i, marker=ef._marker(ef.OUTCOME_SYMBOL[r["outcome"]], ef.COLLAPSED_COLOR, False, 9))
                for i, r in enumerate(data_rows[:stop]) if r["outcome"] in ef.OUTCOME_SYMBOL]

    charts = ef._sheet(wb, "chart")
    placed = 0

    def place(chart: Any) -> None:
        nonlocal placed
        ef._size(chart, 16, 10)
        charts.add_chart(chart, f"{'A' if placed % 2 == 0 else 'K'}{1 + 21 * (placed // 2)}")
        placed += 1

    for metric, y_title, column, err_plus, err_minus, line_key, line_name in (
            ("RMSE", "RMSE (m/s), matched bootstrap 95% CI", "value", "err_plus", "err_minus", "lookup_rmse_line",
             "lookup (0.0405 m/s)"),
            ("Skill", "Skill vs lookup, bootstrap 95% CI", "skill", "skill_err_plus", "skill_err_minus", "zero_line",
             "zero")):
        chart = ef._scatter(f"{metric} vs Dirichlet α (FL K=3, init seed 4601)", "Dirichlet α (log scale)", y_title)

        def points(span: tuple[int, int], title: str, marker: Any, color: str) -> None:
            a, b = span
            s = ef._markers_only(ef._series(table.ref(column, a, b), table.ref("alpha", a, b), title=title), marker)
            s.errBars = ef._custom_error_bars(table, a, b, direction="y", plus=err_plus, minus=err_minus, color=color)
            chart.series.append(s)

        for d, color in zip((1, 2, 3), ef.SEED_COLORS):
            if span := block(role="confirmatory", outcome="trained", draw=d):
                points(span, f"draw {d} (partition seed {PARTITION_SEEDS[d - 1]})",
                       ef._marker(("circle", "square", "triangle")[d - 1], color, True, 7), color)
        for outcome in ("did not train", "collapsed"):
            if span := block(role="confirmatory", outcome=outcome):
                which = "; ".join(f"α={r['alpha']:g} draw {r['draw']}" for r in data_rows[span[0]:span[1]])
                points(span, f"{outcome} ({which}), in every statistic",
                       ef._marker(ef.OUTCOME_SYMBOL[outcome], ef.COLLAPSED_COLOR, False, 9), ef.COLLAPSED_COLOR)
        if span := block(role="init_control"):
            points(span, "init-seed 4602 controls (same partitions as draw 1)",
                   ef._marker("diamond", ef.LOCAL_COLOR, False, 8), ef.LOCAL_COLOR)
        if summary:
            mean = ef._series(s_table.ref(f"summary_{metric.lower()}_mean"), s_table.ref("summary_alpha"),
                              title="mean ± std over draws (all 3, collapsed included)")
            mean.marker = ef._marker("dash", ef.INK, True, 10)
            mean.graphicalProperties = ef._line(ef.INK, 1.0, "dash")
            mean.smooth = False
            mean.errBars = ef._custom_error_bars(s_table, direction="y", plus=f"summary_{metric.lower()}_std",
                                                 minus=f"summary_{metric.lower()}_std", color=ef.INK)
            chart.series.append(mean)
        chart.series.append(ef._line_only(ef._series(line_ref(line_key), lines_x, title=line_name), ef.MUTED, 0.75, "dash"))
        deleted = []
        if centralized:
            key = f"centralized_{metric.lower()}"
            chart.series.append(ef._line_only(ef._series(line_ref(f"{key}_mean"), lines_x,
                                                         title="centralized residual (mean of 3 seeds)"),
                                              ef.CENTRAL_COLOR, 1.25, "dash"))
            for stat in ("min", "max"):
                chart.series.append(ef._line_only(ef._series(line_ref(f"{key}_{stat}"), lines_x,
                                                             title="centralized min–max over 3 seeds"),
                                                  ef.CENTRAL_COLOR, 0.75, "sysDot"))
            deleted.append(len(chart.series) - 1)  # one legend entry for the band's two lines
        ef._style_axis(chart.x_axis, "Dirichlet α (log scale)", log=True, minimum=0.01, maximum=100, gridlines=False,
                       number_format="0.##")
        chart.x_axis.crosses = "min"
        ef._legend(chart, deleted=deleted)
        place(chart)

    conf_end = n_conf
    controls = [i for i, r in enumerate(data_rows) if r["role"] == "init_control"]

    def descriptor_chart(column: str, x_title: str, y_column: str = "value", y_title: str = "RMSE (m/s)",
                         title: str | None = None) -> None:
        chart = ef._scatter(title or f"RMSE vs {x_title}", x_title, y_title)
        if conf_end:
            s = ef._markers_only(
                ef._series(table.ref(y_column, 0, conf_end), table.ref(column, 0, conf_end),
                           title="confirmatory runs (× = collapsed)"),
                ef._marker("circle", ef.FL_COLOR, True, 7))
            s.dPt = collapsed_points(conf_end)
            chart.series.append(s)
        if controls:
            a, b = controls[0], controls[-1] + 1
            chart.series.append(ef._markers_only(
                ef._series(table.ref(y_column, a, b), table.ref(column, a, b), title="init-seed 4602 controls"),
                ef._marker("diamond", ef.LOCAL_COLOR, False, 8)))
        place(chart)

    descriptor_chart("spread_runs", "prediction spread (28 runs)",
                     title="RMSE vs prediction spread, the pre-registered descriptor (15 runs + 2 controls)")
    chart = ef._scatter("Gain on testing_20251124:007 vs loss elsewhere", "Model − lookup SSE, testing_20251124:007",
                        "Model − lookup SSE, other identities")
    if conf_end:
        s = ef._markers_only(
            ef._series(table.ref("dsse_other", 0, conf_end), table.ref("dsse_focus", 0, conf_end),
                       title="confirmatory runs (× = collapsed)"),
            ef._marker("circle", ef.FL_COLOR, True, 7))
        s.dPt = collapsed_points(conf_end)
        chart.series.append(s)
    place(chart)
    for column, label in (("largest_client_share", "largest client's share of training runs"),
                          ("client_run_cv", "CV of client run counts"),
                          ("mean_identity_share_on_largest", "mean identity share on the largest client"),
                          ("focus_share_on_largest", "testing_20251124:007 share on the largest client"),
                          ("posthoc_isolated_identities", "identities on one non-largest client (post hoc)")):
        descriptor_chart(column, label)
    chart = ef._scatter("RMSE vs each identity's share on the largest client", "Identity's share of its runs on the largest client",
                        "RMSE (m/s)")
    for j, identity in enumerate(identities):
        if conf_end:
            s = ef._markers_only(ef._series(table.ref("value", 0, conf_end), table.ref(f"share_on_largest:{identity}", 0, conf_end),
                                            title=identity),
                                 ef._marker(ef.IDENTITY_SYMBOLS[j], ef.IDENTITY_COLORS[j], identity.startswith("test_2"), 6))
            chart.series.append(s)
    place(chart)

    stats = ef._sheet(wb, "stats")
    stats.column_dimensions["A"].width = 58
    for col in "BCDE":
        stats.column_dimensions[col].width = 16
    for j, head in enumerate(("quantity", "rho / value / mean", "p (permutation) / std", "n", "live formula")):
        stats.cell(row=1, column=1 + j, value=head).font = ef._font(bold=True)
    line = 2

    def put(name: str, value: Any = None, p: Any = None, n: Any = None, formula: str | None = None) -> None:
        nonlocal line
        stats.cell(row=line, column=1, value=name).font = ef._font()
        for j, v in enumerate((value, p, n), start=2):
            if isinstance(v, bool):
                v = "yes" if v else "no"
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                cell = stats.cell(row=line, column=j, value=v)
                cell.font = ef._font()
                if isinstance(v, float):
                    cell.number_format = "0.0000"
        if formula:
            stats.cell(row=line, column=5, value=formula).font = ef._font()
        line += 1

    def corr(name: str, entry: Mapping[str, Any], formula: str | None = None) -> None:
        put(name, entry.get("rho"), entry.get("p"), entry.get("n"), formula)

    def ranks(column: str) -> str:
        return table.ref(column, 0, conf_end) if conf_end else ""

    live = conf_end >= 3
    put(f"confirmatory runs measured (of {analysis['confirmatory_runs_planned']})", analysis["confirmatory_runs_measured"])
    put("conclusions final (all 15 confirmatory runs measured)", analysis["complete"])
    corr("PRIMARY: Spearman ρ(α, RMSE)", analysis["primary_alpha_vs_rmse"],
         f"=CORREL({ranks('rank_alpha')},{ranks('rank_rmse')})" if live else None)
    corr("Spearman ρ(α, skill)", analysis["alpha_vs_skill"])
    corr("Spearman ρ(spread, RMSE)", analysis["spread_vs_rmse"],
         f"=CORREL({ranks('rank_spread')},{ranks('rank_rmse')})" if live else None)
    corr("Spearman ρ(α, spread)", analysis["alpha_vs_spread"])
    put("support rule: |ρ(spread,RMSE)| > |ρ(α,RMSE)| and p(α,RMSE) ≥ 0.05", analysis["spread_rule_supported"])
    rule = analysis["variance_rule"]
    means_ref = s_table.ref("summary_rmse_mean") if summary else None
    stds_ref = s_table.ref("summary_rmse_std") if summary else None
    put("between-α spread: sample std of the five per-α mean RMSEs", rule["between_alpha_std_of_means"],
        formula=f"=STDEV({means_ref})" if summary and len(summary) == len(ALPHAS) else None)
    put("within-α spread: mean of the five per-α RMSE stds", rule["within_alpha_mean_of_stds"],
        formula=f"=AVERAGE({stds_ref})" if summary and len(summary) == len(ALPHAS) else None)
    put("variance rule: within ≥ between → dominated by the draw, not α", rule["draw_dominates"])
    for s in analysis["per_alpha"]:
        put(f"per α, every confirmatory run: α={s['alpha']:g} RMSE mean / std (n = {s['n']})", s["rmse_mean"],
            s["rmse_std"])
        put(f"per α, every confirmatory run: α={s['alpha']:g} skill mean / std (n = {s['n']})", s["skill_mean"],
            s["skill_std"])
    line += 1
    sens = analysis["sensitivity_excluding_collapsed"]
    excluded = ", ".join(sens["excluded"]) or "none"
    put(f"SENSITIVITY, NOT PRE-REGISTERED: collapsed runs excluded ({excluded}); the rows above keep them")
    put("  confirmatory runs used", sens["n"])
    corr("  Spearman ρ(α, RMSE)", sens["primary_alpha_vs_rmse"])
    corr("  Spearman ρ(α, skill)", sens["alpha_vs_skill"])
    corr("  Spearman ρ(spread, RMSE)", sens["spread_vs_rmse"])
    corr("  Spearman ρ(α, spread)", sens["alpha_vs_spread"])
    put("  support rule", sens["spread_rule_supported"])
    put("  between-α spread", sens["variance_rule"]["between_alpha_std_of_means"])
    put("  within-α spread", sens["variance_rule"]["within_alpha_mean_of_stds"])
    put("  variance rule: within ≥ between", sens["variance_rule"]["draw_dominates"])
    for s in sens["per_alpha"]:
        put(f"  α={s['alpha']:g}: RMSE mean / std (n = {s['n']})", s["rmse_mean"], s["rmse_std"])
        put(f"  α={s['alpha']:g}: skill mean / std (n = {s['n']})", s["skill_mean"], s["skill_std"])
    line += 1
    put("Exploratory descriptors vs RMSE (not confirmatory)")
    for column, entry in analysis["descriptors_vs_rmse"].items():
        corr(f"  {entry['label']} [{entry['kind']}]", entry)
    for column, entry in analysis["identity_share_vs_rmse"].items():
        corr(f"  {column.split(':', 1)[1]}: share on the largest client [partition]", entry)
    line += 1
    put("Initialization controls (α, draw 1): init 4601 vs init 4602")
    for c in analysis["init_controls"]:
        put(f"  α={c['alpha']:g}: spread (28 runs), init 4601 / 4602", c["spread_runs_init4601"], c["spread_runs_init4602"])
        put(f"  α={c['alpha']:g}: saved window spread, init 4601 / 4602", c["spread_saved_window_init4601"],
            c["spread_saved_window_init4602"])
        put(f"  α={c['alpha']:g}: RMSE, init 4601 / 4602", c["rmse_init4601"], c["rmse_init4602"])
        put(f"  α={c['alpha']:g}: skill, init 4601 / 4602", c["skill_init4601"], c["skill_init4602"])

    missing = [f"α={r['alpha']:g} draw {r['draw']} init {r['init_seed']}: {r['status']}" for r in rows if "rmse" not in r]
    ef.write_notes(
        wb, ef.Build(commit=commit, results=None),  # type: ignore[arg-type]
        "Alpha sweep: FL K=3 Dirichlet regression",
        "RMSE and skill of FL regression across Dirichlet α ∈ {0.05, 0.1, 0.3, 1, 10}, three partition draws "
        "per α (partition seeds 4601 / 5601 / 6601; draw 1 is the relabel partition), init seed 4601, plus "
        "two initialization controls (α = 0.1 and 0.3, draw 1, init seed 4602) that are not in the "
        "confirmatory set. Per-run descriptors and the pre-registered statistics are on the data and stats "
        "sheets. Reference lines: the lookup (0.0405 m/s; skill 0) and the centralized residual model, its "
        "mean over init seeds 4601 / 4602 / 4603 with the min–max over those seeds as two dotted lines (an "
        "Excel scatter chart cannot shade a band). Read from artifacts/alpha_sweep/derived/ and, for the "
        "centralized lines only, the relabel / S5 results.",
        ef._columns_doc(table.columns, {
            "role": "confirmatory (the 15-run set) or init_control (init seed 4602; never in the statistics).",
            "alpha": "Dirichlet concentration α.",
            "draw": "Partition draw 1 / 2 / 3.",
            "partition_seed": "Dirichlet partition seed (4601 = split seed = the relabel partition; 5601; 6601).",
            "init_seed": "Initialization seed.",
            "status": "Queue outcome.",
            "outcome": "Regression gate label: trained / did not train / collapsed. A label, never an exclusion.",
            "value": "Test RMSE (m/s) over the 28 test runs (run-mean speed).",
            "ci_lower": "RMSE matched-bootstrap 95% interval, lower end.",
            "ci_upper": "RMSE matched-bootstrap 95% interval, upper end.",
            "lookup_rmse": "Lookup RMSE (identity train-run mean): 0.0405 m/s on this split.",
            "skill": "1 − SSE_model / SSE_lookup.",
            "skill_ci_lower": "Skill bootstrap 95% interval, lower end.",
            "skill_ci_upper": "Skill bootstrap 95% interval, upper end.",
            "skill_err_plus": "Formula: skill_ci_upper − skill.",
            "skill_err_minus": "Formula: skill − skill_ci_lower.",
            "spread_runs": "Prediction spread: std(predicted residual) / std(actual residual) over the 28 test "
                           "runs (population std). The descriptor of the support rule.",
            "spread_saved_window": "The spread the run saved: std of the 238 window predicted residuals / std "
                                   "of the 28 run actual residuals. Recorded, not tested.",
            "dsse_focus": "Σ over testing_20251124:007's test runs of (model error² − lookup error²), m²/s².",
            "dsse_other": "The same over the other 25 test runs. dsse_focus + dsse_other = SSE_model − SSE_lookup.",
            "client_runs": "Training runs per client (client_0/client_1/client_2).",
            "largest_client_share": "The largest client's training runs / 112.",
            "client_run_cv": "Coefficient of variation (population std / mean) of the client run counts.",
            "mean_identity_share_on_largest": "Mean over the 8 identities of the share of its runs on the "
                                              "largest client.",
            "share_on_largest:<identity>": "That identity's share of its training runs on the largest client.",
            "focus_share_on_largest": "testing_20251124:007's share of its training runs on the largest client.",
            "focus_num_clients": "Clients holding testing_20251124:007 training runs.",
            "posthoc_isolated_identities": "POST HOC / exploratory: identities whose training runs all sit on "
                                           "one client that is not the largest.",
            "posthoc_isolated_list": "Those identities.",
            "rank_alpha / rank_rmse / rank_spread": "Formulas: average ranks among the confirmatory rows, for the "
                                                    "live Spearman (CORREL of ranks) on the stats sheet.",
            "reused_from_relabel": "yes for α = 0.1 and 0.3 draw 1 init 4601: read from artifacts/relabel.",
            "predictions_file": "Provenance: the run's saved test-run predictions, from which derived/ was "
                                "computed. This workbook reads derived/ only.",
            "partition_file": "Provenance: the run's run → client assignment.",
            "summary_* (block right of the table)": "Per α over every confirmatory row, the collapsed run "
                                                    "included: COUNTIFS, AVERAGEIFS and sample std via SUMPRODUCT.",
            "line_alpha and *_line / centralized_*_mean|min|max": "Reference lines (2 points at α = 0.01 and "
                                                                  "100): formulas over the lookup column and the "
                                                                  "centralized block.",
            "centralized_* (block under the lines)": "The centralized residual regression model per init seed: "
                                                      "RMSE, skill, outcome and its predictions file.",
        }),
        [*DERIVED_FILES, *(c["source_file"] for c in centralized or ())],
        extra=[
            "PRE-REGISTERED (CHANGES.md, before any job ran). Primary: Spearman ρ(α, RMSE) over the 15 "
            "confirmatory runs, two-sided Monte Carlo permutation p-value "
            f"({PERMUTATIONS:,} shuffles, generator seed {PERMUTATION_SEED}).",
            "Support rule for 'RMSE follows how far the model deviates from the lookup, not α': "
            "|ρ(spread, RMSE)| > |ρ(α, RMSE)| and p(α, RMSE) ≥ 0.05; ρ(α, spread) reported with it.",
            "Variance rule: between-α spread (sample std of the five per-α mean RMSEs) against within-α spread "
            "(mean of the five per-α sample stds); within ≥ between → the differences are dominated by the "
            "particular draw, not α. Reported whatever the outcome.",
            "Exploratory, not confirmatory: every descriptor correlation, the SSE split, and the post-hoc "
            "descriptor. The SSE split is an identity (SSE_model = SSE_lookup + dsse_focus + dsse_other), not a "
            "test.",
            "All measured confirmatory runs enter every statistic; collapsed / did-not-train runs are labelled, "
            "not excluded.",
            "Display only: in the α charts a collapsed run is drawn as its own × series instead of in its draw's "
            "series, and its row is sorted after the trained rows of the confirmatory block. The per-α mean ± std "
            "formulas, the rank formulas and the stats sheet still include it.",
            "The init-seed 4602 controls use the draw-1 partitions of α = 0.1 and 0.3 and sit at the same α as "
            "their draw-1 partners: the vertical gap is the effect of the training seed alone.",
            "NOT PRE-REGISTERED: the stats sheet's sensitivity block repeats the primary test, the support rule, "
            "the variance rule and the per-α summary without the collapsed confirmatory runs. It was added after "
            "the results were seen and is reported next to, never instead of, the pre-registered result.",
            "Not measured at build time: " + ("; ".join(missing) if missing else "none") + ".",
        ],
        metrics=("skill", "lookup RMSE", "bootstrap 95% interval", "outcome"),
        seeds="Split seed 4601 (112 train / 28 test runs). Init seed 4601 for every confirmatory run; 4602 for the "
              "two controls. Dirichlet partition seeds 4601 / 5601 / 6601 (draws 1 / 2 / 3).")
    return wb


def make_workbook(artifacts_root: Path, out_path: Path, repo: Path) -> Path:
    """Reads ``alpha_sweep/derived/`` (rows and analysis as saved) and, for the centralized
    reference lines, the relabel / S5 results."""
    from .excel_figures import git_commit

    artifacts_root = Path(artifacts_root).resolve()
    out_path = Path(out_path).resolve()
    if artifacts_root in out_path.parents:
        raise ValueError(f"Refusing to write under the results folder: {out_path}")
    rows, analysis = load_derived(artifacts_root / "alpha_sweep")
    wb = build_workbook(rows, analysis, git_commit(Path(repo)), centralized_reference(artifacts_root))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path

"""S5 experiment queue: the job matrix, outcomes, metrics and scheduling rules.

``scripts/run_s5_queue.py`` drives this. Everything that decides *what* runs, in what
order, whether a finished job counts as done, and what its numbers are lives here, so
it can be tested without training anything.

Every job writes into its own folder under one S5 results root, never into the
canonical ``artifacts/`` result folders. A job's ``job.json`` is written only after the
job ends; its ``outcome`` decides what a restart does:

- terminal, skipped on restart: ``completed``, ``completed_with_failed_clients``
  (local-only, some client failed an enforced gate), ``failed_detector`` (an enforced
  regression gate failed; deterministic, so re-running reproduces it),
  ``dropped_fallback`` (the 31 Oct fallback), ``reused_canonical`` (seed-4601
  centralized, scored from the committed run without retraining);
- ``crashed``: the folder is set aside and the job re-runs.

A folder without ``job.json`` is a job that was killed mid-run; it is set aside (renamed,
never deleted) and re-run.

Gates (S4 policy): ``degeneracy`` and ``trained`` are enforced by the pipelines on every
residual path; skill is reported with its CI and never enforced. The queue enforces
nothing itself. It flags **FLAT** convergence and keeps going: a classification run
whose global-train accuracy rises by less than 0.05 from round 0, or that predicts a
single class on the test set, or a regression run that failed the ``trained`` gate.
"""

from __future__ import annotations

import csv
import fnmatch
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .regression_metrics import report_from_run_rows
from .run_level_classification import METRICS_FILENAME, run_level_for_result

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "artifacts/s5"
SPLIT_SEED = 4601
INIT_SEEDS = (4601, 4602, 4603)
TASKS = ("classification", "regression")

#: CHANGES.md S3 "S5 scope": if S5 is not complete by the morning of 31 Oct 2026, drop
#: regression from Dirichlet first, then from rho = 0.375. Past this time, jobs at those
#: fallback levels that have not started are recorded as dropped; classification runs.
FALLBACK_CUTOFF = datetime(2026, 10, 31, 8, 0)
#: --stop-by starts a job only if its estimate x this margin ends before the stop time.
STOP_BY_MARGIN = 1.10
FLAT_MIN_ACCURACY_GAIN = 0.05

TERMINAL_OUTCOMES = frozenset(
    {"completed", "completed_with_failed_clients", "failed_detector", "dropped_fallback", "reused_canonical"}
)
MEASURED_OUTCOMES = frozenset({"completed", "completed_with_failed_clients", "failed_detector"})

#: name -> timing class. k3_iid is also channel overlap rho = 1.0 (run once, used for both).
FL_CONDITIONS = {
    "k2_natural": "k2",
    "k2_iid": "k2",
    "k3_iid": "k3",
    "k3_rho0.375": "k3",
    "k3_rho0": "k3",
    "k3_dirichlet0.1": "dirichlet",
}

#: First-run wall-time estimates in minutes, M5, client_num_cpus 3. Measured where the
#: S3 sweep ran the setting (natural K=2 E=5 persistent: 42.5 / 33.7; IID K=3 regression
#: 28.1); the rest are scaled (CHANGES.md, S5). Replaced by this root's own measurements
#: once a job of the same timing class has finished.
STATIC_MINUTES: dict[tuple[str, str, str], float] = {
    ("fl", "k2", "classification"): 43, ("fl", "k2", "regression"): 34,
    ("fl", "k3", "classification"): 35, ("fl", "k3", "regression"): 28,
    ("fl", "dirichlet", "classification"): 60, ("fl", "dirichlet", "regression"): 48,
    ("local_only", "k2", "classification"): 72, ("local_only", "k2", "regression"): 58,
    ("local_only", "k3", "classification"): 90, ("local_only", "k3", "regression"): 72,
    ("local_only", "dirichlet", "classification"): 90, ("local_only", "dirichlet", "regression"): 72,
    ("centralized", "classification", "classification"): 18,
    ("centralized", "absolute", "regression"): 10,
    ("centralized", "residual", "regression"): 9,
    ("source_probe", "source_probe", "classification"): 18,
    ("subject003_control", "subject003_control", "classification"): 6,
}

CANONICAL_CENTRALIZED = {
    "classification": ROOT / "artifacts/centralized_combined_9ch_no006_nowalking_80_20_60e",
    "absolute": ROOT / "artifacts/centralized_combined_regression_9ch_no006_walking_80_20_60e",
    "residual": ROOT / "artifacts/centralized_combined_residual_runbalanced_9ch_no006_60e",
}
CENTRALIZED_RUNNERS = {
    "classification": "scripts/run_combined_centralized_classification.py",
    "absolute": "scripts/run_combined_centralized_regression.py",
    "residual": "scripts/run_combined_residual_run_regression.py",
}
CENTRALIZED_TASK = {"classification": "classification", "absolute": "regression", "residual": "regression"}
CLASSIFICATION_CHECKPOINTS = {
    "fl": "flower_iid_global_classification_model.pt",
    "local_only": "flower_iid_global_classification_model.pt",
    "centralized": "combined_centralized_cnn_final.pt",
}


@dataclass(frozen=True)
class Job:
    job_id: str  # its folder, relative to the results root
    group: int
    family: str  # fl | local_only | centralized | source_probe | subject003_control
    condition: str
    seed: int
    task: str
    fallback_level: int = 0
    reuse_canonical: bool = False

    @property
    def timing_key(self) -> tuple[str, str, str]:
        size = FL_CONDITIONS.get(self.condition, self.condition)
        return (self.family, size, self.task)

    def directory(self, root: Path) -> Path:
        return root / self.job_id

    def result_dir(self, root: Path) -> Path:
        """Where the runner's own outputs land."""
        if self.reuse_canonical:
            return CANONICAL_CENTRALIZED[self.condition]
        if self.family == "fl":
            return self.directory(root) / self.task
        if self.family == "local_only":
            return self.directory(root) / "local_only" / self.task
        return self.directory(root)


def _fallback_level(condition: str, task: str) -> int:
    if task != "regression":
        return 0
    return {"k3_dirichlet0.1": 1, "k3_rho0.375": 2}.get(condition, 0)


def build_jobs() -> list[Job]:
    """The S5 matrix, most important first (the order approved for the queue)."""
    jobs: list[Job] = []

    def fl(group: int, family: str, condition: str, seed: int) -> None:
        for task in TASKS:
            jobs.append(Job(f"{family}/{condition}/seed{seed}/{task}", group, family, condition, seed, task,
                            _fallback_level(condition, task)))

    for seed in INIT_SEEDS:  # (1) K=2 natural and IID, all seeds
        for condition in ("k2_natural", "k2_iid"):
            fl(1, "fl", condition, seed)
    for condition in ("k2_natural", "k2_iid"):  # (2) their local-only runs
        fl(2, "local_only", condition, SPLIT_SEED)
    for kind in ("classification", "absolute", "residual"):  # (3) centralized, probe, control
        jobs.append(Job(f"centralized/{kind}/seed4601", 3, "centralized", kind, 4601, CENTRALIZED_TASK[kind],
                        reuse_canonical=True))
    for seed in INIT_SEEDS[1:]:
        for kind in ("classification", "absolute", "residual"):
            jobs.append(Job(f"centralized/{kind}/seed{seed}", 3, "centralized", kind, seed, CENTRALIZED_TASK[kind]))
    jobs.append(Job("source_probe/seed4601", 3, "source_probe", "source_probe", 4601, "classification"))
    jobs.append(Job("subject003_control/seed4601", 3, "subject003_control", "subject003_control", 4601, "classification"))
    for condition in ("k3_iid", "k3_rho0.375", "k3_rho0"):  # (4) K=3 channel axis
        fl(4, "fl", condition, SPLIT_SEED)
    fl(5, "fl", "k3_dirichlet0.1", SPLIT_SEED)  # (5) Dirichlet
    for condition in ("k3_iid", "k3_rho0.375", "k3_rho0", "k3_dirichlet0.1"):  # (6) remaining local-only
        fl(6, "local_only", condition, SPLIT_SEED)
    return jobs


def select_jobs(jobs: Sequence[Job], patterns: Iterable[str] | None) -> list[Job]:
    patterns = [p for p in (patterns or []) if p]
    if not patterns:
        return list(jobs)
    return [job for job in jobs if any(fnmatch.fnmatch(job.job_id, p) for p in patterns)]


def command(job: Job, root: Path, python: str, smoke: bool) -> list[str] | None:
    """The child command, or None for a job that does not run a child (or not in smoke)."""
    out = str(job.directory(root))
    if job.reuse_canonical:
        return None
    if job.family in ("fl", "local_only"):
        cmd = [python, str(ROOT / "scripts/run_combined_iid_flower.py"),
               "--config", str(ROOT / f"configs/s5_{job.condition}.yaml"),
               "--task", job.task, "--output-root", out, "--set", f"init_seed={job.seed}"]
        if job.family == "local_only":
            cmd += ["--mode", "local_only"]
        if smoke:
            cmd += ["--rounds", "1"] + (["--allow-degenerate"] if job.task == "regression" else [])
        return cmd
    if job.family == "centralized":
        if smoke:
            return None  # the centralized runners pin 60 epochs; covered by the S4 runner tests
        return [python, str(ROOT / CENTRALIZED_RUNNERS[job.condition]),
                "--config", str(ROOT / f"configs/s5_centralized_{job.condition}_init{job.seed}.yaml"),
                "--output-root", out]
    script = {"source_probe": "scripts/run_source_probe.py", "subject003_control": "scripts/run_subject003_control.py"}[job.family]
    return [python, str(ROOT / script), "--output-root", out] + (["--epochs", "1"] if smoke else [])


# --------------------------------------------------------------------------------------
# job.json and outcomes
# --------------------------------------------------------------------------------------


def read_job_record(job: Job, root: Path) -> dict[str, Any] | None:
    path = job.directory(root) / "job.json"
    return json.loads(path.read_text()) if path.exists() else None


def is_terminal(record: dict[str, Any] | None) -> bool:
    return record is not None and record.get("outcome") in TERMINAL_OUTCOMES


def detect_outcome(job: Job, root: Path, exit_code: int, log_text: str) -> str:
    result = job.result_dir(root)
    if job.family == "local_only":
        summary = result / "local_only_summary.json"
        if summary.exists():
            failed = json.loads(summary.read_text()).get("failed_clients") or []
            if failed:
                return "completed_with_failed_clients"
            return "completed" if exit_code == 0 else "crashed"
        return "crashed"
    expected = {
        "source_probe": "source_probe_summary.json",
        "subject003_control": "subject003_control_summary.json",
    }.get(job.family, "training_summary.json")
    if (result / expected).exists():
        return "completed" if exit_code == 0 else "crashed"
    if (result / "degenerate_model_check_FAILED.json").exists():
        return "failed_detector"
    if job.task == "regression" and "DegenerateModelError" in log_text:
        return "failed_detector"  # the centralized residual path raises without a record
    return "crashed"


# --------------------------------------------------------------------------------------
# metrics and the FLAT flag
# --------------------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def regression_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    report = summary["regression_report"]
    overall = report["overall"]
    metrics: dict[str, Any] = {
        "skill": overall["skill"],
        "skill_ci_low": overall["skill_ci95"][0],
        "skill_ci_high": overall["skill_ci95"][1],
        "model_rmse_mps": overall["model_rmse_mps"],
        "baseline_rmse_mps": overall["baseline_rmse_mps"],
        "r2_within_subject": overall["r2_within_subject"],
    }
    for source, block in report["per_source"].items():
        metrics[f"skill_{source}"] = block["skill"]
        metrics[f"skill_{source}_ci_low"], metrics[f"skill_{source}_ci_high"] = block["skill_ci95"]
    gates = summary.get("degenerate_model_check", {}).get("gates", {})
    for gate in ("degeneracy", "trained"):
        metrics[f"gate_{gate}"] = gates.get(gate, {}).get("passed")
    return metrics


def classification_metrics(run_level: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"window_accuracy": run_level["window_accuracy_all_test_windows"]}
    for method, short in (("majority_vote", "run_vote"), ("mean_probability", "run_meanprob")):
        block = run_level[method]
        metrics[f"{short}_accuracy"] = block["accuracy"]
        metrics[f"{short}_correct"] = block["num_correct"]
        metrics[f"{short}_runs"] = block["num_runs"]
        metrics[f"{short}_ci_low"], metrics[f"{short}_ci_high"] = block["wilson95"]
        metrics[f"{short}_macro_f1"] = block["macro_f1"]
    return metrics


def _regression_summary(result: Path) -> dict[str, Any] | None:
    for name in ("training_summary.json", "degenerate_model_check_FAILED.json"):
        if (result / name).exists():
            return json.loads((result / name).read_text())
    return None


def _flat_classification(result: Path) -> bool:
    predictions = _read_csv(result / "test_predictions.csv")
    if len({row["predicted_class"] for row in predictions}) <= 1:
        return True
    history_path = result / "global_train_history.csv"
    if history_path.exists():  # federated / local-only: round 0 is the initial model
        history = _read_csv(history_path)
        gain = float(history[-1]["global_train_accuracy"]) - float(history[0]["global_train_accuracy"])
        return gain < FLAT_MIN_ACCURACY_GAIN
    history = _read_csv(result / "training_history.csv")  # centralized: no epoch 0
    num_classes = len(json.loads((result / "training_summary.json").read_text())["per_class"])
    return float(history[-1]["train_accuracy"]) < 1.0 / num_classes + FLAT_MIN_ACCURACY_GAIN


def _mean_std(per_client: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for m in per_client.values() for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([m[key] for m in per_client.values() if isinstance(m.get(key), (int, float)) and m.get(key) is not None], dtype=float)
        out[f"{key}_mean"] = float(values.mean()) if values.size else math.nan
        out[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return out


def collect_metrics(job: Job, root: Path, outcome: str, samples_loader: Callable[[], Any]) -> dict[str, Any]:
    """Headline numbers for job.json and the index; writes run-level files as a side effect.

    Returns ``{"metrics": {...}, "flat": bool | None, "per_client": {...}?}``.
    """
    if outcome not in MEASURED_OUTCOMES | {"reused_canonical"}:
        return {"metrics": {}, "flat": None}
    result = job.result_dir(root)
    directory = job.directory(root)

    if job.family == "local_only":
        per_client: dict[str, dict[str, Any]] = {}
        flats = []
        for client_dir in sorted(p for p in result.iterdir() if p.is_dir() and p.name.startswith("client_")):
            if job.task == "classification":
                if not (client_dir / "test_predictions.csv").exists():
                    continue
                run_level = run_level_for_result(client_dir, CLASSIFICATION_CHECKPOINTS["local_only"], samples_loader(), client_dir / "run_level")
                per_client[client_dir.name] = classification_metrics(run_level)
                flats.append(_flat_classification(client_dir))
            else:
                summary = _regression_summary(client_dir)
                if summary is None:
                    continue
                per_client[client_dir.name] = regression_metrics(summary)
                flats.append(per_client[client_dir.name]["gate_trained"] is False)
        local = json.loads((result / "local_only_summary.json").read_text())
        metrics = {**_mean_std(per_client), "num_clients": len(per_client),
                   "failed_clients": json.dumps(local.get("failed_clients") or [])}
        return {"metrics": metrics, "flat": any(flats), "per_client": per_client}

    if job.family == "source_probe":
        summary = json.loads((result / "source_probe_summary.json").read_text())
        run_level = json.loads((result / "run_level" / METRICS_FILENAME).read_text())
        metrics = classification_metrics(run_level)
        for activity, block in summary["test_window_accuracy_by_activity"].items():
            metrics[f"window_accuracy_{activity}"] = block["accuracy"]
        return {"metrics": metrics, "flat": _flat_classification(result)}

    if job.family == "subject003_control":
        summary = json.loads((result / "subject003_control_summary.json").read_text())
        metrics: dict[str, Any] = {}
        for label in ("in_session", "cross_session"):
            block = summary[label]["majority_vote"]
            metrics[f"{label}_run_vote_accuracy"] = block["accuracy"]
            metrics[f"{label}_run_vote_correct"] = block["num_correct"]
            metrics[f"{label}_run_vote_runs"] = block["num_runs"]
            metrics[f"{label}_run_vote_ci_low"], metrics[f"{label}_run_vote_ci_high"] = block["wilson95"]
            metrics[f"{label}_run_meanprob_accuracy"] = summary[label]["mean_probability"]["accuracy"]
        negative = {}
        for row in summary["predicted_label_distribution"]:
            if row["control_set"] == "negative_control":
                votes = {k.removeprefix("runs_majority_"): v for k, v in row.items() if k.startswith("runs_majority_") and v}
                negative[row["subject_id"]] = votes
        metrics["negative_control_run_vote_labels"] = json.dumps(negative, sort_keys=True)
        return {"metrics": metrics, "flat": _flat_classification(result)}

    if job.task == "classification":
        checkpoint = CLASSIFICATION_CHECKPOINTS[job.family]
        out = (directory / "run_level") if job.reuse_canonical else (result / "run_level")
        run_level = run_level_for_result(result, checkpoint, samples_loader(), out)
        return {"metrics": classification_metrics(run_level), "flat": _flat_classification(result)}

    # regression: federated, centralized absolute or residual
    if job.reuse_canonical:
        # The committed seed-4601 summaries predate S4's regression_report; S4 showed the
        # re-run reproduces their predictions byte for byte, so the report is computed
        # from the committed test_run_predictions.csv (read-only). Gates as committed.
        baseline_key = "source_subject_train_run_mean_mps" if job.condition == "absolute" else "source_subject_train_mean_mps"
        report = report_from_run_rows(_read_csv(result / "test_run_predictions.csv"), baseline_key)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "regression_report.json").write_text(json.dumps(report, indent=2))
        committed = json.loads((result / "training_summary.json").read_text())
        check = json.loads(json.dumps(committed.get("degenerate_model_check", {})))
        gates = check.setdefault("gates", {})
        if job.condition == "residual" and "trained" not in gates:
            # Pre-S4 detector had no separate "trained" gate; derive it from the
            # standardized MSE the committed detector recorded (S4: < 0.95).
            threshold = check["thresholds"]["maximum_standardized_mse"]
            gates["trained"] = {"enforced": False, "passed": check["final_standardized_mse"] < threshold,
                                "note": "derived from the committed final_standardized_mse"}
        summary = {"regression_report": report, "degenerate_model_check": check}
    else:
        summary = _regression_summary(result)
    metrics = regression_metrics(summary)
    flat = None if job.condition == "absolute" else (metrics["gate_trained"] is False)
    return {"metrics": metrics, "flat": flat}


def status_text(job: Job, record: dict[str, Any]) -> str:
    m = record.get("metrics") or {}
    outcome = record["outcome"]
    flat = " FLAT" if record.get("flat") else ""
    head = f"{job.job_id} {outcome}{flat}"
    if record.get("wall_seconds") is not None:
        head += f" {record['wall_seconds'] / 60:.1f} min"
    if not m:
        return head

    def mark(value: Any) -> str:
        return "?" if value is None else ("✓" if value else "✗")

    if job.family == "subject003_control":
        detail = (f"in-session runs {m['in_session_run_vote_correct']}/{m['in_session_run_vote_runs']} | "
                  f"cross-session runs {m['cross_session_run_vote_correct']}/{m['cross_session_run_vote_runs']} "
                  f"[{m['cross_session_run_vote_ci_low']:.2f}, {m['cross_session_run_vote_ci_high']:.2f}] | "
                  f"negative control votes {m['negative_control_run_vote_labels']}")
    elif job.family == "local_only" and job.task == "classification":
        detail = (f"{m['num_clients']} clients: run acc (vote) {m.get('run_vote_accuracy_mean', math.nan):.3f} "
                  f"± {m.get('run_vote_accuracy_std', math.nan):.3f}, window acc {m.get('window_accuracy_mean', math.nan):.3f}")
    elif job.family == "local_only":
        detail = (f"{m['num_clients']} clients: skill {m.get('skill_mean', math.nan):+.3f} ± {m.get('skill_std', math.nan):.3f} "
                  f"| failed clients {m['failed_clients']}")
    elif job.task == "classification":
        detail = (f"window acc {m['window_accuracy']:.3f} | run acc (vote) {m['run_vote_correct']}/{m['run_vote_runs']} "
                  f"[{m['run_vote_ci_low']:.2f}, {m['run_vote_ci_high']:.2f}]")
    else:
        detail = (f"skill {m['skill']:+.3f} [{m['skill_ci_low']:+.3f}, {m['skill_ci_high']:+.3f}] | "
                  f"RMSE {m['model_rmse_mps']:.4f} vs lookup {m['baseline_rmse_mps']:.4f} | "
                  f"gates degeneracy{mark(m['gate_degeneracy'])} trained{mark(m['gate_trained'])}")
    return f"{head} | {detail}"


# --------------------------------------------------------------------------------------
# scheduling: estimates, --stop-by, fallback
# --------------------------------------------------------------------------------------


def estimate_minutes(job: Job, root: Path, jobs: Sequence[Job], smoke: bool) -> float:
    """Longest measured wall time of a finished job of the same timing class, else static."""
    if job.reuse_canonical:
        return 0.0
    measured = []
    for other in jobs:
        if other.timing_key != job.timing_key:
            continue
        record = read_job_record(other, root)
        if record and record.get("outcome") in MEASURED_OUTCOMES and bool(record.get("smoke")) == smoke:
            measured.append(float(record["wall_seconds"]) / 60.0)
    return max(measured) if measured else float(STATIC_MINUTES[job.timing_key])


def parse_stop_by(value: str, now: datetime) -> datetime:
    """The next occurrence of HH:MM after ``now`` (today if still ahead, else tomorrow)."""
    try:
        clock = datetime.strptime(value, "%H:%M")
    except ValueError as error:
        raise ValueError(f"--stop-by expects HH:MM, got {value!r}") from error
    candidate = now.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
    return candidate if candidate > now else candidate + timedelta(days=1)


def fits_before(now: datetime, estimate_min: float, stop_by: datetime | None) -> bool:
    return stop_by is None or now + timedelta(minutes=estimate_min * STOP_BY_MARGIN) <= stop_by


def fallback_drops(job: Job, now: datetime, cutoff: datetime = FALLBACK_CUTOFF) -> bool:
    return job.fallback_level > 0 and now >= cutoff


# --------------------------------------------------------------------------------------
# index and derived tables (rebuilt from job.json files after every job)
# --------------------------------------------------------------------------------------

INDEX_BASE_COLUMNS = ["job_id", "group", "family", "condition", "seed", "task", "status", "flat",
                      "wall_minutes", "result_dir"]


def _write_rows(path: Path, rows: Sequence[dict[str, Any]], leading: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(leading) + sorted({k for row in rows for k in row} - set(leading))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def rebuild_index(jobs: Sequence[Job], root: Path) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        record = read_job_record(job, root) or {}
        rows.append({
            "job_id": job.job_id, "group": job.group, "family": job.family, "condition": job.condition,
            "seed": job.seed, "task": job.task, "status": record.get("outcome", "pending"),
            "flat": "FLAT" if record.get("flat") else "",
            "wall_minutes": "" if record.get("wall_seconds") is None else round(record["wall_seconds"] / 60, 1),
            "result_dir": str(job.result_dir(root)),
            **(record.get("metrics") or {}),
        })
    _write_rows(root / "results_index.csv", rows, INDEX_BASE_COLUMNS)
    _write_seed_aggregates(jobs, root)
    _write_run_level_tables(jobs, root)
    return rows


def _write_seed_aggregates(jobs: Sequence[Job], root: Path) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for job in jobs:
        record = read_job_record(job, root)
        if record and record.get("outcome") in MEASURED_OUTCOMES | {"reused_canonical"} and record.get("metrics"):
            groups.setdefault((job.family, job.condition, job.task), []).append({"seed": job.seed, **record["metrics"]})
    rows = []
    for (family, condition, task), members in groups.items():
        if len(members) < 2:
            continue
        row: dict[str, Any] = {"family": family, "condition": condition, "task": task,
                               "seeds": " ".join(str(m["seed"]) for m in members), "n": len(members)}
        for key in sorted({k for m in members for k in m} - {"seed"}):
            values = [m.get(key) for m in members]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                array = np.asarray(values, dtype=float)
                row[f"{key}_mean"] = float(array.mean())
                row[f"{key}_std"] = float(array.std(ddof=1))
        rows.append(row)
    if rows:
        _write_rows(root / "derived/seed_aggregates.csv", rows, ["family", "condition", "task", "seeds", "n"])


def _run_level_files(job: Job, root: Path) -> list[tuple[str, Path]]:
    if job.task != "classification" or job.family in ("source_probe", "subject003_control"):
        return []
    if job.reuse_canonical:
        return [("", job.directory(root) / "run_level" / METRICS_FILENAME)]
    result = job.result_dir(root)
    if job.family == "local_only":
        return [(p.parent.parent.name, p) for p in sorted(result.glob("client_*/run_level/" + METRICS_FILENAME))]
    return [("", result / "run_level" / METRICS_FILENAME)]


def _write_run_level_tables(jobs: Sequence[Job], root: Path) -> None:
    """Run-level classification per job (and client), and the free cross-session look:
    the subject x source rows of every subject recorded in more than one source."""
    summary_rows, cross_rows = [], []
    for job in jobs:
        for client, path in _run_level_files(job, root):
            if not path.exists():
                continue
            run_level = json.loads(path.read_text())
            base = {"job_id": job.job_id, "family": job.family, "condition": job.condition,
                    "seed": job.seed, "client": client}
            for method in ("majority_vote", "mean_probability"):
                block = run_level[method]
                summary_rows.append({**base, "method": method, "num_runs": block["num_runs"],
                                     "num_correct": block["num_correct"], "accuracy": block["accuracy"],
                                     "wilson95_low": block["wilson95"][0], "wilson95_high": block["wilson95"][1],
                                     "macro_f1": block["macro_f1"],
                                     **{f"accuracy_{s}": b["accuracy"] for s, b in block["per_source"].items()}})
            sources_of: dict[str, set[str]] = {}
            for row in run_level["per_subject_source"]:
                sources_of.setdefault(row["subject_id"], set()).add(row["source_id"])
            for row in run_level["per_subject_source"]:
                if len(sources_of[row["subject_id"]]) > 1:
                    cross_rows.append({**base, **row})
    if summary_rows:
        _write_rows(root / "derived/run_level_classification.csv", summary_rows,
                    ["job_id", "family", "condition", "seed", "client", "method"])
    if cross_rows:
        _write_rows(root / "derived/cross_session_free_breakdown.csv", cross_rows,
                    ["job_id", "family", "condition", "seed", "client", "subject_id", "source_id"])

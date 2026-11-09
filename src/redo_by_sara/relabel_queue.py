"""Relabel queue (PI ground truth 2026-11-08): the job matrix, commands and tables.

``scripts/run_relabel_queue.py`` drives this, the same way ``run_s5_queue.py`` drives
``s5_queue``. Outcomes, metrics, gates, labels, the FLAT flag, per-seed reporting and the
status lines are S5's, imported unchanged from ``s5_queue``. What differs:

- the classification label is the (source, subject) identity: 8 identities +
  ``no_walking`` = 9 classes, from ``artifacts/relabel/data/classification``;
- configs are ``configs/relabel_*.yaml``; results go under ``artifacts/relabel``;
- the centralized seed-4601 classifier runs first, because every FL and local-only
  classification job compares against it (its folder is passed with ``--set``);
- no job reuses a committed run and there is no fallback clock.

Kept from S5, not rerun: every regression job whose partition is unchanged (K=2 IID and
natural, all seeds; K=3 rho = 1.0 / 0.375 / 0; their local-only runs), all centralized
regression, and the source probe. Re-scored without training (``relabel_rescore``):
local-only natural K=2 classification and the cross-session control. Everything else is
rerun here. ``s5_disposition`` records this per S5 job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from . import s5_queue
from .identities import DEFAULT_LINKS_PATH, identity_id, linked_identities, load_identity_links
from .s5_queue import Job, MEASURED_OUTCOMES, read_job_record

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "artifacts/relabel"
RELABEL_CLASSIFICATION_ARTIFACT = ROOT / "artifacts/relabel/data/classification/raw_windows_combined.pt"
SPLIT_SEED = 4601
INIT_SEEDS = (4601, 4602, 4603)
CENTRALIZED_REFERENCE_JOB = "centralized/classification/seed4601"

#: condition -> timing class
FL_CONDITIONS = {
    "k2_natural": "k2",
    "k2_iid": "k2",
    "k3_iid": "k3",
    "k3_rho0.375": "k3",
    "k3_rho0": "k3",
    "k3_dirichlet0.1": "dirichlet",
    "k3_dirichlet0.3": "dirichlet",
}
#: First-run estimates in minutes: the longest measured S5 wall time of the same timing
#: class (CHANGES.md, S5 results). alpha = 0.3 is assumed to take as long as alpha = 0.1.
#: Replaced by this root's own measurements once a job of the class has finished.
STATIC_MINUTES: dict[tuple[str, str, str], float] = {
    ("fl", "k2", "classification"): 48.4,
    ("fl", "k3", "classification"): 33.8,
    ("fl", "dirichlet", "classification"): 56.9, ("fl", "dirichlet", "regression"): 31.0,
    ("local_only", "k2", "classification"): 48.2,
    ("local_only", "k3", "classification"): 44.9,
    ("local_only", "dirichlet", "classification"): 50.4, ("local_only", "dirichlet", "regression"): 33.5,
    ("centralized", "classification", "classification"): 8.0,
}


def timing_key(job: Job) -> tuple[str, str, str]:
    return (job.family, FL_CONDITIONS.get(job.condition, job.condition), job.task)


def build_jobs() -> list[Job]:
    """The approved order (PI, 2026-11-08), most important first."""
    jobs: list[Job] = []

    def add(group: int, family: str, condition: str, seed: int, task: str) -> None:
        jobs.append(Job(f"{family}/{condition}/seed{seed}/{task}", group, family, condition, seed, task))

    def centralized(seed: int) -> None:
        jobs.append(Job(f"centralized/classification/seed{seed}", 1, "centralized", "classification", seed, "classification"))

    # (1) K=2 classification, FL natural and IID, all seeds; centralized classification
    # seeds. Seed 4601 centralized first: the FL classification jobs compare against it.
    centralized(4601)
    for seed in INIT_SEEDS:
        for condition in ("k2_natural", "k2_iid"):
            add(1, "fl", condition, seed, "classification")
    for seed in INIT_SEEDS[1:]:
        centralized(seed)

    def k3(group: int, family: str) -> None:
        # rho conditions: classification only; their regression keeps S5's results.
        for condition in ("k3_iid", "k3_rho0.375", "k3_rho0"):
            add(group, family, condition, SPLIT_SEED, "classification")
        for condition in ("k3_dirichlet0.1", "k3_dirichlet0.3"):
            add(group, family, condition, SPLIT_SEED, "classification")
            add(group, family, condition, SPLIT_SEED, "regression")
            for seed in INIT_SEEDS[1:]:
                add(group, family, condition, seed, "classification")

    k3(2, "fl")  # (2) K=3 FL: rho conditions, Dirichlet alpha = 0.1 then 0.3
    add(3, "local_only", "k2_iid", SPLIT_SEED, "classification")  # (3)
    k3(4, "local_only")  # (4) K=3 local-only
    return jobs


def reference_dir(root: Path) -> Path:
    """The centralized seed-4601 classifier every FL / local-only classification job
    compares against (the result folder of ``CENTRALIZED_REFERENCE_JOB``)."""
    return Path(root) / CENTRALIZED_REFERENCE_JOB


def needs_reference(job: Job) -> bool:
    return job.family in ("fl", "local_only") and job.task == "classification"


def command(job: Job, root: Path, python: str) -> list[str]:
    out = str(job.directory(root))
    if job.family in ("fl", "local_only"):
        cmd = [python, str(ROOT / "scripts/run_combined_iid_flower.py"),
               "--config", str(ROOT / f"configs/relabel_{job.condition}.yaml"),
               "--task", job.task, "--output-root", out, "--set", f"init_seed={job.seed}"]
        if needs_reference(job):
            cmd += ["--set", f"centralized_classification_dir={reference_dir(root)}"]
        if job.family == "local_only":
            cmd += ["--mode", "local_only"]
        return cmd
    if job.family == "centralized":
        return [python, str(ROOT / "scripts/run_combined_centralized_classification.py"),
                "--config", str(ROOT / f"configs/relabel_centralized_classification_init{job.seed}.yaml"),
                "--output-root", out]
    raise ValueError(f"Unknown relabel job family {job.family!r}.")


def estimate_minutes(job: Job, root: Path, jobs: Sequence[Job]) -> float:
    """Longest measured wall time of a finished job of the same timing class, else static."""
    measured = []
    for other in jobs:
        if timing_key(other) != timing_key(job):
            continue
        record = read_job_record(other, root)
        if record and record.get("outcome") in MEASURED_OUTCOMES:
            measured.append(float(record["wall_seconds"]) / 60.0)
    return max(measured) if measured else float(STATIC_MINUTES[timing_key(job)])


# --------------------------------------------------------------------------------------
# S5 disposition: what is kept, re-scored or rerun, per S5 job
# --------------------------------------------------------------------------------------

KEPT_REGRESSION_CONDITIONS = ("k2_iid", "k2_natural", "k3_iid", "k3_rho0.375", "k3_rho0")


def s5_disposition(job: Job) -> tuple[str, str]:
    """(disposition, reason) for one S5 job under the relabel."""
    if job.family == "source_probe":
        return "kept", "label is the source, not the subject"
    if job.family == "subject003_control":
        return "re-scored", "trained on one source only; redefined by identity links (relabel_rescore)"
    if job.task == "regression":
        if job.family == "centralized":
            return "kept", "regression lookup, target and evaluation are keyed by (source, subject)"
        if job.condition in KEPT_REGRESSION_CONDITIONS:
            return "kept", "run assignment byte-identical under identities; regression keyed by (source, subject)"
        return "rerun", "Dirichlet groups by identity now: new partition"
    if job.family == "local_only" and job.condition == "k2_natural":
        return "re-scored", "each client saw one source, so each output names one identity"
    return "rerun", "classification label is now the (source, subject) identity"


def s5_disposition_rows() -> list[dict[str, Any]]:
    rows = []
    for job in s5_queue.build_jobs():
        disposition, reason = s5_disposition(job)
        rows.append({"s5_job_id": job.job_id, "family": job.family, "condition": job.condition,
                     "seed": job.seed, "task": job.task, "disposition": disposition, "reason": reason})
    return rows


# --------------------------------------------------------------------------------------
# index and derived tables
# --------------------------------------------------------------------------------------


def _linked_identity_rows(jobs: Sequence[Job], root: Path, links: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """Run-level rows of every identity in a same-person link, per job and client."""
    rows = []
    for job in jobs:
        for client, path in s5_queue._run_level_files(job, root):
            if not path.exists():
                continue
            for row in json.loads(path.read_text())["per_subject_source"]:
                identity = identity_id(row["source_id"], row["subject_id"])
                linked = linked_identities(identity, links)
                if linked:
                    rows.append({"job_id": job.job_id, "family": job.family, "condition": job.condition,
                                 "seed": job.seed, "client": client, "identity": identity,
                                 "same_person_as": " ".join(linked), **row})
    return rows


def rebuild_index(jobs: Sequence[Job], root: Path, links_path: Path = DEFAULT_LINKS_PATH) -> list[dict[str, Any]]:
    """S5's index and seed aggregates, plus the identity-link breakdown and the S5
    disposition table. The S5 "free look" (a subject ID in more than one source) does not
    apply: under identities no class spans sources."""
    groups = s5_queue._seed_groups(jobs, root)
    rows = []
    for job in jobs:
        record = read_job_record(job, root) or {}
        metrics = dict(record.get("metrics") or {})
        label = metrics.pop("training_outcome", "")
        members = groups.get((job.family, job.condition, job.task), [])
        rows.append({
            "job_id": job.job_id, "group": job.group, "family": job.family, "condition": job.condition,
            "seed": job.seed, "task": job.task, "status": record.get("outcome", "pending"),
            "flat": "FLAT" if record.get("flat") else "",
            "training_outcome": label,
            "seed_reporting": s5_queue.seed_reporting(members) if len(members) > 1 and record.get("metrics") else "",
            "wall_minutes": "" if record.get("wall_seconds") is None else round(record["wall_seconds"] / 60, 1),
            "result_dir": str(job.result_dir(root)),
            **metrics,
        })
    s5_queue._write_rows(root / "results_index.csv", rows, s5_queue.INDEX_BASE_COLUMNS)
    s5_queue._write_seed_aggregates(jobs, root)
    _write_run_level_tables(jobs, root, load_identity_links(links_path))
    s5_queue._write_rows(root / "derived/s5_disposition.csv", s5_disposition_rows(),
                         ["s5_job_id", "family", "condition", "seed", "task", "disposition", "reason"])
    return rows


def _write_run_level_tables(jobs: Sequence[Job], root: Path, links: Sequence[Sequence[str]]) -> None:
    summary_rows = []
    for job in jobs:
        for client, path in s5_queue._run_level_files(job, root):
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
    if summary_rows:
        s5_queue._write_rows(root / "derived/run_level_classification.csv", summary_rows,
                             ["job_id", "family", "condition", "seed", "client", "method"])
    linked = _linked_identity_rows(jobs, root, links)
    if linked:
        s5_queue._write_rows(root / "derived/linked_identity_breakdown.csv", linked,
                             ["job_id", "family", "condition", "seed", "client", "identity", "same_person_as"])


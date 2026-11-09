"""Relabel queue: every rerun job of the (source, subject) identity relabel, resumable.

PI ground truth 2026-11-08: the classification label is the (source, subject) identity,
8 identities + no_walking = 9 classes. Order (approved): (1) centralized classification
seed 4601 (the reference every FL classification job compares against), FL K=2 natural
and IID classification at init seeds 4601/4602/4603, centralized classification seeds
4602/4603; (2) K=3 FL: rho = 1.0 (= IID), 0.375, 0 classification, then Dirichlet
alpha = 0.1 and alpha = 0.3 (classification x 3 seeds, regression seed 4601); (3)
local-only K=2 IID classification; (4) K=3 local-only, as (2). Split seed 4601
throughout. Every FL and local-only job runs FedAvg, E=5, persistent client optimizer,
60 rounds, client_num_cpus 3, global 9-channel evaluation, one task per job.

  python scripts/run_relabel_queue.py --dry-run            # the plan, status and ETA
  python scripts/run_relabel_queue.py                      # run everything not yet done
  python scripts/run_relabel_queue.py --stop-by 05:00      # start only jobs that end by 05:00
  python scripts/run_relabel_queue.py --derived-only       # re-score finished jobs, rebuild tables

Restart-safe: finished jobs are skipped; a job killed mid-run is set aside and re-run.
Each job writes only to its own folder under --root; progress goes to queue.log and one
status line per finished job to stdout and status.log. Children run under
`caffeinate -i` on macOS, which prevents idle sleep but not lid-close sleep: keep the
machine open and on power.

Preflight refuses to start unless every kept K=2 / K=3 IID (and natural) run assignment,
rebuilt from the identity-labelled artifact, is byte-identical to every S5 copy: the
condition under which S5's regression results for those partitions are kept.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.relabel_queue import (  # noqa: E402
    CENTRALIZED_REFERENCE_JOB,
    DEFAULT_RESULTS_ROOT,
    RELABEL_CLASSIFICATION_ARTIFACT,
    build_jobs,
    command,
    estimate_minutes,
    needs_reference,
    rebuild_index,
    reference_dir,
)
from redo_by_sara.s5_queue import (  # noqa: E402
    CANONICAL_CENTRALIZED,
    STOP_BY_MARGIN,
    Job,
    collect_metrics,
    detect_outcome,
    fits_before,
    is_terminal,
    parse_stop_by,
    read_job_record,
    recompute_metrics,
    select_jobs,
    status_text,
)

EXPECTED_VERSIONS = {"flwr": "1.29.0", "ray": "2.51.1"}
REGRESSION_ARTIFACT = CANONICAL_CENTRALIZED["absolute"] / "raw_walking_windows_regression.pt"
S5_ROOT = ROOT / "artifacts/s5"
CHILD_GRACE_SECONDS = 60


class Interrupted(Exception):
    pass


def _raise_interrupt(signum: int, frame: object) -> None:
    raise Interrupted(signal.Signals(signum).name)


class Queue:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.root).resolve()
        self.all_jobs = build_jobs()
        self.jobs = select_jobs(self.all_jobs, args.only.split(",") if args.only else None)
        self.started = datetime.now()
        self.stop_by = parse_stop_by(args.stop_by, self.started) if args.stop_by else None
        self._samples = None
        self.head = "unknown"

    # ---------------------------------------------------------------- logging
    def log(self, message: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "queue.log").open("a") as handle:
            handle.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")

    def status(self, line: str) -> None:
        stamped = f"[{datetime.now():%m-%d %H:%M}] {line}"
        print(stamped, flush=True)
        with (self.root / "status.log").open("a") as handle:
            handle.write(stamped + "\n")
        self.log("STATUS " + line)

    # ---------------------------------------------------------------- helpers
    def samples(self):
        if self._samples is None:
            import torch

            self._samples = torch.load(RELABEL_CLASSIFICATION_ARTIFACT, map_location="cpu", weights_only=False)["samples"]
        return self._samples

    def estimate(self, job: Job) -> float:
        return estimate_minutes(job, self.root, self.all_jobs)

    def remaining_minutes(self) -> float:
        return sum(self.estimate(job) for job in self.jobs if not is_terminal(read_job_record(job, self.root)))

    def progress(self) -> tuple[str, str]:
        done = sum(is_terminal(read_job_record(job, self.root)) for job in self.jobs)
        elapsed = (datetime.now() - self.started).total_seconds() / 3600
        return f"{done}/{len(self.jobs)}", f"{elapsed:.1f} h elapsed, ~{self.remaining_minutes() / 60:.1f} h left"

    def write_record(self, job: Job, record: dict) -> None:
        directory = job.directory(self.root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "job.json").write_text(json.dumps(record, indent=2, default=str))

    def base_record(self, job: Job) -> dict:
        return {"job_id": job.job_id, "group": job.group, "family": job.family, "condition": job.condition,
                "seed": job.seed, "split_seed": 4601, "task": job.task, "label_scheme": "source_subject_identity",
                "head_commit": self.head, "result_dir": str(job.result_dir(self.root))}

    # ---------------------------------------------------------------- preflight
    def preflight(self) -> dict:
        problems = []
        if "envs/fl" not in sys.executable:
            problems.append(f"Python is {sys.executable}, not the conda env `fl`.")
        import flwr
        import ray
        import torch

        for name, module in (("flwr", flwr), ("ray", ray)):
            if module.__version__ != EXPECTED_VERSIONS[name]:
                problems.append(f"{name} {module.__version__} != {EXPECTED_VERSIONS[name]}")
        inputs = (RELABEL_CLASSIFICATION_ARTIFACT, REGRESSION_ARTIFACT,
                  CANONICAL_CENTRALIZED["absolute"] / "training_summary.json",
                  CANONICAL_CENTRALIZED["residual"] / "training_summary.json")
        problems += [f"missing input {path}" for path in inputs if not path.exists()]
        artifacts = (ROOT / "artifacts").resolve()
        canonical = [p.resolve() for p in artifacts.iterdir() if p.is_dir() and p.resolve() != self.root]
        if self.root == artifacts or any(self.root == p or p in self.root.parents for p in canonical):
            problems.append(f"--root {self.root} is, or is inside, another artifacts folder (S5 and canonical "
                            "results are never written).")
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts", "configs"],
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            problems.append(f"Uncommitted changes in src/ scripts/ configs/ (commit before the queue):\n{dirty}")
        self.head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

        from redo_by_sara.combined_iid_flower import (
            load_combined_iid_flower_config,
            prepare_combined_iid_flower_experiment,
        )
        from redo_by_sara.relabel_checks import (
            DIRICHLET_CONDITIONS,
            KEPT_ASSIGNMENT_CONDITIONS,
            load_artifact,
            partition_checks,
        )

        verified, configs = [], {}
        for condition in sorted({j.condition for j in self.jobs if j.family in ("fl", "local_only")}):
            config = load_combined_iid_flower_config(ROOT / f"configs/relabel_{condition}.yaml")
            expected = {"aggregation": "FedAvg", "local_epochs": 5, "client_optimizer_state": "persistent",
                        "client_num_cpus": 3.0, "num_rounds": 60, "split_seed": 4601, "init_seed": 4601,
                        "classification_artifact": RELABEL_CLASSIFICATION_ARTIFACT.resolve(),
                        "regression_artifact": REGRESSION_ARTIFACT.resolve()}
            wrong = {k: config[k] for k, v in expected.items() if config[k] != v}
            if wrong:
                problems.append(f"configs/relabel_{condition}.yaml deviates from the relabel setting: {wrong}")
            # Builds the partition and runs its audits with every write disabled.
            prepare_combined_iid_flower_experiment(config, self.root / "_verify", write_outputs=False)
            configs[condition] = config
            verified.append(condition)
        # The kept S5 regression results rest on these assignments being unchanged.
        all_conditions = {c: load_combined_iid_flower_config(ROOT / f"configs/relabel_{c}.yaml")
                          for c in KEPT_ASSIGNMENT_CONDITIONS + DIRICHLET_CONDITIONS}
        partitions = partition_checks(load_artifact(RELABEL_CLASSIFICATION_ARTIFACT), load_artifact(REGRESSION_ARTIFACT),
                                      all_conditions, S5_ROOT)
        if not partitions["passed"]:
            failed = [c for c, e in partitions["conditions"].items() if e.get("assignment_unchanged") is False]
            problems.append(f"Partition check failed (assignment changed or IID identity property broken): {failed}")
        if problems:
            raise SystemExit("Preflight failed:\n- " + "\n- ".join(problems))
        record = {"time": datetime.now().isoformat(timespec="seconds"), "head_commit": self.head,
                  "python": platform.python_version(), "executable": sys.executable, "machine": platform.machine(),
                  "flwr": flwr.__version__, "ray": ray.__version__, "torch": torch.__version__,
                  "only": self.args.only,
                  "stop_by": None if self.stop_by is None else self.stop_by.isoformat(timespec="minutes"),
                  "fl_configs_verified": verified,
                  "kept_assignments_byte_identical_to_s5": {
                      c: partitions["conditions"][c]["assignment_unchanged"] for c in KEPT_ASSIGNMENT_CONDITIONS},
                  "dirichlet_draws": {c: partitions["conditions"][c]["dirichlet"]["draws"] for c in DIRICHLET_CONDITIONS}}
        state_path = self.root / "queue_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state.setdefault("preflight_history", []).append(record)
        self.root.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
        return record

    # ---------------------------------------------------------------- one job
    def set_aside(self, job: Job, reason: str) -> None:
        directory = job.directory(self.root)
        if directory.exists():
            target = directory.with_name(f"{directory.name}.{reason}-{datetime.now():%Y%m%d-%H%M%S}")
            directory.rename(target)
            self.log(f"{job.job_id}: set aside as {target.name}")

    def run_child(self, job: Job, cmd: list[str]) -> tuple[int, float, str]:
        directory = job.directory(self.root)
        directory.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            cmd = ["caffeinate", "-i", *cmd]
        self.log(f"START {job.job_id}: {' '.join(cmd)}")
        start = time.monotonic()
        log_path = directory / "run.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with log_path.open("w") as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                code = child.wait()
            except Interrupted:
                self.log(f"INTERRUPTED during {job.job_id}; stopping child (SIGINT, then SIGKILL after {CHILD_GRACE_SECONDS} s)")
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=CHILD_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        elapsed = time.monotonic() - start
        self.log(f"END {job.job_id}: exit {code} after {elapsed / 60:.1f} min")
        return code, elapsed, log_path.read_text(errors="replace")

    def finish(self, job: Job, record: dict) -> None:
        self.write_record(job, record)
        rebuild_index(self.all_jobs, self.root)
        count, eta = self.progress()
        self.status(f"{count} {status_text(job, record)} | {eta}")

    def run_job(self, job: Job) -> bool:
        """Process one job. Returns False when the queue must stop before it."""
        record = read_job_record(job, self.root)
        if is_terminal(record):
            return True
        if needs_reference(job) and not (reference_dir(self.root) / "training_summary.json").exists():
            self.status(f"BLOCKED: {job.job_id} needs {CENTRALIZED_REFERENCE_JOB} to have completed "
                        f"({reference_dir(self.root)}); stopping.")
            return False
        directory = job.directory(self.root)
        if directory.exists():
            self.set_aside(job, "crashed" if record else "incomplete")
        estimate = self.estimate(job)
        if not fits_before(datetime.now(), estimate, self.stop_by):
            pending = [j for j in self.jobs if not is_terminal(read_job_record(j, self.root))]
            self.status(
                f"STOP-BY {self.stop_by:%m-%d %H:%M}: not starting {job.job_id} "
                f"(estimate {estimate:.0f} min x {STOP_BY_MARGIN:.2f} would end "
                f"{datetime.fromtimestamp(time.time() + estimate * STOP_BY_MARGIN * 60):%H:%M}). "
                f"Skipped this session: {len(pending)} job(s), ~{self.remaining_minutes() / 60:.1f} h. "
                f"Next in line: {job.job_id}."
            )
            return False

        cmd = command(job, self.root, sys.executable)
        started = datetime.now()
        code, elapsed, log_text = self.run_child(job, cmd)
        outcome = detect_outcome(job, self.root, code, log_text)
        record = {**self.base_record(job), "command": cmd, "started": started.isoformat(timespec="seconds"),
                  "finished": datetime.now().isoformat(timespec="seconds"), "exit_code": code,
                  "wall_seconds": elapsed, "estimate_minutes": estimate, "outcome": outcome}
        try:
            record.update(collect_metrics(job, self.root, outcome, self.samples))
        except Exception as error:  # a metrics bug must not lose the run
            record.update({"metrics": {}, "flat": None, "metrics_error": repr(error)})
            self.log(f"{job.job_id}: metrics collection failed: {error!r}")
        self.finish(job, record)
        return True

    # ---------------------------------------------------------------- dry run
    def dry_run(self) -> None:
        clock, stopped = datetime.now(), False
        print(f"{'grp':>3}  {'job':<52} {'status':<30} {'est':>5}", flush=True)
        totals: dict[int, float] = {}
        for job in self.jobs:
            record = read_job_record(job, self.root)
            status = record["outcome"] if is_terminal(record) else ("incomplete folder" if job.directory(self.root).exists() else "pending")
            est = 0.0 if is_terminal(record) else self.estimate(job)
            note = ""
            if self.stop_by is not None and not is_terminal(record):
                # The queue stops at the first job that does not fit; nothing after it runs.
                stopped = stopped or not fits_before(clock, est, self.stop_by)
                if stopped:
                    note = "  <- after --stop-by"
                else:
                    clock += timedelta(minutes=est)
            totals[job.group] = totals.get(job.group, 0.0) + est
            print(f"{job.group:>3}  {job.job_id:<52} {status:<30} {est:>5.0f}{note}", flush=True)
        print("\nRemaining by group (h): " + ", ".join(f"{g}: {m / 60:.1f}" for g, m in sorted(totals.items()))
              + f" | total {sum(totals.values()) / 60:.1f} h over {len(self.jobs)} jobs", flush=True)

    # ---------------------------------------------------------------- main loop
    def run(self) -> int:
        self.log(f"QUEUE START root={self.root} only={self.args.only} stop_by={self.stop_by} pid={os.getpid()}")
        for job in self.jobs:
            if not self.run_job(job):
                self.log("QUEUE STOPPED before " + job.job_id)
                return 0
        rebuild_index(self.all_jobs, self.root)
        pending = [j.job_id for j in self.jobs if not is_terminal(read_job_record(j, self.root))]
        crashed = [j.job_id for j in self.jobs if (read_job_record(j, self.root) or {}).get("outcome") == "crashed"]
        self.status(f"QUEUE END: {len(self.jobs) - len(pending)}/{len(self.jobs)} terminal; crashed (re-run on restart): {crashed or 'none'}")
        return 1 if crashed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_RESULTS_ROOT), help="Results root (default artifacts/relabel).")
    parser.add_argument("--dry-run", action="store_true", help="Print the job list, status and estimates; run nothing.")
    parser.add_argument("--derived-only", action="store_true", help="Re-score finished jobs; rebuild results_index.csv and derived/.")
    parser.add_argument("--stop-by", default=None, metavar="HH:MM",
                        help="Do not start a job whose estimate would end after the next occurrence of HH:MM.")
    parser.add_argument("--only", default=None, metavar="GLOB[,GLOB]", help="Restrict to job ids matching these globs.")
    args = parser.parse_args()

    queue = Queue(args)
    if args.dry_run:
        queue.dry_run()
        return
    if args.derived_only:
        updated = recompute_metrics(queue.all_jobs, queue.root, queue.samples)
        rebuild_index(queue.all_jobs, queue.root)
        queue.log(f"DERIVED-ONLY: recomputed metrics for {len(updated)} finished job(s)")
        print(f"Recomputed {len(updated)} job(s); rebuilt {queue.root / 'results_index.csv'} and {queue.root / 'derived'}", flush=True)
        return
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        queue.preflight()
        code = queue.run()
    except Interrupted as interruption:
        queue.log(f"QUEUE INTERRUPTED ({interruption}); the running job is left without job.json and re-runs on restart")
        print(f"\nInterrupted ({interruption}). Restart the same command to resume.", flush=True)
        raise SystemExit(130)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

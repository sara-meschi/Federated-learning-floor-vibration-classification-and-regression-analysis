"""Alpha sweep queue: FL K=3 Dirichlet regression across alpha and partition draws, resumable.

Design and analysis are in src/redo_by_sara/alpha_sweep.py and were recorded in CHANGES.md
before any job ran. 15 new jobs, one task each (regression), FedAvg, E=5, persistent client
optimizer, 60 rounds, client_num_cpus 3, split seed 4601:
  (1) draw 1 (partition seed 4601) of alpha = 0.05, 1, 10;
  (2) draw 2 (partition seed 5601) and (3) draw 3 (6601) of every alpha;
  (4) initialization controls: alpha = 0.1 and 0.3, draw 1, init seed 4602.
Draw 1 of alpha = 0.1 and 0.3 at init 4601 is the relabel run, read from artifacts/relabel.

  python scripts/run_alpha_sweep.py --dry-run            # the plan, status and ETA
  python scripts/run_alpha_sweep.py                      # run everything not yet done
  python scripts/run_alpha_sweep.py --stop-by 07:00      # start only jobs that end by 07:00
  python scripts/run_alpha_sweep.py --derived-only       # rebuild results_index.csv and derived/

Restart-safe: finished jobs are skipped; a job killed mid-run is set aside and re-run.
Each job writes only to its own folder under --root (default artifacts/alpha_sweep); the
relabel and S5 results are read, never written. Progress goes to queue.log and one status
line per finished job to stdout and status.log. Children run under `caffeinate -i` on
macOS, which prevents idle sleep but not lid-close sleep: keep the machine open and on power.

Preflight refuses to start unless partition seed 4601 rebuilds the saved relabel alpha = 0.1
and alpha = 0.3 run -> client assignments byte for byte, every new draw meets the
2-identity floor, every config carries the relabel setting, and src/ scripts/ configs/ are
committed.
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

from redo_by_sara.alpha_sweep import (  # noqa: E402
    ALPHAS,
    DEFAULT_RESULTS_ROOT,
    PARTITION_SEEDS,
    PERMUTATION_SEED,
    PERMUTATIONS,
    RELABEL_ALPHAS,
    RELABEL_ROOT,
    RUN_TO_CLIENT,
    SweepJob,
    build_jobs,
    command,
    condition_name,
    config_path,
    draw_of,
    estimate_minutes,
    rebuild_index,
    sweep_runs,
    write_derived,
)
from redo_by_sara.s5_queue import (  # noqa: E402
    CANONICAL_CENTRALIZED,
    STOP_BY_MARGIN,
    collect_metrics,
    detect_outcome,
    fits_before,
    is_terminal,
    parse_stop_by,
    read_job_record,
    select_jobs,
    status_text,
)

EXPECTED_VERSIONS = {"flwr": "1.29.0", "ray": "2.51.1"}
RELABEL_CLASSIFICATION_ARTIFACT = ROOT / "artifacts/relabel/data/classification/raw_windows_combined.pt"
REGRESSION_ARTIFACT = CANONICAL_CENTRALIZED["absolute"] / "raw_walking_windows_regression.pt"
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
    def estimate(self, job: SweepJob) -> float:
        return estimate_minutes(job, self.root, self.all_jobs)

    def remaining_minutes(self) -> float:
        return sum(self.estimate(job) for job in self.jobs if not is_terminal(read_job_record(job, self.root)))

    def progress(self) -> tuple[str, str]:
        done = sum(is_terminal(read_job_record(job, self.root)) for job in self.jobs)
        elapsed = (datetime.now() - self.started).total_seconds() / 3600
        return f"{done}/{len(self.jobs)}", f"{elapsed:.1f} h elapsed, ~{self.remaining_minutes() / 60:.1f} h left"

    def write_record(self, job: SweepJob, record: dict) -> None:
        directory = job.directory(self.root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "job.json").write_text(json.dumps(record, indent=2, default=str))

    def base_record(self, job: SweepJob) -> dict:
        return {"job_id": job.job_id, "group": job.group, "family": job.family, "condition": job.condition,
                "alpha": job.alpha, "role": job.role, "seed": job.seed, "init_seed": job.seed,
                "partition_seed": job.partition_seed, "draw": draw_of(job.partition_seed), "split_seed": 4601,
                "task": job.task, "head_commit": self.head, "result_dir": str(job.result_dir(self.root))}

    def rebuild(self) -> None:
        rebuild_index(self.all_jobs, self.root)
        try:
            write_derived(self.root)
        except Exception as error:  # the analysis must never lose a run
            self.log(f"derived analysis failed: {error!r}")

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
        reused = [run for run in sweep_runs(self.root, RELABEL_ROOT) if run.reused]
        inputs = [RELABEL_CLASSIFICATION_ARTIFACT, REGRESSION_ARTIFACT,
                  CANONICAL_CENTRALIZED["absolute"] / "training_summary.json",
                  CANONICAL_CENTRALIZED["residual"] / "training_summary.json"]
        for run in reused:
            inputs += [run.job_dir / "job.json", run.result_dir / "test_run_predictions.csv",
                       run.job_dir / "partitions" / RUN_TO_CLIENT]
        problems += [f"missing input {path}" for path in inputs if not path.exists()]
        artifacts = (ROOT / "artifacts").resolve()
        others = [p.resolve() for p in artifacts.iterdir() if p.is_dir() and p.resolve() != self.root]
        if self.root == artifacts or any(self.root == p or p in self.root.parents for p in others):
            problems.append(f"--root {self.root} is, or is inside, another artifacts folder (relabel, S5 and "
                            "canonical results are never written).")
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts", "configs"],
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            problems.append(f"Uncommitted changes in src/ scripts/ configs/ (commit before the queue):\n{dirty}")
        self.head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

        from redo_by_sara.combined_iid_fl_partitioning import build_combined_partitions
        from redo_by_sara.combined_iid_flower import (
            load_combined_iid_flower_config,
            prepare_combined_iid_flower_experiment,
        )
        from redo_by_sara.relabel_checks import load_artifact, run_to_client_text

        # Every sweep config is the relabel Dirichlet config with only alpha changed.
        reference = load_combined_iid_flower_config(ROOT / "configs/relabel_k3_dirichlet0.1.yaml")
        ignored = {"config_path", "dirichlet_alpha", "output_dir"}
        for alpha in ALPHAS:
            config = load_combined_iid_flower_config(config_path(alpha))
            wrong = {k: config[k] for k in reference if k not in ignored and config[k] != reference[k]}
            if config["dirichlet_alpha"] != alpha or wrong:
                problems.append(f"{config_path(alpha).name} deviates from the relabel Dirichlet setting: "
                                f"alpha {config['dirichlet_alpha']}, {wrong}")

        classification, regression = load_artifact(RELABEL_CLASSIFICATION_ARTIFACT), load_artifact(REGRESSION_ARTIFACT)
        reproduced = {}
        for alpha in RELABEL_ALPHAS:  # draw 1 must be the relabel partition, byte for byte
            _, summary = build_combined_partitions(classification, regression, "dirichlet", 3, seed=4601,
                                                   dirichlet_alpha=alpha, dirichlet_seed=PARTITION_SEEDS[0])
            saved = next(run for run in reused if run.alpha == alpha).job_dir / "partitions" / RUN_TO_CLIENT
            reproduced[condition_name(alpha)] = run_to_client_text(summary).encode("utf-8") == saved.read_bytes()
            if not reproduced[condition_name(alpha)]:
                problems.append(f"partition seed 4601 does not rebuild {saved} byte for byte")
        draws = {}
        for job in self.jobs:  # every draw meets the floor; the setup audits pass with writes disabled
            config = load_combined_iid_flower_config(
                config_path(job.alpha), {"init_seed": job.seed, "partition_seed": job.partition_seed})
            try:
                prepared = prepare_combined_iid_flower_experiment(config, self.root / "_verify", write_outputs=False)
            except Exception as error:
                problems.append(f"{job.job_id}: setup failed: {error!r}")
                continue
            dirichlet = prepared["setup"]["partition"]["dirichlet"]
            draws[job.job_id] = {"draws": dirichlet["draws"], "floor_applied": dirichlet["floor_applied"],
                                 "client_runs": [c["num_runs"] for c in prepared["setup"]["partition"]["clients"]]}
        if problems:
            raise SystemExit("Preflight failed:\n- " + "\n- ".join(problems))
        record = {"time": datetime.now().isoformat(timespec="seconds"), "head_commit": self.head,
                  "python": platform.python_version(), "executable": sys.executable, "machine": platform.machine(),
                  "flwr": flwr.__version__, "ray": ray.__version__, "torch": torch.__version__,
                  "only": self.args.only,
                  "stop_by": None if self.stop_by is None else self.stop_by.isoformat(timespec="minutes"),
                  "draw1_rebuilds_relabel_partition": reproduced, "dirichlet_draws": draws,
                  "permutations": PERMUTATIONS, "permutation_seed": PERMUTATION_SEED}
        state_path = self.root / "queue_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state.setdefault("preflight_history", []).append(record)
        self.root.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
        for job_id, entry in draws.items():
            self.log(f"PREFLIGHT {job_id}: client runs {entry['client_runs']}, draws {entry['draws']}")
        return record

    # ---------------------------------------------------------------- one job
    def set_aside(self, job: SweepJob, reason: str) -> None:
        directory = job.directory(self.root)
        if directory.exists():
            target = directory.with_name(f"{directory.name}.{reason}-{datetime.now():%Y%m%d-%H%M%S}")
            directory.rename(target)
            self.log(f"{job.job_id}: set aside as {target.name}")

    def run_child(self, job: SweepJob, cmd: list[str]) -> tuple[int, float, str]:
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

    def run_job(self, job: SweepJob) -> bool:
        """Process one job. Returns False when the queue must stop before it."""
        record = read_job_record(job, self.root)
        if is_terminal(record):
            return True
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
            record.update(collect_metrics(job, self.root, outcome, lambda: None))
        except Exception as error:  # a metrics bug must not lose the run
            record.update({"metrics": {}, "flat": None, "metrics_error": repr(error)})
            self.log(f"{job.job_id}: metrics collection failed: {error!r}")
        self.write_record(job, record)
        self.rebuild()
        count, eta = self.progress()
        self.status(f"{count} {status_text(job, record)} | {eta}")
        return True

    # ---------------------------------------------------------------- dry run
    def dry_run(self) -> None:
        clock, stopped = datetime.now(), False
        print(f"{'grp':>3}  {'job':<56} {'role':<13} {'status':<28} {'est':>5}", flush=True)
        total = 0.0
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
            total += est
            print(f"{job.group:>3}  {job.job_id:<56} {job.role:<13} {status:<28} {est:>5.0f}{note}", flush=True)
        print("\nReused from artifacts/relabel (draw 1, init 4601; read-only, not run):", flush=True)
        for run in sweep_runs(self.root, RELABEL_ROOT):
            if run.reused:
                present = "present" if (run.job_dir / "job.json").exists() else "MISSING"
                print(f"     alpha={run.alpha:g}  {run.job_dir.relative_to(ROOT)}  {present}", flush=True)
        print(f"\nRemaining: {total / 60:.1f} h over {sum(not is_terminal(read_job_record(j, self.root)) for j in self.jobs)} "
              f"of {len(self.jobs)} jobs" + (f" | stop-by {self.stop_by:%m-%d %H:%M}" if self.stop_by else ""), flush=True)

    # ---------------------------------------------------------------- main loop
    def run(self) -> int:
        self.log(f"QUEUE START root={self.root} only={self.args.only} stop_by={self.stop_by} pid={os.getpid()}")
        for job in self.jobs:
            if not self.run_job(job):
                self.log("QUEUE STOPPED before " + job.job_id)
                return 0
        self.rebuild()
        pending = [j.job_id for j in self.jobs if not is_terminal(read_job_record(j, self.root))]
        crashed = [j.job_id for j in self.jobs if (read_job_record(j, self.root) or {}).get("outcome") == "crashed"]
        self.status(f"QUEUE END: {len(self.jobs) - len(pending)}/{len(self.jobs)} terminal; crashed (re-run on restart): {crashed or 'none'}")
        return 1 if crashed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_RESULTS_ROOT), help="Results root (default artifacts/alpha_sweep).")
    parser.add_argument("--dry-run", action="store_true", help="Print the job list, status and estimates; run nothing.")
    parser.add_argument("--derived-only", action="store_true", help="Rebuild results_index.csv and derived/ only.")
    parser.add_argument("--stop-by", default=None, metavar="HH:MM",
                        help="Do not start a job whose estimate would end after the next occurrence of HH:MM.")
    parser.add_argument("--only", default=None, metavar="GLOB[,GLOB]", help="Restrict to job ids matching these globs.")
    args = parser.parse_args()

    queue = Queue(args)
    if args.dry_run:
        queue.dry_run()
        return
    if args.derived_only:
        rebuild_index(queue.all_jobs, queue.root)
        analysis = write_derived(queue.root)
        queue.log("DERIVED-ONLY: rebuilt results_index.csv and derived/")
        print(f"Rebuilt {queue.root / 'results_index.csv'} and {queue.root / 'derived'} "
              f"({analysis['confirmatory_runs_measured']}/{analysis['confirmatory_runs_planned']} confirmatory runs measured)",
              flush=True)
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

"""S3 R1 sweep driver — implements docs/s3_winner_rule.md, which predates this file.

Stages (run in order; each resumes, skipping configs whose results already exist):

  1        FedAvg, E in {1, 5, 10} x {fresh, persistent}, regression, IID K=3.
  2        FedAdam at the stage-1 winner's local setting.
  confirm  The overall winner on natural K=2, both tasks, pass/fail, with the
           pre-registered fallback down the IID ranking.
  all      1, 2, confirm.

Every child run uses the committed configs, --allow-degenerate OFF, and on macOS runs
under `caffeinate -i` so a long sweep does not stall when the lid timer fires.

Preflight refuses to start unless the rule doc is tracked and unmodified, src/ scripts/
configs/ have no uncommitted changes, and the persistent-mode determinism test passes
(rule §6). The rule doc's commit hash and HEAD are recorded in the sweep summary.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.r1_sweep import (  # noqa: E402
    RULE_DOC,
    SweepConfig,
    classification_confirmation_passes,
    eligible,
    load_regression_result,
    markdown_table,
    pick_winner,
    rank_eligible,
    result_from_dict,
    result_to_dict,
    stage_1_configs,
    stage_2_config,
)

IID_K3_CONFIG = ROOT / "configs/combined_iid_flower_3clients_60r.yaml"
NATURAL_K2_CONFIG = ROOT / "configs/combined_natural_flower_2clients_60r.yaml"
DETERMINISM_TEST = "test_combined_iid_flower.py::test_persistent_optimizer_runs_are_bitwise_reproducible"


class Escalate(RuntimeError):
    """A pre-registered stop condition: stop and escalate, do not improvise."""


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def preflight(skip_determinism_test: bool) -> dict[str, object]:
    try:
        _git("ls-files", "--error-unmatch", RULE_DOC)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{RULE_DOC} is not tracked; commit the rule before sweeping.") from exc
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", RULE_DOC]).returncode:
        raise SystemExit(f"{RULE_DOC} has uncommitted changes; the rule must be the committed one.")
    dirty = _git("status", "--porcelain", "--", "src", "scripts", "configs")
    if dirty:
        raise SystemExit(f"Uncommitted changes in src/ scripts/ configs/:\n{dirty}\nCommit before sweeping.")
    determinism = "skipped by --skip-determinism-test (not valid for the paper sweep)"
    if not skip_determinism_test:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-rs", DETERMINISM_TEST],
            cwd=ROOT, capture_output=True, text=True,
        )
        tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        if completed.returncode != 0 or "1 passed" not in tail or "skipped" in tail:
            raise SystemExit(f"Determinism precondition (rule §6) failed:\n{completed.stdout}\n{completed.stderr}")
        determinism = tail
    import flwr
    import ray
    import torch

    return {
        "head_commit": _git("rev-parse", "HEAD"),
        "rule_doc": RULE_DOC,
        "rule_doc_commit": _git("log", "-1", "--format=%H", "--", RULE_DOC),
        "determinism_precondition": determinism,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "flwr": flwr.__version__,
        "ray": ray.__version__,
        "torch": torch.__version__,
    }


def run_child(base_config: Path, task: str, out_dir: Path, overrides: dict[str, object]) -> tuple[int, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "scripts/run_combined_iid_flower.py"),
        "--config", str(base_config), "--task", task, "--output-root", str(out_dir),
    ]
    for key, value in overrides.items():
        # The child parses VALUE as YAML; round-trip here so a value that would not
        # survive that (e.g. a string that YAML reads as a number) fails loudly now.
        if yaml.safe_load(str(value)) != value:
            raise ValueError(f"Override {key}={value!r} does not round-trip through YAML.")
        command += ["--set", f"{key}={value}"]
    if sys.platform == "darwin":
        command = ["caffeinate", "-i", *command]
    print(f"[sweep] {' '.join(command)}", flush=True)
    start = time.monotonic()
    with (out_dir / "run.log").open("w") as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.monotonic() - start
    print(f"[sweep]   exit {code} after {elapsed / 60:.1f} min -> {out_dir / 'run.log'}", flush=True)
    return code, elapsed


def run_regression_config(config: SweepConfig, base: Path, root: Path, state: dict) -> None:
    if config.name in state["results"] and state["results"][config.name]["status"] != "crashed":
        print(f"[sweep] {config.name}: already done, reusing", flush=True)
        return
    out = root / config.name
    _, elapsed = run_child(base, "regression", out, config.overrides())
    result = load_regression_result(config, out / "regression", elapsed)
    state["results"][config.name] = result_to_dict(result)
    print(f"[sweep] {config.name}: {result.status}, final std. MSE {result.final_global_train_standardized_mse}", flush=True)


def _results(state: dict, stages: set[str]) -> list:
    return [result_from_dict(r) for r in state["results"].values() if r["stage"] in stages]


def stage_1(root: Path, state: dict) -> None:
    for config in stage_1_configs():
        run_regression_config(config, IID_K3_CONFIG, root, state)
        save(root, state)


def stage_2(root: Path, state: dict) -> None:
    winner = pick_winner(eligible(_results(state, {"stage_1"})))
    if winner is None:
        raise Escalate("No stage-1 config is eligible; there is no local setting for FedAdam (rule §3).")
    state["stage_1_winner"] = winner.name
    run_regression_config(stage_2_config(winner), IID_K3_CONFIG, root, state)
    save(root, state)


def confirm(root: Path, state: dict) -> None:
    ranking = rank_eligible(_results(state, {"stage_1", "stage_2"}))
    state["iid_k3_ranking"] = [r.name for r in ranking]
    if not ranking:
        raise Escalate("No eligible config after stages 1 and 2 (rule §3).")
    confirm_root = root / "confirm_natural_k2"
    reference_dir = confirm_root / "reference_fedavg_e01_fresh"
    reference_summary = reference_dir / "classification" / "training_summary.json"
    if not reference_summary.exists():
        run_child(NATURAL_K2_CONFIG, "classification", reference_dir, {
            "federated.aggregation": "FedAvg", "federated.local_epochs": 1,
            "federated.client_optimizer_state": "fresh",
        })
    if not reference_summary.exists():
        raise Escalate("The natural K=2 E=1/fresh classification reference did not complete.")
    reference_accuracy = float(json.loads(reference_summary.read_text())["final_global_train_metrics"]["accuracy"])
    state["confirmation"] = {"reference_final_global_train_accuracy": reference_accuracy, "attempts": []}
    for candidate in ranking:
        config = SweepConfig(candidate.name, "confirm_natural_k2", candidate.aggregation,
                             candidate.local_epochs, candidate.client_optimizer_state)
        out = confirm_root / candidate.name
        regression_done = any(
            (out / "regression" / name).exists()
            for name in ("training_summary.json", "degenerate_model_check_FAILED.json")
        )
        if not regression_done:
            run_child(NATURAL_K2_CONFIG, "both", out, config.overrides())
        cls_summary = out / "classification" / "training_summary.json"
        if not cls_summary.exists():
            raise Escalate(f"{candidate.name}: natural K=2 classification did not complete.")
        accuracy = float(json.loads(cls_summary.read_text())["final_global_train_metrics"]["accuracy"])
        regression = load_regression_result(config, out / "regression")
        attempt = {
            "config": candidate.name,
            "classification_final_global_train_accuracy": accuracy,
            "classification_passes": classification_confirmation_passes(accuracy, reference_accuracy),
            "regression": result_to_dict(regression),
        }
        state["confirmation"]["attempts"].append(attempt)
        save(root, state)
        if not attempt["classification_passes"]:
            raise Escalate(
                f"{candidate.name}: natural K=2 classification accuracy {accuracy:.4f} is more than "
                f"2 pp below the E=1/fresh reference {reference_accuracy:.4f} (rule §4)."
            )
        if regression.detector_passed:
            state["s5_setting"] = {
                "config": candidate.name, "aggregation": candidate.aggregation,
                "local_epochs": candidate.local_epochs,
                "client_optimizer_state": candidate.client_optimizer_state,
            }
            save(root, state)
            print(f"[sweep] CONFIRMED for S5: {state['s5_setting']}", flush=True)
            return
        print(f"[sweep] {candidate.name} failed regression on natural K=2; falling back (rule §4)", flush=True)
    raise Escalate("No eligible config passed the natural K=2 confirmation (rule §4).")


def save(root: Path, state: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep_summary.json").write_text(json.dumps(state, indent=2))
    results = _results(state, {"stage_1", "stage_2"})
    ranking = rank_eligible(results)
    (root / "sweep_table.md").write_text(markdown_table(results, ranking) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("1", "2", "confirm", "all"), default="all")
    parser.add_argument("--sweep-root", default=str(ROOT / "artifacts/s3_r1_sweep"))
    parser.add_argument("--skip-determinism-test", action="store_true",
                        help="Debugging only; the run is then marked as not valid for the paper.")
    args = parser.parse_args()

    root = Path(args.sweep_root).resolve()
    summary_path = root / "sweep_summary.json"
    state = json.loads(summary_path.read_text()) if summary_path.exists() else {"results": {}}
    state.setdefault("preflight_history", []).append(preflight(args.skip_determinism_test))
    save(root, state)

    stages = {"1": [stage_1], "2": [stage_2], "confirm": [confirm], "all": [stage_1, stage_2, confirm]}
    try:
        for stage in stages[args.stage]:
            stage(root, state)
    except Escalate as escalation:
        state["escalation"] = str(escalation)
        save(root, state)
        print(f"\n[sweep] STOP AND ESCALATE: {escalation}", flush=True)
        raise SystemExit(2) from escalation
    print((root / "sweep_table.md").read_text(), flush=True)


if __name__ == "__main__":
    main()

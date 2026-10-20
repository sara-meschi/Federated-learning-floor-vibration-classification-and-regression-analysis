"""The S3 R1 sweep: config grid, result loading, and the pre-registered winner rule.

The rule implemented here is fixed in ``docs/s3_winner_rule.md``, which was committed
before this code was written. ``pick_winner`` / ``rank_eligible`` read only training-side
quantities (final global-train standardized MSE, step count, local setting) plus the
detector's pass/fail. Test-set metrics are carried on each result for reporting and are
never read by the ranking.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

RULE_DOC = "docs/s3_winner_rule.md"

#: §2: E <= 5 for eligibility; E = 10 only if no E <= 5 config passes.
LOCAL_EPOCH_CAP = 5
#: §3: configs within this of the best final standardized MSE are tied.
TIE_TOLERANCE = 0.01
#: §4: classification fails confirmation if more than this below the E=1/fresh reference.
CLASSIFICATION_ACCURACY_MARGIN = 0.02

STAGE_1_GRID = [
    (local_epochs, state)
    for local_epochs in (1, 5, 10)
    for state in ("fresh", "persistent")
]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    stage: str
    aggregation: str
    local_epochs: int
    client_optimizer_state: str

    def overrides(self) -> dict[str, Any]:
        return {
            "federated.aggregation": self.aggregation,
            "federated.local_epochs": self.local_epochs,
            "federated.client_optimizer_state": self.client_optimizer_state,
        }


def config_name(aggregation: str, local_epochs: int, state: str) -> str:
    return f"{aggregation.lower()}_e{local_epochs:02d}_{state}"


def stage_1_configs() -> list[SweepConfig]:
    return [
        SweepConfig(config_name("FedAvg", e, s), "stage_1", "FedAvg", e, s)
        for e, s in STAGE_1_GRID
    ]


def stage_2_config(stage_1_winner: "SweepResult") -> SweepConfig:
    """FedAdam at the local setting the rule selected from stage 1 (§1)."""
    e, s = stage_1_winner.local_epochs, stage_1_winner.client_optimizer_state
    return SweepConfig(config_name("FedAdam", e, s), "stage_2", "FedAdam", e, s)


@dataclass
class SweepResult:
    name: str
    stage: str
    aggregation: str
    local_epochs: int
    client_optimizer_state: str
    status: str  # "passed" | "failed_detector" | "crashed"
    final_global_train_standardized_mse: float | None
    total_gradient_steps: int | None
    detector_passed: bool
    detector: dict[str, Any] | None = None
    # Reporting only. The ranking never reads these.
    test_metrics: dict[str, Any] = field(default_factory=dict)
    wall_seconds: float | None = None
    output_dir: str | None = None

    def to_row(self) -> dict[str, Any]:
        detector = self.detector or {}
        return {
            "name": self.name,
            "stage": self.stage,
            "aggregation": self.aggregation,
            "local_epochs": self.local_epochs,
            "client_optimizer_state": self.client_optimizer_state,
            "status": self.status,
            "final_global_train_standardized_mse": self.final_global_train_standardized_mse,
            "total_gradient_steps": self.total_gradient_steps,
            "detector_passed": self.detector_passed,
            "distinct_window_residuals": detector.get("distinct_predictions"),
            "skill_vs_subject_mean": detector.get("skill_score_vs_subject_mean"),
            "test_run_rmse_mps": self.test_metrics.get("rmse_mps"),
            "test_run_r2": self.test_metrics.get("r2"),
            "wall_seconds": self.wall_seconds,
        }


def eligible(results: Sequence[SweepResult]) -> list[SweepResult]:
    """§2: all three detector gates passed, and E <= 5 unless no E <= 5 config passes."""
    passing = [r for r in results if r.detector_passed]
    capped = [r for r in passing if r.local_epochs <= LOCAL_EPOCH_CAP]
    return capped if capped else passing


def _simplicity_key(result: SweepResult) -> tuple[int, int, int, str]:
    # §3: fewer steps; then FedAvg over FedAdam; then fresh over persistent; name last
    # only so the order is total and deterministic.
    return (
        int(result.total_gradient_steps),
        0 if result.aggregation == "FedAvg" else 1,
        0 if result.client_optimizer_state == "fresh" else 1,
        result.name,
    )


def pick_winner(results: Sequence[SweepResult]) -> SweepResult | None:
    """§3 applied once to an already-eligible set."""
    if not results:
        return None
    best = min(float(r.final_global_train_standardized_mse) for r in results)
    tied = [
        r
        for r in results
        if float(r.final_global_train_standardized_mse) <= best + TIE_TOLERANCE
    ]
    return min(tied, key=_simplicity_key)


def rank_eligible(results: Sequence[SweepResult]) -> list[SweepResult]:
    """The full §3 order of the eligible set: the fallback list for §4.

    Built by applying the winner rule repeatedly to what remains, so rank 1 is exactly
    ``pick_winner`` and every later rank is the winner among the rest.
    """
    remaining = list(eligible(results))
    order: list[SweepResult] = []
    while remaining:
        winner = pick_winner(remaining)
        order.append(winner)
        remaining = [r for r in remaining if r.name != winner.name]
    return order


def classification_confirmation_passes(
    candidate_final_accuracy: float, reference_final_accuracy: float
) -> bool:
    """§4: fails only if more than 2 percentage points below the E=1/fresh reference."""
    return candidate_final_accuracy >= reference_final_accuracy - CLASSIFICATION_ACCURACY_MARGIN


def _final_history_row(history_path: Path) -> dict[str, str] | None:
    if not history_path.exists():
        return None
    with history_path.open() as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def _s3_detector_passed(detector: dict[str, Any]) -> bool:
    """The three S3 gates (rule §2), computed from the report's numbers.

    Independent of which gates a run *enforced*: S4 split the old "skill" gate into
    "trained" + "skill" and stopped enforcing skill, so the report's ``passed`` flag no
    longer means what the S3 rule meant by passing.
    """
    return (
        bool(detector["gates"]["degeneracy"]["passed"])
        and detector["final_standardized_mse"] is not None
        and float(detector["final_standardized_mse"]) < 0.95
        and float(detector["skill_score_vs_subject_mean"]) > 0.0
    )


def load_regression_result(
    config: SweepConfig, task_dir: Path, wall_seconds: float | None = None
) -> SweepResult:
    """Read one finished regression run: a summary if it passed, the failure record if not."""
    summary_path = task_dir / "training_summary.json"
    failure_path = task_dir / "degenerate_model_check_FAILED.json"
    common = dict(
        name=config.name,
        stage=config.stage,
        aggregation=config.aggregation,
        local_epochs=config.local_epochs,
        client_optimizer_state=config.client_optimizer_state,
        wall_seconds=wall_seconds,
        output_dir=str(task_dir),
    )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        detector = summary["degenerate_model_check"]
        if detector.get("allow_degenerate_override_used"):
            raise AssertionError(f"{config.name} ran under --allow-degenerate; not a sweep run.")
        return SweepResult(
            **common,
            status="passed" if _s3_detector_passed(detector) else "failed_detector",
            final_global_train_standardized_mse=float(detector["final_standardized_mse"]),
            total_gradient_steps=int(
                summary["integrity_audit"]["gradient_step_budget"]["total_gradient_steps"]
            ),
            detector_passed=_s3_detector_passed(detector),
            detector=detector,
            test_metrics=summary["test_run_metrics"],
        )
    if failure_path.exists():
        failure = json.loads(failure_path.read_text())
        detector = failure["degenerate_model_check"]
        return SweepResult(
            **common,
            status="failed_detector",
            final_global_train_standardized_mse=float(detector["final_standardized_mse"]),
            total_gradient_steps=int(
                failure["integrity_audit"]["gradient_step_budget"]["total_gradient_steps"]
            ),
            detector_passed=False,
            detector=detector,
        )
    final_row = _final_history_row(task_dir / "global_train_history.csv")
    return SweepResult(
        **common,
        status="crashed",
        final_global_train_standardized_mse=(
            None if final_row is None else float(final_row["global_train_standardized_mse"])
        ),
        total_gradient_steps=None,
        detector_passed=False,
    )


def result_from_dict(payload: dict[str, Any]) -> SweepResult:
    return SweepResult(**payload)


def result_to_dict(result: SweepResult) -> dict[str, Any]:
    return asdict(result)


def markdown_table(results: Sequence[SweepResult], ranking: Sequence[SweepResult]) -> str:
    rank_of = {r.name: i + 1 for i, r in enumerate(ranking)}
    eligible_names = {r.name for r in eligible(results)}
    header = (
        "| config | stage | E | optimizer state | steps | final std. MSE | distinct residuals "
        "| skill | detector | eligible | rank |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in results:
        row = r.to_row()

        def fmt(value: Any, spec: str) -> str:
            return "—" if value is None else format(value, spec)

        lines.append(
            f"| {r.name} | {r.stage} | {r.local_epochs} | {r.client_optimizer_state} "
            f"| {fmt(row['total_gradient_steps'], 'd')} "
            f"| {fmt(row['final_global_train_standardized_mse'], '.4f')} "
            f"| {fmt(row['distinct_window_residuals'], 'd')} "
            f"| {fmt(row['skill_vs_subject_mean'], '+.4f')} "
            f"| {r.status} | {'yes' if r.name in eligible_names else 'no'} "
            f"| {rank_of.get(r.name, '—')} |"
        )
    return "\n".join(lines)

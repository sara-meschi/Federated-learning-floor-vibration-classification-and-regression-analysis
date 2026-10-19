"""The S3 winner rule (docs/s3_winner_rule.md) on synthetic sweep results.

Thresholds are written out as literals here, not imported, so a change to the rule in
code cannot pass silently: the doc, the code and these tests must all move together.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from redo_by_sara.r1_sweep import (
    SweepConfig,
    SweepResult,
    classification_confirmation_passes,
    eligible,
    load_regression_result,
    pick_winner,
    rank_eligible,
    stage_1_configs,
    stage_2_config,
)


def _result(name, e, state="fresh", mse=0.8, steps=None, passed=True, agg="FedAvg", test=None):
    return SweepResult(
        name=name,
        stage="stage_2" if agg == "FedAdam" else "stage_1",
        aggregation=agg,
        local_epochs=e,
        client_optimizer_state=state,
        status="passed" if passed else "failed_detector",
        final_global_train_standardized_mse=mse,
        total_gradient_steps=1800 * e if steps is None else steps,
        detector_passed=passed,
        test_metrics=test or {},
    )


def test_stage_grids() -> None:
    names = [c.name for c in stage_1_configs()]
    assert names == [
        "fedavg_e01_fresh", "fedavg_e01_persistent",
        "fedavg_e05_fresh", "fedavg_e05_persistent",
        "fedavg_e10_fresh", "fedavg_e10_persistent",
    ]
    fedadam = stage_2_config(_result("fedavg_e05_persistent", 5, "persistent"))
    assert (fedadam.aggregation, fedadam.local_epochs, fedadam.client_optimizer_state) == (
        "FedAdam", 5, "persistent"
    )


def test_failing_configs_are_never_eligible() -> None:
    results = [_result("a", 1, mse=0.99, passed=False), _result("b", 5, mse=0.90)]
    assert [r.name for r in eligible(results)] == ["b"]
    assert pick_winner(eligible(results)).name == "b"


def test_e10_is_capped_while_any_e_le_5_passes() -> None:
    results = [_result("e10", 10, mse=0.50), _result("e05", 5, mse=0.90)]
    assert [r.name for r in eligible(results)] == ["e05"]
    assert pick_winner(eligible(results)).name == "e05"


def test_e10_becomes_eligible_only_when_nothing_at_e_le_5_passes() -> None:
    results = [_result("e10", 10, mse=0.70), _result("e05", 5, mse=0.97, passed=False)]
    assert [r.name for r in eligible(results)] == ["e10"]


def test_no_eligible_config_means_no_winner() -> None:
    results = [_result("a", 1, mse=1.0, passed=False), _result("b", 10, mse=1.0, passed=False)]
    assert eligible(results) == []
    assert pick_winner(eligible(results)) is None
    assert rank_eligible(results) == []


def test_tie_within_0_01_goes_to_fewer_steps() -> None:
    results = [
        _result("e05", 5, mse=0.800),
        _result("e01", 1, mse=0.8099),  # within 0.01 of the best
    ]
    assert pick_winner(eligible(results)).name == "e01"


def test_outside_0_01_is_not_a_tie() -> None:
    results = [_result("e05", 5, mse=0.800), _result("e01", 1, mse=0.8101)]
    assert pick_winner(eligible(results)).name == "e05"


def test_equal_steps_prefers_fedavg_then_fresh() -> None:
    results = [
        _result("adam", 5, "fresh", mse=0.800, agg="FedAdam"),
        _result("persist", 5, "persistent", mse=0.801),
        _result("fresh", 5, "fresh", mse=0.805),
    ]
    assert pick_winner(eligible(results)).name == "fresh"


def test_ranking_never_reads_test_metrics() -> None:
    results = [
        _result("a", 1, mse=0.85, test={"rmse_mps": 0.030, "r2": 0.95}),
        _result("b", 5, mse=0.80, test={"rmse_mps": 0.090, "r2": 0.10}),
        _result("c", 5, "persistent", mse=0.70, test={"rmse_mps": 0.060, "r2": 0.50}),
    ]
    swapped = [
        dataclasses.replace(r, test_metrics={"rmse_mps": -1.0, "r2": 99.0}) for r in results
    ]
    assert [r.name for r in rank_eligible(results)] == [r.name for r in rank_eligible(swapped)]


def test_ranking_is_the_winner_rule_applied_repeatedly() -> None:
    results = [
        _result("e01_fresh", 1, mse=0.92),
        _result("e05_fresh", 5, mse=0.75),
        _result("e05_persist", 5, "persistent", mse=0.70),
        _result("e05_adam", 5, "persistent", mse=0.705, agg="FedAdam"),
        _result("e10", 10, mse=0.40),  # capped
        _result("failed", 1, "persistent", mse=0.99, passed=False),
    ]
    order = [r.name for r in rank_eligible(results)]
    # e05_persist and e05_adam tie at equal steps -> FedAvg first; e10 never appears.
    assert order == ["e05_persist", "e05_adam", "e05_fresh", "e01_fresh"]
    assert order[0] == pick_winner(eligible(results)).name


def test_classification_confirmation_margin_is_2_percentage_points() -> None:
    assert classification_confirmation_passes(0.9700, 0.9900)
    assert classification_confirmation_passes(0.9950, 0.9900)
    assert not classification_confirmation_passes(0.9690, 0.9900)


def test_failed_run_is_loaded_from_its_failure_record(tmp_path: Path) -> None:
    config = SweepConfig("fedavg_e01_fresh", "stage_1", "FedAvg", 1, "fresh")
    (tmp_path / "degenerate_model_check_FAILED.json").write_text(json.dumps({
        "degenerate_model_check": {
            "final_standardized_mse": 1.0000407, "distinct_predictions": 1,
            "skill_score_vs_subject_mean": -0.0028, "passed": False,
        },
        "integrity_audit": {"gradient_step_budget": {"total_gradient_steps": 1800}},
    }))
    result = load_regression_result(config, tmp_path)
    assert result.status == "failed_detector"
    assert result.detector_passed is False
    assert result.final_global_train_standardized_mse == 1.0000407
    assert result.total_gradient_steps == 1800
    assert eligible([result]) == []


def test_run_with_no_outputs_is_recorded_as_crashed(tmp_path: Path) -> None:
    config = SweepConfig("fedavg_e10_fresh", "stage_1", "FedAvg", 10, "fresh")
    result = load_regression_result(config, tmp_path)
    assert result.status == "crashed"
    assert not result.detector_passed

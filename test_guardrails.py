"""Tests for the WP0 guardrails.

Every number appearing here is an independent literal, deliberately not imported from the
library modules: accidental drift must still fail loudly, while a deliberate change stays
a one-line test edit.

The degenerate cases are built from the measured shape of the R1 collapse recorded in
``docs/session0_findings.md``: one distinct predicted residual across 238 test rows,
``global_train_standardized_mse`` flat at ~1.0 for 60 rounds, and a run-level skill score
of -0.0028 against a subject-mean baseline of R^2 0.92464.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from redo_by_sara.guardrails import (
    CANONICAL_DETERMINISM_FLAGS,
    DegenerateModelError,
    GuardedWriter,
    TestSetAccessError,
    TestSetAccessGuard,
    assert_channel_index_conversion,
    assert_determinism_flags,
    check_regression_health,
    determinism_flags,
    seeding_record,
    set_random_seeds,
    skill_score,
)


# ----------------------------------------------------------------------------------
# 1. Seeding parity
# ----------------------------------------------------------------------------------


def test_set_random_seeds_establishes_the_canonical_determinism_flags() -> None:
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    set_random_seeds(4601)
    if torch.backends.cudnn.is_available():
        assert determinism_flags() == CANONICAL_DETERMINISM_FLAGS
        assert determinism_flags() == {
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }
    assert_determinism_flags()


def test_seeding_is_reproducible_across_all_three_rngs() -> None:
    import random as py_random

    set_random_seeds(4601)
    first = (py_random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_random_seeds(4601)
    second = (py_random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_assert_determinism_flags_fires_when_a_run_type_diverges() -> None:
    if not torch.backends.cudnn.is_available():
        pytest.skip("cudnn not available; the flags are unset and unenforced")
    set_random_seeds(4601)
    torch.backends.cudnn.benchmark = True  # what the federated copy used to leave unset
    with pytest.raises(AssertionError, match="Determinism flags differ"):
        assert_determinism_flags()
    set_random_seeds(4601)


def test_seeding_record_reports_what_the_summary_needs() -> None:
    record = seeding_record(4601)
    assert record["seed"] == 4601
    assert "python.random" in record["seeded"]
    assert "numpy.random" in record["seeded"]
    assert "torch" in record["seeded"]
    assert record["canonical_determinism_flags"] == CANONICAL_DETERMINISM_FLAGS


# ----------------------------------------------------------------------------------
# 2. Guarded writing
# ----------------------------------------------------------------------------------


def test_guarded_writer_writes_when_enabled(tmp_path) -> None:
    writer = GuardedWriter(tmp_path / "out", enabled=True)
    writer.mkdir()
    writer.write_text(tmp_path / "out" / "setup_audit.json", "{}")
    called: list[int] = []
    writer.call(lambda: called.append(1))
    assert (tmp_path / "out" / "setup_audit.json").read_text() == "{}"
    assert called == [1]


def test_guarded_writer_is_inert_when_disabled(tmp_path) -> None:
    destination = tmp_path / "out"
    writer = GuardedWriter(destination, enabled=False)
    writer.mkdir()
    writer.write_text(destination / "setup_audit.json", "{}")
    called: list[int] = []
    assert writer.call(lambda: called.append(1)) is None

    # Nothing at all reached the filesystem - not even the directory.
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
    assert called == []

    # ...but the run still reports exactly what it would have produced.
    report = writer.report()
    assert report["writes_enabled"] is False
    assert any("setup_audit.json" in entry for entry in report["planned_writes"])


# ----------------------------------------------------------------------------------
# 3. Degenerate-model detector - two separate gates
# ----------------------------------------------------------------------------------


#: Measured from artifacts/combined_iid_flower_3clients_60r/regression. The baseline is
#: keyed on (source_id, subject_id): test_2 x {001,002,003} and testing_20251124 x
#: {003,004,005,007,008}. Not on direction, which would give 13.
NUM_SOURCE_SUBJECT_STRATA = 8
NUM_TEST_RUNS = 28
NUM_TEST_WINDOWS = 238

#: The single value the collapsed federated model emitted for every one of the 238 test
#: windows. The model was the lookup table minus 0.2 mm/s.
COLLAPSED_RESIDUAL_MPS = -0.0002014615893131122


def _r1_collapse_case() -> dict[str, object]:
    """The federated regression collapse, shaped as it actually occurred.

    The point of the fixture is the gap between the two counts. The residual is **one**
    distinct value across all 238 windows. The reported speed is ``baseline + residual``,
    so it takes exactly as many distinct values as there are baselines — 8, measured — at
    both run and window level.

    Eight is below the threshold of 10, so gating on the speed does catch *this* collapse.
    It catches it by a margin of two, on a quantity that counts strata in the lookup table
    rather than anything the network did. A third building, or a few more subjects, puts
    the stratum count over 10 and a completely constant model walks through. Gating on the
    residual gives 1 regardless of how many strata exist.
    """

    rng = np.random.default_rng(0)
    strata = rng.uniform(1.0, 1.5, size=NUM_SOURCE_SUBJECT_STRATA)
    baseline = strata[np.arange(NUM_TEST_RUNS) % NUM_SOURCE_SUBJECT_STRATA]
    # Centre the noise. Otherwise the skill score is 2c*mean(noise) - c^2 over the
    # baseline variance, and a draw whose mean happens to share the offset's sign makes a
    # constant predictor score *better* than the lookup by luck. With zero-mean noise it
    # is -c^2/var: strictly negative, as a constant offset must be.
    noise = rng.normal(0.0, 0.04, size=NUM_TEST_RUNS)
    actual = baseline + (noise - noise.mean())
    predicted = baseline + COLLAPSED_RESIDUAL_MPS
    assert len(set(predicted)) == NUM_SOURCE_SUBJECT_STRATA  # the dilution, reproduced
    return {
        "actual": actual,
        "predicted": predicted,
        "baseline": baseline,
        "model_output": np.full(NUM_TEST_WINDOWS, COLLAPSED_RESIDUAL_MPS),
        "model_output_name": "window predicted residual",
    }


def _absolute_speed_case() -> dict[str, object]:
    """R2 condition A: trains normally, 238 distinct predictions, skill about -0.75."""

    rng = np.random.default_rng(1)
    baseline = rng.uniform(1.0, 1.5, size=28)
    actual = baseline + rng.normal(0.0, 0.04, size=28)
    # Error roughly 1.32x the baseline error in RMS -> skill about 1 - 1.32^2 = -0.75.
    predicted = actual + rng.normal(0.0, 0.04, size=28) * 1.75
    return {"actual": actual, "predicted": predicted, "baseline": baseline}


def test_r1_collapse_fails_both_gates() -> None:
    case = _r1_collapse_case()
    with pytest.raises(DegenerateModelError) as excinfo:
        check_regression_health(
            **case,
            final_standardized_mse=1.0000406,
            enforced_gates=("degeneracy", "skill"),
            run_label="r1-collapse",
        )
    message = str(excinfo.value)
    # The message names the gated quantity, so a reader can tell at a glance that the
    # detector looked at the residual rather than the baseline-inflated speed.
    assert "1 distinct window predicted residual" in message
    assert "skill score" in message

    report = check_regression_health(
        **case,
        final_standardized_mse=1.0000406,
        enforced_gates=("degeneracy", "skill"),
        run_label="r1-collapse",
        allow_degenerate=True,
    )
    assert report["distinct_predictions"] == 1
    assert report["degeneracy_evaluated_on"] == "window predicted residual"
    assert report["num_model_outputs"] == NUM_TEST_WINDOWS
    assert report["gates"]["degeneracy"]["passed"] is False
    assert report["gates"]["skill"]["passed"] is False
    assert report["passed"] is False
    assert report["allow_degenerate_override_used"] is True
    assert "not a valid result" in report["WARNING"]


def test_gating_on_reported_speed_is_diluted_by_the_baseline_lookup() -> None:
    """Why the degeneracy gate reads the model output and not the reported speed.

    Same collapsed model, gated on the speed instead of the residual. The count becomes
    the number of source/subject strata, which is a property of the lookup table rather
    than of the network. Here that is 8 and the gate still fires — but only just, and only
    because this dataset happens to have fewer than 10 strata.
    """

    case = _r1_collapse_case()
    diluted = check_regression_health(
        actual=case["actual"],
        predicted=case["predicted"],
        baseline=case["baseline"],
        final_standardized_mse=1.0000406,
        enforced_gates=(),  # observe the counts without raising
        run_label="r1-collapse-gated-on-speed",
    )
    assert diluted["distinct_predictions"] == NUM_SOURCE_SUBJECT_STRATA == 8

    # With one more building's worth of strata the same constant model clears the
    # threshold entirely, and a speed-gated detector goes quiet.
    rng = np.random.default_rng(9)
    strata = rng.uniform(1.0, 1.5, size=12)
    baseline = strata[np.arange(36) % 12]
    wider = check_regression_health(
        actual=baseline + rng.normal(0.0, 0.04, size=36),
        predicted=baseline + COLLAPSED_RESIDUAL_MPS,
        baseline=baseline,
        final_standardized_mse=1.0000406,
        enforced_gates=(),
        run_label="r1-collapse-more-strata",
    )
    assert wider["distinct_predictions"] == 12
    assert wider["gates"]["degeneracy"]["passed"] is True  # a constant model, undetected


def test_absolute_speed_model_passes_degeneracy_and_records_negative_skill() -> None:
    """A weak model is not a degenerate one, and must not abort the run."""

    case = _absolute_speed_case()
    report = check_regression_health(
        **case,
        final_standardized_mse=None,
        enforced_gates=("degeneracy",),
        run_label="absolute-speed",
        standardized_mse_not_applicable_reason="absolute target, no residual scale",
    )
    assert report["passed"] is True
    assert report["distinct_predictions"] == 28
    assert report["prediction_std"] > 0
    # The negative skill is recorded even though it is not enforced - that is R2's point.
    assert report["skill_score_vs_subject_mean"] < 0
    assert report["gates"]["skill"]["passed"] is False
    assert report["gates"]["skill"]["enforced"] is False
    assert report["final_standardized_mse"] is None
    assert report["standardized_mse_not_applicable_reason"]
    assert report["allow_degenerate_override_used"] is False
    assert "WARNING" not in report


def test_the_skill_gate_itself_works_when_enforced() -> None:
    """The exemption is the per-run-type setting, not a soft threshold."""

    case = _absolute_speed_case()
    with pytest.raises(DegenerateModelError, match="skill score"):
        check_regression_health(
            **case,
            final_standardized_mse=None,
            enforced_gates=("degeneracy", "skill"),
            run_label="absolute-speed-with-skill-enforced",
        )


def test_healthy_residual_run_passes_both_gates() -> None:
    rng = np.random.default_rng(2)
    baseline = rng.uniform(1.0, 1.5, size=28)
    actual = baseline + rng.normal(0.0, 0.04, size=28)
    predicted = actual + rng.normal(0.0, 0.015, size=28)
    report = check_regression_health(
        actual=actual,
        predicted=predicted,
        baseline=baseline,
        final_standardized_mse=0.693081,
        enforced_gates=("degeneracy", "skill"),
        run_label="healthy",
    )
    assert report["passed"] is True
    assert report["skill_score_vs_subject_mean"] > 0
    assert report["distinct_predictions"] == 28


def test_standardized_mse_at_the_trivial_baseline_fails_the_skill_gate() -> None:
    """Standardized MSE of 1.0 means 'predict zero residual' exactly."""

    rng = np.random.default_rng(3)
    baseline = rng.uniform(1.0, 1.5, size=28)
    actual = baseline + rng.normal(0.0, 0.04, size=28)
    predicted = actual + rng.normal(0.0, 0.015, size=28)
    with pytest.raises(DegenerateModelError, match="standardized MSE"):
        check_regression_health(
            actual=actual,
            predicted=predicted,
            baseline=baseline,
            final_standardized_mse=0.999902,
            enforced_gates=("degeneracy", "skill"),
            run_label="flat-standardized-mse",
        )


def test_unknown_gate_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown regression health gate"):
        check_regression_health(
            actual=[1.0, 2.0],
            predicted=[1.0, 2.0],
            baseline=[1.5, 1.5],
            final_standardized_mse=None,
            enforced_gates=("degenaracy",),  # typo must not silently disable the gate
            run_label="typo",
        )


def test_skill_score_definition() -> None:
    actual = np.asarray([1.0, 2.0, 3.0])
    baseline = np.asarray([2.0, 2.0, 2.0])  # MSE 2/3
    assert skill_score(actual, actual, baseline) == 1.0
    assert skill_score(actual, baseline, baseline) == 0.0
    # Twice the baseline error in RMS -> skill = 1 - 4 = -3.
    doubled = actual + 2.0 * (baseline - actual)
    assert np.isclose(skill_score(actual, doubled, baseline), -3.0)


# ----------------------------------------------------------------------------------
# 4. Test-set access accounting (R9)
# ----------------------------------------------------------------------------------


class _FakeArtifact(dict):
    def __init__(self) -> None:
        super().__init__(test_indices=torch.tensor([0, 1, 2]))


def test_counter_starts_at_zero_and_measures_a_real_access() -> None:
    guard = TestSetAccessGuard(_FakeArtifact(), label="regression")
    assert guard.count == 0
    indices = guard.test_indices("final held-out evaluation")
    assert indices == [0, 1, 2]
    assert guard.count == 1
    report = guard.report()
    assert report["test_evaluation_count"] == 1
    assert report["test_evaluations_during_training"] == 0
    assert report["access_log"][0]["reason"] == "final held-out evaluation"


def test_a_sealed_access_both_raises_and_is_recorded() -> None:
    """The guard prevents as well as measures - the count is not merely advisory."""

    guard = TestSetAccessGuard(_FakeArtifact(), label="classification")
    guard.seal()
    with pytest.raises(TestSetAccessError, match="during training"):
        guard.test_indices("peeking mid-round")
    assert guard.count == 1
    assert len(guard.accesses_while_sealed) == 1
    with pytest.raises(TestSetAccessError, match="touched 1 time"):
        guard.assert_untouched_during_training()


def test_untouched_training_then_one_final_evaluation() -> None:
    guard = TestSetAccessGuard(_FakeArtifact(), label="regression")
    guard.seal()
    guard.assert_untouched_during_training()  # nothing touched it: passes
    guard.unseal()
    guard.test_indices("final held-out evaluation after round 60")
    assert guard.report() == {
        "test_evaluation_count": 1,
        "test_evaluations_during_training": 0,
        "access_log": [
            {
                "reason": "final held-out evaluation after round 60",
                "while_sealed": False,
                "access_number": 1,
            }
        ],
        "measured_by": "redo_by_sara.guardrails.TestSetAccessGuard",
    }


# ----------------------------------------------------------------------------------
# C1 - channel index conversion
# ----------------------------------------------------------------------------------


def test_correct_channel_conversion_passes() -> None:
    assert_channel_index_conversion(
        [1, 2, 3, 4, 5, 6, 7, 8, 10], [0, 1, 2, 3, 4, 5, 6, 7, 9]
    )


def test_off_by_one_conversion_fires() -> None:
    """The failure C1 exists to catch: no shift applied at all."""

    with pytest.raises(AssertionError, match="not a strict one-based"):
        assert_channel_index_conversion(
            [1, 2, 3, 4, 5, 6, 7, 8, 10], [1, 2, 3, 4, 5, 6, 7, 8, 10]
        )


def test_naive_range_conversion_fires() -> None:
    """`range(9)` looks plausible and silently includes the dropped y-axis channel."""

    with pytest.raises(AssertionError):
        assert_channel_index_conversion([1, 2, 3, 4, 5, 6, 7, 8, 10], list(range(9)))


def test_dropped_y_axis_channel_must_stay_dropped() -> None:
    with pytest.raises(AssertionError):
        assert_channel_index_conversion(
            [1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )


def test_wrong_channel_count_fires() -> None:
    with pytest.raises(AssertionError, match="a priori hallway selection"):
        assert_channel_index_conversion(list(range(1, 21)), list(range(20)))

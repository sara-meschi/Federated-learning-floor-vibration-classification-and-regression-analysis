"""Run-level guardrails shared by every combined experiment (WP0 §3).

Four independent concerns live here, deliberately in one module so that later sessions
have one import rather than four:

1. **Seeding parity** — one ``set_random_seeds`` including the cudnn determinism block,
   so federated and centralized runs are seeded identically. Four copies previously
   existed and the federated one silently dropped the determinism block.
2. **Guarded writing** — ``--verify-only`` must write nothing. A verification that
   mutates its outputs is not a verification.
3. **Regression health** — the degenerate-model detector that would have caught R1, run
   automatically at the end of every regression run.
4. **Test-set access accounting** — ``TestSetAccessGuard`` makes "the test set was not
   touched during training" a measurement rather than an assertion about a counter that
   was never incremented (R9).

Plus ``assert_channel_index_conversion``, the C1 assertion, which belongs with the other
things that must be provable rather than merely true.

See ``docs/review_response_plan.md`` §3 and ``CLAUDE.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# 1. Seeding parity (§3 item 1)
# --------------------------------------------------------------------------------------

#: The determinism configuration every run type must establish, whatever its task or
#: framework. Federated and centralized runs previously differed here: the federated
#: ``_set_random_seeds`` set the three RNGs but omitted the cudnn block entirely, so the
#: two run types were not comparable even at a fixed seed.
CANONICAL_DETERMINISM_FLAGS = {
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}


def set_random_seeds(seed: int) -> dict[str, Any]:
    """Seed every RNG the pipeline draws from and pin cudnn to deterministic kernels.

    Returns the seeding record to embed in the run summary, so that a reader can confirm
    two runs were seeded the same way without re-reading the source.

    Note that ``torch.backends.cudnn.is_available()`` can be ``True`` on a CPU-only host
    (it reports whether the build has cudnn, not whether a device exists). Setting the
    flags there is harmless — no cudnn kernel runs — and setting them unconditionally is
    what makes the flags comparable across machines.
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seeding_record(seed)


def determinism_flags() -> dict[str, bool]:
    """The live determinism state, as ``CANONICAL_DETERMINISM_FLAGS`` is keyed."""

    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def assert_determinism_flags() -> None:
    """Fail loudly if this run's determinism flags differ from every other run type's."""

    if not torch.backends.cudnn.is_available():
        return
    observed = determinism_flags()
    if observed != CANONICAL_DETERMINISM_FLAGS:
        raise AssertionError(
            "Determinism flags differ from the canonical configuration shared by every "
            f"run type: observed {observed}, expected {CANONICAL_DETERMINISM_FLAGS}. "
            "Federated and centralized runs must be seeded identically or their "
            "comparison is not controlled."
        )


def seeding_record(seed: int) -> dict[str, Any]:
    """The ``seeding`` block embedded in every run summary."""

    return {
        "seed": int(seed),
        "seeded": ["python.random", "numpy.random", "torch"]
        + (["torch.cuda"] if torch.cuda.is_available() else []),
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "determinism_flags": determinism_flags(),
        "canonical_determinism_flags": dict(CANONICAL_DETERMINISM_FLAGS),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "implementation": "redo_by_sara.guardrails.set_random_seeds",
    }


# --------------------------------------------------------------------------------------
# 2. Guarded writing — `--verify-only` must write nothing (§3 item 2)
# --------------------------------------------------------------------------------------


@dataclass
class GuardedWriter:
    """Routes every output write of a run through one switch.

    When ``enabled`` is ``False`` nothing reaches the filesystem — not a directory, not a
    file, not a plot — and the paths that *would* have been written are recorded so that
    ``--verify-only`` can still report exactly what a real run produces.
    """

    destination: Path
    enabled: bool = True
    planned_writes: list[str] = field(default_factory=list)

    def _record(self, path: Path | str) -> Path:
        path = Path(path)
        self.planned_writes.append(str(path))
        return path

    def mkdir(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path is not None else self.destination
        self._record(target)
        if self.enabled:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def write_text(self, path: Path | str, text: str) -> Path:
        target = self._record(path)
        if self.enabled:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        return target

    def call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke a writer function (a plot, a multi-file saver) only when enabled.

        The function is not called at all in verify mode, so nothing it would create is
        created. Its return value is ``None`` in that case; callers must tolerate that.
        """

        label = getattr(function, "__name__", repr(function))
        self.planned_writes.append(f"{label}(...)")
        if not self.enabled:
            return None
        return function(*args, **kwargs)

    def report(self) -> dict[str, Any]:
        return {
            "destination": str(self.destination),
            "writes_enabled": self.enabled,
            "planned_writes": list(self.planned_writes),
        }


# --------------------------------------------------------------------------------------
# 3. Degenerate-model detector (§3 item 4)
# --------------------------------------------------------------------------------------


class DegenerateModelError(RuntimeError):
    """Raised when a regression run failed an enforced health gate."""


#: A model must produce more than this many distinct predictions to count as a model
#: rather than a lookup table. The federated regression collapse produced exactly one
#: distinct predicted residual across all 238 test rows.
MINIMUM_DISTINCT_PREDICTIONS = 10

#: Standardized MSE of 1.0 means "predict zero residual" — the trivial baseline exactly.
MAXIMUM_STANDARDIZED_MSE = 0.95

#: Skill score = 1 - MSE_model / MSE_baseline. Zero means the model matched the
#: source/subject-mean lookup; negative means it did worse than doing nothing.
MINIMUM_SKILL_SCORE = 0.0

GATE_DEGENERACY = "degeneracy"
GATE_SKILL = "skill"
ALL_GATES = (GATE_DEGENERACY, GATE_SKILL)


def skill_score(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
) -> float:
    """1 - MSE_model / MSE_baseline. The primary regression metric (R2).

    Positive means the model beats the source/subject-mean lookup; zero means it matched
    it; negative means it is worse than doing nothing.
    """

    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    baseline_mse = float(np.mean((baseline - actual) ** 2))
    if baseline_mse <= 1e-15:
        raise ValueError("Baseline MSE is zero; skill score is undefined.")
    model_mse = float(np.mean((predicted - actual) ** 2))
    return float(1.0 - model_mse / baseline_mse)


def check_regression_health(
    *,
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
    final_standardized_mse: float | None,
    enforced_gates: Iterable[str],
    run_label: str,
    model_output: Sequence[float] | np.ndarray | None = None,
    model_output_name: str = "prediction",
    allow_degenerate: bool = False,
    standardized_mse_not_applicable_reason: str | None = None,
) -> dict[str, Any]:
    """Run the degenerate-model detector at the end of a regression run.

    Two gates, deliberately separate, because the four diagnostics do not mean the same
    thing and thresholding them as one block would fail a model that is merely weak:

    ``degeneracy``
        Distinct-prediction count and prediction standard deviation. A constant predictor
        is not a model at all. Enforced on **every** regression path; this is the check
        that would have caught R1.

    ``skill``
        Skill score against the source/subject-mean baseline, and final standardized MSE.
        Enforced only where R1's acceptance criteria apply — the residual paths. The
        absolute-speed model is a legitimately trained model with a skill score around
        -0.75, which is a finding to report (R2), not a run to abort. Standardized MSE is
        defined against the train residual scale and has no meaning on the absolute path
        at all, so it is recorded as ``None`` there rather than thresholded.

    ``enforced_gates`` is always passed explicitly by the caller and never inferred from
    the numbers, so the exemption is a stated property of the run type rather than a soft
    threshold that could quietly stop catching things.

    ``model_output`` is the quantity the network actually emits, and the degeneracy gate
    is evaluated on **that**, not on ``predicted``. On the residual paths the reported
    prediction is ``source_subject_train_mean + residual``, so a completely constant model
    still yields one distinct value per source/subject stratum — eight of them here. That
    dilution is exactly what let the R1 collapse read as a near-match to centralized, so
    gating on it would be gating on the symptom's disguise. Pass the residuals and the
    detector sees the one distinct value that is actually there. Defaults to ``predicted``
    for the absolute path, where the model output *is* the reported speed.

    Always returns the full diagnostics regardless of which gates are enforced, so a
    negative skill score is visible in the summary of a run that legitimately passes.
    """

    enforced = tuple(dict.fromkeys(enforced_gates))
    unknown = set(enforced) - set(ALL_GATES)
    if unknown:
        raise ValueError(f"Unknown regression health gate(s): {sorted(unknown)}")

    predicted_array = np.asarray(predicted, dtype=np.float64).reshape(-1)
    output_array = np.asarray(
        predicted if model_output is None else model_output, dtype=np.float64
    ).reshape(-1)
    distinct = int(np.unique(output_array).size)
    prediction_std = float(np.std(output_array))
    score = skill_score(actual, predicted, baseline)

    degeneracy_failures: list[str] = []
    if distinct <= MINIMUM_DISTINCT_PREDICTIONS:
        degeneracy_failures.append(
            f"only {distinct} distinct {model_output_name}(s) across "
            f"{output_array.size} rows (need > {MINIMUM_DISTINCT_PREDICTIONS}); "
            "this is a constant predictor, not a model"
        )
    if prediction_std <= 0.0:
        degeneracy_failures.append(
            f"{model_output_name} std is {prediction_std} (need > 0)"
        )

    skill_failures: list[str] = []
    if score <= MINIMUM_SKILL_SCORE:
        skill_failures.append(
            f"skill score {score:.6f} vs the source/subject-mean baseline "
            f"(need > {MINIMUM_SKILL_SCORE}); the model is not beating the lookup table"
        )
    if final_standardized_mse is not None and (
        float(final_standardized_mse) >= MAXIMUM_STANDARDIZED_MSE
    ):
        skill_failures.append(
            f"final standardized MSE {float(final_standardized_mse):.6f} "
            f"(need < {MAXIMUM_STANDARDIZED_MSE}); 1.0 means 'predict zero residual'"
        )

    gates = {
        GATE_DEGENERACY: {
            "enforced": GATE_DEGENERACY in enforced,
            "passed": not degeneracy_failures,
            "failures": degeneracy_failures,
        },
        GATE_SKILL: {
            "enforced": GATE_SKILL in enforced,
            "passed": not skill_failures,
            "failures": skill_failures,
        },
    }
    enforced_failures = [
        failure
        for gate_name in enforced
        for failure in gates[gate_name]["failures"]
    ]
    passed = not enforced_failures

    report: dict[str, Any] = {
        "run_label": run_label,
        "num_predictions": int(predicted_array.size),
        "degeneracy_evaluated_on": model_output_name,
        "num_model_outputs": int(output_array.size),
        "distinct_predictions": distinct,
        "prediction_std": prediction_std,
        "final_standardized_mse": (
            None if final_standardized_mse is None else float(final_standardized_mse)
        ),
        "skill_score_vs_subject_mean": score,
        "thresholds": {
            "minimum_distinct_predictions": MINIMUM_DISTINCT_PREDICTIONS,
            "maximum_standardized_mse": MAXIMUM_STANDARDIZED_MSE,
            "minimum_skill_score": MINIMUM_SKILL_SCORE,
        },
        "enforced_gates": list(enforced),
        "gates": gates,
        "passed": passed,
        "allow_degenerate_override_used": bool(allow_degenerate and not passed),
    }
    if final_standardized_mse is None and standardized_mse_not_applicable_reason:
        report["standardized_mse_not_applicable_reason"] = (
            standardized_mse_not_applicable_reason
        )

    if passed:
        print(f"[guardrails] regression health OK ({run_label}): {_summary_line(report)}", flush=True)
        return report

    message = (
        f"Regression run {run_label!r} failed enforced gate(s) "
        f"{[g for g in enforced if not gates[g]['passed']]}: "
        + "; ".join(enforced_failures)
    )
    if not allow_degenerate:
        raise DegenerateModelError(message)

    report["WARNING"] = (
        "This run failed an enforced gate and was allowed to complete under "
        "--allow-degenerate. Its numbers are not a valid result."
    )
    print(
        "\n"
        + "!" * 88
        + f"\n[guardrails] DEGENERATE MODEL ALLOWED THROUGH: {message}"
        + "\n[guardrails] This run's outputs are NOT a valid result. "
        "degenerate_model_check.allow_degenerate_override_used is true in its summary.\n"
        + "!" * 88
        + "\n",
        flush=True,
    )
    return report


def _summary_line(report: dict[str, Any]) -> str:
    mse = report["final_standardized_mse"]
    return (
        f"{report['distinct_predictions']} distinct / {report['num_predictions']} rows, "
        f"std={report['prediction_std']:.6g}, "
        f"skill={report['skill_score_vs_subject_mean']:.6f}, "
        f"standardized_mse={'n/a' if mse is None else format(mse, '.6f')}"
    )


# --------------------------------------------------------------------------------------
# 4. Test-set access accounting (R9)
# --------------------------------------------------------------------------------------


class TestSetAccessError(RuntimeError):
    """Raised when held-out data is reached for while training is in progress."""

    __test__ = False  # a domain name, not a pytest test class


class TestSetAccessGuard:
    """The only route to ``artifact["test_indices"]``, so the count is a measurement.

    Previously ``test_evaluation_counter`` was a local dict that was never incremented,
    so the guard that read it could not fail and ``"test_evaluations_during_training": 0``
    was asserted rather than measured (R9). The finalizers kept a *separate* counter the
    training-phase check never saw.

    One guard instance now spans training and finalization. It is **sealed** for the
    duration of federated training: an access there both records itself and raises, so
    the guard prevents as well as measures.
    """

    __test__ = False  # a domain name, not a pytest test class

    def __init__(self, artifact: dict[str, object], label: str = "") -> None:
        self._artifact = artifact
        self.label = label
        self.sealed = False
        self.accesses: list[dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self.accesses)

    @property
    def accesses_while_sealed(self) -> list[dict[str, Any]]:
        return [access for access in self.accesses if access["while_sealed"]]

    def seal(self) -> None:
        """Close the test set for the duration of training."""

        self.sealed = True

    def unseal(self) -> None:
        """Reopen it for the single post-training evaluation."""

        self.sealed = False

    def test_indices(self, reason: str) -> list[int]:
        record = {
            "reason": reason,
            "while_sealed": bool(self.sealed),
            "access_number": self.count + 1,
        }
        self.accesses.append(record)
        if self.sealed:
            raise TestSetAccessError(
                f"Held-out test data was accessed during training ({reason!r}). "
                "Training must never touch the test set. This is a measured access, "
                f"number {record['access_number']} on guard {self.label!r}."
            )
        return self._artifact["test_indices"].tolist()

    def assert_untouched_during_training(self) -> None:
        breaches = self.accesses_while_sealed
        if breaches:
            raise TestSetAccessError(
                f"Held-out test was touched {len(breaches)} time(s) during federated "
                f"rounds: {breaches}"
            )

    def report(self) -> dict[str, Any]:
        return {
            "test_evaluation_count": self.count,
            "test_evaluations_during_training": len(self.accesses_while_sealed),
            "access_log": list(self.accesses),
            "measured_by": "redo_by_sara.guardrails.TestSetAccessGuard",
        }


# --------------------------------------------------------------------------------------
# C1 — the channel index conversion must be provable, not merely correct
# --------------------------------------------------------------------------------------

#: Selected a priori from physical sensor placement, not from model performance. Sensors
#: 1-7 are one unit each at seven hallway points; channels 8 (x) and 10 (z) are two
#: separate uniaxial units co-located at an eighth point. Channel 9 (y) is dropped. The
#: other 11 of the 20 channels were not in the hallway where walking occurred.
#: See docs/sensor_layout.md.
SELECTED_CHANNELS_ONE_BASED = (1, 2, 3, 4, 5, 6, 7, 8, 10)
SELECTED_CHANNELS_ZERO_BASED = (0, 1, 2, 3, 4, 5, 6, 7, 9)


def assert_channel_index_conversion(
    one_based: Sequence[int], zero_based: Sequence[int]
) -> None:
    """Assert the 1-indexed config -> 0-indexed numpy conversion (C1).

    ``CLAUDE.md`` requires this explicitly because a silent off-by-one shifts every
    channel and is undetectable downstream: the arrays keep their shape, training still
    converges, and every number is quietly wrong.
    """

    one_based = [int(value) for value in one_based]
    zero_based = [int(value) for value in zero_based]

    if one_based != list(SELECTED_CHANNELS_ONE_BASED):
        raise AssertionError(
            f"Selected one-based channels {one_based} != the a priori hallway selection "
            f"{list(SELECTED_CHANNELS_ONE_BASED)}."
        )
    if zero_based != [value - 1 for value in one_based]:
        raise AssertionError(
            f"Channel conversion is not a strict one-based -> zero-based shift: "
            f"{one_based} -> {zero_based}."
        )
    if zero_based != list(SELECTED_CHANNELS_ZERO_BASED):
        raise AssertionError(
            f"Zero-based channel indices {zero_based} != "
            f"{list(SELECTED_CHANNELS_ZERO_BASED)}."
        )
    if len(zero_based) != 9:
        raise AssertionError(f"Expected 9 channels, got {len(zero_based)}.")
    if len(set(zero_based)) != len(zero_based):
        raise AssertionError(f"Duplicate channel indices: {zero_based}.")
    if any(b <= a for a, b in zip(zero_based, zero_based[1:])):
        raise AssertionError(f"Channel indices are not strictly increasing: {zero_based}.")
    if 8 in zero_based:
        raise AssertionError(
            "Zero-based index 8 (one-based channel 9, the y axis of the co-located "
            "position-8 unit) must be dropped; channels 8 (x) and 10 (z) are kept."
        )

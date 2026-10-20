"""The one regression metrics implementation (review plan §2 R2).

Two functions:

``point_metrics(actual, predicted)``
    MSE, RMSE, MAE, bias, R² (total) and the within-0.10 m/s share, plus the median
    absolute error. These are the key names every summary and plot already uses; it
    replaces ``regression_metrics``, ``_extended_metrics``,
    ``_extended_regression_metrics`` and the plot module's ``_regression_metrics``.

``regression_report(actual, predicted, baseline, source_ids, subject_ids)``
    The run-level evaluation every regression output reports. Skill score
    (1 − SSE_model / SSE_baseline) against the train-mean lookup is the primary metric,
    always shown with the baseline. It is reported overall and per source, each with a
    run-level bootstrap 95% CI, plus a per-subject × source breakdown of model and lookup
    RMSE. Groups are taken from the IDs passed in; no subject or source is named here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

#: Run-level bootstrap: 10,000 resamples of test runs with replacement. The seed is
#: fixed and every group restarts the generator from it, so two models evaluated on the
#: same test runs see identical index draws and their intervals are comparable.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 4601
WITHIN_TOLERANCE_MPS = 0.10


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def point_metrics(
    actual: Sequence[float] | np.ndarray, predicted: Sequence[float] | np.ndarray
) -> dict[str, float]:
    actual = _as_array(actual)
    predicted = _as_array(predicted)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError(
            f"actual {actual.shape} and predicted {predicted.shape} must be nonempty and aligned."
        )
    residual = predicted - actual
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 1e-12 else 0.0
    return {
        "mse": mse,
        "rmse_mps": float(np.sqrt(mse)),
        "mae_mps": float(np.mean(np.abs(residual))),
        "bias_mps": float(np.mean(residual)),
        "r2": r2,
        "within_0_10_mps_pct": float(100.0 * np.mean(np.abs(residual) <= WITHIN_TOLERANCE_MPS)),
        "median_absolute_error_mps": float(np.median(np.abs(residual))),
    }


def skill_score_from_arrays(
    actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray
) -> float:
    """1 − SSE_model / SSE_baseline. Zero matches the lookup; negative is worse than it."""
    baseline_sse = float(np.sum((baseline - actual) ** 2))
    if baseline_sse <= 0.0:
        raise ValueError("Baseline SSE is zero; skill score is undefined.")
    return float(1.0 - np.sum((predicted - actual) ** 2) / baseline_sse)


def bootstrap_skill_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile interval of skill over resamples of runs with replacement."""
    rng = np.random.default_rng(seed)
    n = actual.size
    indices = rng.integers(0, n, size=(int(resamples), n))
    sse_model = np.sum((predicted[indices] - actual[indices]) ** 2, axis=1)
    sse_base = np.sum((baseline[indices] - actual[indices]) ** 2, axis=1)
    # A resample whose runs all sit exactly on the baseline has an undefined ratio; it
    # is dropped rather than allowed to produce inf, and the count is visible upstream.
    valid = sse_base > 0.0
    values = 1.0 - sse_model[valid] / sse_base[valid]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _group_block(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
    *,
    resamples: int,
    seed: int,
    with_ci: bool,
) -> dict[str, Any]:
    model_rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((baseline - actual) ** 2)))
    actual_residual = actual - baseline
    centered = float(np.sum((actual_residual - actual_residual.mean()) ** 2))
    block: dict[str, Any] = {
        "num_runs": int(actual.size),
        "baseline_rmse_mps": baseline_rmse,
        "model_rmse_mps": model_rmse,
        "delta_rmse_mps": model_rmse - baseline_rmse,
        "r2_total": point_metrics(actual, predicted)["r2"],
        # Within-subject R²: variance of the residual target (actual − lookup) the model
        # explains, centred on the residual's own test mean. Uncentred, this is skill.
        "r2_within_subject": (
            float(1.0 - np.sum((predicted - actual) ** 2) / centered) if centered > 1e-12 else None
        ),
        "skill": skill_score_from_arrays(actual, predicted, baseline),
    }
    if with_ci:
        block["skill_ci95"] = list(
            bootstrap_skill_ci(actual, predicted, baseline, resamples=resamples, seed=seed)
        )
    return block


def regression_report(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
    source_ids: Sequence[str],
    subject_ids: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run-level regression evaluation against the train-mean lookup.

    One element per held-out **run** (not window): runs are the independent unit, since
    adjacent windows share 4 s of signal. ``baseline`` is each run's source/subject
    train-mean lookup.
    """
    actual = _as_array(actual)
    predicted = _as_array(predicted)
    baseline = _as_array(baseline)
    source_ids = [str(value) for value in source_ids]
    subject_ids = [str(value) for value in subject_ids]
    if not (actual.size == predicted.size == baseline.size == len(source_ids) == len(subject_ids)):
        raise ValueError("actual, predicted, baseline and the id lists must align one-to-one.")

    by_source: dict[str, list[int]] = defaultdict(list)
    by_subject_source: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (source, subject) in enumerate(zip(source_ids, subject_ids)):
        by_source[source].append(index)
        by_subject_source[(source, subject)].append(index)

    def block(indices: list[int] | None, with_ci: bool) -> dict[str, Any]:
        take = slice(None) if indices is None else np.asarray(indices)
        return _group_block(
            actual[take], predicted[take], baseline[take],
            resamples=resamples, seed=seed, with_ci=with_ci,
        )

    per_subject = []
    for (source, subject), indices in sorted(by_subject_source.items()):
        group = block(indices, with_ci=False)
        per_subject.append(
            {
                "source_id": source,
                "subject_id": subject,
                "num_test_runs": group["num_runs"],
                "model_rmse_mps": group["model_rmse_mps"],
                "lookup_rmse_mps": group["baseline_rmse_mps"],
                "model_minus_lookup_rmse_mps": group["delta_rmse_mps"],
            }
        )
    return {
        "unit": "held-out test run",
        "primary_metric": "skill = 1 - SSE_model / SSE_lookup",
        "baseline": "source/subject train-run mean lookup",
        "bootstrap": {
            "resamples": int(resamples),
            "seed": int(seed),
            "interval": "95% percentile; per-source intervals resample within the source",
        },
        "overall": block(None, with_ci=True),
        "per_source": {
            source: block(indices, with_ci=True) for source, indices in sorted(by_source.items())
        },
        "per_subject_source": per_subject,
    }


def per_subject_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-subject × source breakdown as CSV-ready rows."""
    return [dict(row) for row in report["per_subject_source"]]


PER_SUBJECT_FILENAME = "test_per_subject_breakdown.csv"


def report_from_run_rows(rows: Sequence[dict[str, Any]], baseline_key: str) -> dict[str, Any]:
    """``regression_report`` over a run-level prediction table (one row per test run)."""
    uids = [str(row["run_uid"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise AssertionError("regression_report needs one row per run; run_uids repeat.")
    return regression_report(
        actual=[float(row["actual_speed_mps"]) for row in rows],
        predicted=[float(row["predicted_speed_mps"]) for row in rows],
        baseline=[float(row[baseline_key]) for row in rows],
        source_ids=[str(row["source_id"]) for row in rows],
        subject_ids=[str(row["subject_id"]) for row in rows],
    )


def write_per_subject_csv(path: Any, report: dict[str, Any]) -> None:
    import csv
    from pathlib import Path

    rows = per_subject_rows(report)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

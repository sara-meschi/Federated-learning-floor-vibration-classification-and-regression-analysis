"""The consolidated regression metrics (review plan §2 R2).

Values are hand-computed or written out as independent literals, never imported from
the module under test.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np
import pytest

from redo_by_sara.regression_metrics import (
    bootstrap_skill_ci,
    point_metrics,
    regression_report,
    report_from_run_rows,
)

ROOT = Path(__file__).resolve().parent


def test_point_metrics_hand_computed() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.5, 2.0, 2.5, 4.0])
    metrics = point_metrics(actual, predicted)
    assert metrics["mse"] == pytest.approx(0.125)
    assert metrics["rmse_mps"] == pytest.approx(math.sqrt(0.125))
    assert metrics["mae_mps"] == pytest.approx(0.25)
    assert metrics["bias_mps"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1 - 0.5 / 5.0)
    assert metrics["within_0_10_mps_pct"] == pytest.approx(50.0)
    assert metrics["median_absolute_error_mps"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        point_metrics([], [])


def _toy():
    # Two sources, three subjects; IDs are arbitrary on purpose.
    actual = np.array([1.0, 1.2, 1.4, 2.0, 2.2, 3.0])
    baseline = np.array([1.1, 1.1, 1.1, 2.1, 2.1, 2.8])
    predicted = np.array([1.05, 1.15, 1.3, 2.05, 2.15, 2.9])
    sources = ["s1", "s1", "s1", "s1", "s1", "s2"]
    subjects = ["A", "A", "A", "B", "B", "C"]
    return actual, predicted, baseline, sources, subjects


def test_report_overall_numbers() -> None:
    actual, predicted, baseline, sources, subjects = _toy()
    report = regression_report(actual, predicted, baseline, sources, subjects, resamples=500)
    overall = report["overall"]
    sse_model = float(np.sum((predicted - actual) ** 2))
    sse_base = float(np.sum((baseline - actual) ** 2))
    assert overall["num_runs"] == 6
    assert overall["skill"] == pytest.approx(1 - sse_model / sse_base)
    assert overall["model_rmse_mps"] == pytest.approx(math.sqrt(sse_model / 6))
    assert overall["baseline_rmse_mps"] == pytest.approx(math.sqrt(sse_base / 6))
    assert overall["delta_rmse_mps"] == pytest.approx(
        overall["model_rmse_mps"] - overall["baseline_rmse_mps"]
    )
    residual = actual - baseline
    centered = float(np.sum((residual - residual.mean()) ** 2))
    assert overall["r2_within_subject"] == pytest.approx(1 - sse_model / centered)
    low, high = overall["skill_ci95"]
    assert low <= high


def test_report_per_source_and_per_subject() -> None:
    actual, predicted, baseline, sources, subjects = _toy()
    report = regression_report(actual, predicted, baseline, sources, subjects, resamples=200)
    assert sorted(report["per_source"]) == ["s1", "s2"]
    assert report["per_source"]["s1"]["num_runs"] == 5
    rows = {(r["source_id"], r["subject_id"]): r for r in report["per_subject_source"]}
    assert sorted(rows) == [("s1", "A"), ("s1", "B"), ("s2", "C")]
    a = rows[("s1", "A")]
    assert a["num_test_runs"] == 3
    assert a["model_rmse_mps"] == pytest.approx(math.sqrt((0.05**2 + 0.05**2 + 0.1**2) / 3))
    assert a["lookup_rmse_mps"] == pytest.approx(math.sqrt((0.1**2 + 0.1**2 + 0.3**2) / 3))
    assert a["model_minus_lookup_rmse_mps"] == pytest.approx(a["model_rmse_mps"] - a["lookup_rmse_mps"])
    # The breakdown has no CI and no skill per subject: it is a description, not a test.
    assert set(a) == {
        "source_id", "subject_id", "num_test_runs", "model_rmse_mps",
        "lookup_rmse_mps", "model_minus_lookup_rmse_mps",
    }


def test_bootstrap_is_deterministic_and_seed_sensitive() -> None:
    # 28 runs, like the real test set; 6 would give percentiles too coarse to differ.
    rng = np.random.default_rng(0)
    baseline = rng.uniform(1.0, 1.5, size=28)
    actual = baseline + rng.normal(0.0, 0.04, size=28)
    predicted = actual + rng.normal(0.0, 0.03, size=28)
    first = bootstrap_skill_ci(actual, predicted, baseline, resamples=1000, seed=7)
    second = bootstrap_skill_ci(actual, predicted, baseline, resamples=1000, seed=7)
    other = bootstrap_skill_ci(actual, predicted, baseline, resamples=1000, seed=8)
    assert first == second
    assert first != other


def test_misaligned_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        regression_report([1.0, 2.0], [1.0], [1.0, 2.0], ["s", "s"], ["A", "A"])


def test_report_from_rows_refuses_window_level_tables() -> None:
    row = {"run_uid": "x:001:000", "actual_speed_mps": 1.0, "predicted_speed_mps": 1.0,
           "lookup": 1.1, "source_id": "x", "subject_id": "001"}
    with pytest.raises(AssertionError, match="one row per run"):
        report_from_run_rows([row, dict(row)], "lookup")


def test_reproduces_the_s3_centralized_bootstrap() -> None:
    """Same estimator as scripts/analyze_s3_regression_skill.py (CHANGES.md, S3)."""
    path = ROOT / "artifacts/centralized_combined_residual_runbalanced_9ch_no006_60e/test_run_predictions.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    report = report_from_run_rows(rows, "source_subject_train_mean_mps")
    overall = report["overall"]
    assert overall["num_runs"] == 28
    assert round(overall["skill"], 3) == 0.121
    assert [round(v, 3) for v in overall["skill_ci95"]] == [-0.288, 0.235]
    assert round(report["per_source"]["test_2"]["skill"], 3) == -0.042
    assert [round(v, 3) for v in report["per_source"]["testing_20251124"]["skill_ci95"]] == [-0.348, 0.295]
    assert len(report["per_subject_source"]) == 8


def test_metrics_module_names_no_subject() -> None:
    """No subject or source ID is hard-coded: groups come from the IDs passed in."""
    source = (ROOT / "src/redo_by_sara/regression_metrics.py").read_text()
    assert not re.search(r"""["']0\d\d["']""", source)
    assert "test_2" not in source and "20251124" not in source

from __future__ import annotations

import math

import numpy as np

from redo_by_sara.combined_iid_flower import (
    aggregate_confusion_matrices,
    aggregate_regression_sufficient_metrics,
    cosine_round_learning_rate,
)
from redo_by_sara.combined_residual_run_regression import (
    build_zero_initialized_residual_model,
)
from redo_by_sara.federated import get_parameters, set_parameters


def test_cosine_round_learning_rate_matches_declared_schedule() -> None:
    assert cosine_round_learning_rate(1, 1e-3, 1e-5, 60) == 1e-3
    assert math.isclose(
        cosine_round_learning_rate(60, 1e-3, 1e-5, 60),
        1.0678380296485953e-5,
        rel_tol=0.0,
        abs_tol=1e-16,
    )


def test_regression_metrics_are_aggregated_from_global_sufficient_statistics() -> None:
    metrics = aggregate_regression_sufficient_metrics(
        [
            {
                "n": 2,
                "sse": 0,
                "sae": 0,
                "error_sum": 0,
                "within_count": 2,
                "sum_y": 4,
                "sum_y2": 10,
            },
            {
                "n": 1,
                "sse": 4,
                "sae": 2,
                "error_sum": -2,
                "within_count": 0,
                "sum_y": 5,
                "sum_y2": 25,
            },
        ]
    )
    assert math.isclose(metrics["mse"], 4 / 3)
    assert math.isclose(metrics["rmse_mps"], math.sqrt(4 / 3))
    assert math.isclose(metrics["mae_mps"], 2 / 3)
    assert math.isclose(metrics["bias_mps"], -2 / 3)
    assert math.isclose(metrics["within_0_10_mps_pct"], 200 / 3)
    assert math.isclose(metrics["r2"], 0.5)


def test_classification_macro_metrics_are_computed_after_confusion_sum() -> None:
    first = np.asarray([[4, 0], [1, 1]])
    second = np.asarray([[0, 2], [0, 4]])
    matrix, metrics = aggregate_confusion_matrices([first, second], ["a", "b"])
    np.testing.assert_array_equal(matrix, np.asarray([[4, 2], [1, 5]]))
    assert math.isclose(metrics["accuracy"], 0.75)
    assert math.isclose(metrics["balanced_accuracy"], 0.75)
    expected_macro_f1 = ((2 * 0.8 * (4 / 6) / (0.8 + 4 / 6)) + (2 * (5 / 7) * (5 / 6) / (5 / 7 + 5 / 6))) / 2
    assert math.isclose(metrics["macro_f1"], expected_macro_f1)


def test_residual_model_zero_head_and_parameter_roundtrip() -> None:
    model = build_zero_initialized_residual_model()
    initial = get_parameters(model)
    assert np.count_nonzero(initial[-2]) == 0
    assert np.count_nonzero(initial[-1]) == 0
    restored = build_zero_initialized_residual_model()
    set_parameters(restored, initial)
    for expected, actual in zip(initial, get_parameters(restored)):
        np.testing.assert_array_equal(actual, expected)

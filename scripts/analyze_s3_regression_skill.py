"""S3 follow-up: run-level bootstrap CIs and subject-007 sensitivity for regression skill.

No training. Reads each run's ``test_run_predictions.csv`` (28 held-out test runs) and
reports skill = 1 - SSE_model / SSE_baseline against the source/subject train-mean
baseline:

- point estimate and a 95% percentile bootstrap interval, resampling test *runs* with
  replacement (10,000 resamples, fixed seed), overall and per source (resampled within
  the source);
- the same with subject 007's test runs removed. 007's three test runs carry ~73% of the
  baseline's squared error, so the overall skill is mostly a statement about them.

Superseded by the consolidated regression metrics function in S4; kept so the S3
numbers in CHANGES.md can be regenerated exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "artifacts/s3_r1_sweep"
RUNS = {
    "centralized_60e": ROOT / "artifacts/centralized_combined_residual_runbalanced_9ch_no006_60e",
    "centralized_300e_matched_budget": ROOT / "artifacts/centralized_combined_residual_runbalanced_9ch_no006_300e",
    "iid_k3_fedavg_e05_persistent (winner)": SWEEP / "fedavg_e05_persistent/regression",
    "iid_k3_fedavg_e10_persistent": SWEEP / "fedavg_e10_persistent/regression",
    "natural_k2_fedavg_e05_persistent": SWEEP / "confirm_natural_k2/fedavg_e05_persistent/regression",
    "natural_k2_fedadam_e05_persistent": SWEEP / "confirm_natural_k2/fedadam_e05_persistent/regression",
}
SEED = 4601
RESAMPLES = 10_000
EXCLUDED_FOR_SENSITIVITY = "007"


def skill(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> float:
    return float(1.0 - np.sum((predicted - actual) ** 2) / np.sum((baseline - actual) ** 2))


def bootstrap(actual, predicted, baseline, rng) -> tuple[float, float]:
    n = actual.size
    indices = rng.integers(0, n, size=(RESAMPLES, n))
    sse_model = np.sum((predicted[indices] - actual[indices]) ** 2, axis=1)
    sse_base = np.sum((baseline[indices] - actual[indices]) ** 2, axis=1)
    values = 1.0 - sse_model / sse_base
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def analyze(run_dir: Path) -> dict[str, object]:
    with (run_dir / "test_run_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 28:
        raise AssertionError(f"{run_dir}: expected 28 test runs, found {len(rows)}")
    groups = {"overall": rows}
    for source in sorted({r["source_id"] for r in rows}):
        groups[source] = [r for r in rows if r["source_id"] == source]
    groups[f"overall_without_{EXCLUDED_FOR_SENSITIVITY}"] = [
        r for r in rows if r["subject_id"] != EXCLUDED_FOR_SENSITIVITY
    ]
    report = {}
    for name, group in groups.items():
        # One generator per group, seeded identically, so every run is resampled with the
        # same index draws and the intervals are directly comparable across runs.
        rng = np.random.default_rng(SEED)
        actual = np.array([float(r["actual_speed_mps"]) for r in group])
        predicted = np.array([float(r["predicted_speed_mps"]) for r in group])
        baseline = np.array([float(r["source_subject_train_mean_mps"]) for r in group])
        low, high = bootstrap(actual, predicted, baseline, rng)
        report[name] = {
            "num_runs": len(group),
            "skill": skill(actual, predicted, baseline),
            "ci95": [low, high],
            "model_rmse_mps": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "baseline_rmse_mps": float(np.sqrt(np.mean((baseline - actual) ** 2))),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(SWEEP / "regression_skill_bootstrap.json"))
    args = parser.parse_args()
    results = {name: analyze(path) for name, path in RUNS.items() if (path / "test_run_predictions.csv").exists()}
    payload = {"seed": SEED, "resamples": RESAMPLES, "unit": "test run", "results": results}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    columns = list(next(iter(results.values())))
    print("| run | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for name, report in results.items():
        cells = [
            f"{report[c]['skill']:+.3f} [{report[c]['ci95'][0]:+.3f}, {report[c]['ci95'][1]:+.3f}] (n={report[c]['num_runs']})"
            for c in columns
        ]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

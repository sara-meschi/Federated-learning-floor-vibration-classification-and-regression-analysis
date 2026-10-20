"""Short end-to-end runs of every runner, into a temporary output folder.

1-2 epochs or rounds each: the point is that every pipeline runs to completion and
writes its outputs (S4 shipped a crash that only a full re-run caught), not that the
models train. Regression runs this short cannot pass the "trained" gate, so they run
with the degenerate override, which stamps the summary as invalid; that is expected.

The centralized classification and absolute-regression configs pin 60 epochs in their
loader, so those tests load the canonical config, shorten it in memory, and call the
same functions their runner scripts call. The federated and local-only tests go through
the actual CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
pytestmark = pytest.mark.requires_data

CLS_CONFIG = ROOT / "configs/centralized_combined_9ch_no006_nowalking_80_20_60e.yaml"
ABS_CONFIG = ROOT / "configs/centralized_combined_regression_9ch_no006_walking_80_20_60e.yaml"
RES_CONFIG = ROOT / "configs/centralized_combined_residual_runbalanced_9ch_no006_60e.yaml"
NATURAL_K2_CONFIG = ROOT / "configs/combined_natural_flower_2clients_60r.yaml"


def _assert_regression_outputs(directory: Path) -> dict:
    summary = json.loads((directory / "training_summary.json").read_text())
    breakdown = (directory / "test_per_subject_breakdown.csv").read_text().splitlines()
    assert breakdown[0] == (
        "source_id,subject_id,num_test_runs,model_rmse_mps,lookup_rmse_mps,"
        "model_minus_lookup_rmse_mps"
    )
    assert len(breakdown) == 1 + 8  # one row per source x subject test group
    report = summary["regression_report"]
    assert report["overall"]["num_runs"] == 28
    assert len(report["overall"]["skill_ci95"]) == 2
    assert sorted(report["per_source"]) == ["test_2", "testing_20251124"]
    return summary


def test_centralized_classification_end_to_end(tmp_path: Path) -> None:
    from redo_by_sara.combined_centralized_classification import (
        load_combined_config,
        train_and_evaluate,
    )

    config = load_combined_config(CLS_CONFIG)
    artifact = torch.load(config["output_dir"] / config["artifact_name"], map_location="cpu", weights_only=False)
    config["output_dir"] = tmp_path
    config["epochs"] = 1
    train_and_evaluate(artifact, config)
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["epochs"] == 1
    assert (tmp_path / "test_predictions.csv").exists()


def test_centralized_absolute_regression_end_to_end(tmp_path: Path) -> None:
    from redo_by_sara.combined_centralized_regression import (
        load_regression_config,
        train_and_evaluate_regression,
    )

    config = load_regression_config(ABS_CONFIG)
    artifact = torch.load(config["output_dir"] / config["artifact_name"], map_location="cpu", weights_only=False)
    config["output_dir"] = tmp_path
    config["epochs"] = 1
    train_and_evaluate_regression(artifact, config)
    summary = _assert_regression_outputs(tmp_path)
    # Absolute path: only degeneracy is enforced; the trained gate has no meaning here.
    assert summary["degenerate_model_check"]["gates"]["trained"]["enforced"] is False


def test_centralized_residual_regression_end_to_end(tmp_path: Path) -> None:
    from redo_by_sara.combined_residual_run_regression import (
        load_residual_run_config,
        train_and_evaluate_residual_run_model,
    )

    config = load_residual_run_config(RES_CONFIG)
    artifact = torch.load(config["artifact_path"], map_location="cpu", weights_only=False)
    config["output_dir"] = tmp_path
    config["epochs"] = 2
    train_and_evaluate_residual_run_model(artifact, config, allow_degenerate=True)
    summary = _assert_regression_outputs(tmp_path)
    gates = summary["degenerate_model_check"]["gates"]
    assert (gates["trained"]["enforced"], gates["skill"]["enforced"]) == (True, False)


def _run_cli(*args: str) -> None:
    command = [sys.executable, str(ROOT / "scripts/run_combined_iid_flower.py"), *args]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]


def test_federated_runner_end_to_end(tmp_path: Path) -> None:
    _run_cli(
        "--config", str(NATURAL_K2_CONFIG), "--task", "both", "--rounds", "1",
        "--allow-degenerate", "--output-root", str(tmp_path),
    )
    classification = json.loads((tmp_path / "classification/training_summary.json").read_text())
    assert classification["rounds"] == 1
    summary = _assert_regression_outputs(tmp_path / "regression")
    assert summary["degenerate_model_check"]["gates"]["skill"]["enforced"] is False


def test_local_only_runner_end_to_end(tmp_path: Path) -> None:
    _run_cli(
        "--config", str(NATURAL_K2_CONFIG), "--task", "both", "--rounds", "1",
        "--mode", "local_only", "--allow-degenerate", "--output-root", str(tmp_path),
    )
    for task in ("classification", "regression"):
        local = json.loads((tmp_path / "local_only" / task / "local_only_summary.json").read_text())
        assert sorted(local["per_client"]) == ["client_0", "client_1"]
        assert local["failed_clients"] == []
        for client_id in ("client_0", "client_1"):
            client_dir = tmp_path / "local_only" / task / client_id
            if task == "classification":
                assert (client_dir / "training_summary.json").exists()
            else:
                _assert_regression_outputs(client_dir)

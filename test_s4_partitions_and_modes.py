"""S4: Dirichlet label skew, channel-availability partitions, seed separation, local-only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from redo_by_sara.combined_iid_fl_partitioning import (
    _assign_dirichlet,
    _build_run_profiles,
    _largest_remainder,
    build_combined_partitions,
    save_combined_iid_partitions,
)
from redo_by_sara.combined_iid_flower import load_combined_iid_flower_config
from redo_by_sara.guardrails import resolve_seeds
from test_combined_iid_fl_partitioning import _synthetic_artifacts

ROOT = Path(__file__).resolve().parent
IID_K3_CONFIG = ROOT / "configs/combined_iid_flower_3clients_60r.yaml"
NATURAL_K2_CONFIG = ROOT / "configs/combined_natural_flower_2clients_60r.yaml"


# --- Dirichlet ------------------------------------------------------------------------


def test_largest_remainder_sums_and_rounds() -> None:
    assert _largest_remainder(np.array([0.5, 0.3, 0.2]), 10) == [5, 3, 2]
    assert _largest_remainder(np.array([0.34, 0.33, 0.33]), 2) == [1, 1, 0]
    assert sum(_largest_remainder(np.array([0.9, 0.05, 0.05]), 7)) == 7


def test_dirichlet_partition_properties() -> None:
    classification, regression = _synthetic_artifacts()
    partitions, summary = build_combined_partitions(
        classification, regression, "dirichlet", 3, seed=4601, dirichlet_alpha=0.1
    )
    run_sets = [set(p.run_uids) for p in partitions]
    assert not (run_sets[0] & run_sets[1] or run_sets[0] & run_sets[2] or run_sets[1] & run_sets[2])
    assert summary["audits"]["every_client_holds_at_least_min_subjects"] is True
    for client in summary["clients"]:
        assert len(client["subjects"]) >= 2
    dirichlet = summary["dirichlet"]
    assert dirichlet["alpha"] == 0.1
    assert dirichlet["draws"] >= 1
    assert dirichlet["floor_applied"] == (dirichlet["draws"] > 1)


def test_dirichlet_is_seeded_and_redraws_are_recorded() -> None:
    classification, regression = _synthetic_artifacts()
    profiles, *_ = _build_run_profiles(classification, regression)
    first, audit = _assign_dirichlet(profiles, 3, 4601, 0.1)
    again, _ = _assign_dirichlet(profiles, 3, 4601, 0.1)
    assert first == again
    # A floor this demanding (4 subjects, 3 clients, every client >= 2) at alpha = 0.1
    # cannot be met by the first draw for this seed: the redraw count is the record.
    assert audit["draws"] > 1 and audit["floor_applied"] is True
    assert str(audit["draws"] - 1) in audit["floor_note"]


def test_dirichlet_needs_alpha_and_only_dirichlet_takes_one() -> None:
    classification, regression = _synthetic_artifacts()
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        build_combined_partitions(classification, regression, "dirichlet", 3)
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        build_combined_partitions(classification, regression, "iid", 3, dirichlet_alpha=0.1)


# --- Channel availability on the IID partition ----------------------------------------


def test_channel_overlap_keeps_runs_and_adds_masks(tmp_path: Path) -> None:
    classification, regression = _synthetic_artifacts()
    plain, _ = build_combined_partitions(classification, regression, "iid", 3)
    masked, summary = build_combined_partitions(
        classification, regression, "iid", 3, channel_overlap=0.375
    )
    assert [p.run_uids for p in plain] == [p.run_uids for p in masked]
    assert [p.positions for p in masked] == [(1, 2, 3, 4, 5), (1, 2, 3, 6, 7), (1, 2, 3, 8)]
    assert all(len(p.channel_mask) == 9 for p in masked)
    assert summary["channel_availability"]["rho"] == 0.375

    # A plain partition serializes exactly as before channel availability existed.
    paths = save_combined_iid_partitions(plain, {"run_to_client": {}}, tmp_path / "plain")
    assert "channel_mask" not in paths["partitions"].read_text()
    paths = save_combined_iid_partitions(masked, {"run_to_client": {}}, tmp_path / "masked")
    assert json.loads(paths["partitions"].read_text())[2]["positions"] == [1, 2, 3, 8]


def test_channel_overlap_only_on_iid_k3() -> None:
    classification, regression = _synthetic_artifacts()
    with pytest.raises(ValueError, match="IID K=3"):
        build_combined_partitions(classification, regression, "iid", 2, channel_overlap=0.0)
    with pytest.raises(ValueError, match="IID K=3"):
        build_combined_partitions(classification, regression, "natural", 2, channel_overlap=0.0)


# --- Seed separation (R6) -------------------------------------------------------------


def test_resolve_seeds() -> None:
    assert resolve_seeds({"seed": 4601}) == {"split_seed": 4601, "init_seed": 4601}
    assert resolve_seeds({"split_seed": 4601, "init_seed": 4602}) == {
        "split_seed": 4601, "init_seed": 4602,
    }
    with pytest.raises(ValueError, match="not both"):
        resolve_seeds({"seed": 1, "init_seed": 2})
    with pytest.raises(ValueError, match="both split_seed and init_seed"):
        resolve_seeds({"split_seed": 1})


def test_init_seed_override_keeps_the_split() -> None:
    config = load_combined_iid_flower_config(NATURAL_K2_CONFIG, {"init_seed": 4602})
    assert (config["split_seed"], config["init_seed"]) == (4601, 4602)
    legacy = load_combined_iid_flower_config(NATURAL_K2_CONFIG)
    assert (legacy["split_seed"], legacy["init_seed"]) == (4601, 4601)


def test_partitions_follow_the_split_seed_only() -> None:
    classification, regression = _synthetic_artifacts()
    a, _ = build_combined_partitions(classification, regression, "iid", 3, seed=4601)
    b, _ = build_combined_partitions(classification, regression, "iid", 3, seed=4601)
    c, _ = build_combined_partitions(classification, regression, "iid", 3, seed=17)
    assert a == b
    assert a != c


def test_s5_config_settings_validate() -> None:
    for overrides in (
        {"federated.channel_overlap": 0.375},
        {"federated.channel_overlap": 0.0},
        {"federated.partition_scheme": "dirichlet", "federated.dirichlet_alpha": 0.1},
    ):
        config = load_combined_iid_flower_config(IID_K3_CONFIG, overrides)
        assert config["num_clients"] == 3


@pytest.mark.requires_data
def test_artifact_split_seed_must_match_config() -> None:
    from redo_by_sara.combined_iid_flower import audit_combined_iid_flower_setup

    config = load_combined_iid_flower_config(NATURAL_K2_CONFIG, {"split_seed": 1234})
    with pytest.raises(AssertionError, match="split with seed 4601"):
        audit_combined_iid_flower_setup(config)


# --- Local-only (R5) ------------------------------------------------------------------


@pytest.mark.requires_data
def test_local_only_matches_the_federated_per_client_budget(tmp_path: Path) -> None:
    """Same client code, same rounds x E, no aggregation; evaluated on the global test."""
    from redo_by_sara.combined_iid_flower import run_local_only_experiment

    config = load_combined_iid_flower_config(
        NATURAL_K2_CONFIG,
        {"federated.local_epochs": 2, "federated.client_optimizer_state": "persistent"},
    )
    summary = run_local_only_experiment(
        config, "regression", num_rounds_override=2, output_root=tmp_path, allow_degenerate=True
    )
    assert sorted(summary["per_client"]) == ["client_0", "client_1"]
    for client_id, runs in (("client_0", 52), ("client_1", 60)):
        client_summary = json.loads(
            (tmp_path / "local_only/regression" / client_id / "training_summary.json").read_text()
        )
        budget = client_summary["integrity_audit"]["gradient_step_budget"]
        # ceil(runs / 4) steps per epoch x 2 epochs x 2 rounds, measured == formula.
        assert budget["total_gradient_steps"] == -(-runs // 4) * 2 * 2
        assert budget["measured_total_gradient_steps"] == budget["total_gradient_steps"]
        assert client_summary["integrity_audit"]["test_evaluations_during_training"] == 0
        assert client_summary["regression_report"]["overall"]["num_runs"] == 28
        assert "local_only_client" in client_summary["experiment"] or client_id in client_summary["experiment"]

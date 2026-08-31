from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from redo_by_sara.combined_iid_fl_partitioning import (
    EXPECTED_CLASSIFICATION_WINDOW_COUNTS,
    EXPECTED_NO_WALKING_WINDOW_COUNTS,
    EXPECTED_REGRESSION_WINDOW_COUNTS,
    EXPECTED_RUN_COUNTS,
    EXPECTED_SOURCE_SUBJECT_GROUPS,
    EXPECTED_SUBJECTS,
    build_combined_iid_partitions,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFICATION_ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/centralized_combined_9ch_no006_nowalking_80_20_60e/raw_windows_combined.pt"
)
REGRESSION_ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/centralized_combined_regression_9ch_no006_walking_80_20_60e/"
    "raw_walking_windows_regression.pt"
)


@lru_cache(maxsize=1)
def _result():
    classification = torch.load(
        CLASSIFICATION_ARTIFACT, map_location="cpu", weights_only=False
    )
    regression = torch.load(REGRESSION_ARTIFACT, map_location="cpu", weights_only=False)
    return classification, regression, *build_combined_iid_partitions(
        classification, regression, num_clients=3, seed=4601
    )


def test_exact_client_counts_and_complete_window_coverage() -> None:
    classification, regression, partitions, summary = _result()
    assert tuple(len(partition.run_uids) for partition in partitions) == EXPECTED_RUN_COUNTS
    assert tuple(len(partition.regression_indices) for partition in partitions) == (
        EXPECTED_REGRESSION_WINDOW_COUNTS
    )
    assert tuple(len(partition.classification_indices) for partition in partitions) == (
        EXPECTED_CLASSIFICATION_WINDOW_COUNTS
    )
    assert tuple(
        client["num_no_walking_windows"] for client in summary["clients"]
    ) == EXPECTED_NO_WALKING_WINDOW_COUNTS

    owned_classification = [
        index for partition in partitions for index in partition.classification_indices
    ]
    owned_regression = [
        index for partition in partitions for index in partition.regression_indices
    ]
    assert len(owned_classification) == len(set(owned_classification)) == 1242
    assert len(owned_regression) == len(set(owned_regression)) == 966
    assert set(owned_classification) == set(classification["train_indices"].tolist())
    assert set(owned_regression) == set(regression["train_indices"].tolist())


def test_no_run_leakage_and_same_owner_for_both_tasks() -> None:
    classification, regression, partitions, summary = _result()
    client_run_sets = [set(partition.run_uids) for partition in partitions]
    assert not (client_run_sets[0] & client_run_sets[1])
    assert not (client_run_sets[0] & client_run_sets[2])
    assert not (client_run_sets[1] & client_run_sets[2])

    classification_test_uids = {
        classification["metadata"][int(index)]["run_uid"]
        for index in classification["test_indices"]
    }
    regression_test_uids = {
        regression["metadata"][int(index)]["run_uid"]
        for index in regression["test_indices"]
    }
    assert classification_test_uids == regression_test_uids
    assert not set().union(*client_run_sets) & classification_test_uids
    assert summary["audits"]["pairwise_client_run_overlap"] == 0
    assert summary["audits"]["train_test_run_overlap"] == 0
    assert summary["audits"]["same_run_owner_for_both_tasks"] is True

    for partition in partitions:
        classification_uids = {
            classification["metadata"][index]["run_uid"]
            for index in partition.classification_indices
        }
        regression_uids = {
            regression["metadata"][index]["run_uid"]
            for index in partition.regression_indices
        }
        assert classification_uids == regression_uids == set(partition.run_uids)


def test_every_client_has_all_subject_source_and_direction_strata() -> None:
    _, _, _, summary = _result()
    for client in summary["clients"]:
        assert client["subjects"] == list(EXPECTED_SUBJECTS)
        assert client["source_subject_groups"] == list(EXPECTED_SOURCE_SUBJECT_GROUPS)
        assert client["direction_run_counts"]["testing_20251124:N_to_S"] == 10
        assert client["direction_run_counts"]["testing_20251124:S_to_N"] == 10
    for block in summary["speed_blocks"]:
        if block["size"] == 3:
            assert set(block["run_to_client"].values()) == {
                "client_0",
                "client_1",
                "client_2",
            }


def test_partition_is_deterministic() -> None:
    classification, regression, original, original_summary = _result()
    rebuilt, rebuilt_summary = build_combined_iid_partitions(
        classification, regression, num_clients=3, seed=4601
    )
    assert rebuilt == original
    assert rebuilt_summary["run_to_client"] == original_summary["run_to_client"]


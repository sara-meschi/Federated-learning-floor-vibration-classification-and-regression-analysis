"""Partitioner tests.

The fast tests build small synthetic artifact pairs and check *properties* at K=2 and
K=3, for both schemes. The ``requires_data`` tests pin the real numbers as independent
literals: they are written out here, never imported from the module, so accidental
drift fails loudly and a deliberate change is a one-line edit.
"""

from __future__ import annotations

import copy
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import yaml

from redo_by_sara.combined_iid_fl_partitioning import (
    build_combined_iid_partitions,
    build_combined_partitions,
)
from redo_by_sara.combined_iid_flower import (
    audit_combined_iid_flower_setup,
    load_combined_iid_flower_config,
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
IID_CONFIG = PROJECT_ROOT / "configs/combined_iid_flower_3clients_60r.yaml"
NATURAL_CONFIG = PROJECT_ROOT / "configs/combined_natural_flower_2clients_60r.yaml"


# --------------------------------------------------------------------------------------
# Synthetic artifacts
# --------------------------------------------------------------------------------------

#: (source, subject, direction) -> (train runs, test runs). Shaped like the real data:
#: Test_2 unidirectional, the 2025 campaign alternating, subject 003 in both.
SYNTHETIC_STRATA = {
    ("test_2", "001", "single_direction_unknown"): (7, 2),
    ("test_2", "002", "single_direction_unknown"): (8, 2),
    ("test_2", "003", "single_direction_unknown"): (7, 2),
    ("testing_20251124", "003", "N_to_S"): (4, 1),
    ("testing_20251124", "003", "S_to_N"): (5, 1),
    ("testing_20251124", "004", "N_to_S"): (4, 1),
    ("testing_20251124", "004", "S_to_N"): (3, 1),
}
WALKING_WINDOWS = 3
NO_WALKING_WINDOWS = 2


def _synthetic_artifacts(strata=SYNTHETIC_STRATA):
    classification = {"metadata": [], "train_indices": [], "test_indices": []}
    regression = {"metadata": [], "train_indices": [], "test_indices": []}
    run_counter: Counter[tuple[str, str]] = Counter()
    for (source, subject, direction), (num_train, num_test) in strata.items():
        for position in range(num_train + num_test):
            run_index = run_counter[(source, subject)]
            run_counter[(source, subject)] += 1
            split = "train" if position < num_train else "test"
            base = {
                "source_id": source,
                "subject_id": subject,
                "run_index": run_index,
                "run_uid": f"{source}:{subject}:{run_index:03d}",
                "direction": direction,
                "split": split,
            }
            speed = 1.0 + 0.01 * position + 0.001 * int(subject)
            for window in range(WALKING_WINDOWS + NO_WALKING_WINDOWS):
                walking = window < WALKING_WINDOWS
                item = {
                    **base,
                    "activity": f"subject_{subject}_walking" if walking else "no_walking",
                    "start_time": float(window),
                }
                classification[f"{split}_indices"].append(len(classification["metadata"]))
                classification["metadata"].append(item)
                if walking:
                    regression[f"{split}_indices"].append(len(regression["metadata"]))
                    regression["metadata"].append(
                        {**base, "start_time": float(window), "speed_mps": speed}
                    )
    return classification, regression


def _assert_common(partitions, summary, classification, regression):
    run_sets = [set(p.run_uids) for p in partitions]
    for left in range(len(run_sets)):
        for right in range(left + 1, len(run_sets)):
            assert not run_sets[left] & run_sets[right]
    train_indices = [int(i) for i in classification["train_indices"]]
    test_indices = [int(i) for i in classification["test_indices"]]
    train_uids = {classification["metadata"][i]["run_uid"] for i in train_indices}
    test_uids = {classification["metadata"][i]["run_uid"] for i in test_indices}
    assert set().union(*run_sets) == train_uids
    assert not set().union(*run_sets) & test_uids
    owned_cls = [i for p in partitions for i in p.classification_indices]
    owned_reg = [i for p in partitions for i in p.regression_indices]
    assert sorted(owned_cls) == sorted(train_indices)
    assert sorted(owned_reg) == sorted(int(i) for i in regression["train_indices"])
    for partition in partitions:
        cls_runs = {
            classification["metadata"][i]["run_uid"] for i in partition.classification_indices
        }
        reg_runs = {regression["metadata"][i]["run_uid"] for i in partition.regression_indices}
        assert cls_runs == reg_runs == set(partition.run_uids)
    assert summary["audits"]["pairwise_client_run_overlap"] == 0
    assert summary["audits"]["train_test_run_overlap"] == 0
    assert summary["audits"]["same_run_owner_for_both_tasks"] is True


# --------------------------------------------------------------------------------------
# IID — properties at K=2 and K=3
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_clients", [2, 3])
def test_iid_partition_properties(num_clients: int) -> None:
    classification, regression = _synthetic_artifacts()
    partitions, summary = build_combined_partitions(
        classification, regression, scheme="iid", num_clients=num_clients, seed=4601
    )
    assert len(partitions) == num_clients
    assert [p.client_id for p in partitions] == [f"client_{i}" for i in range(num_clients)]
    _assert_common(partitions, summary, classification, regression)

    run_counts = [len(p.run_uids) for p in partitions]
    assert max(run_counts) - min(run_counts) <= 1

    meta = {
        m["run_uid"]: (m["source_id"], m["subject_id"], m["direction"])
        for m in classification["metadata"]
        if m["split"] == "train"
    }
    subject_totals = Counter(stratum[1] for stratum in meta.values())
    for partition in partitions:
        strata = [meta[uid] for uid in partition.run_uids]
        assert set(strata) == set(SYNTHETIC_STRATA)  # every stratum, both directions
        held = Counter(s[1] for s in strata)
        assert set(held) == set(subject_totals)  # >= 1 run of every subject
        assert all(held[s] < subject_totals[s] for s in held)  # never all runs of one

    for block in summary["speed_blocks"]:
        if block["block_number"] is not None:
            assert len(set(block["run_to_client"].values())) == num_clients
    assert summary["scheme"] == "iid"
    assert summary["stratification_fields"] == ["source_id", "subject_id", "direction"]


def test_iid_partition_is_deterministic_and_seeded() -> None:
    classification, regression = _synthetic_artifacts()
    first = build_combined_partitions(classification, regression, num_clients=3, seed=4601)
    again = build_combined_partitions(classification, regression, num_clients=3, seed=4601)
    other = build_combined_partitions(classification, regression, num_clients=3, seed=7)
    assert first[0] == again[0]
    assert first[1]["run_to_client"] == again[1]["run_to_client"]
    assert first[1]["run_to_client"] != other[1]["run_to_client"]


def test_iid_wrapper_matches_the_scheme_builder() -> None:
    classification, regression = _synthetic_artifacts()
    assert build_combined_iid_partitions(
        classification, regression, num_clients=2
    )[0] == build_combined_partitions(classification, regression, "iid", num_clients=2)[0]


def test_iid_refuses_a_stratum_smaller_than_k() -> None:
    strata = dict(SYNTHETIC_STRATA)
    strata[("testing_20251124", "004", "S_to_N")] = (2, 1)
    classification, regression = _synthetic_artifacts(strata)
    with pytest.raises(AssertionError, match="fewer than K=3"):
        build_combined_partitions(classification, regression, num_clients=3)
    build_combined_partitions(classification, regression, num_clients=2)


# --------------------------------------------------------------------------------------
# Natural — one client per campaign; IID balance and coverage deliberately not required
# --------------------------------------------------------------------------------------


def test_natural_partition_is_one_client_per_campaign() -> None:
    classification, regression = _synthetic_artifacts()
    partitions, summary = build_combined_partitions(
        classification, regression, scheme="natural", num_clients=2
    )
    _assert_common(partitions, summary, classification, regression)
    assert summary["client_sources"] == {"client_0": "test_2", "client_1": "testing_20251124"}
    by_client = {p.client_id: p for p in partitions}
    assert {uid.split(":")[0] for uid in by_client["client_0"].run_uids} == {"test_2"}
    assert {uid.split(":")[0] for uid in by_client["client_1"].run_uids} == {"testing_20251124"}

    # The heterogeneity is the point: uneven sizes, incomplete subject coverage.
    clients = {c["client_id"]: c for c in summary["clients"]}
    assert clients["client_0"]["subjects"] == ["001", "002", "003"]
    assert clients["client_1"]["subjects"] == ["003", "004"]
    assert clients["client_0"]["num_runs"] != clients["client_1"]["num_runs"]
    assert "every_client_holds_every_subject" not in summary["audits"]
    assert summary["audits"]["one_source_per_client"] is True


@pytest.mark.parametrize(
    ("scheme", "num_clients", "message"),
    [
        ("natural", 3, "one client per source"),
        ("iid", 4, "num_clients must be 2 or 3"),
        ("iid", 1, "num_clients must be 2 or 3"),
        ("dirichlet", 3, "Unknown partition scheme"),
    ],
)
def test_invalid_scheme_or_k_is_refused(scheme: str, num_clients: int, message: str) -> None:
    classification, regression = _synthetic_artifacts()
    with pytest.raises(ValueError, match=message):
        build_combined_partitions(classification, regression, scheme, num_clients=num_clients)


# --------------------------------------------------------------------------------------
# Leakage — the partitioner re-checks the split at load time
# --------------------------------------------------------------------------------------


def test_a_leaky_split_fires_at_partition_load_time() -> None:
    classification, regression = _synthetic_artifacts()
    leaky = copy.deepcopy(classification)
    # Move one window of a training run to the test side: two windows of one run now
    # sit on opposite sides of the split.
    moved = leaky["train_indices"].pop(0)
    leaky["metadata"][moved]["split"] = "test"
    leaky["test_indices"].append(moved)
    with pytest.raises(AssertionError, match="run leakage"):
        build_combined_partitions(leaky, regression, num_clients=3)


def test_excluded_subject_in_training_runs_fires() -> None:
    strata = dict(SYNTHETIC_STRATA)
    strata[("testing_20251124", "006", "N_to_S")] = (3, 1)
    classification, regression = _synthetic_artifacts(strata)
    with pytest.raises(AssertionError, match="Excluded subjects"):
        build_combined_partitions(classification, regression, num_clients=3)


# --------------------------------------------------------------------------------------
# Federated config validator
# --------------------------------------------------------------------------------------


def _write_config(tmp_path: Path, **federated_overrides) -> Path:
    raw = yaml.safe_load(IID_CONFIG.read_text())
    raw["federated"].update(federated_overrides)
    path = tmp_path / "configs" / "flower.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw))
    return path


def test_validator_accepts_k_rounds_and_epochs_as_parameters(tmp_path: Path) -> None:
    config = load_combined_iid_flower_config(
        _write_config(tmp_path, num_clients=2, num_rounds=5, local_epochs=3)
    )
    assert (config["num_clients"], config["num_rounds"], config["local_epochs"]) == (2, 5, 3)
    assert config["partition_scheme"] == "iid"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_clients": 4}, "num_clients must be 2 or 3"),
        ({"partition_scheme": "natural", "num_clients": 3}, "num_clients must be 2"),
        ({"partition_scheme": "dirichlet"}, "Unknown partition_scheme"),
        ({"num_rounds": 0}, "at least 1"),
    ],
)
def test_validator_refuses_invalid_settings(tmp_path: Path, overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_combined_iid_flower_config(_write_config(tmp_path, **overrides))


def test_shipped_configs_declare_their_scheme() -> None:
    iid = load_combined_iid_flower_config(IID_CONFIG)
    natural = load_combined_iid_flower_config(NATURAL_CONFIG)
    assert (iid["partition_scheme"], iid["num_clients"]) == ("iid", 3)
    assert (natural["partition_scheme"], natural["num_clients"]) == ("natural", 2)


# --------------------------------------------------------------------------------------
# Real artifacts — exact numbers as independent literals
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _artifacts():
    classification = torch.load(CLASSIFICATION_ARTIFACT, map_location="cpu", weights_only=False)
    regression = torch.load(REGRESSION_ARTIFACT, map_location="cpu", weights_only=False)
    return classification, regression


@lru_cache(maxsize=None)
def _real(scheme: str, num_clients: int):
    classification, regression = _artifacts()
    return build_combined_partitions(
        classification, regression, scheme=scheme, num_clients=num_clients, seed=4601
    )


@pytest.mark.requires_data
def test_canonical_split_reference_numbers() -> None:
    classification, regression = _artifacts()
    cls_summary = classification["summary"]
    assert cls_summary["num_recorded_runs"] == 161
    assert cls_summary["run_status_counts"] == {
        "excluded": 15, "skipped": 6, "test": 28, "train": 112
    }
    assert cls_summary["num_usable_runs"] == 140
    assert (
        len(classification["metadata"]), cls_summary["num_train"], cls_summary["num_test"]
    ) == (1549, 1242, 307)
    assert (
        len(regression["metadata"]),
        regression["summary"]["num_train"],
        regression["summary"]["num_test"],
    ) == (1204, 966, 238)
    test_support = torch.bincount(
        classification["classification_targets"][classification["test_indices"]],
        minlength=len(classification["index_to_class"]),
    ).tolist()
    assert test_support == [69, 32, 43, 56, 25, 25, 27, 30]


@pytest.mark.requires_data
@pytest.mark.parametrize(
    ("scheme", "num_clients", "runs", "classification", "regression", "no_walking"),
    [
        # The pre-S2 brute-force partitioner pinned (37, 38, 37) / (412, 418, 412) /
        # (321, 324, 321) / (91, 94, 91). Those are retired; see CHANGES.md.
        ("iid", 3, (37, 38, 37), (413, 421, 408), (321, 328, 317), (92, 93, 91)),
        ("iid", 2, (56, 56), (618, 624), (480, 486), (138, 138)),
        ("natural", 2, (52, 60), (587, 655), (431, 535), (156, 120)),
    ],
)
def test_real_partition_counts(
    scheme, num_clients, runs, classification, regression, no_walking
) -> None:
    partitions, summary = _real(scheme, num_clients)
    cls_artifact, reg_artifact = _artifacts()
    _assert_common(partitions, summary, cls_artifact, reg_artifact)
    assert tuple(len(p.run_uids) for p in partitions) == runs
    assert tuple(len(p.classification_indices) for p in partitions) == classification
    assert tuple(len(p.regression_indices) for p in partitions) == regression
    assert tuple(c["num_no_walking_windows"] for c in summary["clients"]) == no_walking
    assert sum(runs) == 112
    assert sum(classification) == 1242
    assert sum(regression) == 966


@pytest.mark.requires_data
@pytest.mark.parametrize("num_clients", [2, 3])
def test_real_iid_every_client_holds_every_subject_and_stratum(num_clients: int) -> None:
    _, summary = _real("iid", num_clients)
    for client in summary["clients"]:
        assert client["subjects"] == ["001", "002", "003", "004", "005", "007", "008"]
        assert client["source_subject_groups"] == [
            "test_2:001", "test_2:002", "test_2:003",
            "testing_20251124:003", "testing_20251124:004", "testing_20251124:005",
            "testing_20251124:007", "testing_20251124:008",
        ]
        assert set(client["direction_run_counts"]) == {
            "test_2:single_direction_unknown",
            "testing_20251124:N_to_S",
            "testing_20251124:S_to_N",
        }


@pytest.mark.requires_data
def test_real_natural_split_subjects() -> None:
    _, summary = _real("natural", 2)
    clients = {c["client_id"]: c for c in summary["clients"]}
    assert clients["client_0"]["subjects"] == ["001", "002", "003"]
    assert clients["client_1"]["subjects"] == ["003", "004", "005", "007", "008"]


@pytest.mark.requires_data
def test_real_partition_is_deterministic() -> None:
    classification, regression = _artifacts()
    original, original_summary = _real("iid", 3)
    rebuilt, rebuilt_summary = build_combined_iid_partitions(
        classification, regression, num_clients=3, seed=4601
    )
    assert rebuilt == original
    assert rebuilt_summary["run_to_client"] == original_summary["run_to_client"]


@pytest.mark.requires_data
@pytest.mark.parametrize(
    ("config_path", "scheme"), [(IID_CONFIG, "iid"), (NATURAL_CONFIG, "natural")]
)
def test_federated_setup_audit_runs_for_both_schemes(config_path: Path, scheme: str) -> None:
    config = load_combined_iid_flower_config(config_path)
    _, _, partitions, setup = audit_combined_iid_flower_setup(config)
    assert setup["partition_scheme"] == scheme
    assert setup["partition"]["scheme"] == scheme
    assert len(partitions) == config["num_clients"]
    assert setup["residual_baseline"]["num_train_runs"] == 112

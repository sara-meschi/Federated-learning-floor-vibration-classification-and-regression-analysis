"""Leakage-safe client partitioning for the combined federated experiments.

The indivisible partitioning unit in this module is a *complete recording run*.
Classification and regression windows from the same ``run_uid`` are therefore
always owned by the same client.  This is important because adjacent five-second
windows overlap by four seconds in the combined artifacts.

Two schemes, both over the K clients of a real ``num_clients`` parameter (K <= 3,
``docs/review_response_plan.md`` §1.2):

``iid``
    Within every ``(source, subject, direction)`` stratum, runs are ordered by APDM
    gait speed and dealt round-robin in blocks of K: each complete block gives exactly
    one run to every client, through a seeded permutation, and each remainder run goes
    to a distinct client with the fewest runs so far.  Run counts end within one of
    balanced, and every client sees every stratum across the whole speed range.

    The third stratum key, ``direction``, is there for **signal coverage, not speed
    balance**.  R11 showed walking speed is direction-invariant (max |delta| 0.028 m/s
    against a within-stratum std of 0.032 m/s), so stratifying on direction buys no
    speed balance at all.  What it buys is that direction reverses the order in which
    the corridor sensors are excited: a client that only ever saw N->S passes would
    be trained on one temporal ordering of the footstep wavefront across the nine
    channels and may fail on S->N.  Keying the strata on direction guarantees every
    client sees both orderings wherever a source recorded both (``20251124_Testing``
    alternates by run parity; ``Test_2`` is unidirectional, S->N, and contributes a
    single direction value per subject).  See ``docs/session0_findings.md``.

``natural``
    One client per collection campaign: ``test_2`` -> ``client_0``,
    ``testing_20251124`` -> ``client_1`` (K must equal the number of sources, 2).
    Both campaigns are the same instrumented corridor 19.6 months apart, so this is a
    cross-session / cross-silo split, not a cross-building one.  Its client sizes are
    uneven and its subject coverage is deliberately incomplete (3 subjects vs 5):
    that heterogeneity is the point, so the IID balance and coverage properties are
    *not* asserted for it.

Properties asserted for every scheme: client run and window sets are disjoint and
together cover the training set exactly once; no client owns a held-out test run;
both tasks give every run the same owner; every window a client owns belongs to one
of its runs; excluded subjects are absent.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .guardrails import assert_split_integrity


PARTITION_SCHEMES = ("iid", "natural")
#: §1.2: with 112 training runs, K > 3 leaves ~4 optimizer steps per local epoch at
#: batch size 4, which structurally reproduces the R1 collapse.
MAX_CLIENTS = 3
#: Subject 006 is excluded from every experiment (data collection problems). Asserted
#: absent here as a property of any partition, not as an expected outcome.
EXCLUDED_SUBJECT_IDS = ("006",)


def client_ids_for(num_clients: int) -> tuple[str, ...]:
    return tuple(f"client_{index}" for index in range(num_clients))


@dataclass(frozen=True)
class ClientPartition:
    """Global artifact indices owned by one simulated Flower client."""

    client_id: str
    run_uids: tuple[str, ...]
    classification_indices: tuple[int, ...]
    regression_indices: tuple[int, ...]


@dataclass(frozen=True)
class _RunProfile:
    run_uid: str
    source_id: str
    subject_id: str
    run_index: int
    direction: str
    speed_mps: float
    classification_indices: tuple[int, ...]
    regression_indices: tuple[int, ...]
    walking_classification_indices: tuple[int, ...]
    no_walking_classification_indices: tuple[int, ...]

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.source_id, self.subject_id, self.direction)

    @property
    def walking_windows(self) -> int:
        return len(self.regression_indices)

    @property
    def no_walking_windows(self) -> int:
        return len(self.no_walking_classification_indices)


def _indices(value: Any, name: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    result = tuple(int(index) for index in value)
    if len(result) != len(set(result)):
        raise AssertionError(f"{name} contains duplicate indices.")
    return result


def _canonical_run_uid(item: Mapping[str, Any]) -> str:
    source_id = str(item["source_id"])
    subject_id = f"{int(item['subject_id']):03d}"
    run_index = int(item["run_index"])
    expected = f"{source_id}:{subject_id}:{run_index:03d}"
    actual = str(item["run_uid"])
    if actual != expected:
        raise AssertionError(
            f"Non-canonical run_uid {actual!r}; expected {expected!r}. "
            "The source component is mandatory for cross-dataset uniqueness."
        )
    return actual


def _metadata_by_run(
    artifact: Mapping[str, Any], indices: Sequence[int], artifact_name: str
) -> dict[str, list[tuple[int, Mapping[str, Any]]]]:
    metadata = artifact.get("metadata")
    if not isinstance(metadata, Sequence):
        raise TypeError(f"{artifact_name} artifact metadata must be a sequence.")
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index in indices:
        if index < 0 or index >= len(metadata):
            raise IndexError(f"{artifact_name} index {index} is outside its metadata.")
        item = metadata[index]
        if not isinstance(item, Mapping):
            raise TypeError(f"{artifact_name} metadata[{index}] is not a mapping.")
        run_uid = _canonical_run_uid(item)
        grouped[run_uid].append((index, item))
    return dict(grouped)


def _split_run_uids(
    artifact: Mapping[str, Any], artifact_name: str
) -> tuple[set[str], set[str], tuple[int, ...], tuple[int, ...]]:
    # Load-time half of CLAUDE.md invariant 2; the build-time half runs in the builders.
    assert_split_integrity(artifact, f"{artifact_name} artifact (partition load)")
    train_indices = _indices(artifact["train_indices"], f"{artifact_name}.train_indices")
    test_indices = _indices(artifact["test_indices"], f"{artifact_name}.test_indices")
    train_uids = set(_metadata_by_run(artifact, train_indices, artifact_name))
    test_uids = set(_metadata_by_run(artifact, test_indices, artifact_name))
    return train_uids, test_uids, train_indices, test_indices


def _build_run_profiles(
    classification_artifact: Mapping[str, Any],
    regression_artifact: Mapping[str, Any],
) -> tuple[
    dict[str, _RunProfile], tuple[int, ...], tuple[int, ...], set[str]
]:
    (
        classification_train_uids,
        classification_test_uids,
        classification_train_indices,
        _,
    ) = _split_run_uids(classification_artifact, "classification")
    (
        regression_train_uids,
        regression_test_uids,
        regression_train_indices,
        _,
    ) = _split_run_uids(regression_artifact, "regression")
    if classification_train_uids != regression_train_uids:
        raise AssertionError(
            "Classification and regression training run sets differ: "
            f"classification-only={sorted(classification_train_uids - regression_train_uids)}, "
            f"regression-only={sorted(regression_train_uids - classification_train_uids)}."
        )
    if classification_test_uids != regression_test_uids:
        raise AssertionError(
            "Classification and regression test run sets differ: "
            f"classification-only={sorted(classification_test_uids - regression_test_uids)}, "
            f"regression-only={sorted(regression_test_uids - classification_test_uids)}."
        )

    classification_by_run = _metadata_by_run(
        classification_artifact, classification_train_indices, "classification"
    )
    regression_by_run = _metadata_by_run(
        regression_artifact, regression_train_indices, "regression"
    )
    profiles: dict[str, _RunProfile] = {}
    for run_uid in sorted(classification_train_uids):
        classification_rows = classification_by_run[run_uid]
        regression_rows = regression_by_run[run_uid]
        classification_first = classification_rows[0][1]
        regression_first = regression_rows[0][1]
        identity_fields = ("source_id", "subject_id", "run_index", "direction")
        for field in identity_fields:
            if str(classification_first[field]) != str(regression_first[field]):
                raise AssertionError(
                    f"Artifacts disagree on {field} for {run_uid}: "
                    f"{classification_first[field]!r} != {regression_first[field]!r}."
                )

        speed_values = {float(item["speed_mps"]) for _, item in regression_rows}
        if len(speed_values) != 1:
            raise AssertionError(f"Run {run_uid} has multiple regression targets: {speed_values}")
        speed_mps = next(iter(speed_values))
        if not np.isfinite(speed_mps) or speed_mps <= 0:
            raise AssertionError(f"Run {run_uid} has invalid gait speed {speed_mps}.")

        walking_rows = [
            (index, item)
            for index, item in classification_rows
            if str(item.get("activity", "")) != "no_walking"
        ]
        no_walking_rows = [
            (index, item)
            for index, item in classification_rows
            if str(item.get("activity", "")) == "no_walking"
        ]
        if not walking_rows or not no_walking_rows:
            raise AssertionError(
                f"Classification run {run_uid} must contain walking and no-walking windows."
            )
        classification_walking_keys = {
            (str(item["run_uid"]), round(float(item["start_time"]), 9))
            for _, item in walking_rows
        }
        regression_walking_keys = {
            (str(item["run_uid"]), round(float(item["start_time"]), 9))
            for _, item in regression_rows
        }
        if classification_walking_keys != regression_walking_keys:
            raise AssertionError(
                f"Classification/regression walking windows differ for {run_uid}."
            )
        if len(walking_rows) != len(regression_rows):
            raise AssertionError(
                f"Classification/regression walking counts differ for {run_uid}."
            )

        profiles[run_uid] = _RunProfile(
            run_uid=run_uid,
            source_id=str(regression_first["source_id"]),
            subject_id=f"{int(regression_first['subject_id']):03d}",
            run_index=int(regression_first["run_index"]),
            direction=str(regression_first["direction"]),
            speed_mps=speed_mps,
            classification_indices=tuple(index for index, _ in classification_rows),
            regression_indices=tuple(index for index, _ in regression_rows),
            walking_classification_indices=tuple(index for index, _ in walking_rows),
            no_walking_classification_indices=tuple(index for index, _ in no_walking_rows),
        )
    return (
        profiles,
        classification_train_indices,
        regression_train_indices,
        classification_test_uids,
    )


def _seed_rank(seed: int, *parts: Any) -> int:
    payload = ":".join(str(part) for part in (seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _speed_ordered_strata(
    profiles: Mapping[str, _RunProfile],
) -> list[tuple[tuple[str, str, str], list[_RunProfile]]]:
    by_stratum: dict[tuple[str, str, str], list[_RunProfile]] = defaultdict(list)
    for profile in profiles.values():
        by_stratum[profile.stratum].append(profile)
    return [
        (stratum, sorted(members, key=lambda item: (item.speed_mps, item.run_uid)))
        for stratum, members in sorted(by_stratum.items())
    ]


def _block_audit_row(
    block_number: int | None,
    stratum: tuple[str, str, str],
    block: Sequence[_RunProfile],
    owners: Sequence[int],
) -> dict[str, Any]:
    return {
        "block_number": block_number,
        "source_id": stratum[0],
        "subject_id": stratum[1],
        "direction": stratum[2],
        "size": len(block),
        "minimum_speed_mps": min(record.speed_mps for record in block),
        "maximum_speed_mps": max(record.speed_mps for record in block),
        "run_to_client": {
            record.run_uid: f"client_{owner}" for record, owner in zip(block, owners)
        },
    }


def _assign_iid(
    profiles: Mapping[str, _RunProfile], num_clients: int, seed: int
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Round-robin over speed-ordered blocks of K within each stratum."""

    assignment: dict[str, int] = {}
    run_totals = [0] * num_clients
    block_audit: list[dict[str, Any]] = []
    complete_block_number = 0
    for stratum, ordered in _speed_ordered_strata(profiles):
        if len(ordered) < num_clients:
            raise AssertionError(
                f"Stratum {stratum} has {len(ordered)} training runs, fewer than "
                f"K={num_clients}; an IID partition cannot give every client a run of it."
            )
        for start in range(0, len(ordered), num_clients):
            block = ordered[start : start + num_clients]
            if len(block) == num_clients:
                # A complete block: one run to every client, permuted by seed so no
                # client systematically gets the slowest member of each block.
                owners = sorted(
                    range(num_clients),
                    key=lambda client: _seed_rank(seed, *stratum, start, client),
                )
                block_number: int | None = complete_block_number
                complete_block_number += 1
            else:
                # A remainder: distinct clients, fewest runs first, ties by seed. Taking
                # the r least-loaded clients keeps every client's total within one.
                owners = sorted(
                    range(num_clients),
                    key=lambda client: (
                        run_totals[client],
                        _seed_rank(seed, *stratum, "remainder", client),
                    ),
                )[: len(block)]
                block_number = None
            for record, owner in zip(block, owners):
                assignment[record.run_uid] = owner
                run_totals[owner] += 1
            block_audit.append(_block_audit_row(block_number, stratum, block, owners))
    return assignment, block_audit


def _assign_natural(
    profiles: Mapping[str, _RunProfile], num_clients: int
) -> tuple[dict[str, int], list[str]]:
    """One client per collection campaign, in sorted source order."""

    sources = sorted({profile.source_id for profile in profiles.values()})
    if len(sources) != num_clients:
        raise ValueError(
            f"The natural partition needs one client per source: {len(sources)} sources "
            f"({sources}) but num_clients={num_clients}."
        )
    assignment = {
        run_uid: sources.index(profile.source_id) for run_uid, profile in profiles.items()
    }
    return assignment, sources


def _speed_distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum_mps": float(array.min()),
        "maximum_mps": float(array.max()),
        "mean_mps": float(array.mean()),
        "std_mps": float(array.std()),
    }


def _assert_common_properties(
    partitions: Sequence[ClientPartition],
    profiles: Mapping[str, _RunProfile],
    classification_artifact: Mapping[str, Any],
    regression_artifact: Mapping[str, Any],
    classification_train_indices: Sequence[int],
    regression_train_indices: Sequence[int],
    test_uids: set[str],
) -> dict[str, Any]:
    """Properties every scheme must satisfy (CLAUDE.md invariants 1-3)."""

    expected_train_uids = set(profiles)
    run_sets = [set(partition.run_uids) for partition in partitions]
    classification_sets = [set(partition.classification_indices) for partition in partitions]
    regression_sets = [set(partition.regression_indices) for partition in partitions]

    pairwise_run_overlap = 0
    pairwise_classification_overlap = 0
    pairwise_regression_overlap = 0
    for left, right in itertools.combinations(range(len(partitions)), 2):
        pairwise_run_overlap += len(run_sets[left] & run_sets[right])
        pairwise_classification_overlap += len(
            classification_sets[left] & classification_sets[right]
        )
        pairwise_regression_overlap += len(regression_sets[left] & regression_sets[right])
    if pairwise_run_overlap:
        raise AssertionError("Client run sets overlap.")
    if pairwise_classification_overlap:
        raise AssertionError("Client classification-window sets overlap.")
    if pairwise_regression_overlap:
        raise AssertionError("Client regression-window sets overlap.")
    if set().union(*run_sets) != expected_train_uids:
        raise AssertionError("Client run union does not equal the complete training run set.")
    if set().union(*classification_sets) != set(classification_train_indices):
        raise AssertionError(
            "Client classification-window union does not equal all training windows."
        )
    if set().union(*regression_sets) != set(regression_train_indices):
        raise AssertionError(
            "Client regression-window union does not equal all training windows."
        )
    if expected_train_uids & test_uids:
        raise AssertionError("Training and held-out test run sets overlap.")
    if any(run_set & test_uids for run_set in run_sets):
        raise AssertionError("A client owns a held-out test run.")

    # Run atomicity, measured from the windows rather than inferred from construction:
    # every window a client owns, in either task, belongs to one of that client's runs.
    for partition, run_set in zip(partitions, run_sets):
        for task, artifact, indices in (
            ("classification", classification_artifact, partition.classification_indices),
            ("regression", regression_artifact, partition.regression_indices),
        ):
            window_runs = {str(artifact["metadata"][index]["run_uid"]) for index in indices}
            if window_runs != run_set:
                raise AssertionError(
                    f"{partition.client_id} {task} windows come from runs "
                    f"{sorted(window_runs ^ run_set)[:5]} that it does not own exactly."
                )

    present_excluded = sorted(
        {profile.subject_id for profile in profiles.values()} & set(EXCLUDED_SUBJECT_IDS)
    )
    if present_excluded:
        raise AssertionError(f"Excluded subjects present in training runs: {present_excluded}.")

    return {
        "classification_and_regression_train_run_sets_identical": True,
        "classification_and_regression_test_run_sets_identical": True,
        "canonical_source_aware_run_uids": True,
        "pairwise_client_run_overlap": pairwise_run_overlap,
        "pairwise_client_classification_window_overlap": pairwise_classification_overlap,
        "pairwise_client_regression_window_overlap": pairwise_regression_overlap,
        "client_run_union_equals_all_training_runs": True,
        "classification_window_coverage_exactly_once": True,
        "regression_window_coverage_exactly_once": True,
        "train_test_run_overlap": len(expected_train_uids & test_uids),
        "test_runs_owned_by_clients": sum(len(run_set & test_uids) for run_set in run_sets),
        "same_run_owner_for_both_tasks": True,
        "client_windows_belong_to_client_runs": True,
        "excluded_subjects_absent": list(EXCLUDED_SUBJECT_IDS),
    }


def _assert_iid_properties(
    partitions: Sequence[ClientPartition],
    profiles: Mapping[str, _RunProfile],
    block_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """IID-only properties (CLAUDE.md invariant 4, review plan §3 item 3)."""

    run_counts = [len(partition.run_uids) for partition in partitions]
    if max(run_counts) - min(run_counts) > 1:
        raise AssertionError(f"IID client run counts are not within one: {run_counts}.")

    all_subjects = {profile.subject_id for profile in profiles.values()}
    all_strata = {profile.stratum for profile in profiles.values()}
    subject_totals = Counter(profile.subject_id for profile in profiles.values())
    for partition in partitions:
        client_profiles = [profiles[run_uid] for run_uid in partition.run_uids]
        subject_counts = Counter(profile.subject_id for profile in client_profiles)
        missing_subjects = all_subjects - set(subject_counts)
        if missing_subjects:
            raise AssertionError(
                f"{partition.client_id} holds no run of subjects {sorted(missing_subjects)}."
            )
        missing_strata = all_strata - {profile.stratum for profile in client_profiles}
        if missing_strata:
            raise AssertionError(
                f"{partition.client_id} holds no run of strata {sorted(missing_strata)}."
            )
        hoarded = [
            subject for subject, count in subject_counts.items()
            if count == subject_totals[subject]
        ]
        if hoarded:
            raise AssertionError(
                f"{partition.client_id} holds every training run of subjects {hoarded}."
            )

    for block in block_audit:
        if block["block_number"] is not None and len(set(block["run_to_client"].values())) != len(
            partitions
        ):
            raise AssertionError(f"Complete speed block {block['block_number']} is not 1/client.")

    return {
        "client_run_count_spread": max(run_counts) - min(run_counts),
        "every_client_holds_every_subject": True,
        "every_client_holds_every_source_subject_direction_stratum": True,
        "no_client_holds_all_runs_of_a_subject": True,
        "complete_speed_blocks_split_one_per_client": True,
    }


def _assert_natural_properties(
    partitions: Sequence[ClientPartition],
    profiles: Mapping[str, _RunProfile],
    sources: Sequence[str],
) -> dict[str, Any]:
    """Natural-only properties. Balance and subject coverage are deliberately absent."""

    for partition, source_id in zip(partitions, sources):
        client_sources = {profiles[run_uid].source_id for run_uid in partition.run_uids}
        if client_sources != {source_id}:
            raise AssertionError(
                f"{partition.client_id} must hold exactly source {source_id!r}, "
                f"holds {sorted(client_sources)}."
            )
    return {
        "one_source_per_client": True,
        "each_source_in_exactly_one_client": True,
    }


def _client_summary(
    partition: ClientPartition, profiles: Mapping[str, _RunProfile]
) -> dict[str, Any]:
    client_profiles = [profiles[run_uid] for run_uid in partition.run_uids]
    walking_class_counts: Counter[str] = Counter()
    for profile in client_profiles:
        walking_class_counts[f"subject_{profile.subject_id}_walking"] += profile.walking_windows
    no_walking_count = sum(profile.no_walking_windows for profile in client_profiles)
    return {
        "client_id": partition.client_id,
        "num_runs": len(client_profiles),
        "num_classification_windows": len(partition.classification_indices),
        "num_regression_windows": len(partition.regression_indices),
        "num_walking_windows": sum(profile.walking_windows for profile in client_profiles),
        "num_no_walking_windows": no_walking_count,
        "subjects": sorted({profile.subject_id for profile in client_profiles}),
        "source_subject_groups": sorted(
            {f"{profile.source_id}:{profile.subject_id}" for profile in client_profiles}
        ),
        "subject_run_counts": dict(
            sorted(Counter(profile.subject_id for profile in client_profiles).items())
        ),
        "source_run_counts": dict(
            sorted(Counter(profile.source_id for profile in client_profiles).items())
        ),
        "source_subject_run_counts": dict(
            sorted(
                Counter(
                    f"{profile.source_id}:{profile.subject_id}" for profile in client_profiles
                ).items()
            )
        ),
        "direction_run_counts": dict(
            sorted(
                Counter(
                    f"{profile.source_id}:{profile.direction}" for profile in client_profiles
                ).items()
            )
        ),
        "classification_class_window_counts": {
            "no_walking": no_walking_count,
            **dict(sorted(walking_class_counts.items())),
        },
        "run_speed_distribution": _speed_distribution(
            [profile.speed_mps for profile in client_profiles]
        ),
        "run_uids": list(partition.run_uids),
    }


def build_combined_partitions(
    classification_artifact: Mapping[str, Any],
    regression_artifact: Mapping[str, Any],
    scheme: str = "iid",
    num_clients: int = 3,
    seed: int = 4601,
) -> tuple[list[ClientPartition], dict[str, Any]]:
    """Build a run-atomic client partition shared by both tasks, plus its audit.

    The returned indices refer to the original, global artifact tensors.  Local
    training code may shuffle those indices after partitioning, but must not
    reassign individual windows to another client.
    """

    if scheme not in PARTITION_SCHEMES:
        raise ValueError(f"Unknown partition scheme {scheme!r}; expected {PARTITION_SCHEMES}.")
    if not 2 <= num_clients <= MAX_CLIENTS:
        raise ValueError(
            f"num_clients must be 2 or 3, got {num_clients}. K > {MAX_CLIENTS} leaves too few "
            "local optimizer steps per client (review plan §1.2)."
        )

    (
        profiles,
        classification_train_indices,
        regression_train_indices,
        test_uids,
    ) = _build_run_profiles(classification_artifact, regression_artifact)

    sources: list[str] = []
    block_audit: list[dict[str, Any]] = []
    if scheme == "iid":
        assignment, block_audit = _assign_iid(profiles, num_clients, seed)
    else:
        assignment, sources = _assign_natural(profiles, num_clients)

    client_ids = client_ids_for(num_clients)
    partitions: list[ClientPartition] = []
    for client_index, client_id in enumerate(client_ids):
        run_uids = tuple(
            sorted(run_uid for run_uid, owner in assignment.items() if owner == client_index)
        )
        partitions.append(
            ClientPartition(
                client_id=client_id,
                run_uids=run_uids,
                classification_indices=tuple(
                    sorted(
                        index
                        for run_uid in run_uids
                        for index in profiles[run_uid].classification_indices
                    )
                ),
                regression_indices=tuple(
                    sorted(
                        index
                        for run_uid in run_uids
                        for index in profiles[run_uid].regression_indices
                    )
                ),
            )
        )

    audits = _assert_common_properties(
        partitions=partitions,
        profiles=profiles,
        classification_artifact=classification_artifact,
        regression_artifact=regression_artifact,
        classification_train_indices=classification_train_indices,
        regression_train_indices=regression_train_indices,
        test_uids=test_uids,
    )
    if scheme == "iid":
        audits.update(_assert_iid_properties(partitions, profiles, block_audit))
    else:
        audits.update(_assert_natural_properties(partitions, profiles, sources))

    summary: dict[str, Any] = {
        "schema_version": 2,
        "experiment": f"combined_{num_clients}client_run_grouped_{scheme}",
        "scheme": scheme,
        "seed": seed,
        "num_clients": num_clients,
        "client_ids": list(client_ids),
        "partition_unit": "complete_run_before_window_shuffling",
        "canonical_run_uid": "{source_id}:{subject_id_3_digits}:{run_index_3_digits}",
        "num_training_runs": len(profiles),
        "num_test_runs": len(test_uids),
        "num_classification_training_windows": len(classification_train_indices),
        "num_regression_training_windows": len(regression_train_indices),
        "clients": [_client_summary(partition, profiles) for partition in partitions],
        "run_to_client": {
            run_uid: client_ids[owner] for run_uid, owner in sorted(assignment.items())
        },
        "audits": audits,
    }
    if scheme == "iid":
        summary.update(
            {
                "stratification_fields": ["source_id", "subject_id", "direction"],
                "direction_stratification_reason": (
                    "signal coverage, not speed balance: direction reverses the order in "
                    "which corridor sensors are excited; speed is direction-invariant (R11)"
                ),
                "speed_block_size": num_clients,
                "speed_ordering": "ascending_APDM_mean_left_right_gait_speed_then_run_uid",
                "speed_blocks": block_audit,
            }
        )
    else:
        summary["client_sources"] = {
            client_ids[index]: source_id for index, source_id in enumerate(sources)
        }
    return partitions, summary


def build_combined_iid_partitions(
    classification_artifact: Mapping[str, Any],
    regression_artifact: Mapping[str, Any],
    num_clients: int = 3,
    seed: int = 4601,
) -> tuple[list[ClientPartition], dict[str, Any]]:
    """The IID scheme of :func:`build_combined_partitions`, kept for existing callers."""

    return build_combined_partitions(
        classification_artifact,
        regression_artifact,
        scheme="iid",
        num_clients=num_clients,
        seed=seed,
    )


def save_combined_iid_partitions(
    partitions: Sequence[ClientPartition],
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save indices, the run-owner manifest, and the full audit summary."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    partitions_path = destination / "combined_iid_client_partitions.json"
    manifest_path = destination / "combined_iid_run_to_client.csv"
    summary_path = destination / "combined_iid_partition_summary.json"

    partitions_path.write_text(
        json.dumps([asdict(partition) for partition in partitions], indent=2)
    )
    run_to_client = summary.get("run_to_client")
    if not isinstance(run_to_client, Mapping):
        raise TypeError("Partition summary is missing run_to_client.")
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_uid", "client_id"])
        writer.writeheader()
        for run_uid, client_id in sorted(run_to_client.items()):
            writer.writerow({"run_uid": run_uid, "client_id": client_id})
    summary_path.write_text(json.dumps(dict(summary), indent=2))
    return {
        "partitions": partitions_path,
        "run_manifest": manifest_path,
        "summary": summary_path,
    }

"""Leakage-safe, shared IID client partitioning for the combined experiments.

The indivisible partitioning unit in this module is a *complete recording run*.
Classification and regression windows from the same ``run_uid`` are therefore
always owned by the same client.  This is important because adjacent five-second
windows overlap by four seconds in the combined artifacts.

The experiment is intentionally fixed to the three-client, seed-4601 design used
for the combined nine-channel data.  Within every
``(source, subject, direction)`` stratum, runs are ordered by APDM gait speed and
placed into neighboring blocks of three.  Each complete block contributes one
run to every client.  Deterministic, seed-derived tie breaking assigns the small
remainders and the members of each block while meeting the audited run and
window totals.
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


CLIENT_IDS = ("client_0", "client_1", "client_2")
EXPECTED_SUBJECTS = ("001", "002", "003", "004", "005", "007", "008")
EXPECTED_SOURCE_SUBJECT_GROUPS = (
    "test_2:001",
    "test_2:002",
    "test_2:003",
    "testing_20251124:003",
    "testing_20251124:004",
    "testing_20251124:005",
    "testing_20251124:007",
    "testing_20251124:008",
)
EXPECTED_RUN_COUNTS = (37, 38, 37)
EXPECTED_REGRESSION_WINDOW_COUNTS = (321, 324, 321)
EXPECTED_NO_WALKING_WINDOW_COUNTS = (91, 94, 91)
EXPECTED_CLASSIFICATION_WINDOW_COUNTS = (412, 418, 412)
EXPECTED_SUBJECT_RUN_COUNTS: dict[str, tuple[int, int, int]] = {
    "001": (6, 6, 5),
    "002": (6, 6, 6),
    "003": (9, 10, 10),
    "004": (4, 4, 4),
    "005": (5, 4, 4),
    "007": (3, 4, 4),
    "008": (4, 4, 4),
}
EXPECTED_WALKING_CLASS_COUNTS: dict[str, tuple[int, int, int]] = {
    "001": (54, 55, 47),
    "002": (50, 49, 52),
    "003": (76, 79, 83),
    "004": (34, 34, 34),
    "005": (41, 35, 33),
    "007": (28, 35, 35),
    "008": (38, 37, 37),
}


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
    train_indices = _indices(artifact["train_indices"], f"{artifact_name}.train_indices")
    test_indices = _indices(artifact["test_indices"], f"{artifact_name}.test_indices")
    if set(train_indices) & set(test_indices):
        raise AssertionError(f"{artifact_name} train/test indices overlap.")
    train_metadata = _metadata_by_run(artifact, train_indices, artifact_name)
    test_metadata = _metadata_by_run(artifact, test_indices, artifact_name)
    train_uids = set(train_metadata)
    test_uids = set(test_metadata)
    overlap = train_uids & test_uids
    if overlap:
        raise AssertionError(
            f"{artifact_name} has run leakage between train and test: {sorted(overlap)}"
        )
    for index in train_indices:
        if str(artifact["metadata"][index].get("split")) != "train":
            raise AssertionError(f"{artifact_name} train index {index} is not marked train.")
    for index in test_indices:
        if str(artifact["metadata"][index].get("split")) != "test":
            raise AssertionError(f"{artifact_name} test index {index} is not marked test.")
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


def _form_speed_blocks(
    profiles: Mapping[str, _RunProfile], num_clients: int
) -> tuple[
    list[tuple[tuple[str, str, str], tuple[_RunProfile, ...]]],
    list[tuple[tuple[str, str, str], tuple[_RunProfile, ...]]],
]:
    by_stratum: dict[tuple[str, str, str], list[_RunProfile]] = defaultdict(list)
    for profile in profiles.values():
        by_stratum[(profile.source_id, profile.subject_id, profile.direction)].append(
            profile
        )

    blocks: list[tuple[tuple[str, str, str], tuple[_RunProfile, ...]]] = []
    remainders: list[tuple[tuple[str, str, str], tuple[_RunProfile, ...]]] = []
    for stratum, stratum_profiles in sorted(by_stratum.items()):
        ordered = sorted(
            stratum_profiles, key=lambda item: (item.speed_mps, item.run_uid)
        )
        complete_count = len(ordered) // num_clients * num_clients
        for start in range(0, complete_count, num_clients):
            blocks.append((stratum, tuple(ordered[start : start + num_clients])))
        if complete_count < len(ordered):
            remainders.append((stratum, tuple(ordered[complete_count:])))
    return blocks, remainders


def _base_counts_from_blocks(
    blocks: Sequence[tuple[tuple[str, str, str], tuple[_RunProfile, ...]]],
) -> tuple[
    dict[str, list[int]], dict[str, list[int]], dict[tuple[str, str], list[int]]
]:
    subject_counts = {subject_id: [0, 0, 0] for subject_id in EXPECTED_SUBJECTS}
    source_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    direction_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for (source_id, subject_id, direction), _ in blocks:
        for client_index in range(3):
            subject_counts[subject_id][client_index] += 1
            source_counts[source_id][client_index] += 1
            direction_counts[(source_id, direction)][client_index] += 1
    return subject_counts, source_counts, direction_counts


def _remainder_assignments(
    blocks: Sequence[tuple[tuple[str, str, str], tuple[_RunProfile, ...]]],
    remainders: Sequence[tuple[tuple[str, str, str], tuple[_RunProfile, ...]]],
    seed: int,
) -> list[dict[str, int]]:
    subject_base, source_base, direction_base = _base_counts_from_blocks(blocks)
    options: list[
        list[tuple[tuple[str, str, str], tuple[_RunProfile, ...], tuple[int, ...]]]
    ] = []
    for stratum, records in remainders:
        assignments = list(itertools.permutations(range(3), len(records)))
        assignments.sort(
            key=lambda assignment: _seed_rank(
                seed,
                *stratum,
                *(record.run_uid for record in records),
                *assignment,
            )
        )
        options.append([(stratum, records, assignment) for assignment in assignments])

    valid: list[tuple[int, dict[str, int]]] = []
    for selection in itertools.product(*options):
        subject_counts = {key: value.copy() for key, value in subject_base.items()}
        source_counts = {key: value.copy() for key, value in source_base.items()}
        direction_counts = {key: value.copy() for key, value in direction_base.items()}
        assignment_by_uid: dict[str, int] = {}
        for _, records, assignments in selection:
            for record, client_index in zip(records, assignments):
                assignment_by_uid[record.run_uid] = client_index
                subject_counts[record.subject_id][client_index] += 1
                source_counts.setdefault(record.source_id, [0, 0, 0])[client_index] += 1
                direction_counts.setdefault(
                    (record.source_id, record.direction), [0, 0, 0]
                )[client_index] += 1

        run_counts = [len(blocks) + sum(value == index for value in assignment_by_uid.values())
                      for index in range(3)]
        if tuple(run_counts) != EXPECTED_RUN_COUNTS:
            continue
        if any(tuple(subject_counts[subject]) != EXPECTED_SUBJECT_RUN_COUNTS[subject]
               for subject in EXPECTED_SUBJECTS):
            continue
        if tuple(source_counts.get("test_2", [])) != (17, 18, 17):
            continue
        for direction in ("N_to_S", "S_to_N"):
            if tuple(direction_counts.get(("testing_20251124", direction), [])) != (
                10,
                10,
                10,
            ):
                break
        else:
            rank = sum(
                _seed_rank(seed, run_uid, client_index)
                for run_uid, client_index in assignment_by_uid.items()
            )
            valid.append((rank, assignment_by_uid))
    valid.sort(key=lambda item: item[0])
    return [assignment for _, assignment in valid]


def _solve_subject_blocks(
    subject_blocks: Sequence[tuple[tuple[str, str, str], tuple[_RunProfile, ...]]],
    initial_windows: Sequence[int],
    target_windows: Sequence[int],
    seed: int,
) -> list[tuple[int, ...]] | None:
    """Find deterministic block permutations that hit exact per-subject totals."""

    permutations = list(itertools.permutations(range(3)))
    # Only the first two counts need to be keys; the third follows from the total.
    states: dict[tuple[int, int], tuple[int, list[tuple[int, ...]]]] = {
        (int(initial_windows[0]), int(initial_windows[1])): (0, [])
    }
    cumulative_windows = int(sum(initial_windows))
    for block_index, (stratum, block) in enumerate(subject_blocks):
        cumulative_windows += sum(record.walking_windows for record in block)
        choices: list[tuple[int, tuple[int, ...], tuple[int, int, int]]] = []
        for permutation in permutations:
            additions = [0, 0, 0]
            for record, client_index in zip(block, permutation):
                additions[client_index] += record.walking_windows
            rank = _seed_rank(seed, *stratum, block_index, *permutation)
            choices.append((rank, permutation, tuple(additions)))
        choices.sort(key=lambda item: item[0])

        next_states: dict[tuple[int, int], tuple[int, list[tuple[int, ...]]]] = {}
        for (count_0, count_1), (path_rank, path) in states.items():
            for choice_rank, permutation, additions in choices:
                next_0 = count_0 + additions[0]
                next_1 = count_1 + additions[1]
                next_2 = cumulative_windows - next_0 - next_1
                if (
                    next_0 > target_windows[0]
                    or next_1 > target_windows[1]
                    or next_2 > target_windows[2]
                ):
                    continue
                key = (next_0, next_1)
                candidate_rank = path_rank + choice_rank
                if key not in next_states or candidate_rank < next_states[key][0]:
                    next_states[key] = (candidate_rank, [*path, permutation])
        states = next_states

    solution = states.get((int(target_windows[0]), int(target_windows[1])))
    return None if solution is None else solution[1]


def _assign_profiles(
    profiles: Mapping[str, _RunProfile], num_clients: int, seed: int
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    blocks, remainders = _form_speed_blocks(profiles, num_clients)
    remainder_candidates = _remainder_assignments(blocks, remainders, seed)
    if not remainder_candidates:
        raise RuntimeError("No remainder allocation satisfies the fixed IID audit targets.")

    final_assignment: dict[str, int] | None = None
    for remainder_assignment in remainder_candidates:
        assignment = remainder_assignment.copy()
        complete = True
        for subject_id in EXPECTED_SUBJECTS:
            subject_blocks = [block for block in blocks if block[0][1] == subject_id]
            initial_windows = [0, 0, 0]
            for run_uid, client_index in remainder_assignment.items():
                profile = profiles[run_uid]
                if profile.subject_id == subject_id:
                    initial_windows[client_index] += profile.walking_windows
            permutations = _solve_subject_blocks(
                subject_blocks=subject_blocks,
                initial_windows=initial_windows,
                target_windows=EXPECTED_WALKING_CLASS_COUNTS[subject_id],
                seed=seed,
            )
            if permutations is None:
                complete = False
                break
            for (_, block), permutation in zip(subject_blocks, permutations):
                for profile, client_index in zip(block, permutation):
                    assignment[profile.run_uid] = client_index
        if complete and set(assignment) == set(profiles):
            final_assignment = assignment
            break
    if final_assignment is None:
        raise RuntimeError(
            "No speed-block allocation satisfies the fixed run and window audit targets."
        )

    block_audit: list[dict[str, Any]] = []
    for block_number, (stratum, block) in enumerate(blocks):
        owners = [final_assignment[record.run_uid] for record in block]
        if set(owners) != set(range(num_clients)):
            raise AssertionError(f"Complete speed block {block_number} is not split 1/client.")
        block_audit.append(
            {
                "block_number": block_number,
                "source_id": stratum[0],
                "subject_id": stratum[1],
                "direction": stratum[2],
                "size": len(block),
                "minimum_speed_mps": min(record.speed_mps for record in block),
                "maximum_speed_mps": max(record.speed_mps for record in block),
                "run_to_client": {
                    record.run_uid: CLIENT_IDS[final_assignment[record.run_uid]]
                    for record in block
                },
            }
        )
    for stratum, records in remainders:
        owners = [final_assignment[record.run_uid] for record in records]
        if len(owners) != len(set(owners)):
            raise AssertionError(f"Remainder stratum {stratum} duplicates a client.")
        block_audit.append(
            {
                "block_number": None,
                "source_id": stratum[0],
                "subject_id": stratum[1],
                "direction": stratum[2],
                "size": len(records),
                "minimum_speed_mps": min(record.speed_mps for record in records),
                "maximum_speed_mps": max(record.speed_mps for record in records),
                "run_to_client": {
                    record.run_uid: CLIENT_IDS[final_assignment[record.run_uid]]
                    for record in records
                },
            }
        )
    return final_assignment, block_audit


def _speed_distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum_mps": float(array.min()),
        "maximum_mps": float(array.max()),
        "mean_mps": float(array.mean()),
        "std_mps": float(array.std()),
    }


def _audit_and_summarize(
    partitions: Sequence[ClientPartition],
    profiles: Mapping[str, _RunProfile],
    assignment: Mapping[str, int],
    block_audit: Sequence[Mapping[str, Any]],
    classification_train_indices: Sequence[int],
    regression_train_indices: Sequence[int],
    test_uids: set[str],
    seed: int,
) -> dict[str, Any]:
    expected_train_uids = set(profiles)
    run_sets = [set(partition.run_uids) for partition in partitions]
    classification_sets = [set(partition.classification_indices) for partition in partitions]
    regression_sets = [set(partition.regression_indices) for partition in partitions]

    pairwise_run_overlap = 0
    pairwise_classification_overlap = 0
    pairwise_regression_overlap = 0
    for left, right in itertools.combinations(range(3), 2):
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

    client_summaries: list[dict[str, Any]] = []
    for client_index, partition in enumerate(partitions):
        client_profiles = [profiles[run_uid] for run_uid in partition.run_uids]
        subjects = sorted({profile.subject_id for profile in client_profiles})
        source_subject_groups = sorted(
            {f"{profile.source_id}:{profile.subject_id}" for profile in client_profiles}
        )
        subject_run_counts = Counter(profile.subject_id for profile in client_profiles)
        source_subject_run_counts = Counter(
            f"{profile.source_id}:{profile.subject_id}" for profile in client_profiles
        )
        direction_run_counts = Counter(
            f"{profile.source_id}:{profile.direction}" for profile in client_profiles
        )
        source_run_counts = Counter(profile.source_id for profile in client_profiles)
        walking_class_counts = Counter()
        for profile in client_profiles:
            walking_class_counts[f"subject_{profile.subject_id}_walking"] += (
                profile.walking_windows
            )
        no_walking_count = sum(profile.no_walking_windows for profile in client_profiles)
        classification_class_counts = {
            "no_walking": no_walking_count,
            **dict(sorted(walking_class_counts.items())),
        }
        walking_count = sum(profile.walking_windows for profile in client_profiles)

        if subjects != list(EXPECTED_SUBJECTS):
            raise AssertionError(
                f"{partition.client_id} does not contain every subject: {subjects}."
            )
        if source_subject_groups != list(EXPECTED_SOURCE_SUBJECT_GROUPS):
            raise AssertionError(
                f"{partition.client_id} does not contain all eight source/subject groups."
            )
        if len(client_profiles) != EXPECTED_RUN_COUNTS[client_index]:
            raise AssertionError(f"Unexpected run count for {partition.client_id}.")
        if walking_count != EXPECTED_REGRESSION_WINDOW_COUNTS[client_index]:
            raise AssertionError(f"Unexpected walking-window count for {partition.client_id}.")
        if no_walking_count != EXPECTED_NO_WALKING_WINDOW_COUNTS[client_index]:
            raise AssertionError(f"Unexpected no-walking count for {partition.client_id}.")
        if len(partition.classification_indices) != EXPECTED_CLASSIFICATION_WINDOW_COUNTS[
            client_index
        ]:
            raise AssertionError(
                f"Unexpected classification-window count for {partition.client_id}."
            )
        if len(partition.regression_indices) != EXPECTED_REGRESSION_WINDOW_COUNTS[
            client_index
        ]:
            raise AssertionError(f"Unexpected regression-window count for {partition.client_id}.")
        for subject_id in EXPECTED_SUBJECTS:
            if subject_run_counts[subject_id] != EXPECTED_SUBJECT_RUN_COUNTS[subject_id][
                client_index
            ]:
                raise AssertionError(
                    f"Unexpected {subject_id} run count for {partition.client_id}."
                )
            if walking_class_counts[f"subject_{subject_id}_walking"] != (
                EXPECTED_WALKING_CLASS_COUNTS[subject_id][client_index]
            ):
                raise AssertionError(
                    f"Unexpected {subject_id} window count for {partition.client_id}."
                )
        for direction in ("N_to_S", "S_to_N"):
            if direction_run_counts[f"testing_20251124:{direction}"] != 10:
                raise AssertionError(
                    f"{partition.client_id} must have 10 newer-dataset {direction} runs."
                )

        client_summaries.append(
            {
                "client_id": partition.client_id,
                "num_runs": len(client_profiles),
                "num_classification_windows": len(partition.classification_indices),
                "num_regression_windows": len(partition.regression_indices),
                "num_walking_windows": walking_count,
                "num_no_walking_windows": no_walking_count,
                "subjects": subjects,
                "source_subject_groups": source_subject_groups,
                "subject_run_counts": dict(sorted(subject_run_counts.items())),
                "source_run_counts": dict(sorted(source_run_counts.items())),
                "source_subject_run_counts": dict(sorted(source_subject_run_counts.items())),
                "direction_run_counts": dict(sorted(direction_run_counts.items())),
                "classification_class_window_counts": classification_class_counts,
                "run_speed_distribution": _speed_distribution(
                    [profile.speed_mps for profile in client_profiles]
                ),
                "run_uids": list(partition.run_uids),
            }
        )

    run_to_client = {
        run_uid: CLIENT_IDS[client_index]
        for run_uid, client_index in sorted(assignment.items())
    }
    return {
        "schema_version": 1,
        "experiment": "combined_3client_run_grouped_approximate_iid",
        "seed": seed,
        "num_clients": 3,
        "client_ids": list(CLIENT_IDS),
        "partition_unit": "complete_run_before_window_shuffling",
        "stratification_fields": ["source_id", "subject_id", "direction"],
        "speed_block_size": 3,
        "speed_ordering": "ascending_APDM_mean_left_right_gait_speed_then_run_uid",
        "canonical_run_uid": "{source_id}:{subject_id_3_digits}:{run_index_3_digits}",
        "num_training_runs": len(expected_train_uids),
        "num_test_runs": len(test_uids),
        "num_classification_training_windows": len(classification_train_indices),
        "num_regression_training_windows": len(regression_train_indices),
        "clients": client_summaries,
        "run_to_client": run_to_client,
        "speed_blocks": list(block_audit),
        "audits": {
            "classification_and_regression_train_run_sets_identical": True,
            "classification_and_regression_test_run_sets_identical": True,
            "canonical_source_aware_run_uids": True,
            "pairwise_client_run_overlap": pairwise_run_overlap,
            "pairwise_client_classification_window_overlap": pairwise_classification_overlap,
            "pairwise_client_regression_window_overlap": pairwise_regression_overlap,
            "client_run_union_equals_all_training_runs": True,
            "classification_window_coverage_exactly_once": True,
            "regression_window_coverage_exactly_once": True,
            "train_test_run_overlap": 0,
            "test_runs_owned_by_clients": 0,
            "same_run_owner_for_both_tasks": True,
            "all_subjects_present_on_every_client": True,
            "all_source_subject_groups_present_on_every_client": True,
            "newer_source_each_direction_runs_per_client": 10,
        },
    }


def build_combined_iid_partitions(
    classification_artifact: Mapping[str, Any],
    regression_artifact: Mapping[str, Any],
    num_clients: int = 3,
    seed: int = 4601,
) -> tuple[list[ClientPartition], dict[str, Any]]:
    """Build the shared three-client run partition and its leakage audit.

    The returned indices refer to the original, global artifact tensors.  Local
    training code may shuffle those indices after partitioning, but must not
    reassign individual windows to another client.
    """

    if num_clients != 3:
        raise ValueError("The combined IID experiment is fixed to exactly three clients.")
    if seed != 4601:
        raise ValueError("The audited combined IID experiment is fixed to seed 4601.")

    (
        profiles,
        classification_train_indices,
        regression_train_indices,
        test_uids,
    ) = _build_run_profiles(classification_artifact, regression_artifact)
    if len(profiles) != 112 or len(test_uids) != 28:
        raise AssertionError(
            f"Expected 112 training and 28 test runs; found {len(profiles)}/{len(test_uids)}."
        )
    if sorted({profile.subject_id for profile in profiles.values()}) != list(
        EXPECTED_SUBJECTS
    ):
        raise AssertionError("The combined artifact has an unexpected subject set.")
    source_subject_groups = sorted(
        {f"{profile.source_id}:{profile.subject_id}" for profile in profiles.values()}
    )
    if source_subject_groups != list(EXPECTED_SOURCE_SUBJECT_GROUPS):
        raise AssertionError("The combined artifact has unexpected source/subject groups.")

    assignment, block_audit = _assign_profiles(profiles, num_clients, seed)
    partitions: list[ClientPartition] = []
    for client_index, client_id in enumerate(CLIENT_IDS):
        run_uids = tuple(
            sorted(run_uid for run_uid, owner in assignment.items() if owner == client_index)
        )
        classification_indices = tuple(
            sorted(
                index
                for run_uid in run_uids
                for index in profiles[run_uid].classification_indices
            )
        )
        regression_indices = tuple(
            sorted(
                index
                for run_uid in run_uids
                for index in profiles[run_uid].regression_indices
            )
        )
        partitions.append(
            ClientPartition(
                client_id=client_id,
                run_uids=run_uids,
                classification_indices=classification_indices,
                regression_indices=regression_indices,
            )
        )

    summary = _audit_and_summarize(
        partitions=partitions,
        profiles=profiles,
        assignment=assignment,
        block_audit=block_audit,
        classification_train_indices=classification_train_indices,
        regression_train_indices=regression_train_indices,
        test_uids=test_uids,
        seed=seed,
    )
    return partitions, summary


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


from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/iid_partitioning.py"
raise ImportError(
    "Quarantined legacy module: legacy/iid_partitioning.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import random
from collections import defaultdict
from typing import Sequence

import torch

from .federated import ClientPartition


def _as_index_list(indices: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(indices, torch.Tensor):
        return [int(index) for index in indices.tolist()]
    return [int(index) for index in indices]


def build_iid_run_partitions(
    artifact: dict[str, object],
    num_clients: int,
    seed: int,
) -> tuple[list[ClientPartition], dict[str, list[str]]]:
    """Shard global train by subject-owned full runs, not individual windows.

    Each client receives at least one full training run from every subject. Keeping
    a run wholly on one client avoids splitting heavily-overlapping windows from
    the same trial across clients.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")

    global_train_indices = _as_index_list(artifact["train_indices"])
    global_val_indices = set(_as_index_list(artifact["val_indices"]))
    global_test_indices = set(_as_index_list(artifact["test_indices"]))
    metadata = artifact["metadata"]

    subject_runs: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in global_train_indices:
        subject_id = str(metadata[index]["subject_id"])
        run_index = int(metadata[index]["run_index"])
        subject_runs[subject_id][run_index].append(index)

    subject_ids = sorted(subject_runs)
    for subject_id, runs in subject_runs.items():
        if len(runs) < num_clients:
            raise ValueError(
                f"Subject {subject_id} has {len(runs)} training runs, "
                f"but IID partitioning needs at least {num_clients}."
            )

    partitions: dict[str, list[int]] = {str(client_id): [] for client_id in range(num_clients)}
    client_subjects = {str(client_id): list(subject_ids) for client_id in range(num_clients)}

    for subject_id in subject_ids:
        run_items = sorted(subject_runs[subject_id].items())
        rng = random.Random(f"{seed}:{subject_id}:iid-runs")
        rng.shuffle(run_items)

        per_subject_counts = {str(client_id): 0 for client_id in range(num_clients)}
        for run_index, run_indices in run_items:
            client_id = min(
                per_subject_counts,
                key=lambda candidate: (per_subject_counts[candidate], len(partitions[candidate]), candidate),
            )
            sorted_indices = sorted(run_indices)
            partitions[client_id].extend(sorted_indices)
            per_subject_counts[client_id] += len(sorted_indices)

    seen_indices: set[int] = set()
    client_partitions: list[ClientPartition] = []
    for client_id in sorted(partitions, key=int):
        indices = sorted(partitions[client_id])
        if not indices:
            raise ValueError(f"Client {client_id} has no training windows.")

        overlap = seen_indices & set(indices)
        if overlap:
            raise ValueError(f"Client {client_id} overlaps another client on indices: {sorted(overlap)}")
        if set(indices) & global_val_indices:
            raise ValueError(f"Client {client_id} overlaps the global validation split.")
        if set(indices) & global_test_indices:
            raise ValueError(f"Client {client_id} overlaps the global test split.")

        subjects_present = {str(metadata[index]["subject_id"]) for index in indices}
        missing_subjects = sorted(set(subject_ids) - subjects_present)
        if missing_subjects:
            raise ValueError(f"Client {client_id} is missing subjects: {missing_subjects}")

        seen_indices.update(indices)
        client_partitions.append(
            ClientPartition(
                client_id=client_id,
                subject_ids=list(subject_ids),
                train_indices=indices,
            )
        )

    if seen_indices != set(global_train_indices):
        missing_indices = sorted(set(global_train_indices) - seen_indices)
        raise ValueError(f"IID partitions do not cover the full global train split: {missing_indices}")

    return client_partitions, client_subjects


def create_iid_partition_summary(
    artifact: dict[str, object],
    client_partitions: Sequence[ClientPartition],
    client_subjects: dict[str, list[str]],
) -> dict[str, object]:
    metadata = artifact["metadata"]
    clients = []
    for partition in client_partitions:
        subject_windows: dict[str, int] = defaultdict(int)
        subject_runs: dict[str, set[int]] = defaultdict(set)
        for index in partition.train_indices:
            subject_id = str(metadata[index]["subject_id"])
            run_index = int(metadata[index]["run_index"])
            subject_windows[subject_id] += 1
            subject_runs[subject_id].add(run_index)

        clients.append(
            {
                "client_id": partition.client_id,
                "subjects": partition.subject_ids,
                "num_train_windows": len(partition.train_indices),
                "num_train_runs": sum(len(runs) for runs in subject_runs.values()),
                "subject_window_counts": dict(sorted(subject_windows.items())),
                "subject_run_counts": {
                    subject_id: len(subject_runs[subject_id])
                    for subject_id in sorted(subject_runs)
                },
            }
        )

    return {
        "partition_mode": "iid_by_subject_full_train_runs",
        "client_subjects": client_subjects,
        "clients": clients,
        "num_global_train": len(_as_index_list(artifact["train_indices"])),
        "num_global_val": len(_as_index_list(artifact["val_indices"])),
        "num_global_test": len(_as_index_list(artifact["test_indices"])),
        "split_rule": (
            "Global train/val/test remains fixed by subject+run. Only global train is "
            "distributed to clients. For each subject, whole training runs are assigned "
            "to clients so every client sees every subject while overlapping windows "
            "from the same run stay on one client."
        ),
    }

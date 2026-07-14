from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .training import EvalResult, create_model


class IndexedArtifactDataset(Dataset):
    def __init__(self, artifact: dict[str, object], indices: Sequence[int], task: str) -> None:
        self.indices = [int(index) for index in indices]
        self.samples = artifact["samples"]
        self.mean = artifact["channel_mean"]
        self.std = artifact["channel_std"]
        if task == "regression":
            self.targets = artifact["regression_targets"]
        elif task == "classification":
            self.targets = artifact["classification_targets"]
        else:
            raise ValueError(f"Unsupported task: {task}")
        self.task = task

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_index = self.indices[index]
        x = self.samples[item_index]
        x = (x - self.mean.squeeze(0)) / self.std.squeeze(0)
        y = self.targets[item_index]
        if self.task == "regression":
            y = y.unsqueeze(0)
        return x.float(), y


@dataclass
class ClientPartition:
    client_id: str
    subject_ids: list[str]
    train_indices: list[int]


def _as_index_list(indices: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(indices, torch.Tensor):
        return [int(index) for index in indices.tolist()]
    return [int(index) for index in indices]


def _criterion_for_task(task: str) -> nn.Module:
    if task == "regression":
        return nn.MSELoss()
    if task == "classification":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported task: {task}")


def _score_for_task(task: str, outputs: torch.Tensor, targets: torch.Tensor) -> float:
    if task == "regression":
        return torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
    predictions = torch.argmax(outputs, dim=1)
    return (predictions == targets).float().mean().item()


def regression_r2_score(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = outputs.reshape(-1).float()
    actual = targets.reshape(-1).float()
    ss_res = torch.sum((actual - predictions) ** 2)
    ss_tot = torch.sum((actual - torch.mean(actual)) ** 2)
    if ss_tot.item() <= 1e-12:
        return 0.0
    return (1.0 - ss_res / ss_tot).item()


def round_robin_subject_map(subject_ids: Sequence[str], num_clients: int) -> dict[str, list[str]]:
    mapping = {str(client_id): [] for client_id in range(num_clients)}
    for index, subject_id in enumerate(sorted(subject_ids)):
        mapping[str(index % num_clients)].append(str(subject_id))
    return mapping


def build_client_partitions(
    artifact: dict[str, object],
    num_clients: int,
    client_subjects: dict[str, list[str]] | None = None,
) -> tuple[list[ClientPartition], dict[str, list[str]]]:
    """
    Build client partitions for federated learning, supporting both exclusive and shared subjects.
    
    A subject can now be assigned to multiple clients (shared subjects). When a subject is shared,
    its training data is replicated to all owning clients. Non-shared subjects are exclusive to one client.
    
    Args:
        artifact: Data artifact containing train/val/test indices and metadata
        num_clients: Number of clients
        client_subjects: Optional mapping of client_id -> list of subject_ids. 
                        Subjects can appear in multiple clients (shared) or just one (exclusive).
    
    Returns:
        Tuple of (client_partitions, resolved_subjects)
    """
    global_train_indices = _as_index_list(artifact["train_indices"])
    global_val_indices = set(_as_index_list(artifact["val_indices"]))
    global_test_indices = set(_as_index_list(artifact["test_indices"]))
    metadata = artifact["metadata"]

    train_subjects = sorted({str(metadata[index]["subject_id"]) for index in global_train_indices})
    expected_client_ids = {str(client_id) for client_id in range(num_clients)}
    resolved_subjects = round_robin_subject_map(train_subjects, num_clients) if client_subjects is None else {
        str(client_id): [str(subject_id) for subject_id in subject_ids]
        for client_id, subject_ids in client_subjects.items()
    }

    if set(resolved_subjects) != expected_client_ids:
        raise ValueError(
            f"Client ids must match {sorted(expected_client_ids)}, got {sorted(resolved_subjects)}."
        )

    # Build subject_owners: subject -> set of client IDs (supports multiple clients per subject)
    subject_owners: dict[str, set[str]] = {}
    for client_id, subject_ids in sorted(resolved_subjects.items()):
        for subject_id in subject_ids:
            if subject_id not in subject_owners:
                subject_owners[subject_id] = set()
            subject_owners[subject_id].add(client_id)
    
    # Identify shared subjects (subjects owned by multiple clients)
    shared_subjects: set[str] = {subject_id for subject_id, owners in subject_owners.items() if len(owners) > 1}

    missing_subjects = sorted(set(train_subjects) - set(subject_owners))
    if missing_subjects:
        raise ValueError(
            f"Training subjects missing from client assignment: {missing_subjects}."
        )
    unknown_subjects = sorted(set(subject_owners) - set(train_subjects))
    if unknown_subjects:
        raise ValueError(
            f"Client assignment includes subjects not present in the training split: {unknown_subjects}."
        )

    # Build partitions: each client gets all indices for their exclusive subjects and all shared subjects they own
    partitions: dict[str, list[int]] = {client_id: [] for client_id in sorted(expected_client_ids)}
    for index in global_train_indices:
        subject_id = str(metadata[index]["subject_id"])
        owners = subject_owners.get(subject_id)
        if owners is None:
            raise ValueError(f"No client assignment found for subject {subject_id}.")
        # Add this index to all clients that own the subject
        for owner_client_id in owners:
            partitions[owner_client_id].append(index)

    # Validate partitions
    all_seen_indices: set[int] = set()  # Track all unique indices seen (accounting for replication)
    exclusive_subject_indices: dict[str, set[int]] = {}  # Track exclusive subject indices per client
    
    # First pass: identify exclusive subject indices for overlap checking
    for client_id in sorted(partitions):
        exclusive_subject_indices[client_id] = set()
        for index in partitions[client_id]:
            subject_id = str(metadata[index]["subject_id"])
            if subject_id not in shared_subjects:
                exclusive_subject_indices[client_id].add(index)
        all_seen_indices.update(partitions[client_id])
    
    # Second pass: validate no inter-client overlap for exclusive subjects
    for i, client_id_1 in enumerate(sorted(partitions)):
        for client_id_2 in sorted(partitions)[i+1:]:
            # Check overlap only for exclusive subject indices
            overlap = exclusive_subject_indices[client_id_1] & exclusive_subject_indices[client_id_2]
            if overlap:
                raise ValueError(
                    f"Exclusive subject indices overlap between client {client_id_1} and {client_id_2}: {sorted(overlap)}"
                )

    # Validate no overlap with val/test and build final client partitions
    client_partitions: list[ClientPartition] = []
    for client_id in sorted(partitions):
        indices = sorted(partitions[client_id])
        if not indices:
            raise ValueError(f"Client {client_id} has no training windows.")
        if set(indices) & global_val_indices:
            raise ValueError(f"Client {client_id} overlaps the global validation split.")
        if set(indices) & global_test_indices:
            raise ValueError(f"Client {client_id} overlaps the global test split.")
        client_partitions.append(
            ClientPartition(
                client_id=client_id,
                subject_ids=list(resolved_subjects[client_id]),
                train_indices=indices,
            )
        )

    # Validate that all global training indices are covered (accounting for shared subject replication)
    if all_seen_indices != set(global_train_indices):
        missing_indices = sorted(set(global_train_indices) - all_seen_indices)
        raise ValueError(f"Client partitions do not cover the full global train split: {missing_indices}")

    return client_partitions, resolved_subjects


def build_loader(
    artifact: dict[str, object],
    indices: Sequence[int] | torch.Tensor,
    task: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = IndexedArtifactDataset(artifact=artifact, indices=_as_index_list(indices), task=task)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def run_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> EvalResult:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    gathered_outputs: list[torch.Tensor] = []
    gathered_targets: list[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad()

        outputs = model(x)
        if task == "regression":
            outputs = outputs.float()
            y = y.float()
        loss = criterion(outputs, y)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        gathered_outputs.append(outputs.detach().cpu())
        gathered_targets.append(y.detach().cpu())

    outputs = torch.cat(gathered_outputs, dim=0)
    targets = torch.cat(gathered_targets, dim=0)
    return EvalResult(
        loss=total_loss / max(total_examples, 1),
        score=_score_for_task(task, outputs, targets),
        outputs=outputs,
        targets=targets,
    )


def train_local_model(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> EvalResult:
    criterion = _criterion_for_task(task)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(epochs):
        run_loader(model, loader, criterion, device, task, optimizer=optimizer)
    return run_loader(model, loader, criterion, device, task, optimizer=None)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
) -> EvalResult:
    criterion = _criterion_for_task(task)
    return run_loader(model, loader, criterion, device, task, optimizer=None)


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: Iterable[np.ndarray]) -> None:
    current_state = model.state_dict()
    new_state = OrderedDict()
    for (key, current_value), new_value in zip(current_state.items(), parameters, strict=True):
        tensor = torch.from_numpy(np.asarray(new_value)).to(dtype=current_value.dtype)
        new_state[key] = tensor
    model.load_state_dict(new_state, strict=True)


def create_partition_summary(
    artifact: dict[str, object],
    client_partitions: Sequence[ClientPartition],
    client_subjects: dict[str, list[str]],
) -> dict[str, object]:
    metadata = artifact["metadata"]
    client_rows = []
    subject_clients: dict[str, list[str]] = {}
    for client_id, subject_ids in client_subjects.items():
        for subject_id in subject_ids:
            subject_clients.setdefault(str(subject_id), []).append(str(client_id))
    shared_subjects = {
        subject_id: sorted(client_ids)
        for subject_id, client_ids in subject_clients.items()
        if len(client_ids) > 1
    }

    for partition in client_partitions:
        run_groups = {
            (str(metadata[index]["subject_id"]), int(metadata[index]["run_index"]))
            for index in partition.train_indices
        }
        client_rows.append(
            {
                "client_id": partition.client_id,
                "subjects": partition.subject_ids,
                "num_train_windows": len(partition.train_indices),
                "num_train_runs": len(run_groups),
            }
        )

    return {
        "client_subjects": client_subjects,
        "shared_subjects": shared_subjects,
        "total_client_train_windows": sum(row["num_train_windows"] for row in client_rows),
        "clients": client_rows,
        "num_global_train": len(_as_index_list(artifact["train_indices"])),
        "num_global_val": len(_as_index_list(artifact["val_indices"])),
        "num_global_test": len(_as_index_list(artifact["test_indices"])),
        "split_rule": "Global train/val/test is fixed first by subject+run. Only global train is sharded across clients by subject ownership.",
    }


def save_partition_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def save_round_history(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    preferred = [
        "round",
        "client_id",
        "num_examples",
        "train_loss",
        "train_score",
        "train_r2",
        "val_loss",
        "val_score",
        "val_r2",
    ]
    discovered = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in discovered]
    fieldnames.extend(sorted(discovered - set(fieldnames)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_federated_model(artifact: dict[str, object], task: str, mask_aware: bool = False) -> nn.Module:
    return create_model(artifact=artifact, task=task, mask_aware=mask_aware)

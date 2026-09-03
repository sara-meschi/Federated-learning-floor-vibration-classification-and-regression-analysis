from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/training.py"
raise ImportError(
    "Quarantined legacy module: legacy/training.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from redo_by_sara.models import SimpleCNN1D
from .mask_aware_model import MaskAwareSimpleCNN1D


class ArtifactDataset(Dataset):
    def __init__(self, artifact: dict[str, object], split: str, task: str) -> None:
        index_keys = {
            "train": "train_indices",
            "val": "val_indices",
            "test": "test_indices",
        }
        self.indices = artifact[index_keys[split]].tolist()
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
class EvalResult:
    loss: float
    score: float
    outputs: torch.Tensor
    targets: torch.Tensor


MetricLogger = Callable[[dict[str, float]], None]


def create_model(artifact: dict[str, object], task: str, mask_aware: bool = False) -> nn.Module:
    in_channels = int(artifact["samples"].shape[1])
    if task == "regression":
        if mask_aware:
            raise ValueError("Mask-aware model is currently implemented for classification only.")
        return SimpleCNN1D(in_channels=in_channels, output_dim=1)
    num_classes = len(artifact["subject_to_class"])
    if mask_aware:
        return MaskAwareSimpleCNN1D(signal_channels=in_channels, output_dim=num_classes)
    return SimpleCNN1D(in_channels=in_channels, output_dim=num_classes)


def create_loaders(
    artifact: dict[str, object], task: str, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = ArtifactDataset(artifact, split="train", task=task)
    val_dataset = ArtifactDataset(artifact, split="val", task=task)
    test_dataset = ArtifactDataset(artifact, split="test", task=task)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def _regression_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def _classification_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    task: str,
) -> EvalResult:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    gathered_outputs = []
    gathered_targets = []

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
    if task == "regression":
        score = _regression_score(outputs, targets)
    else:
        score = _classification_score(outputs, targets)
    return EvalResult(
        loss=total_loss / max(total_examples, 1),
        score=score,
        outputs=outputs,
        targets=targets,
    )


def fit(
    artifact: dict[str, object],
    task: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    seed: int,
    metric_logger: MetricLogger | None = None,
    run_logger: Any | None = None,
) -> tuple[nn.Module, list[dict[str, float]], EvalResult]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(artifact, task).to(device)
    train_loader, val_loader, test_loader = create_loaders(artifact, task, batch_size, num_workers)

    if run_logger is not None:
        run_logger.watch(model, log="gradients", log_freq=max(10, len(train_loader)))

    criterion: nn.Module
    if task == "regression":
        criterion = nn.MSELoss()
        best_is_better = lambda current, best: current < best
        best_score = float("inf")
    else:
        criterion = nn.CrossEntropyLoss()
        best_is_better = lambda current, best: current > best
        best_score = float("-inf")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, optimizer, device, task)
        val_result = run_epoch(model, val_loader, criterion, None, device, task)
        row = {
            "epoch": float(epoch),
            "train_loss": train_result.loss,
            "train_score": train_result.score,
            "val_loss": val_result.loss,
            "val_score": val_result.score,
        }
        history.append(row)
        if metric_logger is not None:
            metric_logger(row)
        if best_is_better(val_result.score, best_score):
            best_score = val_result.score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_result = run_epoch(model, test_loader, criterion, None, device, task)
    return model, history, test_result

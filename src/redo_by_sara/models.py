"""Model architectures for the combined experiments.

``SimpleCNN1D`` is the only architecture the paper uses, for both tasks: nine input
channels for classification (eight class logits) and nine for regression (one
standardized residual).  Nine input channels on every client is what keeps parameter
shapes identical across clients so FedAvg can average them, which is why
channel-availability heterogeneity is implemented by **masking** the input in the
dataset's ``__getitem__`` after normalization, never by slicing it or by making the
model mask-aware.  See ``CLAUDE.md`` invariant 5.

``MaskAwareSimpleCNN1D`` used to live here.  It doubled the input to 18 channels by
concatenating a mask-indicator channel, which breaks that invariant, and it is now
quarantined at ``legacy/mask_aware_model.py``.  Do not resurrect it.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels: int, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)

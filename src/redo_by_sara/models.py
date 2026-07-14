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


class MaskAwareSimpleCNN1D(SimpleCNN1D):
    def __init__(self, signal_channels: int, output_dim: int) -> None:
        super().__init__(in_channels=signal_channels * 2, output_dim=output_dim)
        self.signal_channels = signal_channels

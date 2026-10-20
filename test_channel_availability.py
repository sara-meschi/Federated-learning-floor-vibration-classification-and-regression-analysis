"""Channel-availability masks over corridor positions (review plan §1.3)."""

from __future__ import annotations

import pytest
import torch

from redo_by_sara.channel_availability import (
    apply_channel_mask,
    artifact_column,
    channel_audit,
    channel_layout,
    positions_to_mask,
    raw_numpy_index,
)

CLIENTS = ("client_0", "client_1", "client_2")


def test_one_indexed_conversions() -> None:
    channels = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    assert [raw_numpy_index(c) for c in channels] == [0, 1, 2, 3, 4, 5, 6, 7, 9]
    assert [artifact_column(c) for c in channels] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    with pytest.raises(ValueError):
        artifact_column(9)  # channel 9 (8y) is dropped


def test_position_8_carries_channels_8_and_10_together() -> None:
    assert positions_to_mask([8]) == (False,) * 7 + (True, True)
    assert positions_to_mask([1, 2, 3]) == (True, True, True) + (False,) * 6


@pytest.mark.parametrize(
    ("rho", "positions", "channels", "shared"),
    [
        (1.0, [[1, 2, 3, 4, 5, 6, 7, 8]] * 3, [9, 9, 9], [1, 2, 3, 4, 5, 6, 7, 8]),
        (0.375, [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [1, 2, 3, 8]], [5, 5, 5], [1, 2, 3]),
        (0.0, [[1, 2, 3], [4, 5, 6], [7, 8]], [3, 3, 3], []),
    ],
)
def test_layouts(rho, positions, channels, shared) -> None:
    audit = channel_audit(rho, 3, CLIENTS)
    assert audit["rho"] == rho
    assert audit["shared_positions"] == shared
    assert [c["positions"] for c in audit["clients"]] == positions
    assert [c["num_channels"] for c in audit["clients"]] == channels
    covered = set().union(*(set(c["positions"]) for c in audit["clients"]))
    assert covered == set(range(1, 9))  # every position instrumented somewhere


def test_position_8_holder_is_logged() -> None:
    assert channel_audit(0.0, 3, CLIENTS)["position_8_holder"] == ["client_2"]
    assert channel_audit(0.375, 3, CLIENTS)["position_8_holder"] == ["client_2"]


def test_disjoint_layout_has_no_shared_channel() -> None:
    masks = [positions_to_mask(p) for p in channel_layout(0.0, 3)]
    for column in range(9):
        assert sum(mask[column] for mask in masks) == 1


def test_only_declared_layouts_and_k3() -> None:
    with pytest.raises(ValueError):
        channel_layout(0.33, 3)
    with pytest.raises(ValueError):
        channel_layout(0.0, 2)


def test_masking_zeroes_after_normalization_and_keeps_the_rest() -> None:
    sample = torch.randn(9, 50)
    mask = torch.tensor(positions_to_mask([1, 2, 3]))
    masked = apply_channel_mask(sample, mask)
    assert masked.shape == sample.shape  # never sliced
    assert torch.equal(masked[:3], sample[:3])
    assert torch.count_nonzero(masked[3:]) == 0
    windows = torch.randn(4, 9, 50)
    assert torch.count_nonzero(apply_channel_mask(windows, mask)[:, 3:]) == 0
    assert apply_channel_mask(sample, None) is sample
    full = apply_channel_mask(sample, torch.ones(9, dtype=torch.bool))
    assert torch.equal(full, sample)  # rho = 1.0 is bitwise the unmasked input

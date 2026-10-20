"""Channel-availability heterogeneity by masking (review plan §1.3).

The 9 selected channels ``[1, 2, 3, 4, 5, 6, 7, 8, 10]`` (1-indexed, a priori from
hallway placement) cover **8 distinct corridor positions**: sensors 1-7 are one
uniaxial unit each, and channels 8 (x) and 10 (z) are separate units co-located at an
eighth point (``docs/sensor_layout.md``). The partition unit is therefore the
**position**; channels 8 and 10 always travel together.

Every client keeps the full 9-channel input. Unavailable channels are zeroed in the
dataset's ``__getitem__`` *after* normalization, so a masked channel equals the global
train mean ("no information"). The model stays plain ``SimpleCNN1D`` with 9 input
channels, so every client has identical parameter shapes and FedAvg can average them.
This is covariate shift / sensor-availability heterogeneity in a shared feature space,
i.e. horizontal FL, not vertical FL.

Overlap ratio ``rho = |positions held by every client| / 8``. Three contiguous-block
layouts at K=3, fixed before any run:

``rho = 1.0``    every client holds all 8 positions (identical to IID K=3)
``rho = 0.375``  shared core {1, 2, 3}; private {4, 5} / {6, 7} / {8}
``rho = 0.0``    disjoint {1, 2, 3} / {4, 5, 6} / {7, 8}

Positions 1 -> 8 run down the corridor, so contiguous blocks are contiguous corridor
segments. Channel 8 is the only horizontal-axis channel; the audit names which client
holds position 8.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

#: The 9 selected sensor channels, 1-indexed, as in every config.
SELECTED_CHANNELS_ONE_BASED = (1, 2, 3, 4, 5, 6, 7, 8, 10)
NUM_POSITIONS = 8
#: Corridor position -> the 1-indexed channels physically located there.
POSITION_CHANNELS_ONE_BASED: dict[int, tuple[int, ...]] = {
    1: (1,), 2: (2,), 3: (3,), 4: (4,), 5: (5,), 6: (6,), 7: (7,), 8: (8, 10),
}

#: Layouts keyed by their exact overlap ratio. Fixed in the S4 plan, not tuned.
CHANNEL_LAYOUTS_K3: dict[float, tuple[tuple[int, ...], ...]] = {
    1.0: ((1, 2, 3, 4, 5, 6, 7, 8),) * 3,
    0.375: ((1, 2, 3, 4, 5), (1, 2, 3, 6, 7), (1, 2, 3, 8)),
    0.0: ((1, 2, 3), (4, 5, 6), (7, 8)),
}
CHANNEL_OVERLAPS = tuple(sorted(CHANNEL_LAYOUTS_K3, reverse=True))


def raw_numpy_index(channel_one_based: int) -> int:
    """1-indexed sensor channel -> 0-indexed column of the raw 20-channel recording."""
    return int(channel_one_based) - 1


def artifact_column(channel_one_based: int) -> int:
    """1-indexed sensor channel -> column of the 9-channel artifact tensor."""
    return SELECTED_CHANNELS_ONE_BASED.index(int(channel_one_based))


# The two conversions, asserted at import: a silent off-by-one would shift every
# channel and is undetectable downstream (CLAUDE.md).
assert [raw_numpy_index(c) for c in SELECTED_CHANNELS_ONE_BASED] == [0, 1, 2, 3, 4, 5, 6, 7, 9]
assert [artifact_column(c) for c in SELECTED_CHANNELS_ONE_BASED] == list(range(9))
assert sorted(c for chans in POSITION_CHANNELS_ONE_BASED.values() for c in chans) == list(
    SELECTED_CHANNELS_ONE_BASED
)


def positions_to_mask(positions: Sequence[int]) -> tuple[bool, ...]:
    """Positions held -> boolean availability over the 9 artifact columns."""
    available = {
        artifact_column(channel)
        for position in positions
        for channel in POSITION_CHANNELS_ONE_BASED[int(position)]
    }
    return tuple(column in available for column in range(len(SELECTED_CHANNELS_ONE_BASED)))


def overlap_ratio(layout: Sequence[Sequence[int]]) -> float:
    shared = set.intersection(*(set(positions) for positions in layout))
    return len(shared) / NUM_POSITIONS


def channel_layout(rho: float, num_clients: int) -> tuple[tuple[int, ...], ...]:
    if num_clients != 3:
        raise ValueError("Channel-availability layouts are defined at K=3 only.")
    for key, layout in CHANNEL_LAYOUTS_K3.items():
        if abs(float(rho) - key) < 1e-9:
            return layout
    raise ValueError(f"No channel layout for rho={rho}; defined: {CHANNEL_OVERLAPS}.")


def channel_audit(rho: float, num_clients: int, client_ids: Sequence[str]) -> dict[str, Any]:
    """Masks plus the properties §1.3 asks to be asserted and logged."""
    layout = channel_layout(rho, num_clients)
    if overlap_ratio(layout) != float(rho):
        raise AssertionError(f"Layout overlap {overlap_ratio(layout)} != declared rho {rho}.")
    union = set().union(*(set(positions) for positions in layout))
    if union != set(POSITION_CHANNELS_ONE_BASED):
        raise AssertionError(f"Positions {sorted(set(POSITION_CHANNELS_ONE_BASED) - union)} held by no client.")
    clients = []
    for client_id, positions in zip(client_ids, layout):
        mask = positions_to_mask(positions)
        channels = [c for c, keep in zip(SELECTED_CHANNELS_ONE_BASED, mask) if keep]
        # Channels 8 and 10 share position 8: a mask that holds one must hold both.
        if (8 in channels) != (10 in channels):
            raise AssertionError(f"{client_id}: channels 8 and 10 split across a mask.")
        clients.append(
            {
                "client_id": client_id,
                "positions": list(positions),
                "num_positions": len(positions),
                "channels_one_based": channels,
                "num_channels": len(channels),
                "channel_mask": list(mask),
                "holds_position_8_horizontal_channel": 8 in positions,
            }
        )
    return {
        "rho": float(rho),
        "rho_definition": "|positions held by every client| / 8",
        "shared_positions": sorted(set.intersection(*(set(p) for p in layout))),
        "layout": "contiguous corridor blocks",
        "masking": "zeroed after normalization in Dataset.__getitem__; model input stays 9 channels",
        "normalization": "global train-window statistics over all 9 channels, for comparability across rho",
        "evaluation_protocol": "global held-out test set with all 9 channels",
        "position_8_holder": [c["client_id"] for c in clients if c["holds_position_8_horizontal_channel"]],
        "clients": clients,
    }


def apply_channel_mask(sample: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero unavailable channels of an already-normalized sample.

    ``sample`` is ``(channels, time)`` or ``(windows, channels, time)``; the channel
    axis is the second-to-last.
    """
    if mask is None:
        return sample
    if sample.shape[-2] != mask.numel():
        raise ValueError(f"Mask over {mask.numel()} channels; sample has {sample.shape[-2]}.")
    return sample * mask.to(sample.dtype).unsqueeze(-1)

"""Quarantined mask-aware model, moved out of ``src/redo_by_sara/models.py``.

Kept as a record of an approach that must not be re-derived, not as something to use.

``MaskAwareSimpleCNN1D`` doubles the input to ``signal_channels * 2`` by concatenating a
mask-indicator channel. That is wrong for this project on two counts:

1. It breaks ``CLAUDE.md`` invariant 5. Channel-availability heterogeneity must be built
   by **masking a 9-channel input**, never by changing its width. Nine channels on every
   client is what keeps first-layer parameter shapes identical, and identical shapes are
   the only reason FedAvg can average client updates at all.
2. It buys nothing here. Each client's channel mask is **static**, so there is no
   information in a mask-indicator channel that the weights cannot absorb.

The mask belongs in the dataset's ``__getitem__``, applied **after** normalization, so a
masked channel reads as the training mean ("no information") rather than an extreme value.
The model stays plain ``SimpleCNN1D`` with 9 input channels.

``legacy/training.py`` raises ``"Mask-aware model is currently implemented for
classification only."`` — the regression half was never written.

See ``docs/session0_findings.md`` C3.
"""

from __future__ import annotations

_QUARANTINED_LEGACY_MODULE = "legacy/mask_aware_model.py"
raise ImportError(
    "Quarantined legacy module: legacy/mask_aware_model.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


from redo_by_sara.models import SimpleCNN1D


class MaskAwareSimpleCNN1D(SimpleCNN1D):
    def __init__(self, signal_channels: int, output_dim: int) -> None:
        super().__init__(in_channels=signal_channels * 2, output_dim=output_dim)
        self.signal_channels = signal_channels

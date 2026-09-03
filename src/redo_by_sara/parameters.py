"""Model parameter marshalling shared by every federated run mode.

Flower exchanges parameters as a list of ``numpy`` arrays ordered exactly like
``model.state_dict()``.  These two functions are the only conversion between that
representation and a ``torch`` module, and they are deliberately kept in a module
with no training, partitioning, or configuration dependencies so that any run mode
can import them.

The pair must round trip exactly: ``set_parameters(model, get_parameters(model))``
leaves every tensor bit-identical.  ``_run_flower_core`` asserts this at round 0
before the first fit, because a silent dtype or ordering change here would corrupt
FedAvg without producing an error.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
from torch import nn


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: Iterable[np.ndarray]) -> None:
    current_state = model.state_dict()
    new_state = OrderedDict()
    for (key, current_value), new_value in zip(current_state.items(), parameters, strict=True):
        tensor = torch.from_numpy(np.asarray(new_value)).to(dtype=current_value.dtype)
        new_state[key] = tensor
    model.load_state_dict(new_state, strict=True)

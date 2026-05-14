from __future__ import annotations

import torch
from torch import nn


def build_loss(
    name: str = "cross_entropy",
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    if name != "cross_entropy":
        raise ValueError(f"Unsupported loss: {name}")
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

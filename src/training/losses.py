from __future__ import annotations
from torch import nn

def build_loss(name: str = "cross_entropy") -> nn.Module:
    if name != "cross_entropy":
        raise ValueError(f"Unsupported loss: {name}")
    return nn.CrossEntropyLoss()

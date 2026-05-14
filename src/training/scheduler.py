from __future__ import annotations
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

def build_scheduler(
    optimizer: Optimizer,
    name: str,
    t_max: int,
    eta_min: float = 0.0,
) -> LRScheduler:
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")
    return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

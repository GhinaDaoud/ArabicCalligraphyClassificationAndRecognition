from __future__ import annotations
from torch.optim import AdamW, Optimizer

def build_optimizer(
    model,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> Optimizer:
    return AdamW(
        [
            {"params": model.backbone_parameters(), "lr": backbone_lr},
            {"params": model.head_parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

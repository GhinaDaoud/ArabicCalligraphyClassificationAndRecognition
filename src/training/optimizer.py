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


def set_optimizer_lrs(optimizer: Optimizer, backbone_lr: float, head_lr: float) -> None:
    if len(optimizer.param_groups) < 2:
        raise ValueError("Expected optimizer with separate backbone and head parameter groups.")
    optimizer.param_groups[0]["lr"] = backbone_lr
    optimizer.param_groups[1]["lr"] = head_lr

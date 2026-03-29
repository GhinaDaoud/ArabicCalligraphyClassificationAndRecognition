from __future__ import annotations

from pathlib import Path

import torch

def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    config: dict,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint

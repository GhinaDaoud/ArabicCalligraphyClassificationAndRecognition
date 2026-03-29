from __future__ import annotations

import time

import torch

from src.evaluation.metrics import classification_metrics_from_confusion, confusion_matrix

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


def _run_epoch(
    model,
    dataloader,
    criterion,
    device: torch.device,
    optimizer=None,
    epoch_index: int | None = None,
    total_epochs: int | None = None,
    phase_name: str = "train",
    throttle_seconds_per_batch: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    total_examples = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    if epoch_index is not None and total_epochs is not None:
        description = f"{phase_name.capitalize()} {epoch_index:03d}/{total_epochs:03d}"
    else:
        description = phase_name.capitalize()

    if tqdm is not None:
        iterator = tqdm(
            dataloader,
            desc=description,
            leave=False,
            dynamic_ncols=True,
        )
    else:
        iterator = dataloader

    for step_index, (inputs, targets) in enumerate(iterator, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size

        predictions = torch.argmax(logits, dim=1)
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if tqdm is not None:
            iterator.set_postfix(
                loss=f"{running_loss / max(total_examples, 1):.4f}",
                batch=step_index,
            )
        if throttle_seconds_per_batch > 0:
            time.sleep(throttle_seconds_per_batch)

    predictions = torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long)
    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)

    num_classes = len(getattr(dataloader.dataset, "class_to_index", {}))
    if hasattr(dataloader.dataset, "dataset") and hasattr(dataloader.dataset.dataset, "class_to_index"):
        num_classes = len(dataloader.dataset.dataset.class_to_index)

    matrix = confusion_matrix(predictions, targets, num_classes=num_classes)
    metrics = classification_metrics_from_confusion(matrix)
    metrics["loss"] = running_loss / total_examples if total_examples > 0 else 0.0
    return metrics


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    epoch_index: int | None = None,
    total_epochs: int | None = None,
    throttle_seconds_per_batch: float = 0.0,
) -> dict[str, float]:
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        phase_name="train",
        throttle_seconds_per_batch=throttle_seconds_per_batch,
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device: torch.device,
    epoch_index: int | None = None,
    total_epochs: int | None = None,
    throttle_seconds_per_batch: float = 0.0,
) -> dict[str, float]:
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=None,
        device=device,
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        phase_name="val",
        throttle_seconds_per_batch=throttle_seconds_per_batch,
    )

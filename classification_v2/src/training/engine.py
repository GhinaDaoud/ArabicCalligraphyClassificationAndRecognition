from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.metrics import accuracy_from_confmat, confusion_matrix_torch, macro_f1_from_confmat, per_class_recall_from_confmat
from utils.io import ensure_dir

try:
    # Newer PyTorch API
    from torch.amp import GradScaler as TorchGradScaler  # type: ignore
    from torch.amp import autocast as torch_autocast  # type: ignore
except Exception:
    # Older PyTorch API
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore
    from torch.cuda.amp import autocast as torch_autocast  # type: ignore


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float
    per_class_recall: list[float]
    confusion_matrix: torch.Tensor


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: TorchGradScaler | None = None,
    amp_enabled: bool = True,
    num_classes: int = 7,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if amp_enabled and device.type == "cuda":
            try:
                # Newer API
                amp_ctx = torch_autocast(device_type=device.type, enabled=True)
            except TypeError:
                # Older API
                amp_ctx = torch_autocast(enabled=True)
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            logits = model(x)
            loss = criterion(logits, y)

        if is_train:
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

        pred = logits.argmax(dim=1)
        all_true.append(y.detach().cpu())
        all_pred.append(pred.detach().cpu())

    if total_samples == 0:
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        return EpochResult(loss=0.0, accuracy=0.0, macro_f1=0.0, per_class_recall=[0.0] * num_classes, confusion_matrix=cm)

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    cm = confusion_matrix_torch(y_true, y_pred, num_classes=num_classes)
    return EpochResult(
        loss=total_loss / total_samples,
        accuracy=accuracy_from_confmat(cm),
        macro_f1=macro_f1_from_confmat(cm),
        per_class_recall=per_class_recall_from_confmat(cm),
        confusion_matrix=cm,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    amp_enabled: bool = True,
) -> EpochResult:
    return _run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        amp_enabled=amp_enabled,
        num_classes=num_classes,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer_warmup: torch.optim.Optimizer,
    optimizer_finetune: torch.optim.Optimizer,
    scheduler_finetune: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    num_classes: int,
    warmup_epochs: int,
    finetune_epochs: int,
    amp_enabled: bool,
    early_stopping_patience: int,
    monitor: str,
    ckpt_dir: str | Path,
    stage_callbacks: dict[str, callable] | None = None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    ckpt_path = ensure_dir(ckpt_dir)
    best_path = ckpt_path / "best.pt"
    scaler = TorchGradScaler(enabled=(amp_enabled and device.type == "cuda"))

    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_epoch = -1
    no_improve = 0

    total_epochs = warmup_epochs + finetune_epochs
    last_stage = ""

    for epoch in range(1, total_epochs + 1):
        stage = "warmup" if epoch <= warmup_epochs else "finetune"
        if stage != last_stage and stage_callbacks and stage in stage_callbacks:
            stage_callbacks[stage]()
        last_stage = stage
        optimizer = optimizer_warmup if stage == "warmup" else optimizer_finetune

        tr = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            num_classes=num_classes,
        )
        va = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
            num_classes=num_classes,
        )

        if stage == "finetune" and scheduler_finetune is not None:
            scheduler_finetune.step()

        row = {
            "epoch": float(epoch),
            "stage": 0.0 if stage == "warmup" else 1.0,
            "train_loss": tr.loss,
            "train_acc": tr.accuracy,
            "train_macro_f1": tr.macro_f1,
            "val_loss": va.loss,
            "val_acc": va.accuracy,
            "val_macro_f1": va.macro_f1,
        }
        history.append(row)

        score = va.macro_f1 if monitor == "macro_f1" else -va.loss
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": va.macro_f1,
                    "val_loss": va.loss,
                    "val_confusion_matrix": va.confusion_matrix.tolist(),
                },
                best_path,
            )
        else:
            no_improve += 1

        print(
            f"[Epoch {epoch:03d}/{total_epochs:03d}] stage={stage} "
            f"train(loss={tr.loss:.4f}, acc={tr.accuracy:.4f}, f1={tr.macro_f1:.4f}) "
            f"val(loss={va.loss:.4f}, acc={va.accuracy:.4f}, f1={va.macro_f1:.4f})"
        )

        if no_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}, best_epoch={best_epoch}")
            break

    best = {"best_score": best_score, "best_epoch": float(best_epoch), "checkpoint_path": str(best_path)}
    return history, best

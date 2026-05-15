from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import ProcessedCalligraphyDataset, build_transforms
from models.efficientnet_b0 import EfficientNetB0Classifier
from training.engine import evaluate_model, train_model
from training.imbalance import build_weighted_sampler, effective_num_class_weights
from utils.config import load_yaml
from utils.io import ensure_dir, write_history_csv, write_json
from utils.seed import set_global_seed


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "classification_v2" / "configs" / "efficientnet_b0_baseline.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 for classification_v2.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return p.parse_args(argv)


def _build_optimizer_finetune(
    model: EfficientNetB0Classifier,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": backbone_lr},
            {"params": model.classifier_parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def run_train(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml(args.config)

    exp_name = cfg["experiment"]["name"]
    seed = int(cfg["experiment"].get("seed", 42))
    set_global_seed(seed)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    imb_cfg = cfg["imbalance"]
    opt_cfg = cfg["optimization"]
    tr_cfg = cfg["training"]

    run_root = ensure_dir(REPO_ROOT / "classification_v2" / "outputs" / "training")
    run_name = f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(run_root / run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_ds = ProcessedCalligraphyDataset(data_cfg["train_csv"], transform=build_transforms(train=True))
    val_ds = ProcessedCalligraphyDataset(data_cfg["val_csv"], transform=build_transforms(train=False))
    test_ds = ProcessedCalligraphyDataset(data_cfg["test_csv"], transform=build_transforms(train=False))

    batch_size = int(tr_cfg.get("batch_size", 8))
    num_workers = int(tr_cfg.get("num_workers", 0))

    weighted_sampler_enabled = bool(imb_cfg.get("weighted_sampler", True))
    sampler_power = float(imb_cfg.get("sampler_power", 0.5))
    if weighted_sampler_enabled:
        train_sampler = build_weighted_sampler(train_ds.labels, power=sampler_power)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    num_classes = int(model_cfg["num_classes"])
    model = EfficientNetB0Classifier(
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )
    model.to(device)

    beta = float(imb_cfg.get("effective_num_beta", 0.999))
    class_w = effective_num_class_weights(train_ds.labels, num_classes=num_classes, beta=beta).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_w,
        label_smoothing=float(opt_cfg.get("label_smoothing", 0.05)),
    )

    head_lr = float(opt_cfg.get("head_lr", 3e-4))
    backbone_lr = float(opt_cfg.get("backbone_lr", 5e-5))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))

    model.freeze_backbone()
    optimizer_warmup = torch.optim.AdamW(model.classifier_parameters(), lr=head_lr, weight_decay=weight_decay)
    optimizer_finetune = _build_optimizer_finetune(
        model=model,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
    )

    warmup_epochs = int(tr_cfg.get("warmup_epochs", 4))
    finetune_epochs = int(tr_cfg.get("finetune_epochs", 16))
    if finetune_epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, T_max=finetune_epochs)
    else:
        scheduler = None

    unfreeze_last_blocks = int(tr_cfg.get("unfreeze_last_blocks", 2))
    stage_callbacks = {
        "warmup": lambda: model.freeze_backbone(),
        "finetune": lambda: model.unfreeze_last_feature_blocks(unfreeze_last_blocks),
    }

    print("Training setup")
    print(f"  Device: {device.type}")
    print(f"  Run dir: {run_dir}")
    print(f"  Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: warmup={warmup_epochs}, finetune={finetune_epochs}")
    print(f"  Weighted sampler: {weighted_sampler_enabled}")
    print(f"  Class weights beta: {beta}")

    history, best = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer_warmup=optimizer_warmup,
        optimizer_finetune=optimizer_finetune,
        scheduler_finetune=scheduler,
        device=device,
        num_classes=num_classes,
        warmup_epochs=warmup_epochs,
        finetune_epochs=finetune_epochs,
        amp_enabled=bool(tr_cfg.get("amp", True)),
        early_stopping_patience=int(tr_cfg.get("early_stopping_patience", 5)),
        monitor=str(tr_cfg.get("monitor", "macro_f1")),
        ckpt_dir=ckpt_dir,
        stage_callbacks=stage_callbacks,
    )

    history_path = run_dir / "history.csv"
    write_history_csv(history_path, history)
    losses_path = run_dir / "losses.csv"
    loss_rows = [
        {
            "epoch": row["epoch"],
            "stage": row["stage"],
            "train_loss": row["train_loss"],
            "val_loss": row["val_loss"],
            "lr_backbone": row.get("lr_backbone", 0.0),
            "lr_head": row.get("lr_head", 0.0),
        }
        for row in history
    ]
    write_history_csv(losses_path, loss_rows)

    best_ckpt = torch.load(best["checkpoint_path"], map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_eval = evaluate_model(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        amp_enabled=bool(tr_cfg.get("amp", True)),
    )
    test_eval = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        amp_enabled=bool(tr_cfg.get("amp", True)),
    )

    summary = {
        "run_name": run_name,
        "config_path": str(args.config),
        "best": best,
        "val": {
            "loss": val_eval.loss,
            "accuracy": val_eval.accuracy,
            "macro_f1": val_eval.macro_f1,
            "per_class_recall": val_eval.per_class_recall,
            "confusion_matrix": val_eval.confusion_matrix.tolist(),
        },
        "test": {
            "loss": test_eval.loss,
            "accuracy": test_eval.accuracy,
            "macro_f1": test_eval.macro_f1,
            "per_class_recall": test_eval.per_class_recall,
            "confusion_matrix": test_eval.confusion_matrix.tolist(),
        },
    }
    write_json(run_dir / "summary.json", summary)

    print(f"Saved history: {history_path}")
    print(f"Saved losses: {losses_path}")
    print(f"Saved summary: {run_dir / 'summary.json'}")
    print(
        f"Best val macro-F1={summary['val']['macro_f1']:.4f} | "
        f"Test macro-F1={summary['test']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    run_train()

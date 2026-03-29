from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import CalligraphyStyleDataset
from src.datasets.transforms import build_eval_transforms, build_train_transforms
from src.models.local_global_model import (
    ResNet18GlobalOnlyClassifier,
    ResNet18LocalGlobalClassifier,
)
from src.training.engine import train_one_epoch, validate_one_epoch
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.seed import set_seed


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to load config files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def configure_runtime(training_cfg: dict) -> None:
    max_cpu_threads = int(training_cfg.get("max_cpu_threads", 0) or 0)
    max_interop_threads = int(training_cfg.get("max_interop_threads", 0) or 0)

    if max_cpu_threads > 0:
        torch.set_num_threads(max_cpu_threads)
    if max_interop_threads > 0:
        try:
            torch.set_num_interop_threads(max_interop_threads)
        except RuntimeError:
            pass


def build_model(model_cfg: dict):
    name = model_cfg["name"]
    kwargs = {
        "num_classes": model_cfg["num_classes"],
        "pretrained": model_cfg.get("pretrained", True),
        "dropout": model_cfg.get("dropout", 0.3),
    }
    if name == "resnet18_local_global":
        return ResNet18LocalGlobalClassifier(**kwargs)
    if name == "resnet18_global_only":
        return ResNet18GlobalOnlyClassifier(**kwargs)
    raise ValueError(f"Unsupported model: {name}")


def split_indices(length: int, validation_split: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator).tolist()
    val_size = int(length * validation_split)
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    return train_indices, val_indices


def build_datasets(data_cfg: dict, training_cfg: dict):
    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()

    image_root = resolve_path(data_cfg["image_root"])
    train_csv = resolve_path(data_cfg["train_csv"])
    val_csv = resolve_path(data_cfg.get("val_csv"))
    validation_split = float(data_cfg.get("validation_split", 0.2))

    if val_csv:
        train_dataset = CalligraphyStyleDataset(
            csv_path=train_csv,
            image_root=image_root,
            transform=train_transform,
        )
        val_dataset = CalligraphyStyleDataset(
            csv_path=val_csv,
            image_root=image_root,
            transform=eval_transform,
            class_to_index=train_dataset.class_to_index,
        )
        return train_dataset, val_dataset

    full_train = CalligraphyStyleDataset(
        csv_path=train_csv,
        image_root=image_root,
        transform=train_transform,
    )
    eval_full = CalligraphyStyleDataset(
        csv_path=train_csv,
        image_root=image_root,
        transform=eval_transform,
        class_to_index=full_train.class_to_index,
    )
    train_indices, val_indices = split_indices(
        length=len(full_train),
        validation_split=validation_split,
        seed=int(training_cfg["seed"]),
    )
    return Subset(full_train, train_indices), Subset(eval_full, val_indices)


def extract_class_to_index(dataset):
    if hasattr(dataset, "class_to_index"):
        return dataset.class_to_index
    return dataset.dataset.class_to_index


def make_dataloaders(train_dataset, val_dataset, training_cfg: dict) -> tuple[DataLoader, DataLoader]:
    batch_size = int(training_cfg["batch_size"])
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def metrics_for_logging(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
    }


def optimizer_lrs(optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def dataset_length(dataset) -> int:
    return len(dataset)


def history_paths(logs_dir: Path) -> tuple[Path, Path]:
    return logs_dir / "history.json", logs_dir / "history.csv"


def load_history(logs_dir: Path) -> list[dict]:
    history_json, _ = history_paths(logs_dir)
    if not history_json.exists():
        return []
    with history_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_history(history: list[dict], logs_dir: Path) -> None:
    history_json, history_csv = history_paths(logs_dir)
    with history_json.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    fieldnames = [
        "epoch",
        "stage",
        "lr_backbone",
        "lr_head",
        "train_loss",
        "train_accuracy",
        "train_precision_macro",
        "train_recall_macro",
        "train_f1_macro",
        "val_loss",
        "val_accuracy",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
    ]
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "stage": row["stage"],
                    "lr_backbone": row["learning_rates"][0],
                    "lr_head": row["learning_rates"][1],
                    "train_loss": row["train"]["loss"],
                    "train_accuracy": row["train"]["accuracy"],
                    "train_precision_macro": row["train"]["precision_macro"],
                    "train_recall_macro": row["train"]["recall_macro"],
                    "train_f1_macro": row["train"]["f1_macro"],
                    "val_loss": row["val"]["loss"],
                    "val_accuracy": row["val"]["accuracy"],
                    "val_precision_macro": row["val"]["precision_macro"],
                    "val_recall_macro": row["val"]["recall_macro"],
                    "val_f1_macro": row["val"]["f1_macro"],
                }
            )


def print_startup_summary(
    config: dict,
    device: torch.device,
    train_dataset,
    val_dataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_to_index: dict[str, int],
) -> None:
    print("Training setup")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Model: {config['model']['name']}")
    print(f"  Num classes: {len(class_to_index)}")
    print(f"  Train samples: {dataset_length(train_dataset)}")
    print(f"  Val samples: {dataset_length(val_dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(
        f"  Epochs: stage1={config['training']['epochs_stage1']}, "
        f"stage2={config['training']['epochs_stage2']}"
    )
    print(
        f"  Optimizer: AdamW | backbone_lr={config['optimizer']['backbone_lr']} | "
        f"head_lr={config['optimizer']['head_lr']} | wd={config['optimizer']['weight_decay']}"
    )
    print(
        f"  Scheduler: {config['scheduler']['name']} | "
        f"t_max={config['scheduler']['t_max']}"
    )
    print(
        f"  CPU threads: intra_op={torch.get_num_threads()} | "
        f"interop={torch.get_num_interop_threads()}"
    )
    print(
        f"  Throttle per batch: {config['training'].get('throttle_seconds_per_batch', 0.0)} sec"
    )
    print("")


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    stage_name: str,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    learning_rates: list[float],
    best_f1: float,
) -> None:
    print(
        f"[Epoch {epoch:03d}/{total_epochs:03d}] "
        f"stage={stage_name} | "
        f"lr_backbone={learning_rates[0]:.6g} | "
        f"lr_head={learning_rates[1]:.6g}"
    )
    print(
        f"  train: loss={train_metrics['loss']:.4f} | "
        f"acc={train_metrics['accuracy']:.4f} | "
        f"f1={train_metrics['f1_macro']:.4f} | "
        f"precision={train_metrics['precision_macro']:.4f} | "
        f"recall={train_metrics['recall_macro']:.4f}"
    )
    print(
        f"  val:   loss={val_metrics['loss']:.4f} | "
        f"acc={val_metrics['accuracy']:.4f} | "
        f"f1={val_metrics['f1_macro']:.4f} | "
        f"precision={val_metrics['precision_macro']:.4f} | "
        f"recall={val_metrics['recall_macro']:.4f} | "
        f"best_f1={best_f1:.4f}"
    )
    print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Arabic calligraphy style classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "resnet18_local_global.yaml",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint path. Defaults to latest.pt if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config["training"]["seed"]))
    configure_runtime(config["training"])

    train_dataset, val_dataset = build_datasets(
        data_cfg=config["data"],
        training_cfg=config["training"],
    )
    class_to_index = extract_class_to_index(train_dataset)
    config["model"]["num_classes"] = len(class_to_index)

    train_loader, val_loader = make_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_cfg=config["training"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)
    criterion = build_loss(config["loss"]["name"])
    optimizer = build_optimizer(
        model=model,
        backbone_lr=float(config["optimizer"]["backbone_lr"]),
        head_lr=float(config["optimizer"]["head_lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    total_epochs = int(config["training"]["epochs_stage1"]) + int(config["training"]["epochs_stage2"])
    scheduler = build_scheduler(
        optimizer=optimizer,
        name=config["scheduler"]["name"],
        t_max=max(1, total_epochs),
    )

    output_root = Path(resolve_path(config["paths"]["output_root"]))
    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print_startup_summary(
        config=config,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        class_to_index=class_to_index,
    )

    model.freeze_backbone()
    best_f1 = float("-inf")
    history = load_history(logs_dir)
    start_epoch = 0

    resume_path = args.resume
    if resume_path is None:
        candidate = checkpoints_dir / "latest.pt"
        resume_path = candidate if candidate.exists() else None

    if resume_path is not None and Path(resume_path).exists():
        checkpoint = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        checkpoint_metrics = checkpoint.get("metrics", {})
        best_f1 = float(checkpoint_metrics.get("val", {}).get("f1_macro", best_f1))
        print(f"Resumed from checkpoint: {resume_path}")
        print(f"  Starting at epoch: {start_epoch + 1}")
        print(f"  Best F1 so far: {best_f1:.4f}\n")

    if start_epoch >= int(config["training"]["epochs_stage1"]):
        model.unfreeze_backbone()

    try:
        for epoch in range(start_epoch, total_epochs):
            stage_name = "warmup" if epoch < int(config["training"]["epochs_stage1"]) else "finetune"
            if epoch == int(config["training"]["epochs_stage1"]):
                model.unfreeze_backbone()
                print("Switching to stage 2 fine-tuning: backbone unfrozen.\n")

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch_index=epoch + 1,
                total_epochs=total_epochs,
                throttle_seconds_per_batch=float(
                    config["training"].get("throttle_seconds_per_batch", 0.0)
                ),
            )
            val_metrics = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch_index=epoch + 1,
                total_epochs=total_epochs,
            )
            scheduler.step()
            current_lrs = optimizer_lrs(optimizer)

            epoch_record = {
                "epoch": epoch + 1,
                "stage": stage_name,
                "learning_rates": current_lrs,
                "train": metrics_for_logging(train_metrics),
                "val": metrics_for_logging(val_metrics),
            }
            history.append(epoch_record)
            write_history(history, logs_dir)

            save_checkpoint(
                path=checkpoints_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics=epoch_record,
                config=config,
            )

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = float(val_metrics["f1_macro"])
                save_checkpoint(
                    path=checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    metrics=epoch_record,
                    config=config,
                )

            print_epoch_summary(
                epoch=epoch + 1,
                total_epochs=total_epochs,
                stage_name=stage_name,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rates=current_lrs,
                best_f1=best_f1,
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Resume later from latest.pt.")
    finally:
        write_history(history, logs_dir)
        history_json, history_csv = history_paths(logs_dir)
        print(f"Training history saved to {history_json}")
        print(f"Training history CSV saved to {history_csv}")


if __name__ == "__main__":
    main()

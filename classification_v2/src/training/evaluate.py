from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import ProcessedCalligraphyDataset, build_transforms
from models.efficientnet_b0 import EfficientNetB0Classifier
from training.engine import evaluate_model
from training.imbalance import effective_num_class_weights
from utils.config import load_yaml
from utils.io import ensure_dir, write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "classification_v2" / "configs" / "efficientnet_b0_baseline.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 checkpoint on val/test splits.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint (.pt).")
    p.add_argument("--run-dir", type=Path, default=None, help="Run directory containing checkpoints/best.pt.")
    return p.parse_args(argv)


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.run_dir is not None:
        return args.run_dir / "checkpoints" / "best.pt"
    raise ValueError("Provide either --checkpoint or --run-dir.")


def _write_confusion_matrix_csv(path: Path, cm: list[list[int]], id_to_label: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [id_to_label[i] for i in sorted(id_to_label)]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, row in enumerate(cm):
            w.writerow([labels[i]] + row)


def _id_to_label_from_rows(rows: list[dict[str, str]]) -> dict[int, str]:
    pairs = {}
    for r in rows:
        pairs[int(r["label_id"])] = r["label"]
    return dict(sorted(pairs.items()))


def run_evaluate(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml(args.config)
    ckpt_path = _resolve_checkpoint(args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimization"]
    imb_cfg = cfg["imbalance"]
    tr_cfg = cfg["training"]

    run_dir = ckpt_path.parent.parent
    eval_dir = ensure_dir(run_dir / "evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    batch_size = int(tr_cfg.get("batch_size", 8))
    num_workers = int(tr_cfg.get("num_workers", 0))

    train_ds = ProcessedCalligraphyDataset(data_cfg["train_csv"], transform=build_transforms(train=False))
    val_ds = ProcessedCalligraphyDataset(data_cfg["val_csv"], transform=build_transforms(train=False))
    test_ds = ProcessedCalligraphyDataset(data_cfg["test_csv"], transform=build_transforms(train=False))

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    num_classes = int(model_cfg["num_classes"])
    model = EfficientNetB0Classifier(
        num_classes=num_classes,
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.3)),
    ).to(device)

    beta = float(imb_cfg.get("effective_num_beta", 0.999))
    class_w = effective_num_class_weights(train_ds.labels, num_classes=num_classes, beta=beta).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_w,
        label_smoothing=float(opt_cfg.get("label_smoothing", 0.05)),
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

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

    id_to_label = _id_to_label_from_rows(test_ds.rows)
    _write_confusion_matrix_csv(eval_dir / "val_confusion_matrix.csv", val_eval.confusion_matrix.tolist(), id_to_label)
    _write_confusion_matrix_csv(eval_dir / "test_confusion_matrix.csv", test_eval.confusion_matrix.tolist(), id_to_label)

    summary = {
        "checkpoint": str(ckpt_path),
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
    write_json(eval_dir / "evaluation_summary.json", summary)

    print(f"Evaluation saved under: {eval_dir}")
    print(
        f"Val macro-F1={summary['val']['macro_f1']:.4f}, "
        f"Test macro-F1={summary['test']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    run_evaluate()


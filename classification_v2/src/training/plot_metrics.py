from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training metrics from history/losses CSV files.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory containing history.csv/losses.csv.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots (default: <run-dir>/plots).")
    return p.parse_args(argv)


def run_plot_metrics(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir
    history_path = run_dir / "history.csv"
    losses_path = run_dir / "losses.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")

    out_dir = args.out_dir if args.out_dir is not None else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    history = _read_csv(history_path)
    losses = _read_csv(losses_path) if losses_path.exists() else history

    epochs_h = [int(float(r["epoch"])) for r in history]
    train_f1 = [float(r["train_macro_f1"]) for r in history]
    val_f1 = [float(r["val_macro_f1"]) for r in history]
    train_acc = [float(r["train_acc"]) for r in history]
    val_acc = [float(r["val_acc"]) for r in history]

    epochs_l = [int(float(r["epoch"])) for r in losses]
    train_loss = [float(r["train_loss"]) for r in losses]
    val_loss = [float(r["val_loss"]) for r in losses]
    lr_head = [float(r.get("lr_head", 0.0)) for r in losses]
    lr_backbone = [float(r.get("lr_backbone", 0.0)) for r in losses]

    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_l, train_loss, marker="o", label="train_loss")
    plt.plot(epochs_l, val_loss, marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_png = out_dir / "loss_curves.png"
    plt.savefig(loss_png, dpi=150)
    plt.close()

    # Macro-F1 curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_h, train_f1, marker="o", label="train_macro_f1")
    plt.plot(epochs_h, val_f1, marker="o", label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    f1_png = out_dir / "macro_f1_curves.png"
    plt.savefig(f1_png, dpi=150)
    plt.close()

    # Accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_h, train_acc, marker="o", label="train_acc")
    plt.plot(epochs_h, val_acc, marker="o", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    acc_png = out_dir / "accuracy_curves.png"
    plt.savefig(acc_png, dpi=150)
    plt.close()

    # LR curves (only if available; legacy runs may not contain lr columns)
    lr_png = out_dir / "lr_curves.png"
    if any(v != 0.0 for v in lr_head) or any(v != 0.0 for v in lr_backbone):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_l, lr_head, marker="o", label="lr_head")
        plt.plot(epochs_l, lr_backbone, marker="o", label="lr_backbone")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Curves")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(lr_png, dpi=150)
        plt.close()

    print(f"Saved: {loss_png}")
    print(f"Saved: {f1_png}")
    print(f"Saved: {acc_png}")
    if lr_png.exists():
        print(f"Saved: {lr_png}")
    else:
        print("Skipped lr_curves.png (no LR columns in run logs).")


if __name__ == "__main__":
    run_plot_metrics()

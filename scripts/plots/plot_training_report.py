from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "training" / "resnet18_local_global_grouped_split"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def history_series(history_rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in history_rows]


def history_epochs(history_rows: list[dict[str, str]]) -> list[int]:
    return [int(row["epoch"]) for row in history_rows]


def stage_transition_epoch(history_rows: list[dict[str, str]]) -> int | None:
    for row in history_rows:
        if row["stage"] == "finetune":
            return int(row["epoch"])
    return None


def add_stage_marker(ax, transition_epoch: int | None) -> None:
    if transition_epoch is None:
        return
    ax.axvline(transition_epoch, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        transition_epoch + 0.15,
        0.98,
        "finetune start",
        transform=ax.get_xaxis_transform(),
        va="top",
        ha="left",
        fontsize=9,
        color="gray",
    )


def plot_loss_curves(history_rows: list[dict[str, str]], output_path: Path) -> None:
    epochs = history_epochs(history_rows)
    train_loss = history_series(history_rows, "train_loss")
    val_loss = history_series(history_rows, "val_loss")
    transition_epoch = stage_transition_epoch(history_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, train_loss, label="train_loss", linewidth=2.0)
    axes[0].plot(epochs, val_loss, label="val_loss", linewidth=2.0)
    add_stage_marker(axes[0], transition_epoch)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_loss, label="train_loss", linewidth=2.0)
    axes[1].plot(epochs, val_loss, label="val_loss", linewidth=2.0)
    add_stage_marker(axes[1], transition_epoch)
    axes[1].set_title("Loss Curves (Log Scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(history_rows: list[dict[str, str]], output_path: Path) -> None:
    epochs = history_epochs(history_rows)
    transition_epoch = stage_transition_epoch(history_rows)

    train_acc = history_series(history_rows, "train_accuracy")
    val_acc = history_series(history_rows, "val_accuracy")
    train_f1 = history_series(history_rows, "train_f1_macro")
    val_f1 = history_series(history_rows, "val_f1_macro")
    train_precision = history_series(history_rows, "train_precision_macro")
    val_precision = history_series(history_rows, "val_precision_macro")
    train_recall = history_series(history_rows, "train_recall_macro")
    val_recall = history_series(history_rows, "val_recall_macro")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, train_acc, label="train_acc", linewidth=2.0)
    axes[0].plot(epochs, val_acc, label="val_acc", linewidth=2.0)
    axes[0].plot(epochs, train_f1, label="train_f1", linewidth=2.0)
    axes[0].plot(epochs, val_f1, label="val_f1", linewidth=2.0)
    add_stage_marker(axes[0], transition_epoch)
    axes[0].set_title("Accuracy and Macro F1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_precision, label="train_precision", linewidth=2.0)
    axes[1].plot(epochs, val_precision, label="val_precision", linewidth=2.0)
    axes[1].plot(epochs, train_recall, label="train_recall", linewidth=2.0)
    axes[1].plot(epochs, val_recall, label="val_recall", linewidth=2.0)
    add_stage_marker(axes[1], transition_epoch)
    axes[1].set_title("Macro Precision and Recall")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_learning_rates(history_rows: list[dict[str, str]], output_path: Path) -> None:
    epochs = history_epochs(history_rows)
    lr_backbone = history_series(history_rows, "lr_backbone")
    lr_head = history_series(history_rows, "lr_head")
    transition_epoch = stage_transition_epoch(history_rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(epochs, lr_backbone, label="backbone_lr", linewidth=2.0)
    ax.plot(epochs, lr_head, label="head_lr", linewidth=2.0)
    add_stage_marker(ax, transition_epoch)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(report_rows: list[dict[str, str]], output_path: Path) -> None:
    class_names = [row["class_name"] for row in report_rows]
    precision = [float(row["precision"]) for row in report_rows]
    recall = [float(row["recall"]) for row in report_rows]
    f1 = [float(row["f1"]) for row in report_rows]
    accuracy = [float(row["accuracy"]) for row in report_rows]

    x_positions = list(range(len(class_names)))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([x - 1.5 * width for x in x_positions], precision, width=width, label="precision")
    ax.bar([x - 0.5 * width for x in x_positions], recall, width=width, label="recall")
    ax.bar([x + 0.5 * width for x in x_positions], f1, width=width, label="f1")
    ax.bar([x + 1.5 * width for x in x_positions], accuracy, width=width, label="accuracy")
    ax.set_title("Per-Class Metrics")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_support(report_rows: list[dict[str, str]], output_path: Path) -> None:
    class_names = [row["class_name"] for row in report_rows]
    supports = [int(float(row["support"])) for row in report_rows]

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    bars = ax.bar(class_names, supports, color="#4C78A8")
    ax.set_title("Validation Support Per Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, padding=3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot report-ready training and evaluation figures.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--history-csv", type=Path, default=None)
    parser.add_argument("--per-class-csv", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--eval-subdir", type=str, default="val_best")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    history_csv = args.history_csv or (output_root / "logs" / "history.csv")
    per_class_csv = args.per_class_csv or (output_root / "evaluation" / args.eval_subdir / "per_class_report.csv")
    figures_dir = args.figures_dir or (output_root / "figures")

    history_rows = read_csv_rows(history_csv)
    report_rows = read_csv_rows(per_class_csv)

    plot_loss_curves(history_rows, figures_dir / "loss_curves.png")
    plot_metric_curves(history_rows, figures_dir / "metric_curves.png")
    plot_learning_rates(history_rows, figures_dir / "learning_rates.png")
    plot_per_class_metrics(report_rows, figures_dir / f"per_class_metrics_{args.eval_subdir}.png")
    plot_per_class_support(report_rows, figures_dir / f"per_class_support_{args.eval_subdir}.png")

    print("Saved figures:")
    print(f"  {figures_dir / 'loss_curves.png'}")
    print(f"  {figures_dir / 'metric_curves.png'}")
    print(f"  {figures_dir / 'learning_rates.png'}")
    print(f"  {figures_dir / f'per_class_metrics_{args.eval_subdir}.png'}")
    print(f"  {figures_dir / f'per_class_support_{args.eval_subdir}.png'}")


if __name__ == "__main__":
    main()

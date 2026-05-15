from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_confusion_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"Invalid confusion matrix CSV: {path}")

    labels = rows[0][1:]
    matrix = []
    for r in rows[1:]:
        matrix.append([int(x) for x in r[1:]])
    arr = np.array(matrix, dtype=np.int64)
    return labels, arr


def _plot_confusion(labels: list[str], cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    vmax = max(1, int(cm.max()))
    thresh = vmax / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot confusion matrix PNGs from evaluation CSV files.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory that contains evaluation/*.csv")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <run-dir>/evaluation)")
    return p.parse_args(argv)


def run_plot_confusion(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    eval_dir = args.run_dir / "evaluation"
    val_csv = eval_dir / "val_confusion_matrix.csv"
    test_csv = eval_dir / "test_confusion_matrix.csv"
    if not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Missing confusion CSVs in {eval_dir}. Expected val_confusion_matrix.csv and test_confusion_matrix.csv."
        )

    out_dir = args.out_dir if args.out_dir is not None else eval_dir
    val_labels, val_cm = _read_confusion_csv(val_csv)
    test_labels, test_cm = _read_confusion_csv(test_csv)

    val_png = out_dir / "val_confusion_matrix.png"
    test_png = out_dir / "test_confusion_matrix.png"
    _plot_confusion(val_labels, val_cm, "Validation Confusion Matrix", val_png)
    _plot_confusion(test_labels, test_cm, "Test Confusion Matrix", test_png)

    print(f"Saved: {val_png}")
    print(f"Saved: {test_png}")


if __name__ == "__main__":
    run_plot_confusion()


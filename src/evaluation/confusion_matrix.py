from __future__ import annotations

import csv
from pathlib import Path

import torch


def matrix_to_rows(matrix: torch.Tensor, class_names: list[str]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    matrix = matrix.to(torch.int64).cpu()
    for row_index, class_name in enumerate(class_names):
        row: dict[str, int | str] = {"true_class": class_name}
        for col_index, predicted_name in enumerate(class_names):
            row[predicted_name] = int(matrix[row_index, col_index].item())
        rows.append(row)
    return rows


def write_confusion_matrix_csv(
    path: str | Path,
    matrix: torch.Tensor,
    class_names: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = matrix_to_rows(matrix, class_names)
    fieldnames = ["true_class", *class_names]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix_markdown(
    path: str | Path,
    matrix: torch.Tensor,
    class_names: list[str],
) -> None:
    rows = matrix_to_rows(matrix, class_names)
    lines = [
        "# Confusion Matrix",
        "",
        "| true_class | " + " | ".join(class_names) + " |",
        "| --- | " + " | ".join("---" for _ in class_names) + " |",
    ]
    for row in rows:
        values = [str(row[class_name]) for class_name in class_names]
        lines.append(f"| {row['true_class']} | " + " | ".join(values) + " |")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot_confusion_matrix(
    path: str | Path,
    matrix: torch.Tensor,
    class_names: list[str],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    matrix_np = matrix.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix_np, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix_np.max() / 2.0 if matrix_np.size else 0.0
    for row_index in range(matrix_np.shape[0]):
        for col_index in range(matrix_np.shape[1]):
            ax.text(
                col_index,
                row_index,
                int(matrix_np[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if matrix_np[row_index, col_index] > threshold else "black",
            )

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True

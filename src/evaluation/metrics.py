from __future__ import annotations

import torch


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        matrix[target.long(), prediction.long()] += 1
    return matrix


def classification_metrics_from_confusion(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.to(torch.float64)
    total = matrix.sum().item()
    correct = torch.diag(matrix).sum().item()

    true_positive = torch.diag(matrix)
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive
    support = matrix.sum(dim=1)

    precision_per_class = true_positive / torch.clamp(true_positive + false_positive, min=1.0)
    recall_per_class = true_positive / torch.clamp(true_positive + false_negative, min=1.0)
    f1_per_class = 2 * precision_per_class * recall_per_class / torch.clamp(
        precision_per_class + recall_per_class,
        min=1e-12,
    )
    class_accuracy = true_positive / torch.clamp(support, min=1.0)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "precision_macro": precision_per_class.mean().item(),
        "recall_macro": recall_per_class.mean().item(),
        "f1_macro": f1_per_class.mean().item(),
        "per_class_accuracy": class_accuracy.tolist(),
        "support_per_class": support.tolist(),
    }


def per_class_report_from_confusion(
    matrix: torch.Tensor,
    class_names: list[str],
) -> list[dict[str, float | int | str]]:
    matrix = matrix.to(torch.float64)
    true_positive = torch.diag(matrix)
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive
    support = matrix.sum(dim=1)

    precision_per_class = true_positive / torch.clamp(true_positive + false_positive, min=1.0)
    recall_per_class = true_positive / torch.clamp(true_positive + false_negative, min=1.0)
    f1_per_class = 2 * precision_per_class * recall_per_class / torch.clamp(
        precision_per_class + recall_per_class,
        min=1e-12,
    )
    class_accuracy = true_positive / torch.clamp(support, min=1.0)

    rows: list[dict[str, float | int | str]] = []
    for index, class_name in enumerate(class_names):
        rows.append(
            {
                "class_name": class_name,
                "support": int(support[index].item()),
                "precision": float(precision_per_class[index].item()),
                "recall": float(recall_per_class[index].item()),
                "f1": float(f1_per_class[index].item()),
                "accuracy": float(class_accuracy[index].item()),
                "true_positive": int(true_positive[index].item()),
                "false_positive": int(false_positive[index].item()),
                "false_negative": int(false_negative[index].item()),
            }
        )
    return rows

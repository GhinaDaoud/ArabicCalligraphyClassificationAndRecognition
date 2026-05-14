from __future__ import annotations

import torch


def confusion_matrix_torch(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1_from_confmat(cm: torch.Tensor) -> float:
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    precision = tp / torch.clamp(tp + fp, min=1e-12)
    recall = tp / torch.clamp(tp + fn, min=1e-12)
    f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-12)
    return float(f1.mean().item())


def accuracy_from_confmat(cm: torch.Tensor) -> float:
    total = cm.sum().item()
    if total == 0:
        return 0.0
    correct = torch.diag(cm).sum().item()
    return float(correct / total)


def per_class_recall_from_confmat(cm: torch.Tensor) -> list[float]:
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fn = cm.sum(dim=1) - tp
    recall = tp / torch.clamp(tp + fn, min=1e-12)
    return [float(x.item()) for x in recall]


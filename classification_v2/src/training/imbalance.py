from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


def class_counts_from_labels(labels: list[int]) -> Counter[int]:
    return Counter(labels)


def build_weighted_sampler(
    labels: list[int],
    power: float = 0.5,
) -> WeightedRandomSampler:
    counts = class_counts_from_labels(labels)
    sample_weights = []
    for y in labels:
        c = counts[y]
        w = 1.0 / (float(c) ** power)
        sample_weights.append(w)

    weight_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weight_tensor, num_samples=len(labels), replacement=True)


def effective_num_class_weights(
    labels: list[int],
    num_classes: int,
    beta: float = 0.999,
    normalize_mean_one: bool = True,
) -> torch.Tensor:
    counts = class_counts_from_labels(labels)
    out = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        n = float(counts.get(c, 0))
        if n <= 0:
            out[c] = 0.0
            continue
        effective_num = 1.0 - (beta**n)
        out[c] = (1.0 - beta) / max(effective_num, 1e-12)

    if normalize_mean_one:
        nonzero = out > 0
        if nonzero.any():
            out[nonzero] = out[nonzero] / out[nonzero].mean()
    return out


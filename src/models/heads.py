from __future__ import annotations
import torch
from torch import nn

class LocalProjectionHead(nn.Module):
    """Project mid-level feature maps into a compact local descriptor."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class GlobalPoolingHead(nn.Module):
    """Pool the deepest backbone feature map into a global descriptor."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return torch.flatten(x, 1)


class FusionClassifierHead(nn.Module):
    """Fuse local and global descriptors and classify style labels."""

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(local_dim + global_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([local_features, global_features], dim=1)
        return self.classifier(fused)


class GlobalClassifierHead(nn.Module):
    """Baseline single-branch classifier using only the deepest pooled feature."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(global_features)

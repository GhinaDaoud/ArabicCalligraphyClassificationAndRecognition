from __future__ import annotations
import torch
from torch import nn
from src.models.backbone import ResNet18FeatureBackbone
from src.models.heads import (
    FusionClassifierHead,
    GlobalClassifierHead,
    GlobalPoolingHead,
    LocalProjectionHead,
)

class ResNet18LocalGlobalClassifier(nn.Module):
    """ResNet18 classifier with local-global multi-scale feature fusion."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
        local_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = ResNet18FeatureBackbone(pretrained=pretrained)
        self.local_head = LocalProjectionHead(
            in_channels=self.backbone.layer3_channels,
            out_channels=local_dim,
        )
        self.global_head = GlobalPoolingHead()
        self.classifier = FusionClassifierHead(
            local_dim=local_dim,
            global_dim=self.backbone.layer4_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        local_features = self.local_head(features.layer3)
        global_features = self.global_head(features.layer4)
        return local_features, global_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features, global_features = self.forward_features(x)
        return self.classifier(local_features, global_features)

    def freeze_backbone(self) -> None:
        self.backbone.set_trainable(False)

    def unfreeze_backbone(self) -> None:
        self.backbone.set_trainable(True)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        head_modules = [self.local_head, self.global_head, self.classifier]
        for module in head_modules:
            yield from module.parameters()


class ResNet18GlobalOnlyClassifier(nn.Module):
    """Baseline ResNet18 classifier using only the final pooled global feature."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = ResNet18FeatureBackbone(pretrained=pretrained)
        self.global_head = GlobalPoolingHead()
        self.classifier = GlobalClassifierHead(
            in_dim=self.backbone.layer4_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.global_head(features.layer4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_features = self.forward_features(x)
        return self.classifier(global_features)

    def freeze_backbone(self) -> None:
        self.backbone.set_trainable(False)

    def unfreeze_backbone(self) -> None:
        self.backbone.set_trainable(True)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        head_modules = [self.global_head, self.classifier]
        for module in head_modules:
            yield from module.parameters()

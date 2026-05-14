from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self) -> None:
        for p in self.model.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.model.features.parameters():
            p.requires_grad = True

    def unfreeze_last_feature_blocks(self, n_blocks: int = 2) -> None:
        self.freeze_backbone()
        total = len(self.model.features)
        n_blocks = max(1, min(n_blocks, total))
        for i in range(total - n_blocks, total):
            for p in self.model.features[i].parameters():
                p.requires_grad = True

    def classifier_parameters(self):
        return self.model.classifier.parameters()

    def backbone_parameters(self):
        return self.model.features.parameters()


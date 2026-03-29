from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class BackboneFeatures:
    layer3: torch.Tensor
    layer4: torch.Tensor


class ResNet18FeatureBackbone(nn.Module):
    """Shared ResNet18 backbone that exposes layer3 and layer4 features."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.layer3_channels = 256
        self.layer4_channels = 512

    def forward(self, x: torch.Tensor) -> BackboneFeatures:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        layer3 = self.layer3(x)
        layer4 = self.layer4(layer3)
        return BackboneFeatures(layer3=layer3, layer4=layer4)

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = trainable

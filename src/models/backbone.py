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

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def set_trainable_stage(self, mode: str) -> None:
        mode = mode.strip().lower()
        self.set_trainable(False)

        if mode == "none":
            return
        if mode == "full":
            self.set_trainable(True)
            return
        if mode == "layer4":
            self._set_module_trainable(self.layer4, True)
            return
        if mode in {"layer3_layer4", "layer3+layer4"}:
            self._set_module_trainable(self.layer3, True)
            self._set_module_trainable(self.layer4, True)
            return
        raise ValueError(f"Unsupported fine-tuning mode: {mode}")

    def freeze_batchnorm_stats(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

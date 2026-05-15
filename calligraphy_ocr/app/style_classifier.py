"""
Stage 1 classifier wrapper — ResNet18 local-global fusion architecture from
the ECE 693 progress report.

If the env var RESNET18_CKPT or the file ./style_classifier.pth points to a
trained state_dict, it's loaded. Otherwise the wrapper exposes a clear
'no checkpoint loaded' state instead of fabricating predictions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

CLASSES = ["Naskh", "Ruq'ah", "Diwani", "Thuluth", "Kufic", "Muhaqaq", "Nasta'liq"]


class LocalGlobalFusionResNet18(nn.Module):
    """Reproduces the architecture described in section IV of the progress
    report: ResNet18 backbone, layer3 → 1×1 conv + BN + ReLU + GAP for the
    local branch, layer4 → GAP for the global branch, concat → Linear(256) +
    ReLU + Dropout(0.3) → Linear(num_classes)."""

    def __init__(self, num_classes: int = 7, local_dim: int = 256):
        super().__init__()
        backbone = resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3   # 256 channels
        self.layer4 = backbone.layer4   # 512 channels

        self.local_branch = nn.Sequential(
            nn.Conv2d(256, local_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(local_dim),
            nn.ReLU(inplace=True),
        )

        fused = local_dim + 512
        self.head = nn.Sequential(
            nn.Linear(fused, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)

        local = self.local_branch(f3)
        local = F.adaptive_avg_pool2d(local, 1).flatten(1)
        glob = F.adaptive_avg_pool2d(f4, 1).flatten(1)

        fused = torch.cat([local, glob], dim=1)
        return self.head(fused)


@dataclass
class ClassifierResult:
    style: str
    confidence: float
    probabilities: dict[str, float]
    note: str = ""


class StyleClassifier:
    def __init__(self, ckpt_path: Optional[str] = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = LocalGlobalFusionResNet18(num_classes=len(CLASSES))
        self.loaded = False
        self.ckpt_path = ckpt_path or os.environ.get("RESNET18_CKPT", "")

        if self.ckpt_path and Path(self.ckpt_path).is_file():
            sd = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # Tolerate minor key prefix differences (DataParallel etc.)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            self.loaded = len(missing) == 0
            self.load_note = (
                f"Loaded {self.ckpt_path}"
                if self.loaded
                else f"Loaded with missing keys: {missing[:3]}..."
            )
        else:
            self.load_note = (
                "No checkpoint loaded. Set RESNET18_CKPT env var or place "
                "style_classifier.pth alongside app.py."
            )

        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> ClassifierResult:
        if not self.loaded:
            return ClassifierResult(
                style="—",
                confidence=0.0,
                probabilities={c: 0.0 for c in CLASSES},
                note=self.load_note,
            )
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        idx = int(torch.tensor(probs).argmax().item())
        return ClassifierResult(
            style=CLASSES[idx],
            confidence=float(probs[idx]),
            probabilities={c: float(p) for c, p in zip(CLASSES, probs)},
            note="",
        )

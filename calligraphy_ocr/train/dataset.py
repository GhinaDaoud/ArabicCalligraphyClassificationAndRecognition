"""
Paired calligraphy dataset for QLoRA fine-tuning of AIN-7B / Qwen-VL family.

Reads the already-paired transcripts from HICMA Sets 1-3 and DuwaBench
labels.csv (same source files the eval set uses). Each row is (image_path,
style, transcript). Builds the chat-template input format Qwen2-VL expects
and supplies a style-conditioned prompt prefix.

Splits:
    - hold out the 49 eval images (anything in eval_gt.csv) entirely
    - 90/10 train/val on what remains, stratified by style
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../693
DATASET_ROOT = REPO_ROOT / "dataset"
EVAL_GT_CSV = REPO_ROOT / "calligraphy_ocr" / "eval_gt.csv"

CLASS_ALIASES = {
    "Muhaquaq": "Muhaqaq",
    "Nastaliq": "Nasta'liq",
    "Ruqah": "Ruq'ah",
    "Ruqaah": "Ruq'ah",
    "Ruqa": "Ruq'ah",
    "Ruqʿah": "Ruq'ah",
    "Ruq‘ah": "Ruq'ah",
}

STYLES = ["Naskh", "Ruq'ah", "Diwani", "Thuluth", "Kufic", "Muhaqaq", "Nasta'liq"]


def _norm_class(c: object) -> str:
    s = str(c).strip().strip('"').strip("'")
    return CLASS_ALIASES.get(s, s)


def _flatten_label(x: object) -> str:
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple)):
                return " ".join(str(s) for s in parsed)
        except (ValueError, SyntaxError):
            pass
    return str(x)


def load_paired_corpus() -> pd.DataFrame:
    """Returns a DataFrame with columns: filename, style, transcript, abs_path, source."""
    rows = []
    for setn in (1, 2, 3):
        csv = DATASET_ROOT / f"HICMA/Set{setn}/labels.csv"
        df = pd.read_csv(csv)
        df["source"] = f"HICMA-Set{setn}"
        df["abs_path"] = df["img_name"].apply(
            lambda n, s=setn: str(DATASET_ROOT / f"HICMA/Set{s}/images" / n)
        )
        rows.append(df[["img_name", "class", "label", "source", "abs_path"]])

    duwa = pd.read_csv(DATASET_ROOT / "DuwaBench/labels.csv")
    duwa["source"] = "DuwaBench"
    duwa["abs_path"] = duwa["img_name"].apply(
        lambda n: str(DATASET_ROOT / "DuwaBench/images" / n)
    )
    duwa["label"] = duwa["label"].apply(_flatten_label)
    rows.append(duwa[["img_name", "class", "label", "source", "abs_path"]])

    full = pd.concat(rows, ignore_index=True)
    full["style"] = full["class"].apply(_norm_class)
    full = full.rename(columns={"img_name": "filename", "label": "transcript"})
    full["transcript"] = full["transcript"].astype(str).str.strip()

    # Drop rows with empty transcripts, missing image files, or unknown styles.
    full = full[full["transcript"].str.len() > 0]
    full = full[full["style"].isin(STYLES)]
    full = full[full["abs_path"].apply(lambda p: Path(p).is_file())].reset_index(drop=True)
    return full[["filename", "style", "transcript", "abs_path", "source"]]


def build_train_val_split(
    seed: int = 42,
    val_frac: float = 0.10,
    hold_out_eval: bool = True,
    max_per_style: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split. If hold_out_eval, excludes filenames present in
    eval_gt.csv so we never train on what we benchmarked."""
    full = load_paired_corpus()

    if hold_out_eval and EVAL_GT_CSV.is_file():
        eval_df = pd.read_csv(EVAL_GT_CSV)
        held = set(eval_df["filename"].astype(str).tolist())
        full = full[~full["filename"].isin(held)].reset_index(drop=True)

    train_rows, val_rows = [], []
    for style in STYLES:
        sub = full[full["style"] == style]
        if max_per_style is not None and len(sub) > max_per_style:
            sub = sub.sample(n=max_per_style, random_state=seed)
        sub = sub.sample(frac=1.0, random_state=seed)  # shuffle
        n_val = max(1, int(round(len(sub) * val_frac)))
        val_rows.append(sub.iloc[:n_val])
        train_rows.append(sub.iloc[n_val:])
    train = pd.concat(train_rows, ignore_index=True).sample(frac=1.0, random_state=seed)
    val = pd.concat(val_rows, ignore_index=True).sample(frac=1.0, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True)


# ── Prompt templating ──────────────────────────────────────────────────────

# We always include a style-conditioned hint at training time. At inference
# time the prompt is built the same way using the predicted style from Stage 1.
def build_user_prompt(style: Optional[str]) -> str:
    base = (
        "Transcribe the Arabic text in this image exactly as written. "
        "Output only the Arabic characters as plain text. "
        "No HTML, no markdown, no commentary, no translation."
    )
    if style and style in STYLES:
        return f"This image contains {style} Arabic calligraphy. " + base
    return base


# ── Augmentation (calligraphy-safe) ─────────────────────────────────────────
#
# Design notes:
# - Rotation limited to ±8° (vs ±12° in the Stage-1 augmentation pipeline).
#   OCR is more rotation-sensitive than classification — large rotations can
#   change which character a stroke looks like.
# - NO elastic deformation. Used in Stage 1 because it preserved style cues
#   while perturbing stroke shapes. For OCR it's risky: it can morph one
#   letter into the visual shape of another, creating noisy labels.
# - Light brightness/contrast jitter (±0.15) — robustness to scan quality.
# - Mild gaussian noise (var 5..25) — same.
# - Mild gaussian blur — same.
# - All transforms applied independently with low individual probabilities so
#   the cumulative effect per sample is moderate.

def build_ocr_safe_augmentation(intensity: str = "light"):
    """Returns an albumentations Compose. Apply to numpy uint8 RGB images.

    intensity='light': used for majority classes (Naskh, Thuluth, Diwani) —
        ±8° rotation, mild brightness/contrast, occasional blur or noise.
    intensity='heavy': used for minority classes (Nasta'liq, Muhaqaq, Ruq'ah,
        Kufic) — stronger to break exact-image memorization on tiny pools.
        Adds perspective transform, larger rotation, brightness range,
        and random rectangular erasing.
    """
    import albumentations as A
    import cv2

    if intensity == "heavy":
        return A.Compose([
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, fill=255, p=0.8),
            A.Perspective(scale=(0.02, 0.05), fit_output=False, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                       contrast_limit=0.25, p=0.7),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.6),
            A.CoarseDropout(num_holes_range=(1, 3),
                            hole_height_range=(0.05, 0.10),
                            hole_width_range=(0.05, 0.10),
                            fill=255, p=0.3),
        ])
    return A.Compose([
        A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, fill=255, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
        ], p=0.4),
    ])


MINORITY_STYLES = {"Nasta'liq", "Muhaqaq", "Ruq'ah", "Kufic"}


# ── Torch Dataset ──────────────────────────────────────────────────────────

@dataclass
class _Row:
    filename: str
    style: str
    transcript: str
    abs_path: str
    source: str


class CalligraphyOCRDataset(Dataset):
    """Yields dicts: { 'messages': [...], 'image': PIL.Image, 'answer': str, 'style': str }.

    The collator (TRL or custom) handles tokenization + applying the chat
    template. We keep the Dataset format generic so we can also drive HF Trainer
    or a custom training loop."""

    def __init__(
        self,
        df: pd.DataFrame,
        style_dropout: float = 0.0,
        load_images: bool = True,
        augment: bool = False,
    ):
        """
        style_dropout: probability of OMITTING the style hint per sample, so
            the model also learns the unconditional baseline.
        load_images: if False, returns image paths (cheap; useful for testing).
        augment: if True, apply on-the-fly calligraphy-safe augmentation.
            Recommended True for train, False for val.
        """
        self.rows = [_Row(**{k: r[k] for k in ("filename", "style", "transcript", "abs_path", "source")})
                     for _, r in df.iterrows()]
        self.style_dropout = style_dropout
        self.load_images = load_images
        self.augment = augment
        self._aug_light = build_ocr_safe_augmentation("light") if augment else None
        self._aug_heavy = build_ocr_safe_augmentation("heavy") if augment else None

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def styles(self) -> list[str]:
        """Per-row style list, used by samplers."""
        return [r.style for r in self.rows]

    def __getitem__(self, idx: int) -> dict:
        import random

        r = self.rows[idx]
        use_style = random.random() >= self.style_dropout
        style_for_prompt = r.style if use_style else None
        user_prompt = build_user_prompt(style_for_prompt)

        # Load + (optionally) augment.
        if self.load_images:
            image = Image.open(r.abs_path).convert("RGB")
            if self.augment:
                pipe = self._aug_heavy if r.style in MINORITY_STYLES else self._aug_light
                arr = np.array(image)
                arr = pipe(image=arr)["image"]
                image = Image.fromarray(arr)
            image_for_message = image
        else:
            image = r.abs_path
            image_for_message = r.abs_path

        messages = [
            {"role": "user", "content": [
                # When augmenting, pass the augmented PIL.Image directly so
                # qwen_vl_utils.process_vision_info uses our processed pixels
                # instead of re-reading the original file.
                {"type": "image", "image": image_for_message},
                {"type": "text", "text": user_prompt},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": r.transcript},
            ]},
        ]

        return {
            "messages": messages,
            "image": image,
            "answer": r.transcript,
            "style": r.style,
            "filename": r.filename,
        }


def make_style_balanced_sampler(dataset: "CalligraphyOCRDataset",
                                mode: str = "sqrt"):
    """WeightedRandomSampler with selectable rebalancing strength.

    mode='uniform': each style drawn with equal probability (aggressive
        rebalance; oversamples minorities ~count_max/count_minor times per
        epoch — caused Nasta'liq memorisation in the first training run).
    mode='sqrt': weight per sample ∝ 1/sqrt(class_count). Softer rebalance —
        minority classes still get more exposure than under natural sampling
        but without the extreme oversampling that caused overfitting on
        small classes (Nasta'liq=51, Muhaqaq=9).
    mode='natural': no rebalancing (proportional to raw counts).
    """
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler
    import math
    import torch

    styles = dataset.styles
    counts = Counter(styles)

    if mode == "uniform":
        n_classes = len(counts)
        weights = [1.0 / (n_classes * counts[s]) for s in styles]
    elif mode == "sqrt":
        weights = [1.0 / math.sqrt(counts[s]) for s in styles]
    elif mode == "natural":
        weights = [1.0 for _ in styles]
    else:
        raise ValueError(f"unknown sampler mode: {mode!r}")

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )


if __name__ == "__main__":
    # Smoke print.
    train, val = build_train_val_split()
    print(f"train rows: {len(train)} | val rows: {len(val)}")
    print("\nPer-style counts:")
    for style in STYLES:
        n_train = (train["style"] == style).sum()
        n_val = (val["style"] == style).sum()
        print(f"  {style:<11s}  train={n_train:>5d}  val={n_val:>4d}")

    ds = CalligraphyOCRDataset(train.head(3))
    print(f"\nfirst sample keys: {list(ds[0].keys())}")
    print(f"first messages[0]['content'][1]['text']: {ds[0]['messages'][0]['content'][1]['text'][:120]}")
    print(f"first answer: {ds[0]['answer'][:120]}")

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from data.dataset import build_transforms
from models.efficientnet_b0 import EfficientNetB0Classifier
from utils.config import load_yaml


def _id_to_label(csv_path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[int(row["label_id"])] = row["label"]
    return dict(sorted(mapping.items()))


def _tight_crop_bbox(rgb_img: Image.Image, ink_threshold: int = 245, min_fg_ratio: float = 0.001, margin_ratio: float = 0.02):
    gray = np.array(rgb_img.convert("L"))
    h, w = gray.shape
    mask = gray < ink_threshold
    if float(mask.mean()) < min_fg_ratio:
        return (0, 0, w, h)
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    mx = max(2, int(round((x1 - x0) * margin_ratio)))
    my = max(2, int(round((y1 - y0) * margin_ratio)))
    return (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))


def _resize_pad(rgb_img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    src_w, src_h = rgb_img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), color=(255, 255, 255))
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict a single image using a trained EfficientNet-B0 checkpoint.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--raw-image", action="store_true", help="Apply crop+resize+pad before inference.")
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--save-json", type=Path, default=None)
    return p.parse_args(argv)


def run_predict_single(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    label_map = _id_to_label(Path(data_cfg["train_csv"]))
    num_classes = int(model_cfg["num_classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0Classifier(
        num_classes=num_classes,
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.3)),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image_arg = str(args.image)
    if "<path_to_image>" in image_arg or "path_to_image" == Path(image_arg).name.lower():
        raise ValueError("Replace <path_to_image> with a real image path.")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    with Image.open(args.image) as im:
        rgb = im.convert("RGB")

    if args.raw_image:
        bbox = _tight_crop_bbox(rgb)
        rgb = rgb.crop(bbox)
        rgb = _resize_pad(rgb, target_h=args.height, target_w=args.width)

    tfm = build_transforms(train=False)
    x = tfm(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    topk = max(1, min(args.topk, num_classes))
    vals, inds = torch.topk(probs, k=topk)
    preds = []
    for p, i in zip(vals.tolist(), inds.tolist()):
        preds.append(
            {
                "label_id": int(i),
                "label": label_map.get(int(i), str(i)),
                "probability": float(p),
            }
        )

    result = {
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "topk": preds,
    }
    print(json.dumps(result, indent=2))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    run_predict_single()

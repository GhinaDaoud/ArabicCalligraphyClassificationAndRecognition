from __future__ import annotations

import argparse
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "processed" / "augmented_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "augmented"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize images with aspect-ratio preservation and white padding."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Input dataset root containing images/ and labels.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output dataset root that will contain resized images/ and labels.csv.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Final canvas width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Final canvas height.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker threads for image processing.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality if output images are saved as JPEG.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def validate_dataset_root(root: Path) -> tuple[Path, Path]:
    images_dir = root / "images"
    labels_csv = root / "labels.csv"
    if not images_dir.exists():
        raise SystemExit(f"Missing images directory: {images_dir}")
    if not labels_csv.exists():
        raise SystemExit(f"Missing labels file: {labels_csv}")
    return images_dir, labels_csv


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit("labels.csv is empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def resize_and_pad(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    offset_x = (width - new_w) // 2
    offset_y = (height - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def process_one_image(
    image_path: Path,
    output_path: Path,
    width: int,
    height: int,
    jpeg_quality: int,
) -> tuple[str, bool]:
    try:
        with Image.open(image_path) as image:
            processed = resize_and_pad(image, width=width, height=height)
            save_kwargs = {"optimize": True}
            if output_path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = jpeg_quality
            processed.save(output_path, **save_kwargs)
        return image_path.name, True
    except Exception:
        return image_path.name, False


def main() -> None:
    args = parse_args()
    input_images_dir, input_labels_csv = validate_dataset_root(args.input_root)

    output_root = args.output_root
    output_images_dir = output_root / "images"
    output_labels_csv = output_root / "labels.csv"

    if output_root.exists():
        if not args.force:
            raise SystemExit(
                f"Output directory already exists: {output_root}\n"
                "Use --force to overwrite it."
            )
        clear_directory(output_root)

    ensure_dir(output_images_dir)

    labels = read_labels(input_labels_csv)
    image_paths = list_images(input_images_dir)
    expected_names = {row["img_name"] for row in labels if row.get("img_name")}
    available_names = {path.name for path in image_paths}
    missing = sorted(expected_names - available_names)
    if missing:
        raise SystemExit(
            f"labels.csv references {len(missing)} missing image files. "
            f"First missing file: {missing[0]}"
        )

    tasks = [
        (
            image_path,
            output_images_dir / image_path.name,
            args.width,
            args.height,
            args.jpeg_quality,
        )
        for image_path in image_paths
        if image_path.name in expected_names
    ]

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for name, ok in executor.map(lambda item: process_one_image(*item), tasks):
            if not ok:
                failures.append(name)

    if failures:
        raise SystemExit(
            f"Failed to process {len(failures)} image(s). First failed file: {failures[0]}"
        )

    write_labels(output_labels_csv, labels)

    print(f"Wrote resized dataset to: {output_root}")
    print(f"Images: {len(tasks)}")
    print(f"Canvas size: {args.width}x{args.height}")
    print(f"Workers: {args.workers}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "raw" / "dataset"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "augmented_v1"
DEFAULT_METADATA_DIR = REPO_ROOT / "data" / "metadata"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SELECTED_DATASETS = ("AC_data", "DuwaBench", "HICMA", "Rufa")


@dataclass(frozen=True)
class Sample:
    source_dataset: str
    subset: str
    image_path: Path
    raw_class: str
    text_label: str
    label_source: str


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def discover_image_sources() -> list[tuple[str, str, Path, Path | None, str]]:
    sources: list[tuple[str, str, Path, Path | None, str]] = []
    seen: set[Path] = set()

    for dataset_name in SELECTED_DATASETS:
        root = DATASET_ROOT / dataset_name
        if not root.exists():
            continue

        for subdir in root.rglob("*"):
            if not subdir.is_dir() or subdir in seen:
                continue

            image_files = list_image_files(subdir)
            if not image_files:
                continue

            seen.add(subdir)
            labels_csv = subdir / "labels.csv"
            if not labels_csv.exists():
                parent_csv = subdir.parent / "labels.csv"
                labels_csv = parent_csv if parent_csv.exists() else None

            rel = subdir.relative_to(root)
            parts = [part for part in rel.parts if part != "images"]
            subset = "/".join(parts) if parts else dataset_name

            if labels_csv is None:
                sources.append((dataset_name, subset, subdir, None, "folder"))
            else:
                sources.append((dataset_name, subset, subdir, labels_csv, "csv"))

    return sorted(sources, key=lambda item: (item[0], item[1], str(item[2])))


def collect_samples(skip_mixed: bool = True) -> tuple[list[Sample], dict[str, int]]:
    samples: list[Sample] = []
    skipped = {"mixed_labels": 0, "orphan_images": 0}

    for dataset_name, subset, image_dir, labels_csv, label_source in discover_image_sources():
        image_files = list_image_files(image_dir)

        if labels_csv is None:
            folder_class = image_dir.name.strip()
            if skip_mixed and "," in folder_class:
                skipped["mixed_labels"] += len(image_files)
                continue
            for image_path in image_files:
                samples.append(
                    Sample(
                        source_dataset=dataset_name,
                        subset=subset,
                        image_path=image_path,
                        raw_class=folder_class,
                        text_label="",
                        label_source="folder",
                    )
                )
            continue

        rows = read_csv_rows(labels_csv)
        label_lookup = {row["img_name"]: row for row in rows if row.get("img_name")}
        for image_path in image_files:
            row = label_lookup.get(image_path.name)
            if row is None:
                skipped["orphan_images"] += 1
                continue

            raw_class = row.get("class", "").strip()
            if not raw_class:
                continue
            if skip_mixed and "," in raw_class:
                skipped["mixed_labels"] += 1
                continue

            samples.append(
                Sample(
                    source_dataset=dataset_name,
                    subset=subset,
                    image_path=image_path,
                    raw_class=raw_class,
                    text_label=row.get("label", "").strip(),
                    label_source="csv",
                )
            )

    return samples, skipped


def class_targets(class_counts: Counter[str], target_ratio: float) -> dict[str, int]:
    if not class_counts:
        return {}
    max_count = max(class_counts.values())
    target = max(1, int(round(max_count * target_ratio)))
    return {class_name: max(count, target) for class_name, count in class_counts.items()}


def build_augmentation_plan(
    samples: list[Sample],
    target_ratio: float,
    max_aug_per_image: int,
) -> tuple[dict[Path, int], list[dict[str, object]], dict[str, int]]:
    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.raw_class].append(sample)

    class_counts = Counter({class_name: len(items) for class_name, items in by_class.items()})
    targets = class_targets(class_counts, target_ratio=target_ratio)

    plan: dict[Path, int] = {}
    plan_rows: list[dict[str, object]] = []

    for class_name, class_samples in sorted(by_class.items()):
        current = len(class_samples)
        target = targets[class_name]
        needed = max(0, target - current)
        per_image = 0 if needed == 0 else min(max_aug_per_image, math.ceil(needed / current))

        remaining = needed
        for index, sample in enumerate(sorted(class_samples, key=lambda item: str(item.image_path))):
            if remaining <= 0 or per_image == 0:
                copies = 0
            else:
                copies = min(per_image, remaining)
            plan[sample.image_path] = copies
            remaining -= copies

        planned_total = sum(plan[sample.image_path] for sample in class_samples)
        plan_rows.append(
            {
                "raw_class": class_name,
                "original_count": current,
                "target_count": target,
                "planned_augmented_copies": planned_total,
                "projected_total": current + planned_total,
                "per_image_cap": max_aug_per_image,
            }
        )

    return plan, plan_rows, dict(class_counts)


def random_transform(image: Image.Image, rng: random.Random) -> Image.Image:
    # Keep the pipeline conservative so style geometry is not damaged.
    image = image.convert("RGB")

    if rng.random() < 0.8:
        angle = rng.uniform(-6.0, 6.0)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))

    if rng.random() < 0.7:
        width, height = image.size
        shift_x = int(rng.uniform(-0.02, 0.02) * width)
        shift_y = int(rng.uniform(-0.02, 0.02) * height)
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1.0, 0.0, shift_x, 0.0, 1.0, shift_y),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )

    if rng.random() < 0.7:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.12))

    if rng.random() < 0.6:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.94, 1.08))

    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))

    if rng.random() < 0.15:
        image = image.filter(ImageFilter.SHARPEN)

    return image


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metadata_rows(
    samples: list[Sample],
    augment_plan: dict[Path, int],
    images_dir: Path,
    rng_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    class_counters: dict[str, int] = defaultdict(int)
    aug_counters: dict[str, int] = defaultdict(int)

    for sample in sorted(samples, key=lambda item: (item.raw_class, str(item.image_path))):
        original_name = f"{sample.raw_class}_{class_counters[sample.raw_class]:06d}.png"
        class_counters[sample.raw_class] += 1

        destination = images_dir / original_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, destination)

        rows.append(
            {
                "img_name": original_name,
                "class": sample.raw_class,
                "label": sample.text_label,
                "source_dataset": sample.source_dataset,
                "source_subset": sample.subset,
                "source_path": str(sample.image_path.relative_to(REPO_ROOT)),
                "is_augmented": False,
                "augmentation_index": 0,
            }
        )

        copies = augment_plan.get(sample.image_path, 0)
        if copies <= 0:
            continue

        with Image.open(sample.image_path) as image:
            for copy_index in range(1, copies + 1):
                aug_name = f"{sample.raw_class}_9{aug_counters[sample.raw_class]:05d}.png"
                aug_counters[sample.raw_class] += 1
                aug_image = random_transform(image, random.Random(rng_seed + aug_counters[sample.raw_class]))
                aug_image.save(images_dir / aug_name, format="PNG", optimize=True)

                rows.append(
                    {
                        "img_name": aug_name,
                        "class": sample.raw_class,
                        "label": sample.text_label,
                        "source_dataset": sample.source_dataset,
                        "source_subset": sample.subset,
                        "source_path": str(sample.image_path.relative_to(REPO_ROOT)),
                        "is_augmented": True,
                        "augmentation_index": copy_index,
                    }
                )

    return rows


def build_summary_markdown(
    plan_rows: list[dict[str, object]],
    class_counts: dict[str, int],
    skipped: dict[str, int],
    args: argparse.Namespace,
) -> str:
    total_original = sum(class_counts.values())
    total_augmented = sum(int(row["planned_augmented_copies"]) for row in plan_rows)
    total_final = total_original + total_augmented

    lines = [
        "# Augmentation Plan",
        "",
        f"- target_ratio: {args.target_ratio}",
        f"- max_aug_per_image: {args.max_aug_per_image}",
        f"- skip_mixed: {not args.keep_mixed}",
        f"- random_seed: {args.seed}",
        f"- originals: {total_original}",
        f"- planned_augmented_copies: {total_augmented}",
        f"- projected_total: {total_final}",
        f"- skipped_mixed_labels: {skipped['mixed_labels']}",
        f"- skipped_orphan_images: {skipped['orphan_images']}",
        "",
        "## Per-Class Plan",
        "",
        "| raw_class | original_count | target_count | planned_augmented_copies | projected_total | per_image_cap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in plan_rows:
        lines.append(
            f"| {row['raw_class']} | {row['original_count']} | {row['target_count']} | "
            f"{row['planned_augmented_copies']} | {row['projected_total']} | {row['per_image_cap']} |"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a conservative offline augmented dataset.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--target-ratio", type=float, default=0.25)
    parser.add_argument("--max-aug-per-image", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-mixed", action="store_true", help="Keep mixed-label rows instead of skipping them.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and save the plan without writing images.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_dir = ensure_dir(args.metadata_dir)

    samples, skipped = collect_samples(skip_mixed=not args.keep_mixed)
    augment_plan, plan_rows, class_counts = build_augmentation_plan(
        samples=samples,
        target_ratio=args.target_ratio,
        max_aug_per_image=args.max_aug_per_image,
    )

    plan_csv = metadata_dir / "augmentation_plan_v1.csv"
    summary_md = metadata_dir / "augmentation_plan_v1.md"
    write_csv(
        plan_csv,
        plan_rows,
        ["raw_class", "original_count", "target_count", "planned_augmented_copies", "projected_total", "per_image_cap"],
    )
    summary_md.write_text(build_summary_markdown(plan_rows, class_counts, skipped, args), encoding="utf-8")
    print(f"Wrote {plan_csv}")
    print(f"Wrote {summary_md}")

    if args.dry_run:
        print("Dry run only. No images were written.")
        return

    output_root = args.output_root
    images_dir = output_root / "images"
    labels_csv = output_root / "labels.csv"

    if output_root.exists():
        if not args.force:
            raise SystemExit(
                f"Output directory already exists: {output_root}\n"
                "Use --force after deleting it yourself or point to a new output path."
            )
        shutil.rmtree(output_root)

    ensure_dir(images_dir)
    rows = build_metadata_rows(samples, augment_plan, images_dir=images_dir, rng_seed=args.seed)
    write_csv(
        labels_csv,
        rows,
        ["img_name", "class", "label", "source_dataset", "source_subset", "source_path", "is_augmented", "augmentation_index"],
    )
    print(f"Wrote {labels_csv}")
    print(f"Wrote augmented images under {images_dir}")


if __name__ == "__main__":
    main()

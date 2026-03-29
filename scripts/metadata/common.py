from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "raw" / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "metadata"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageSource:
    dataset: str
    subset: str
    image_dir: Path
    labels_csv: Path | None
    label_source: str
    folder_label: str | None = None


def selected_dataset_names(include_misc: bool = False) -> list[str]:
    datasets = ["AC_data", "DuwaBench", "HICMA", "Rufa"]
    if include_misc:
        datasets.append("Misc")
    return datasets


def dataset_roots(include_misc: bool = False) -> list[Path]:
    return [DATASET_ROOT / name for name in selected_dataset_names(include_misc=include_misc)]


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def list_image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def relative_subset(dataset_root: Path, image_dir: Path) -> str:
    rel = image_dir.relative_to(dataset_root)
    parts = [part for part in rel.parts if part != "images"]
    return "/".join(parts) if parts else dataset_root.name


def discover_sources(include_misc: bool = False) -> list[ImageSource]:
    sources: list[ImageSource] = []
    seen: set[Path] = set()

    for root in dataset_roots(include_misc=include_misc):
        if not root.exists():
            continue

        for subdir in root.rglob("*"):
            if not subdir.is_dir() or subdir in seen:
                continue

            image_files = list_image_files(subdir)
            if not image_files:
                continue

            seen.add(subdir)
            labels_path = subdir / "labels.csv"
            if not labels_path.exists():
                parent_labels = subdir.parent / "labels.csv"
                labels_path = parent_labels if parent_labels.exists() else None

            subset = relative_subset(root, subdir)
            if labels_path is not None:
                sources.append(
                    ImageSource(
                        dataset=root.name,
                        subset=subset,
                        image_dir=subdir,
                        labels_csv=labels_path,
                        label_source="csv",
                    )
                )
            else:
                sources.append(
                    ImageSource(
                        dataset=root.name,
                        subset=subset,
                        image_dir=subdir,
                        labels_csv=None,
                        label_source="folder",
                        folder_label=subdir.name,
                    )
                )

    return sorted(sources, key=lambda item: (item.dataset, item.subset, str(item.image_dir)))


def load_labels_csv(labels_csv: Path) -> list[dict[str, str]]:
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]

    if not rows:
        return []

    sample_keys = rows[0].keys()
    if "img_name" not in sample_keys or "class" not in sample_keys:
        raise ValueError(f"Missing required columns in {labels_csv}")
    return rows


def csv_label_lookup(labels_csv: Path) -> dict[str, str]:
    rows = load_labels_csv(labels_csv)
    return {
        row["img_name"]: row["class"]
        for row in rows
        if row["img_name"]
    }


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return None
    return width, height


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows: Iterable[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])

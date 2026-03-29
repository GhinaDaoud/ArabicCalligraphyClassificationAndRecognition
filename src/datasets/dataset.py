from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    img_name: str
    class_name: str
    image_path: Path
    text_label: str
    source_dataset: str
    source_subset: str
    source_path: str
    is_augmented: bool
    augmentation_index: int


class CalligraphyStyleDataset(Dataset):
    """Image classification dataset backed by a labels CSV and image directory."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        transform: Callable | None = None,
        class_to_index: dict[str, int] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform

        self.samples = self._load_samples()
        self.class_names = sorted({sample.class_name for sample in self.samples})
        self.class_to_index = class_to_index or {
            class_name: index for index, class_name in enumerate(self.class_names)
        }
        self.index_to_class = {
            index: class_name for class_name, index in self.class_to_index.items()
        }

    def _load_samples(self) -> list[SampleRecord]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing labels CSV: {self.csv_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_root}")

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        samples: list[SampleRecord] = []
        for row in rows:
            img_name = (row.get("img_name") or "").strip()
            class_name = (row.get("class") or "").strip()
            if not img_name or not class_name:
                continue

            image_path = self.image_root / img_name
            if not image_path.exists():
                raise FileNotFoundError(f"Image referenced in CSV was not found: {image_path}")

            samples.append(
                SampleRecord(
                    img_name=img_name,
                    class_name=class_name,
                    image_path=image_path,
                    text_label=(row.get("label") or "").strip(),
                    source_dataset=(row.get("source_dataset") or "").strip(),
                    source_subset=(row.get("source_subset") or "").strip(),
                    source_path=(row.get("source_path") or "").strip(),
                    is_augmented=str(row.get("is_augmented", "")).strip().lower() == "true",
                    augmentation_index=int((row.get("augmentation_index") or "0").strip()),
                )
            )

        if not samples:
            raise ValueError(f"No valid samples were found in {self.csv_path}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        target = self.class_to_index[sample.class_name]
        return image, target

    def get_class_distribution(self) -> dict[str, int]:
        counts = {class_name: 0 for class_name in self.class_to_index}
        for sample in self.samples:
            counts[sample.class_name] += 1
        return counts

    def get_sample_metadata(self, index: int) -> SampleRecord:
        return self.samples[index]

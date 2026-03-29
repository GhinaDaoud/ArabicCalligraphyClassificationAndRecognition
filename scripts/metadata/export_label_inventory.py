from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    csv_label_lookup,
    discover_sources,
    ensure_output_dir,
    list_image_files,
    write_csv,
    write_markdown_table,
)


def build_label_inventory(include_misc: bool = False) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    label_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    for source in discover_sources(include_misc=include_misc):
        image_files = list_image_files(source.image_dir)
        matched_images = 0
        orphan_images = 0
        class_counts: Counter[str] = Counter()

        if source.labels_csv is not None:
            label_lookup = csv_label_lookup(source.labels_csv)
            for image_path in image_files:
                raw_class = label_lookup.get(image_path.name)
                if raw_class is None:
                    orphan_images += 1
                    continue
                class_counts[raw_class] += 1
                matched_images += 1
        else:
            raw_class = source.folder_label or source.subset
            class_counts[raw_class] = len(image_files)
            matched_images = len(image_files)

        source_rows.append(
            {
                "dataset": source.dataset,
                "subset": source.subset,
                "label_source": source.label_source,
                "image_dir": str(source.image_dir.relative_to(REPO_ROOT)),
                "labels_csv": (
                    str(source.labels_csv.relative_to(REPO_ROOT))
                    if source.labels_csv is not None
                    else ""
                ),
                "matched_images": matched_images,
                "orphan_images": orphan_images,
                "unique_raw_classes": len(class_counts),
            }
        )

        for raw_class, count in sorted(class_counts.items()):
            label_rows.append(
                {
                    "dataset": source.dataset,
                    "subset": source.subset,
                    "label_source": source.label_source,
                    "raw_class": raw_class,
                    "image_count": count,
                    "is_mixed_label": "," in raw_class,
                }
            )

    labels = sorted(label_rows, key=lambda row: (str(row["dataset"]), str(row["subset"]), str(row["raw_class"])))
    sources = sorted(source_rows, key=lambda row: (str(row["dataset"]), str(row["subset"])))
    return labels, sources


def build_markdown(labels: list[dict[str, object]], sources: list[dict[str, object]]) -> str:
    pure_labels = sorted({str(row["raw_class"]) for row in labels if not row["is_mixed_label"]})
    mixed_labels = sorted({str(row["raw_class"]) for row in labels if row["is_mixed_label"]})

    lines = [
        "# Raw Label Inventory",
        "",
        "Generated from the selected raw datasets before any relabeling.",
        "",
        "## Canonical Raw Labels Present",
        "",
    ]

    for label in pure_labels:
        lines.append(f"- {label}")

    lines.extend(
        [
            "",
            "## Mixed Labels Present",
            "",
        ]
    )

    if mixed_labels:
        for label in mixed_labels:
            lines.append(f"- {label}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Source Summary",
            "",
            write_markdown_table(
                sources,
                [
                    "dataset",
                    "subset",
                    "label_source",
                    "image_dir",
                    "labels_csv",
                    "matched_images",
                    "orphan_images",
                    "unique_raw_classes",
                ],
            ),
            "",
            "## Label Counts",
            "",
            write_markdown_table(
                labels,
                ["dataset", "subset", "label_source", "raw_class", "image_count", "is_mixed_label"],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw label inventory into data/metadata.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where metadata outputs will be written.",
    )
    parser.add_argument(
        "--include-misc",
        action="store_true",
        help="Include the Misc dataset in addition to the selected project datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    labels, sources = build_label_inventory(include_misc=args.include_misc)

    labels_csv = output_dir / "raw_label_inventory.csv"
    sources_csv = output_dir / "raw_label_sources.csv"
    markdown_path = output_dir / "raw_label_inventory.md"

    write_csv(
        labels_csv,
        labels,
        ["dataset", "subset", "label_source", "raw_class", "image_count", "is_mixed_label"],
    )
    write_csv(
        sources_csv,
        sources,
        ["dataset", "subset", "label_source", "image_dir", "labels_csv", "matched_images", "orphan_images", "unique_raw_classes"],
    )
    markdown_path.write_text(build_markdown(labels, sources), encoding="utf-8")

    print(f"Wrote {labels_csv}")
    print(f"Wrote {sources_csv}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()

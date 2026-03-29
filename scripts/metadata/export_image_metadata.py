from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    discover_sources,
    ensure_output_dir,
    image_size,
    list_image_files,
    write_csv,
    write_markdown_table,
)


def build_image_summary(include_misc: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for source in discover_sources(include_misc=include_misc):
        image_files = list_image_files(source.image_dir)
        if not image_files:
            continue

        min_area = None
        max_area = None
        min_file = ""
        max_file = ""
        min_width = None
        max_width = None
        min_height = None
        max_height = None
        min_size = ""
        max_size = ""
        valid_images = 0
        unreadable_images = 0

        for image_path in image_files:
            size = image_size(image_path)
            if size is None:
                unreadable_images += 1
                continue

            width, height = size
            area = width * height
            valid_images += 1

            if min_area is None or area < min_area:
                min_area = area
                min_file = image_path.name
                min_size = f"{width}x{height}"
            if max_area is None or area > max_area:
                max_area = area
                max_file = image_path.name
                max_size = f"{width}x{height}"

            min_width = width if min_width is None else min(min_width, width)
            max_width = width if max_width is None else max(max_width, width)
            min_height = height if min_height is None else min(min_height, height)
            max_height = height if max_height is None else max(max_height, height)

        rows.append(
            {
                "dataset": source.dataset,
                "subset": source.subset,
                "label_source": source.label_source,
                "image_dir": str(source.image_dir.relative_to(REPO_ROOT)),
                "image_count": valid_images,
                "unreadable_images": unreadable_images,
                "min_size_by_area": min_size,
                "min_size_file": min_file,
                "max_size_by_area": max_size,
                "max_size_file": max_file,
                "min_width": min_width,
                "max_width": max_width,
                "min_height": min_height,
                "max_height": max_height,
            }
        )

    return sorted(rows, key=lambda row: (str(row["dataset"]), str(row["subset"])))


def build_markdown(summary_rows: list[dict[str, object]]) -> str:
    lines = [
        "# Image Metadata Summary",
        "",
        "Resolution ranges for the selected raw datasets before resizing or preprocessing.",
        "",
        write_markdown_table(
            summary_rows,
            [
                "dataset",
                "subset",
                "label_source",
                "image_dir",
                "image_count",
                "unreadable_images",
                "min_size_by_area",
                "min_size_file",
                "max_size_by_area",
                "max_size_file",
                "min_width",
                "max_width",
                "min_height",
                "max_height",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export image resolution metadata into data/metadata.")
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

    summary_rows = build_image_summary(include_misc=args.include_misc)

    summary_csv = output_dir / "image_resolution_summary.csv"
    summary_md = output_dir / "image_resolution_summary.md"

    write_csv(
        summary_csv,
        summary_rows,
        [
            "dataset",
            "subset",
            "label_source",
            "image_dir",
            "image_count",
            "unreadable_images",
            "min_size_by_area",
            "min_size_file",
            "max_size_by_area",
            "max_size_file",
            "min_width",
            "max_width",
            "min_height",
            "max_height",
        ],
    )
    summary_md.write_text(build_markdown(summary_rows), encoding="utf-8")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()

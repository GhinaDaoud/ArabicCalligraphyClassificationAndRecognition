from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from common import DEFAULT_OUTPUT_DIR, ensure_output_dir, write_csv, write_markdown_table


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_balance_rows(rows: list[dict[str, str]], include_mixed: bool) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected = rows if include_mixed else [row for row in rows if row["is_mixed_label"] == "False"]

    counts: Counter[str] = Counter()
    for row in selected:
        counts[row["raw_class"]] += int(row["image_count"])

    total = sum(counts.values())
    max_count = max(counts.values()) if counts else 0
    min_count = min(counts.values()) if counts else 0

    balance_rows: list[dict[str, object]] = []
    for class_name in sorted(counts):
        count = counts[class_name]
        balance_rows.append(
            {
                "raw_class": class_name,
                "image_count": count,
                "share_of_total": f"{(count / total):.6f}" if total else "0.000000",
                "share_of_max_class": f"{(count / max_count):.6f}" if max_count else "0.000000",
            }
        )

    summary = {
        "total_images": total,
        "num_classes": len(counts),
        "largest_class_count": max_count,
        "smallest_class_count": min_count,
        "min_to_max_ratio": f"{(min_count / max_count):.6f}" if max_count else "0.000000",
        "include_mixed": include_mixed,
    }
    return balance_rows, summary


def build_markdown(balance_rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    lines = [
        "# Overall Class Balance",
        "",
        "Computed from `data/metadata/raw_label_inventory.csv`.",
        "",
        "## Summary",
        "",
        f"- total_images: {summary['total_images']}",
        f"- num_classes: {summary['num_classes']}",
        f"- largest_class_count: {summary['largest_class_count']}",
        f"- smallest_class_count: {summary['smallest_class_count']}",
        f"- min_to_max_ratio: {summary['min_to_max_ratio']}",
        f"- include_mixed: {summary['include_mixed']}",
        "",
        "## Per-Class Counts",
        "",
        write_markdown_table(
            balance_rows,
            ["raw_class", "image_count", "share_of_total", "share_of_max_class"],
        ),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export overall class balance into data/metadata.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "raw_label_inventory.csv",
        help="Raw label inventory CSV to aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where balance outputs will be written.",
    )
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help="Include mixed-label rows in the aggregation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    rows = read_rows(args.input_csv)
    balance_rows, summary = build_balance_rows(rows, include_mixed=args.include_mixed)

    stem = "overall_class_balance_with_mixed" if args.include_mixed else "overall_class_balance"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"

    write_csv(csv_path, balance_rows, ["raw_class", "image_count", "share_of_total", "share_of_max_class"])
    md_path.write_text(build_markdown(balance_rows, summary), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

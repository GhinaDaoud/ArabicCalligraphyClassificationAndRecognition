from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_CSV = REPO_ROOT / "data" / "processed" / "augmented" / "labels.csv"
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_METADATA_DIR = REPO_ROOT / "data" / "metadata"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def group_rows(rows: list[dict[str, str]]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for row in rows:
        source_path = (row.get("source_path") or "").strip()
        group_key = source_path or (row.get("img_name") or "").strip()
        class_name = (row.get("class") or "").strip()
        if not group_key or not class_name:
            continue

        if group_key not in grouped:
            grouped[group_key] = {
                "class_name": class_name,
                "rows": [],
            }
        grouped[group_key]["rows"].append(row)
    return grouped


def split_group_keys(
    grouped: dict[str, dict],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for group_key, payload in grouped.items():
        by_class[payload["class_name"]].append(group_key)

    rng = random.Random(seed)
    train_keys: set[str] = set()
    val_keys: set[str] = set()

    for class_name, group_keys in sorted(by_class.items()):
        shuffled = list(group_keys)
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            train_keys.update(shuffled)
            continue

        proposed = int(round(len(shuffled) * val_ratio))
        val_count = min(len(shuffled) - 1, max(1, proposed))
        val_subset = shuffled[:val_count]
        train_subset = shuffled[val_count:]

        val_keys.update(val_subset)
        train_keys.update(train_subset)

    return train_keys, val_keys


def build_split_rows(
    grouped: dict[str, dict],
    train_keys: set[str],
    val_keys: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []

    for group_key, payload in grouped.items():
        rows = payload["rows"]
        if group_key in train_keys:
            train_rows.extend(rows)
        elif group_key in val_keys:
            val_rows.extend(row for row in rows if not parse_bool(row.get("is_augmented", "")))

    return train_rows, val_rows


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[(row.get("class") or "").strip()] += 1
    return dict(sorted(counts.items()))


def write_report(
    path: Path,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    train_keys: set[str],
    val_keys: set[str],
) -> None:
    train_counts = class_counts(train_rows)
    val_counts = class_counts(val_rows)
    overlap = train_keys & val_keys

    lines = [
        "# Train/Validation Split Report",
        "",
        f"- train_rows: {len(train_rows)}",
        f"- val_rows: {len(val_rows)}",
        f"- train_unique_sources: {len(train_keys)}",
        f"- val_unique_sources: {len(val_keys)}",
        f"- source_overlap: {len(overlap)}",
        "",
        "## Train Class Counts",
        "",
    ]
    for class_name, count in train_counts.items():
        lines.append(f"- {class_name}: {count}")

    lines.extend(["", "## Validation Class Counts", ""])
    for class_name, count in val_counts.items():
        lines.append(f"- {class_name}: {count}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped train/validation splits that keep augmented siblings out of validation."
    )
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.labels_csv)
    grouped = group_rows(rows)
    train_keys, val_keys = split_group_keys(grouped, val_ratio=args.val_ratio, seed=args.seed)
    train_rows, val_rows = build_split_rows(grouped, train_keys=train_keys, val_keys=val_keys)

    train_csv = args.splits_dir / "train.csv"
    val_csv = args.splits_dir / "val.csv"
    report_md = args.metadata_dir / "train_val_split_report.md"

    write_rows(train_csv, train_rows)
    write_rows(val_csv, val_rows)
    write_report(report_md, train_rows, val_rows, train_keys=train_keys, val_keys=val_keys)

    print(f"Wrote {train_csv}")
    print(f"Wrote {val_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()

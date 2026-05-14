from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_history_csv(path: str | Path, rows: Iterable[dict[str, float]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_list[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


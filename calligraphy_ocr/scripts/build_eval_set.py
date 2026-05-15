"""
Build a stratified evaluation subset from HICMA + DuwaBench.

Output: calligraphy_ocr/eval_gt.csv with columns:
    filename, style, transcript, source, abs_path, notes

Usage:
    python scripts/build_eval_set.py [--per-style 7] [--seed 42]
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../693
DATASET_ROOT = REPO_ROOT / "dataset"
OUT_PATH = REPO_ROOT / "calligraphy_ocr" / "eval_gt.csv"

STYLES = ["Naskh", "Ruq'ah", "Diwani", "Thuluth", "Kufic", "Muhaqaq", "Nasta'liq"]

CLASS_ALIASES = {
    "Muhaquaq": "Muhaqaq",
    "Nastaliq": "Nasta'liq",
    "Ruqah": "Ruq'ah",
    "Ruqaah": "Ruq'ah",
    "Ruqa": "Ruq'ah",
    "Ruqʿah": "Ruq'ah",
    "Ruq‘ah": "Ruq'ah",
}


def normalize_class(c: object) -> str:
    s = str(c).strip().strip('"').strip("'")
    return CLASS_ALIASES.get(s, s)


def flatten_label(x: object) -> str:
    """DuwaBench stores some labels as Python list-strings: ['text1', 'text2']."""
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple)):
                return " ".join(str(s) for s in parsed)
        except (ValueError, SyntaxError):
            pass
    return str(x)


def load_hicma() -> pd.DataFrame:
    rows = []
    for setn in (1, 2, 3):
        csv = DATASET_ROOT / f"HICMA/Set{setn}/labels.csv"
        df = pd.read_csv(csv)
        df["source"] = f"HICMA-Set{setn}"
        df["abs_path"] = df["img_name"].apply(
            lambda n, s=setn: str(DATASET_ROOT / f"HICMA/Set{s}/images" / n)
        )
        rows.append(df[["img_name", "class", "label", "source", "abs_path"]])
    return pd.concat(rows, ignore_index=True)


def load_duwabench() -> pd.DataFrame:
    csv = DATASET_ROOT / "DuwaBench/labels.csv"
    df = pd.read_csv(csv)
    df["source"] = "DuwaBench"
    df["abs_path"] = df["img_name"].apply(
        lambda n: str(DATASET_ROOT / "DuwaBench/images" / n)
    )
    df["label"] = df["label"].apply(flatten_label)
    return df[["img_name", "class", "label", "source", "abs_path"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-style", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hicma = load_hicma()
    duwa = load_duwabench()
    full = pd.concat([hicma, duwa], ignore_index=True)

    full["class_norm"] = full["class"].apply(normalize_class)
    full["label"] = full["label"].astype(str).str.strip()

    # Drop rows with empty labels or where the image file is missing.
    full = full[full["label"].str.len() > 0].copy()
    full["exists"] = full["abs_path"].apply(lambda p: Path(p).is_file())
    missing = (~full["exists"]).sum()
    if missing:
        print(f"warn: dropping {missing} rows whose image file is missing", file=sys.stderr)
    full = full[full["exists"]].drop(columns=["exists"])

    # Per-style stratified sample.
    picked = []
    for style in STYLES:
        pool = full[full["class_norm"] == style]
        n_avail = len(pool)
        if n_avail == 0:
            print(f"warn: no rows for style {style!r}", file=sys.stderr)
            continue
        take = min(args.per_style, n_avail)
        sample = pool.sample(n=take, random_state=args.seed)
        picked.append(sample)
        print(f"{style:<10s}  picked {take}/{n_avail}")

    eval_df = pd.concat(picked, ignore_index=True)
    eval_df = eval_df.rename(
        columns={"img_name": "filename", "class_norm": "style", "label": "transcript"}
    )
    eval_df["notes"] = ""
    eval_df = eval_df[["filename", "style", "transcript", "source", "abs_path", "notes"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {len(eval_df)} rows to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

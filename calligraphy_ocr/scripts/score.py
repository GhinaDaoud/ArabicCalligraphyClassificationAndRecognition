"""
Score zero-shot predictions vs. ground truth.

Reads results/zero_shot/<model>_predictions.csv for each model that has a
predictions file present, applies Arabic normalization, and computes:
    - per-style CER, WER, BLEU
    - overall (micro & macro)
    - 3 best / 3 worst predictions per (model, style) by CER

Output: results/zero_shot/summary.md

Usage:
    python scripts/score.py
    python scripts/score.py --models qari-v0.3 sherif-handwritten-v3

Required:
    pip install jiwer sacrebleu
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = REPO_ROOT / "calligraphy_ocr" / "results" / "zero_shot"
SUMMARY_PATH = PRED_DIR / "summary.md"

# ── Arabic normalization ────────────────────────────────────────────────────

# Tashkeel (harakat) range: U+064B..U+0652 plus a few extras.
TASHKEEL_RE = re.compile(
    "[" "ً-ٟ" "ٰ" "ۖ-ۭ" "]"
)
TATWEEL_RE = re.compile("ـ")  # kashida
NON_ARABIC_PUNCT_RE = re.compile(r"[^؀-ۿ\s]")  # strip latin/punct/digits


def normalize_arabic(s: str, *, strip_punct: bool = True) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = TASHKEEL_RE.sub("", s)
    s = TATWEEL_RE.sub("", s)
    # Alif variants → bare alif.
    s = s.replace("آ", "ا")  # آ
    s = s.replace("أ", "ا")  # أ
    s = s.replace("إ", "ا")  # إ
    # Yaa variants → bare yaa.
    s = s.replace("ى", "ي")  # ى → ي
    # Taa marbuta → haa (canonical for OCR scoring).
    s = s.replace("ة", "ه")  # ة → ه
    # Hamza on waw/yaa → drop hamza.
    s = s.replace("ؤ", "و")  # ؤ → و
    s = s.replace("ئ", "ي")  # ئ → ي
    if strip_punct:
        s = NON_ARABIC_PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── Metrics ─────────────────────────────────────────────────────────────────

def cer_one(ref: str, hyp: str) -> float:
    """Character error rate via Levenshtein distance, character-level."""
    import jiwer

    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


def wer_one(ref: str, hyp: str) -> float:
    import jiwer

    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0
    return jiwer.wer(ref, hyp)


def bleu_one(ref: str, hyp: str) -> float:
    import sacrebleu

    if not ref.strip() and not hyp.strip():
        return 100.0
    # Sentence-BLEU at the character level would over-reward; word-level is
    # standard for OCR papers. Use sacrebleu sentence-BLEU (smoothed).
    return sacrebleu.sentence_bleu(hyp, [ref]).score


def score_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds normalized fields and per-row metrics."""
    df = df.copy()
    df["transcript_norm"] = df["transcript"].apply(normalize_arabic)
    df["prediction_norm"] = df["prediction"].fillna("").apply(normalize_arabic)
    df["cer"] = [
        cer_one(r, h) for r, h in zip(df["transcript_norm"], df["prediction_norm"])
    ]
    df["wer"] = [
        wer_one(r, h) for r, h in zip(df["transcript_norm"], df["prediction_norm"])
    ]
    df["bleu"] = [
        bleu_one(r, h) for r, h in zip(df["transcript_norm"], df["prediction_norm"])
    ]
    return df


def per_style_table(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("style")[["cer", "wer", "bleu"]].mean()
    overall = pd.DataFrame(
        df[["cer", "wer", "bleu"]].mean().to_dict(), index=["__overall__"]
    )
    return pd.concat([grp, overall])


def fmt_table(table: pd.DataFrame) -> str:
    """Render a pandas DataFrame as a github-flavored markdown table."""
    cols = list(table.columns)
    lines = ["| style | " + " | ".join(cols) + " |"]
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for idx, row in table.iterrows():
        cells = [f"{row[c]:.3f}" if c != "bleu" else f"{row[c]:.2f}" for c in cols]
        label = "**overall**" if idx == "__overall__" else str(idx)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_worst_examples(df: pd.DataFrame, k: int = 3) -> dict[str, dict[str, list]]:
    """For each style, k best (lowest CER) and k worst (highest CER) rows."""
    out: dict[str, dict[str, list]] = {}
    for style, sub in df.groupby("style"):
        sub_sorted = sub.sort_values("cer")
        best = sub_sorted.head(k)
        worst = sub_sorted.tail(k).iloc[::-1]
        out[style] = {
            "best": best[["filename", "transcript", "prediction", "cer"]].to_dict("records"),
            "worst": worst[["filename", "transcript", "prediction", "cer"]].to_dict("records"),
        }
    return out


def render_examples(examples_by_style: dict, model_id: str) -> str:
    parts = [f"\n#### Qualitative — {model_id}\n"]
    for style, blob in examples_by_style.items():
        parts.append(f"\n##### {style}\n")
        parts.append("**3 best (lowest CER):**\n")
        for row in blob["best"]:
            parts.append(
                f"- `{row['filename']}` — CER {row['cer']:.3f}\n"
                f"  - GT:   {row['transcript']}\n"
                f"  - Pred: {row['prediction']}"
            )
        parts.append("\n**3 worst (highest CER):**\n")
        for row in blob["worst"]:
            parts.append(
                f"- `{row['filename']}` — CER {row['cer']:.3f}\n"
                f"  - GT:   {row['transcript']}\n"
                f"  - Pred: {row['prediction']}"
            )
    return "\n".join(parts)


def render_perf_table(df: pd.DataFrame) -> str:
    """Latency / VRAM aggregate for sanity checks."""
    if "latency_s" not in df.columns:
        return ""
    mean_lat = df["latency_s"].mean()
    p95_lat = df["latency_s"].quantile(0.95)
    peak = df.get("peak_vram_mb", pd.Series([0])).max()
    return (
        f"- mean latency: **{mean_lat:.2f}s**, p95: **{p95_lat:.2f}s** per image\n"
        f"- peak VRAM observed: **{peak:.0f} MB**\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                        help="restrict to a subset of model ids; default = all "
                             "*_predictions.csv files in results/zero_shot/")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    pred_files = sorted(PRED_DIR.glob("*_predictions.csv"))
    if not pred_files:
        print(f"error: no *_predictions.csv files found in {PRED_DIR}", file=sys.stderr)
        return 1

    md = ["# Zero-shot Arabic calligraphy OCR — summary\n"]
    md.append(
        "_Eval set: 49 stratified samples (7 per style) from HICMA + DuwaBench, "
        "with ground-truth Arabic transcripts. Arabic normalization (alif/yaa/"
        "taa-marbuta unification, tashkeel + kashida stripped) applied before "
        "scoring._\n"
    )

    # Headline: aggregate model comparison.
    headline_rows = []

    for f in pred_files:
        model_id = f.stem.replace("_predictions", "")
        if args.models and model_id not in args.models:
            continue

        df = pd.read_csv(f)
        df = score_df(df)
        table = per_style_table(df)

        md.append(f"\n## {model_id}\n")
        md.append(f"_Source: `{f.relative_to(REPO_ROOT)}` ({len(df)} rows)_\n")
        md.append(render_perf_table(df))
        md.append("\n### Per-style metrics (lower CER/WER, higher BLEU is better)\n")
        md.append(fmt_table(table))
        md.append("")

        examples = best_worst_examples(df, k=args.top_k)
        md.append(render_examples(examples, model_id))

        # Save the scored CSV alongside.
        scored_path = f.with_name(f.stem + "_scored.csv")
        df.to_csv(scored_path, index=False)

        overall = table.loc["__overall__"]
        headline_rows.append({
            "model": model_id, "CER": overall["cer"],
            "WER": overall["wer"], "BLEU": overall["bleu"],
        })

    if len(headline_rows) > 1:
        head = pd.DataFrame(headline_rows).set_index("model")
        head_md = ["\n# Headline comparison (overall)\n",
                   "| model | CER | WER | BLEU |",
                   "|---|---|---|---|"]
        for m, row in head.iterrows():
            head_md.append(f"| {m} | {row['CER']:.3f} | {row['WER']:.3f} | {row['BLEU']:.2f} |")
        md = [md[0], md[1], "\n".join(head_md)] + md[2:]

    SUMMARY_PATH.write_text("\n".join(md))
    print(f"wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

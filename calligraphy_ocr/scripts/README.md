# Step 2 — Zero-shot benchmark runbook

## Files in this directory

| File | What it does |
|------|--------------|
| `build_eval_set.py` | Reads HICMA + DuwaBench `labels.csv` files, samples a stratified eval subset (7 imgs/style × 7 styles = 49 imgs), writes `../eval_gt.csv`. Already run; rerun only if you change the eval set. |
| `run_zero_shot.py` | Loads one of the 3 shortlisted models, runs inference on `eval_gt.csv`, saves `../results/zero_shot/<model>_predictions.csv` with predictions, latency, peak VRAM. |
| `score.py` | Reads all `*_predictions.csv` in `../results/zero_shot/`, applies Arabic normalization, computes CER/WER/BLEU per style, writes `../results/zero_shot/summary.md`. |

## One-time setup

```bash
# In your conda env / venv:
pip install "transformers>=4.49.0" accelerate qwen-vl-utils Pillow jiwer sacrebleu
```

`transformers>=4.49.0` is required for `Qwen2_5_VLForConditionalGeneration`
(used by the sherif1313 handwritten model). The Qari models use the older
`Qwen2VLForConditionalGeneration` which is in any recent transformers version.

## Run order

```bash
# (Step 0) Build the eval set if eval_gt.csv doesn't already exist
python calligraphy_ocr/scripts/build_eval_set.py

# (Step 1) Inference — once per model. Each will download ~5–7 GB to ~/.cache.
# Each model on ~50 images at bf16 should take ~5–15 minutes on one RTX 2080.
python calligraphy_ocr/scripts/run_zero_shot.py --model sherif-handwritten-v3
python calligraphy_ocr/scripts/run_zero_shot.py --model qari-v0.3
python calligraphy_ocr/scripts/run_zero_shot.py --model qari-v0.2.2.1

# (Step 2) Score everything we have predictions for
python calligraphy_ocr/scripts/score.py
# → writes calligraphy_ocr/results/zero_shot/summary.md
```

## Smoke-test before committing to a full run

If you want to verify a model loads + runs without burning 15 min:

```bash
python calligraphy_ocr/scripts/run_zero_shot.py --model qari-v0.3 --limit 3
```

## VRAM tuning

If a model OOMs on your 2080 (8 GB), the knob to turn is `--max-pixels`. The
default budget is `1280 * 28 * 28 = 1003520` pixel-tokens. Try halving it:

```bash
python calligraphy_ocr/scripts/run_zero_shot.py --model qari-v0.3 --max-pixels 501760
```

Lowering this trades resolution detail for memory. For calligraphy you want
this as high as fits — fine details matter.

## What to look at in `summary.md`

1. **Headline table** at the top — which model has the lowest overall CER. Expect:
   - `qari-v0.2.2.1` will be best on clean printed-style images, worst on cursive/decorative
   - `sherif-handwritten-v3` should win on Ruq'ah / Nasta'liq (handwriting-heavy styles)
   - `qari-v0.3` should sit between the two
2. **Per-style breakdown** — large CER spreads between styles tell us style conditioning will help
3. **Qualitative 3-best/3-worst** — read these. Numbers can lie; eyeballing the predictions tells you whether the model is making sensible word-level errors or hallucinating.

## Notes / known caveats

- The Arabic normalization in `score.py` strips tashkeel (harakat) and kashida,
  unifies alif/yaa/taa-marbuta variants, and unifies hamza-on-waw/yaa to
  base-waw/yaa. Standalone hamza `ء` is **kept**. This matches the convention
  most Arabic OCR papers use; if the project later wants strict-tashkeel CER,
  pass un-normalized strings to `cer_one`.
- BLEU is sentence-level (sacrebleu) on space-tokenized words. For short
  calligraphic phrases (often 1–5 words) BLEU is noisy — treat CER as the
  primary metric and BLEU as a sanity check.
- All three models share a single zero-shot prompt (in `run_zero_shot.py`
  under `PROMPT`) so results are comparable. If you later want to test
  prompt sensitivity, that's the variable to ablate.

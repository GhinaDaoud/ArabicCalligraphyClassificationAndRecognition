# Calligraphy OCR — Stage 2 of the ECE 693 Pipeline

Stage 2 of the Arabic calligraphy recognition project. Stage 1 (style classifier) lives in
the parent project. This directory holds the OCR survey, zero-shot benchmark harness, and
fine-tuning scaffold for the chosen baseline.

## Layout

```
calligraphy_ocr/
├── models_survey.md          # Step 1 — done. Candidate model fact-extraction + verdicts.
├── eval_gt.csv               # Step 2 — schema stub for the evaluation ground truth.
│                             #   columns: filename, style, transcript, source, notes
├── scripts/                  # Step 2 — eval harness (build_eval_set.py, run_zero_shot.py,
│                             #   score.py). To be filled in next.
├── results/
│   └── zero_shot/            # Per-model predictions + summary.md
├── train/                    # Step 3 — fine-tuning scaffold (dataset, LoRA, train loop,
│                             #   smoke test).
└── data/                     # Symlink target for paired calligraphy data (HICMA/DuwaBench).
```

## eval_gt.csv schema

| column | description |
|--------|-------------|
| `filename` | image basename (resolved against `data/<source>/images/`) |
| `style` | one of: Naskh, Ruq'ah, Diwani, Thuluth, Kufic, Muhaqaq, Nasta'liq |
| `transcript` | ground-truth Arabic text. For HICMA/DuwaBench this can be auto-populated from their `labels.csv`; for Misc/Rufa it must be hand-transcribed. |
| `source` | one of: HICMA-Set1, HICMA-Set2, HICMA-Set3, DuwaBench, Misc, Rufa |
| `notes` | free-text — legibility flags, partial text, etc. |

## Status

- [x] Step 1 — Model survey (`models_survey.md`)
- [ ] Step 2 — Zero-shot benchmark harness + run
- [ ] Step 3 — Fine-tuning scaffold + smoke test

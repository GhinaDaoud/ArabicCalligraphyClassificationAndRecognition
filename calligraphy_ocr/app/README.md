# Arabic Calligraphy Recognition — Demo App

Two-stage pipeline demo:
1. **Stage 1** — ResNet18 local-global fusion classifier predicts the calligraphy style.
2. **Stage 2** — A Qwen-VL OCR model transcribes the Arabic text. Optional style-conditioned prompt prefix.

## Run

```bash
/home/ahmad/anaconda3/envs/calligraphy_ocr/bin/python calligraphy_ocr/app/app.py
```

Open http://localhost:7860 in a browser.

## Plug in the trained Stage-1 classifier

```bash
RESNET18_CKPT=/path/to/style_classifier.pth \
  /home/ahmad/anaconda3/envs/calligraphy_ocr/bin/python calligraphy_ocr/app/app.py
```

Or place the file at `calligraphy_ocr/app/style_classifier.pth` and the wrapper finds it automatically (set `RESNET18_CKPT=$(pwd)/calligraphy_ocr/app/style_classifier.pth`).

If no checkpoint is loaded, the Stage-1 panel shows a "checkpoint not loaded" notice and the OCR side still works.

## Files

| File | Role |
|------|------|
| `app.py` | Gradio Blocks layout + custom CSS (manuscript palette, Amiri Arabic font, RTL output zone) |
| `style_classifier.py` | `LocalGlobalFusionResNet18` (matches §IV of the progress report) + `StyleClassifier` wrapper |
| `ocr_backend.py` | Lazy-loading wrapper for the 3 Qwen-VL candidates; `transcribe(image, model_name, style_hint)` |

## Default OCR backend

`Qari-OCR v0.3 (handwriting-aware)`. Switchable in the dropdown to:
- `Sherif Handwritten v3` (Qwen2.5-VL-3B trained on Muharaf+KHATT)
- `Qari-OCR v0.2.2.1 (printed)` — printed-Arabic upper bound

After the Step-2 zero-shot benchmark picks a winner, change `DEFAULT_MODEL` in `ocr_backend.py` to that model.

## Aesthetic notes

- Palette: deep teal ink (#0d3d3a), illumination gold (#c9a961), aged parchment (#f5efe0), maroon accent (#6e2a3e)
- Typography: **Amiri** for Arabic (designed at the Bibliotheca Alexandrina), **Cormorant Garamond** for Latin headings
- Subtle 8-fold-star geometric pattern in the page background (CSS-only, low opacity)
- RTL output zone with right-aligned ink-on-parchment styling

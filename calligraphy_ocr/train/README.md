# Step 3 — QLoRA fine-tuning of AIN-7B on Arabic calligraphy

Stage-2 OCR fine-tuning scaffold. Uses 4-bit NF4 quantized AIN (Qwen2-VL-7B
Arabic base) + LoRA adapters, with style-conditioned prompt prefix.

## Files

| File | Role |
|------|------|
| `dataset.py` | Reads paired HICMA + DuwaBench transcripts (5,611 train + 624 val rows), excludes the 49 eval images, exposes `CalligraphyOCRDataset` and `build_train_val_split()` |
| `train.py` | QLoRA training loop. Smoke-test mode (`--smoke`) verifies the pipeline in ~3 min; full training mode trains all epochs and saves a LoRA adapter. |
| `ckpt/` | Output dir for saved LoRA adapters (created on first save) |

## Run

```bash
# 1. Smoke test — 20 steps on tiny sub-sample, ~3 min on a 4090.
#    Verifies loss decreases without OOM.
python calligraphy_ocr/train/train.py --smoke

# 2. Full training — 3 epochs, default settings.
python calligraphy_ocr/train/train.py \
    --epochs 3 \
    --per-style-cap 500 \
    --output-dir calligraphy_ocr/train/ckpt/ain-7b-calligraphy-lora
```

## Smoke test result (recorded 2026-05-14)

- trainable params: 80,740,352 / 8,372,115,968 (0.96%)
- vision encoder + projector: frozen (391/730 params)
- VRAM steady, no OOM
- loss: 2.20 → 1.08 over 20 steps (rough trend)
- ~8.5 s/step on RTX 4090, bs=2, grad_accum=4 (effective batch 8)

## Design summary

**Quantization:** 4-bit NF4 with double-quant + bf16 compute dtype (QLoRA).
The 7B base needs ~16 GB in bf16 for inference alone; QLoRA drops the
resident base to ~4-5 GB, freeing budget for optimizer + activations.

**LoRA targets:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` —
all attention and MLP projection matrices in the 28 Qwen2 transformer blocks.
`r=32, alpha=64, dropout=0.05`.

**Frozen:** vision encoder (ViT), vision→LLM projector, embeddings, lm_head.
We don't train the visual side because Qwen2-VL's ViT is already broadly
capable and Arabic generation errors dominate over vision errors in our
zero-shot benchmark.

**Style conditioning:** every training example's user prompt is prepended
with `"This image contains <style> Arabic calligraphy."` where `<style>` is
the ground-truth style from HICMA/DuwaBench. At inference time the predicted
style from the Stage-1 ResNet18 classifier is used. The collator masks all
user-side tokens out of the loss via `-100` so only the assistant's Arabic
transcription is supervised.

**Style dropout:** 10% of training samples omit the style hint, so the model
also learns an unconditional baseline. Useful for ablations and as a fallback
when Stage-1 confidence is low.

**Per-style cap:** Naskh and Thuluth dominate the corpus (3,400 + 1,500 imgs
vs. Muhaqaq's 9). We cap each style at 500 training samples to prevent the
model from collapsing to a Naskh-style transcriber. Set `--per-style-cap` to
override.

## Saved artifacts

After full training, `ckpt/ain-7b-calligraphy-lora/` contains:
- `adapter_model.safetensors` — the LoRA weights (~300 MB)
- `adapter_config.json` — LoRA hyperparameters
- `tokenizer.json`, `preprocessor_config.json`, etc. — from the base processor

Load for inference:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base = Qwen2VLForConditionalGeneration.from_pretrained("MBZUAI/AIN", ...)
model = PeftModel.from_pretrained(base, "calligraphy_ocr/train/ckpt/ain-7b-calligraphy-lora")
processor = AutoProcessor.from_pretrained("calligraphy_ocr/train/ckpt/ain-7b-calligraphy-lora")
```

## Open knobs to tune for full training

- `--epochs`: 3 is the default LoRA convention; can try 2 if val loss plateaus
- `--lr`: 1e-4 is the LoRA standard; consider 5e-5 if you see instability
- `--batch-size` / `--grad-accum`: effective batch 8 by default; effective 16 is
  worth trying if VRAM is comfortable
- `--max-pixels`: bigger = more detail = slower. Calligraphy benefits from
  high resolution; we kept it at 1280×28×28 = 1.0M pixels, same as inference
- `--style-dropout`: try 0.0 if Stage-1 is highly accurate; bump to 0.2 if
  Stage-1 is unsure

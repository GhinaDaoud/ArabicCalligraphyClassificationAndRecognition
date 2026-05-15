"""
Zero-shot Arabic OCR inference on the eval set.

Loads one of the three shortlisted Qwen-VL family models, runs inference on
calligraphy_ocr/eval_gt.csv, and writes per-image predictions + latency + GPU
peak memory to results/zero_shot/<model_id>_predictions.csv.

Usage:
    python scripts/run_zero_shot.py --model sherif-handwritten-v3
    python scripts/run_zero_shot.py --model qari-v0.3
    python scripts/run_zero_shot.py --model qari-v0.2.2.1

    # All three back-to-back:
    for m in sherif-handwritten-v3 qari-v0.3 qari-v0.2.2.1; do
        python scripts/run_zero_shot.py --model $m
    done

Required:
    pip install "transformers>=4.49.0" accelerate qwen-vl-utils Pillow

GPU memory note: each of these is a 2-3B Qwen-VL model. bf16 fits comfortably
in 8GB. If OOM, lower --max-pixels (default 1280*28*28 = 1003520).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../693
EVAL_GT = REPO_ROOT / "calligraphy_ocr" / "eval_gt.csv"
OUT_DIR = REPO_ROOT / "calligraphy_ocr" / "results" / "zero_shot"

# Model registry. `loader` selects the transformers class.
# `dtype` overrides bf16 default per-model (some Qwen2.5-VL fine-tunes
# produce NaN-logits in bf16, manifesting as a stream of `!` tokens).
# `adapter_base` indicates this repo holds a PEFT/LoRA adapter only — load
# the named base model first, then attach the adapter.
MODELS = {
    "sherif-handwritten-v3": {
        "repo": "sherif1313/Arabic-English-handwritten-OCR-v3",
        "loader": "qwen2_5_vl",
        "dtype": "float16",
    },
    "qari-v0.3": {
        "repo": "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
        "loader": "qwen2_vl",
        "dtype": "bfloat16",
    },
    "qari-v0.2.2.1": {
        "repo": "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct",
        "loader": "qwen2_vl",
        "dtype": "bfloat16",
        "adapter_base": "Qwen/Qwen2-VL-2B-Instruct",
    },
    # Round 2 candidates — see models_survey.md "Round 2" section.
    "baseer-nakba": {
        # NakbaNLP-2026-winning Qwen2.5-VL-3B fine-tune for historical Arabic
        # manuscripts. Same tied-embeddings pattern as sherif1313 → needs fix.
        "repo": "Misraj/Baseer__Nakba",
        "loader": "qwen2_5_vl",
        "dtype": "bfloat16",
    },
    "ain-7b": {
        # MBZUAI/AIN — Qwen2-VL-7B Arabic generalist, MIT, full lm_head shipped.
        "repo": "MBZUAI/AIN",
        "loader": "qwen2_vl",
        "dtype": "bfloat16",
    },
    "ketaba-lora": {
        # LoRA on top of sherif-handwritten-v3 (NakbaNLP per-line winner).
        # Base inherits tied-embeddings issue → fix applies to the base.
        "repo": "HassanB4/Ketaba-OCR-LoRA",
        "loader": "qwen2_5_vl",
        "dtype": "float16",
        "adapter_base": "sherif1313/Arabic-English-handwritten-OCR-v3",
    },
    "ain-7b-lora": {
        # OUR fine-tune: QLoRA adapter on MBZUAI/AIN, trained on paired
        # HICMA+DuwaBench calligraphy with style-prefix prompts.
        "repo": "/home/ahmad/Desktop/beebot/693/calligraphy_ocr/train/ckpt/ain-7b-calligraphy-lora",
        "loader": "qwen2_vl",
        "dtype": "bfloat16",
        "adapter_base": "MBZUAI/AIN",
    },
    "ain-7b-lora-ep1": {
        "repo": "/home/ahmad/Desktop/beebot/693/calligraphy_ocr/train/ckpt/ain-7b-calligraphy-lora/checkpoint-702",
        "loader": "qwen2_vl",
        "dtype": "bfloat16",
        "adapter_base": "MBZUAI/AIN",
    },
    "ain-7b-lora-ep2": {
        "repo": "/home/ahmad/Desktop/beebot/693/calligraphy_ocr/train/ckpt/ain-7b-calligraphy-lora/checkpoint-1404",
        "loader": "qwen2_vl",
        "dtype": "bfloat16",
        "adapter_base": "MBZUAI/AIN",
    },
}

# A single zero-shot prompt used for all three models so results are comparable.
# Stronger anti-HTML phrasing because Qari was trained on HTML-formatted Arabic
# documents and otherwise wraps output in <b>/<i> tags.
PROMPT = (
    "Below is an image of Arabic calligraphic text. "
    "Transcribe the Arabic text exactly as written in the image. "
    "Output ONLY the Arabic characters as plain text. "
    "Do NOT use any HTML tags, markdown, or formatting. "
    "Do NOT add commentary, translation, repetition, or padding. "
    "If unclear, output what is visible and stop."
)


def load_model_and_processor(model_key: str, max_pixels: int):
    cfg = MODELS[model_key]
    repo = cfg["repo"]
    loader = cfg["loader"]
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    adapter_base = cfg.get("adapter_base")

    from transformers import AutoProcessor
    if loader == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration as Cls
    elif loader == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    else:
        raise ValueError(f"unknown loader: {loader}")

    def _force_tie_embeddings(m):
        """Workaround for repos that ship tied embeddings (lm_head.weight not in
        safetensors; expected to alias model.embed_tokens.weight). transformers
        5.x doesn't always re-tie on load, leaving lm_head randomly initialized
        and the model emits garbage. Forcibly re-tie — but ONLY when the model's
        config actually requests tied embeddings, otherwise we'd overwrite a
        properly trained separate lm_head (e.g. MBZUAI/AIN)."""
        try:
            cfg = getattr(m, "config", None)
            if cfg is None:
                return
            tie = bool(getattr(cfg, "tie_word_embeddings", False))
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                tie = tie or bool(getattr(text_cfg, "tie_word_embeddings", False))
            if not tie:
                return
            in_emb = m.get_input_embeddings()
            out_emb = m.get_output_embeddings()
            if out_emb is not None and in_emb is not None:
                out_emb.weight = in_emb.weight
        except (AttributeError, RuntimeError):
            pass

    if adapter_base:
        # PEFT adapter on top of a public base model.
        from peft import PeftModel

        base = Cls.from_pretrained(adapter_base, torch_dtype=dtype, device_map="auto")
        # Apply the tied-embedding fix to the BASE before wrapping — required if
        # the base itself ships tied weights (e.g. sherif1313/...-v3).
        _force_tie_embeddings(base)
        model = PeftModel.from_pretrained(base, repo)
        # The adapter repo may not ship a processor; use the base's.
        processor = AutoProcessor.from_pretrained(
            adapter_base, min_pixels=256 * 28 * 28, max_pixels=max_pixels
        )
    else:
        model = Cls.from_pretrained(repo, torch_dtype=dtype, device_map="auto")
        _force_tie_embeddings(model)
        processor = AutoProcessor.from_pretrained(
            repo, min_pixels=256 * 28 * 28, max_pixels=max_pixels
        )

    return model, processor


def run_one(model, processor, image_path: str, max_new_tokens: int) -> tuple[str, float, int]:
    """Returns (prediction_text, latency_seconds, peak_vram_bytes)."""
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
        )
    latency = time.perf_counter() - t0

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return decoded.strip(), latency, peak_mem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28,
                        help="vision token budget; lower = less VRAM, less detail")
    parser.add_argument("--limit", type=int, default=None,
                        help="run on the first N rows only (smoke test)")
    args = parser.parse_args()

    if not EVAL_GT.is_file():
        print(f"error: missing {EVAL_GT}. Run build_eval_set.py first.", file=sys.stderr)
        return 1

    eval_df = pd.read_csv(EVAL_GT)
    if args.limit:
        eval_df = eval_df.head(args.limit)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.model}_predictions.csv"

    print(f"loading {MODELS[args.model]['repo']}...")
    model, processor = load_model_and_processor(args.model, args.max_pixels)
    model.eval()
    print(f"model loaded on device(s): {set(p.device for p in model.parameters())}")

    rows = []
    for i, row in eval_df.iterrows():
        img = row["abs_path"]
        if not Path(img).is_file():
            print(f"  [{i+1}/{len(eval_df)}] MISSING {img}", file=sys.stderr)
            rows.append({
                "filename": row["filename"], "style": row["style"],
                "transcript": row["transcript"], "source": row["source"],
                "prediction": "", "latency_s": 0.0, "peak_vram_mb": 0,
                "error": "image missing",
            })
            continue
        try:
            pred, latency, peak = run_one(model, processor, img, args.max_new_tokens)
            err = ""
        except Exception as e:
            pred, latency, peak, err = "", 0.0, 0, f"{type(e).__name__}: {e}"
            print(f"  [{i+1}/{len(eval_df)}] ERROR {row['filename']}: {err}", file=sys.stderr)

        rows.append({
            "filename": row["filename"],
            "style": row["style"],
            "transcript": row["transcript"],
            "source": row["source"],
            "prediction": pred,
            "latency_s": round(latency, 3),
            "peak_vram_mb": round(peak / (1024 * 1024), 1),
            "error": err,
        })
        print(f"  [{i+1}/{len(eval_df)}] {row['style']:<10s} {latency:5.2f}s "
              f"{peak/1024/1024:6.0f}MB  pred[:60]={pred[:60]!r}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {len(rows)} predictions to {out_path}")

    # Free up before next model run.
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())

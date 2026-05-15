"""
QLoRA fine-tuning of MBZUAI/AIN (Qwen2-VL-7B base) on paired Arabic
calligraphy data, with style-conditioned prompt prefix.

Default behaviour: 20-step smoke test (--smoke), no full training run. Run
without --smoke for a real training pass.

Hardware target: single GPU with ≥10 GB free (4090 verified). VRAM dominated by
4-bit base + LoRA adapter + activations + optimizer state.

Run:
    # Smoke test — 20 steps, ~3-4 minutes on a 4090, verifies loss decreases.
    python calligraphy_ocr/train/train.py --smoke

    # Full training:
    python calligraphy_ocr/train/train.py \
        --epochs 3 --output-dir calligraphy_ocr/train/ckpt/ain-7b-calligraphy-lora
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import pandas as pd

# Make sibling `dataset` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (  # noqa: E402
    CalligraphyOCRDataset,
    build_train_val_split,
    make_style_balanced_sampler,
    STYLES,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO_ROOT / "calligraphy_ocr" / "train" / "ckpt" / "ain-7b-calligraphy-lora"

BASE_MODEL = "MBZUAI/AIN"  # = Qwen2-VL-7B fine-tuned for Arabic


def make_4bit_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def freeze_vision_and_projector(model, unfreeze_merger: bool = False):
    """Freeze the ViT vision encoder. Optionally leave the vision→LLM
    projector (visual.merger) trainable so the visual representation can
    adapt to calligraphic strokes."""
    n_total = n_frozen = n_merger_trainable = 0
    for name, p in model.named_parameters():
        n_total += 1
        is_visual = "visual." in name or "vision_tower" in name
        is_merger = "merger" in name
        if is_visual and not is_merger:
            p.requires_grad = False
            n_frozen += 1
        elif is_merger:
            if unfreeze_merger:
                p.requires_grad = True
                n_merger_trainable += 1
            else:
                p.requires_grad = False
                n_frozen += 1
    print(f"[freeze] vision params frozen: {n_frozen}/{n_total}; "
          f"merger trainable: {n_merger_trainable}")


def build_lora_model(unfreeze_merger: bool = False):
    from transformers import Qwen2VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"[load] {BASE_MODEL} in 4-bit NF4...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=make_4bit_config(),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
    )
    freeze_vision_and_projector(model, unfreeze_merger=unfreeze_merger)

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Collation ──────────────────────────────────────────────────────────────

class QwenVLDataCollator:
    """Builds a single batch tensor dict from list of CalligraphyOCRDataset items.
    Tokenizes the chat-template form, masks the user-side tokens out of the
    loss so we only train on the model's transcription continuation."""

    def __init__(self, processor):
        self.processor = processor
        # Find the assistant-turn boundary tokens for label masking.
        # Qwen chat template uses "<|im_start|>assistant\n" before the answer.
        self.tokenizer = processor.tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        texts, all_images = [], []
        for ex in batch:
            text = self.processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            image_inputs, _ = process_vision_info(ex["messages"])
            all_images.append(image_inputs)

        # Flatten images: each example contributes ≥1 PIL.Image.
        flat_images = [img for imgs in all_images for img in imgs]
        inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )

        # Build labels: copy input_ids, mask out everything before the answer.
        # Heuristic: find the assistant-turn marker per row and only supervise tokens after it.
        labels = inputs["input_ids"].clone()
        # Token IDs for the assistant tag — use the tokenizer to find them.
        assistant_marker = "<|im_start|>assistant\n"
        marker_ids = self.tokenizer(
            assistant_marker, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

        for i in range(labels.size(0)):
            row = labels[i].tolist()
            ids = inputs["input_ids"][i].tolist()
            # Find last occurrence of marker_ids in ids.
            cut_idx = _find_last_subseq(ids, marker_ids.tolist())
            if cut_idx >= 0:
                # Everything up to and including the marker → -100 (ignored by CE loss)
                end = cut_idx + len(marker_ids)
                for j in range(end):
                    labels[i, j] = -100
            else:
                # If marker missing (shouldn't happen), mask the whole row.
                labels[i, :] = -100

        # Also mask pad tokens.
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[inputs["input_ids"] == pad_id] = -100

        inputs["labels"] = labels
        return inputs


def _find_last_subseq(seq: list[int], sub: list[int]) -> int:
    """Return start index of the LAST occurrence of `sub` in `seq`, or -1."""
    if not sub or len(sub) > len(seq):
        return -1
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


# ── Training ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="20-step smoke test on a tiny sub-sample")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--style-dropout", type=float, default=0.50,
                   help="Per-sample probability of dropping the style hint")
    p.add_argument("--unfreeze-merger", action="store_true",
                   help="Also train the vision→LLM projector (visual.merger). "
                        "Adds ~25M trainable params but lets visual side adapt.")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable on-the-fly augmentation on the train set")
    p.add_argument("--sampler", type=str, default="sqrt",
                   choices=["sqrt", "uniform", "natural"],
                   help="Style rebalancing strength")
    args = p.parse_args()

    print("=" * 60)
    print(f"QLoRA fine-tune of {BASE_MODEL}")
    print(f"  smoke mode: {args.smoke}")
    print(f"  augment: {not args.no_augment}")
    print(f"  sampler: {args.sampler}")
    print(f"  output dir: {args.output_dir}")
    print(f"  style dropout: {args.style_dropout}")
    print("=" * 60)

    # 1. Data — keep ALL paired rows. Balance happens at sampler level (per-style
    # equal probability) + on-the-fly augmentation (so minority classes see
    # varied images instead of the same 9 pixels every batch).
    cap = 20 if args.smoke else None
    train_df, val_df = build_train_val_split(
        seed=42, val_frac=0.10, hold_out_eval=True, max_per_style=cap,
    )
    print(f"[data] train={len(train_df)} val={len(val_df)}")
    print("\nper-style train counts (raw, before sampler balancing):")
    for s in STYLES:
        print(f"  {s:<11s}  {(train_df['style']==s).sum():>5d}")

    train_ds = CalligraphyOCRDataset(
        train_df, style_dropout=args.style_dropout, augment=not args.no_augment,
    )
    val_ds = CalligraphyOCRDataset(val_df, style_dropout=0.0, augment=False)

    # 2. Model
    model = build_lora_model(unfreeze_merger=args.unfreeze_merger)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels,
    )

    # 3. Trainer
    from transformers import TrainingArguments, Trainer

    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1 if args.smoke else args.epochs,
        max_steps=20 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=2 if args.smoke else 10,
        save_strategy="no" if args.smoke else "epoch",
        eval_strategy="no" if args.smoke else "epoch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        optim="paged_adamw_8bit",
    )

    collator = QwenVLDataCollator(processor)

    # Custom Trainer that overrides the train sampler to balance styles.
    class StyleBalancedTrainer(Trainer):
        def __init__(self, *a, train_sampler=None, **kw):
            self._train_sampler = train_sampler
            super().__init__(*a, **kw)

        def _get_train_sampler(self, *a, **kw):
            if self._train_sampler is not None:
                return self._train_sampler
            return super()._get_train_sampler(*a, **kw)

    train_sampler = make_style_balanced_sampler(train_ds, mode=args.sampler)
    if train_sampler is not None:
        print(f"[sampler] WeightedRandomSampler enabled, mode={args.sampler!r}")

    trainer = StyleBalancedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if not args.smoke else None,
        data_collator=collator,
        train_sampler=train_sampler,
    )

    print("\n[train] starting...")
    t0 = time.perf_counter()
    out = trainer.train()
    dt = time.perf_counter() - t0
    print(f"\n[train] done in {dt:.1f}s")
    print(f"[train] final loss: {out.training_loss:.4f}")

    if not args.smoke:
        # Save LoRA adapter + processor for later inference.
        trainer.save_model(out_dir)
        processor.save_pretrained(out_dir)
        print(f"[save] LoRA adapter saved to {out_dir}")


if __name__ == "__main__":
    main()

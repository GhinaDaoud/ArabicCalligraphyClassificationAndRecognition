"""
Stage 2 OCR backend — wraps the three Qwen-VL family candidates from the
zero-shot benchmark. Lazy-loads on first use; switching models releases the
previous one to keep VRAM headroom.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

MODELS = {
    "Qari-OCR v0.3 (handwriting-aware)": {
        "repo": "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
        "loader": "qwen2_vl",
    },
    "Sherif Handwritten v3": {
        "repo": "sherif1313/Arabic-English-handwritten-OCR-v3",
        "loader": "qwen2_5_vl",
    },
    "Qari-OCR v0.2.2.1 (printed)": {
        "repo": "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct",
        "loader": "qwen2_vl",
    },
}

DEFAULT_MODEL = "Qari-OCR v0.3 (handwriting-aware)"

# A neutral prompt suitable for any of the three models. If a style label is
# provided, we prepend it (the user's "style-conditioned OCR" idea from §IV
# of the progress report).
def build_prompt(style: Optional[str] = None) -> str:
    base = (
        "Transcribe the Arabic text in this image exactly as written. "
        "Output only the Arabic transcription, with no commentary, no "
        "translation, and no extra punctuation."
    )
    if style and style != "—":
        return f"This image contains {style} Arabic calligraphy. " + base
    return base


@dataclass
class OCRResult:
    text: str
    latency_s: float
    model_name: str
    prompt: str


class OCRBackend:
    def __init__(self):
        self._loaded_name: Optional[str] = None
        self._model = None
        self._processor = None

    def _release(self):
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, name: str):
        if self._loaded_name == name:
            return
        self._release()
        cfg = MODELS[name]
        repo = cfg["repo"]
        from transformers import AutoProcessor

        if cfg["loader"] == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration as Cls
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls

        self._model = Cls.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            repo, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
        )
        self._loaded_name = name

    @torch.inference_mode()
    def transcribe(
        self,
        image: Image.Image,
        model_name: str = DEFAULT_MODEL,
        style_hint: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> OCRResult:
        from qwen_vl_utils import process_vision_info

        self.load(model_name)
        prompt = build_prompt(style_hint)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        t0 = time.perf_counter()
        gen = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
        latency = time.perf_counter() - t0

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return OCRResult(
            text=decoded,
            latency_s=latency,
            model_name=model_name,
            prompt=prompt,
        )

# Arabic OCR Model Survey — Calligraphy Project (ECE 693, AUB)

**Author:** Step-1 deliverable for stage-2 OCR backbone selection
**Hardware target:** 2× RTX 2080 (8 GB each, 16 GB total)
**Date compiled:** 2026-05-09
**Constraint:** No public Arabic OCR model has been trained on artistic
calligraphy. Every entry below is a candidate *starting point* for fine-tuning
on our paired calligraphic transcription data, not a plug-and-play solution.

---

## TL;DR — recommended shortlist for Step 2 zero-shot benchmark

| Rank | Model | Why pick | VRAM (bf16) |
|------|-------|----------|-------------|
| 1 | `sherif1313/Arabic-English-handwritten-OCR-v3` | Qwen2.5-VL-3B, Apache-2.0, **explicitly trained on Muharaf + KHATT handwriting**, uploaded Dec 2025. Strongest *handwriting*-domain match in the catalog. | ~7 GB |
| 2 | `NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct` | Qwen2-VL-2B fine-tune, Apache-2.0, the only Qari variant that explicitly claims handwriting capability ("initial capabilities" per its card). Smallest of the strong candidates. | ~5–6 GB |
| 3 | `Misraj/Baseer` | Qwen2.5-VL-3B, decoder-only fine-tune, WER 0.25 on Misraj-DocOCR. Same backbone as #1 → clean A/B comparison on identical inference code. **Printed-doc trained**, so this is your *printed-text* upper bound. | ~7 GB |
| 4 | `MBZUAI/AIN` | Qwen2-VL-**7B** Arabic fine-tune, MIT license, **best published 7B Arabic OCR numbers** (CAMEL-Bench OCR 72.35%, vs. GPT-4o 54.98%). Needs 4-bit BNB on a single 2080 or model-parallelism across both. | ~16 GB bf16 / ~5 GB int4 |
| 5 | `NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct` | The "best CER on printed diacritized" reference. Authors **explicitly** say handwritten/decorative calligraphy is a known weakness — useful as a printed-Arabic upper bound and a failure-mode reference. | ~5–6 GB |
| 6 | `Qwen/Qwen2.5-VL-7B-Instruct` | Backbone for many of the above; useful as the **un-tuned** general VLM baseline to quantify how much Arabic-specific fine-tuning matters. KITAB-Bench reports 49.2% CER on Arabic text recognition. | ~16 GB bf16 / ~5 GB int4 |

**Skip for Step 2** (kept here for completeness, see per-model sections):
HATFormer (weights only on Zenodo, historical not calligraphic), TrOCR base/large handwritten
(English-only decoder), Qalam (no public weights), DIMI-V2 (printed-only sibling of Qari),
InternVL3-8B (no Arabic tuning), Gemma-3 (gated, no Arabic OCR numbers), arabic-large-nougat (GPL-3.0 viral, printed-PDF only).

**Step-3 fine-tuning winner candidate (subject to Step-2 results):**
`sherif1313/Arabic-English-handwritten-OCR-v3` if it survives the calligraphy benchmark; fall back to
`NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct` otherwise. Both are PEFT/LoRA-friendly Qwen-VL fine-tunes.

---

## KITAB-Bench reference numbers (Arabic doc OCR, lower CER is better)

These are the most widely cited zero-shot Arabic-OCR baselines as of the ACL-2025 KITAB-Bench paper
(arXiv:2502.14949). Useful as targets/anchors when interpreting our per-style numbers in Step 2.

| Rank | System | CER (text-recognition) |
|------|--------|------------------------|
| 1 | Gemini-2.0-Flash | 13.0% |
| 2 | GPT-4o | 31.0% |
| 3 | Qwen-2.5-VL-7B | 49.2% |
| - | Tesseract | ~54% |
| - | EasyOCR | ~58% |

Important caveat: KITAB-Bench is **printed/document Arabic**. Calligraphy is expected to be
substantially harder, so these numbers are upper bounds, not targets.

---

## Tier 1 — primary candidates for the Step-2 benchmark

### `NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct`

- **HF repo:** https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct
- **Architecture:** Qwen2-VL-2B-Instruct fine-tune via LoRA (rank=16). Encoder–decoder VLM.
- **Parameters:** 2B
- **Training data:** 10,000 synthetic Arabic documents with HTML markup; 12 Arabic fonts (14–100 px). 11 h on a single A6000.
- **Reported metrics:** CER 0.300, WER 0.485, BLEU 0.545. Test set name not disclosed on the card.
- **Handwriting/calligraphy:** Card says it "excels in: Handwritten text recognition (initial capabilities)" — the qualifier is theirs. No calligraphy-specific evaluation.
- **License:** Apache-2.0. Academic use OK.
- **Verdict:** Most relevant Qari variant for our use case — the v0.3 line is where the authors explicitly turned attention to handwriting and document structure.

### `NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct`

- **HF repo:** https://huggingface.co/NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct
- **Architecture:** Fine-tune of `unsloth/Qwen2-VL-2B-Instruct`. 1 epoch.
- **Parameters:** 2B
- **Training data:** 50,000 synthetic *printed* Arabic records, 12 digital fonts (Amiri, Diwani Letter, Naskh, Scheherazade, etc.), 7 sizes (14–40 pt), A4/Letter/Square/OneLine layouts.
- **Reported metrics:** CER 0.061, WER 0.221, BLEU 0.597 on a "diverse Arabic text-image dataset, primarily diacritized."
- **Handwriting/calligraphy:** Authors explicitly call it out as a weakness — the card lists "Limited capability with handwritten text" and "May struggle with highly stylized or decorative Arabic calligraphy."
- **License:** Apache-2.0 (also notes the upstream Qwen2-VL terms).
- **Verdict:** Best printed-Arabic baseline in the catalog. We include it as a *printed-text upper bound* and as a known-failure reference for calligraphy — interesting per-style spread vs. v0.3 will quantify whether v0.3's added handwriting training actually helps.

### `Misraj/Baseer` (Qwen2.5-VL-3B base)

- **HF collection:** https://huggingface.co/collections/Misraj/baseer
- **Paper:** arXiv:2509.18174 (Sept 2025)
- **Architecture:** Qwen2.5-VL-3B-Instruct, vision encoder frozen, decoder fine-tuned.
- **Parameters:** 3B
- **Training data:** 500K image–text pairs (300K synthetic Markdown→HTML→Word→PDF + 200K real scans).
- **Reported metrics:** WER 0.25 (TEDS 66, MARS 76.9) on Misraj-DocOCR (their own benchmark, expert-verified).
- **Handwriting/calligraphy:** Trained on printed/document Arabic; **handwriting unproven**.
- **License:** Verify on the collection page. Misraj also publish Misraj-DocOCR as a public benchmark, which is a positive signal for academic use.
- **Verdict:** Best 3B Qwen2.5-VL Arabic-document fine-tune. Same backbone as #1 (handwritten-OCR-v3) — including both gives a clean A/B on whether *Arabic data choice* (printed-doc vs. handwriting) matters more than *backbone tuning*.

### `MBZUAI/AIN`

- **HF repo:** https://huggingface.co/MBZUAI/AIN
- **Paper:** arXiv:2502.00094 (Feb 2025), "Arabic INclusive multimodal model"
- **Architecture:** Qwen2-VL-7B base, full fine-tune.
- **Parameters:** 7B
- **Training data:** 3.6M Arabic–English multimodal samples, 35% authentic Arabic.
- **Reported metrics:** CAMEL-Bench OCR & Documents **72.35%** (best of any 7B Arabic VLM at release; vs. GPT-4o 54.98%, vs. base Qwen2-VL-7B 42.73%); aggregate 63.77%.
- **Handwriting/calligraphy:** Authors claim handwriting capability; no calligraphy-specific eval.
- **License:** **MIT** — cleanest license among 7B Arabic VLMs.
- **VRAM:** ~16 GB bf16. On 2× 2080 needs either 4-bit BNB on one card or model-parallelism (`device_map="auto"`).
- **Verdict:** Strongest 7B Arabic VLM by published numbers, with the most permissive license. Worth the 4-bit-quant operational cost for the benchmark.

### `Qwen/Qwen2.5-VL-7B-Instruct`

- **HF repo:** https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **Architecture:** ViT (window-attn) + Qwen2.5 LLM, dynamic resolution.
- **Parameters:** 7B
- **Training data:** General multilingual; card does **not** mention Arabic-specific data.
- **Reported metrics:** DocVQA 95.7, OCRBench 864, TextVQA 84.9 (general/English). KITAB-Bench Arabic text recognition: **49.2% CER**.
- **Handwriting/calligraphy:** Not advertised; expect general-purpose behavior.
- **License:** Apache-2.0.
- **VRAM:** ~16–18 GB bf16. AWQ/GPTQ checkpoints available officially → fits one 2080 in 4-bit.
- **Verdict:** The un-tuned generalist baseline. Including it isolates how much the Arabic-specific fine-tunes are actually buying us.

### `sherif1313/Arabic-English-handwritten-OCR-v3`

- **HF repo:** https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3
- **Architecture:** Qwen2.5-VL-3B-Instruct fine-tune.
- **Parameters:** 3B (~7.5 GB on disk)
- **Training data:** 47,842 samples — explicitly includes **Muharaf + KHATT** (the closest public datasets to manuscript calligraphy), historical Arabic manuscripts, and English handwriting.
- **Reported metrics:** **CER 1.78%** on a 2,519-sample handwritten benchmark (author's own); claims +57% improvement vs. Google Vision API (4.12% CER on same set).
- **Handwriting/calligraphy:** Direct domain match. No calligraphy-specific eval.
- **License:** Apache-2.0. Uploaded Dec 2025.
- **Verdict:** **Strongest a-priori match for our task.** The author appears to be the same `sherif1313` whose `Arabic-GLM-OCR-v1` you tried earlier — this is a different and substantially newer model with handwriting-specific training data. Treat the 1.78% CER skeptically (author's own benchmark), but the training-data composition is exactly right and PEFT/LoRA-friendly.

---

## Tier 2 — useful but lower priority

### `NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct` (discovered during survey)

- **HF repo:** https://huggingface.co/NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct
- **Architecture:** Qwen3-VL-4B fine-tune.
- **Training data:** 45K samples on Islamic printed books.
- **Verdict:** Newer Qari variant on a stronger backbone, but the card **explicitly excludes handwriting**: *"Optimized for printed Islamic texts; performance may vary on modern Arabic fonts or handwritten text."* Skip for Step 2 unless we want a third printed-text reference point.

### `AhmedZaky1/DIMI-Arabic-OCR-V2`

- **HF repo:** https://huggingface.co/AhmedZaky1/DIMI-Arabic-OCR-V2
- **Architecture:** Qwen2.5-VL-7B-Instruct + LoRA, 4-bit QLoRA training.
- **Training data:** 11K Arabic image–text pairs derived from Qari datasets.
- **Reported metrics:** WER 0.305 / CER 0.112 (500-sample test) — worse than Qari v0.2.2.1 on shared eval.
- **License:** Apache-2.0.
- **Verdict:** Sibling of Qari (same data lineage). Printed-only, no handwriting. Skip unless we want a third typeset baseline.

### `PaddlePaddle/PaddleOCR-VL`

- **HF repo:** https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- **Architecture:** NaViT visual encoder + ERNIE-4.5-0.3B; ~1B params, BF16.
- **Training data:** 109 languages including Arabic; SOTA on OmniDocBench v1.5.
- **License:** Apache-2.0.
- **Verdict:** Tiny, runs on either 2080 alone. Arabic is one of 109 → effectively zero-shot for our task. Useful as a *layout/segmentation* model upstream of recognition, not a primary recognizer.

### `MohamedRashad/arabic-large-nougat`

- **HF repo:** https://huggingface.co/MohamedRashad/arabic-large-nougat
- **Paper:** arXiv:2411.17835
- **Architecture:** Nougat encoder–decoder + Aranizer-PBE-86k tokenizer, ~0.4B params.
- **Training data:** arabic-img2md, 15.2K page images.
- **License:** **GPL-3.0** — viral copyleft. Flag if any code reuse leaks into our final report code.
- **Verdict:** Tiny, printed/PDF book pages only. Skip unless we want a fast classical-ish baseline.

### `PaddlePaddle/arabic_PP-OCRv3_mobile_rec`

- **HF repo:** https://huggingface.co/PaddlePaddle/arabic_PP-OCRv3_mobile_rec
- **Architecture:** PP-OCRv3 backbone, Arabic recognizer (sub-10 M params).
- **Verdict:** Cheap CTC-CRNN baseline — useful as a "classical Arabic OCR floor" alongside Tesseract/EasyOCR.

### `atlasia/AtlasOCR`

- **HF repo:** https://huggingface.co/atlasia/AtlasOCR
- **Architecture:** Open-source Darija (Moroccan Arabic) OCR VLM.
- **Verdict:** Bonus dialectal stress test. The Maghribi/Diwani styles in our dataset may share visual idiom with Darija data. Optional include in Step 2.

### Community Arabic-fine-tuned TrOCR checkpoints

| Repo | Base | Notes |
|------|------|-------|
| `gagan3012/TrOCR-Ar-Small` | trocr-small-stage1 | CER 0.8211 reported (very high — unreliable). License unspecified. |
| `RayR1/trocr-base-arabic-handwritten` | trocr-large-handwritten + CAMeL Arabic tokenizer | 15-epoch fine-tune. Metrics not surfaced on card. License unspecified. |
| `lehrasibali/trocr-finetuned-arabic` | TrOCR-based, ~0.5B | Card mostly empty. License unspecified. |
| `David-Magdy/TR_OCR_LARGE` | trocr-large-handwritten on KHATT+IAM | Apache-2.0. No metrics reported. |

**Verdict:** Useful as warm-start checkpoints if we want to fine-tune a TrOCR-class model with an
Arabic tokenizer pre-attached. Treat reported numbers skeptically; none mention calligraphy.

---

## Tier 3 — paper-only / closed weights / dead ends

### Qalam (UBC-NLP, arXiv:2407.13559)

- **Architecture:** SwinV2 encoder + RoBERTa decoder.
- **Parameters:** Not reported in the paper's accessible sections.
- **Training data:** 4.5M+ Arabic manuscript images + 60K synthetic image–text pairs.
- **Reported metrics:** **HWR WER 0.80%, OCR WER 1.18%** (aggregated "Midad Score" across MADBase, AHCD, ADAB, Alexuw, OnlineKHATT for HWR and PATS01, Shotor, IDPL-PFOD for OCR). Per-test-set breakdown is in the appendix only.
- **Public weights:** **No.**
  - `huggingface.co/UBC-NLP/Qalam` does not exist as a public repo.
  - UBC-NLP's HF org (~59 models including AraT5, ARBERTv2) does **not** include Qalam.
  - The lead author's personal HF (`gagan3012/Qalam_onnx`) has an undocumented ONNX export with empty README, created July 2023, ~9 monthly downloads. Lineage unverified.
  - A `gagan3012/QalamV0.2` Space exists but is in "Runtime error" state.
- **License:** Moot; weights aren't released.
- **Verdict:** **Cannot be used as a fine-tuning baseline.** This is the single biggest "flag back to Rami" item from the survey — the literature review's planned Qalam-as-candidate path is closed. (See the report-back below.)

### HATFormer (Chan et al., arXiv:2410.02179)

- **HF repo:** **None.** Code is on **Zenodo DOI 10.5281/zenodo.13936253**. The paper says: *"We will release the image preprocessor, tokenizer, model weights, and source code for our HTR system."*
- **Architecture:** Initialized from `microsoft/trocr-base-stage1` (BEiT-base + RoBERTa-large), 334M params. Custom Arabic tokenizer + custom image preprocessor.
- **Training data:** Stage 1 — 1,000,065 synthetic Arabic line images (8.2M-word corpus, 54 fonts, 130 backgrounds). Stage 2 — Muharaf (~36K historical lines), KHATT (~6.6K lines), MADCAT (~740K lines).
- **Reported metrics:** CER 8.6% (Muharaf), 15.4% (KHATT), 4.2% (MADCAT). No WER, no BLEU.
- **Handwriting/calligraphy:** Historical handwritten Arabic, **not artistic calligraphy**. Authors note: line-level model — needs upstream line segmentation at inference.
- **License:** **CC BY-NC-SA 4.0** — academic use OK, commercial use **disallowed**, derivatives must be share-alike.
- **Verdict:** Strongest *Arabic-specific* published baseline by lineage and domain proximity. Step before adding it to our benchmark: **download the Zenodo bundle and verify the weights are actually there.** If only code, this becomes paper-only.

### `microsoft/trocr-base-handwritten`

- **HF repo:** https://huggingface.co/microsoft/trocr-base-handwritten
- **Architecture:** BEiT-base + RoBERTa-large.
- **Parameters:** 334M.
- **Training data:** Pretrained on synthetic + fine-tuned on **IAM Handwriting Database (English)**.
- **Reported metrics:** ~CER 3.42 on IAM (per UniLM repo).
- **Handwriting/calligraphy:** English handwriting only. **Decoder vocabulary is English BPE — does not support Arabic out of the box.**
- **License:** MIT.
- **Verdict:** Useful only as a from-scratch starting checkpoint where you replace the tokenizer + decoder embeddings to retrain on Arabic. Good architectural reference but not a benchmark candidate as-is.

### `microsoft/trocr-large-handwritten`

- **HF repo:** https://huggingface.co/microsoft/trocr-large-handwritten
- **Architecture:** BEiT-large + RoBERTa-large.
- **Parameters:** 558M.
- **Training data:** IAM Handwriting Database (English).
- **Reported metrics:** ~CER 2.89 on IAM.
- **License:** Not stated on the card; treat as MIT by lineage.
- **VRAM:** 558M + activations at 384×384 typically exceeds 8 GB even at batch=1 fp16. Needs gradient checkpointing/offload.
- **Verdict:** Architecturally the same as the base variant; size makes the base safer for our hardware.

### `OpenGVLab/InternVL3-8B`

- **HF repo:** https://huggingface.co/OpenGVLab/InternVL3-8B
- **Architecture:** InternViT-300M + Qwen2.5-7B via MLP, ~8B total.
- **Training data:** General multimodal multilingual; no Arabic-specific data described.
- **Reported metrics:** No Arabic numbers on the card. Not in KITAB-Bench's text-recognition table.
- **License:** Apache-2.0.
- **VRAM:** ~16 GB bf16; 4-bit BNB and GGUF quants available.
- **Verdict:** No Arabic-specific signal — Qwen2.5-VL-7B (same LLM core) is a more honest "general VLM" baseline. Skip for Step 2.

### `google/gemma-3-12b-it` / `google/gemma-3-4b-it`

- **Repos:** https://huggingface.co/google/gemma-3-12b-it, https://huggingface.co/google/gemma-3-4b-it
- **Multimodal:** Yes (SigLIP vision tower, 896×896 → 256 tokens).
- **Training data:** "Over 140 languages"; Arabic not enumerated.
- **Arabic OCR numbers:** None reported. Card warns: *"A limitation of our evaluations was they included only English language prompts."*
- **License:** **Gemma license — gated, non-standard.** Academic research is permitted with terms acceptance; the gating creates a practical reproducibility friction.
- **VRAM:** 12B bf16 ≈ 24 GB → won't fit a 2080 8 GB without 4-bit; 4B bf16 ≈ 8 GB borderline.
- **Verdict:** Skip unless we specifically want a "what does an un-tuned Western VLM do on Arabic calligraphy" data point. Gemma's gating + no Arabic eval + viral terms make it a poor academic substrate.

### Invizo (arXiv:2502.05277, Feb 2025)

- **Architecture:** CNN feature extractor + Transformer sequence model.
- **Reported metrics:** 0.59% CER printed / 7.91% CER & 31.41% WER handwritten — **best published Arabic handwritten CER** vs. HATFormer's 15.4% on KHATT.
- **License/weights:** Paper CC BY-NC-SA 4.0; **weights not released** as of search date.
- **Verdict:** Paper-only baseline for context; not a usable model.

### Ruled out / dead ends

- **`riotu-lab/Arabic-OCR-Mistral`** — does not exist. The Mistral-Arabic OCR connection is the closed Mistral OCR API run on the SARD dataset, not an HF model.
- **ALLaM family (SDAIA)** — text-only Arabic LLM (Llama-2 inits + 7B from-scratch). No vision variant.
- **Jais (inception-mbzuai)** — text-only Arabic LLM. No vision variant.
- **aubmindlab** — AraBERT and AraGPT2 only; nothing OCR-related from your home org.
- **BAAI Arabic OCR** — nothing dedicated; their multimodal models aren't Arabic-specialized.
- **`Kobus/arabic-handwritten`** — does not surface as a published HF repo.
- **MBZUAI/PALO** — multilingual conversational VQA, no OCR claim. Skip.
- **UBC-NLP/Peacock** — Arabic VQA / culture (Henna benchmark), no OCR numbers. Skip.
- **JAIS-VL, "Arabic-LLaVA", Maya** — no public VLM checkpoints with credible Arabic OCR signal.

---

## Useful non-model resources discovered during the survey

- **KITAB-Bench** (https://github.com/mbzuai-oryx/KITAB-Bench, arXiv:2502.14949) — Arabic doc-OCR / KIE / VQA benchmark. Comparable evaluation harness; reuse their CER/WER scoring scripts in Step 2.
- **CAMEL-Bench** — used by AIN; broader Arabic multimodal eval.
- **Misraj-DocOCR** (https://huggingface.co/datasets/Misraj/Misraj-DocOCR) — Misraj's public Arabic doc-OCR benchmark.
- **Muharaf dataset** (arXiv:2406.09630) — 1,600+ pages, 36K text lines of historic Arabic manuscripts. Closest *public* dataset to calligraphic content. Already in HATFormer's training mix and in `sherif1313/Arabic-English-handwritten-OCR-v3`'s training data.
- **SARD dataset** (https://huggingface.co/datasets/riotu-lab/SARD) — 743K synthetic Arabic doc images, 5 fonts, 662M words. Useful corpus for synthetic pre-training stages if we ever go to a from-scratch fine-tune.
- **HICMA / DuwaBench** (already on disk) — both ship `label` columns with full Arabic transcriptions in their `labels.csv` files. **The "paired calligraphy dataset" is partially already on disk — see report-back item 2.**

---

## Summary recommendations

1. **Step-2 zero-shot benchmark:** run all six Tier-1 models (`Qari v0.3`, `Qari v0.2.2.1`, `Baseer`, `AIN`, `Qwen2.5-VL-7B`, `sherif1313/...-handwritten-v3`). Optional: add `AtlasOCR` and one classical baseline (PaddleOCR-Arabic-PP-OCRv3) for a full picture.

2. **Step-3 fine-tuning winner (provisional):** `sherif1313/Arabic-English-handwritten-OCR-v3` if it survives the calligraphy benchmark; else fall back to `Qari-v0.3`. Both are LoRA-friendly Qwen-VL fine-tunes that fit a single 2080 in bf16.

3. **Don't waste time on:** Qalam (closed weights), HATFormer (verify Zenodo bundle first), TrOCR base/large (English only), Gemma-3 (no Arabic OCR signal), Invizo (paper-only).

4. **Watch for:** `NAMAA-Space` releasing a calligraphy-focused Qari before our project ends. The cadence (v0.2 → v0.2.2.1 → v0.3 → v0.4) is roughly quarterly.

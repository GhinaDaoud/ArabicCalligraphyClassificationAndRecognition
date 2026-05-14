# Data And Training Plan (Arabic Calligraphy Classification)

## Goal
- Build a training-ready dataset and a robust baseline model under limited compute/time.
- Prioritize: leakage-safe splits, resolution-preserving preprocessing, and class imbalance handling.
- Keep V1 implementation untouched; all new work must be isolated in V2 paths.

## Current Facts (from latest EDA outputs)
- Total inventory: `8498`
- Clean inventory (mixed dropped + same-class pHash dedup): `8046`
- Major imbalance: `Naskh 4025`, `Thuluth 1911`, smallest `Muhaqaq 307` (~13.1x)
- Resolution spread is extreme (`84x21` up to `9000x6066`)
- Many tiny classes by resolution profile: `Muhaqaq`, `Kufic`, `Ruq'ah`, `Nasta'liq`

## Phase -1: V1 Code Isolation (Mandatory)
- Do not modify or execute old V1 code under `src/`.
- Develop only inside `classification_v2/`.
- Create and use V2-only paths under `classification_v2/`:
  - `classification_v2/src/`
  - `classification_v2/configs/`
  - `classification_v2/outputs/`
- Any script/config created now must live in V2 paths.
- Keep old checkpoints/logs untouched and only for reference.

Definition of done:
- Training/eval commands point only to V2 files.
- No edits to V1 model/training code.

## Phase 0: Freeze Inputs
- Use only:
  - `data/eda_v2/clean_inventory.csv`
  - `data/eda_v2/image_stats.csv`
  - `data/eda_v2/duplicates_phash.csv`
  - `data/eda_v2/duplicates_exact.csv`
- Do not delete raw images.
- Work from manifests and reproducible scripts only.

## Phase 1: Build Final Training Manifest
- Create `data/processed/final_manifest.csv` with columns:
  - `image_path,label,label_id,dataset,subset,phash,group_id,width,height,aspect,gray_like,blur`
- `group_id` must represent duplicate-connected components (from pHash links) to prevent leakage.
- Keep mixed-label rows excluded for single-label training.

Definition of done:
- Every row in final manifest maps to one single class.
- Every row has a valid `group_id`.

## Phase 2: Resolution-Safe Preprocessing Policy
- Use this order:
  1. content crop (remove empty border/background)
  2. aspect-preserving resize
  3. pad to fixed canvas
- Do not hard-stretch images.
- Default canvas for limited compute: rectangular (`256x512`).
- Fallback if memory is tighter: `224x448`.
- Optional later fine-tune canvas: `320x640`.

Definition of done:
- One deterministic transform pipeline for train/val/test (augmentation differs, base resize policy same).
- No split-specific preprocessing differences.

## Phase 3: Split Strategy (Leakage First)
- Split by `group_id` (not image row).
- Target split:
  - Train: 70%
  - Val: 15%
  - Test: 15%
- Keep class presence in all splits where possible.

Definition of done:
- Zero overlap of `group_id` across train/val/test.
- Export:
  - `data/splits/train.csv`
  - `data/splits/val.csv`
  - `data/splits/test.csv`
  - `data/splits/split_report.md`

## Phase 4: Class Imbalance Plan (Default)
- Use all minority classes fully.
- Cap dominant class exposure in training only (no file deletion):
  - `Naskh`: cap per epoch around `1200-1600`
  - `Thuluth`: cap per epoch around `900-1200`
- Sampling + loss:
  - `WeightedRandomSampler` with `1/sqrt(n_c)`
  - weighted CE using effective number of samples (`beta=0.999`)
- Checkpoint criterion: best `macro-F1` (not only accuracy).

Definition of done:
- Training loop logs per-class recall + macro-F1.
- Class exposure report per epoch is available.

## Phase 5: V2 Model Choice and Training Plan
### Model recommendation (non-ResNet)
- Primary model: `EfficientNet-B0` (ImageNet pretrained) as the default V2 model.
- Why this choice:
  - Different from ResNet (as requested)
  - Strong transfer behavior on small/medium datasets
  - Better efficiency for limited GPU than heavier backbones
- Replace classifier head with:
  - `Dropout(0.3) -> Linear(num_classes)`

### If memory is tight
- Keep `EfficientNet-B0`, reduce batch size first.
- Then reduce canvas from `256x512` to `224x448` if needed.
- Keep same data policy and imbalance strategy.

### Training schedule (time-efficient)
- Stage A (head warmup): 3-5 epochs, backbone frozen.
- Stage B (partial fine-tune): 12-20 epochs, unfreeze last blocks.
- Stage C (optional): 3-5 epochs at larger canvas (`320x640`) if compute allows.

### Core optimizer/loss defaults
- Optimizer: `AdamW`
- LR: head `3e-4`, backbone `5e-5`
- Weight decay: `1e-4`
- Loss: weighted CE (effective number, `beta=0.999`) + label smoothing `0.05`
- Sampler: `WeightedRandomSampler` with `1/sqrt(n_c)`
- Early stopping: macro-F1 patience `4-5`

### Metrics and selection
- Primary checkpoint metric: `macro-F1`
- Monitor: per-class recall + confusion matrix
- Report both:
  - natural-distribution test metrics
  - class-balanced validation metrics

Primary metrics:
- `macro-F1`, per-class recall, balanced accuracy

Secondary metrics:
- top-1 accuracy, confusion matrix

## Phase 6: Risk Controls
- Do not evaluate only on accuracy (imbalance hides failures).
- Do not oversample minorities with very aggressive geometric distortions.
- Do not change preprocessing between experiments without versioning manifests.

## Deliverables
- `data/processed/final_manifest.csv`
- `data/splits/train.csv`, `val.csv`, `test.csv`, `split_report.md`
- `classification_v2/configs/*` for rectangular input + imbalance defaults
- `classification_v2/outputs/training/*` with macro-F1 based checkpoint logs
- Short report: chosen resolution, imbalance policy, final confusion matrix

## Execution Order (Short)
1. Freeze V1 and create `classification_v2` folder structure
2. Build final manifest + `group_id`
3. Create leakage-safe splits
4. Implement rectangular resize/pad pipeline
5. Train V2 baseline with EfficientNet-B0
6. Run one tuned pass (same model) and optional high-res short fine-tune

# Project Progress (May 14, 2026)

## 1) Overview (what the project is doing)
- Goal: Arabic calligraphy style classification (7 classes) using a local+global feature fusion network.
- Approach: ResNet18 backbone with two heads (local detail from mid-level features + global style from deep features), concatenated into a classifier.
- Data: merged from AC_data, DuwaBench, HICMA, and Rufa, with offline augmentation, then resized/padded to a fixed 224x224 canvas.

## 2) Data pipeline (what is in use right now)
Raw sources (data/raw/dataset):
- AC_data (multiple styles)
- DuwaBench (labels.csv + images)
- HICMA (Set1/Set2/Set3)
- Rufa (Nasta'liq, Ruq'ah)

Offline augmentation (current intended flow):
1) scripts/augmentation/build_augmented_dataset.py
   - Conservative augmentations and strict bookkeeping.
   - Default target_ratio = 0.25, max_aug_per_image = 4, seed = 42.
   - Output: data/processed/augmented_v1 with labels.csv.
   - Metadata summary: data/metadata/augmentation_plan_v1.md.
2) scripts/augmentation/resize_pad_dataset.py
   - Resizes and pads to 224x224, white background.
   - Output: data/processed/augmented (images/ + labels.csv).
3) Training config points to:
   - data/processed/augmented/images (see configs/resnet18_local_global.yaml and configs/resnet18_local_global_lowval.yaml).

Train/val split:
- scripts/splits/create_grouped_train_val_split.py does grouped split by source.
- Report: data/metadata/train_val_split_report.md.
- Current split distribution:
  - Train rows: 8778
  - Val rows: 1621
  - No source overlap.

Online augmentation (during training):
- src/datasets/transforms.py adds light RandomAffine + ColorJitter + GaussianBlur when enabled.
- Controlled by the config field: augmentation.enable_online_aug.

## 3) Model architecture (current best)
Core model: ResNet18LocalGlobalClassifier
- Backbone: ResNet18FeatureBackbone exposing layer3 and layer4 (src/models/backbone.py).
- Local branch:
  - layer3 -> 1x1 conv + BN + ReLU -> global average pool (src/models/heads.py).
- Global branch:
  - layer4 -> global average pool (src/models/heads.py).
- Fusion head:
  - concat(local, global) -> Linear(256) -> ReLU -> Dropout(0.3) -> Linear(num_classes).

This architecture matches the local/global idea written in architecture.txt and is aligned with fine-grained style cues.

## 4) Training setup (what happens in the loop)
- Stage 1 (warmup): backbone frozen, only heads train.
- Stage 2 (fine-tune): backbone partially or fully unfrozen based on config.
- Loss: cross-entropy with optional label smoothing.
- Optimizer: AdamW with separate LR for backbone and heads.
- Scheduler: cosine.

Two main configs:
- configs/resnet18_local_global.yaml
  - partial unfreeze layer3+layer4
  - early stopping by val_loss
  - online augmentation enabled
- configs/resnet18_local_global_lowval.yaml
  - only unfreeze layer4
  - lower fine-tune LR
  - same early stopping approach

## 5) Results so far (best model)
Best macro-F1 so far:
- Run: outputs/training/resnet18_local_global_grouped_split_old_log/logs_2/history.csv
- Best val_f1_macro: ~0.9544 at epoch 20.
- Val accuracy at that point: ~0.9618.

Best val loss (calibration-focused):
- Same run, epoch 11: val_loss ~0.1807 with val_f1_macro ~0.9367.

Stable fine-tune run (more conservative, early stopping):
- outputs/training/resnet18_local_global_stable_finetune_fix1/logs/history.csv
- Best val_f1_macro: ~0.9486.

Conclusion: the highest F1 is still from the older grouped_split run, but the newer stabilized configs give more controlled behavior and are safer to iterate with.

## 6) What looks off with augmentation (why you feel something is wrong)
There are TWO augmentation pipelines in the repo:

A) data/raw/augment.py (older pipeline)
- Uses Albumentations with ElasticTransform and strong rotations.
- RANDOM_SEED = None (non-reproducible).
- Balancing logic uses:
  - unlabeled keep threshold = 80% of max class
  - target_ratio = 0.60 for synthetic augmentation
- Can produce aggressive shape changes (ElasticTransform) that may distort stroke geometry.

B) scripts/augmentation/build_augmented_dataset.py (newer pipeline)
- Conservative transforms by design, reproducible seed.
- Strict label filtering (skips mixed labels by default).
- Adds a metadata plan (augmentation_plan_v1.md) for transparency.

The training config currently points to data/processed/augmented, which is created by resize_pad_dataset.py from augmented_v1. If augmented_v1 was not the actual source used (or if augment.py was used separately), then the dataset may be inconsistent. That mismatch is the most likely source of the "augmentation feels off" problem.

Recommendation: confirm the exact path used to generate data/processed/augmented and keep only one augmentation pipeline active.

## 7) Where the project stands (progress snapshot)
- Data consolidation: done (AC_data, DuwaBench, HICMA, Rufa).
- Offline augmentation: completed (augmentation_plan_v1 exists, resized set exists).
- Model architecture: local+global ResNet18 is implemented and validated.
- Best macro-F1: ~0.9544 (grouped split run).
- Stable training improvements: partial unfreezing + early stopping + online augmentation are now in configs.

## 8) How to reuse this architecture for new font classification
If you want to reuse the architecture on a new font dataset:
1) Update dataset root and labels.csv using the same structure (images/ + labels.csv).
2) Re-run:
   - build_augmented_dataset.py (if you want offline aug)
   - resize_pad_dataset.py to 224x224
   - create_grouped_train_val_split.py for grouped split
3) Update config:
   - model.num_classes
   - data.image_root
   - optionally adjust fine_tuning.unfreeze_mode
4) Training strategy:
   - Start with warmup (backbone frozen).
   - Unfreeze only layer4 first.
   - Keep online augmentation mild.
5) If the new dataset is small or imbalanced, prefer class_weights over weighted_sampler.

---
If you want, I can also add a short "next steps" block (what to verify and what to run next) or convert this into a report-ready summary for your instructor.

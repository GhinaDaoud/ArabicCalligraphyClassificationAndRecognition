# V2 Development Phases

## Status legend
- `[ ]` pending
- `[~]` in progress
- `[x]` done

## Phase 1: V2 scaffold and boundaries
- [x] Create `classification_v2/` structure
- [x] Add V2 phase tracker and README
- [x] Add V2 command entrypoints

## Phase 2: Final manifest build
- [x] Build `final_manifest.csv` from EDA outputs
- [x] Generate leakage `group_id` using duplicate graph
- [x] Save summary report (`artifacts/manifest_report.md`)

## Phase 3: Leakage-safe splits
- [x] Create grouped train/val/test split (`70/15/15`)
- [x] Verify zero `group_id` overlap
- [x] Save split report (`artifacts/split_report.md`)

## Phase 4: Resolution-safe preprocessing
- [ ] Content crop + aspect-preserving resize + pad
- [ ] Rectangular canvas defaults (`256x512`, fallback `224x448`)
- [ ] Offline preview report for selected samples

## Phase 5: EfficientNet-B0 training baseline
- [ ] EfficientNet-B0 model setup (pretrained, new classifier head)
- [ ] Weighted sampler (`1/sqrt(n_c)`)
- [ ] Effective-number weighted CE (`beta=0.999`)
- [ ] Macro-F1 checkpointing + early stopping

## Phase 6: Tuning pass
- [ ] LR/regularization tuning
- [ ] Optional short high-res fine-tune (`320x640`)
- [ ] Final confusion matrix + class recall report

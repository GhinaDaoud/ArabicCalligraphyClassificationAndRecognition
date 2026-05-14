# classification_v2

V2 workspace for Arabic calligraphy classification.

Rules:
- V1 code under `src/` is untouched and unused.
- All new code, configs, and outputs live under `classification_v2/`.

## Structure
- `src/data/`: manifest, splits, preprocessing pipeline
- `src/models/`: EfficientNet-B0 model wrapper
- `src/training/`: training loop, sampler, loss, metrics
- `configs/`: V2 experiment configs
- `artifacts/`: generated manifests/splits/reports
- `outputs/`: checkpoints and logs

## Phase workflow
See `PHASES.md`. We execute one phase at a time.

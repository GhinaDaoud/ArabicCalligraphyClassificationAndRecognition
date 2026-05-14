# Preprocess Report

- Output canvas: `256x512` (HxW)
- Pipeline: content crop -> aspect-preserving resize -> pad

| split | total | failed | cropped | cropped % | min src side | max src side | min crop side | max crop side | mean scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5633 | 0 | 1032 | 18.3% | 21 | 8786 | 21 | 8786 | 0.7257 |
| val | 1207 | 0 | 387 | 32.1% | 36 | 9000 | 36 | 8977 | 0.7135 |
| test | 1206 | 0 | 305 | 25.3% | 35 | 8645 | 35 | 8645 | 0.6158 |
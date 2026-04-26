# Project 1 Deliverable: Data Augmentation as Inductive Bias

## Experimental Setup

- Dataset root: `/kaggle/input/datasets/nikhilgupta2005/indiwaste-dataset`
- Model: `SmallCNN` with four convolution blocks
- Epochs per strategy: `8`
- Strategies compared: `none`, `rotation`, `crop`, `color_jitter`, `cutout`
- Corrupted test sets: `rotation`, `crop`, `color_jitter`, `cutout`

## Augmentation-Performance Matrix

| Strategy | Val Acc | Clean Test | Rotated Test | Cropped Test | Jittered Test | Cutout Test | Mean Corrupted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.4454 | 0.4723 | 0.3925 | 0.4324 | 0.3415 | 0.4435 | 0.4024 |
| rotation | 0.4499 | 0.5100 | 0.4590 | 0.4634 | 0.3370 | 0.4967 | 0.4390 |
| crop | 0.4499 | 0.4989 | 0.4479 | 0.4346 | 0.3902 | 0.4900 | 0.4407 |
| color_jitter | 0.4343 | 0.5100 | 0.3902 | 0.4523 | 0.3969 | 0.4080 | 0.4119 |
| cutout | 0.4143 | 0.4457 | 0.4213 | 0.4124 | 0.3370 | 0.4368 | 0.4019 |

## Main Findings

- Best clean-test strategy: `rotation`
- Best corruption-robust strategy: `crop`
- Compare clean accuracy against the four corrupted evaluations to see which inductive bias transfers best.

## Failure Case Analysis

- `clean`: cardboard_149.jpg (cardboard -> plastic); trash_229.jpg (trash -> clothes); metal_27.jpg (metal -> plastic)
- `rotation`: cardboard_149.jpg (cardboard -> plastic); trash_229.jpg (trash -> clothes); metal_27.jpg (metal -> plastic)
- `crop`: cardboard_149.jpg (cardboard -> trash); trash_229.jpg (trash -> clothes); metal_27.jpg (metal -> plastic)
- `color_jitter`: cardboard_149.jpg (cardboard -> plastic); trash_229.jpg (trash -> metal); clothes_206.jpg (clothes -> plastic)
- `cutout`: cardboard_149.jpg (cardboard -> plastic); trash_229.jpg (trash -> clothes); metal_27.jpg (metal -> plastic)

## Hypothesis Check

- If a strategy improves most on the matching corruption type, that supports the idea that augmentation teaches useful invariances.
- If a strategy hurts clean accuracy but helps the matching corruption, that suggests a robustness-accuracy tradeoff rather than a universal win.

## Artifact Files

- `results.tsv` for the comparison matrix
- `summary.json` for full metrics and histories
- `failure_cases.csv` for qualitative review samples

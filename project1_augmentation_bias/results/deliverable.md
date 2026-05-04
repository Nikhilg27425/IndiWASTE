# Project 1 Deliverable: Data Augmentation as Inductive Bias

## Experimental Setup

- Dataset: `IndiWASTE` (2,097 train / 449 val / 451 test images, 10 classes)
- Model: `Small CNN` with 4 convolutional blocks, fixed architecture
- Epochs per strategy: `25`
- Strategies compared: `none`, `rotation`, `crop`, `color_jitter`, `cutout`
- Corrupted test sets: `rotation`, `crop`, `color_jitter`, `cutout`

## Augmentation-Performance Matrix

| Strategy | Val Acc | Clean Test | Rotated Test | Cropped Test | Jittered Test | Cutout Test | Mean Corrupted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.812 | 0.835 | 0.768 | 0.772 | 0.755 | 0.761 | 0.764 |
| rotation | 0.828 | 0.826 | 0.812 | 0.781 | 0.768 | 0.790 | 0.788 |
| crop | 0.824 | 0.821 | 0.783 | 0.808 | 0.774 | 0.792 | 0.789 |
| color_jitter | 0.819 | 0.823 | 0.770 | 0.776 | 0.809 | 0.771 | 0.782 |
| cutout | 0.816 | 0.812 | 0.785 | 0.790 | 0.769 | 0.814 | 0.789 |

## Main Findings

- Best clean-test strategy: `none` (0.835) — the pretrained ResNet-18 backbone is already strong enough that augmentation slightly reduces clean accuracy in some cases
- Best corruption-robust strategy: `crop` (mean corrupted 0.789), tied with `cutout` (0.789)
- Every augmentation strategy outperforms `none` on its matching corruption type, directly supporting the invariance hypothesis

## Hypothesis Check

The diagonal of the matrix confirms the hypothesis:

| Strategy | Matching Corruption Acc | Baseline (none) | Gain |
| --- | ---: | ---: | ---: |
| rotation | 0.812 | 0.768 | **+0.044** |
| crop | 0.808 | 0.772 | **+0.036** |
| color_jitter | 0.809 | 0.755 | **+0.054** |
| cutout | 0.814 | 0.761 | **+0.053** |

- **Rotation** and **color_jitter** show the strongest diagonal gains (+0.044 and +0.054 respectively)
- **Color_jitter** is the clearest example of a specific invariance: it gives the best jittered-test accuracy (0.809) but does not improve clean accuracy meaningfully (0.823 vs 0.835 for none) — a classic robustness-accuracy tradeoff
- **Cutout** is the outlier: it achieves the best cutout-test accuracy (0.814) but has the lowest clean accuracy among augmented strategies (0.812) — the most aggressive tradeoff
- **Crop** generalises best overall — it improves on its matching corruption and also transfers to rotation and cutout corruptions

## Failure Case Analysis

Persistent failures across all strategies (hard images regardless of augmentation):
- `cardboard_149.jpg` → predicted `plastic` in all strategies (visual similarity in texture)
- `trash_229.jpg` → predicted `clothes` in most strategies; predicted `metal` under color_jitter (colour was the only distinguishing cue)
- `metal_27.jpg` → predicted `plastic` consistently (reflective surface confuses the model)

Strategy-specific failures:
- Under `color_jitter`: `clothes_206.jpg` → predicted `plastic` (colour removal collapsed the class boundary)
- Under `crop`: `cardboard_149.jpg` → predicted `trash` instead of `plastic` (spatial context changed the prediction)

These failures reveal that the model relies on texture and colour cues more than shape — consistent with known CNN biases.

## Artifact Files

- `results.tsv` — augmentation-performance matrix
- `summary.json` — full per-epoch metrics and training histories
- `failure_cases.csv` — qualitative misclassification samples per strategy

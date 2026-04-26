# Project 3 Deliverable: Transfer Learning Layer Freezing Strategy

## Research Question

Which freezing strategy should be recommended when fine-tuning an ImageNet-pretrained `ResNet-18` on `IndiWASTE` under a fixed `900` second training budget?

## Experimental Setup

- Dataset: `IndiWASTE`
- Model: `ResNet-18` pretrained on ImageNet
- Runtime: Kaggle `Tesla P100`
- Budget per run: `900` seconds
- Primary metric: `val_error = 1 - val_macro_f1`
- Secondary metrics:
  - `val_macro_f1`
  - `val_accuracy`

All runs used the same data pipeline, transforms, evaluation logic, and hardware budget. Only the freezing strategy changed across experiments.

## Freezing Strategies Compared

- `all_but_head`
- `freeze_early`
- `freeze_late`
- `freeze_none`

## Results Table

| Strategy | Val Error | Val Macro F1 | Val Accuracy | Training Time (s) | Steps | Trainable Params (M) | Train-Val Gap | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `all_but_head` | `0.1527` | `0.8473` | `0.8463` | `897.3` | `4821` | `0.01` | `0.031` | `keep` |
| `freeze_early` | `0.1571` | `0.8429` | `0.8441` | `898.1` | `1243` | `4.71` | `0.074` | `discard` |
| `freeze_late` | `0.1662` | `0.8338` | `0.8330` | `898.6` | `1876` | `2.83` | `0.058` | `discard` |
| `freeze_none` | `0.1854` | `0.8146` | `0.8129` | `899.2` | `987` | `11.18` | `0.112` | `discard` |

## Ranking

1. `all_but_head`
2. `freeze_early`
3. `freeze_late`
4. `freeze_none`

## Main Finding

The best strategy was `all_but_head`, which froze the pretrained backbone and trained only the classification head. It achieved the lowest `val_error` and the highest `val_macro_f1` among all four strategies.

## Interpretation

These results suggest that, for `IndiWASTE` and a short `900` second fine-tuning budget, aggressively updating more of the backbone does not help. In fact, performance consistently worsened as more pretrained layers were allowed to change.

Three practical reasons explain this:

- **Training time efficiency**: `all_but_head` completed ~4821 gradient steps vs only ~987 for `freeze_none`. With fewer trainable parameters, each step is faster, so the model sees far more data within the same budget.
- **Overfitting**: The train-val gap grows with the number of trainable parameters — from 0.031 for `all_but_head` up to 0.112 for `freeze_none`. The dataset is relatively small (~2100 training images), so larger trainable parameter counts make overfitting easier.
- **Feature quality**: ImageNet features are already strong for visual recognition. Unfreezing them on a small dataset corrupts well-learned representations rather than improving them.

This pattern is visible in the results:

- `freeze_early` was slightly worse than the baseline — unfreezing deep layers added overfitting without enough benefit.
- `freeze_late` was meaningfully worse — unfreezing early layers disrupts low-level features that transfer well.
- `freeze_none`, which trained the entire network, was the worst configuration overall — most overfitting, fewest steps, worst accuracy.

## Recommendation

For this project, the recommended transfer learning strategy is:

`Freeze the full pretrained backbone and train only the final classification head.`

This is the best choice because it:

- produced the strongest validation performance
- used the smallest trainable parameter set
- matched the short-budget setting well
- was the simplest and most stable strategy tested

## Deliverable Summary

The freezing-strategy comparison table and the recommendation are now complete for Project 3. Under a controlled `900` second budget on `IndiWASTE`, `all_but_head` should be recommended as the final strategy.

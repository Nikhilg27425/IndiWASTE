# Project 2: The Attention Surgery Experiment

This folder runs a systematic attention-head ablation study on a pretrained ViT
(`vit_b_16` from torchvision) applied to the `IndiWASTE` dataset.

## Question

What happens when you remove (mask to uniform distribution) specific attention
heads in a Vision Transformer?

## Approach

1. Load pretrained `vit_b_16` (ImageNet weights), freeze all parameters.
2. For each of the 12 layers × 12 heads = 144 heads, inject a hook that replaces
   that head's attention weights with a uniform distribution (ablation).
3. Evaluate the ablated model on three probes:
   - **texture** — color-jitter-corrupted test split (disrupts texture cues)
   - **shape** — rotation-corrupted test split (disrupts pose but preserves shape)
   - **spatial** — crop-corrupted test split (disrupts spatial layout)
4. Head importance = accuracy drop vs. the intact baseline on each probe.
5. Produce a 12×12 importance heatmap per probe and collect failure examples for
   the most critical heads.

## Outputs

Running the experiment writes:

- `summary.json`        — full per-head accuracy/f1 for every probe
- `results.tsv`         — flat table: layer, head, probe, accuracy_drop
- `head_importance.json`— importance scores shaped [probe][layer][head]
- `failure_cases.csv`   — misclassified examples for the top-3 critical heads
- `project2_deliverable.md` — narrative report with heatmap table

## Run

```bash
python run_project2.py
```

Quick smoke-test (fewer heads):

```bash
python run_project2.py --layers 0 1 --heads 0 1 2
```

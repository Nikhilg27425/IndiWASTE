# Kaggle Workflow for Project 2

Use Kaggle as the GPU execution environment for the attention surgery study.

## What This Setup Does

- Keeps source code local in this repo
- Stages a Kaggle kernel for `project2_attention_surgery`
- Runs the full 144-head ablation (12 layers × 12 heads) on the IndiWASTE dataset
- Downloads `run.log`, `results.tsv`, `summary.json`, `head_importance.json`,
  `failure_cases.csv`, and `project2_deliverable.md`

## One-Time Setup

The dataset is already uploaded from Project 1 — reuse the same `indiwaste-dataset`.

## Config

The `kaggle_sync_config.json` is pre-filled with your credentials.

To ablate only a subset (faster testing), set `layers` and `heads` in the config:

```json
"layers": [0, 5, 11],
"heads": [0, 1, 2, 3]
```

Leave both as `[]` to run all 144 heads (full experiment, ~30-60 min on GPU).

## Run Flow

```bash
python kaggle_sync.py stage-kernel
python kaggle_sync.py push
python kaggle_sync.py watch
```

## Expected Outputs

After the Kaggle run completes:

- `kaggle_outputs/run.log`
- `kaggle_outputs/results.tsv`
- `kaggle_outputs/summary.json`
- `kaggle_outputs/head_importance.json`
- `kaggle_outputs/failure_cases.csv`
- `kaggle_outputs/project2_deliverable.md`

# Project 1: Data Augmentation as Inductive Bias

This folder runs a controlled augmentation study on the `IndiWASTE` dataset.

We keep the CNN architecture fixed and only change the training augmentation:

- `none`
- `rotation`
- `crop`
- `color_jitter`
- `cutout`

Each trained model is then evaluated on:

- the clean test split
- a rotated test split
- a cropped test split
- a color-jittered test split
- a cutout-corrupted test split

## Goal

Measure which augmentation helps the most, and whether the benefit transfers mainly to the matching corruption type.

## Outputs

Running the experiment writes:

- `summary.json`
- `results.tsv`
- `failure_cases.csv`
- `project1_deliverable.md`

For Kaggle GPU runs, also see:

- `kaggle_sync.py`
- `kaggle_workflow.md`
- `stage_kaggle_dataset.py`

## Run

```bash
cd /Users/nikhilgupta/Desktop/dlcv_project/project1_augmentation_bias
python3 run_project1.py
```

You can also shorten the run while testing:

```bash
python3 run_project1.py --epochs 3 --strategies none rotation
```

# Kaggle Workflow for Project 1

Use Kaggle as the GPU execution environment for the augmentation study.

## What This Setup Does

- keeps your source code local in this repo
- stages a Kaggle kernel for `project1_augmentation_bias`
- runs the augmentation comparison on your uploaded IndiWASTE dataset
- downloads `run.log`, `results.tsv`, `summary.json`, `failure_cases.csv`, and `project1_deliverable.md`

## One-Time Setup

1. Stage your current repo dataset for Kaggle upload:
   `python stage_kaggle_dataset.py`
2. Upload the staged dataset:
   `.venv_project1/bin/kaggle datasets create -p .kaggle_stage/dataset -r zip`
   or use `datasets version` if the dataset already exists.
3. Install the Kaggle CLI locally:
   `python -m pip install kaggle`
4. Put your Kaggle API token in the standard Kaggle location so `kaggle --version` works.

## Config

From this folder:

```bash
python kaggle_sync.py prepare-config
```

Then edit `kaggle_sync_config.json` and set:

- `kernel_id`
- `kernel_title`
- `dataset_source`
- `dataset_mount_slug`
- `dataset_subdir` if your upload contains an extra top-level folder

This project expects the Kaggle dataset mount to contain `images/` and `splits/` directly.

## Run Flow

```bash
python kaggle_sync.py stage-kernel
python kaggle_sync.py push
python kaggle_sync.py watch
```

## Expected Outputs

After the Kaggle run completes, check:

- `kaggle_outputs/run.log`
- `kaggle_outputs/results.tsv`
- `kaggle_outputs/summary.json`
- `kaggle_outputs/failure_cases.csv`
- `kaggle_outputs/project1_deliverable.md`

## Notes

- Kaggle GPU is the recommended path for this project because the full five-strategy experiment is slow on local CPU.
- If the dataset is mounted as `/kaggle/input/indiwaste-dataset/IndiWASTE`, set `dataset_subdir` to `IndiWASTE`.
- If you use `stage_kaggle_dataset.py`, leave `dataset_subdir` empty because the staged upload puts `images/` and `splits/` at the top level.
- The Kaggle runner will automatically unpack archived directories like `images.zip` and `splits.zip` before training.
- You can reduce runtime by lowering `epochs` in the config file.

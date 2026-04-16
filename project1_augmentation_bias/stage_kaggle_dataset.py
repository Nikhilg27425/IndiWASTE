from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_DIR.parent
DEFAULT_STAGE_DIR = PROJECT_DIR / ".kaggle_stage" / "dataset"
DATASET_METADATA_PATH = DEFAULT_STAGE_DIR / "dataset-metadata.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Stage the IndiWASTE dataset for Kaggle upload.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--slug", default="indiwaste-dataset")
    parser.add_argument("--title", default="IndiWASTE Dataset")
    parser.add_argument("--public", action="store_true")
    return parser.parse_args()


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    stage_dir = args.stage_dir.resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)

    required_dirs = ["images", "splits"]
    optional_dirs = ["annotations", "metadata", "docs"]

    for name in required_dirs:
        src = dataset_root / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required dataset directory: {src}")
        copy_tree(src, stage_dir / name)

    for name in optional_dirs:
        src = dataset_root / name
        if src.exists():
            copy_tree(src, stage_dir / name)

    metadata = {
        "title": args.title,
        "id": f"nikhilgupta2005/{args.slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    DATASET_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Staged Kaggle dataset at {stage_dir}")
    print(f"Wrote {DATASET_METADATA_PATH}")
    print("Upload with:")
    print(f"  .venv_project1/bin/kaggle datasets create -p {stage_dir} -r zip")
    print(f"  .venv_project1/bin/kaggle datasets version -p {stage_dir} -m \"Update IndiWASTE dataset\" -r zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

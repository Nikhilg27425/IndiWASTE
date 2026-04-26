from __future__ import annotations

import json
import random
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

IMAGE_SIZE = 224
NUM_CLASSES = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_EPOCHS = 8
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
NUM_WORKERS = 0

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

TRAIN_STRATEGIES = ["none", "rotation", "crop", "color_jitter", "cutout"]
CORRUPTION_SPLITS = ["clean", "rotation", "crop", "color_jitter", "cutout"]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _default_dataset_root() -> Path:
    explicit = Path(os.environ.get("INDIWASTE_ROOT", Path.cwd())).expanduser()
    candidate_roots = [
        explicit,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent / "IndiWASTE",
        Path("/kaggle/input"),
    ]
    seen: set[Path] = set()
    for root in candidate_roots:
        root = root.resolve() if root.exists() else root
        if root in seen:
            continue
        seen.add(root)
        if (root / "splits").exists() and (root / "images").exists():
            return root.resolve()
        if root.exists():
            for train_csv in root.rglob("train.csv"):
                if train_csv.parent.name != "splits":
                    continue
                possible_root = train_csv.parent.parent
                if (possible_root / "images").exists():
                    return possible_root.resolve()
    raise FileNotFoundError(
        "Could not locate dataset root with `splits/` and `images/`. "
        "Set INDIWASTE_ROOT to the repo or dataset directory."
    )


DATASET_ROOT = _default_dataset_root()


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fixed_color_jitter(image: Image.Image) -> Image.Image:
    image = F.adjust_brightness(image, 1.35)
    image = F.adjust_contrast(image, 1.2)
    image = F.adjust_saturation(image, 0.65)
    image = F.adjust_hue(image, 0.04)
    return image


def _fixed_cutout(tensor: torch.Tensor) -> torch.Tensor:
    _, height, width = tensor.shape
    cutout = int(min(height, width) * 0.3)
    top = (height - cutout) // 2
    left = (width - cutout) // 2
    tensor[:, top : top + cutout, left : left + cutout] = 0.0
    return tensor


def build_train_transform(strategy: str):
    if strategy not in TRAIN_STRATEGIES:
        raise ValueError(f"Unknown training strategy: {strategy}")

    ops = [transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)]
    if strategy == "rotation":
        ops.append(transforms.RandomRotation(degrees=25))
    elif strategy == "crop":
        ops = [transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1))]
    elif strategy == "color_jitter":
        ops.append(
            transforms.ColorJitter(
                brightness=0.35, contrast=0.25, saturation=0.35, hue=0.04
            )
        )

    ops.extend([transforms.ToTensor()])
    if strategy == "cutout":
        ops.append(transforms.RandomErasing(p=0.8, scale=(0.08, 0.2), ratio=(0.8, 1.2), value=0.0))
    ops.append(transforms.Normalize(mean=MEAN, std=STD))
    return transforms.Compose(ops)


def build_eval_transform(corruption: str = "clean"):
    if corruption not in CORRUPTION_SPLITS:
        raise ValueError(f"Unknown corruption split: {corruption}")

    ops = [transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)]
    if corruption == "rotation":
        ops.append(transforms.Lambda(lambda image: F.rotate(image, angle=25, fill=(0, 0, 0))))
    elif corruption == "crop":
        ops.append(
            transforms.Lambda(
                lambda image: F.resized_crop(
                    image,
                    top=24,
                    left=24,
                    height=max(IMAGE_SIZE - 48, 1),
                    width=max(IMAGE_SIZE - 48, 1),
                    size=[IMAGE_SIZE, IMAGE_SIZE],
                )
            )
        )
    elif corruption == "color_jitter":
        ops.append(transforms.Lambda(_fixed_color_jitter))

    ops.extend([transforms.ToTensor()])
    if corruption == "cutout":
        ops.append(transforms.Lambda(_fixed_cutout))
    ops.append(transforms.Normalize(mean=MEAN, std=STD))
    return transforms.Compose(ops)


class IndiWasteSplitDataset(Dataset):
    def __init__(self, root: Path, split: str, transform):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        frame = pd.read_csv(self.root / "splits" / f"{split}.csv")
        self.rows = frame.to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        label_name = row["label"]
        image_path = self.root / "images" / label_name / row["filename"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "label": CLASS_TO_IDX[label_name],
            "filename": row["filename"],
            "label_name": label_name,
        }


def make_dataloaders(
    dataset_root: Path | str | None = None,
    train_strategy: str = "none",
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
):
    root = Path(dataset_root).resolve() if dataset_root is not None else DATASET_ROOT
    train_loader = DataLoader(
        IndiWasteSplitDataset(root, "train", build_train_transform(train_strategy)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        IndiWasteSplitDataset(root, "val", build_eval_transform("clean")),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loaders = {
        corruption: DataLoader(
            IndiWasteSplitDataset(root, "test", build_eval_transform(corruption)),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        for corruption in CORRUPTION_SPLITS
    }
    return train_loader, val_loader, test_loaders


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    confusion: list[list[int]]
    per_class_f1: dict[str, float]
    failures: list[dict[str, object]]


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    failures: list[dict[str, object]] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        for idx, (target, pred) in enumerate(zip(targets.cpu().tolist(), preds.cpu().tolist())):
            if target != pred:
                failures.append(
                    {
                        "filename": batch["filename"][idx],
                        "true_label": IDX_TO_CLASS[target],
                        "predicted_label": IDX_TO_CLASS[pred],
                    }
                )

    confusion = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
    per_class = f1_score(all_targets, all_preds, labels=list(range(NUM_CLASSES)), average=None)
    return EvalResult(
        accuracy=float(accuracy_score(all_targets, all_preds)),
        macro_f1=float(f1_score(all_targets, all_preds, average="macro")),
        confusion=confusion.tolist(),
        per_class_f1={IDX_TO_CLASS[idx]: float(score) for idx, score in enumerate(per_class)},
        failures=failures,
    )


def dataset_summary(dataset_root: Path | str | None = None) -> dict[str, object]:
    root = Path(dataset_root).resolve() if dataset_root is not None else DATASET_ROOT
    summary = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(root / "splits" / f"{split}.csv")
        summary[split] = {
            "num_images": int(len(frame)),
            "class_counts": frame["label"].value_counts().sort_index().to_dict(),
        }
    return summary


def save_json(payload: dict[str, object], output_path: Path | str) -> None:
    Path(output_path).write_text(json.dumps(payload, indent=2))

import os
import shlex
import sys
import tarfile
import zipfile
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

DEFAULT_DATASET_ROOT = r"/kaggle/input/indiwaste-dataset"
P100_TORCH_INDEX_URL = r"https://download.pytorch.org/whl/cu113"
P100_TORCH_VERSION = r"1.12.1+cu113"
P100_TORCHVISION_VERSION = r"0.13.1+cu113"

PREPARE_SRC = 'from __future__ import annotations\n\nimport json\nimport random\nimport os\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom PIL import Image\nfrom sklearn.metrics import accuracy_score, confusion_matrix, f1_score\nfrom torch.utils.data import DataLoader, Dataset\nfrom torchvision import transforms\nfrom torchvision.transforms import functional as F\n\nIMAGE_SIZE = 224\nNUM_CLASSES = 10\nDEFAULT_BATCH_SIZE = 32\nDEFAULT_EVAL_BATCH_SIZE = 64\nDEFAULT_EPOCHS = 8\nDEFAULT_LR = 1e-3\nDEFAULT_WEIGHT_DECAY = 1e-4\nRANDOM_SEED = 42\nNUM_WORKERS = 0\n\nCLASS_NAMES = [\n    "battery",\n    "biological",\n    "cardboard",\n    "clothes",\n    "glass",\n    "metal",\n    "paper",\n    "plastic",\n    "shoes",\n    "trash",\n]\nCLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}\nIDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}\n\nTRAIN_STRATEGIES = ["none", "rotation", "crop", "color_jitter", "cutout"]\nCORRUPTION_SPLITS = ["clean", "rotation", "crop", "color_jitter", "cutout"]\n\nMEAN = (0.485, 0.456, 0.406)\nSTD = (0.229, 0.224, 0.225)\n\n\ndef _default_dataset_root() -> Path:\n    explicit = Path(os.environ.get("INDIWASTE_ROOT", Path.cwd())).expanduser()\n    candidate_roots = [\n        explicit,\n        Path(__file__).resolve().parent.parent,\n        Path(__file__).resolve().parent.parent / "IndiWASTE",\n        Path("/kaggle/input"),\n    ]\n    seen: set[Path] = set()\n    for root in candidate_roots:\n        root = root.resolve() if root.exists() else root\n        if root in seen:\n            continue\n        seen.add(root)\n        if (root / "splits").exists() and (root / "images").exists():\n            return root.resolve()\n        if root.exists():\n            for train_csv in root.rglob("train.csv"):\n                if train_csv.parent.name != "splits":\n                    continue\n                possible_root = train_csv.parent.parent\n                if (possible_root / "images").exists():\n                    return possible_root.resolve()\n    raise FileNotFoundError(\n        "Could not locate dataset root with `splits/` and `images/`. "\n        "Set INDIWASTE_ROOT to the repo or dataset directory."\n    )\n\n\nDATASET_ROOT = _default_dataset_root()\n\n\ndef set_seed(seed: int = RANDOM_SEED) -> None:\n    random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(seed)\n\n\ndef _fixed_color_jitter(image: Image.Image) -> Image.Image:\n    image = F.adjust_brightness(image, 1.35)\n    image = F.adjust_contrast(image, 1.2)\n    image = F.adjust_saturation(image, 0.65)\n    image = F.adjust_hue(image, 0.04)\n    return image\n\n\ndef _fixed_cutout(tensor: torch.Tensor) -> torch.Tensor:\n    _, height, width = tensor.shape\n    cutout = int(min(height, width) * 0.3)\n    top = (height - cutout) // 2\n    left = (width - cutout) // 2\n    tensor[:, top : top + cutout, left : left + cutout] = 0.0\n    return tensor\n\n\ndef build_train_transform(strategy: str):\n    if strategy not in TRAIN_STRATEGIES:\n        raise ValueError(f"Unknown training strategy: {strategy}")\n\n    ops = [transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)]\n    if strategy == "rotation":\n        ops.append(transforms.RandomRotation(degrees=25))\n    elif strategy == "crop":\n        ops = [transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1))]\n    elif strategy == "color_jitter":\n        ops.append(\n            transforms.ColorJitter(\n                brightness=0.35, contrast=0.25, saturation=0.35, hue=0.04\n            )\n        )\n\n    ops.extend([transforms.ToTensor()])\n    if strategy == "cutout":\n        ops.append(transforms.RandomErasing(p=0.8, scale=(0.08, 0.2), ratio=(0.8, 1.2), value=0.0))\n    ops.append(transforms.Normalize(mean=MEAN, std=STD))\n    return transforms.Compose(ops)\n\n\ndef build_eval_transform(corruption: str = "clean"):\n    if corruption not in CORRUPTION_SPLITS:\n        raise ValueError(f"Unknown corruption split: {corruption}")\n\n    ops = [transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)]\n    if corruption == "rotation":\n        ops.append(transforms.Lambda(lambda image: F.rotate(image, angle=25, fill=(0, 0, 0))))\n    elif corruption == "crop":\n        ops.append(\n            transforms.Lambda(\n                lambda image: F.resized_crop(\n                    image,\n                    top=24,\n                    left=24,\n                    height=max(IMAGE_SIZE - 48, 1),\n                    width=max(IMAGE_SIZE - 48, 1),\n                    size=[IMAGE_SIZE, IMAGE_SIZE],\n                )\n            )\n        )\n    elif corruption == "color_jitter":\n        ops.append(transforms.Lambda(_fixed_color_jitter))\n\n    ops.extend([transforms.ToTensor()])\n    if corruption == "cutout":\n        ops.append(transforms.Lambda(_fixed_cutout))\n    ops.append(transforms.Normalize(mean=MEAN, std=STD))\n    return transforms.Compose(ops)\n\n\nclass IndiWasteSplitDataset(Dataset):\n    def __init__(self, root: Path, split: str, transform):\n        self.root = Path(root)\n        self.split = split\n        self.transform = transform\n        frame = pd.read_csv(self.root / "splits" / f"{split}.csv")\n        self.rows = frame.to_dict("records")\n\n    def __len__(self) -> int:\n        return len(self.rows)\n\n    def __getitem__(self, index: int):\n        row = self.rows[index]\n        label_name = row["label"]\n        image_path = self.root / "images" / label_name / row["filename"]\n        image = Image.open(image_path).convert("RGB")\n        image = self.transform(image)\n        return {\n            "image": image,\n            "label": CLASS_TO_IDX[label_name],\n            "filename": row["filename"],\n            "label_name": label_name,\n        }\n\n\ndef make_dataloaders(\n    dataset_root: Path | str | None = None,\n    train_strategy: str = "none",\n    batch_size: int = DEFAULT_BATCH_SIZE,\n    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,\n):\n    root = Path(dataset_root).resolve() if dataset_root is not None else DATASET_ROOT\n    train_loader = DataLoader(\n        IndiWasteSplitDataset(root, "train", build_train_transform(train_strategy)),\n        batch_size=batch_size,\n        shuffle=True,\n        num_workers=NUM_WORKERS,\n        pin_memory=torch.cuda.is_available(),\n    )\n    val_loader = DataLoader(\n        IndiWasteSplitDataset(root, "val", build_eval_transform("clean")),\n        batch_size=eval_batch_size,\n        shuffle=False,\n        num_workers=NUM_WORKERS,\n        pin_memory=torch.cuda.is_available(),\n    )\n    test_loaders = {\n        corruption: DataLoader(\n            IndiWasteSplitDataset(root, "test", build_eval_transform(corruption)),\n            batch_size=eval_batch_size,\n            shuffle=False,\n            num_workers=NUM_WORKERS,\n            pin_memory=torch.cuda.is_available(),\n        )\n        for corruption in CORRUPTION_SPLITS\n    }\n    return train_loader, val_loader, test_loaders\n\n\n@dataclass\nclass EvalResult:\n    accuracy: float\n    macro_f1: float\n    confusion: list[list[int]]\n    per_class_f1: dict[str, float]\n    failures: list[dict[str, object]]\n\n\n@torch.no_grad()\ndef evaluate_classifier(model, loader, device):\n    model.eval()\n    all_targets: list[int] = []\n    all_preds: list[int] = []\n    failures: list[dict[str, object]] = []\n\n    for batch in loader:\n        images = batch["image"].to(device, non_blocking=True)\n        targets = batch["label"].to(device, non_blocking=True)\n        logits = model(images)\n        preds = logits.argmax(dim=1)\n\n        all_targets.extend(targets.cpu().tolist())\n        all_preds.extend(preds.cpu().tolist())\n\n        for idx, (target, pred) in enumerate(zip(targets.cpu().tolist(), preds.cpu().tolist())):\n            if target != pred:\n                failures.append(\n                    {\n                        "filename": batch["filename"][idx],\n                        "true_label": IDX_TO_CLASS[target],\n                        "predicted_label": IDX_TO_CLASS[pred],\n                    }\n                )\n\n    confusion = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))\n    per_class = f1_score(all_targets, all_preds, labels=list(range(NUM_CLASSES)), average=None)\n    return EvalResult(\n        accuracy=float(accuracy_score(all_targets, all_preds)),\n        macro_f1=float(f1_score(all_targets, all_preds, average="macro")),\n        confusion=confusion.tolist(),\n        per_class_f1={IDX_TO_CLASS[idx]: float(score) for idx, score in enumerate(per_class)},\n        failures=failures,\n    )\n\n\ndef dataset_summary(dataset_root: Path | str | None = None) -> dict[str, object]:\n    root = Path(dataset_root).resolve() if dataset_root is not None else DATASET_ROOT\n    summary = {}\n    for split in ("train", "val", "test"):\n        frame = pd.read_csv(root / "splits" / f"{split}.csv")\n        summary[split] = {\n            "num_images": int(len(frame)),\n            "class_counts": frame["label"].value_counts().sort_index().to_dict(),\n        }\n    return summary\n\n\ndef save_json(payload: dict[str, object], output_path: Path | str) -> None:\n    Path(output_path).write_text(json.dumps(payload, indent=2))\n'
TRAIN_SRC = 'from __future__ import annotations\n\nimport argparse\nimport json\nfrom pathlib import Path\n\nimport torch\nimport torch.nn as nn\nfrom torch.optim import AdamW\n\nfrom prepare import (\n    CLASS_NAMES,\n    CORRUPTION_SPLITS,\n    DATASET_ROOT,\n    DEFAULT_BATCH_SIZE,\n    DEFAULT_EPOCHS,\n    DEFAULT_EVAL_BATCH_SIZE,\n    DEFAULT_LR,\n    DEFAULT_WEIGHT_DECAY,\n    TRAIN_STRATEGIES,\n    dataset_summary,\n    evaluate_classifier,\n    make_dataloaders,\n    save_json,\n    set_seed,\n)\n\n\nclass SmallCNN(nn.Module):\n    def __init__(self, num_classes: int = len(CLASS_NAMES)):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(3, 32, kernel_size=3, padding=1),\n            nn.BatchNorm2d(32),\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(2),\n            nn.Conv2d(32, 64, kernel_size=3, padding=1),\n            nn.BatchNorm2d(64),\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(2),\n            nn.Conv2d(64, 128, kernel_size=3, padding=1),\n            nn.BatchNorm2d(128),\n            nn.ReLU(inplace=True),\n            nn.MaxPool2d(2),\n            nn.Conv2d(128, 256, kernel_size=3, padding=1),\n            nn.BatchNorm2d(256),\n            nn.ReLU(inplace=True),\n            nn.AdaptiveAvgPool2d((1, 1)),\n        )\n        self.classifier = nn.Sequential(\n            nn.Flatten(),\n            nn.Dropout(p=0.3),\n            nn.Linear(256, num_classes),\n        )\n\n    def forward(self, x):\n        return self.classifier(self.features(x))\n\n\ndef train_one_epoch(model, loader, optimizer, criterion, device):\n    model.train()\n    running_loss = 0.0\n    total = 0\n    correct = 0\n\n    for batch in loader:\n        images = batch["image"].to(device, non_blocking=True)\n        targets = batch["label"].to(device, non_blocking=True)\n\n        optimizer.zero_grad(set_to_none=True)\n        logits = model(images)\n        loss = criterion(logits, targets)\n        loss.backward()\n        optimizer.step()\n\n        running_loss += loss.item() * images.size(0)\n        preds = logits.argmax(dim=1)\n        correct += (preds == targets).sum().item()\n        total += images.size(0)\n\n    return {\n        "loss": running_loss / max(total, 1),\n        "accuracy": correct / max(total, 1),\n    }\n\n\ndef run_strategy(strategy: str, args, device):\n    set_seed(args.seed)\n    print(f"=== Strategy: {strategy} ===")\n    train_loader, val_loader, test_loaders = make_dataloaders(\n        dataset_root=args.dataset_root,\n        train_strategy=strategy,\n        batch_size=args.batch_size,\n        eval_batch_size=args.eval_batch_size,\n    )\n\n    model = SmallCNN().to(device)\n    criterion = nn.CrossEntropyLoss()\n    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)\n\n    best_val_acc = -1.0\n    best_state = None\n    history = []\n\n    for epoch in range(1, args.epochs + 1):\n        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)\n        val_metrics = evaluate_classifier(model, val_loader, device)\n        print(\n            "epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "\n            "val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}".format(\n                epoch=epoch,\n                train_loss=train_metrics["loss"],\n                train_acc=train_metrics["accuracy"],\n                val_acc=val_metrics.accuracy,\n                val_macro_f1=val_metrics.macro_f1,\n            )\n        )\n        history.append(\n            {\n                "epoch": epoch,\n                "train_loss": train_metrics["loss"],\n                "train_accuracy": train_metrics["accuracy"],\n                "val_accuracy": val_metrics.accuracy,\n                "val_macro_f1": val_metrics.macro_f1,\n            }\n        )\n        if val_metrics.accuracy > best_val_acc:\n            best_val_acc = val_metrics.accuracy\n            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}\n\n    if best_state is None:\n        raise RuntimeError(f"No model checkpoint captured for strategy {strategy}")\n\n    model.load_state_dict(best_state)\n    evaluations = {\n        corruption: evaluate_classifier(model, loader, device)\n        for corruption, loader in test_loaders.items()\n    }\n    for corruption, result in evaluations.items():\n        print(\n            f"test_split={corruption} accuracy={result.accuracy:.4f} macro_f1={result.macro_f1:.4f}"\n        )\n    return {\n        "strategy": strategy,\n        "history": history,\n        "best_val_accuracy": best_val_acc,\n        "test": {\n            corruption: {\n                "accuracy": result.accuracy,\n                "macro_f1": result.macro_f1,\n                "confusion": result.confusion,\n                "per_class_f1": result.per_class_f1,\n                "failures": result.failures,\n            }\n            for corruption, result in evaluations.items()\n        },\n    }\n\n\ndef write_results_tsv(results: list[dict[str, object]], output_path: Path) -> None:\n    columns = [\n        "strategy",\n        "best_val_accuracy",\n        "clean_accuracy",\n        "rotation_accuracy",\n        "crop_accuracy",\n        "color_jitter_accuracy",\n        "cutout_accuracy",\n        "mean_corrupted_accuracy",\n    ]\n    lines = ["\\t".join(columns)]\n    for item in results:\n        test = item["test"]\n        corrupted_mean = sum(\n            float(test[name]["accuracy"]) for name in CORRUPTION_SPLITS if name != "clean"\n        ) / 4.0\n        lines.append(\n            "\\t".join(\n                [\n                    item["strategy"],\n                    f"{float(item[\'best_val_accuracy\']):.4f}",\n                    f"{float(test[\'clean\'][\'accuracy\']):.4f}",\n                    f"{float(test[\'rotation\'][\'accuracy\']):.4f}",\n                    f"{float(test[\'crop\'][\'accuracy\']):.4f}",\n                    f"{float(test[\'color_jitter\'][\'accuracy\']):.4f}",\n                    f"{float(test[\'cutout\'][\'accuracy\']):.4f}",\n                    f"{corrupted_mean:.4f}",\n                ]\n            )\n        )\n    output_path.write_text("\\n".join(lines) + "\\n")\n\n\ndef write_failure_cases(results: list[dict[str, object]], output_path: Path) -> None:\n    best = max(results, key=lambda item: float(item["test"]["clean"]["accuracy"]))\n    lines = ["strategy,corruption,filename,true_label,predicted_label"]\n    for corruption in CORRUPTION_SPLITS:\n        failures = best["test"][corruption]["failures"][:20]\n        for failure in failures:\n            lines.append(\n                ",".join(\n                    [\n                        best["strategy"],\n                        corruption,\n                        failure["filename"],\n                        failure["true_label"],\n                        failure["predicted_label"],\n                    ]\n                )\n            )\n    output_path.write_text("\\n".join(lines) + "\\n")\n\n\ndef write_deliverable(results: list[dict[str, object]], output_path: Path) -> None:\n    best_clean = max(results, key=lambda item: float(item["test"]["clean"]["accuracy"]))\n    best_corrupted = max(\n        results,\n        key=lambda item: sum(\n            float(item["test"][name]["accuracy"]) for name in CORRUPTION_SPLITS if name != "clean"\n        )\n        / 4.0,\n    )\n\n    table_lines = [\n        "| Strategy | Val Acc | Clean Test | Rotated Test | Cropped Test | Jittered Test | Cutout Test | Mean Corrupted |",\n        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",\n    ]\n    for item in results:\n        corrupted_mean = sum(\n            float(item["test"][name]["accuracy"]) for name in CORRUPTION_SPLITS if name != "clean"\n        ) / 4.0\n        table_lines.append(\n            "| {strategy} | {val:.4f} | {clean:.4f} | {rotation:.4f} | {crop:.4f} | {color:.4f} | {cutout:.4f} | {mean:.4f} |".format(\n                strategy=item["strategy"],\n                val=float(item["best_val_accuracy"]),\n                clean=float(item["test"]["clean"]["accuracy"]),\n                rotation=float(item["test"]["rotation"]["accuracy"]),\n                crop=float(item["test"]["crop"]["accuracy"]),\n                color=float(item["test"]["color_jitter"]["accuracy"]),\n                cutout=float(item["test"]["cutout"]["accuracy"]),\n                mean=corrupted_mean,\n            )\n        )\n\n    failure_hints = []\n    for corruption in CORRUPTION_SPLITS:\n        failures = best_corrupted["test"][corruption]["failures"][:3]\n        if failures:\n            example = "; ".join(\n                f"{item[\'filename\']} ({item[\'true_label\']} -> {item[\'predicted_label\']})"\n                for item in failures\n            )\n            failure_hints.append(f"- `{corruption}`: {example}")\n    if not failure_hints:\n        failure_hints = ["- No failure cases were recorded."]\n\n    output_lines = [\n        "# Project 1 Deliverable: Data Augmentation as Inductive Bias",\n        "",\n        "## Experimental Setup",\n        "",\n        f"- Dataset root: `{Path(DATASET_ROOT).resolve()}`",\n        "- Model: `SmallCNN` with four convolution blocks",\n        f"- Epochs per strategy: `{results[0][\'history\'][-1][\'epoch\'] if results and results[0][\'history\'] else 0}`",\n        "- Strategies compared: `none`, `rotation`, `crop`, `color_jitter`, `cutout`",\n        "- Corrupted test sets: `rotation`, `crop`, `color_jitter`, `cutout`",\n        "",\n        "## Augmentation-Performance Matrix",\n        "",\n        *table_lines,\n        "",\n        "## Main Findings",\n        "",\n        f"- Best clean-test strategy: `{best_clean[\'strategy\']}`",\n        f"- Best corruption-robust strategy: `{best_corrupted[\'strategy\']}`",\n        "- Compare clean accuracy against the four corrupted evaluations to see which inductive bias transfers best.",\n        "",\n        "## Failure Case Analysis",\n        "",\n        *failure_hints,\n        "",\n        "## Hypothesis Check",\n        "",\n        "- If a strategy improves most on the matching corruption type, that supports the idea that augmentation teaches useful invariances.",\n        "- If a strategy hurts clean accuracy but helps the matching corruption, that suggests a robustness-accuracy tradeoff rather than a universal win.",\n        "",\n        "## Artifact Files",\n        "",\n        "- `results.tsv` for the comparison matrix",\n        "- `summary.json` for full metrics and histories",\n        "- `failure_cases.csv` for qualitative review samples",\n        "",\n    ]\n    output = "\\n".join(output_lines)\n    output_path.write_text(output)\n\n\ndef parse_args():\n    parser = argparse.ArgumentParser(description="Project 1 augmentation experiment runner.")\n    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)\n    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)\n    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)\n    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)\n    parser.add_argument("--lr", type=float, default=DEFAULT_LR)\n    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)\n    parser.add_argument("--seed", type=int, default=42)\n    parser.add_argument(\n        "--strategies",\n        nargs="+",\n        default=TRAIN_STRATEGIES,\n        choices=TRAIN_STRATEGIES,\n    )\n    return parser.parse_args()\n\n\ndef _get_device() -> torch.device:\n    """Pick the best available device, falling back to CPU if CUDA is present but\n    incompatible with the installed PyTorch build (e.g. Tesla P100 / sm_60 on a\n    PyTorch build that only supports sm_70+)."""\n    if torch.cuda.is_available():\n        try:\n            torch.zeros(1).cuda()  # will raise if kernel image is missing\n            return torch.device("cuda")\n        except Exception as exc:\n            print(\n                f"WARNING: CUDA reported as available but failed a smoke-test "\n                f"({exc.__class__.__name__}: {exc}). Falling back to CPU."\n            )\n    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():\n        return torch.device("mps")\n    return torch.device("cpu")\n\n\ndef main():\n    args = parse_args()\n    device = _get_device()\n    output_dir = Path(__file__).resolve().parent\n\n    print(json.dumps(dataset_summary(args.dataset_root), indent=2))\n    print(f"Running on device: {device}")\n\n    results = [run_strategy(strategy, args, device) for strategy in args.strategies]\n    save_json(\n        {\n            "dataset_root": str(Path(args.dataset_root).resolve()),\n            "device": str(device),\n            "strategies": results,\n        },\n        output_dir / "summary.json",\n    )\n    write_results_tsv(results, output_dir / "results.tsv")\n    write_failure_cases(results, output_dir / "failure_cases.csv")\n    write_deliverable(results, output_dir / "project1_deliverable.md")\n    print(f"Wrote artifacts to {output_dir}")\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
RUN_SRC = 'from train import main\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'

Path("prepare.py").write_text(PREPARE_SRC, encoding="utf-8")
Path("train.py").write_text(TRAIN_SRC, encoding="utf-8")
Path("run_project1.py").write_text(RUN_SRC, encoding="utf-8")
sys.path.insert(0, os.getcwd())


def _extract_if_needed(dataset_root: Path) -> Path:
    if (dataset_root / "splits").exists() and (dataset_root / "images").exists():
        return dataset_root

    extract_root = Path("/kaggle/working/indiwaste_dataset")
    extract_root.mkdir(parents=True, exist_ok=True)
    extracted_any = False

    for name in ("images", "splits", "annotations", "metadata", "docs"):
        zip_path = dataset_root / f"{name}.zip"
        tar_path = dataset_root / f"{name}.tar"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_root)
            extracted_any = True
        elif tar_path.exists():
            with tarfile.open(tar_path) as archive:
                archive.extractall(extract_root)
            extracted_any = True

    if extracted_any and (extract_root / "splits").exists() and (extract_root / "images").exists():
        return extract_root
    return dataset_root


os.environ.setdefault("INDIWASTE_ROOT", str(_extract_if_needed(Path(DEFAULT_DATASET_ROOT))))


def _gpu_name():
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        print(f"gpu_probe_error: {type(exc).__name__}")
        return ""
    return (probe.stdout or probe.stderr or "").strip()


def _ensure_p100_compatible_torch():
    gpu_name = _gpu_name()
    if gpu_name:
        print(f"gpu_name: {gpu_name}")
    if "P100" not in gpu_name:
        return

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-input",
        "--no-cache-dir",
        "--upgrade",
        "--index-url",
        P100_TORCH_INDEX_URL,
        f"torch=={P100_TORCH_VERSION}",
        f"torchvision=={P100_TORCHVISION_VERSION}",
    ]
    print("+", " ".join(install_cmd))
    try:
        subprocess.run(install_cmd, check=True)
        print("P100-compatible PyTorch installed successfully.")
    except Exception as exc:
        print(
            f"WARNING: Could not install P100-compatible PyTorch ({exc.__class__.__name__}: {exc}). "
            "Will attempt to run anyway -- training will fall back to CPU if CUDA is incompatible."
        )


_ensure_p100_compatible_torch()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


from train import main


if __name__ == "__main__":
    argv = [
        "run_project1.py",
        "--epochs",
        "8",
        "--batch-size",
        "32",
        "--eval-batch-size",
        "64",
        "--strategies",
        *shlex.split('none rotation crop color_jitter cutout'),
    ]
    sys.argv = argv
    with open("run.log", "w", encoding="utf-8") as f:
        tee_out = Tee(sys.stdout, f)
        tee_err = Tee(sys.stderr, f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            raise SystemExit(main())

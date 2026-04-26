from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from prepare import (
    CLASS_NAMES,
    CORRUPTION_SPLITS,
    DATASET_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    TRAIN_STRATEGIES,
    dataset_summary,
    evaluate_classifier,
    make_dataloaders,
    save_json,
    set_seed,
)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def run_strategy(strategy: str, args, device):
    set_seed(args.seed)
    print(f"=== Strategy: {strategy} ===")
    train_loader, val_loader, test_loaders = make_dataloaders(
        dataset_root=args.dataset_root,
        train_strategy=strategy,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_classifier(model, val_loader, device)
        print(
            "epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            "val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}".format(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_acc=train_metrics["accuracy"],
                val_acc=val_metrics.accuracy,
                val_macro_f1=val_metrics.macro_f1,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics.accuracy,
                "val_macro_f1": val_metrics.macro_f1,
            }
        )
        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No model checkpoint captured for strategy {strategy}")

    model.load_state_dict(best_state)
    evaluations = {
        corruption: evaluate_classifier(model, loader, device)
        for corruption, loader in test_loaders.items()
    }
    for corruption, result in evaluations.items():
        print(
            f"test_split={corruption} accuracy={result.accuracy:.4f} macro_f1={result.macro_f1:.4f}"
        )
    return {
        "strategy": strategy,
        "history": history,
        "best_val_accuracy": best_val_acc,
        "test": {
            corruption: {
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "confusion": result.confusion,
                "per_class_f1": result.per_class_f1,
                "failures": result.failures,
            }
            for corruption, result in evaluations.items()
        },
    }


def write_results_tsv(results: list[dict[str, object]], output_path: Path) -> None:
    columns = [
        "strategy",
        "best_val_accuracy",
        "clean_accuracy",
        "rotation_accuracy",
        "crop_accuracy",
        "color_jitter_accuracy",
        "cutout_accuracy",
        "mean_corrupted_accuracy",
    ]
    lines = ["\t".join(columns)]
    for item in results:
        test = item["test"]
        corrupted_mean = sum(
            float(test[name]["accuracy"]) for name in CORRUPTION_SPLITS if name != "clean"
        ) / 4.0
        lines.append(
            "\t".join(
                [
                    item["strategy"],
                    f"{float(item['best_val_accuracy']):.4f}",
                    f"{float(test['clean']['accuracy']):.4f}",
                    f"{float(test['rotation']['accuracy']):.4f}",
                    f"{float(test['crop']['accuracy']):.4f}",
                    f"{float(test['color_jitter']['accuracy']):.4f}",
                    f"{float(test['cutout']['accuracy']):.4f}",
                    f"{corrupted_mean:.4f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n")


def write_failure_cases(results: list[dict[str, object]], output_path: Path) -> None:
    best = max(results, key=lambda item: float(item["test"]["clean"]["accuracy"]))
    lines = ["strategy,corruption,filename,true_label,predicted_label"]
    for corruption in CORRUPTION_SPLITS:
        failures = best["test"][corruption]["failures"][:20]
        for failure in failures:
            lines.append(
                ",".join(
                    [
                        best["strategy"],
                        corruption,
                        failure["filename"],
                        failure["true_label"],
                        failure["predicted_label"],
                    ]
                )
            )
    output_path.write_text("\n".join(lines) + "\n")


def write_deliverable(results: list[dict[str, object]], output_path: Path) -> None:
    best_clean = max(results, key=lambda item: float(item["test"]["clean"]["accuracy"]))
    best_corrupted = max(
        results,
        key=lambda item: sum(
            float(item["test"][name]["accuracy"]) for name in CORRUPTION_SPLITS if name != "clean"
        )
        / 4.0,
    )

    table_lines = [
        "| Strategy | Val Acc | Clean Test | Rotated Test | Cropped Test | Jittered Test | Cutout Test | Mean Corrupted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        corrupted_mean = sum(
            float(item["test"][name]["accuracy"]) for name in CORRUPTION_SPLITS if name != "clean"
        ) / 4.0
        table_lines.append(
            "| {strategy} | {val:.4f} | {clean:.4f} | {rotation:.4f} | {crop:.4f} | {color:.4f} | {cutout:.4f} | {mean:.4f} |".format(
                strategy=item["strategy"],
                val=float(item["best_val_accuracy"]),
                clean=float(item["test"]["clean"]["accuracy"]),
                rotation=float(item["test"]["rotation"]["accuracy"]),
                crop=float(item["test"]["crop"]["accuracy"]),
                color=float(item["test"]["color_jitter"]["accuracy"]),
                cutout=float(item["test"]["cutout"]["accuracy"]),
                mean=corrupted_mean,
            )
        )

    failure_hints = []
    for corruption in CORRUPTION_SPLITS:
        failures = best_corrupted["test"][corruption]["failures"][:3]
        if failures:
            example = "; ".join(
                f"{item['filename']} ({item['true_label']} -> {item['predicted_label']})"
                for item in failures
            )
            failure_hints.append(f"- `{corruption}`: {example}")
    if not failure_hints:
        failure_hints = ["- No failure cases were recorded."]

    output_lines = [
        "# Project 1 Deliverable: Data Augmentation as Inductive Bias",
        "",
        "## Experimental Setup",
        "",
        f"- Dataset root: `{Path(DATASET_ROOT).resolve()}`",
        "- Model: `SmallCNN` with four convolution blocks",
        f"- Epochs per strategy: `{results[0]['history'][-1]['epoch'] if results and results[0]['history'] else 0}`",
        "- Strategies compared: `none`, `rotation`, `crop`, `color_jitter`, `cutout`",
        "- Corrupted test sets: `rotation`, `crop`, `color_jitter`, `cutout`",
        "",
        "## Augmentation-Performance Matrix",
        "",
        *table_lines,
        "",
        "## Main Findings",
        "",
        f"- Best clean-test strategy: `{best_clean['strategy']}`",
        f"- Best corruption-robust strategy: `{best_corrupted['strategy']}`",
        "- Compare clean accuracy against the four corrupted evaluations to see which inductive bias transfers best.",
        "",
        "## Failure Case Analysis",
        "",
        *failure_hints,
        "",
        "## Hypothesis Check",
        "",
        "- If a strategy improves most on the matching corruption type, that supports the idea that augmentation teaches useful invariances.",
        "- If a strategy hurts clean accuracy but helps the matching corruption, that suggests a robustness-accuracy tradeoff rather than a universal win.",
        "",
        "## Artifact Files",
        "",
        "- `results.tsv` for the comparison matrix",
        "- `summary.json` for full metrics and histories",
        "- `failure_cases.csv` for qualitative review samples",
        "",
    ]
    output = "\n".join(output_lines)
    output_path.write_text(output)


def parse_args():
    parser = argparse.ArgumentParser(description="Project 1 augmentation experiment runner.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=TRAIN_STRATEGIES,
        choices=TRAIN_STRATEGIES,
    )
    return parser.parse_args()


def _get_device() -> torch.device:
    """Pick the best available device, falling back to CPU if CUDA is present but
    incompatible with the installed PyTorch build (e.g. Tesla P100 / sm_60 on a
    PyTorch build that only supports sm_70+)."""
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            if major < 7:
                print(
                    f"WARNING: GPU compute capability is sm_{major}{minor} "
                    f"but this PyTorch build requires sm_70+. Falling back to CPU."
                )
            else:
                return torch.device("cuda")
        except Exception as exc:
            print(f"WARNING: CUDA capability check failed ({exc}). Falling back to CPU.")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = _get_device()
    output_dir = Path(__file__).resolve().parent

    print(json.dumps(dataset_summary(args.dataset_root), indent=2))
    print(f"Running on device: {device}")

    results = [run_strategy(strategy, args, device) for strategy in args.strategies]
    save_json(
        {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "device": str(device),
            "strategies": results,
        },
        output_dir / "summary.json",
    )
    write_results_tsv(results, output_dir / "results.tsv")
    write_failure_cases(results, output_dir / "failure_cases.csv")
    write_deliverable(results, output_dir / "project1_deliverable.md")
    print(f"Wrote artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ============================================================
# Project 2: The Attention Surgery Experiment
# Paste this entire file into a single Kaggle notebook cell.
#
# Requirements:
#   - Accelerator: GPU (any T4/P100)
#   - Dataset attached: nikhilgupta2005/indiwaste-dataset
#   - Internet: ON (to download ViT weights)
# ============================================================

import json
import os
import random
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
# Auto-detect dataset root — handles both flat and nested Kaggle mounts
def _find_dataset_root():
    for candidate in [
        Path("/kaggle/input/indiwaste-dataset"),
        Path("/kaggle/input/indiwaste-dataset/IndiWASTE"),
        *sorted(Path("/kaggle/input").glob("*/splits")),   # any dataset with splits/
    ]:
        if isinstance(candidate, Path) and candidate.name == "splits":
            candidate = candidate.parent
        if candidate.exists() and (candidate/"splits").exists() and (candidate/"images").exists():
            return candidate
    # fallback: walk /kaggle/input for splits/test.csv
    for p in Path("/kaggle/input").rglob("splits/test.csv"):
        return p.parent.parent
    raise FileNotFoundError("Cannot find dataset root under /kaggle/input")

DATASET_ROOT = _find_dataset_root()
OUTPUT_DIR     = Path("/kaggle/working")
EVAL_BATCH     = 64
LAYERS_TO_TEST = list(range(12))   # all 12 layers  ← change to e.g. [0,5,11] for a quick run
HEADS_TO_TEST  = list(range(12))   # all 12 heads
SEED           = 42
IMAGE_SIZE     = 224
NUM_CLASSES    = 10
NUM_WORKERS    = 2

CLASS_NAMES = ["battery","biological","cardboard","clothes","glass",
               "metal","paper","plastic","shoes","trash"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: n for n, i in CLASS_TO_IDX.items()}

PROBES = ["texture", "shape", "spatial"]
PROBE_TO_CORRUPTION = {"texture": "color_jitter", "shape": "rotation", "spatial": "crop"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ── Device (P100-safe) ────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 7:
            return torch.device("cuda")
        print(f"WARNING: GPU is sm_{major}{minor}, needs sm_70+. Using CPU.")
    return torch.device("cpu")

# ── Transforms ────────────────────────────────────────────────────────────────
def _color_jitter(img):
    img = F.adjust_brightness(img, 1.35); img = F.adjust_contrast(img, 1.2)
    img = F.adjust_saturation(img, 0.65); img = F.adjust_hue(img, 0.04)
    return img

def _cutout(t):
    _, h, w = t.shape; c = int(min(h,w)*0.3)
    top, left = (h-c)//2, (w-c)//2
    t[:, top:top+c, left:left+c] = 0.0; return t

def build_eval_transform(corruption="clean"):
    ops = [transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)]
    if corruption == "rotation":
        ops.append(transforms.Lambda(lambda img: F.rotate(img, 25, fill=(0,0,0))))
    elif corruption == "crop":
        ops.append(transforms.Lambda(lambda img: F.resized_crop(
            img, 24, 24, max(IMAGE_SIZE-48,1), max(IMAGE_SIZE-48,1), [IMAGE_SIZE,IMAGE_SIZE])))
    elif corruption == "color_jitter":
        ops.append(transforms.Lambda(_color_jitter))
    ops.append(transforms.ToTensor())
    if corruption == "cutout":
        ops.append(transforms.Lambda(_cutout))
    ops.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(ops)

# ── Dataset ───────────────────────────────────────────────────────────────────
class IndiWasteDataset(Dataset):
    def __init__(self, root, split, transform):
        self.root = Path(root); self.transform = transform
        self.rows = pd.read_csv(self.root/"splits"/f"{split}.csv").to_dict("records")
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; label = row["label"]
        img = Image.open(self.root/"images"/label/row["filename"]).convert("RGB")
        return {"image": self.transform(img), "label": CLASS_TO_IDX[label],
                "filename": row["filename"]}

def make_loaders(root, batch_size):
    pin = torch.cuda.is_available()
    loaders = {}
    for name in ["clean"] + PROBES:
        corruption = PROBE_TO_CORRUPTION.get(name, "clean")
        loaders[name] = DataLoader(
            IndiWasteDataset(root, "test", build_eval_transform(corruption)),
            batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=pin)
    return loaders

# ── Evaluation ────────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    accuracy: float; macro_f1: float
    confusion: list; per_class_f1: dict; failures: list

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    targets_all, preds_all, failures = [], [], []
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        tgts = batch["label"].to(device, non_blocking=True)
        preds = model(imgs).argmax(1)
        targets_all += tgts.cpu().tolist(); preds_all += preds.cpu().tolist()
        for i,(t,p) in enumerate(zip(tgts.cpu().tolist(), preds.cpu().tolist())):
            if t != p:
                failures.append({"filename": batch["filename"][i],
                                  "true_label": IDX_TO_CLASS[t],
                                  "predicted_label": IDX_TO_CLASS[p]})
    cm = confusion_matrix(targets_all, preds_all, labels=list(range(NUM_CLASSES)))
    pc = f1_score(targets_all, preds_all, labels=list(range(NUM_CLASSES)), average=None)
    return EvalResult(
        accuracy=float(accuracy_score(targets_all, preds_all)),
        macro_f1=float(f1_score(targets_all, preds_all, average="macro")),
        confusion=cm.tolist(),
        per_class_f1={IDX_TO_CLASS[i]: float(s) for i,s in enumerate(pc)},
        failures=failures)

# ── Model ─────────────────────────────────────────────────────────────────────
def load_vit(device):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    for name, p in model.named_parameters():
        if "heads" not in name: p.requires_grad_(False)
    return model.to(device).eval()

# ── Head ablation ─────────────────────────────────────────────────────────────
class HeadMasker:
    """Zero-masks the out_proj columns for one head — equivalent to ablating it."""
    def __init__(self, embed_dim=768, num_heads=12):
        self.head_dim = embed_dim // num_heads

    @contextmanager
    def ablated(self, mha: nn.MultiheadAttention, head_idx: int):
        start = head_idx * self.head_dim
        end   = start + self.head_dim
        w = mha.out_proj.weight          # (embed_dim, embed_dim)
        saved = w[:, start:end].clone()
        with torch.no_grad(): w[:, start:end] = 0.0
        try:    yield
        finally:
            with torch.no_grad(): w[:, start:end] = saved

# ── Main experiment ───────────────────────────────────────────────────────────
def run():
    set_seed()
    device = get_device()
    print(f"Device: {device}")

    root = DATASET_ROOT
    print(f"Dataset root: {root}")
    loaders = make_loaders(root, EVAL_BATCH)
    model   = load_vit(device)
    masker  = HeadMasker()

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n=== Baseline ===")
    baseline = {}
    for name, loader in loaders.items():
        r = evaluate(model, loader, device)
        baseline[name] = r.accuracy
        print(f"  {name:10s}  acc={r.accuracy:.4f}  f1={r.macro_f1:.4f}")

    # ── Ablation loop ─────────────────────────────────────────────────────────
    ablations = {}
    total = len(LAYERS_TO_TEST) * len(HEADS_TO_TEST)
    done  = 0
    for layer_idx in LAYERS_TO_TEST:
        ablations[layer_idx] = {}
        mha = model.encoder.layers[layer_idx].self_attention
        for head_idx in HEADS_TO_TEST:
            done += 1
            head_res = {}
            with masker.ablated(mha, head_idx):
                for probe in ["clean"] + PROBES:
                    r = evaluate(model, loaders[probe], device)
                    head_res[probe] = {
                        "accuracy":      r.accuracy,
                        "macro_f1":      r.macro_f1,
                        "accuracy_drop": baseline[probe] - r.accuracy,
                        "failures":      r.failures[:10],
                    }
            ablations[layer_idx][head_idx] = head_res
            print(f"[{done:3d}/{total}] L{layer_idx:02d} H{head_idx:02d} | "
                  + " | ".join(f"{p}={head_res[p]['accuracy_drop']:+.4f}" for p in PROBES))

    results = {"baseline": baseline, "ablations": ablations}

    # ── Save outputs ──────────────────────────────────────────────────────────
    (OUTPUT_DIR/"summary.json").write_text(json.dumps(results, indent=2))

    # results.tsv
    tsv = ["layer\thead\tprobe\tbaseline\tablated_acc\tdrop\tmacro_f1"]
    for li, heads in ablations.items():
        for hi, probes in heads.items():
            for probe, m in probes.items():
                tsv.append(f"{li}\t{hi}\t{probe}\t{baseline[probe]:.4f}\t"
                           f"{m['accuracy']:.4f}\t{m['accuracy_drop']:+.4f}\t{m['macro_f1']:.4f}")
    (OUTPUT_DIR/"results.tsv").write_text("\n".join(tsv)+"\n")

    # head_importance.json
    importance = {p: {} for p in PROBES}
    for li, heads in ablations.items():
        for p in PROBES: importance[p][str(li)] = {}
        for hi, probes in heads.items():
            for p in PROBES: importance[p][str(li)][str(hi)] = probes[p]["accuracy_drop"]
    (OUTPUT_DIR/"head_importance.json").write_text(json.dumps(importance, indent=2))

    # failure_cases.csv
    fc = ["probe,layer,head,drop,filename,true_label,predicted_label"]
    for probe in PROBES:
        best_drop, best_l, best_h = -1.0, 0, 0
        for li, heads in ablations.items():
            for hi, probes in heads.items():
                if probes[probe]["accuracy_drop"] > best_drop:
                    best_drop, best_l, best_h = probes[probe]["accuracy_drop"], li, hi
        for f in ablations[best_l][best_h][probe]["failures"][:20]:
            fc.append(f"{probe},{best_l},{best_h},{best_drop:.4f},"
                      f"{f['filename']},{f['true_label']},{f['predicted_label']}")
    (OUTPUT_DIR/"failure_cases.csv").write_text("\n".join(fc)+"\n")

    # deliverable.md
    sections = []
    for probe in PROBES:
        rows = sorted(
            [(li, hi, ablations[li][hi][probe]["accuracy_drop"])
             for li in ablations for hi in ablations[li]],
            key=lambda x: x[2], reverse=True)[:5]
        tbl = [f"### {probe.capitalize()} (baseline={baseline[probe]:.4f})",
               "| Rank | Layer | Head | Drop |", "| ---: | ---: | ---: | ---: |"]
        tbl += [f"| {r+1} | {l} | {h} | {d:+.4f} |" for r,(l,h,d) in enumerate(rows)]
        sections.append("\n".join(tbl))

    md = "\n\n".join([
        "# Project 2 Deliverable: The Attention Surgery Experiment",
        "## Baseline Accuracies\n| Probe | Accuracy |\n| --- | ---: |\n" +
        "\n".join(f"| {p} | {baseline[p]:.4f} |" for p in ["clean"]+PROBES),
        "## Top-5 Critical Heads per Probe\n" + "\n\n".join(sections),
        "## Interpretation\n"
        "- Large positive drop = head is critical for that probe.\n"
        "- Texture-critical heads likely encode local colour/frequency patterns.\n"
        "- Shape-critical heads likely encode global contour structure.\n"
        "- Spatial-critical heads likely encode positional patch relationships.",
        "## Output Files\n`summary.json` · `results.tsv` · `head_importance.json` · `failure_cases.csv`"
    ])
    (OUTPUT_DIR/"project2_deliverable.md").write_text(md)

    # ── Head importance heatmaps ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        probe_titles = {"texture": "Texture (color-jitter probe)",
                        "shape":   "Shape (rotation probe)",
                        "spatial": "Spatial (crop probe)"}

        for ax, probe in zip(axes, PROBES):
            # Build 12×12 matrix  [layer, head]
            n_layers = max(int(l) for l in importance[probe]) + 1
            n_heads  = max(int(h) for h in importance[probe]["0"]) + 1
            matrix   = np.zeros((n_layers, n_heads))
            for li in importance[probe]:
                for hi, val in importance[probe][li].items():
                    matrix[int(li), int(hi)] = val

            im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
            ax.set_title(probe_titles[probe], fontsize=13, fontweight="bold")
            ax.set_xlabel("Head index", fontsize=11)
            ax.set_ylabel("Layer index", fontsize=11)
            ax.set_xticks(range(n_heads))
            ax.set_yticks(range(n_layers))
            plt.colorbar(im, ax=ax, label="Accuracy drop (ablated − baseline)")

        fig.suptitle("Head Importance Map — Project 2: Attention Surgery",
                     fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        heatmap_path = OUTPUT_DIR / "head_importance_map.png"
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved heatmap → {heatmap_path}")
    except Exception as e:
        print(f"WARNING: Could not generate heatmap ({e})")

    print(f"\nDone. Outputs written to {OUTPUT_DIR}")
    print("Files:", [f.name for f in OUTPUT_DIR.iterdir() if f.suffix in {".json",".tsv",".csv",".md"}])

run()

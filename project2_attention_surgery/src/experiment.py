from __future__ import annotations

"""
Core experiment logic for Project 2: Attention Surgery.

Loads pretrained ViT-B/16, injects per-head ablation hooks, and measures
accuracy drop on three probes (texture / shape / spatial).
"""

import argparse
import json
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from prepare import (
    DATASET_ROOT,
    DEFAULT_EVAL_BATCH_SIZE,
    PROBES,
    VIT_NUM_HEADS,
    VIT_NUM_LAYERS,
    dataset_summary,
    evaluate_model,
    make_probe_loaders,
    save_json,
    set_seed,
)


# ---------------------------------------------------------------------------
# Device selection (P100-safe, mirrors project 1)
# ---------------------------------------------------------------------------
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        # Check compute capability directly — P100 is sm_60, modern PyTorch
        # requires sm_70+. Avoid running any kernel that would fail silently.
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


# ---------------------------------------------------------------------------
# ViT attention-head ablation hook
# ---------------------------------------------------------------------------
class _UniformAttentionHook:
    """
    Replaces the attention weights of a single head with a uniform distribution
    (all tokens attend equally) so that head contributes no selective information.

    Hooks into torchvision's MultiheadAttention by wrapping the scaled_dot_product
    attention call via a forward hook on the encoder block's self_attention module.
    """

    def __init__(self, head_idx: int, num_heads: int):
        self.head_idx = head_idx
        self.num_heads = num_heads
        self._handle = None

    def _hook(self, module, args, kwargs, output):
        # output of MultiheadAttention.forward is (attn_output, attn_weights)
        # attn_output shape: (B, seq_len, embed_dim)
        # We need to recompute the output for the ablated head.
        # torchvision ViT uses F.multi_head_attention_forward internally;
        # we intercept at the module level and patch the projection.
        # Strategy: zero out the contribution of this head in the output projection.
        # The in_proj weight is [3*embed_dim, embed_dim]; out_proj is [embed_dim, embed_dim].
        # Simpler & exact: register a pre-forward hook that masks the out_proj rows
        # corresponding to this head. We do that via a separate weight-mask approach below.
        return output  # placeholder — see _HeadOutputMasker below

    def register(self, mha_module: nn.MultiheadAttention):
        self._handle = mha_module.register_forward_hook(self._hook, with_kwargs=True)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class _HeadOutputMasker:
    """
    Zero-masks the out_proj rows that correspond to `head_idx` so that head
    contributes nothing to the residual stream — equivalent to ablating it.

    This is cleaner than patching attention weights because torchvision's MHA
    fuses the attention computation and doesn't expose per-head weights easily.
    """

    def __init__(self, head_idx: int, embed_dim: int, num_heads: int):
        self.head_idx = head_idx
        self.head_dim = embed_dim // num_heads
        self.start = head_idx * self.head_dim
        self.end = self.start + self.head_dim
        self._handle = None
        self._saved_weight: torch.Tensor | None = None
        self._saved_bias: torch.Tensor | None = None

    @contextmanager
    def ablated(self, mha_module: nn.MultiheadAttention):
        """Context manager: temporarily zero the head's out_proj slice."""
        w = mha_module.out_proj.weight  # (embed_dim, embed_dim)
        b = mha_module.out_proj.bias    # (embed_dim,)

        saved_w = w[:, self.start: self.end].clone()
        saved_b = b.clone() if b is not None else None

        with torch.no_grad():
            w[:, self.start: self.end] = 0.0

        try:
            yield
        finally:
            with torch.no_grad():
                w[:, self.start: self.end] = saved_w
                if saved_b is not None and b is not None:
                    b.copy_(saved_b)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def load_vit(device: torch.device) -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # Replace the classification head for 10 classes
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 10)
    # Freeze everything except the new head
    for name, param in model.named_parameters():
        if "heads" not in name:
            param.requires_grad_(False)
    model = model.to(device)
    model.eval()
    return model


def _get_encoder_block(model: nn.Module, layer_idx: int) -> nn.Module:
    return model.encoder.layers[layer_idx]


def _get_mha(model: nn.Module, layer_idx: int) -> nn.MultiheadAttention:
    return _get_encoder_block(model, layer_idx).self_attention


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------
def run_ablation(
    model: nn.Module,
    probe_loaders: dict,
    device: torch.device,
    layers: list[int],
    heads: list[int],
) -> dict[str, object]:
    """
    For each (layer, head) pair, ablate that head and evaluate on all probes.
    Returns a nested dict: results[layer][head][probe] = {accuracy, macro_f1, drop}
    """
    embed_dim = model.hidden_dim  # 768 for ViT-B/16

    # Baseline: evaluate intact model on all probes
    print("=== Baseline (no ablation) ===")
    baseline: dict[str, float] = {}
    for probe_name, loader in probe_loaders.items():
        result = evaluate_model(model, loader, device)
        baseline[probe_name] = result.accuracy
        print(f"  probe={probe_name} accuracy={result.accuracy:.4f} macro_f1={result.macro_f1:.4f}")

    results: dict[str, object] = {"baseline": baseline, "ablations": {}}
    ablations = results["ablations"]

    total = len(layers) * len(heads)
    done = 0
    for layer_idx in layers:
        ablations[layer_idx] = {}
        mha = _get_mha(model, layer_idx)
        masker = _HeadOutputMasker(
            head_idx=0,  # placeholder; we'll set per head
            embed_dim=embed_dim,
            num_heads=VIT_NUM_HEADS,
        )
        for head_idx in heads:
            done += 1
            masker.head_idx = head_idx
            masker.start = head_idx * masker.head_dim
            masker.end = masker.start + masker.head_dim

            head_results: dict[str, object] = {}
            with masker.ablated(mha):
                for probe_name, loader in probe_loaders.items():
                    r = evaluate_model(model, loader, device)
                    drop = baseline[probe_name] - r.accuracy
                    head_results[probe_name] = {
                        "accuracy": r.accuracy,
                        "macro_f1": r.macro_f1,
                        "accuracy_drop": drop,
                        "failures": r.failures[:10],
                    }

            ablations[layer_idx][head_idx] = head_results
            print(
                f"[{done}/{total}] layer={layer_idx} head={head_idx} | "
                + " | ".join(
                    f"{p}=drop{head_results[p]['accuracy_drop']:+.4f}"
                    for p in PROBES
                )
            )

    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_results_tsv(results: dict, output_path: Path) -> None:
    lines = ["layer\thead\tprobe\tbaseline_accuracy\tablated_accuracy\taccuracy_drop\tmacro_f1"]
    baseline = results["baseline"]
    for layer_idx, heads in results["ablations"].items():
        for head_idx, probes in heads.items():
            for probe_name, metrics in probes.items():
                lines.append(
                    f"{layer_idx}\t{head_idx}\t{probe_name}\t"
                    f"{baseline[probe_name]:.4f}\t"
                    f"{metrics['accuracy']:.4f}\t"
                    f"{metrics['accuracy_drop']:+.4f}\t"
                    f"{metrics['macro_f1']:.4f}"
                )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_head_importance(results: dict, output_path: Path) -> None:
    """
    Writes a JSON with shape importance[probe][layer][head] = accuracy_drop.
    Positive drop = head was important (ablating it hurt accuracy).
    """
    importance: dict[str, dict] = {p: {} for p in PROBES}
    for layer_idx, heads in results["ablations"].items():
        for probe in PROBES:
            importance[probe][str(layer_idx)] = {}
        for head_idx, probes in heads.items():
            for probe in PROBES:
                importance[probe][str(layer_idx)][str(head_idx)] = probes[probe]["accuracy_drop"]
    output_path.write_text(json.dumps(importance, indent=2), encoding="utf-8")


def write_failure_cases(results: dict, output_path: Path) -> None:
    """
    For each probe, find the single most critical head (highest drop) and
    write up to 20 of its failure cases.
    """
    lines = ["probe,layer,head,accuracy_drop,filename,true_label,predicted_label"]
    for probe in PROBES:
        best_drop = -1.0
        best_layer = best_head = 0
        for layer_idx, heads in results["ablations"].items():
            for head_idx, probes in heads.items():
                drop = probes[probe]["accuracy_drop"]
                if drop > best_drop:
                    best_drop = drop
                    best_layer = layer_idx
                    best_head = head_idx
        failures = results["ablations"][best_layer][best_head][probe]["failures"][:20]
        for f in failures:
            lines.append(
                f"{probe},{best_layer},{best_head},{best_drop:.4f},"
                f"{f['filename']},{f['true_label']},{f['predicted_label']}"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_deliverable(results: dict, output_path: Path) -> None:
    baseline = results["baseline"]

    # Build heatmap summary table per probe (top-5 most critical heads)
    probe_sections = []
    for probe in PROBES:
        rows = []
        for layer_idx, heads in results["ablations"].items():
            for head_idx, probes in heads.items():
                rows.append((layer_idx, head_idx, probes[probe]["accuracy_drop"]))
        rows.sort(key=lambda x: x[2], reverse=True)
        top5 = rows[:5]
        table = [
            f"### {probe.capitalize()} probe (baseline acc={baseline[probe]:.4f})",
            "",
            "| Rank | Layer | Head | Accuracy Drop |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for rank, (l, h, d) in enumerate(top5, 1):
            table.append(f"| {rank} | {l} | {h} | {d:+.4f} |")
        probe_sections.append("\n".join(table))

    output_lines = [
        "# Project 2 Deliverable: The Attention Surgery Experiment",
        "",
        "## Setup",
        "",
        "- Model: `vit_b_16` (ImageNet pretrained, classification head replaced for 10 classes)",
        "- Ablation: zero-mask the out_proj slice for each head independently",
        f"- Layers ablated: {sorted(results['ablations'].keys())}",
        "- Probes: texture (color-jitter), shape (rotation), spatial (crop)",
        "",
        "## Baseline Accuracies",
        "",
        "| Probe | Accuracy |",
        "| --- | ---: |",
        *[f"| {p} | {baseline[p]:.4f} |" for p in ["clean"] + PROBES],
        "",
        "## Top-5 Most Critical Heads per Probe",
        "",
        *[s + "\n" for s in probe_sections],
        "## Interpretation",
        "",
        "- A large positive drop when a head is ablated means that head carries information",
        "  critical for that probe's task.",
        "- Heads that are critical for **texture** but not shape/spatial likely encode",
        "  local colour/frequency patterns.",
        "- Heads critical for **shape** likely encode global contour or pose-invariant structure.",
        "- Heads critical for **spatial** likely encode positional relationships between patches.",
        "",
        "## Artifact Files",
        "",
        "- `summary.json` — full per-head metrics",
        "- `results.tsv` — flat table for analysis",
        "- `head_importance.json` — importance[probe][layer][head]",
        "- `failure_cases.csv` — misclassified examples for top critical heads",
        "",
    ]
    output_path.write_text("\n".join(output_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Project 2: Attention Surgery experiment runner.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layers", nargs="+", type=int,
        default=list(range(VIT_NUM_LAYERS)),
        help="Which ViT encoder layers to ablate (0-indexed). Default: all 12.",
    )
    parser.add_argument(
        "--heads", nargs="+", type=int,
        default=list(range(VIT_NUM_HEADS)),
        help="Which attention heads to ablate per layer (0-indexed). Default: all 12.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)  # suppress PIL palette warnings
    set_seed(args.seed)
    device = _get_device()
    output_dir = Path(__file__).resolve().parent

    print(json.dumps(dataset_summary(args.dataset_root), indent=2))
    print(f"Running on device: {device}")
    print(f"Ablating layers={args.layers}, heads={args.heads}")

    probe_loaders = make_probe_loaders(args.dataset_root, args.eval_batch_size)
    model = load_vit(device)

    results = run_ablation(
        model=model,
        probe_loaders=probe_loaders,
        device=device,
        layers=args.layers,
        heads=args.heads,
    )

    save_json(results, output_dir / "summary.json")
    write_results_tsv(results, output_dir / "results.tsv")
    write_head_importance(results, output_dir / "head_importance.json")
    write_failure_cases(results, output_dir / "failure_cases.csv")
    write_deliverable(results, output_dir / "project2_deliverable.md")
    print(f"Wrote artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

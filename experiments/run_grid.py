"""
AI Anatomy — Experiment Runner

Runs the 10×10 grid: 10 tools × 10 tasks × 5 variants = 500 tool runs.
Each cell scored 0/1/2 based on whether the tool finds task-relevant features.

Scoring criteria:
  0 = Tool finds no features with logit effect on target token above threshold
  1 = Tool finds relevant features (logit effect on target > threshold)
  2 = Tool finds interpretable circuit (multi-hop path to target in attribution graph)

Score 2 only applicable for CLT/crosscoder tools.
"""

import sys
sys.path.insert(0, "/mnt/storage/sandeep/priyansh/Gemma-Scope-2-Study")

import torch
import numpy as np
import json
import time
import gc
import os
from dataclasses import dataclass, field

from src.loader import (
    load_gemma3_1b, load_sae, load_transcoder, load_crosscoder, load_clt,
    GEMMA3_1B_CROSSCODER_LAYERS, GEMMA3_1B_NUM_LAYERS,
)
from src.hooks import (
    gather_residual_activations, gather_mlp_out_activations,
    gather_attn_out_activations, gather_transcoder_activations,
    gather_crosscoder_activations, gather_clt_activations,
)
from src.metrics import compute_fvu, compute_l0, cross_entropy_loss
from src.attribution import build_attribution_graph, prune_graph, compute_graph_metrics


# ============================================================
# Data structures
# ============================================================

@dataclass
class CellResult:
    """Result for one (tool, task, variant) combination."""
    tool_id: str
    task_id: str
    variant_idx: int
    prompt_text: str
    target_tokens: list

    # Metrics
    fvu: float = 0.0
    l0: float = 0.0
    delta_loss: float = 0.0

    # Task-relevance
    n_relevant_features: int = 0
    top_relevant_feature: int = -1
    top_logit_effect_on_target: float = 0.0
    target_token_rank: int = -1  # where target appears in model's predictions

    # Circuit (CLT only)
    has_circuit: bool = False
    circuit_path_length: float = 0.0
    circuit_n_nodes: int = 0

    # Score
    score: int = 0  # 0, 1, or 2

    # Timing
    run_time: float = 0.0


@dataclass
class GridResult:
    """Full 10×10 grid results."""
    cells: list = field(default_factory=list)  # list of CellResult
    metadata: dict = field(default_factory=dict)

    def get_cell(self, tool_id, task_id, variant_idx=None):
        """Get cell(s) for a specific tool-task pair."""
        matches = [c for c in self.cells
                   if c.tool_id == tool_id and c.task_id == task_id]
        if variant_idx is not None:
            matches = [c for c in matches if c.variant_idx == variant_idx]
        return matches

    def get_score(self, tool_id, task_id):
        """Get aggregated score for a tool-task pair (max across variants)."""
        cells = self.get_cell(tool_id, task_id)
        if not cells:
            return 0
        return max(c.score for c in cells)

    def get_mean_fvu(self, tool_id, task_id):
        """Get mean FVU across variants."""
        cells = self.get_cell(tool_id, task_id)
        if not cells:
            return 0.0
        return np.mean([c.fvu for c in cells])


# ============================================================
# Tool definitions
# ============================================================

TOOL_IDS = [
    "resid_sae",
    "mlp_sae",
    "attn_sae",
    "transcoder_noskip",
    "transcoder_skip",
    "crosscoder",
    "clt",
]

TOOL_LABELS = {
    "resid_sae": "Residual SAE",
    "mlp_sae": "MLP SAE",
    "attn_sae": "Attention SAE",
    "transcoder_noskip": "Transcoder",
    "transcoder_skip": "Skip Transcoder",
    "crosscoder": "Crosscoder",
    "clt": "CLT",
}


# ============================================================
# Core analysis: check if tool finds task-relevant features
# ============================================================

def check_target_relevance(
    model, tokenizer, features, decoder_weights, target_tokens,
    d_model_match=True,
    top_k_features=20,
    relevance_threshold=0.3,
):
    """
    Check if any active features have logit effects on the target token.

    Args:
        features: (seq_len, d_sae) — feature activations
        decoder_weights: (d_sae, d_model) — decoder matrix
        target_tokens: list of acceptable target token strings
        d_model_match: if False, skip logit effect (attn SAE has different d_model)
        top_k_features: how many top features to check
        relevance_threshold: minimum logit effect to count as relevant

    Returns:
        (n_relevant, best_feature_idx, best_logit_effect, target_rank)
    """
    if not d_model_match:
        # Can't compute logit effects for attention SAE (different space)
        # Fall back to counting active features as a proxy
        n_active = (features[1:] > 0).any(dim=0).sum().item()
        return 0, -1, 0.0, -1

    # Get target token IDs
    target_ids = []
    for t in target_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        target_ids.extend(ids)
    # Also try with space prefix
    for t in target_tokens:
        ids = tokenizer.encode(" " + t, add_special_tokens=False)
        target_ids.extend(ids)
    target_ids = list(set(target_ids))

    if not target_ids:
        return 0, -1, 0.0, -1

    # Effective unembedding
    w_unembed = model.lm_head.weight
    ln_weight = model.model.norm.weight
    w_eff = (w_unembed * ln_weight).float()

    # Get top features at last position by activation
    last_pos_acts = features[-1]  # (d_sae,)
    top_vals, top_idxs = last_pos_acts.topk(min(top_k_features, (last_pos_acts > 0).sum().item()))

    n_relevant = 0
    best_idx = -1
    best_effect = 0.0

    for val, idx in zip(top_vals, top_idxs):
        if val.item() <= 0:
            continue

        # Compute this feature's logit effect on target tokens
        dec_vec = decoder_weights[idx.item()].float()
        logit_effects = w_eff @ dec_vec  # (vocab,)

        for tid in target_ids:
            effect = logit_effects[tid].item()
            if abs(effect) > relevance_threshold:
                n_relevant += 1
                if abs(effect) > abs(best_effect):
                    best_effect = effect
                    best_idx = idx.item()

    # Also check where target appears in model's overall predictions
    # (even if no single feature promotes it)
    target_rank = -1
    # This would need the model's actual logits — skip for now, compute separately

    return n_relevant, best_idx, best_effect, target_rank


def check_model_prediction(model, tokenizer, inputs, target_tokens):
    """Check where target token appears in model's top predictions."""
    with torch.no_grad():
        logits = model(inputs).logits[0, -1]

    target_ids = set()
    for t in target_tokens:
        target_ids.update(tokenizer.encode(t, add_special_tokens=False))
        target_ids.update(tokenizer.encode(" " + t, add_special_tokens=False))

    # Get rank of best target token
    sorted_ids = logits.argsort(descending=True)
    for rank, tid in enumerate(sorted_ids[:100]):
        if tid.item() in target_ids:
            return rank + 1  # 1-indexed
    return -1  # not in top 100


# ============================================================
# Per-tool runners (lightweight, metrics only)
# ============================================================

def run_single_layer_tool(
    model, tokenizer, inputs, layer, tool_id, target_tokens,
    width="65k", l0="medium", cache_dir=None, device="cuda",
):
    """Run a single-layer tool and compute relevance metrics."""
    t0 = time.time()

    if tool_id == "resid_sae":
        sae = load_sae(layer, "resid_post", width, l0, device=device, cache_dir=cache_dir)
        acts = gather_residual_activations(model, layer, inputs)
        features = sae.encode(acts.float())
        recon = sae.decode(features)
        fvu = compute_fvu(recon, acts).item()
        n_rel, best_f, best_eff, _ = check_target_relevance(
            model, tokenizer, features, sae.w_dec, target_tokens)
        del sae

    elif tool_id == "mlp_sae":
        sae = load_sae(layer, "mlp_out", width, l0, device=device, cache_dir=cache_dir)
        acts = gather_mlp_out_activations(model, layer, inputs)
        features = sae.encode(acts.float())
        recon = sae.decode(features)
        fvu = compute_fvu(recon, acts).item()
        n_rel, best_f, best_eff, _ = check_target_relevance(
            model, tokenizer, features, sae.w_dec, target_tokens)
        del sae

    elif tool_id == "attn_sae":
        sae = load_sae(layer, "attn_out", width, l0, device=device, cache_dir=cache_dir)
        acts = gather_attn_out_activations(model, layer, inputs)
        features = sae.encode(acts.float())
        recon = sae.decode(features)
        fvu = compute_fvu(recon, acts).item()
        # Attention SAE is in different space — can't compute logit effects directly
        n_rel, best_f, best_eff, _ = check_target_relevance(
            model, tokenizer, features, sae.w_dec, target_tokens, d_model_match=False)
        del sae

    elif tool_id == "transcoder_noskip":
        tc = load_transcoder(layer, width, l0, affine=False, device=device, cache_dir=cache_dir)
        cache = gather_transcoder_activations(model, layer, inputs)
        features = tc.encode(cache["input"].float())
        recon = tc.forward(cache["input"].float())
        fvu = compute_fvu(recon, cache["target"]).item()
        n_rel, best_f, best_eff, _ = check_target_relevance(
            model, tokenizer, features, tc.w_dec, target_tokens)
        del tc

    elif tool_id == "transcoder_skip":
        tc = load_transcoder(layer, width, l0, affine=True, device=device, cache_dir=cache_dir)
        cache = gather_transcoder_activations(model, layer, inputs)
        features = tc.encode(cache["input"].float())
        recon = tc.forward(cache["input"].float())
        fvu = compute_fvu(recon, cache["target"]).item()
        n_rel, best_f, best_eff, _ = check_target_relevance(
            model, tokenizer, features, tc.w_dec, target_tokens)
        del tc

    else:
        raise ValueError(f"Unknown single-layer tool: {tool_id}")

    l0_val = compute_l0(features).item()
    run_time = time.time() - t0

    torch.cuda.empty_cache()

    return {
        "fvu": fvu,
        "l0": l0_val,
        "n_relevant": n_rel,
        "best_feature": best_f,
        "best_logit_effect": best_eff,
        "run_time": run_time,
    }


def run_crosscoder_tool(
    model, tokenizer, inputs, target_tokens,
    width="262k", l0="medium", cache_dir=None, device="cuda",
):
    """Run crosscoder and compute relevance metrics."""
    t0 = time.time()
    cc = load_crosscoder(width, l0, device=device, cache_dir=cache_dir)
    layers = GEMMA3_1B_CROSSCODER_LAYERS

    cc_input = gather_crosscoder_activations(model, layers, inputs).float()
    features = cc.encode(cc_input)
    recon = cc.forward(cc_input)

    fvu = compute_fvu(recon, cc_input).item()
    l0_val = compute_l0(features).item()

    # Check relevance across all layers
    # Use the last crosscoder layer's features for logit relevance
    last_layer_idx = len(layers) - 1
    last_features = features[:, last_layer_idx, :]  # (seq, d_sae)
    last_decoder = cc.w_dec.data[last_layer_idx, :, last_layer_idx, :]  # (d_sae, d_model)

    n_rel, best_f, best_eff, _ = check_target_relevance(
        model, tokenizer, last_features, last_decoder, target_tokens)

    run_time = time.time() - t0
    del cc
    torch.cuda.empty_cache()

    return {
        "fvu": fvu,
        "l0": l0_val,
        "n_relevant": n_rel,
        "best_feature": best_f,
        "best_logit_effect": best_eff,
        "run_time": run_time,
    }


def run_clt_tool(
    model, tokenizer, inputs, target_tokens, clt,
    device="cuda",
):
    """Run CLT (pre-loaded) and compute relevance + circuit metrics."""
    t0 = time.time()

    clt_inputs, clt_targets = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)
    if next(clt.parameters()).dtype == torch.float16:
        clt_inputs = clt_inputs.half()
        clt_targets = clt_targets.half()

    features = clt.encode(clt_inputs)
    recon = clt.forward(clt_inputs)

    fvu = compute_fvu(recon, clt_targets).item()
    l0_val = compute_l0(features).item()

    # Check relevance: use last few layers' features
    # Sum decoder contributions to residual for features at layers 20-25
    w_eff = (model.lm_head.weight * model.model.norm.weight).float()
    target_ids = set()
    for t in target_tokens:
        target_ids.update(tokenizer.encode(t, add_special_tokens=False))
        target_ids.update(tokenizer.encode(" " + t, add_special_tokens=False))

    n_relevant = 0
    best_feature = (-1, -1)  # (layer, idx)
    best_effect = 0.0

    for layer in range(20, 26):
        layer_features = features[-1, layer, :]  # last position
        active_mask = layer_features > 0
        active_idxs = active_mask.nonzero(as_tuple=True)[0]

        for idx in active_idxs:
            dec_sum = clt.w_dec.data[layer, idx.item(), layer:].sum(dim=0).float()
            for tid in target_ids:
                effect = (w_eff[tid] * dec_sum).sum().item()
                if abs(effect) > 0.3:
                    n_relevant += 1
                    if abs(effect) > abs(best_effect):
                        best_effect = effect
                        best_feature = (layer, idx.item())

    # Try attribution graph for circuit detection
    has_circuit = False
    path_length = 0.0
    n_nodes = 0

    try:
        prompt_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
        graph = build_attribution_graph(
            model, clt, tokenizer, prompt_text,
            top_k_output_tokens=5,
            min_ff_edge_weight=50.0,
            min_fl_edge_weight=5.0,
        )
        pruned = prune_graph(graph, top_k_edges_per_node=3, max_feature_nodes=30, min_edge_weight=10.0)
        metrics = compute_graph_metrics(pruned)

        n_nodes = metrics.get("num_feature_nodes", 0)
        ff_edges = metrics.get("feature_to_feature_edges", 0)
        path_length = metrics.get("avg_path_length", 0.0)

        if ff_edges > 0 and path_length > 1.0:
            has_circuit = True
    except Exception as e:
        print(f"    Attribution graph failed: {e}")

    run_time = time.time() - t0

    return {
        "fvu": fvu,
        "l0": l0_val,
        "n_relevant": n_relevant,
        "best_feature": best_feature,
        "best_logit_effect": best_effect,
        "has_circuit": has_circuit,
        "path_length": path_length,
        "n_nodes": n_nodes,
        "run_time": run_time,
    }


# ============================================================
# Main grid runner
# ============================================================

def run_grid(
    model, tokenizer, tasks_module, clt,
    target_layer=17,
    width="65k",
    l0="medium",
    cache_dir=None,
    device="cuda",
    max_variants=5,
    skip_tools=None,
):
    """
    Run the full 10×10 grid.

    Args:
        tasks_module: the prompts module (has TASKS dict)
        clt: pre-loaded CLT
        target_layer: layer for single-layer tools
        max_variants: how many prompt variants per task (1-5)
        skip_tools: list of tool_ids to skip (for incremental runs)
    """
    if skip_tools is None:
        skip_tools = []

    grid = GridResult(metadata={
        "target_layer": target_layer,
        "width": width,
        "l0": l0,
        "max_variants": max_variants,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    task_ids = tasks_module.get_all_task_ids()
    total = len(task_ids) * len(TOOL_IDS) * max_variants
    completed = 0

    for task_id in task_ids:
        task_info = tasks_module.get_task_info(task_id)
        prompts = tasks_module.get_task_prompts(task_id)[:max_variants]

        print(f"\n{'='*60}")
        print(f"TASK: {task_info['name']} ({task_id})")
        print(f"{'='*60}")

        for vi, prompt in enumerate(prompts):
            text = prompt["text"]
            targets = prompt["target"]
            inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)

            # Check model prediction rank for this target
            target_rank = check_model_prediction(model, tokenizer, inputs, targets)

            for tool_id in TOOL_IDS:
                if tool_id in skip_tools:
                    continue

                completed += 1
                label = TOOL_LABELS[tool_id]
                print(f"  [{completed}/{total}] {label} on v{vi+1}: '{text[:50]}...'", end="")

                cell = CellResult(
                    tool_id=tool_id,
                    task_id=task_id,
                    variant_idx=vi,
                    prompt_text=text,
                    target_tokens=targets,
                    target_token_rank=target_rank,
                )

                try:
                    if tool_id in ["resid_sae", "mlp_sae", "attn_sae",
                                   "transcoder_noskip", "transcoder_skip"]:
                        result = run_single_layer_tool(
                            model, tokenizer, inputs, target_layer,
                            tool_id, targets, width, l0, cache_dir, device,
                        )
                        cell.fvu = result["fvu"]
                        cell.l0 = result["l0"]
                        cell.n_relevant_features = result["n_relevant"]
                        cell.top_relevant_feature = result["best_feature"]
                        cell.top_logit_effect_on_target = result["best_logit_effect"]
                        cell.run_time = result["run_time"]

                        # Score: 0 or 1 (single-layer tools can't get 2)
                        cell.score = 1 if result["n_relevant"] > 0 else 0

                    elif tool_id == "crosscoder":
                        result = run_crosscoder_tool(
                            model, tokenizer, inputs, targets,
                            "262k", l0, cache_dir, device,
                        )
                        cell.fvu = result["fvu"]
                        cell.l0 = result["l0"]
                        cell.n_relevant_features = result["n_relevant"]
                        cell.top_logit_effect_on_target = result["best_logit_effect"]
                        cell.run_time = result["run_time"]
                        cell.score = 1 if result["n_relevant"] > 0 else 0

                    elif tool_id == "clt":
                        result = run_clt_tool(
                            model, tokenizer, inputs, targets, clt, device,
                        )
                        cell.fvu = result["fvu"]
                        cell.l0 = result["l0"]
                        cell.n_relevant_features = result["n_relevant"]
                        cell.top_logit_effect_on_target = result["best_logit_effect"]
                        cell.has_circuit = result["has_circuit"]
                        cell.circuit_path_length = result["path_length"]
                        cell.circuit_n_nodes = result["n_nodes"]
                        cell.run_time = result["run_time"]

                        # Score: 0, 1, or 2
                        if result["has_circuit"]:
                            cell.score = 2
                        elif result["n_relevant"] > 0:
                            cell.score = 1
                        else:
                            cell.score = 0

                    print(f"  → score={cell.score}, rel={cell.n_relevant_features}, "
                          f"fvu={cell.fvu:.4f}, {cell.run_time:.1f}s")

                except Exception as e:
                    print(f"  → FAILED: {e}")
                    cell.score = -1

                grid.cells.append(cell)

                # Periodic save
                if completed % 20 == 0:
                    save_grid(grid, "/mnt/storage/sandeep/priyansh/AI-Anatomy/outputs/grid_checkpoint.json")

    return grid


# ============================================================
# Grid visualization and export
# ============================================================

def get_score_matrix(grid, task_ids, tool_ids):
    """Extract the score matrix as a 2D numpy array."""
    matrix = np.zeros((len(task_ids), len(tool_ids)))
    for i, task_id in enumerate(task_ids):
        for j, tool_id in enumerate(tool_ids):
            matrix[i, j] = grid.get_score(tool_id, task_id)
    return matrix


def get_fvu_matrix(grid, task_ids, tool_ids):
    """Extract mean FVU matrix."""
    matrix = np.zeros((len(task_ids), len(tool_ids)))
    for i, task_id in enumerate(task_ids):
        for j, tool_id in enumerate(tool_ids):
            matrix[i, j] = grid.get_mean_fvu(tool_id, task_id)
    return matrix


def plot_grid_heatmap(grid, tasks_module, save_path):
    """Generate the centerpiece 10×10 heatmap (Figure 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    task_ids = tasks_module.get_all_task_ids()
    task_labels = [tasks_module.get_task_info(t)["name"] for t in task_ids]
    tool_labels = [TOOL_LABELS[t] for t in TOOL_IDS]

    matrix = get_score_matrix(grid, task_ids, TOOL_IDS)

    # Custom colormap: 0=white, 1=light blue, 2=dark blue, -1=gray
    cmap = mcolors.ListedColormap(["#f0f0f0", "#a8d5e2", "#1a5276"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(tool_labels)))
    ax.set_xticklabels(tool_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(task_labels)))
    ax.set_yticklabels(task_labels, fontsize=9)

    # Add score text in each cell
    for i in range(len(task_ids)):
        for j in range(len(TOOL_IDS)):
            score = int(matrix[i, j])
            text_color = "white" if score == 2 else "black"
            ax.text(j, i, str(score), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_title("AI Anatomy: Tool × Task Coverage Grid\n"
                 "0 = No relevant features | 1 = Relevant features found | 2 = Circuit traced",
                 fontsize=11, pad=15)

    plt.colorbar(im, ax=ax, ticks=[0, 1, 2], label="Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved grid heatmap: {save_path}")
    plt.close()


def save_grid(grid, path):
    """Save grid results to JSON."""
    data = {
        "metadata": grid.metadata,
        "cells": [],
    }
    for c in grid.cells:
        data["cells"].append({
            "tool_id": c.tool_id,
            "task_id": c.task_id,
            "variant_idx": c.variant_idx,
            "prompt_text": c.prompt_text,
            "target_tokens": c.target_tokens,
            "fvu": c.fvu,
            "l0": c.l0,
            "delta_loss": c.delta_loss,
            "n_relevant_features": c.n_relevant_features,
            "top_relevant_feature": c.top_relevant_feature
                if not isinstance(c.top_relevant_feature, tuple)
                else list(c.top_relevant_feature),
            "top_logit_effect_on_target": c.top_logit_effect_on_target,
            "target_token_rank": c.target_token_rank,
            "has_circuit": c.has_circuit,
            "circuit_path_length": c.circuit_path_length,
            "circuit_n_nodes": c.circuit_n_nodes,
            "score": c.score,
            "run_time": c.run_time,
        })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved grid: {path}")

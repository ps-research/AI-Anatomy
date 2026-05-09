"""
AI Anatomy — Experiment Runner (v3 — full 10x10)

10 tools:
  1. Behavioral Probing — does the model predict the target? relevance = 1/rank
  2. Linear Probe — sklearn LogisticRegression on cached activations
  3. Attention SAE — with W_O projection for logit effects
  4. MLP SAE
  5. Residual SAE
  6. Transcoder (no skip)
  7. Skip Transcoder (affine)
  8. Crosscoder (weakly causal, 4 layers)
  9. CLT (no affine skip)
  10. CLT + Affine

KNOWN ISSUES (to fix on vast.ai):
  - MLP SAE/transcoder relevance may be low due to threshold 1.0
  - CLT score=2 doesn't require relevant features (just any circuit)
"""

import sys
sys.path.insert(0, "/workspace/Gemma-Scope-2-Study")

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
    tool_id: str
    task_id: str
    variant_idx: int
    prompt_text: str
    target_tokens: list

    fvu: float = 0.0
    l0: float = 0.0

    n_relevant_features: int = 0
    max_logit_effect_on_target: float = 0.0
    mean_logit_effect_on_target: float = 0.0
    target_token_rank: int = -1

    # Probe-specific
    probe_accuracy: float = 0.0

    # Circuit (CLT only)
    has_circuit: bool = False
    circuit_path_length: float = 0.0
    circuit_n_nodes: int = 0
    circuit_ff_edges: int = 0

    score: int = 0
    run_time: float = 0.0


@dataclass
class GridResult:
    cells: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_cell(self, tool_id, task_id, variant_idx=None):
        matches = [c for c in self.cells
                   if c.tool_id == tool_id and c.task_id == task_id]
        if variant_idx is not None:
            matches = [c for c in matches if c.variant_idx == variant_idx]
        return matches

    def get_score(self, tool_id, task_id):
        cells = self.get_cell(tool_id, task_id)
        return max((c.score for c in cells), default=0)

    def get_mean_fvu(self, tool_id, task_id):
        cells = self.get_cell(tool_id, task_id)
        return np.mean([c.fvu for c in cells]) if cells else 0.0

    def get_mean_relevance(self, tool_id, task_id):
        cells = self.get_cell(tool_id, task_id)
        return np.mean([c.max_logit_effect_on_target for c in cells]) if cells else 0.0

    def get_mean_n_relevant(self, tool_id, task_id):
        cells = self.get_cell(tool_id, task_id)
        return np.mean([c.n_relevant_features for c in cells]) if cells else 0.0

    def get_mean_probe_acc(self, tool_id, task_id):
        cells = self.get_cell(tool_id, task_id)
        return np.mean([c.probe_accuracy for c in cells]) if cells else 0.0


# ============================================================
# Tool definitions — FULL 10
# ============================================================

TOOL_IDS = [
    "behavioral",
    "linear_probe",
    "attn_sae",
    "mlp_sae",
    "resid_sae",
    "transcoder_noskip",
    "transcoder_skip",
    "crosscoder",
    "clt_noskip",
    "clt_affine",
]

TOOL_LABELS = {
    "behavioral": "Behavioral",
    "linear_probe": "Probes",
    "attn_sae": "Attn SAE",
    "mlp_sae": "MLP SAE",
    "resid_sae": "Resid SAE",
    "transcoder_noskip": "Skip-TC",
    "transcoder_skip": "Skip-TC+Affine",
    "crosscoder": "Crosscoder",
    "clt_noskip": "CLT",
    "clt_affine": "CLT+Affine",
}


# ============================================================
# Helpers
# ============================================================

def resolve_target_token_ids(tokenizer, target_tokens):
    target_ids = set()
    for t in target_tokens:
        target_ids.update(tokenizer.encode(t, add_special_tokens=False))
        target_ids.update(tokenizer.encode(" " + t, add_special_tokens=False))
    return target_ids


def check_model_prediction(model, tokenizer, inputs, target_tokens):
    with torch.no_grad():
        logits = model(inputs).logits[0, -1]
    target_ids = resolve_target_token_ids(tokenizer, target_tokens)
    sorted_ids = logits.argsort(descending=True)
    for rank, tid in enumerate(sorted_ids[:100]):
        if tid.item() in target_ids:
            return rank + 1
    return -1


def get_effective_unembed(model):
    w_u = model.lm_head.weight
    ln_w = model.model.norm.weight
    return (w_u * ln_w).float()


def get_attn_wo_projection(model, layer):
    return model.model.layers[layer].self_attn.o_proj.weight.float()


def compute_relevance(decoder_weights, features_last_pos, w_eff, target_ids,
                      top_k_features=20, relevance_threshold=1.0):
    n_active = (features_last_pos > 0).sum().item()
    if n_active == 0:
        return 0, 0.0, 0.0

    top_vals, top_idxs = features_last_pos.topk(min(top_k_features, n_active))
    relevant_features = {}

    for val, idx in zip(top_vals, top_idxs):
        if val.item() <= 0:
            continue
        feat_idx = idx.item()
        dec_vec = decoder_weights[feat_idx].float()
        logit_effects = w_eff @ dec_vec

        best_effect = 0.0
        for tid in target_ids:
            effect = logit_effects[tid].item()
            if abs(effect) > abs(best_effect):
                best_effect = effect

        if abs(best_effect) > relevance_threshold:
            relevant_features[feat_idx] = best_effect

    n_relevant = len(relevant_features)
    if n_relevant == 0:
        return 0, 0.0, 0.0

    effects = list(relevant_features.values())
    max_effect = max(effects, key=abs)
    mean_effect = np.mean([abs(e) for e in effects])
    return n_relevant, max_effect, mean_effect


# ============================================================
# Tool 1: Behavioral Probing
# ============================================================

def run_behavioral(model, tokenizer, inputs, target_tokens, device="cuda"):
    """Simply check model's prediction rank for the target."""
    t0 = time.time()
    rank = check_model_prediction(model, tokenizer, inputs, target_tokens)

    # Relevance = inverse rank (higher = better prediction)
    if rank > 0 and rank <= 100:
        relevance = 1.0 / rank
    else:
        relevance = 0.0

    return {
        "target_rank": rank,
        "max_logit_effect": relevance,  # use relevance slot for 1/rank
        "n_relevant": 1 if rank <= 10 else 0,
        "run_time": time.time() - t0,
    }


# ============================================================
# Tool 2: Linear Probe
# ============================================================

def run_linear_probe(model, tokenizer, inputs, target_tokens, target_layer,
                     corpus_texts, device="cuda"):
    """
    Train a logistic regression probe to predict whether the target token
    follows, based on residual stream activations at target_layer.

    Uses the input prompt's activations as positive example and
    corpus_texts as negative examples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    t0 = time.time()

    # Positive: activation at last position of the input prompt
    acts = gather_residual_activations(model, target_layer, inputs)
    pos_act = acts[-1].float().cpu().numpy().reshape(1, -1)  # (1, d_model)

    # Negatives: last-position activations from corpus texts
    neg_acts = []
    for text in corpus_texts:
        neg_inputs = tokenizer.encode(text, return_tensors="pt",
                                      add_special_tokens=True,
                                      truncation=True, max_length=128).to(device)
        neg_act = gather_residual_activations(model, target_layer, neg_inputs)
        neg_acts.append(neg_act[-1].float().cpu().numpy())

    if len(neg_acts) < 3:
        return {"probe_accuracy": 0.0, "max_logit_effect": 0.0,
                "n_relevant": 0, "run_time": time.time() - t0}

    neg_matrix = np.stack(neg_acts)  # (n_neg, d_model)

    # Build dataset
    X = np.vstack([pos_act, neg_matrix])
    y = np.array([1] + [0] * len(neg_acts))

    # If we have too few samples for cross-val, just fit and score
    if len(y) < 5:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        acc = clf.score(X, y)
    else:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        try:
            scores = cross_val_score(clf, X, y, cv=min(3, len(y)), scoring="accuracy")
            acc = scores.mean()
        except Exception:
            clf.fit(X, y)
            acc = clf.score(X, y)

    # Use accuracy as the relevance metric
    # Above chance (0.5) = relevant
    relevance = max(0, acc - 0.5) * 2  # scale to [0, 1]

    return {
        "probe_accuracy": acc,
        "max_logit_effect": relevance,
        "n_relevant": 1 if acc > 0.6 else 0,
        "run_time": time.time() - t0,
    }


# ============================================================
# Tools 3-7: Single-layer Gemma Scope 2 tools
# ============================================================

def run_single_layer_tool(model, tokenizer, inputs, layer, tool_id, target_tokens,
                          width="65k", l0="medium", cache_dir=None, device="cuda"):
    t0 = time.time()
    target_ids = resolve_target_token_ids(tokenizer, target_tokens)
    w_eff = get_effective_unembed(model)

    if tool_id == "resid_sae":
        sae = load_sae(layer, "resid_post", width, l0, device=device, cache_dir=cache_dir)
        acts = gather_residual_activations(model, layer, inputs)
        features = sae.encode(acts.float())
        recon = sae.decode(features)
        fvu = compute_fvu(recon, acts).item()
        n_rel, max_eff, mean_eff = compute_relevance(
            sae.w_dec, features[-1], w_eff, target_ids)
        del sae

    elif tool_id == "mlp_sae":
        sae = load_sae(layer, "mlp_out", width, l0, device=device, cache_dir=cache_dir)
        acts = gather_mlp_out_activations(model, layer, inputs)
        features = sae.encode(acts.float())
        recon = sae.decode(features)
        fvu = compute_fvu(recon, acts).item()
        n_rel, max_eff, mean_eff = compute_relevance(
            sae.w_dec, features[-1], w_eff, target_ids)
        del sae

    elif tool_id == "attn_sae":
        sae = load_sae(layer, "attn_out", width, l0, device=device, cache_dir=cache_dir)
        acts = gather_attn_out_activations(model, layer, inputs)
        features = sae.encode(acts.float())
        recon = sae.decode(features)
        fvu = compute_fvu(recon, acts).item()
        w_o = get_attn_wo_projection(model, layer)
        projected_decoder = (sae.w_dec.float() @ w_o.T)
        n_rel, max_eff, mean_eff = compute_relevance(
            projected_decoder, features[-1], w_eff, target_ids)
        del sae, projected_decoder

    elif tool_id == "transcoder_noskip":
        tc = load_transcoder(layer, width, l0, affine=False, device=device, cache_dir=cache_dir)
        cache = gather_transcoder_activations(model, layer, inputs)
        features = tc.encode(cache["input"].float())
        recon = tc.forward(cache["input"].float())
        fvu = compute_fvu(recon, cache["target"]).item()
        n_rel, max_eff, mean_eff = compute_relevance(
            tc.w_dec, features[-1], w_eff, target_ids)
        del tc

    elif tool_id == "transcoder_skip":
        tc = load_transcoder(layer, width, l0, affine=True, device=device, cache_dir=cache_dir)
        cache = gather_transcoder_activations(model, layer, inputs)
        features = tc.encode(cache["input"].float())
        recon = tc.forward(cache["input"].float())
        fvu = compute_fvu(recon, cache["target"]).item()
        n_rel, max_eff, mean_eff = compute_relevance(
            tc.w_dec, features[-1], w_eff, target_ids)
        del tc

    else:
        raise ValueError(f"Unknown tool: {tool_id}")

    l0_val = compute_l0(features).item()
    run_time = time.time() - t0
    torch.cuda.empty_cache()

    return {
        "fvu": fvu, "l0": l0_val,
        "n_relevant": n_rel, "max_logit_effect": max_eff,
        "mean_logit_effect": mean_eff, "run_time": run_time,
    }


# ============================================================
# Tool 8: Crosscoder
# ============================================================

def run_crosscoder_tool(model, tokenizer, inputs, target_tokens,
                        width="262k", l0="medium", cache_dir=None, device="cuda"):
    t0 = time.time()
    target_ids = resolve_target_token_ids(tokenizer, target_tokens)
    w_eff = get_effective_unembed(model)

    cc = load_crosscoder(width, l0, device=device, cache_dir=cache_dir)
    layers = GEMMA3_1B_CROSSCODER_LAYERS

    cc_input = gather_crosscoder_activations(model, layers, inputs).float()
    features = cc.encode(cc_input)
    recon = cc.forward(cc_input)

    fvu = compute_fvu(recon, cc_input).item()
    l0_val = compute_l0(features).item()

    last_idx = len(layers) - 1
    last_features = features[-1, last_idx, :]
    last_decoder = cc.w_dec.data[last_idx, :, last_idx, :]

    n_rel, max_eff, mean_eff = compute_relevance(
        last_decoder, last_features, w_eff, target_ids)

    run_time = time.time() - t0
    del cc
    torch.cuda.empty_cache()

    return {
        "fvu": fvu, "l0": l0_val,
        "n_relevant": n_rel, "max_logit_effect": max_eff,
        "mean_logit_effect": mean_eff, "run_time": run_time,
    }


# ============================================================
# Tools 9-10: CLT (no-skip and affine)
# ============================================================

def run_clt_tool(model, tokenizer, inputs, target_tokens, clt, device="cuda"):
    t0 = time.time()
    target_ids = resolve_target_token_ids(tokenizer, target_tokens)
    w_eff = get_effective_unembed(model)

    clt_inputs, clt_targets = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)
    if next(clt.parameters()).dtype == torch.float16:
        clt_inputs = clt_inputs.half()
        clt_targets = clt_targets.half()

    features = clt.encode(clt_inputs)
    recon = clt.forward(clt_inputs)

    fvu = compute_fvu(recon, clt_targets).item()
    l0_val = compute_l0(features).item()

    # Check relevance across late layers
    relevant_features = {}
    for layer in range(18, 26):
        layer_features = features[-1, layer, :]
        n_active = (layer_features > 0).sum().item()
        if n_active == 0:
            continue
        top_vals, top_idxs = layer_features.topk(min(20, n_active))
        for val, idx in zip(top_vals, top_idxs):
            if val.item() <= 0:
                continue
            dec_sum = clt.w_dec.data[layer, idx.item(), layer:].sum(dim=0).float()
            for tid in target_ids:
                effect = (w_eff[tid] * dec_sum).sum().item()
                if abs(effect) > 1.0:
                    key = (layer, idx.item())
                    if key not in relevant_features or abs(effect) > abs(relevant_features[key]):
                        relevant_features[key] = effect

    n_rel = len(relevant_features)
    max_eff = max(relevant_features.values(), key=abs) if relevant_features else 0.0
    mean_eff = np.mean([abs(v) for v in relevant_features.values()]) if relevant_features else 0.0

    # Attribution graph
    has_circuit = False
    path_length = 0.0
    n_nodes = 0
    ff_edges = 0

    try:
        prompt_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
        graph = build_attribution_graph(
            model, clt, tokenizer, prompt_text,
            top_k_output_tokens=5,
            min_ff_edge_weight=50.0,
            min_fl_edge_weight=5.0,
        )
        pruned = prune_graph(graph, top_k_edges_per_node=3,
                            max_feature_nodes=30, min_edge_weight=10.0)
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
        "fvu": fvu, "l0": l0_val,
        "n_relevant": n_rel, "max_logit_effect": max_eff,
        "mean_logit_effect": mean_eff,
        "has_circuit": has_circuit, "path_length": path_length,
        "n_nodes": n_nodes, "ff_edges": ff_edges,
        "run_time": run_time,
    }


# ============================================================
# Negative corpus for linear probes
# ============================================================

PROBE_NEGATIVE_CORPUS = [
    "The weather today is sunny and warm.",
    "She went to the store to buy groceries.",
    "The football match ended in a draw.",
    "He likes to read books before sleeping.",
    "The conference will be held in March.",
    "They decided to paint the house blue.",
    "The train arrives at platform 3.",
    "She ordered a coffee with extra cream.",
    "The garden was full of colorful flowers.",
    "He fixed the broken shelf with glue.",
    "The movie starts at eight o'clock tonight.",
    "She finished her homework before dinner.",
    "The cat sat on the windowsill quietly.",
    "He drove to work through heavy traffic.",
    "The restaurant serves excellent Italian food.",
]


# ============================================================
# Main grid runner
# ============================================================

def run_grid(
    model, tokenizer, tasks_module,
    clt_affine, clt_noskip,
    target_layer=17,
    width="65k", l0="medium",
    cache_dir=None, device="cuda",
    max_variants=5, skip_tools=None,
):
    if skip_tools is None:
        skip_tools = []

    grid = GridResult(metadata={
        "target_layer": target_layer, "width": width, "l0": l0,
        "max_variants": max_variants,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    task_ids = tasks_module.get_all_task_ids()
    active_tools = [t for t in TOOL_IDS if t not in skip_tools]
    total = len(task_ids) * len(active_tools) * max_variants
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
            inputs = tokenizer.encode(text, return_tensors="pt",
                                      add_special_tokens=True).to(device)

            target_rank = check_model_prediction(model, tokenizer, inputs, targets)

            for tool_id in active_tools:
                completed += 1
                label = TOOL_LABELS[tool_id]
                print(f"  [{completed}/{total}] {label} on v{vi+1}: "
                      f"'{text[:45]}...'", end="", flush=True)

                cell = CellResult(
                    tool_id=tool_id, task_id=task_id, variant_idx=vi,
                    prompt_text=text, target_tokens=targets,
                    target_token_rank=target_rank,
                )

                try:
                    # ---- Behavioral ----
                    if tool_id == "behavioral":
                        result = run_behavioral(model, tokenizer, inputs, targets, device)
                        cell.max_logit_effect_on_target = result["max_logit_effect"]
                        cell.n_relevant_features = result["n_relevant"]
                        cell.target_token_rank = result["target_rank"]
                        cell.run_time = result["run_time"]
                        cell.score = 1 if result["n_relevant"] > 0 else 0

                    # ---- Linear Probe ----
                    elif tool_id == "linear_probe":
                        result = run_linear_probe(
                            model, tokenizer, inputs, targets,
                            target_layer, PROBE_NEGATIVE_CORPUS, device)
                        cell.probe_accuracy = result["probe_accuracy"]
                        cell.max_logit_effect_on_target = result["max_logit_effect"]
                        cell.n_relevant_features = result["n_relevant"]
                        cell.run_time = result["run_time"]
                        cell.score = 1 if result["n_relevant"] > 0 else 0

                    # ---- Single-layer tools ----
                    elif tool_id in ["resid_sae", "mlp_sae", "attn_sae",
                                     "transcoder_noskip", "transcoder_skip"]:
                        result = run_single_layer_tool(
                            model, tokenizer, inputs, target_layer,
                            tool_id, targets, width, l0, cache_dir, device)
                        cell.fvu = result["fvu"]
                        cell.l0 = result["l0"]
                        cell.n_relevant_features = result["n_relevant"]
                        cell.max_logit_effect_on_target = result["max_logit_effect"]
                        cell.mean_logit_effect_on_target = result["mean_logit_effect"]
                        cell.run_time = result["run_time"]
                        cell.score = 1 if result["n_relevant"] > 0 else 0

                    # ---- Crosscoder ----
                    elif tool_id == "crosscoder":
                        result = run_crosscoder_tool(
                            model, tokenizer, inputs, targets,
                            "262k", l0, cache_dir, device)
                        cell.fvu = result["fvu"]
                        cell.l0 = result["l0"]
                        cell.n_relevant_features = result["n_relevant"]
                        cell.max_logit_effect_on_target = result["max_logit_effect"]
                        cell.mean_logit_effect_on_target = result["mean_logit_effect"]
                        cell.run_time = result["run_time"]
                        cell.score = 1 if result["n_relevant"] > 0 else 0

                    # ---- CLT variants ----
                    elif tool_id in ["clt_noskip", "clt_affine"]:
                        clt = clt_affine if tool_id == "clt_affine" else clt_noskip
                        result = run_clt_tool(
                            model, tokenizer, inputs, targets, clt, device)
                        cell.fvu = result["fvu"]
                        cell.l0 = result["l0"]
                        cell.n_relevant_features = result["n_relevant"]
                        cell.max_logit_effect_on_target = result["max_logit_effect"]
                        cell.mean_logit_effect_on_target = result["mean_logit_effect"]
                        cell.has_circuit = result["has_circuit"]
                        cell.circuit_path_length = result["path_length"]
                        cell.circuit_n_nodes = result["n_nodes"]
                        cell.circuit_ff_edges = result["ff_edges"]
                        cell.run_time = result["run_time"]

                        if result["has_circuit"]:
                            cell.score = 2
                        elif result["n_relevant"] > 0:
                            cell.score = 1
                        else:
                            cell.score = 0

                    print(f"  rel={cell.n_relevant_features}, "
                          f"eff={cell.max_logit_effect_on_target:+.3f}, "
                          f"{cell.run_time:.1f}s")

                except Exception as e:
                    print(f"  FAILED: {e}")
                    cell.score = -1

                grid.cells.append(cell)

                if completed % 25 == 0:
                    save_grid(grid, os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "grid_checkpoint.json"))

    return grid


# ============================================================
# Visualization
# ============================================================

def plot_grid_heatmap(grid, tasks_module, save_path, metric="relevance"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    task_ids = tasks_module.get_all_task_ids()
    task_labels = [tasks_module.get_task_info(t)["name"] for t in task_ids]

    active_tools = []
    for t in TOOL_IDS:
        if any(c.tool_id == t for c in grid.cells):
            active_tools.append(t)
    tool_labels = [TOOL_LABELS[t] for t in active_tools]

    n_tasks = len(task_ids)
    n_tools = len(active_tools)
    matrix = np.zeros((n_tasks, n_tools))

    for i, task_id in enumerate(task_ids):
        for j, tool_id in enumerate(active_tools):
            if metric == "relevance":
                matrix[i, j] = grid.get_mean_relevance(tool_id, task_id)
            elif metric == "score":
                matrix[i, j] = grid.get_score(tool_id, task_id)
            elif metric == "fvu":
                matrix[i, j] = grid.get_mean_fvu(tool_id, task_id)
            elif metric == "n_relevant":
                matrix[i, j] = grid.get_mean_n_relevant(tool_id, task_id)
            elif metric == "probe_acc":
                matrix[i, j] = grid.get_mean_probe_acc(tool_id, task_id)

    if metric == "relevance":
        cmap = "RdBu_r"
        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
        label = "Max Logit Effect on Target"
        title = "AI Anatomy: Tool Relevance to Cognitive Tasks"
    elif metric == "score":
        cmap = mcolors.ListedColormap(["#f0f0f0", "#a8d5e2", "#1a5276"])
        label = "Score"
        title = "AI Anatomy: Tool x Task Coverage Grid"
        vmax = None
    elif metric == "fvu":
        cmap = "YlOrRd"
        label = "FVU (lower = better)"
        title = "AI Anatomy: Reconstruction Quality"
        vmax = None
    elif metric == "n_relevant":
        cmap = "YlOrRd"
        label = "Number of Relevant Features"
        title = "AI Anatomy: Feature Relevance Count"
        vmax = None
    else:
        cmap = "YlOrRd"
        label = metric
        title = f"AI Anatomy: {metric}"
        vmax = None

    fig, ax = plt.subplots(figsize=(max(14, n_tools * 1.4), max(8, n_tasks * 0.8)))

    if metric == "score":
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    elif metric == "relevance":
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n_tools))
    ax.set_xticklabels(tool_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_labels, fontsize=9)

    for i in range(n_tasks):
        for j in range(n_tools):
            val = matrix[i, j]
            if metric == "score":
                text = str(int(val))
                color = "white" if val == 2 else "black"
            elif metric == "fvu":
                text = f"{val:.3f}" if val > 0 else "-"
                color = "white" if val > (matrix[matrix > 0].max() * 0.6 if (matrix > 0).any() else 1) else "black"
            elif metric == "relevance":
                text = f"{val:+.2f}" if val != 0 else "0"
                color = "white" if abs(val) > vmax * 0.5 else "black"
            else:
                text = f"{val:.1f}" if val != 0 else "0"
                color = "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)

    ax.set_title(title, fontsize=11, pad=15)
    plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()

    plt.savefig(save_path, dpi=800, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def save_grid(grid, path):
    data = {"metadata": grid.metadata, "cells": []}
    for c in grid.cells:
        data["cells"].append({
            "tool_id": c.tool_id, "task_id": c.task_id,
            "variant_idx": c.variant_idx, "prompt_text": c.prompt_text,
            "target_tokens": c.target_tokens,
            "fvu": c.fvu, "l0": c.l0,
            "n_relevant_features": c.n_relevant_features,
            "max_logit_effect_on_target": c.max_logit_effect_on_target,
            "mean_logit_effect_on_target": c.mean_logit_effect_on_target,
            "target_token_rank": c.target_token_rank,
            "probe_accuracy": c.probe_accuracy,
            "has_circuit": c.has_circuit,
            "circuit_path_length": c.circuit_path_length,
            "circuit_n_nodes": c.circuit_n_nodes,
            "circuit_ff_edges": c.circuit_ff_edges,
            "score": c.score, "run_time": c.run_time,
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved grid: {path}")

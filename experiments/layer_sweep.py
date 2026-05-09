"""
Per-Layer Sweep — Run residual SAE across all 26 layers for each task.

Produces a (tasks × layers) relevance matrix showing WHERE each
cognitive capacity is computed in the network.

This is AI Anatomy Experiment 2.
"""

import sys
sys.path.insert(0, "/workspace/Gemma-Scope-2-Study")

import torch
import numpy as np
import json
import time
import os

from src.loader import load_sae, GEMMA3_1B_NUM_LAYERS
from src.hooks import gather_residual_activations
from src.metrics import compute_fvu, compute_l0


def resolve_target_token_ids(tokenizer, target_tokens):
    target_ids = set()
    for t in target_tokens:
        target_ids.update(tokenizer.encode(t, add_special_tokens=False))
        target_ids.update(tokenizer.encode(" " + t, add_special_tokens=False))
    return target_ids


def get_effective_unembed(model):
    w_u = model.lm_head.weight
    ln_w = model.model.norm.weight
    return (w_u * ln_w).float()


def sweep_single_task(
    model, tokenizer, prompt_text, target_tokens,
    width="16k", l0="big",
    cache_dir=None, device="cuda",
):
    """
    Run residual SAE at every layer for a single prompt.

    Uses 16k/small by default because these are available at ALL layers
    (65k/262k only available at layers 7,13,17,22).

    Returns:
        dict with per-layer results: relevance, fvu, l0, n_relevant
    """
    inputs = tokenizer.encode(prompt_text, return_tensors="pt",
                              add_special_tokens=True).to(device)
    target_ids = resolve_target_token_ids(tokenizer, target_tokens)
    w_eff = get_effective_unembed(model)

    results = {}

    for layer in range(GEMMA3_1B_NUM_LAYERS):
        t0 = time.time()

        sae = load_sae(layer, "resid_post_all", width, l0, device=device, cache_dir=cache_dir)
        acts = gather_residual_activations(model, layer, inputs)
        features = sae.encode(acts.float())
        recon = sae.decode(features)

        fvu = compute_fvu(recon, acts).item()
        l0_val = compute_l0(features).item()

        # Compute relevance: max logit effect on target at last position
        last_features = features[-1]
        n_active = (last_features > 0).sum().item()
        max_effect = 0.0
        n_relevant = 0

        if n_active > 0:
            top_vals, top_idxs = last_features.topk(min(20, n_active))
            for val, idx in zip(top_vals, top_idxs):
                if val.item() <= 0:
                    continue
                dec_vec = sae.w_dec[idx.item()].float()
                logit_effects = w_eff @ dec_vec

                best_for_feature = 0.0
                for tid in target_ids:
                    effect = logit_effects[tid].item()
                    if abs(effect) > abs(best_for_feature):
                        best_for_feature = effect

                if abs(best_for_feature) > abs(max_effect):
                    max_effect = best_for_feature
                if abs(best_for_feature) > 0.5:
                    n_relevant += 1

        run_time = time.time() - t0

        results[layer] = {
            "fvu": fvu,
            "l0": l0_val,
            "max_relevance": max_effect,
            "n_relevant": n_relevant,
            "run_time": run_time,
        }

        del sae, features, recon, acts
        torch.cuda.empty_cache()

    return results


def sweep_all_tasks(
    model, tokenizer, tasks_module,
    width="16k", l0="big",
    max_variants=1,
    cache_dir=None, device="cuda",
):
    """
    Run per-layer sweep for all tasks.

    Returns:
        dict: task_id -> {layer -> metrics}
        Also returns numpy matrices for easy plotting.
    """
    task_ids = tasks_module.get_all_task_ids()
    n_tasks = len(task_ids)
    n_layers = GEMMA3_1B_NUM_LAYERS

    all_results = {}
    # Matrices for plotting
    relevance_matrix = np.zeros((n_tasks, n_layers))
    fvu_matrix = np.zeros((n_tasks, n_layers))

    for ti, task_id in enumerate(task_ids):
        task_info = tasks_module.get_task_info(task_id)
        prompts = tasks_module.get_task_prompts(task_id)[:max_variants]

        print(f"\n  [{ti+1}/{n_tasks}] {task_info['name']}")

        task_relevance = np.zeros(n_layers)
        task_fvu = np.zeros(n_layers)

        for vi, prompt in enumerate(prompts):
            print(f"    Variant {vi+1}: '{prompt['text'][:50]}...'")
            result = sweep_single_task(
                model, tokenizer, prompt["text"], prompt["target"],
                width, l0, cache_dir, device,
            )

            for layer, metrics in result.items():
                task_relevance[layer] += abs(metrics["max_relevance"])
                task_fvu[layer] += metrics["fvu"]

        # Average across variants
        task_relevance /= max_variants
        task_fvu /= max_variants

        relevance_matrix[ti] = task_relevance
        fvu_matrix[ti] = task_fvu
        all_results[task_id] = {
            "relevance_by_layer": task_relevance.tolist(),
            "fvu_by_layer": task_fvu.tolist(),
        }

        # Print peak layers
        top_layers = np.argsort(task_relevance)[-3:][::-1]
        print(f"    Peak layers: {', '.join(f'L{l}({task_relevance[l]:.2f})' for l in top_layers)}")

    return all_results, relevance_matrix, fvu_matrix


def plot_layer_landscape(relevance_matrix, tasks_module, save_path):
    """
    Plot the tasks × layers heatmap — AI Anatomy Figure 2.

    This shows WHERE in the network each cognitive capacity is computed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_ids = tasks_module.get_all_task_ids()
    task_labels = [tasks_module.get_task_info(t)["name"] for t in task_ids]
    n_tasks, n_layers = relevance_matrix.shape

    fig, ax = plt.subplots(figsize=(16, 7))

    im = ax.imshow(relevance_matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="gaussian")

    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xticklabels([str(i) for i in range(0, n_layers, 2)], fontsize=8)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_labels, fontsize=9)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Cognitive Task", fontsize=11)
    ax.set_title("Where Does Each Cognitive Capacity Live?\n"
                 "Residual SAE Feature Relevance Across Layers",
                 fontsize=12, pad=15)

    # Add value annotations for peak cells
    for i in range(n_tasks):
        peak_layer = np.argmax(relevance_matrix[i])
        peak_val = relevance_matrix[i, peak_layer]
        if peak_val > 0:
            ax.text(peak_layer, i, f"{peak_val:.1f}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    plt.colorbar(im, ax=ax, label="Feature Relevance (|max logit effect on target|)",
                 shrink=0.8)
    plt.tight_layout()

    plt.savefig(save_path, dpi=600, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def save_sweep_results(all_results, relevance_matrix, fvu_matrix, path):
    data = {
        "task_results": all_results,
        "relevance_matrix": relevance_matrix.tolist(),
        "fvu_matrix": fvu_matrix.tolist(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")

#!/usr/bin/env python3
"""
AI ANATOMY — COMPLETE EXPERIMENT SUITE
======================================
"Cognitive Mapping of Model Biology Across the MI Spectrum"

Runs all 4 experiments, saves all raw data as structured JSON.
Figures are generated separately from the saved data.

Experiments:
  1. 10×10 Grid: 10 tools × 10 tasks × N variants
  2. Per-Layer Landscape: residual SAE across all 26 layers
  3. FVU vs Delta Loss: reconstruction quality vs functional impact
  4. Skip Fraction: what % of MLP computation is linear?

Usage:
  # Quick verification (~30 min)
  python run_suite.py --quick

  # Full run, 1 variant (~2 hr)
  python run_suite.py --variants 1

  # Full run, all 5 variants (~8 hr)
  python run_suite.py

  # Run specific experiment only
  python run_suite.py --only grid
  python run_suite.py --only sweep
  python run_suite.py --only fvu_dl
  python run_suite.py --only skip

Requires: Gemma 3 1B PT, all Gemma Scope 2 tools
GPU Memory: ~33 GB peak (model + 2 CLTs)
"""

import sys
import os
import json
import time
import argparse
import traceback
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict

# ============================================================
# Path setup
# ============================================================
WORKSPACE = "/workspace"
INFRA = os.path.join(WORKSPACE, "Gemma-Scope-2-Study")
PROJECT = os.path.join(WORKSPACE, "AI-Anatomy")

sys.path.insert(0, INFRA)
sys.path.insert(0, PROJECT)

import torch
import numpy as np
torch.set_grad_enabled(False)

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

from prompts.prompts_ai_anatomy import TASKS, get_all_task_ids, get_task_info, get_task_prompts

# ============================================================
# Setup
# ============================================================

CACHE = os.path.join(INFRA, "cache")
OUT = os.path.join(PROJECT, "outputs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUT, "experiment.log"), mode="w"),
    ],
)
log = logging.getLogger("ai_anatomy")


def save_results(data, filename):
    """Save results with metadata."""
    path = os.path.join(OUT, filename)
    data["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_memory_gb": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Saved: {path} ({os.path.getsize(path)/1024:.0f} KB)")


# ============================================================
# Helpers
# ============================================================

def resolve_targets(tokenizer, target_tokens):
    ids = set()
    for t in target_tokens:
        ids.update(tokenizer.encode(t, add_special_tokens=False))
        ids.update(tokenizer.encode(" " + t, add_special_tokens=False))
    return ids


def get_effective_unembed(model):
    return (model.lm_head.weight * model.model.norm.weight).float()


def get_w_o(model, layer):
    return model.model.layers[layer].self_attn.o_proj.weight.float()


def compute_relevance(decoder_weights, features_last_pos, w_eff, target_ids,
                      top_k=20, threshold=1.0):
    n_active = (features_last_pos > 0).sum().item()
    if n_active == 0:
        return {"n_relevant": 0, "max_effect": 0.0, "mean_effect": 0.0, "feature_effects": {}}

    top_vals, top_idxs = features_last_pos.topk(min(top_k, n_active))
    relevant = {}

    for val, idx in zip(top_vals, top_idxs):
        if val.item() <= 0:
            continue
        dec = decoder_weights[idx.item()].float()
        effects = w_eff @ dec
        best = 0.0
        for tid in target_ids:
            e = effects[tid].item()
            if abs(e) > abs(best):
                best = e
        if abs(best) > threshold:
            relevant[idx.item()] = {"effect": best, "activation": val.item()}

    if not relevant:
        return {"n_relevant": 0, "max_effect": 0.0, "mean_effect": 0.0, "feature_effects": {}}

    effects = [v["effect"] for v in relevant.values()]
    return {
        "n_relevant": len(relevant),
        "max_effect": max(effects, key=abs),
        "mean_effect": float(np.mean([abs(e) for e in effects])),
        "feature_effects": {str(k): v for k, v in relevant.items()},
    }


def model_prediction_rank(model, tokenizer, inputs, target_tokens):
    with torch.no_grad():
        logits = model(inputs).logits[0, -1]
    target_ids = resolve_targets(tokenizer, target_tokens)
    sorted_ids = logits.argsort(descending=True)
    for rank, tid in enumerate(sorted_ids[:200]):
        if tid.item() in target_ids:
            return rank + 1, tokenizer.decode([tid.item()]).strip()
    return -1, ""


def compute_delta_loss_resid(model, sae, layer, inputs):
    """Delta loss for residual SAE."""
    try:
        model_out = model.forward(inputs, output_hidden_states=True)
        logits_clean = model_out.logits[0]
        acts = model_out.hidden_states[layer + 1][0]
        recon = sae.forward(acts[1:].float())

        def _hook(mod, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            if tensor.dim() == 3:
                tensor[0, 1:] = recon.to(tensor.dtype)
            else:
                tensor[1:] = recon.to(tensor.dtype)
            return out

        h = model.model.layers[layer].register_forward_hook(_hook)
        try:
            logits_patched = model.forward(inputs).logits[0]
        finally:
            h.remove()

        tokens = inputs[0]
        lc = cross_entropy_loss(logits_clean, tokens).mean().item()
        lp = cross_entropy_loss(logits_patched, tokens).mean().item()
        return lp - lc
    except Exception as e:
        log.warning(f"Delta loss failed: {e}")
        return 0.0


def compute_delta_loss_transcoder(model, tc, layer, inputs):
    """Delta loss for transcoder."""
    try:
        cached = [None]

        def _cache(mod, inp, out):
            cached[0] = (out[0] if isinstance(out, tuple) else out).clone()

        h1 = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(_cache)
        try:
            logits_clean = model.forward(inputs).logits[0]
        finally:
            h1.remove()

        recon = tc.forward(cached[0].squeeze(0)[1:].float())

        def _inject(mod, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            if tensor.dim() == 3:
                tensor[0, 1:] = recon.to(tensor.dtype)
            else:
                tensor[1:] = recon.to(tensor.dtype)
            return out

        h2 = model.model.layers[layer].post_feedforward_layernorm.register_forward_hook(_inject)
        try:
            logits_patched = model.forward(inputs).logits[0]
        finally:
            h2.remove()

        tokens = inputs[0]
        lc = cross_entropy_loss(logits_clean, tokens).mean().item()
        lp = cross_entropy_loss(logits_patched, tokens).mean().item()
        return lp - lc
    except Exception as e:
        log.warning(f"Transcoder delta loss failed: {e}")
        return 0.0


# ============================================================
# EXPERIMENT 1: 10×10 Grid
# ============================================================

def run_exp1_grid(model, tokenizer, clt_affine, clt_noskip,
                  target_layer=17, width="65k", l0="medium",
                  max_variants=5):
    """Run all 10 tools on all 10 tasks."""
    log.info("=" * 70)
    log.info("EXPERIMENT 1: 10×10 GRID")
    log.info("=" * 70)

    w_eff = get_effective_unembed(model)
    task_ids = get_all_task_ids()
    results = {"config": {"target_layer": target_layer, "width": width, "l0": l0,
                          "max_variants": max_variants}, "cells": []}

    # Probe negatives (for linear probe)
    probe_negatives = [
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

    # Pre-cache negative activations for probes
    log.info("Caching probe negatives...")
    neg_acts = []
    for text in probe_negatives:
        inp = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True,
                               truncation=True, max_length=128).to("cuda")
        act = gather_residual_activations(model, target_layer, inp)
        neg_acts.append(act[-1].float().cpu().numpy())
    neg_matrix = np.stack(neg_acts)

    total_cells = len(task_ids) * 10 * max_variants
    cell_num = 0

    for task_id in task_ids:
        task_info = get_task_info(task_id)
        prompts = get_task_prompts(task_id)[:max_variants]

        log.info(f"\n{'='*50}")
        log.info(f"TASK: {task_info['name']}")
        log.info(f"{'='*50}")

        for vi, prompt in enumerate(prompts):
            text = prompt["text"]
            targets = prompt["target"]
            inputs = tokenizer.encode(text, return_tensors="pt",
                                      add_special_tokens=True).to("cuda")
            target_ids = resolve_targets(tokenizer, targets)

            # Model prediction check
            pred_rank, pred_token = model_prediction_rank(model, tokenizer, inputs, targets)

            # ---- TOOL 1: Behavioral ----
            cell_num += 1
            log.info(f"  [{cell_num}/{total_cells}] Behavioral v{vi+1}")
            cell = {
                "tool": "behavioral", "task": task_id, "variant": vi,
                "prompt": text[:100], "targets": targets,
                "target_rank": pred_rank, "predicted_token": pred_token,
                "relevance": 1.0 / pred_rank if pred_rank > 0 else 0.0,
                "n_relevant": 1 if 0 < pred_rank <= 10 else 0,
            }
            results["cells"].append(cell)

            # ---- TOOL 2: Linear Probe ----
            cell_num += 1
            log.info(f"  [{cell_num}/{total_cells}] Probe v{vi+1}")
            try:
                from sklearn.linear_model import LogisticRegression
                pos_act = gather_residual_activations(model, target_layer, inputs)[-1].float().cpu().numpy().reshape(1, -1)
                X = np.vstack([pos_act, neg_matrix])
                y = np.array([1] + [0] * len(neg_matrix))
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(X, y)
                acc = clf.score(X, y)
                cell = {
                    "tool": "linear_probe", "task": task_id, "variant": vi,
                    "probe_accuracy": acc,
                    "relevance": max(0, (acc - 0.5) * 2),
                    "n_relevant": 1 if acc > 0.6 else 0,
                }
            except Exception as e:
                log.error(f"  Probe failed: {e}")
                cell = {"tool": "linear_probe", "task": task_id, "variant": vi, "error": str(e)}
            results["cells"].append(cell)

            # ---- TOOLS 3-7: Single-layer Gemma Scope 2 ----
            single_layer_tools = [
                ("attn_sae", "attn_out", False, False),
                ("mlp_sae", "mlp_out", False, False),
                ("resid_sae", "resid_post", False, False),
                ("transcoder_noskip", "transcoder_noskip", False, False),
                ("transcoder_skip", "transcoder_skip", True, False),
            ]

            for tool_name, site_key, is_affine, _ in single_layer_tools:
                cell_num += 1
                log.info(f"  [{cell_num}/{total_cells}] {tool_name} v{vi+1}")
                try:
                    t0 = time.time()

                    if tool_name == "attn_sae":
                        sae = load_sae(target_layer, "attn_out", width, l0, cache_dir=CACHE)
                        acts = gather_attn_out_activations(model, target_layer, inputs)
                        features = sae.encode(acts.float())
                        recon = sae.decode(features)
                        fvu = compute_fvu(recon, acts).item()
                        projected = sae.w_dec.float() @ get_w_o(model, target_layer).T
                        rel = compute_relevance(projected, features[-1], w_eff, target_ids)
                        delta_loss = 0.0
                        del sae, projected

                    elif tool_name == "mlp_sae":
                        sae = load_sae(target_layer, "mlp_out", width, l0, cache_dir=CACHE)
                        acts = gather_mlp_out_activations(model, target_layer, inputs)
                        features = sae.encode(acts.float())
                        recon = sae.decode(features)
                        fvu = compute_fvu(recon, acts).item()
                        rel = compute_relevance(sae.w_dec, features[-1], w_eff, target_ids)
                        delta_loss = 0.0
                        del sae

                    elif tool_name == "resid_sae":
                        sae = load_sae(target_layer, "resid_post", width, l0, cache_dir=CACHE)
                        acts = gather_residual_activations(model, target_layer, inputs)
                        features = sae.encode(acts.float())
                        recon = sae.decode(features)
                        fvu = compute_fvu(recon, acts).item()
                        rel = compute_relevance(sae.w_dec, features[-1], w_eff, target_ids)
                        delta_loss = compute_delta_loss_resid(model, sae, target_layer, inputs)
                        del sae

                    elif tool_name == "transcoder_noskip":
                        tc = load_transcoder(target_layer, width, l0, affine=False, cache_dir=CACHE)
                        cache = gather_transcoder_activations(model, target_layer, inputs)
                        features = tc.encode(cache["input"].float())
                        recon = tc.forward(cache["input"].float())
                        fvu = compute_fvu(recon, cache["target"]).item()
                        rel = compute_relevance(tc.w_dec, features[-1], w_eff, target_ids)
                        delta_loss = compute_delta_loss_transcoder(model, tc, target_layer, inputs)
                        del tc

                    elif tool_name == "transcoder_skip":
                        tc = load_transcoder(target_layer, width, l0, affine=True, cache_dir=CACHE)
                        cache = gather_transcoder_activations(model, target_layer, inputs)
                        features = tc.encode(cache["input"].float())
                        recon = tc.forward(cache["input"].float())
                        fvu = compute_fvu(recon, cache["target"]).item()
                        rel = compute_relevance(tc.w_dec, features[-1], w_eff, target_ids)
                        delta_loss = compute_delta_loss_transcoder(model, tc, target_layer, inputs)
                        del tc

                    l0_val = compute_l0(features).item()
                    elapsed = time.time() - t0

                    cell = {
                        "tool": tool_name, "task": task_id, "variant": vi,
                        "fvu": fvu, "l0": l0_val, "delta_loss": delta_loss,
                        "n_relevant": rel["n_relevant"],
                        "max_effect": rel["max_effect"],
                        "mean_effect": rel["mean_effect"],
                        "feature_effects": rel["feature_effects"],
                        "time": elapsed,
                    }
                    del features, recon
                    torch.cuda.empty_cache()

                except Exception as e:
                    log.error(f"  {tool_name} FAILED: {e}")
                    log.error(traceback.format_exc())
                    cell = {"tool": tool_name, "task": task_id, "variant": vi, "error": str(e)}

                results["cells"].append(cell)

            # ---- TOOL 8: Crosscoder ----
            cell_num += 1
            log.info(f"  [{cell_num}/{total_cells}] Crosscoder v{vi+1}")
            try:
                t0 = time.time()
                cc = load_crosscoder("262k", l0, cache_dir=CACHE)
                layers = GEMMA3_1B_CROSSCODER_LAYERS
                cc_input = gather_crosscoder_activations(model, layers, inputs).float()
                features = cc.encode(cc_input)
                recon = cc.forward(cc_input)
                fvu = compute_fvu(recon, cc_input).item()
                l0_val = compute_l0(features).item()

                last_idx = len(layers) - 1
                rel = compute_relevance(
                    cc.w_dec.data[last_idx, :, last_idx, :],
                    features[-1, last_idx, :], w_eff, target_ids)

                cell = {
                    "tool": "crosscoder", "task": task_id, "variant": vi,
                    "fvu": fvu, "l0": l0_val,
                    "n_relevant": rel["n_relevant"],
                    "max_effect": rel["max_effect"],
                    "time": time.time() - t0,
                }
                del cc, features, recon
                torch.cuda.empty_cache()
            except Exception as e:
                log.error(f"  Crosscoder FAILED: {e}")
                cell = {"tool": "crosscoder", "task": task_id, "variant": vi, "error": str(e)}
            results["cells"].append(cell)

            # ---- TOOLS 9-10: CLT variants ----
            for clt_name, clt_obj in [("clt_noskip", clt_noskip), ("clt_affine", clt_affine)]:
                if clt_obj is None:
                    cell_num += 1
                    results["cells"].append({"tool": clt_name, "task": task_id, "variant": vi, "skipped": True})
                    continue
                cell_num += 1
                log.info(f"  [{cell_num}/{total_cells}] {clt_name} v{vi+1}")
                try:
                    t0 = time.time()
                    clt_in, clt_tgt = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)
                    if next(clt_obj.parameters()).dtype == torch.float16:
                        clt_in = clt_in.half()
                        clt_tgt = clt_tgt.half()

                    features = clt_obj.encode(clt_in)
                    recon = clt_obj.forward(clt_in)
                    fvu = compute_fvu(recon, clt_tgt).item()
                    l0_val = compute_l0(features).item()

                    # Relevance across late layers
                    clt_relevant = {}
                    for layer in range(18, 26):
                        lf = features[-1, layer, :]
                        n_act = (lf > 0).sum().item()
                        if n_act == 0:
                            continue
                        tv, ti = lf.topk(min(20, n_act))
                        for v, idx in zip(tv, ti):
                            if v.item() <= 0:
                                continue
                            dec = clt_obj.w_dec.data[layer, idx.item(), layer:].sum(dim=0).float()
                            for tid in target_ids:
                                eff = (w_eff[tid] * dec).sum().item()
                                if abs(eff) > 1.0:
                                    key = f"L{layer}_f{idx.item()}"
                                    if key not in clt_relevant or abs(eff) > abs(clt_relevant[key]):
                                        clt_relevant[key] = eff

                    max_eff = max(clt_relevant.values(), key=abs) if clt_relevant else 0.0

                    # Attribution graph
                    circuit = {"has_circuit": False}
                    try:
                        graph = build_attribution_graph(
                            model, clt_obj, tokenizer, text,
                            top_k_output_tokens=5,
                            min_ff_edge_weight=50.0, min_fl_edge_weight=5.0)
                        pruned = prune_graph(graph, top_k_edges_per_node=3,
                                            max_feature_nodes=30, min_edge_weight=10.0)
                        metrics = compute_graph_metrics(pruned)
                        circuit = {
                            "has_circuit": metrics.get("feature_to_feature_edges", 0) > 0 and metrics.get("avg_path_length", 0) > 1.0,
                            "ff_edges": metrics.get("feature_to_feature_edges", 0),
                            "fl_edges": metrics.get("feature_to_logit_edges", 0),
                            "path_length": metrics.get("avg_path_length", 0),
                            "n_nodes": metrics.get("num_feature_nodes", 0),
                            "layer_distribution": metrics.get("layer_distribution", {}),
                        }
                    except Exception as e:
                        log.warning(f"  {clt_name} attribution failed: {e}")

                    cell = {
                        "tool": clt_name, "task": task_id, "variant": vi,
                        "fvu": fvu, "l0": l0_val,
                        "n_relevant": len(clt_relevant),
                        "max_effect": max_eff,
                        "feature_effects": clt_relevant,
                        "circuit": circuit,
                        "time": time.time() - t0,
                    }
                    del features, recon, clt_in, clt_tgt
                    torch.cuda.empty_cache()

                except Exception as e:
                    log.error(f"  {clt_name} FAILED: {e}")
                    log.error(traceback.format_exc())
                    cell = {"tool": clt_name, "task": task_id, "variant": vi, "error": str(e)}

                results["cells"].append(cell)

            # Checkpoint every task
            save_results(results, "exp1_grid_checkpoint.json")

    save_results(results, "exp1_grid_final.json")
    log.info(f"Experiment 1 complete: {len(results['cells'])} cells")
    return results


# ============================================================
# EXPERIMENT 2: Per-Layer Sweep
# ============================================================

def run_exp2_sweep(model, tokenizer, max_variants=1):
    """Run residual SAE at all 26 layers for each task."""
    log.info("=" * 70)
    log.info("EXPERIMENT 2: PER-LAYER SWEEP")
    log.info("=" * 70)

    w_eff = get_effective_unembed(model)
    task_ids = get_all_task_ids()
    results = {"config": {"width": "16k", "l0": "big", "max_variants": max_variants},
               "sweeps": {}}

    for task_id in task_ids:
        task_info = get_task_info(task_id)
        prompts = get_task_prompts(task_id)[:max_variants]

        log.info(f"\n  TASK: {task_info['name']}")
        task_data = {"layers": {}}

        for vi, prompt in enumerate(prompts):
            text = prompt["text"]
            targets = prompt["target"]
            inputs = tokenizer.encode(text, return_tensors="pt",
                                      add_special_tokens=True).to("cuda")
            target_ids = resolve_targets(tokenizer, targets)

            for layer in range(GEMMA3_1B_NUM_LAYERS):
                try:
                    sae = load_sae(layer, "resid_post_all", "16k", "big", cache_dir=CACHE)
                    acts = gather_residual_activations(model, layer, inputs)
                    features = sae.encode(acts.float())
                    recon = sae.decode(features)

                    fvu = compute_fvu(recon, acts).item()
                    l0_val = compute_l0(features).item()
                    rel = compute_relevance(sae.w_dec, features[-1], w_eff, target_ids,
                                           threshold=0.5)

                    key = str(layer)
                    if key not in task_data["layers"]:
                        task_data["layers"][key] = []
                    task_data["layers"][key].append({
                        "variant": vi,
                        "fvu": fvu, "l0": l0_val,
                        "max_relevance": rel["max_effect"],
                        "n_relevant": rel["n_relevant"],
                    })

                    del sae, features, recon, acts
                    torch.cuda.empty_cache()

                except Exception as e:
                    log.error(f"  Layer {layer} failed: {e}")

        # Compute averages
        task_data["avg_relevance_by_layer"] = {}
        task_data["avg_fvu_by_layer"] = {}
        for layer_key, entries in task_data["layers"].items():
            task_data["avg_relevance_by_layer"][layer_key] = float(np.mean(
                [abs(e["max_relevance"]) for e in entries]))
            task_data["avg_fvu_by_layer"][layer_key] = float(np.mean(
                [e["fvu"] for e in entries]))

        peak_layer = max(task_data["avg_relevance_by_layer"],
                        key=lambda k: task_data["avg_relevance_by_layer"][k])
        log.info(f"    Peak: L{peak_layer} (rel={task_data['avg_relevance_by_layer'][peak_layer]:.3f})")

        results["sweeps"][task_id] = task_data

    save_results(results, "exp2_layer_sweep.json")
    log.info("Experiment 2 complete")
    return results


# ============================================================
# EXPERIMENT 3: FVU vs Delta Loss
# ============================================================

def run_exp3_fvu_dl(model, tokenizer, target_layer=17, width="65k", l0="medium",
                    max_variants=1):
    """Compute FVU and delta loss for resid SAE, transcoder, skip-TC across all tasks."""
    log.info("=" * 70)
    log.info("EXPERIMENT 3: FVU vs DELTA LOSS")
    log.info("=" * 70)

    task_ids = get_all_task_ids()
    results = {"config": {"layer": target_layer, "width": width, "l0": l0}, "data": []}

    tools = [
        ("resid_sae", "resid"),
        ("transcoder_noskip", "transcoder"),
        ("transcoder_skip", "transcoder_skip"),
    ]

    for task_id in task_ids:
        prompts = get_task_prompts(task_id)[:max_variants]
        task_info = get_task_info(task_id)
        log.info(f"\n  {task_info['name']}")

        for vi, prompt in enumerate(prompts):
            inputs = tokenizer.encode(prompt["text"], return_tensors="pt",
                                      add_special_tokens=True).to("cuda")

            for tool_name, tool_type in tools:
                try:
                    if tool_type == "resid":
                        sae = load_sae(target_layer, "resid_post", width, l0, cache_dir=CACHE)
                        acts = gather_residual_activations(model, target_layer, inputs)
                        features = sae.encode(acts.float())
                        recon = sae.decode(features)
                        fvu = compute_fvu(recon, acts).item()
                        dl = compute_delta_loss_resid(model, sae, target_layer, inputs)
                        del sae

                    elif tool_type == "transcoder":
                        tc = load_transcoder(target_layer, width, l0, affine=False, cache_dir=CACHE)
                        cache = gather_transcoder_activations(model, target_layer, inputs)
                        features = tc.encode(cache["input"].float())
                        recon = tc.forward(cache["input"].float())
                        fvu = compute_fvu(recon, cache["target"]).item()
                        dl = compute_delta_loss_transcoder(model, tc, target_layer, inputs)
                        del tc

                    elif tool_type == "transcoder_skip":
                        tc = load_transcoder(target_layer, width, l0, affine=True, cache_dir=CACHE)
                        cache = gather_transcoder_activations(model, target_layer, inputs)
                        features = tc.encode(cache["input"].float())
                        recon = tc.forward(cache["input"].float())
                        fvu = compute_fvu(recon, cache["target"]).item()
                        dl = compute_delta_loss_transcoder(model, tc, target_layer, inputs)
                        del tc

                    l0_val = compute_l0(features).item()
                    results["data"].append({
                        "tool": tool_name, "task": task_id, "variant": vi,
                        "fvu": fvu, "delta_loss": dl, "l0": l0_val,
                        "amplification": dl / fvu if fvu > 0 else 0,
                    })
                    log.info(f"    {tool_name}: FVU={fvu:.4f}, dL={dl:.4f}")

                    del features, recon
                    torch.cuda.empty_cache()

                except Exception as e:
                    log.error(f"    {tool_name} failed: {e}")

    save_results(results, "exp3_fvu_delta_loss.json")
    log.info("Experiment 3 complete")
    return results


# ============================================================
# EXPERIMENT 4: Skip Fraction
# ============================================================

def run_exp4_skip(model, tokenizer, target_layer=17, width="65k", l0="medium",
                  max_variants=1):
    """Compare transcoder vs skip-TC to quantify linear MLP fraction."""
    log.info("=" * 70)
    log.info("EXPERIMENT 4: SKIP FRACTION")
    log.info("=" * 70)

    task_ids = get_all_task_ids()
    results = {"config": {"layer": target_layer, "width": width, "l0": l0}, "data": []}

    for task_id in task_ids:
        prompts = get_task_prompts(task_id)[:max_variants]
        task_info = get_task_info(task_id)

        fvu_noskip_list = []
        fvu_skip_list = []

        for vi, prompt in enumerate(prompts):
            inputs = tokenizer.encode(prompt["text"], return_tensors="pt",
                                      add_special_tokens=True).to("cuda")

            try:
                # No skip
                tc = load_transcoder(target_layer, width, l0, affine=False, cache_dir=CACHE)
                cache = gather_transcoder_activations(model, target_layer, inputs)
                recon = tc.forward(cache["input"].float())
                fvu_ns = compute_fvu(recon, cache["target"]).item()
                fvu_noskip_list.append(fvu_ns)
                del tc, recon

                # Skip
                tc = load_transcoder(target_layer, width, l0, affine=True, cache_dir=CACHE)
                recon = tc.forward(cache["input"].float())
                fvu_s = compute_fvu(recon, cache["target"]).item()
                fvu_skip_list.append(fvu_s)
                del tc, recon, cache

                torch.cuda.empty_cache()

            except Exception as e:
                log.error(f"  {task_id} v{vi} skip fraction failed: {e}")

        if fvu_noskip_list and fvu_skip_list:
            avg_ns = float(np.mean(fvu_noskip_list))
            avg_s = float(np.mean(fvu_skip_list))
            skip_frac = 1 - avg_s / avg_ns if avg_ns > 0 else 0

            results["data"].append({
                "task": task_id, "task_name": task_info["name"],
                "fvu_noskip": avg_ns, "fvu_skip": avg_s,
                "skip_fraction": skip_frac,
                "fvu_noskip_all": fvu_noskip_list,
                "fvu_skip_all": fvu_skip_list,
            })
            log.info(f"  {task_info['name']}: skip_frac={skip_frac:.1%} "
                     f"(FVU: {avg_ns:.4f} → {avg_s:.4f})")

    save_results(results, "exp4_skip_fraction.json")
    log.info("Experiment 4 complete")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AI Anatomy — Complete Experiment Suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 variant, skip heavy")
    parser.add_argument("--variants", type=int, default=None, help="Override variants (1-5)")
    parser.add_argument("--layer", type=int, default=17, help="Target layer for single-layer tools")
    parser.add_argument("--only", type=str, default=None,
                        choices=["grid", "sweep", "fvu_dl", "skip"],
                        help="Run only one experiment")
    args = parser.parse_args()

    max_variants = 1 if args.quick else (args.variants or 5)

    log.info("=" * 70)
    log.info("AI ANATOMY — COMPLETE EXPERIMENT SUITE")
    log.info(f"  Variants: {max_variants}")
    log.info(f"  Layer: {args.layer}")
    log.info(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    log.info(f"  Only: {args.only or 'ALL'}")
    log.info(f"  Time: {datetime.now().isoformat()}")
    log.info("=" * 70)

    # Load model
    log.info("\nLoading Gemma 3 1B PT...")
    model, tokenizer = load_gemma3_1b("pt", device="cuda")

    # Load CLTs (only if running grid)
    clt_affine = None
    clt_noskip = None
    if args.only is None or args.only == "grid":
        log.info("Loading CLT+Affine...")
        clt_affine = load_clt(width="262k", l0="big", affine=True,
                              device="cuda", half_precision=True, cache_dir=CACHE)
        if not args.quick:
            log.info("Loading CLT (no affine)...")
            clt_noskip = load_clt(width="262k", l0="big", affine=False,
                                  device="cuda", half_precision=True, cache_dir=CACHE)

    log.info(f"\nGPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    t_start = time.time()
    run_experiments = []

    # Run experiments
    if args.only is None or args.only == "grid":
        log.info("\n\n")
        r = run_exp1_grid(model, tokenizer, clt_affine, clt_noskip,
                          target_layer=args.layer, max_variants=max_variants)
        run_experiments.append(("grid", len(r["cells"])))

        # Free CLTs for other experiments
        del clt_affine, clt_noskip
        clt_affine = clt_noskip = None
        torch.cuda.empty_cache()

    if args.only is None or args.only == "sweep":
        log.info("\n\n")
        r = run_exp2_sweep(model, tokenizer, max_variants=max(1, max_variants // 2))
        run_experiments.append(("sweep", len(r["sweeps"])))

    if args.only is None or args.only == "fvu_dl":
        log.info("\n\n")
        r = run_exp3_fvu_dl(model, tokenizer, target_layer=args.layer,
                            max_variants=max_variants)
        run_experiments.append(("fvu_dl", len(r["data"])))

    if args.only is None or args.only == "skip":
        log.info("\n\n")
        r = run_exp4_skip(model, tokenizer, target_layer=args.layer,
                          max_variants=max_variants)
        run_experiments.append(("skip", len(r["data"])))

    # Final summary
    total_time = time.time() - t_start
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT SUITE COMPLETE")
    log.info("=" * 70)
    log.info(f"  Total time: {total_time/60:.1f} min ({total_time:.0f}s)")
    log.info(f"  GPU peak: {torch.cuda.max_memory_allocated()/(1024**3):.2f} GB")
    for name, count in run_experiments:
        log.info(f"  {name}: {count} data points")
    log.info(f"\n  Results saved to: {OUT}/")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"):
            size = os.path.getsize(os.path.join(OUT, f)) / 1024
            log.info(f"    {f}: {size:.0f} KB")


if __name__ == "__main__":
    main()

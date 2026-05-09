"""
Verify per-layer sweep on 2 tasks before running full suite.

Run: python verify_layer_sweep.py
"""
import sys
import os
import time

sys.path.insert(0, "/workspace/Gemma-Scope-2-Study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b
from prompts import prompts_ai_anatomy as tasks_module

# Import from the new module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from layer_sweep import sweep_single_task, plot_layer_landscape

import numpy as np

CACHE = "/workspace/Gemma-Scope-2-Study/cache"

print("=" * 70)
print("VERIFY: Per-Layer Sweep")
print("=" * 70)

model, tokenizer = load_gemma3_1b("pt", device="cuda")

# Test on 2 tasks: T1 (simple) and T7 (complex)
test_tasks = ["T1_semantic_retrieval", "T7_theory_of_mind"]
task_labels = []
results_list = []

for task_id in test_tasks:
    info = tasks_module.get_task_info(task_id)
    prompt = tasks_module.get_task_prompts(task_id)[0]
    task_labels.append(info["name"])

    print(f"\n{'='*50}")
    print(f"TASK: {info['name']}")
    print(f"  Prompt: '{prompt['text']}'")
    print(f"  Target: {prompt['target']}")
    print(f"{'='*50}")

    t0 = time.time()
    result = sweep_single_task(
        model, tokenizer, prompt["text"], prompt["target"],
        width="16k", l0="big", cache_dir=CACHE,
    )
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/26:.1f}s per layer)")

    # Print results
    print(f"\n  {'Layer':>5s}  {'Relevance':>10s}  {'FVU':>8s}  {'L0':>6s}  {'#Rel':>5s}")
    print(f"  {'-'*40}")
    for layer in range(26):
        r = result[layer]
        marker = " ***" if abs(r["max_relevance"]) > 1.0 else ""
        print(f"  L{layer:>3d}  {r['max_relevance']:>+10.3f}  {r['fvu']:>8.4f}  "
              f"{r['l0']:>6.1f}  {r['n_relevant']:>5d}{marker}")

    # Collect for plotting
    relevance = np.array([result[l]["max_relevance"] for l in range(26)])
    results_list.append(np.abs(relevance))

    peak = np.argmax(np.abs(relevance))
    print(f"\n  Peak layer: L{peak} (relevance={relevance[peak]:+.3f})")

# Quick test plot
matrix = np.stack(results_list)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

# Mini plot to verify rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for i, (label, relev) in enumerate(zip(task_labels, results_list)):
    axes[i].bar(range(26), relev, color="steelblue", edgecolor="black", linewidth=0.3)
    axes[i].set_xlabel("Layer")
    axes[i].set_ylabel("|Relevance|")
    axes[i].set_title(f"{label}\nPeak: L{np.argmax(relev)}")
    axes[i].set_xticks(range(0, 26, 5))

plt.tight_layout()
plt.savefig(f"{OUT}/verify_layer_sweep.png", dpi=150, bbox_inches="tight")
print(f"\nSaved verification plot: {OUT}/verify_layer_sweep.png")
plt.close()

print(f"\nGPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
print(f"\n{'='*70}")
print("VERIFICATION COMPLETE")
print(f"{'='*70}")

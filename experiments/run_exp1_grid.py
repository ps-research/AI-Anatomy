"""
AI Anatomy — Main Experiment Runner (v3 — full 10x10)

10 tools × 10 tasks × N variants

Usage:
  # Quick test (~20 min, 1 variant)
  CUDA_VISIBLE_DEVICES=0 python experiments/run_exp1_grid.py --quick

  # Full grid (5 variants) — ~4-5 hours
  CUDA_VISIBLE_DEVICES=0 python experiments/run_exp1_grid.py

  # 1 variant full — ~50 min
  CUDA_VISIBLE_DEVICES=0 python experiments/run_exp1_grid.py --variants 1
"""

import sys
import os
import argparse
import time

sys.path.insert(0, "/workspace/Gemma-Scope-2-Study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt

from prompts import prompts_ai_anatomy as tasks_module
from experiments.run_grid import (
    run_grid, save_grid, plot_grid_heatmap,
    TOOL_IDS, TOOL_LABELS,
)


def main():
    parser = argparse.ArgumentParser(description="AI Anatomy Experiment 1: 10x10 Grid")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 variant)")
    parser.add_argument("--variants", type=int, default=None, help="Override variants (1-5)")
    parser.add_argument("--layer", type=int, default=17, help="Target layer")
    parser.add_argument("--width", type=str, default="65k", help="SAE width")
    parser.add_argument("--l0", type=str, default="medium", help="L0 target")
    args = parser.parse_args()

    # Paths — relative to AI-Anatomy root
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE = os.path.join(ROOT, "Gemma-Scope-2-Study", "cache")
    # Fallback if submodule cache doesn't exist
    if not os.path.exists(CACHE):
        CACHE = "/workspace/Gemma-Scope-2-Study/cache"
    OUT = os.path.join(ROOT, "outputs")
    FIG = os.path.join(ROOT, "figures")
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    max_variants = 1 if args.quick else (args.variants or 5)

    print("=" * 70)
    print("AI ANATOMY — EXPERIMENT 1: 10×10 GRID (v3)")
    print("=" * 70)
    print(f"  Tools: {len(TOOL_IDS)} ({', '.join(TOOL_LABELS[t] for t in TOOL_IDS)})")
    print(f"  Tasks: {len(tasks_module.get_all_task_ids())}")
    print(f"  Variants per task: {max_variants}")
    print(f"  Layer: {args.layer}, Width: {args.width}, L0: {args.l0}")
    print(f"  Total cells: {len(TOOL_IDS) * len(tasks_module.get_all_task_ids()) * max_variants}")
    print()

    # Validate prompts
    for task_id in tasks_module.get_all_task_ids():
        info = tasks_module.get_task_info(task_id)
        n = len(tasks_module.get_task_prompts(task_id))
        print(f"  {info['name']}: {n} variants")

    # ============================================================
    # Load models
    # ============================================================
    print(f"\nLoading Gemma 3 1B PT...")
    model, tokenizer = load_gemma3_1b("pt", device="cuda")

    print(f"\nLoading CLT+Affine...")
    clt_affine = load_clt(
        width="262k", l0="big", affine=True,
        device="cuda", half_precision=True, cache_dir=CACHE,
    )

    print(f"\nLoading CLT (no affine)...")
    clt_noskip = load_clt(
        width="262k", l0="big", affine=False,
        device="cuda", half_precision=True, cache_dir=CACHE,
    )

    print(f"\nGPU after loading: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    # Also need sklearn for probes
    try:
        from sklearn.linear_model import LogisticRegression
        print("sklearn: available")
    except ImportError:
        print("WARNING: sklearn not installed. Run: pip install scikit-learn")
        print("Linear probe column will fail.")

    # ============================================================
    # Run grid
    # ============================================================
    t0 = time.time()
    grid = run_grid(
        model, tokenizer, tasks_module,
        clt_affine=clt_affine,
        clt_noskip=clt_noskip,
        target_layer=args.layer,
        width=args.width,
        l0=args.l0,
        cache_dir=CACHE,
        device="cuda",
        max_variants=max_variants,
    )
    total_time = time.time() - t0

    # Save
    save_grid(grid, f"{OUT}/grid_10x10.json")

    # ============================================================
    # Generate figures
    # ============================================================
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")

    plot_grid_heatmap(grid, tasks_module,
                      f"{FIG}/fig1_relevance.pdf", metric="relevance")
    plot_grid_heatmap(grid, tasks_module,
                      f"{FIG}/fig1_score.pdf", metric="score")
    plot_grid_heatmap(grid, tasks_module,
                      f"{FIG}/fig1_fvu.pdf", metric="fvu")
    plot_grid_heatmap(grid, tasks_module,
                      f"{FIG}/fig1_feature_count.pdf", metric="n_relevant")

    # ============================================================
    # Print tables
    # ============================================================
    active_tools = [t for t in TOOL_IDS if any(c.tool_id == t for c in grid.cells)]
    tool_labels = [TOOL_LABELS[t] for t in active_tools]
    task_ids = tasks_module.get_all_task_ids()
    task_labels = [tasks_module.get_task_info(t)["name"] for t in task_ids]

    def print_matrix(title, getter):
        print(f"\n{'='*70}")
        print(title)
        print(f"{'='*70}")
        header = f"{'Task':<25s}"
        for tl in tool_labels:
            header += f" {tl[:10]:>10s}"
        print(header)
        print("-" * len(header))
        for i, task_id in enumerate(task_ids):
            row = f"{task_labels[i]:<25s}"
            for tool_id in active_tools:
                val = getter(tool_id, task_id)
                row += f" {val:>10.3f}"
            print(row)

    print_matrix("RELEVANCE MATRIX", grid.get_mean_relevance)
    print_matrix("FEATURE COUNT", grid.get_mean_n_relevant)
    print_matrix("FVU MATRIX", grid.get_mean_fvu)

    # Score matrix (discrete)
    print(f"\n{'='*70}")
    print("SCORE MATRIX")
    print(f"{'='*70}")
    header = f"{'Task':<25s}"
    for tl in tool_labels:
        header += f" {tl[:10]:>10s}"
    print(header)
    print("-" * len(header))
    for i, task_id in enumerate(task_ids):
        row = f"{task_labels[i]:<25s}"
        for tool_id in active_tools:
            score = grid.get_score(tool_id, task_id)
            row += f" {score:>10d}"
        print(row)

    # Model prediction check
    print(f"\n{'='*70}")
    print("MODEL PREDICTION CHECK")
    print(f"{'='*70}")
    for task_id in task_ids:
        info = tasks_module.get_task_info(task_id)
        cells = grid.get_cell("behavioral", task_id)
        if cells:
            ranks = [c.target_token_rank for c in cells if c.target_token_rank > 0]
            avg_rank = sum(ranks) / max(len(ranks), 1)
            known = sum(1 for r in ranks if r <= 10)
            print(f"  {info['name']:<25s}  avg_rank={avg_rank:>5.1f}  in_top10={known}/{len(cells)}")

    # CLT circuit results
    for clt_tool in ["clt_noskip", "clt_affine"]:
        clt_cells = [c for c in grid.cells if c.tool_id == clt_tool]
        if clt_cells:
            label = TOOL_LABELS[clt_tool]
            print(f"\n{'='*70}")
            print(f"{label} CIRCUIT DETECTION")
            print(f"{'='*70}")
            for c in clt_cells:
                info = tasks_module.get_task_info(c.task_id)
                status = "CIRCUIT" if c.has_circuit else "no circuit"
                print(f"  {info['name']:<25s}  {status}  ff={c.circuit_ff_edges}  "
                      f"path={c.circuit_path_length:.1f}  nodes={c.circuit_n_nodes}  "
                      f"rel={c.n_relevant_features}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Grid: {len(task_ids)} tasks × {len(active_tools)} tools × {max_variants} variants")
    print(f"  Total cells: {len(grid.cells)}")
    print(f"  Total time: {total_time/60:.1f} min ({total_time:.0f}s)")
    print(f"  Avg per cell: {total_time/max(len(grid.cells),1):.1f}s")
    print(f"  GPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    scores = [c.score for c in grid.cells if c.score >= 0]
    print(f"  Scores: 0={scores.count(0)}, 1={scores.count(1)}, 2={scores.count(2)}")

    print(f"\n  Figures:")
    for f in sorted(os.listdir(FIG)):
        if f.endswith(('.pdf', '.png')):
            size = os.path.getsize(os.path.join(FIG, f)) / 1024
            print(f"    {f}: {size:.0f} KB")

    print(f"\n{'='*70}")
    print("EXPERIMENT 1 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

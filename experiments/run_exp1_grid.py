"""
AI Anatomy — Main Experiment Runner

Usage:
  # Full grid (all tools, all tasks, all variants) — ~2-3 hours
  CUDA_VISIBLE_DEVICES=1 python experiments/run_exp1_grid.py

  # Quick test (1 variant per task, skip heavy tools) — ~10 min
  CUDA_VISIBLE_DEVICES=1 python experiments/run_exp1_grid.py --quick

  # Resume from checkpoint
  CUDA_VISIBLE_DEVICES=1 python experiments/run_exp1_grid.py --resume
"""

import sys
import os
import argparse
import time

sys.path.insert(0, "/mnt/storage/sandeep/priyansh/Gemma-Scope-2-Study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt

# Import AI Anatomy modules
from prompts import prompts_ai_anatomy as tasks_module
from experiments.run_grid import (
    run_grid, save_grid, plot_grid_heatmap,
    get_score_matrix, get_fvu_matrix, TOOL_IDS, TOOL_LABELS,
)


def main():
    parser = argparse.ArgumentParser(description="AI Anatomy Experiment 1: Grid")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 variant, skip heavy)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--layer", type=int, default=17, help="Target layer for single-layer tools")
    parser.add_argument("--width", type=str, default="65k", help="SAE width")
    parser.add_argument("--l0", type=str, default="medium", help="L0 target")
    args = parser.parse_args()

    CACHE = "/mnt/storage/sandeep/priyansh/Gemma-Scope-2-Study/cache"
    OUT = "/mnt/storage/sandeep/priyansh/AI-Anatomy/outputs"
    FIG = "/mnt/storage/sandeep/priyansh/AI-Anatomy/figures"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    print("=" * 70)
    print("AI ANATOMY — EXPERIMENT 1: THE GRID")
    print("=" * 70)
    print(f"  Layer: {args.layer}, Width: {args.width}, L0: {args.l0}")
    print(f"  Quick mode: {args.quick}")
    print()

    # Validate prompts
    all_prompts = tasks_module.get_all_prompts()
    print(f"Total prompts defined: {len(all_prompts)}")
    for task_id in tasks_module.get_all_task_ids():
        info = tasks_module.get_task_info(task_id)
        n = len(tasks_module.get_task_prompts(task_id))
        print(f"  {info['name']}: {n} variants")

    # Load model
    print(f"\nLoading model...")
    model, tokenizer = load_gemma3_1b("pt", device="cuda")

    # Load CLT (pre-load to avoid reloading per prompt)
    print(f"Loading CLT...")
    clt = load_clt(
        width="262k", l0="big", affine=True,
        device="cuda", half_precision=True,
        cache_dir=CACHE,
    )

    # Configure run
    max_variants = 1 if args.quick else 5
    skip_tools = ["crosscoder", "clt"] if args.quick else []

    # Run the grid
    print(f"\n{'='*70}")
    print(f"RUNNING GRID: {len(tasks_module.get_all_task_ids())} tasks × "
          f"{len(TOOL_IDS) - len(skip_tools)} tools × {max_variants} variants")
    print(f"{'='*70}")

    t0 = time.time()
    grid = run_grid(
        model, tokenizer, tasks_module, clt,
        target_layer=args.layer,
        width=args.width,
        l0=args.l0,
        cache_dir=CACHE,
        device="cuda",
        max_variants=max_variants,
        skip_tools=skip_tools,
    )
    total_time = time.time() - t0

    # Save results
    save_grid(grid, f"{OUT}/grid_full.json")

    # Generate Figure 1: The Grid Heatmap
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")

    plot_grid_heatmap(grid, tasks_module, f"{FIG}/figure1_grid_heatmap.pdf")

    # Print summary table
    print(f"\n{'='*70}")
    print("SCORE MATRIX")
    print(f"{'='*70}")

    task_ids = tasks_module.get_all_task_ids()
    task_labels = [tasks_module.get_task_info(t)["name"] for t in task_ids]
    tool_labels = [TOOL_LABELS[t] for t in TOOL_IDS if t not in skip_tools]
    active_tools = [t for t in TOOL_IDS if t not in skip_tools]

    # Header
    header = f"{'Task':<25s}"
    for tl in tool_labels:
        header += f" {tl[:12]:>12s}"
    print(header)
    print("-" * len(header))

    # Rows
    for i, task_id in enumerate(task_ids):
        row = f"{task_labels[i]:<25s}"
        for tool_id in active_tools:
            score = grid.get_score(tool_id, task_id)
            fvu = grid.get_mean_fvu(tool_id, task_id)
            row += f" {score:>5d}({fvu:.3f})"
        print(row)

    # Model prediction check
    print(f"\n{'='*70}")
    print("MODEL PREDICTION CHECK (does Gemma 3 1B know the answer?)")
    print(f"{'='*70}")

    for task_id in task_ids:
        info = tasks_module.get_task_info(task_id)
        cells = [c for c in grid.cells if c.task_id == task_id and c.tool_id == active_tools[0]]
        ranks = [c.target_token_rank for c in cells]
        avg_rank = sum(r for r in ranks if r > 0) / max(sum(1 for r in ranks if r > 0), 1)
        known = sum(1 for r in ranks if 0 < r <= 10)
        print(f"  {info['name']:<25s}  avg_rank={avg_rank:>5.1f}  "
              f"in_top10={known}/{len(ranks)}")

    # Stats
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total cells computed: {len(grid.cells)}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Avg time per cell: {total_time/max(len(grid.cells),1):.1f}s")
    print(f"  GPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    # Score distribution
    scores = [c.score for c in grid.cells if c.score >= 0]
    print(f"  Score distribution: 0={scores.count(0)}, 1={scores.count(1)}, 2={scores.count(2)}")

    print(f"\n  Outputs saved to: {OUT}/")
    print(f"  Figures saved to: {FIG}/")
    print(f"\n{'='*70}")
    print("EXPERIMENT 1 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

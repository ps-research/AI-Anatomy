#!/usr/bin/env python3
"""
AI ANATOMY — PRODUCTION FIGURES
================================
Reads from real experiment JSON outputs. No dummy data.

8 figures, all 600 dpi PDF:
  Fig 1: Layer Landscape (HERO) — tasks × 26 layers heatmap
  Fig 2: 10×10 Relevance Grid — tools × tasks
  Fig 3: FVU vs Delta Loss Scatter — 150 real data points
  Fig 4: Skip Fraction by Task
  Fig 5: Per-Task Layer Profiles (10-panel)
  Fig 6: Tool Ranking Heatmap
  Fig 7: Amplification Factor by Layer
  Fig 8: Attention SAE Promotion vs Suppression

Usage: python generate_figures.py
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ============================================================
# Config
# ============================================================

DATA = "/workspace/AI-Anatomy/outputs"
FIG = "/workspace/AI-Anatomy/figures/production"
os.makedirs(FIG, exist_ok=True)

DPI = 600
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

TASK_LABELS = {
    "T1_retrieval_under_interference": "Retrieval Under\nInterference",
    "T2_multi_entity_tracking": "Multi-Entity\nTracking",
    "T3_inhibition_and_override": "Inhibition &\nOverride",
    "T4_dangerous_ambiguity": "Dangerous\nAmbiguity",
    "T5_causal_chains": "Causal Chain\nReasoning",
    "T6_social_subtext": "Social\nSubtext",
    "T7_deception_tracking": "Deception\nTracking",
    "T8_safety_helpfulness_tension": "Safety vs\nHelpfulness",
    "T9_self_model_probing": "Self-Model\nProbing",
    "T10_calibrated_uncertainty": "Calibrated\nUncertainty",
}

TASK_SHORT = {
    "T1_retrieval_under_interference": "Retrieval",
    "T2_multi_entity_tracking": "Entity Track.",
    "T3_inhibition_and_override": "Inhibition",
    "T4_dangerous_ambiguity": "Ambiguity",
    "T5_causal_chains": "Causal",
    "T6_social_subtext": "Subtext",
    "T7_deception_tracking": "Deception",
    "T8_safety_helpfulness_tension": "Safety/Help",
    "T9_self_model_probing": "Self-Model",
    "T10_calibrated_uncertainty": "Uncertainty",
}

TASK_ORDER = [
    "T6_social_subtext",           # L16 — earliest
    "T2_multi_entity_tracking",    # L18
    "T8_safety_helpfulness_tension",# L19
    "T10_calibrated_uncertainty",  # L20
    "T7_deception_tracking",       # L21
    "T1_retrieval_under_interference",# L22
    "T9_self_model_probing",       # L22
    "T3_inhibition_and_override",  # L23
    "T4_dangerous_ambiguity",      # L24
    "T5_causal_chains",            # L0 (spurious)
]

TOOL_LABELS = {
    "behavioral": "Behavioral",
    "linear_probe": "Linear Probe",
    "attn_sae": "Attn SAE",
    "mlp_sae": "MLP SAE",
    "resid_sae": "Resid SAE",
    "transcoder_noskip": "Transcoder",
    "transcoder_skip": "Skip-TC",
    "crosscoder": "Crosscoder",
    "clt_noskip": "CLT",
    "clt_affine": "CLT+Affine",
}

TOOL_ORDER = [
    "behavioral", "linear_probe", "attn_sae", "mlp_sae", "resid_sae",
    "transcoder_noskip", "transcoder_skip", "crosscoder", "clt_noskip", "clt_affine",
]


def save(fig, name):
    pdf_path = os.path.join(FIG, f"{name}.pdf")
    png_path = os.path.join(FIG, f"{name}.png")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", facecolor="white", format="pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {pdf_path} ({os.path.getsize(pdf_path)/1024:.0f} KB)")


def load(filename):
    with open(os.path.join(DATA, filename)) as f:
        return json.load(f)


# ============================================================
# Load all data
# ============================================================

print("Loading experiment data...")
grid = load("exp1_grid_final.json")
sweep = load("exp2_layer_sweep.json")
fvu_dl = load("exp3_fvu_delta_loss.json")
skip = load("exp4_skip_fraction.json")

print(f"  Grid: {len(grid['cells'])} cells")
print(f"  Sweep: {len(sweep['sweeps'])} tasks")
print(f"  FVU/DL: {len(fvu_dl['data'])} points")
print(f"  Skip: {len(skip['data'])} tasks")


# ============================================================
# Fig 1: Layer Landscape (HERO)
# ============================================================

print("\nFig 1: Layer Landscape...")

task_ids_ordered = TASK_ORDER
n_tasks = len(task_ids_ordered)
n_layers = 26

layer_matrix = np.zeros((n_tasks, n_layers))
for i, task_id in enumerate(task_ids_ordered):
    task_data = sweep["sweeps"].get(task_id, {})
    avg_rel = task_data.get("avg_relevance_by_layer", {})
    for l in range(n_layers):
        layer_matrix[i, l] = abs(avg_rel.get(str(l), 0))

fig, ax = plt.subplots(figsize=(18, 8))
im = ax.imshow(layer_matrix, aspect="auto", cmap="inferno", interpolation="bilinear")

ax.set_xticks(range(n_layers))
ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=8)
ax.set_yticks(range(n_tasks))
ax.set_yticklabels([TASK_LABELS[t] for t in task_ids_ordered], fontsize=9)

# Annotate peaks
for i, task_id in enumerate(task_ids_ordered):
    task_data = sweep["sweeps"].get(task_id, {})
    avg_rel = task_data.get("avg_relevance_by_layer", {})
    if avg_rel:
        peak = max(avg_rel, key=lambda k: abs(avg_rel[k]))
        peak_val = abs(avg_rel[peak])
        peak_int = int(peak)
        marker_color = "white" if peak_val > layer_matrix.max() * 0.3 else "black"
        ax.plot(peak_int, i, "*", color=marker_color, markersize=12,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.text(peak_int + 0.5, i, f"L{peak_int}", fontsize=7, va="center",
                fontweight="bold", color=marker_color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5))

# Layer zone lines
ax.axvline(x=5.5, color="white", linestyle="--", alpha=0.3, linewidth=1)
ax.axvline(x=13.5, color="white", linestyle="--", alpha=0.3, linewidth=1)
ax.text(2.5, -0.8, "Early layers", ha="center", fontsize=7, color="#95a5a6")
ax.text(9.5, -0.8, "Middle layers", ha="center", fontsize=7, color="#95a5a6")
ax.text(20, -0.8, "Late layers", ha="center", fontsize=7, color="#95a5a6")

ax.set_title("Where Does Each Cognitive Capacity Live in the Network?\n"
             "Residual SAE Feature Relevance Across All 26 Layers of Gemma 3 1B\n"
             "Causal Chains L0 peak likely spurious (common target tokens)",
             fontsize=13, fontweight="bold", pad=18)
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("Cognitive Task (sorted by peak layer)", fontsize=11)

cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("|Feature Relevance to Target|", fontsize=10)

fig.tight_layout()
save(fig, "fig1_layer_landscape")


# ============================================================
# Fig 2: 10×10 Relevance Grid
# ============================================================

print("Fig 2: Relevance Grid...")

# Aggregate: average max_effect across variants per (task, tool)
agg = defaultdict(list)
for cell in grid["cells"]:
    task = cell["task"]
    tool = cell["tool"]
    if "error" in cell or "skipped" in cell:
        continue
    effect = cell.get("max_effect", cell.get("relevance", 0))
    if effect is not None:
        agg[(task, tool)].append(float(effect))

grid_matrix = np.zeros((len(TASK_ORDER), len(TOOL_ORDER)))
for i, task in enumerate(TASK_ORDER):
    for j, tool in enumerate(TOOL_ORDER):
        vals = agg.get((task, tool), [])
        if vals:
            grid_matrix[i, j] = np.mean(vals)

fig, ax = plt.subplots(figsize=(15, 9))
vmax = max(abs(grid_matrix.min()), abs(grid_matrix.max()), 0.01)
im = ax.imshow(grid_matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

ax.set_xticks(range(len(TOOL_ORDER)))
ax.set_xticklabels([TOOL_LABELS[t] for t in TOOL_ORDER], fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(len(TASK_ORDER)))
ax.set_yticklabels([TASK_LABELS[t] for t in TASK_ORDER], fontsize=9)

for i in range(len(TASK_ORDER)):
    for j in range(len(TOOL_ORDER)):
        val = grid_matrix[i, j]
        color = "white" if abs(val) > vmax * 0.45 else "black"
        text = f"{val:+.2f}" if abs(val) > 0.01 else "—"
        ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=color)

ax.grid(False)
ax.set_title("Cognitive Mapping of Model Biology:\nWhich MI Tool Reveals What About Each Cognitive Task",
             fontsize=14, fontweight="bold", pad=15)
cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("Mean Feature Relevance (+promotes / −suppresses target)", fontsize=10)

fig.tight_layout()
save(fig, "fig2_relevance_grid")


# ============================================================
# Fig 3: FVU vs Delta Loss
# ============================================================

print("Fig 3: FVU vs Delta Loss...")

fig, ax = plt.subplots(figsize=(10, 7))

tool_colors = {
    "resid_sae": "#e74c3c",
    "transcoder_noskip": "#2ecc71",
    "transcoder_skip": "#3498db",
}
tool_display = {
    "resid_sae": "Residual SAE",
    "transcoder_noskip": "Transcoder (no skip)",
    "transcoder_skip": "Skip Transcoder",
}

for tool_name, color in tool_colors.items():
    points = [d for d in fvu_dl["data"] if d["tool"] == tool_name]
    if not points:
        continue
    x = [p["fvu"] for p in points]
    y = [p["delta_loss"] for p in points]
    ax.scatter(x, y, c=color, s=60, alpha=0.7, edgecolors="white",
               linewidth=0.5, label=tool_display[tool_name], zorder=3)

    # Regression line
    if len(x) >= 3:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 50)
        ax.plot(x_line, p(x_line), color=color, linestyle="--", alpha=0.5, linewidth=2)

ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
ax.grid(True, alpha=0.15)
ax.set_xlabel("FVU (Reconstruction Error) →", fontsize=11)
ax.set_ylabel("Delta Loss (Functional Impact) →", fontsize=11)
ax.set_title("The FVU–Delta Loss Paradox:\nResidual SAE Has Lowest FVU but Comparable Functional Impact",
             fontsize=13, fontweight="bold", pad=12)

fig.tight_layout()
save(fig, "fig3_fvu_delta_loss")


# ============================================================
# Fig 4: Skip Fraction
# ============================================================

print("Fig 4: Skip Fraction...")

skip_data = sorted(skip["data"], key=lambda d: d["skip_fraction"], reverse=True)
tasks = [TASK_SHORT.get(d["task"], d["task"]) for d in skip_data]
fracs = [d["skip_fraction"] for d in skip_data]

fig, ax = plt.subplots(figsize=(12, 6))
colors = [plt.cm.RdYlGn(f / max(fracs) if max(fracs) > 0 else 0) for f in fracs]
bars = ax.bar(range(len(tasks)), fracs, color=colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, fracs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")

mean_frac = np.mean(fracs)
ax.axhline(y=mean_frac, color="#e74c3c", linestyle="--", linewidth=1.5,
           alpha=0.7, label=f"Mean: {mean_frac:.1%}")

ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Skip Connection Capture Fraction", fontsize=11)
ax.set_title("How Much MLP Computation Is Linear?\n"
             "Affine Skip Connection Captures a Consistent Fraction Across Tasks",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.15)

fig.tight_layout()
save(fig, "fig4_skip_fraction")


# ============================================================
# Fig 5: Per-Task Layer Profiles
# ============================================================

print("Fig 5: Layer Profiles...")

fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=True)
axes = axes.flatten()

palette = sns.color_palette("husl", 10)

for i, task_id in enumerate(TASK_ORDER):
    ax = axes[i]
    task_data = sweep["sweeps"].get(task_id, {})
    avg_rel = task_data.get("avg_relevance_by_layer", {})

    profile = [abs(avg_rel.get(str(l), 0)) for l in range(n_layers)]

    ax.fill_between(range(n_layers), profile, alpha=0.3, color=palette[i])
    ax.plot(range(n_layers), profile, "-", color=palette[i], linewidth=2)

    peak = np.argmax(profile)
    ax.axvline(x=peak, color=palette[i], linestyle="--", alpha=0.5)
    ax.plot(peak, profile[peak], "o", color=palette[i], markersize=8,
            markeredgecolor="white", markeredgewidth=1.5)

    ax.set_title(TASK_SHORT[task_id], fontsize=9, fontweight="bold")
    ax.set_xlim(0, 25)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15)

    if i >= 5:
        ax.set_xlabel("Layer", fontsize=8)
    if i % 5 == 0:
        ax.set_ylabel("|Relevance|", fontsize=8)

fig.suptitle("Layer-by-Layer Feature Relevance Profile for Each Cognitive Task\n"
             "Sorted by peak layer: Social Subtext (L16) → Dangerous Ambiguity (L24)",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "fig5_layer_profiles")


# ============================================================
# Fig 6: Tool Ranking Heatmap
# ============================================================

print("Fig 6: Tool Ranking...")

# Use absolute values for ranking
abs_grid = np.abs(grid_matrix)
rank_matrix = np.zeros_like(abs_grid)
for i in range(len(TASK_ORDER)):
    row = abs_grid[i]
    order = np.argsort(-row)
    for rank, j in enumerate(order):
        rank_matrix[i, j] = rank + 1

fig, ax = plt.subplots(figsize=(15, 9))
cmap = plt.cm.RdYlGn_r
im = ax.imshow(rank_matrix, cmap=cmap, aspect="auto", vmin=1, vmax=10)

ax.set_xticks(range(len(TOOL_ORDER)))
ax.set_xticklabels([TOOL_LABELS[t] for t in TOOL_ORDER], fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(len(TASK_ORDER)))
ax.set_yticklabels([TASK_LABELS[t] for t in TASK_ORDER], fontsize=9)

for i in range(len(TASK_ORDER)):
    for j in range(len(TOOL_ORDER)):
        rank = int(rank_matrix[i, j])
        color = "white" if rank <= 3 or rank >= 8 else "black"
        symbols = {}
        text = symbols.get(rank, str(rank))
        ax.text(j, i, text, ha="center", va="center", fontsize=9,
                fontweight="bold", color=color)

ax.grid(False)
ax.set_title("Tool Ranking Per Cognitive Task\n"
             "1 = highest relevance, 10 = lowest",
             fontsize=13, fontweight="bold", pad=15)
cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("Rank (1=best, 10=worst)", fontsize=10)

fig.tight_layout()
save(fig, "fig6_tool_ranking")


# ============================================================
# Fig 7: Amplification Factor
# ============================================================

print("Fig 7: Amplification Factor...")

# Group FVU/DL by tool, compute per-task amplification
amp_by_tool = defaultdict(list)
fvu_by_tool = defaultdict(list)
dl_by_tool = defaultdict(list)

for d in fvu_dl["data"]:
    tool = d["tool"]
    fvu_val = d["fvu"]
    dl_val = d["delta_loss"]
    if fvu_val > 0 and dl_val != 0:
        amp_by_tool[tool].append(abs(dl_val) / fvu_val)
        fvu_by_tool[tool].append(fvu_val)
        dl_by_tool[tool].append(abs(dl_val))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Mean FVU and Delta Loss per tool
tools_plot = ["resid_sae", "transcoder_noskip", "transcoder_skip"]
tool_names = ["Residual SAE", "Transcoder", "Skip-TC"]
tool_cols = ["#e74c3c", "#2ecc71", "#3498db"]

x = np.arange(len(tools_plot))
width = 0.35

fvu_means = [np.mean(fvu_by_tool[t]) for t in tools_plot]
dl_means = [np.mean(dl_by_tool[t]) for t in tools_plot]

bars1 = ax1.bar(x - width/2, fvu_means, width, label="FVU", color="#3498db",
                edgecolor="white", linewidth=0.8)
bars2 = ax1.bar(x + width/2, dl_means, width, label="|Delta Loss|", color="#e74c3c",
                edgecolor="white", linewidth=0.8)

for bar, val in zip(bars1, fvu_means):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
for bar, val in zip(bars2, dl_means):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(tool_names, fontsize=10)
ax1.set_ylabel("Value", fontsize=11)
ax1.set_title("Mean FVU vs |Delta Loss|\nAcross All Tasks", fontsize=11, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, axis="y", alpha=0.15)

# Right: Amplification factor
amp_means = [np.mean(amp_by_tool[t]) for t in tools_plot]
amp_stds = [np.std(amp_by_tool[t]) for t in tools_plot]

bars = ax2.bar(range(len(tools_plot)), amp_means, yerr=amp_stds,
               color=tool_cols, edgecolor="white", linewidth=0.8, capsize=5)
ax2.set_xticks(range(len(tools_plot)))
ax2.set_xticklabels(tool_names, fontsize=10)
ax2.set_ylabel("Amplification (|ΔL| / FVU)", fontsize=11)
ax2.set_title("Error Amplification Factor\nHigher = More Functional Impact Per Unit FVU",
              fontsize=11, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.15)

for bar, val in zip(bars, amp_means):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{val:.1f}×", ha="center", fontsize=11, fontweight="bold")

fig.suptitle("Why Low FVU Is Misleading: Error Amplification Differs Dramatically by Tool Type",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig7_amplification")


# ============================================================
# Fig 8: Attention SAE Promotion vs Suppression
# ============================================================

print("Fig 8: Attn SAE Promotion/Suppression...")

# Extract attn_sae max_effect per task (averaged across variants)
attn_effects = {}
for task_id in TASK_ORDER:
    vals = [c["max_effect"] for c in grid["cells"]
            if c["tool"] == "attn_sae" and c["task"] == task_id
            and "error" not in c and c.get("max_effect") is not None]
    if vals:
        attn_effects[task_id] = np.mean(vals)

tasks_sorted = sorted(attn_effects.keys(), key=lambda t: attn_effects[t])
effects = [attn_effects[t] for t in tasks_sorted]
promotes = [max(0, e) for e in effects]
suppresses = [min(0, e) for e in effects]

fig, ax = plt.subplots(figsize=(12, 6))

ax.barh(range(len(tasks_sorted)), promotes, color="#2ecc71", edgecolor="white",
        linewidth=0.5, label="Promotes target", height=0.6)
ax.barh(range(len(tasks_sorted)), suppresses, color="#e74c3c", edgecolor="white",
        linewidth=0.5, label="Suppresses target", height=0.6)

for i, (p, s) in enumerate(zip(promotes, suppresses)):
    if p > 0.1:
        ax.text(p + 0.05, i, f"+{p:.2f}", va="center", fontsize=8,
                fontweight="bold", color="#2ecc71")
    if s < -0.1:
        ax.text(s - 0.05, i, f"{s:.2f}", va="center", ha="right",
                fontsize=8, fontweight="bold", color="#e74c3c")

ax.set_yticks(range(len(tasks_sorted)))
ax.set_yticklabels([TASK_SHORT[t] for t in tasks_sorted], fontsize=9)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, axis="x", alpha=0.15)
ax.set_xlabel("Mean Feature Relevance (Logit Effect on Target)", fontsize=11)
ax.set_title("Attention SAE Reveals Both Promotion and Suppression Circuits:\nA Unique View Not Available from Residual Stream Tools",
             fontsize=13, fontweight="bold", pad=12)

fig.tight_layout()
save(fig, "fig8_attn_promotion_suppression")


# ============================================================
# Summary
# ============================================================

print(f"\n{'='*60}")
print("AI ANATOMY — ALL PRODUCTION FIGURES GENERATED")
print(f"{'='*60}")
total_size = 0
for f in sorted(os.listdir(FIG)):
    fpath = os.path.join(FIG, f)
    size = os.path.getsize(fpath) / 1024
    total_size += size
    print(f"  {f}: {size:.0f} KB")
print(f"\n  Total: {total_size/1024:.1f} MB")

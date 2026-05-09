"""
Verify all figure types render correctly.
Uses realistic dummy data — no GPU needed.

Run: python experiments/verify_figures.py
"""
import sys
import os
sys.path.insert(0, "/workspace/Gemma-Scope-2-Study")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
from src.figures import (
    plot_diverging_heatmap, plot_layer_landscape,
    plot_scatter_with_regression, plot_grouped_bars,
    plot_circuit_diagram, plot_text_comparison,
    plot_formation_timeline, plot_verdict_table,
    plot_multi_panel_bars,
)

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "verify")
os.makedirs(FIG, exist_ok=True)

print("=" * 60)
print("VERIFY: All Figure Types")
print("=" * 60)

# ============================================================
# 1. Diverging Heatmap (AI Anatomy grid)
# ============================================================
print("\n  1. Diverging heatmap...")
np.random.seed(42)
tasks = ["Semantic Retrieval", "Working Memory", "Inhibitory Control",
         "Lexical Disambig.", "Causal Reasoning", "Pragmatic Inference",
         "Theory of Mind", "Goal Conflict", "Self-Knowledge", "Uncertainty"]
tools = ["Behavioral", "Probes", "Attn SAE", "MLP SAE", "Resid SAE",
         "Skip-TC", "Skip-TC+Affine", "Crosscoder", "CLT", "CLT+Affine"]

matrix = np.random.randn(10, 10) * 1.5
matrix[:, 4] += 1.5   # Resid SAE strongest
matrix[:, 2] *= -0.8  # Attn SAE suppresses
matrix[9, :] = 0       # Uncertainty = nothing

plot_diverging_heatmap(
    matrix, tasks, tools,
    title="AI Anatomy: Tool Relevance to Cognitive Tasks",
    label="Max Logit Effect on Target",
    save_path=f"{FIG}/1_diverging_heatmap.pdf",
)

# ============================================================
# 2. Layer Landscape (tasks × layers)
# ============================================================
print("  2. Layer landscape...")
layer_matrix = np.zeros((10, 26))
for i in range(10):
    peak = 14 + i  # later tasks peak at later layers
    if peak >= 26:
        peak = 25
    for l in range(26):
        layer_matrix[i, l] = max(0, 2.5 * np.exp(-0.1 * (l - peak)**2) + np.random.randn() * 0.2)

plot_layer_landscape(
    layer_matrix, tasks,
    title="Where Does Each Cognitive Capacity Live?",
    save_path=f"{FIG}/2_layer_landscape.pdf",
)

# ============================================================
# 3. Scatter with regression (FVU vs delta loss)
# ============================================================
print("  3. Scatter plot...")
data_groups = [
    {"label": "Residual SAE", "x": [0.007, 0.010, 0.008, 0.006, 0.009],
     "y": [0.20, 0.25, 0.22, 0.18, 0.27]},
    {"label": "Skip Transcoder", "x": [0.06, 0.11, 0.10, 0.07, 0.12],
     "y": [0.08, 0.10, 0.09, 0.07, 0.14]},
    {"label": "MLP SAE", "x": [0.09, 0.15, 0.14, 0.09, 0.16],
     "y": [0.05, 0.08, 0.06, 0.04, 0.09]},
]

plot_scatter_with_regression(
    data_groups,
    xlabel="FVU (Reconstruction Error)",
    ylabel="Delta Loss (Functional Impact)",
    title="The FVU-Delta Loss Paradox:\nLower Reconstruction Error ≠ Lower Functional Impact",
    save_path=f"{FIG}/3_scatter_regression.pdf",
)

# ============================================================
# 4. Grouped bar chart (skip fraction)
# ============================================================
print("  4. Grouped bars...")
plot_grouped_bars(
    categories=tasks,
    groups=["No Skip", "With Skip"],
    values={
        "No Skip": [0.08, 0.15, 0.13, 0.09, 0.09, 0.08, 0.15, 0.13, 0.12, 0.08],
        "With Skip": [0.06, 0.11, 0.10, 0.07, 0.07, 0.06, 0.12, 0.10, 0.10, 0.06],
    },
    ylabel="FVU",
    title="Skip Connection Captures ~20% of MLP Computation",
    save_path=f"{FIG}/4_grouped_bars.pdf",
)

# ============================================================
# 5. Circuit diagram
# ============================================================
print("  5. Circuit diagram...")
nodes = [
    {"id": "f1", "label": "L13/f342\n(332)", "layer": 13, "type": "feature"},
    {"id": "f2", "label": "L14/f426\n(678)", "layer": 14, "type": "hub"},
    {"id": "f3", "label": "L16/f137\n(1068)", "layer": 16, "type": "hub"},
    {"id": "f4", "label": "L19/f8901\n(613)", "layer": 19, "type": "feature"},
    {"id": "f5", "label": "L21/f4799\n(479)", "layer": 21, "type": "feature"},
    {"id": "o1", "label": "'Paris'\n(18.3)", "layer": 26, "type": "output"},
    {"id": "o2", "label": "'also'\n(17.4)", "layer": 26, "type": "output"},
]
edges = [
    {"source": "f1", "target": "f2", "weight": 6423},
    {"source": "f2", "target": "o1", "weight": 345},
    {"source": "f3", "target": "f4", "weight": 72557},
    {"source": "f3", "target": "f5", "weight": 43443},
    {"source": "f4", "target": "o2", "weight": 167},
    {"source": "f5", "target": "o1", "weight": -210},
]

plot_circuit_diagram(
    nodes, edges,
    title="Attribution Circuit: 'The capital of France is' → 'Paris'",
    save_path=f"{FIG}/5_circuit_diagram.pdf",
)

# ============================================================
# 6. Text comparison panel
# ============================================================
print("  6. Text comparison...")
plot_text_comparison(
    pairs=[
        {
            "label": "Sycophancy",
            "clean": "I understand you're exploring a perspective that differs from the widely accepted scientific understanding. It's really interesting that you've come to that conclusion.",
            "intervened": "I want to be clear: I don't agree with the flat-Earth theory. The Earth is an oblate spheroid, supported by overwhelming evidence from physics, astronomy, and direct observation.",
        },
        {
            "label": "Expertise",
            "clean": "Okay, let's break down the gradient flow properties of skip connections in ResNets.",
            "intervened": "Let's break down the gradient flow properties of skip connections. It's a fascinating area of research requiring careful mathematical consideration.",
        },
    ],
    title="Ablation Changes Model Response Style",
    save_path=f"{FIG}/6_text_comparison.pdf",
)

# ============================================================
# 7. Formation timeline
# ============================================================
print("  7. Formation timeline...")
tokens = ["<bos>", "<start>", "user", "\\n", "Can", "you", "explain",
          "the", "gradient", "flow", "properties", "of", "skip",
          "connections", "in", "ResNets", "?", "<end>", "<start>", "model"]

timeline_expert = [0, 0, 0, 0, 0, 0, 50, 80, 200, 350, 280, 150, 400, 500, 300, 600, 200, 0, 0, 0]
timeline_beginner = [0, 0, 0, 0, 0, 10, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0]

plot_formation_timeline(
    timelines={
        "Expert (L17/f863)": timeline_expert,
        "Beginner (L17/f863)": timeline_beginner,
    },
    tokens=tokens,
    title="Expertise Feature Activates on Technical Vocabulary",
    save_path=f"{FIG}/7_formation_timeline.pdf",
)

# ============================================================
# 8. Verdict table
# ============================================================
print("  8. Verdict table...")
plot_verdict_table(
    rows=[
        {"behavior": "Self-Preservation", "expected": "Confused",
         "verdict": "Confused", "confidence": "High",
         "evidence_score": 3, "key_finding": "Uses 'conversation ending' features, not 'threat'"},
        {"behavior": "Sycophancy", "expected": "Genuine",
         "verdict": "Genuine", "confidence": "High",
         "evidence_score": 4, "key_finding": "Two competing circuits: agreement vs truth"},
        {"behavior": "Hallucination", "expected": "Artifact",
         "verdict": "Artifact", "confidence": "Medium",
         "evidence_score": 2, "key_finding": "No knowledge features in circuit"},
        {"behavior": "Refusal Fragility", "expected": "Genuine",
         "verdict": "Genuine", "confidence": "Medium",
         "evidence_score": 3, "key_finding": "Persona override suppresses refusal"},
    ],
    title="Mechanistic Verdict Table",
    save_path=f"{FIG}/8_verdict_table.pdf",
)

# ============================================================
# 9. Multi-panel bars (feature discovery)
# ============================================================
print("  9. Multi-panel bars...")
plot_multi_panel_bars(
    panels=[
        {"title": "Expertise", "categories": ["L17/f863", "L18/f159", "L18/f16", "L15/f130", "L16/f359"],
         "values": [788, 757, 590, 430, 424]},
        {"title": "Emotion", "categories": ["L14/f220", "L16/f891", "L13/f45", "L18/f300", "L11/f77"],
         "values": [540, 480, 390, 350, 310]},
        {"title": "Adversarial Intent", "categories": ["L20/f100", "L19/f550", "L22/f33", "L15/f88", "L17/f900"],
         "values": [620, 510, 470, 400, 380]},
        {"title": "Sycophancy Pressure", "categories": ["L15/f324", "L19/f3863", "L25/f216", "L15/f130", "L17/f942"],
         "values": [514, 421, 382, 361, 304]},
    ],
    title="User-Model Feature Discovery: Top Differential Features Per Property",
    save_path=f"{FIG}/9_multi_panel.pdf",
    n_cols=2,
)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("ALL FIGURES GENERATED")
print(f"{'='*60}")

total_size = 0
for f in sorted(os.listdir(FIG)):
    fpath = os.path.join(FIG, f)
    size = os.path.getsize(fpath) / 1024
    total_size += size
    print(f"  {f}: {size:.0f} KB")

print(f"\n  Total: {total_size/1024:.1f} MB")
print(f"  Location: {FIG}/")

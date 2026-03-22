# coding: utf-8
"""
PRISM Architecture Diagram
Run: python plot_architecture.py
Output: plots/prism_architecture.pdf / .png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

os.makedirs('plots', exist_ok=True)

fig, ax = plt.subplots(figsize=(20, 9))
ax.set_xlim(0, 20)
ax.set_ylim(0, 9)
ax.axis('off')

# ── Color palette ──────────────────────────────────────────
C_BEH    = '#AED6F1'   # behavior view (blue)
C_MULTI  = '#A9DFBF'   # multimodal view (green)
C_PRISM  = '#F1948A'   # PRISM innovations (red/orange)
C_LOSS   = '#D7BDE2'   # loss functions (purple)
C_GRAPH  = '#FAD7A0'   # graphs (yellow)
C_GCN    = '#AEB6BF'   # GCN (gray)
C_OUT    = '#F9E79F'   # output embeddings (light yellow)
C_MERGE  = '#D5DBDB'   # merge operations

BOXR   = 0.3           # box corner radius
FS     = 7.5           # font size inside boxes
FS_SM  = 6.5           # small font size

# ── Helper functions ────────────────────────────────────────

def box(ax, x, y, w, h, label, color, sublabel=None, fontsize=FS, bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f'round,pad=0.05',
                           facecolor=color, edgecolor='#2C3E50',
                           linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    if sublabel:
        ax.text(x, y + 0.12, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, zorder=4)
        ax.text(x, y - 0.18, sublabel, ha='center', va='center',
                fontsize=FS_SM, color='#555555', zorder=4, style='italic')
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, zorder=4)

def arrow(ax, x1, y1, x2, y2, label='', color='#2C3E50', lw=1.2,
          connectionstyle='arc3,rad=0.0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle=connectionstyle),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.15, label, ha='center', va='bottom',
                fontsize=6, color='#555555', zorder=4)

def bipartite(ax, cx, cy, n_left=4, n_right=3, color_l='#5DADE2',
              color_r='#E59866', radius=0.13):
    """Draw a small bipartite graph."""
    lx = cx - 0.35
    rx = cx + 0.35
    gap_l = 0.38
    gap_r = 0.45
    ly = [cy + (i - (n_left-1)/2) * gap_l  for i in range(n_left)]
    ry = [cy + (i - (n_right-1)/2) * gap_r for i in range(n_right)]
    # edges (sparse) — generated dynamically to avoid index out of range
    np.random.seed(42)
    edges = [(i, j) for i in range(n_left) for j in range(n_right)
             if np.random.rand() < 0.5]
    if not edges:
        edges = [(0, 0), (1, 1)]
    for li, ri in edges:
        ax.plot([lx, rx], [ly[li], ry[ri]],
                color='#AAB7B8', lw=0.7, zorder=1, alpha=0.7)
    for y_ in ly:
        c = plt.Circle((lx, y_), radius, color=color_l,
                        ec='#2C3E50', lw=0.8, zorder=2)
        ax.add_patch(c)
    for y_ in ry:
        c = plt.Circle((rx, y_), radius, color=color_r,
                        ec='#2C3E50', lw=0.8, zorder=2)
        ax.add_patch(c)

def gcn_layers(ax, cx, cy, n=3, color=C_GCN):
    """Draw stacked GCN layers."""
    w, h = 0.9, 0.38
    labels = [f'layer {i}' for i in range(n)]
    for i, lbl in enumerate(labels):
        yi = cy + (i - (n-1)/2) * 0.42
        rect = FancyBboxPatch((cx - w/2, yi - h/2), w, h,
                               boxstyle='round,pad=0.03',
                               facecolor=color, edgecolor='#2C3E50',
                               linewidth=0.9, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, yi, lbl, ha='center', va='center',
                fontsize=6, zorder=4)
    ax.text(cx, cy + (n-1)/2 * 0.42 + 0.32, 'GCN',
            ha='center', va='bottom', fontsize=7, fontweight='bold', zorder=4)


# ════════════════════════════════════════════════════════════
# SECTION LABELS (far left)
# ════════════════════════════════════════════════════════════

ax.text(0.25, 6.6, 'Behavior\n(User–Item)\nView',
        ha='center', va='center', fontsize=8, fontweight='bold',
        rotation=90, color='#1A5276')
ax.text(0.25, 2.5, 'Multimodal\n(Item–Item)\nView',
        ha='center', va='center', fontsize=8, fontweight='bold',
        rotation=90, color='#1E8449')

# Horizontal divider
ax.axhline(y=4.6, xmin=0.03, xmax=0.97,
           color='#BDC3C7', lw=1.0, linestyle='--', zorder=1)

# ════════════════════════════════════════════════════════════
# TOP ROW — Behavior View  (y ≈ 6.5)
# ════════════════════════════════════════════════════════════

Y_TOP = 6.5

# 1. User-item bipartite graph R
bipartite(ax, 1.3, Y_TOP, color_l='#5DADE2', color_r='#E59866')
ax.text(1.3, Y_TOP - 1.1, 'User-Item Graph (R)',
        ha='center', va='center', fontsize=FS_SM, color='#333')

# 2. Behavioural Simulator
box(ax, 3.2, Y_TOP, 1.5, 0.7, 'Behavioural\nSimulator', C_BEH,
    sublabel='(LightGCN)', fontsize=7)

# 3. User-wise Calibration ← PRISM INNOVATION (highlighted)
box(ax, 5.1, Y_TOP, 1.6, 0.8, 'User-wise\nCalibration (τ)', C_PRISM,
    fontsize=7, bold=True)
# Innovation star marker
ax.text(5.85, Y_TOP + 0.52, '★ PRISM',
        ha='left', va='center', fontsize=6,
        color='#C0392B', fontweight='bold', zorder=5)

# 4. Calibrated R*
bipartite(ax, 7.0, Y_TOP, n_left=4, n_right=3,
          color_l='#F1948A', color_r='#F1948A')
ax.text(7.0, Y_TOP - 1.1, 'Calibrated R*',
        ha='center', va='center', fontsize=FS_SM,
        color='#C0392B', fontweight='bold')

# 5. Merge: β*R* + (1-β)*R
box(ax, 8.85, Y_TOP, 1.5, 0.65, 'β·R* + (1-β)·R', C_MERGE, fontsize=7)

# 6. Refined Interaction Graph
bipartite(ax, 10.6, Y_TOP, n_left=4, n_right=3,
          color_l='#7FB3D3', color_r='#7FB3D3')
ax.text(10.6, Y_TOP - 1.1, 'Refined\nInteraction Graph',
        ha='center', va='center', fontsize=FS_SM, color='#333')

# 7. GCN (behavior)
gcn_layers(ax, 12.3, Y_TOP, n=3, color=C_BEH)

# 8. Output embeddings (behavior)
box(ax, 14.1, Y_TOP + 0.45, 1.5, 0.55, 'h_item,beh', C_OUT, fontsize=7)
box(ax, 14.1, Y_TOP - 0.45, 1.5, 0.55, 'h_user',     C_OUT, fontsize=7)

# ── Arrows — top row ──
arrow(ax, 2.0, Y_TOP, 2.45, Y_TOP)               # graph → simulator
arrow(ax, 3.95, Y_TOP, 4.3, Y_TOP)               # simulator → calibration
arrow(ax, 5.9, Y_TOP, 6.3, Y_TOP)                # calibration → R*
arrow(ax, 7.7, Y_TOP, 8.1, Y_TOP)                # R* → merge
# R (original) also feeds into merge
arrow(ax, 1.3, Y_TOP - 1.2, 8.85, Y_TOP - 0.35,
      connectionstyle='arc3,rad=-0.25', color='#7F8C8D')
arrow(ax, 9.6, Y_TOP, 9.9, Y_TOP)               # merge → refined graph
arrow(ax, 11.3, Y_TOP, 11.85, Y_TOP)             # refined graph → GCN
arrow(ax, 12.75, Y_TOP + 0.35, 13.35, Y_TOP + 0.45)  # GCN → h_item_beh
arrow(ax, 12.75, Y_TOP - 0.35, 13.35, Y_TOP - 0.45)  # GCN → h_user


# ════════════════════════════════════════════════════════════
# BOTTOM ROW — Multimodal View  (y ≈ 2.5)
# ════════════════════════════════════════════════════════════

Y_BOT = 2.5

# 1a. Textual graph node
box(ax, 1.3, Y_BOT + 0.55, 1.2, 0.5, 'Textual\nGraph', C_MULTI, fontsize=7)
# 1b. Visual graph node
box(ax, 1.3, Y_BOT - 0.55, 1.2, 0.5, 'Visual\nGraph',  C_MULTI, fontsize=7)

# 2. Weighted Fusion
box(ax, 3.2, Y_BOT, 1.5, 0.65, 'Weighted Fusion\n(0.9·T + 0.1·V)', C_MULTI, fontsize=7)

# 3. Item-Item Graph (with progressive refinement marker)
bipartite(ax, 5.1, Y_BOT, n_left=3, n_right=3,
          color_l='#58D68D', color_r='#58D68D')
ax.text(5.1, Y_BOT - 1.0, 'Item-Item Graph',
        ha='center', va='center', fontsize=FS_SM, color='#333')

# Progressive refinement annotation ← PRISM INNOVATION
box(ax, 5.1, Y_BOT + 1.45, 1.7, 0.55,
    '★ Progressive\nRefinement', C_PRISM, fontsize=6.5, bold=True)
ax.annotate('', xy=(5.1, Y_BOT + 0.65), xytext=(5.1, Y_BOT + 1.18),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.3), zorder=5)
# feedback arrow from GCN embeddings back to refinement
ax.annotate('', xy=(5.75, Y_BOT + 1.45), xytext=(12.3, Y_BOT + 1.45),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0,
                            connectionstyle='arc3,rad=0.0'),
            zorder=2)
ax.text(9.0, Y_BOT + 1.6, 'item embeddings feedback',
        ha='center', va='bottom', fontsize=6, color='#C0392B', style='italic')

# 4. GCN (multimodal)
gcn_layers(ax, 7.0, Y_BOT, n=3, color=C_MULTI)

# 5. Gated Mechanism
box(ax, 9.0, Y_BOT, 1.6, 0.72, 'Gated\nMechanism', C_MULTI,
    sublabel='(g_i)', fontsize=7)

# 6. h_item_pref (preference-relevant)
box(ax, 10.95, Y_BOT + 0.55, 1.5, 0.52, 'h_item,pref', C_OUT,
    fontsize=7)
box(ax, 10.95, Y_BOT - 0.55, 1.5, 0.52, 'h_item,irrel', C_GCN,
    fontsize=7)

# L_ortho
box(ax, 12.5, Y_BOT - 0.55, 0.9, 0.4, 'L_ortho', C_LOSS, fontsize=7)

# ── Arrows — bottom row ──
arrow(ax, 1.3, Y_BOT + 0.3,  2.45, Y_BOT + 0.12)   # text → fusion
arrow(ax, 1.3, Y_BOT - 0.3,  2.45, Y_BOT - 0.12)   # visual → fusion
arrow(ax, 3.95, Y_BOT, 4.3,  Y_BOT)                  # fusion → item graph
arrow(ax, 5.85, Y_BOT, 6.55, Y_BOT)                  # item graph → GCN
arrow(ax, 7.45, Y_BOT, 8.2,  Y_BOT)                  # GCN → gated
arrow(ax, 9.8,  Y_BOT + 0.22, 10.2, Y_BOT + 0.5)    # gated → h_pref
arrow(ax, 9.8,  Y_BOT - 0.22, 10.2, Y_BOT - 0.5)    # gated → h_irrel
arrow(ax, 11.7, Y_BOT - 0.55, 12.05, Y_BOT - 0.55)  # h_irrel → L_ortho


# ════════════════════════════════════════════════════════════
# RIGHT SIDE — Combination + Losses
# ════════════════════════════════════════════════════════════

# Enhanced item representation: h_item_beh + h_item_pref → h_item
box(ax, 15.9, 5.1, 1.4, 0.58, 'h_item\n(enhanced)', C_OUT,
    fontsize=7, bold=False)
ax.text(15.2, 5.5, '+', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2C3E50')

# InfoNCE between h_item_beh and h_item_pref
box(ax, 15.9, 3.6, 1.4, 0.55, 'L_InfoNCE', C_LOSS, fontsize=7)

# Final prediction + BPR
box(ax, 17.8, 5.1, 1.5, 0.58, 'ŷ  →  L_bpr', C_LOSS,
    fontsize=7, bold=True)

# ── Arrows — right side ──
# h_item_beh → h_item (enhanced)
arrow(ax, 14.85, Y_TOP + 0.45, 15.2, 5.35)
# h_item_pref → h_item (enhanced)
arrow(ax, 11.7, Y_BOT + 0.55, 15.2, 4.95,
      connectionstyle='arc3,rad=-0.15')
# h_user → prediction
arrow(ax, 14.85, Y_TOP - 0.45, 17.8, 5.35,
      connectionstyle='arc3,rad=0.2')
# h_item → prediction
arrow(ax, 16.6, 5.1, 17.05, 5.1)
# h_item_beh → InfoNCE
arrow(ax, 14.85, Y_TOP + 0.3, 15.2, 3.8,
      connectionstyle='arc3,rad=0.2', color='#8E44AD')
# h_item_pref → InfoNCE
arrow(ax, 11.7, Y_BOT + 0.4, 15.2, 3.6,
      connectionstyle='arc3,rad=-0.1', color='#8E44AD')

# ════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════

legend_items = [
    mpatches.Patch(facecolor=C_BEH,   edgecolor='#2C3E50', label='Behavior Module'),
    mpatches.Patch(facecolor=C_MULTI, edgecolor='#2C3E50', label='Multimodal Module'),
    mpatches.Patch(facecolor=C_PRISM, edgecolor='#2C3E50', label='PRISM Innovation ★'),
    mpatches.Patch(facecolor=C_LOSS,  edgecolor='#2C3E50', label='Loss Function'),
    mpatches.Patch(facecolor=C_OUT,   edgecolor='#2C3E50', label='Output Embedding'),
]
ax.legend(handles=legend_items, loc='lower right',
          fontsize=7.5, framealpha=0.9,
          bbox_to_anchor=(0.99, 0.01))

# Title
ax.text(10, 8.7, 'PRISM: Progressive Refinement with calIbrated Simulation for Multimodal Recommendation',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#1A252F')

plt.tight_layout()
plt.savefig('plots/prism_architecture.pdf', bbox_inches='tight', dpi=300)
plt.savefig('plots/prism_architecture.png', bbox_inches='tight', dpi=300)
plt.close()
print('Saved: plots/prism_architecture.pdf  and  plots/prism_architecture.png')

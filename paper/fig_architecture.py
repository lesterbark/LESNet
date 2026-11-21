"""Figure 1: one LESNet layer, drawn.  run.bat -u fig_architecture.py -> figures/architecture.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

INK, MUTE, ACC, ACC2, BG = "#1d2330", "#8a93a6", "#d1495b", "#2e86ab", "#ffffff"
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(11.5, 3.9), dpi=170, facecolor=BG)

# (a) the set and one receiver's partner list ---------------------------------------------------
ax = fig.add_axes([0.01, 0.06, 0.28, 0.82]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
pts = rng.uniform(0.08, 0.92, (22, 2))
i = 7
pts[i] = (0.5, 0.5)
d = np.linalg.norm(pts - pts[i], axis=1)
order = np.argsort(d)[1:7]
for j in order:
    ax.add_patch(FancyArrowPatch(pts[j], pts[i], arrowstyle="-|>", mutation_scale=9, lw=1.2, color=ACC2, alpha=0.9,
                                 shrinkA=3, shrinkB=4))
ax.add_patch(plt.Circle(pts[i], d[order[-1]] + 0.01, fill=False, ls=(0, (3, 3)), lw=0.9, ec=MUTE))
ax.scatter(pts[:, 0], pts[:, 1], s=34, c=MUTE, zorder=3)
ax.scatter(pts[order, 0], pts[order, 1], s=40, c=ACC2, zorder=4)
ax.scatter(*pts[i], s=90, c=ACC, zorder=5)
ax.text(0.5, 0.97, "a  a set of entities", ha="center", va="top", fontsize=10, color=INK, weight="bold")
ax.text(0.5, 0.01, "receiver i (red) and its K nearest partners j (blue)", ha="center", fontsize=7.5, color=MUTE)

# (b) the pair filter ----------------------------------------------------------------------------
ax = fig.add_axes([0.31, 0.06, 0.36, 0.82]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 0.97, "b  one pair, one slice k", ha="center", va="top", fontsize=10, color=INK, weight="bold")


def box(x, y, w, h, text, fc, ec=INK, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


box(0.02, 0.70, 0.22, 0.13, "$a_i[k]=x_i\\!\\cdot\\!A_k$", "#fde8ea")
box(0.02, 0.50, 0.22, 0.13, "$b_j[k]=x_j\\!\\cdot\\!B_k$", "#e3f1f7")
box(0.02, 0.30, 0.22, 0.13, "$E_k\\!\\cdot\\!pf_{ij}$", "#eef0f4")
box(0.02, 0.10, 0.22, 0.13, "$c_k$", "#eef0f4")
box(0.40, 0.40, 0.17, 0.20, "$\\Sigma$", "#ffffff", fs=14)
for y in (0.765, 0.565, 0.365, 0.165):
    ax.add_patch(FancyArrowPatch((0.245, y), (0.40, 0.50), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=MUTE))
box(0.66, 0.40, 0.17, 0.20, "leaky", "#ffffff")
ax.add_patch(FancyArrowPatch((0.57, 0.50), (0.66, 0.50), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK))
ax.text(0.915, 0.50, "$v_{ij}[k]$", fontsize=10, va="center", ha="center", color=INK)
ax.add_patch(FancyArrowPatch((0.83, 0.50), (0.865, 0.50), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK))
ax.text(0.5, 0.01, "no hidden layer: the filter is one linear map per slice", ha="center", fontsize=7.5, color=MUTE)

# (c) the softmax combine -----------------------------------------------------------------------
ax = fig.add_axes([0.70, 0.06, 0.29, 0.82]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 0.97, "c  softmax combine over partners", ha="center", va="top", fontsize=10, color=INK, weight="bold")
v = np.array([0.2, 1.1, -0.3, 0.6, 1.7, 0.1])
w = np.exp(v - v.max()); w /= w.sum()
x0 = 0.06
for n, (vv, ww) in enumerate(zip(v, w)):
    xx = x0 + n * 0.145
    ax.add_patch(plt.Rectangle((xx, 0.62), 0.10, 0.12 * (vv + 0.6) / 2.3 + 0.01, fc=ACC2 if n != 4 else ACC, ec="none", alpha=0.9))
    ax.add_patch(plt.Rectangle((xx, 0.30), 0.10, 0.28 * ww / w.max(), fc=INK, ec="none", alpha=0.85))
ax.text(0.5, 0.80, "values $v_{ij}$ (one per partner)", ha="center", fontsize=8, color=INK)
ax.text(0.5, 0.15, "weights $w_{ij}=\\mathrm{softmax}_j(v_{ij})$; the value is its own score", ha="center", fontsize=7.6, color=INK)
ax.text(0.5, 0.02, "$y_i[k]=\\mathrm{leaky}(\\sum_j w_{ij}v_{ij})$", ha="center", fontsize=10, color=ACC)

fig.savefig(os.path.join(HERE, "figures", "architecture.png"), facecolor=BG)
print("wrote figures/architecture.png")

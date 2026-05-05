"""Side-by-side architecture diagram: original NeRF vs. IGM NeRF.

Produces a single PNG showing the two MLPs as block diagrams. Color coding:
- Sky-blue boxes: components that are identical between the two
- Salmon boxes: components that are replaced or absent in IGM NeRF
- Yellow halo: the differentiable physics rendering operator (the contribution)

Output: experiments/nerf/talk/figures/nerf_vs_igmnerf_arch.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# -----------------------------------------------------------------------------
# Layout constants
# -----------------------------------------------------------------------------
FIG_W, FIG_H = 14.0, 10.5           # inches
COL_X = {"nerf": 3.5, "igm": 10.5}  # column centers
BOX_W = 4.0
BOX_H = 0.55
GAP = 0.20

KEPT = "#a8d8ea"      # sky blue — identical
DIFFERS = "#ffadad"   # salmon — replaced / absent
ACCENT = "#ffd166"    # yellow — physics-rendering accent

# -----------------------------------------------------------------------------
# Box helper
# -----------------------------------------------------------------------------
def box(ax, x, y, label, color=KEPT, w=BOX_W, h=BOX_H, fontsize=10, weight="normal"):
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor="#333333", facecolor=color, zorder=2,
    )
    ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, style="-|>", color="#333", lw=1.2, zorder=1):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, zorder=zorder,
    )
    ax.add_patch(a)


def skip_arrow(ax, x_in, y_in, x_out, y_out, side_x, color="#888", lw=1.0):
    """U-shaped skip connection routed around the column."""
    ax.plot(
        [x_in, side_x, side_x, x_out],
        [y_in, y_in, y_out, y_out],
        color=color, linewidth=lw, linestyle="--", zorder=1,
    )
    arrow(ax, side_x, y_out, x_out, y_out, color=color, lw=lw)


# -----------------------------------------------------------------------------
# Build figure
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=150)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

# Color-key strip (top of figure, above column headers) -----------------------
def _key_swatch(x, y, color, label):
    p = FancyBboxPatch(
        (x - 0.18, y - 0.18), 0.36, 0.36,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.0, edgecolor="#333", facecolor=color,
    )
    ax.add_patch(p)
    ax.text(x + 0.30, y, label, ha="left", va="center", fontsize=12)

_key_swatch(3.0, FIG_H - 0.40, KEPT, "Kept identical from NeRF")
_key_swatch(8.5, FIG_H - 0.40, DIFFERS, "Replaced / different in IGM NeRF")

# Column titles ---------------------------------------------------------------
ax.text(COL_X["nerf"], FIG_H - 1.20, "Original NeRF",
        ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(COL_X["igm"], FIG_H - 1.20, "IGM NeRF (this work)",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color="#a01313")
ax.text(COL_X["nerf"], FIG_H - 1.55,
        "(Mildenhall et al., 2020)", ha="center", va="center",
        fontsize=9, style="italic", color="#555")
ax.text(COL_X["igm"], FIG_H - 1.55,
        "differentiable physics rendering",
        ha="center", va="center", fontsize=9, style="italic", color="#a01313")

# Layout y positions ----------------------------------------------------------
y0 = FIG_H - 2.4   # leave space for legend + headers
ys = [y0 - i * (BOX_H + GAP) for i in range(11)]

# -----------------------------------------------------------------------------
# Left column — original NeRF
# -----------------------------------------------------------------------------
xn = COL_X["nerf"]

box(ax, xn, ys[0], r"Input: $(x, y, z, \theta, \phi)$  —  5D", color=DIFFERS,
    weight="bold")
box(ax, xn, ys[1], r"Fourier $\gamma(\cdot)$  ($L_{\rm pos}=10$, $L_{\rm dir}=4$)",
    color=KEPT)
box(ax, xn, ys[2], r"FC 256 + ReLU  (layer 1)", color=KEPT)
box(ax, xn, ys[3], r"FC 256 + ReLU  (layer 2)", color=KEPT)
box(ax, xn, ys[4], r"FC 256 + ReLU  (layer 3)", color=KEPT)
box(ax, xn, ys[5], r"FC 256 + ReLU  (layer 4)", color=KEPT)
box(ax, xn, ys[6], r"$\oplus$  skip: concat $\gamma(\mathbf{x})$", color=KEPT,
    weight="bold")
box(ax, xn, ys[7], r"FC 256 + ReLU  (layer 5–7)", color=KEPT)
box(ax, xn, ys[8], r"FC 256 + ReLU  (layer 8)", color=KEPT)
box(ax, xn, ys[9],
    r"Heads:  density $\sigma$  +  view-cond. RGB $(r,g,b)$",
    color=DIFFERS, weight="bold")
box(ax, xn, ys[10],
    r"Volume rendering integral $C(\mathbf{r})=\int T\,\sigma\,c\,dt$",
    color=DIFFERS, weight="bold", h=0.7)

# Arrows
for i in range(10):
    arrow(ax, xn, ys[i] - BOX_H / 2, xn, ys[i + 1] + BOX_H / 2)

# Skip connection (dashed, around the right side of column)
skip_arrow(
    ax,
    x_in=xn + BOX_W / 2,
    y_in=ys[1],
    x_out=xn + BOX_W / 2,
    y_out=ys[6],
    side_x=xn + BOX_W / 2 + 0.5,
)

# -----------------------------------------------------------------------------
# Right column — IGM NeRF
# -----------------------------------------------------------------------------
xi = COL_X["igm"]

box(ax, xi, ys[0],
    r"Input: $(x, y, z)$  —  3D  ($\mathit{no\ view\ direction}$)",
    color=DIFFERS, weight="bold")
box(ax, xi, ys[1], r"Fourier $\gamma(\cdot)$  ($L=10$, position only)",
    color=KEPT)
box(ax, xi, ys[2], r"FC 256 + ReLU  (layer 1)", color=KEPT)
box(ax, xi, ys[3], r"FC 256 + ReLU  (layer 2)", color=KEPT)
box(ax, xi, ys[4], r"FC 256 + ReLU  (layer 3)", color=KEPT)
box(ax, xi, ys[5], r"FC 256 + ReLU  (layer 4)", color=KEPT)
box(ax, xi, ys[6], r"$\oplus$  skip: concat $\gamma(\mathbf{x})$", color=KEPT,
    weight="bold")
box(ax, xi, ys[7], r"FC 256 + ReLU  (layer 5–7)", color=KEPT)
box(ax, xi, ys[8], r"FC 256 + ReLU  (layer 8)", color=KEPT)
box(ax, xi, ys[9],
    r"Heads:  $\rho,\ T,\ X_{\rm HI},\ v_{\rm pec}$  (bounded physics)",
    color=DIFFERS, weight="bold")
box(ax, xi, ys[10],
    r"Voigt–Hjerting + RSD:  $\tau(v_{\rm obs})\!=\!\mathcal{A}\!\sum_{\rm src}\!\frac{n_{\rm HI}\,H(a,x)}{b\sqrt{\pi}}$",
    color=DIFFERS, weight="bold", h=0.7)

# Arrows
for i in range(10):
    arrow(ax, xi, ys[i] - BOX_H / 2, xi, ys[i + 1] + BOX_H / 2)

# Skip connection
skip_arrow(
    ax,
    x_in=xi + BOX_W / 2,
    y_in=ys[1],
    x_out=xi + BOX_W / 2,
    y_out=ys[6],
    side_x=xi + BOX_W / 2 + 0.5,
)

# -----------------------------------------------------------------------------
# Legend
# -----------------------------------------------------------------------------
# Color key is the swatch strip at the top of the figure (drawn above).

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
out_dir = Path("experiments/nerf/talk/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "nerf_vs_igmnerf_arch.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved {out_path}  ({out_path.stat().st_size // 1024} KB)")

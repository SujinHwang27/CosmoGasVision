"""
Generate papers/shared/figures/method_pipeline.png mirroring the LEDGER §0
amended Mermaid diagram (post [D-06]/[D-07]/[D-24]/[D-11]/[D-21]/[D-23]).

Produces a 4-subgraph pipeline diagram with matching colors/labels:
  1. Input Space & Encoding              (gray   #f9f9f9 / stroke #333333)
  2. Latent IGM Neural Field (MLP)       (blue   #e1f5fe / stroke #01579b)
  3. Differentiable Physics Bridge       (orange #fff3e0 / stroke #e65100)
  4. Optimization Objectives             (green  #f1f8e9 / stroke #33691e)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- canvas ---------------------------------------------------------------
FIG_W, FIG_H = 16.0, 9.0          # inches  -> 2400x1350 px @ 150 dpi
DPI = 150
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "papers" / "shared" / "figures" / "method_pipeline.png"
)

# Subgraph color palette (Mermaid styling in LEDGER §0)
COLORS = {
    "input":   {"fill": "#f9f9f9", "stroke": "#333333"},
    "neural":  {"fill": "#e1f5fe", "stroke": "#01579b"},
    "physics": {"fill": "#fff3e0", "stroke": "#e65100"},
    "supvis":  {"fill": "#f1f8e9", "stroke": "#33691e"},
}
NODE_FILL = "#ffffff"


# ---- helpers --------------------------------------------------------------
def add_subgraph(ax, x, y, w, h, title, palette, title_color=None):
    """Rounded rectangle + title for a subgraph (group container)."""
    p = COLORS[palette]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=2.0, edgecolor=p["stroke"], facecolor=p["fill"],
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.18, y + h - 0.30, title,
        fontsize=12, fontweight="bold",
        color=title_color or p["stroke"], zorder=2,
    )


def add_node(ax, cx, cy, w, h, text, palette,
             dashed=False, fontsize=9.5, fontweight="normal"):
    """A leaf node (white box, colored stroke) centered at (cx, cy)."""
    p = COLORS[palette]
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        linewidth=1.6, edgecolor=p["stroke"], facecolor=NODE_FILL,
        linestyle=("--" if dashed else "-"), zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        cx, cy, text, ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, color="#111111",
        zorder=4,
    )
    return (cx, cy, w, h)


def edge_point(node, side):
    """Return a point on the boundary of a node for arrow attachment."""
    cx, cy, w, h = node
    if side == "right":  return (cx + w / 2, cy)
    if side == "left":   return (cx - w / 2, cy)
    if side == "top":    return (cx, cy + h / 2)
    if side == "bottom": return (cx, cy - h / 2)
    raise ValueError(side)


def connect(ax, src, dst, src_side="right", dst_side="left",
            color="#222", lw=1.4, dashed=False, label=None,
            label_offset=(0.0, 0.18), rad=0.0):
    a = edge_point(src, src_side)
    b = edge_point(dst, dst_side)
    style = "->" if not dashed else "->"
    arrow = FancyArrowPatch(
        a, b,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color,
        linestyle=("--" if dashed else "-"),
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (a[0] + b[0]) / 2 + label_offset[0]
        my = (a[1] + b[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=8.5, color=color,
                ha="center", va="center", style="italic",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1.2),
                zorder=5)


# ---- build figure ---------------------------------------------------------
def build():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # ===== Subgraph 1: Input Space & Encoding ==============================
    add_subgraph(ax, 0.20, 5.20, 3.20, 3.40,
                 "1. Input Space & Encoding", "input")
    coord = add_node(
        ax, 1.80, 7.55, 2.60, 0.85,
        "3D Comoving Coordinates\n(x, y, z)", "input",
    )
    fourier = add_node(
        ax, 1.80, 6.05, 2.60, 0.95,
        "Fourier Positional\nEncoding (L=10)", "input",
    )
    connect(ax, coord, fourier, "bottom", "top")

    # ===== Subgraph 2: Latent IGM Neural Field (MLP) =======================
    add_subgraph(ax, 3.80, 4.40, 4.20, 4.20,
                 "2. Latent IGM Neural Field (MLP)", "neural")
    mlp = add_node(
        ax, 4.85, 6.50, 1.85, 1.10,
        "8-Layer MLP  (F$_\\theta$)\n256 Hidden Units",
        "neural", dashed=True, fontweight="bold",
    )
    rho = add_node(ax, 7.10, 7.70, 1.55, 0.55,
                   r"$\rho$  (Softplus)", "neural")
    temp = add_node(ax, 7.10, 7.00, 1.55, 0.55,
                    r"$T$  (Softplus$\times$1e4 + 1e3)",
                    "neural", fontsize=8.5)
    xhi = add_node(ax, 7.10, 6.30, 1.55, 0.55,
                   r"$X_{HI}$  (Sigmoid)", "neural")
    vpec = add_node(ax, 7.10, 5.60, 1.55, 0.55,
                    r"$v_{\rm pec}$  (Tanh$\times$500)",
                    "neural", fontsize=8.5)

    # Fourier -> MLP
    connect(ax, fourier, mlp, "right", "left")
    # MLP -> 4 outputs
    for out in (rho, temp, xhi, vpec):
        connect(ax, mlp, out, "right", "left", lw=1.2)

    # ===== Subgraph 3: Differentiable Physics Bridge =======================
    add_subgraph(ax, 8.40, 4.40, 3.50, 4.20,
                 "3. Differentiable Physics Bridge", "physics")
    sightlines = add_node(
        ax, 10.15, 7.45, 3.05, 0.85,
        "1D Sightline\nSampling Grid (RSD-convolved [D-06])",
        "physics", fontsize=9,
    )
    voigt = add_node(
        ax, 10.15, 6.20, 3.05, 0.95,
        "Analytic Voigt Kernel\n(Tepper-García 2006)",
        "physics",
    )
    tau_rend = add_node(
        ax, 10.15, 4.95, 3.05, 0.95,
        r"Rendered Optical Depth"
        "\n"
        r"$\tau_{\rm rendered}$  (redshift-space)",
        "physics",
    )

    # MLP outputs -> sightlines (collapse 4 arrows visually)
    for out in (rho, temp, xhi, vpec):
        connect(ax, out, sightlines, "right", "left",
                lw=0.9, color="#666", rad=-0.15)
    connect(ax, sightlines, voigt, "bottom", "top")
    connect(ax, voigt, tau_rend, "bottom", "top")

    # ===== Subgraph 4: Optimization Objectives =============================
    add_subgraph(ax, 0.20, 0.30, 11.70, 3.80,
                 "4. Optimization Objectives  [D-24] / [D-11] / [D-21]",
                 "supvis")

    # ground truth on the left
    truth = add_node(
        ax, 1.55, 3.10, 2.40, 0.85,
        "Simulation Ground Truth\n" + r"$\tau_{\rm GT}$",
        "supvis",
    )
    # mask
    mask = add_node(
        ax, 4.85, 3.10, 3.20, 0.95,
        "Saturated-Absorber Mask  [D-24]\n"
        r"core $\tau_{\rm GT}\!>\!10^{5}$;  wing $\tau_{\rm GT}\!>\!10$ CC",
        "supvis", fontsize=8.5,
    )
    connect(ax, truth, mask, "right", "left")

    # L_data
    l_data = add_node(
        ax, 4.85, 1.55, 3.85, 1.05,
        r"$\mathcal{L}_{\rm data}$  (log1p+cap+mask MSE)  [D-24]"
        "\n"
        r"$\langle (\log(1{+}\tau^{\rm eff}_{\rm pred}){-}\log(1{+}\tau^{\rm eff}_{\rm GT}))^{2}\rangle_{\rm non\!-\!DLA}$"
        "\n"
        r"$\tau^{\rm eff}=\min(\tau,\, \tau_{\max}{=}10)$",
        "supvis", fontsize=8.2,
    )
    # L_meanF
    l_meanf = add_node(
        ax, 9.10, 1.55, 4.10, 1.05,
        r"$\mathcal{L}_{\rm meanF}$  (mean-flux soft anchor)  [D-11]"
        "\n"
        r"$\lambda_{F}\,(\langle F_{\rm pred}\rangle - \langle F\rangle_{\rm obs})^{2}$"
        "\n"
        r"$\langle F\rangle_{\rm obs}=0.979$  (Kirkman+2007, $z\!=\!0.3$)",
        "supvis", fontsize=8.2,
    )

    # tau_GT and mask -> L_data ; tau_rendered -> L_data ; mask -> L_meanF ; tau_rendered -> L_meanF
    connect(ax, truth, l_data, "bottom", "top", rad=-0.10)
    connect(ax, mask, l_data, "bottom", "top")
    connect(ax, tau_rend, l_data, "bottom", "right",
            color="#e65100", lw=1.4, rad=0.25)
    connect(ax, mask, l_meanf, "bottom", "top", rad=0.15)
    connect(ax, tau_rend, l_meanf, "bottom", "top",
            color="#e65100", lw=1.4, rad=-0.05)

    # Two-pass surrogate
    twopass = add_node(
        ax, 14.10, 1.55, 3.40, 2.45,
        "[D-21] Two-Pass Surrogate\n\n"
        "Pass 1:  cycle-mean\n"
        r"$\langle F\rangle_{\rm cycle}$  (no\_grad)"
        "\n\n"
        "Pass 2:  per-microbatch\n"
        "linearized backward\n\n"
        r"$\Rightarrow$ accum-step invariant",
        "supvis", fontsize=8.6,
    )
    # The two-pass surrogate node is intentionally placed in the right
    # margin (outside the supvis subgraph) to make the backprop arrow to
    # the MLP visually clear.

    connect(ax, l_data, twopass, "right", "left")
    connect(ax, l_meanf, twopass, "right", "left")

    # Backprop: twopass -.-> MLP (dashed, labeled)
    connect(ax, twopass, mlp, "top", "bottom",
            dashed=True, color="#33691e", lw=1.8,
            label="Backpropagation", label_offset=(-0.4, 0.25),
            rad=-0.30)

    # ---- header / footer --------------------------------------------------
    ax.text(
        8.0, 8.78,
        "CosmoGasVision  —  NeRF Track Methodology Pipeline",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="#222222",
    )
    ax.text(
        8.0, 0.05,
        "Mirrors LEDGER §0 (post [D-06]/[D-07]/[D-24]/[D-11]/[D-21]/[D-23] amendments)",
        ha="center", va="bottom", fontsize=8.5,
        color="#555555", style="italic",
    )

    return fig


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = build()
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    size = os.path.getsize(OUT_PATH)
    print(f"WROTE {OUT_PATH}  ({size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()

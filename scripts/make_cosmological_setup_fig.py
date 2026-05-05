"""Schematic figure for the astrophysical setup — for the defense talk.

Shows (top) a cosmological volume with filamentary intergalactic gas, a
background quasar, a sightline through the box to the observer; (bottom)
the resulting 1D absorption spectrum that the sightline produces — the
Lyman-alpha forest, with one saturated DLA and many forest features.

Output: experiments/nerf/talk/figures/cosmological_setup.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MPath

rng = np.random.default_rng(20260504)

# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(14.0, 7.5), dpi=150)

# Top axes: cosmological volume schematic
ax_top = fig.add_axes([0.05, 0.55, 0.90, 0.42])
ax_top.set_xlim(0, 10)
ax_top.set_ylim(0, 4)
ax_top.set_aspect("equal")
ax_top.axis("off")

# Bottom axes: 1D Lyman-alpha forest spectrum
ax_bot = fig.add_axes([0.10, 0.10, 0.80, 0.28])
ax_bot.set_xlim(0, 5400)         # km/s, full Sherwood range
ax_bot.set_ylim(-0.2, 6.2)

# -----------------------------------------------------------------------------
# Top panel: cosmological volume + sightline
# -----------------------------------------------------------------------------

# Sky-blue background patch for "cosmic" feel
ax_top.add_patch(Rectangle((0, 0), 10, 4, facecolor="#0e1626", zorder=0))

# Volume box (oblique 2D for a hint of 3D)
box_x0, box_y0 = 2.5, 0.6
box_w, box_h = 5.0, 2.6
rear_dx, rear_dy = 0.6, 0.4
front_color = "#1a2840"
side_color = "#0f1c34"

# Front face
ax_top.add_patch(Rectangle((box_x0, box_y0), box_w, box_h,
                           facecolor=front_color, edgecolor="#5a7fb0",
                           linewidth=1.2, zorder=2))
# Top edge (perspective)
ax_top.plot([box_x0, box_x0 + rear_dx, box_x0 + rear_dx + box_w, box_x0 + box_w],
            [box_y0 + box_h, box_y0 + box_h + rear_dy,
             box_y0 + box_h + rear_dy, box_y0 + box_h],
            color="#5a7fb0", linewidth=1.2, zorder=3)
# Right edge (perspective)
ax_top.plot([box_x0 + box_w, box_x0 + box_w + rear_dx],
            [box_y0, box_y0 + rear_dy],
            color="#5a7fb0", linewidth=1.2, zorder=3)
ax_top.plot([box_x0 + box_w + rear_dx, box_x0 + box_w + rear_dx],
            [box_y0 + rear_dy, box_y0 + rear_dy + box_h],
            color="#5a7fb0", linewidth=1.2, linestyle=":", zorder=3)

# "Cosmic web" filaments inside the box: random Gaussian blobs + filament lines
N_filaments = 9
for _ in range(N_filaments):
    cx = box_x0 + 0.4 + rng.uniform(0, box_w - 0.8)
    cy = box_y0 + 0.3 + rng.uniform(0, box_h - 0.6)
    rad = rng.uniform(0.06, 0.22)
    intensity = rng.uniform(0.4, 0.95)
    ax_top.add_patch(Circle((cx, cy), rad,
                            facecolor=(intensity, intensity * 0.85, 0.65),
                            alpha=0.55, edgecolor="none", zorder=4))
# Connect a few blobs with filament-like curves (Bezier)
for _ in range(7):
    p0 = (box_x0 + rng.uniform(0.3, box_w - 0.3),
          box_y0 + rng.uniform(0.3, box_h - 0.3))
    p1 = (box_x0 + rng.uniform(0.3, box_w - 0.3),
          box_y0 + rng.uniform(0.3, box_h - 0.3))
    cp = ((p0[0] + p1[0]) / 2 + rng.uniform(-0.5, 0.5),
          (p0[1] + p1[1]) / 2 + rng.uniform(-0.5, 0.5))
    verts = [p0, cp, p1]
    codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
    path = MPath(verts, codes)
    from matplotlib.patches import PathPatch
    ax_top.add_patch(PathPatch(path, facecolor="none",
                               edgecolor="#ffd2a8", alpha=0.35,
                               linewidth=1.0, zorder=4))

# Background quasar (left, outside box)
qx, qy = 0.7, box_y0 + box_h / 2
star_outer = plt.scatter([qx], [qy], s=900, marker="*",
                         color="#ffd166", edgecolors="#ffae0d",
                         linewidths=1.2, zorder=6)
ax_top.text(qx, qy + 0.55, "background\nquasar",
            ha="center", va="center", fontsize=11,
            color="#ffd166", fontweight="bold", zorder=7)

# Observer (right, outside box)
ox, oy = 9.3, box_y0 + box_h / 2
ax_top.scatter([ox], [oy], s=400, marker="o",
               facecolor="#a8d8ea", edgecolor="#1f6f8b",
               linewidths=1.5, zorder=6)
ax_top.text(ox, oy + 0.55, "observer\n(us)",
            ha="center", va="center", fontsize=11,
            color="#a8d8ea", fontweight="bold", zorder=7)

# Sightline (yellow ray, with arrowhead at observer)
sight_y = qy
ax_top.add_patch(FancyArrowPatch(
    (qx + 0.25, qy), (ox - 0.25, oy),
    arrowstyle="-|>", mutation_scale=15,
    linewidth=2.0, color="#ffd166", zorder=5,
))
ax_top.text((qx + ox) / 2, sight_y + 0.18,
            "sightline (ray through the volume)",
            ha="center", va="bottom", fontsize=10,
            color="#ffd166", style="italic", zorder=7)

# Volume label  (placed inside the box near the top so it never clips)
ax_top.text(box_x0 + box_w / 2, box_y0 + box_h - 0.18,
            r"60 Mpc/h cosmological volume    ($z = 0.3$, Sherwood sim)",
            ha="center", va="center",
            fontsize=11, color="#e9ecef", fontweight="bold", zorder=7)
ax_top.text(box_x0 + box_w / 2, box_y0 - 0.30,
            r"intergalactic gas:  $\rho,\ T,\ X_{\rm HI},\ v_{\rm pec}$  (3D fields)",
            ha="center", va="center",
            fontsize=11, color="#a8d8ea", zorder=7)

# -----------------------------------------------------------------------------
# Bottom panel: τ(v) spectrum with realistic Sherwood-style structure
#
# The real Lyman-α forest at z = 0.3 is "thin": sparse, sharp, narrow
# absorbers on a near-zero baseline (the universe has expanded, gas is
# clumpy, most of the LOS is empty). A single dominant absorber dwarfs
# the rest. This panel matches that profile.
# -----------------------------------------------------------------------------
v = np.linspace(0, 5400, 2048)
tau = np.full_like(v, 0.02)  # near-zero baseline

# Narrow absorbers: (center_v in km/s, peak τ, FWHM in km/s)
absorbers = [
    (350,  0.45, 16),
    (470,  0.20, 10),
    (1180, 0.30, 14),
    (1520, 0.18, 12),
    (1640, 0.22, 14),
    (2640, 5.20, 22),   # dominant absorber
    (2900, 0.45, 14),
    (3050, 0.55, 16),
    (3220, 0.40, 14),
    (3560, 0.50, 18),
    (4470, 0.60, 18),
    (4880, 0.45, 16),
    (5180, 1.70, 18),   # late cluster, three close peaks
    (5260, 1.20, 14),
    (5320, 1.00, 14),
]
for v0, peak, fwhm in absorbers:
    sigma = fwhm / 2.355
    tau += peak * np.exp(-((v - v0) / sigma) ** 2)

ax_bot.plot(v, tau, color="#1f77b4", linewidth=1.2)
ax_bot.axhline(y=0, color="#bbb", linewidth=0.5, zorder=0)

# Annotate the dominant absorber
v_dom = 2640
ax_bot.annotate("dominant absorber",
                xy=(v_dom, 5.20),
                xytext=(v_dom + 800, 5.15),
                fontsize=10, color="#a01313",
                arrowprops=dict(arrowstyle="->", color="#a01313", lw=1.0),
                zorder=5)

# Caption — what the field is called
ax_bot.text(150, 4.0, r"Lyman-$\alpha$ forest spectrum  ($\tau$ vs. velocity)",
            fontsize=11, color="#222", fontweight="bold")

# Axes
ax_bot.set_xlabel(r"observed velocity  $v_{\rm obs}$  (km/s)",
                  fontsize=11, color="#222")
ax_bot.set_ylabel(r"optical depth  $\tau$", fontsize=11, color="#222")
ax_bot.tick_params(colors="#444", labelsize=9)
for spine in ("top", "right"):
    ax_bot.spines[spine].set_visible(False)

# Bridge text between the two panels
fig.text(0.5, 0.49,
         "         each sightline produces a 1D absorption spectrum         ",
         ha="center", va="center", fontsize=12, fontweight="bold",
         color="#222",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffd166",
                   edgecolor="#ffae0d"))

# (figure title omitted — the slide that hosts the figure provides its own title)

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
out_dir = Path("experiments/nerf/talk/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "cosmological_setup.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved {out_path}  ({out_path.stat().st_size // 1024} KB)")

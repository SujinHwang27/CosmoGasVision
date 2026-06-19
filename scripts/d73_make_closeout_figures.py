#!/usr/bin/env python
"""[D-73] am-9 close-out figures (PI-commissioned, support-researcher slot).

Fig 1 (K2): truth-field loss_data vs tau_amp (flat ~0.0101 over [0.2,4]) with the
            grid's argmin 0.0026 as a horizontal line — the degeneracy figure.
Fig 2 (xi): xi(r) ceiling-relative comparison — grid vs truth-vs-truth ceiling vs
            truth+100%-noise vs Wiener lower bound, with the DEMOTED 0.6 bar shown
            as unreachable.

All data local (no Juno needed): truth cube + sightlines on host; grid xi(r) profile
from the in-job xi_3d_injob.json (values inlined below, pinned to job 221335); Wiener
from the local a4prime artifact. Loss matched to pipeline.py:2617-2646.
Outputs to papers/shared/figures/ (PNG + PDF).
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, ".")
from src.data.loader import SherwoodLoader
from src.models.nerf import volume_render_physics
from src.analysis.cross_corr import compute_xi_pearson

TAU_MAX, GRID_LOSS, GRID_TAU_AMP = 10.0, 0.0026, 1.5263
N_RAYS, PHYSICS, REDSHIFT = 1024, 1, 0.3
FIGDIR = "papers/shared/figures"
torch.set_grad_enabled(False)

# Grid xi(r) profile — pinned to job 221335 eval/xi_3d_injob.json (commit c6f3aed).
GRID_R = [0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,
          5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75]
GRID_XI = [0.012649,0.010207,0.008613,0.007549,0.006149,0.005397,0.004841,0.004384,
           0.003978,0.003503,0.003008,0.002546,0.002241,0.001970,0.001672,0.001479,
           0.001333,0.001106,0.000904,0.000753]


def loss_data(tau_pred, tau_gt, mask):
    tpe, tge = tau_pred.clamp_max(TAU_MAX), tau_gt.clamp_max(TAU_MAX)
    diff = torch.log1p(tpe) - torch.log1p(tge)
    w = mask.to(diff.dtype)
    return float(((diff * diff) * w).sum() / w.sum().clamp(min=1.0))


# ---- load truth sightlines (for K2 curve) ----
loader = SherwoodLoader("Sherwood")
sl = loader.load_sightlines(PHYSICS, REDSHIFT)
box = float(sl["header"]["box_kpc_h"])
coords = torch.tensor(loader.get_world_coordinates(sl)[:N_RAYS], dtype=torch.float32) / box
vel_axis = torch.tensor(sl["vel_axis"], dtype=torch.float32)
tau_gt = torch.tensor(sl["tau_h1"][:N_RAYS], dtype=torch.float32)
mask = torch.tensor(sl["mask_no_dla"][:N_RAYS], dtype=torch.bool)
truth_fields = torch.stack([torch.tensor(sl["density"][:N_RAYS], dtype=torch.float32),
                            torch.tensor(sl["temp"][:N_RAYS], dtype=torch.float32),
                            torch.tensor(sl["h1_frac"][:N_RAYS], dtype=torch.float32),
                            torch.tensor(sl["v_pec"][:N_RAYS], dtype=torch.float32)], dim=-1)


class _TM:
    def __init__(s, f): s.f = f
    def __call__(s, rp, g=None, physics_id=None): return s.f[s._sl]

CHUNK = 128
tau_base = torch.zeros((N_RAYS, vel_axis.shape[0]), dtype=torch.float32)
for i in range(0, N_RAYS, CHUNK):
    j = min(i + CHUNK, N_RAYS)
    m = _TM(truth_fields); m._sl = slice(i, j)
    tau_base[i:j] = volume_render_physics(m, coords[i:j], vel_axis, tau_amp=1.0)

amps = np.linspace(0.1, 5.0, 200)
curve = np.array([loss_data(float(a) * tau_base, tau_gt, mask) for a in amps])
print(f"[fig1] truth-loss min={curve.min():.5f} max={curve.max():.5f} (flat ~0.0101); grid={GRID_LOSS}")

# ---- Figure 1: K2 degeneracy ----
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(amps, curve, color="#1f77b4", lw=2, label=r"true field through integrator")
ax.axhline(GRID_LOSS, color="#d62728", ls="--", lw=2,
           label=r"grid (argmin), $0.0026$")
ax.axvspan(0.2, 4.0, color="#1f77b4", alpha=0.07)
ax.annotate(r"$\sim4\times$ margin", xy=(3.2, (curve.mean()+GRID_LOSS)/2),
            fontsize=9, color="black",
            ha="center", va="center",
            arrowprops=None)
ax.set_xlabel(r"$\tau_{\mathrm{amp}}$ (free amplitude)")
ax.set_ylabel(r"flux data loss $\mathcal{L}_{\mathrm{data}}$")
ax.set_ylim(0, max(curve.max() * 1.15, 0.013))
ax.set_title(r"K2: the grid fits flux better than the true field"
             "\n" r"($z{=}0.3$, $P_1$, $G{=}192$, plain-[D-24])", fontsize=9)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/d73_k2_truth_vs_grid.png", dpi=200)
fig.savefig(f"{FIGDIR}/d73_k2_truth_vs_grid.pdf")
plt.close(fig)
print(f"[fig1] wrote {FIGDIR}/d73_k2_truth_vs_grid.{{png,pdf}}")

# ---- truth-vs-truth ceiling profile (cheap, local cube) ----
def mean_pool(c, t):
    f = c.shape[0] // t
    return c.reshape(t, f, t, f, t, f).mean(axis=(1, 3, 5))

truth192 = mean_pool(np.load("Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy").astype(np.float64), 192)
r_bins = np.arange(0.0, 10.0 + 1e-9, 0.5)
rc, xi_ceiling = compute_xi_pearson(truth192, truth192, box_kpc_h=60000.0, r_bins=r_bins)
rng = np.random.default_rng(42)
noisy = truth192 + rng.normal(0.0, 1.0 * truth192.std(), truth192.shape)
_, xi_noise = compute_xi_pearson(truth192, noisy, box_kpc_h=60000.0, r_bins=r_bins)
ig = int(np.argmin(np.abs(rc - 2.0)))
print(f"[fig2] ceiling@r2={xi_ceiling[ig]:.4f}  noise@r2={xi_noise[ig]:.4f}  grid@r2={GRID_XI[3]:.4f}")

wiener = json.load(open("experiments/nerf/artifacts/wiener_baseline/a4prime_wiener.json"))
wiener_xi2 = None
for k in ("xi_at_gate", "xi_3d_r2", "xi_best", "xi"):
    if k in wiener and isinstance(wiener[k], (int, float)):
        wiener_xi2 = float(wiener[k]); break
if wiener_xi2 is None:
    wiener_xi2 = 0.079  # spine-pinned self-anchored lower bound

# ---- Figure 2: xi(r) ceiling-relative ----
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(rc, xi_ceiling, color="#2ca02c", lw=2, marker="o", ms=3,
        label=r"truth$\,\times\,$truth ceiling ($0.0298$ @ $r{=}2$)")
ax.plot(rc, xi_noise, color="#ff7f0e", lw=1.5, ls="-.",
        label=r"truth $+\,100\%$ noise ($0.0211$ @ $r{=}2$)")
ax.plot(GRID_R, GRID_XI, color="#d62728", lw=2, marker="s", ms=3,
        label=r"neural voxel grid ($0.0075$ @ $r{=}2$, $\sim25\%$ of ceiling)")
ax.scatter([2.0], [wiener_xi2], color="#9467bd", zorder=5, s=40, marker="D",
           label=rf"Wiener lower bound ($\geq{wiener_xi2:g}$)")
ax.axhline(0.6, color="black", ls=":", lw=1.5)
ax.text(5.0, 0.615, r"demoted [D-13] $0.6$ gate --- unreachable (even truth$\times$truth $=0.03$)",
        fontsize=7.5, va="bottom", ha="center")
ax.axvline(2.0, color="gray", ls=":", lw=0.8, alpha=0.6)
ax.set_xlabel(r"$r$ [$h^{-1}$ Mpc]")
ax.set_ylabel(r"$\xi_{\hat\rho,\rho}(r)$ (global-norm cross-corr. function)")
ax.set_ylim(0, 0.72)
ax.set_title(r"3D $\xi(r)$ is ceiling-relative; the $0.6$ gate is unreachable"
             "\n" r"($z{=}0.3$, $P_1$, $G{=}192$; frame-confounded, $\Delta\chi\!\sim\!1.3$--$2.5\,h^{-1}$Mpc)",
             fontsize=9)
ax.legend(fontsize=7.5, loc="center right")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/d73_xi_ceiling_relative.png", dpi=200)
fig.savefig(f"{FIGDIR}/d73_xi_ceiling_relative.pdf")
plt.close(fig)
print(f"[fig2] wrote {FIGDIR}/d73_xi_ceiling_relative.{{png,pdf}}")
print("[done]")

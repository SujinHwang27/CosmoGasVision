#!/usr/bin/env python
"""
[D-73] (1d') verification battery — host-side, no Juno needed.

Settles the defense-panel blockers/caveats on the voxel-grid close-out:
  K2 (DECISIVE): truth-field forward-model loss vs the grid's achieved 0.0026.
                 truth_loss >= 0.0026  -> DEGENERACY (grid at flux optimum; strong result)
                 truth_loss <  0.0026  -> STUCK (grid never reached flux optimum)
  S5: xi estimator self-check (xi(truth,truth) ~= 1 at r=2; perturbed -> xi >> 0.0075).
  S7-dchi: RSD displacement bound v_pec/H(z) vs the r=2 h^-1Mpc gate point.

Loss matched exactly to pipeline.py:2617-2646 (plain [D-24], sat_band_weight=1.0):
  masked, uniform-weight, clamp_max(TAU_MAX=10), mean of (log1p(tau_pred)-log1p(tau_gt))^2.
Each section is independently guarded so one failure does not kill the decisive K2.
"""
import json, sys, traceback
import numpy as np
import torch

sys.path.insert(0, ".")
from src.data.loader import SherwoodLoader
from src.models.nerf import volume_render_physics
from src.analysis.cross_corr import compute_xi_pearson

TAU_MAX = 10.0
GRID_LOSS_DATA = 0.0026      # job 221335 achieved loss_data (step 5000->50000, flat)
GRID_TAU_AMP = 1.5263        # job 221335 final tau_amp (from driver log)
N_RAYS = 1024
PHYSICS, REDSHIFT = 1, 0.3
DATA_ROOT = "Sherwood"
TRUTH_CUBE = "Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy"
OUT_JSON = "experiments/nerf/artifacts/d73_1dprime/verification_battery.json"

torch.set_grad_enabled(False)
results = {"meta": {"grid_loss_data": GRID_LOSS_DATA, "grid_tau_amp": GRID_TAU_AMP,
                    "n_rays": N_RAYS, "physics": PHYSICS, "redshift": REDSHIFT}}


def loss_data(tau_pred, tau_gt, mask):
    """Exact pipeline.py:2617-2646 plain-[D-24] form (sat_band_weight=1.0)."""
    tpe = tau_pred.clamp_max(TAU_MAX)
    tge = tau_gt.clamp_max(TAU_MAX)
    diff = torch.log1p(tpe) - torch.log1p(tge)
    ds = diff * diff
    w = mask.to(ds.dtype)
    return float((ds * w).sum() / w.sum().clamp(min=1.0))


# ---------------------------------------------------------------- load truth sightlines
loader = SherwoodLoader(DATA_ROOT)
sl = loader.load_sightlines(PHYSICS, REDSHIFT)
box_max = float(sl["header"]["box_kpc_h"])
coords_raw = loader.get_world_coordinates(sl)             # (n_los, n_bins, 3) world kpc/h
coords = torch.tensor(coords_raw[:N_RAYS], dtype=torch.float32) / box_max
vel_axis = torch.tensor(sl["vel_axis"], dtype=torch.float32)
tau_gt = torch.tensor(sl["tau_h1"][:N_RAYS], dtype=torch.float32)
mask = torch.tensor(sl["mask_no_dla"][:N_RAYS], dtype=torch.bool)
density = torch.tensor(sl["density"][:N_RAYS], dtype=torch.float32)
temp = torch.tensor(sl["temp"][:N_RAYS], dtype=torch.float32)
h1 = torch.tensor(sl["h1_frac"][:N_RAYS], dtype=torch.float32)
vpec = torch.tensor(sl["v_pec"][:N_RAYS], dtype=torch.float32)
truth_fields = torch.stack([density, temp, h1, vpec], dim=-1)   # (n_rays, n_bins, 4)
print(f"[load] rays={coords.shape[0]} bins={coords.shape[1]} n_obs={vel_axis.shape[0]} "
      f"box={box_max:.0f} kpc/h  <F_gt>={float(torch.exp(-tau_gt).mean()):.4f}", flush=True)


# ================================================================ K2 (DECISIVE)
try:
    class _TruthModel:
        """A 'model' whose forward returns the truth fields for the given ray slice."""
        def __init__(self, fields): self.f = fields
        def __call__(self, ray_points, g=None, physics_id=None):
            return self.f[self._sl]
        def eval(self): return self

    # tau scales linearly with tau_amp (nerf.py:362 `tau = tau * tau_amp`), so render once at 1.0.
    CHUNK = 128
    tau_base = torch.zeros((N_RAYS, vel_axis.shape[0]), dtype=torch.float32)
    for i in range(0, N_RAYS, CHUNK):
        j = min(i + CHUNK, N_RAYS)
        m = _TruthModel(truth_fields); m._sl = slice(i, j)
        tau_base[i:j] = volume_render_physics(m, coords[i:j], vel_axis, tau_amp=1.0)
    # minimize loss over tau_amp (1D); also report amp=1 and the grid's final amp.
    amps = np.concatenate([np.linspace(0.2, 4.0, 96), [1.0, GRID_TAU_AMP]])
    losses = {float(a): loss_data(a * tau_base, tau_gt, mask) for a in amps}
    best_amp = min(losses, key=losses.get)
    k2 = {
        "truth_loss_best": losses[best_amp], "best_tau_amp": best_amp,
        "truth_loss_at_amp1": losses[1.0],
        "truth_loss_at_grid_amp": losses[float(GRID_TAU_AMP)],
        "grid_loss_data": GRID_LOSS_DATA,
        "verdict": ("DEGENERACY (grid reached flux optimum; wrong 3D structure is "
                    "genuinely flux-equivalent)" if losses[best_amp] >= GRID_LOSS_DATA
                    else "STUCK (truth field drives loss below grid's; grid did not "
                         "reach the flux optimum)"),
        "margin_truth_minus_grid": losses[best_amp] - GRID_LOSS_DATA,
        "variant": "(a) truth sightline fields at exact ray points (upper-bound fidelity)",
    }
    results["K2"] = k2
    print(f"[K2] truth_loss(best amp={best_amp:.3f})={losses[best_amp]:.6f}  "
          f"amp1={losses[1.0]:.6f}  grid_amp={losses[float(GRID_TAU_AMP)]:.6f}  "
          f"vs grid 0.0026 -> {k2['verdict'].split('(')[0].strip()}", flush=True)
except Exception:
    results["K2"] = {"error": traceback.format_exc()}
    print("[K2] FAILED:\n" + results["K2"]["error"], flush=True)


# ================================================================ S5 (xi self-check)
try:
    def mean_pool(cube, target):
        f = cube.shape[0] // target
        return cube.reshape(target, f, target, f, target, f).mean(axis=(1, 3, 5))

    truth768 = np.load(TRUTH_CUBE).astype(np.float64)
    truth192 = mean_pool(truth768, 192)
    r_bins = np.arange(0.0, 10.0 + 1e-9, 0.5)          # edges -> centers 0.25..9.75 (consumer-matched)
    rc, xi_self = compute_xi_pearson(truth192, truth192, box_kpc_h=60000.0, r_bins=r_bins)
    ig = int(np.argmin(np.abs(rc - 2.0)))
    rng = np.random.default_rng(42)
    std_t = truth192.std()
    pert = {}
    for snr in (0.25, 0.5, 1.0):
        noisy = truth192 + rng.normal(0.0, snr * std_t, truth192.shape)
        _, xi_p = compute_xi_pearson(truth192, noisy, box_kpc_h=60000.0, r_bins=r_bins)
        pert[f"noise_std_{snr}x"] = float(xi_p[ig])
    results["S5"] = {
        "xi_truth_vs_truth_at_r2": float(xi_self[ig]),
        "xi_perturbed_at_r2": pert,
        "grid_xi_at_r2": 0.0075,
        "note": "xi(truth,truth)~=1 validates estimator; perturbed >> 0.0075 shows "
                "0.0075 is not the estimator floor for a 192^3 field.",
    }
    print(f"[S5] xi(truth,truth)@r2={xi_self[ig]:.4f}  perturbed={pert}", flush=True)
except Exception:
    results["S5"] = {"error": traceback.format_exc()}
    print("[S5] FAILED:\n" + results["S5"]["error"], flush=True)


# ================================================================ S7-dchi (RSD bound)
try:
    hdr = sl["header"]
    def gh(*names, default=None):
        for n in names:
            if n in hdr: return float(hdr[n])
        return default
    Om = gh("omega_m", "omega_matter", "Om", default=0.308)
    OL = gh("omega_l", "omega_lambda", "OL", default=0.692)
    h100 = gh("h100", "hubble", "h", default=0.678)
    Hz = 100.0 * np.sqrt(Om * (1 + REDSHIFT) ** 3 + OL)    # km/s / (Mpc/h)  [h-units: 100*E(z)]
    vp = sl["v_pec"][:N_RAYS]
    vrms = float(np.sqrt(np.mean(vp ** 2)))
    vp95 = float(np.percentile(np.abs(vp), 95))
    results["S7_dchi"] = {
        "H_z_kms_per_mpc_h": Hz, "omega_m": Om, "omega_l": OL, "h100": h100,
        "vpec_rms_kms": vrms, "vpec_abs_p95_kms": vp95,
        "dchi_rms_mpc_h": vrms / Hz, "dchi_p95_mpc_h": vp95 / Hz,
        "gate_r_mpc_h": 2.0,
        "note": "RSD displacement vs the r=2 gate point; cubes compared in real space "
                "while supervision is redshift-space flux (frame mismatch, S7).",
    }
    print(f"[S7] H(z)={Hz:.1f} km/s/(Mpc/h)  vpec_rms={vrms:.1f}  "
          f"dchi_rms={vrms/Hz:.3f}  dchi_p95={vp95/Hz:.3f} Mpc/h  (gate r=2)", flush=True)
except Exception:
    results["S7_dchi"] = {"error": traceback.format_exc()}
    print("[S7] FAILED:\n" + results["S7_dchi"]["error"], flush=True)


import os
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\n[done] wrote {OUT_JSON}", flush=True)

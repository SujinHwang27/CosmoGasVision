#!/usr/bin/env python
"""[D-73] am-9 P_F sharpener (PI non-gating; retires defense-panel Item 1a).

Computes the grid's actual [D-13] P_F gate |ΔP_F/P_F| on the step_050000.pt
checkpoint, to fill the §5 "P_F (grid)" cell (currently UNKNOWN) and to let the
pre-registered Branch-A "PASS-trainability-but-P_F-fails" disjunct fire verbatim
rather than via the var_pf proxy.

Convention matched to the eval-side gate (diag_pf_per_bin.py + src.analysis.p_flux):
ensemble-mean P_F over sightlines, per-bin |ΔP_F/P_F| averaged over the inertial
band k∥∈[10^-2.5,10^-1.5] s/km; PASS if < 0.10. (MLP pub-t1 baseline was 0.4155.)
"""
import json, math, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from src.data.loader import SherwoodLoader
from src.models.nerf import volume_render_physics
from src.models.voxel_grid_field import VoxelGridField
from src.analysis.p_flux import compute_p_flux

CKPT = "cloud_runs/d73-1dprime-voxel192-P1-z0.3-c6f3aed-20260618-000035-8b4e90/checkpoints/step_050000.pt"
N_RAYS, PHYSICS, REDSHIFT, G = 1024, 1, 0.3, 192
K_LO, K_HI, GATE = 10 ** -2.5, 10 ** -1.5, 0.10
OUT = "experiments/nerf/artifacts/d73_1dprime/pf_sharpener.json"
torch.set_grad_enabled(False)

# ---- truth sightlines ----
loader = SherwoodLoader("Sherwood")
sl = loader.load_sightlines(PHYSICS, REDSHIFT)
box = float(sl["header"]["box_kpc_h"])
coords = torch.tensor(loader.get_world_coordinates(sl)[:N_RAYS], dtype=torch.float32) / box
vel_axis = torch.tensor(sl["vel_axis"], dtype=torch.float32)
vel_np = sl["vel_axis"].astype(np.float64)
tau_gt = torch.tensor(sl["tau_h1"][:N_RAYS], dtype=torch.float32)
F_truth = torch.exp(-tau_gt).cpu().numpy().astype(np.float64)

# ---- grid checkpoint ----
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = VoxelGridField(grid_size=G, density_head="softplus")
model.load_state_dict(ck["model_state"])
model.eval()
tau_amp = math.exp(float(ck["log_tau_amp"]))
print(f"[pf] loaded grid G={G}, tau_amp=exp({ck['log_tau_amp']:.5f})={tau_amp:.4f}, step={ck['step']}")

# ---- grid -> flux (microbatched) ----
CHUNK = 128
tau_pred = torch.zeros((N_RAYS, vel_axis.shape[0]), dtype=torch.float32)
for i in range(0, N_RAYS, CHUNK):
    j = min(i + CHUNK, N_RAYS)
    tau_pred[i:j] = volume_render_physics(model, coords[i:j], vel_axis, tau_amp=tau_amp)
F_pred = torch.exp(-tau_pred).cpu().numpy().astype(np.float64)
print(f"[pf] <F_pred>={F_pred.mean():.4f}  <F_truth>={F_truth.mean():.4f}")

# ---- ensemble P_F + band residual (eval convention) ----
k, pf_pred = compute_p_flux(F_pred, vel_np)
_, pf_tru = compute_p_flux(F_truth, vel_np)
in_band = (k >= K_LO) & (k <= K_HI)
rel = (pf_pred - pf_tru) / pf_tru
band_resid = float(np.nanmean(np.abs(rel[in_band])))
band_ratio = float(np.nanmean((pf_pred / pf_tru)[in_band]))
passed = bool(band_resid < GATE)

res = {
    "checkpoint": CKPT, "tau_amp": tau_amp, "step": int(ck["step"]),
    "n_rays": N_RAYS, "physics": PHYSICS, "redshift": REDSHIFT,
    "k_band_s_per_km": [K_LO, K_HI], "n_kbins_in_band": int(in_band.sum()),
    "mean_abs_rel_diff_in_band": band_resid,
    "band_ratio_mean": band_ratio,
    "gate": GATE, "PASS": passed,
    "mlp_pubt1_baseline": 0.4155,
    "F_pred_mean": float(F_pred.mean()), "F_truth_mean": float(F_truth.mean()),
}
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"[pf] |ΔP_F/P_F| band-mean = {band_resid:.4f}  (ratio {band_ratio:.4f})  "
      f"vs {GATE} gate -> {'PASS' if passed else 'FAIL'}")
print(f"[pf] wrote {OUT}")

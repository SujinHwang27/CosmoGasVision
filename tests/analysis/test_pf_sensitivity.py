"""Sensitivity-floor checks for compute_PF_1d (success-check item 4).

These are the two qualitative properties the PI will spot-check:

  (A) P_F(k) computed on a *random* MLP's predicted tau differs from
      the truth tau by >= 50% in the [D-13] band -- i.e., the metric
      is sensitive to a wrong reconstruction.

  (B) P_F(k) computed truth-vs-truth has < 1% residual -- i.e., the
      metric is stable against the trivial identity comparison.

Both run on a small slice of Sherwood Physics-1 z=0.3 sightlines (or
synthetic data if the simulation files are not present).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.analysis.flux_power import compute_PF_1d
from src.models.nerf import IGMNeRF, volume_render_physics


_HAS_SHERWOOD = os.path.exists(
    os.path.join("Sherwood", "Physics1_nofeedback")
)


def _band_residual(P_a: np.ndarray, P_b: np.ndarray, k: np.ndarray) -> float:
    band = (k >= 10 ** -2.5) & (k <= 10 ** -1.5) & np.isfinite(P_a) & np.isfinite(P_b)
    return float(np.nanmean(np.abs(P_a[band] - P_b[band]) / P_b[band]))


def test_pf_truth_vs_truth_is_stable():
    """truth-vs-truth residual must be ~0 (numerical floor)."""
    if not _HAS_SHERWOOD:
        pytest.skip("Sherwood/ not present; truth-vs-truth check needs real tau")
    from src.data.loader import SherwoodLoader

    sl = SherwoodLoader("Sherwood").load_sightlines(1, 0.3)
    tau = np.asarray(sl["tau_h1"][:200], dtype=np.float64)
    vel = np.asarray(sl["vel_axis"], dtype=np.float64)
    k_a, P_a = compute_PF_1d(tau, vel)
    k_b, P_b = compute_PF_1d(tau, vel)
    resid = _band_residual(P_a, P_b, k_a)
    print(f"\n[truth-vs-truth] band-mean |dP/P| = {resid:.3e}")
    assert resid < 0.01, f"truth-vs-truth residual = {resid:.3e}, expected < 1%"


def test_pf_random_network_is_far_from_truth():
    """A random-init network's tau must yield >= 50% band residual."""
    if not _HAS_SHERWOOD:
        pytest.skip("Sherwood/ not present; need real coords/vel for tau render")
    from src.data.loader import SherwoodLoader

    sherwood = SherwoodLoader("Sherwood")
    sl = sherwood.load_sightlines(1, 0.3)
    box = float(sl["header"]["box_kpc_h"])
    coords_w = sherwood.get_world_coordinates(sl)
    n = 8  # small for runtime + memory; PF estimator is fine at small n_rays
    coords = torch.tensor(coords_w[:n] / box, dtype=torch.float32)
    vel_axis = np.asarray(sl["vel_axis"], dtype=np.float64)
    vel_t = torch.tensor(vel_axis, dtype=torch.float32)
    tau_truth = np.asarray(sl["tau_h1"][:n], dtype=np.float64)

    torch.manual_seed(0)
    model = IGMNeRF(hidden_dim=256, num_layers=8, L=10).eval()
    with torch.no_grad():
        tau_pred = volume_render_physics(model, coords, vel_t).cpu().numpy()

    k_t, P_t = compute_PF_1d(tau_truth, vel_axis)
    k_p, P_p = compute_PF_1d(tau_pred, vel_axis)
    resid = _band_residual(P_p, P_t, k_t)
    print(f"\n[random-net-vs-truth] band-mean |dP/P| = {resid:.3e}")
    assert resid >= 0.5, (
        f"random-net residual = {resid:.3e}, expected >= 50% — metric not sensitive"
    )

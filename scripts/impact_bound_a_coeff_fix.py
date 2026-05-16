"""Empirical impact-bound: how much do the τ predictions / mean-flux / P_F change
between the buggy `a = 4.7e-4 / b` and the corrected `a = 6.063e-3 / b`?

This is forward-model-only (no MLP) — we feed simulator-truth-like (ρ, T, X_HI, v_pec)
into a standalone re-implementation of the integrator with parameterized `a_coeff`,
then compare τ statistics between the two conventions.

Discharges the "impact-bound Stage 2b/3 results" obligation from the [D-57] fix
([COSMO] flag from [D-54] gate-4a panel).
"""
from __future__ import annotations
import math
import numpy as np
import torch

torch.manual_seed(20260516)
np.random.seed(20260516)

# ------------------------------------------------------------------------------
# Re-implementation of the Voigt forward model with parameterized a_coeff.
# Mirrors src/models/nerf.py volume_render_physics exactly except for taking
# (rho, temp, h1_frac, vpec) directly (no MLP) and exposing `a_coeff`.
# ------------------------------------------------------------------------------

import sys
sys.path.insert(0, ".")
from src.models.nerf import tepper_garcia_voigt


def render_tau_with_a_coeff(
    density: torch.Tensor,
    temp: torch.Tensor,
    h1_frac: torch.Tensor,
    vpec: torch.Tensor,
    vel_axis: torch.Tensor,
    a_coeff: float,
    tau_amp: float = 1.0,
    window: int = 64,
) -> torch.Tensor:
    """Standalone integrator. Same logic as nerf.py:220-262 with a_coeff parameterized.

    Args:
        density, temp, h1_frac, vpec: (n_rays, n_src). Simulator-truth fields.
        vel_axis: (n_obs,). Velocity grid in km/s.
        a_coeff: damping-parameter prefactor. Production uses a = a_coeff / b.

    Returns:
        tau: (n_rays, n_obs)
    """
    b = 12.85 * torch.sqrt(temp / 10000.0)
    a = a_coeff / b
    n_hi = density * h1_frac

    n_obs = vel_axis.shape[0]
    n_rays, n_src = density.shape
    device = vel_axis.device
    dtype = density.dtype

    dv_per_bin = (vel_axis[-1] - vel_axis[0]) / (n_obs - 1)
    v_source = vel_axis[None, :] + vpec  # (n_rays, n_src)

    center_idx = ((v_source - vel_axis[0]) / dv_per_bin).long()
    offsets = torch.arange(-window, window + 1, device=device)
    obs_idx = center_idx[..., None] + offsets[None, None, :]
    valid_mask = (obs_idx >= 0) & (obs_idx < n_obs)
    obs_idx_safe = obs_idx.clamp(0, n_obs - 1)

    v_obs_window = vel_axis[obs_idx_safe]
    dv_window = v_obs_window - v_source[..., None]
    x = dv_window / b[..., None]
    H = tepper_garcia_voigt(a[..., None], x)
    H = H * valid_mask.to(dtype)

    sqrt_pi = torch.sqrt(torch.tensor(math.pi, device=device, dtype=dtype))
    contrib = (n_hi[..., None] * H) / (b[..., None] * sqrt_pi)

    tau = torch.zeros(n_rays, n_obs, device=device, dtype=dtype)
    obs_idx_flat = obs_idx_safe.reshape(n_rays, -1)
    contrib_flat = contrib.reshape(n_rays, -1)
    tau.scatter_add_(1, obs_idx_flat, contrib_flat)

    return tau_amp * tau


# ------------------------------------------------------------------------------
# Generate synthetic-but-physical simulator-truth fields representative of
# Sherwood IGM at z=0.3.
# ------------------------------------------------------------------------------

n_rays = 256
n_src = 1024            # source bins per ray (= n_obs; Sherwood-style)
n_obs = n_src

# Velocity grid spanning ~6000 km/s (Sherwood at z=0.3 box scale)
vel_max = 6000.0  # km/s
vel_axis = torch.linspace(0.0, vel_max, n_obs, dtype=torch.float64)

# Log-normal overdensity (Sherwood-like; log10(rho/<rho>) ~ N(0, 0.8))
log10_rho = 0.8 * torch.randn(n_rays, n_src, dtype=torch.float64)
density = 10.0 ** log10_rho  # rho / <rho>

# Temperature: T-rho relation (Lyman-alpha forest at z~0.3)
# log10(T/K) = 4.0 + 0.55 * log10(rho/<rho>) (typical photoionized-IGM scaling)
log10_T = 4.0 + 0.55 * log10_rho
temp = 10.0 ** log10_T

# X_HI from FGPA-ish scaling: n_HI / n_H ~ (rho/<rho>)^a * T^-0.7 (simplified)
# Set typical X_HI ~ 1e-5 at mean density
xhi_baseline = 1.0e-5
h1_frac = xhi_baseline * (density ** 1.6) * (temp / 1e4) ** (-0.7)

# Peculiar velocity ~ Gaussian sigma=50 km/s
vpec = 50.0 * torch.randn(n_rays, n_src, dtype=torch.float64)

print("=" * 72)
print("Impact-bound forward-model comparison: a_coeff = 4.7e-4 (buggy) vs 6.063e-3 (fixed)")
print("=" * 72)
print(f"n_rays = {n_rays}, n_src = {n_src}, vel_max = {vel_max} km/s")
print(f"density: mean={density.mean():.3f}, median={density.median():.3f}, max={density.max():.1f}")
print(f"temp:    mean={temp.mean():.3e}, median={temp.median():.3e} K")
print(f"h1_frac: mean={h1_frac.mean():.3e}, max={h1_frac.max():.3e}")
print(f"vpec:    sigma={vpec.std():.1f} km/s")
print()

# ------------------------------------------------------------------------------
# Compare τ at the two a_coeff values
# ------------------------------------------------------------------------------

# tau_amp is a learned scalar in the production loop. For an impact bound we
# need a representative value. From Stage 2b headline (LEDGER §1), tau_amp
# settled in the ~1.0-1.2 range post-[D-24]. Use 1.0 as canonical.
tau_amp = 1.0

print("Computing τ with BUGGY a_coeff = 4.7e-4 ...")
tau_old = render_tau_with_a_coeff(density, temp, h1_frac, vpec, vel_axis, 4.7e-4, tau_amp=tau_amp)
print("Computing τ with FIXED a_coeff = 6.063e-3 ...")
tau_new = render_tau_with_a_coeff(density, temp, h1_frac, vpec, vel_axis, 6.063e-3, tau_amp=tau_amp)
print()

# ------------------------------------------------------------------------------
# Comparison statistics
# ------------------------------------------------------------------------------

dtau = tau_new - tau_old
abs_dtau = dtau.abs()

# Apply [D-24] tau-cap: cap tau at 10 before computing flux
tau_max = 10.0
tau_old_capped = tau_old.clamp(max=tau_max)
tau_new_capped = tau_new.clamp(max=tau_max)

F_old = torch.exp(-tau_old_capped)
F_new = torch.exp(-tau_new_capped)

print("--- τ statistics ---")
print(f"τ_old:  mean={tau_old.mean():.4e}, median={tau_old.median():.4e}, max={tau_old.max():.3e}")
print(f"τ_new:  mean={tau_new.mean():.4e}, median={tau_new.median():.4e}, max={tau_new.max():.3e}")
print(f"|Δτ|:   max={abs_dtau.max():.4e}, mean={abs_dtau.mean():.4e}, median={abs_dtau.median():.4e}")
print(f"|Δτ| / max(|τ_old|, 1e-9): max_rel={(abs_dtau / tau_old.abs().clamp(min=1e-9)).max():.3e}")
print()

print("--- Mean-flux statistics ---")
print(f"<F_old>: {F_old.mean():.6f}")
print(f"<F_new>: {F_new.mean():.6f}")
print(f"|Δ<F>|:  {(F_new.mean() - F_old.mean()).abs():.4e}")
print(f"|Δ<F>| / <F_old>: {((F_new.mean() - F_old.mean()).abs() / F_old.mean()):.4e}")
print()

# Per-bin RMS flux change
delta_F = F_new - F_old
print(f"--- Per-bin flux delta ---")
print(f"|ΔF|:    max={delta_F.abs().max():.4e}, RMS={(delta_F**2).mean().sqrt():.4e}")
print()

# Fraction of cells saturated under each convention
saturated_old = (tau_old > 1.0).float().mean()
saturated_new = (tau_new > 1.0).float().mean()
print(f"--- Saturated-cell fraction (τ > 1) ---")
print(f"old: {saturated_old:.4%}")
print(f"new: {saturated_new:.4%}")
print()

# 1D flux power spectrum bound: compare P_F at low-k and high-k
# Sample a single ray for illustration
def p_flux(F: torch.Tensor, dv_kms: float):
    """Simple Hann-windowed 1D periodogram for one ray."""
    F = F - F.mean()
    win = torch.hann_window(F.shape[0], dtype=F.dtype)
    Fw = F * win
    P = torch.fft.rfft(Fw).abs() ** 2
    norm = (win ** 2).sum() * dv_kms / F.shape[0]
    P = P * dv_kms ** 2 / norm
    k = torch.fft.rfftfreq(F.shape[0], d=dv_kms) * 2 * math.pi
    return k, P

dv_per_bin = (vel_axis[-1] - vel_axis[0]) / (n_obs - 1)
k_old, P_old = p_flux(F_old[0], dv_per_bin.item())
k_new, P_new = p_flux(F_new[0], dv_per_bin.item())

# Inertial range [10^-2.5, 10^-1.5] s/km per [D-13]
k_mask = (k_old > 10 ** -2.5) & (k_old < 10 ** -1.5)
if k_mask.any():
    rel_pf = ((P_new - P_old).abs() / P_old.clamp(min=1e-30))[k_mask]
    print(f"--- P_F change on ray 0 over inertial range [10^-2.5, 10^-1.5] s/km ---")
    print(f"|ΔP_F / P_F|: max={rel_pf.max():.4e}, mean={rel_pf.mean():.4e}")
    print()
else:
    print("(inertial range not covered by the chosen grid)")
    print()

print("=" * 72)
print("VERDICT")
print("=" * 72)
print(f"Mean-flux relative shift: {((F_new.mean() - F_old.mean()).abs() / F_old.mean()):.2%}")
print(f"Max per-cell flux delta:  {delta_F.abs().max():.4e}")
print(f"For ratio reference: [D-13] mean-flux 5% gate; [D-13] P_F 10% gate.")

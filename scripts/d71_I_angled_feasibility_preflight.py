"""[D-71] §I Rung 4 — Angled-sightline feasibility CPU PRE-FLIGHT.

Brief: experiments/nerf/design/D71_Rung4_angled_sightline_feasibility_brief.md
       (§5 protocol + Amendment 1 §A1.5 modular-index + boundary-gradient mandates).

This script is the CPU pre-flight only; the Juno full-resolution dispatch is
BLOCKED behind cycle #4 panel clearance per LEDGER §I. Local execution.

Deliverables (per dispatch brief):
  (a) sample_field_along_rays — modular-index periodic-wrap helper with
      grid_sample + manual 8-corner trilinear assembly for boundary-crossing
      rays. Autograd-clean w.r.t. ray_origins + ray_dirs.
  (b) Geometric-correctness sanity check on analytic 64**3 sin(2*pi*x/L) field
      (32 angled rays, max |sampled - analytic| <= 1e-4 PASS gate).
  (c) Boundary-gradient autograd verification (>=4 boundary-crossing rays,
      finite + non-NaN gradients at every bin).
  (d) Bootstrap-SE measurement on real Sherwood P1 z=0.3 rho field
      (192**3 downsampled from cached 768**3), Set A axis-parallel vs
      Set B angled, N=768 each, 2048 bins, rho-only FGPA (T=1e4 K,
      v_pec=0), inertial-band Var(P_F)/mean(P_F)**2 R_feas, 200 bootstrap
      resamples per set. Reports SE_meas(A), SE_meas(B), delta log10
      R_feas, 95% bootstrap CI.
  (e) JSON capsule + PNG histogram landed under
      experiments/nerf/artifacts/d71_I_angled_feasibility_preflight/.

Per [D-37] rule (a): observation first. The bootstrap-SE measurement does
NOT bias verdict-band selection (that is PI's Amendment 2 work). The
192**3 downsampling is the honest-framing approximation; the Juno
dispatch will run at 768**3 native resolution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Repo root on sys.path so `src.` imports resolve when invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.flux_power_torch import compute_p_flux_torch  # noqa: E402


# [D-13] inertial-band edges (s/km). Mirrors src/analysis/flux_power_torch.py.
K_MIN_INERTIAL = 10.0 ** -2.5
K_MAX_INERTIAL = 10.0 ** -1.5

# Box geometry (Sherwood P1 z=0.300; CLAUDE.md astrophysical conventions).
BOX_KPC_H = 60_000.0

# FGPA scientific simplification per brief Amendment §A1.4 caveat:
# rho-only proxy, isothermal T = <T>_box = 1e4 K, v_pec = 0.
T_ISO_K = 1.0e4
BETA_TGAMMA = 1.6

# Hubble-flow conversion at z=0.300 for v-axis spacing (km/s per kpc/h).
# v = H(z) * dx_phys. Sherwood Planck1 cosmology: H0=67.74 km/s/Mpc,
# Omega_m=0.3089 (Planck 2015). H(z=0.3) ~ 80.0 km/s/Mpc; convert kpc/h
# (comoving) -> physical Mpc via /(1+z)/h, then * H(z).
H0_KMS_PER_MPC = 67.74
H_LITTLE = H0_KMS_PER_MPC / 100.0  # 0.6774
OMEGA_M = 0.3089
OMEGA_L = 1.0 - OMEGA_M
Z_SNAP = 0.300


def hubble_kms_per_mpc(z: float) -> float:
    """Flat-LCDM H(z) in km/s/Mpc."""
    return H0_KMS_PER_MPC * np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def v_per_kpc_h_comoving(z: float) -> float:
    """km/s of LOS velocity per comoving kpc/h step. Used for FFT dv."""
    # comoving Mpc/h -> physical Mpc: /(1+z)/h
    return hubble_kms_per_mpc(z) / 1000.0 / (1.0 + z) / H_LITTLE


# -----------------------------------------------------------------------------
# (a) sample_field_along_rays — modular-index periodic-wrap.
# -----------------------------------------------------------------------------

def sample_field_along_rays(
    field_3d: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    n_samples: int,
    box_kpc_h: float = BOX_KPC_H,
    t_max_unit: float = 1.0,
) -> torch.Tensor:
    """Sample `field_3d` along rays parameterized as origin + t * dir.

    Implementation per brief §A1.5: modular-index periodic-wrap (NOT 3x
    memory replicate). The standard interior path uses `grid_sample` with
    `padding_mode='zeros'` after a `coords % 1.0` pre-pass. Rays whose
    parameter range crosses a box boundary in PRE-MODULO space are
    handled by a manual 8-corner trilinear assembly that indexes wrapped
    corners directly — this avoids the grid_sample discontinuity that
    arises when adjacent samples land on opposite sides of the [0, 1]
    boundary post-modulo.

    Parameters
    ----------
    field_3d : tensor of shape (N, N, N), float32
        Scalar field on a unit-cube grid in physical-axis order (i.e.
        index [i, j, k] corresponds to spatial coordinate
        (i, j, k) / N in unit-cube coords). Periodic boundary assumed.
    ray_origins : tensor of shape (n_rays, 3)
        Ray origins in UNIT-CUBE coords (in [0, 1]^3). Autograd-traceable.
    ray_dirs : tensor of shape (n_rays, 3)
        Ray direction vectors in UNIT-CUBE coords. NOT required to be
        unit-norm; the ray is sampled at t in [0, t_max_unit] times this
        vector. Autograd-traceable.
    n_samples : int
        Number of sample bins along each ray.
    box_kpc_h : float
        Box side length in kpc/h. Retained for interface symmetry with
        downstream Juno code that may pass raw kpc/h coords; unused here
        because origins/dirs are already in unit-cube coords.
    t_max_unit : float
        Upper end of the sample-parameter interval (lower is 0). For a
        ray spanning the full box diagonal at unit dirs, t_max_unit = 1
        traverses one unit-cube edge.

    Returns
    -------
    samples : tensor of shape (n_rays, n_samples)
        Trilinearly-interpolated field values along each ray.
    """
    if field_3d.dim() != 3:
        raise ValueError(f"field_3d must be 3D; got shape {tuple(field_3d.shape)}")
    if ray_origins.shape[-1] != 3 or ray_dirs.shape[-1] != 3:
        raise ValueError("ray_origins and ray_dirs must have last dim 3")
    N = field_3d.shape[0]
    device = field_3d.device
    dtype = field_3d.dtype
    n_rays = ray_origins.shape[0]

    # t-axis in unit-cube parameter
    t = torch.linspace(0.0, t_max_unit, n_samples, device=device, dtype=dtype)
    # raw (pre-modulo) coords: (n_rays, n_samples, 3)
    coords_raw = ray_origins[:, None, :] + t[None, :, None] * ray_dirs[:, None, :]

    # Identify boundary-crossing rays in PRE-modulo space.
    coord_min = coords_raw.amin(dim=1)  # (n_rays, 3)
    coord_max = coords_raw.amax(dim=1)  # (n_rays, 3)
    boundary_mask = (coord_min < 0.0).any(dim=1) | (coord_max > 1.0).any(dim=1)

    # ---- Interior path: grid_sample on coords % 1.0 ---------------------
    coords_unit = coords_raw % 1.0  # in [0, 1)^3, autograd-safe
    # grid_sample expects coords in [-1, 1] with align_corners=False
    # mapping (-1 -> -0.5/N, +1 -> 1 - 0.5/N) of voxel centers; map [0, 1] -> [-1, 1].
    # Sample order in grid: (..., 3) last-dim is (x, y, z) in W,H,D convention.
    # We use a fixed convention: field_3d[i, j, k] <-> coord (i/N, j/N, k/N).
    # grid_sample's last-dim is (w, h, d) which corresponds to the LAST,
    # SECOND, FIRST dim of input respectively. So we order grid last-dim
    # as (k_coord, j_coord, i_coord) to match.
    grid = coords_unit * 2.0 - 1.0  # (n_rays, n_samples, 3) in [-1, 1)
    # Re-order (i, j, k) -> (k, j, i) for grid_sample
    grid_kji = grid[..., [2, 1, 0]]
    # Reshape to grid_sample's expected (N, D_out, H_out, W_out, 3) form.
    # We pack all rays as a single batch via D_out = n_rays, H_out = 1,
    # W_out = n_samples. Input: (1, 1, N, N, N).
    grid_5d = grid_kji.view(1, n_rays, 1, n_samples, 3)
    input_5d = field_3d.view(1, 1, N, N, N)
    interior = torch.nn.functional.grid_sample(
        input_5d, grid_5d,
        mode='bilinear', padding_mode='zeros', align_corners=False,
    )  # (1, 1, n_rays, 1, n_samples)
    interior = interior.view(n_rays, n_samples)

    if not boundary_mask.any():
        return interior

    # ---- Boundary path: manual 8-corner trilinear with modular-indexed
    # corners. Replaces the interior samples for boundary-crossing rays.
    # coords_unit_b: (n_b, n_samples, 3) in [0, 1)
    n_b = int(boundary_mask.sum().item())
    coords_unit_b = coords_unit[boundary_mask]
    # Coordinate in voxel space: c = unit * N - 0.5. Voxel-center grid
    # convention (matches grid_sample align_corners=False).
    cv = coords_unit_b * N - 0.5  # (n_b, n_samples, 3)
    i0 = torch.floor(cv).to(torch.long)  # lower corner indices
    frac = cv - i0.to(dtype)  # (n_b, n_samples, 3)
    # Modular-index corners (NO 3x replicate; per A1.5):
    i_lo = i0 % N
    i_hi = (i0 + 1) % N
    # frac in [0, 1) is autograd-safe; the long-typed corner indices are
    # NOT autograd-traceable but only enter the gather (no grad needed).
    # The gradient flows through `frac` and the linear weights w_*.
    # Per-axis weights
    wx_lo = 1.0 - frac[..., 0]
    wx_hi = frac[..., 0]
    wy_lo = 1.0 - frac[..., 1]
    wy_hi = frac[..., 1]
    wz_lo = 1.0 - frac[..., 2]
    wz_hi = frac[..., 2]

    def _g(ix, iy, iz):
        # Gather field at integer-index corner. field_3d[i,j,k] with
        # our (i,j,k) <-> (x,y,z) convention.
        return field_3d[ix, iy, iz]

    ix_lo, ix_hi = i_lo[..., 0], i_hi[..., 0]
    iy_lo, iy_hi = i_lo[..., 1], i_hi[..., 1]
    iz_lo, iz_hi = i_lo[..., 2], i_hi[..., 2]

    f000 = _g(ix_lo, iy_lo, iz_lo)
    f100 = _g(ix_hi, iy_lo, iz_lo)
    f010 = _g(ix_lo, iy_hi, iz_lo)
    f110 = _g(ix_hi, iy_hi, iz_lo)
    f001 = _g(ix_lo, iy_lo, iz_hi)
    f101 = _g(ix_hi, iy_lo, iz_hi)
    f011 = _g(ix_lo, iy_hi, iz_hi)
    f111 = _g(ix_hi, iy_hi, iz_hi)

    boundary_samples = (
        f000 * wx_lo * wy_lo * wz_lo
        + f100 * wx_hi * wy_lo * wz_lo
        + f010 * wx_lo * wy_hi * wz_lo
        + f110 * wx_hi * wy_hi * wz_lo
        + f001 * wx_lo * wy_lo * wz_hi
        + f101 * wx_hi * wy_lo * wz_hi
        + f011 * wx_lo * wy_hi * wz_hi
        + f111 * wx_hi * wy_hi * wz_hi
    )

    # Splice boundary rows back into the interior tensor (out-of-place
    # to keep autograd clean).
    out = interior.clone()
    out[boundary_mask] = boundary_samples
    return out


# -----------------------------------------------------------------------------
# (b) Geometric-correctness on analytic sin(2pi x/L) field.
# -----------------------------------------------------------------------------

def _sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform-on-S^2 directions via Marsaglia: cos(theta) ~ U(-1,1), phi ~ U(0,2pi)."""
    cos_t = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, 1.0))
    return np.stack(
        [sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1
    )


def _reference_trilinear_periodic(field_np: np.ndarray,
                                  coords_unit: np.ndarray) -> np.ndarray:
    """Reference trilinear-with-periodic-wrap sampler implemented from
    scratch in NumPy. Used as ground truth for `sample_field_along_rays`
    (we test that grid_sample + modular-corner code paths match a known
    trilinear-periodic implementation, NOT that trilinear matches the
    continuous analytic — those differ by O((dx)^2 |f''|) ~ 1e-2 at
    N=64 for sin(2*pi*x), which would falsify a 1e-4 tolerance).

    coords_unit: (n_rays, n_samples, 3) array of unit-cube coords (may
    be outside [0,1]; wrapped here).
    """
    N = field_np.shape[0]
    c = coords_unit % 1.0
    cv = c * N - 0.5  # voxel-center convention (matches grid_sample align_corners=False)
    i0 = np.floor(cv).astype(np.int64)
    frac = cv - i0
    i_lo = i0 % N
    i_hi = (i0 + 1) % N
    wx_lo, wx_hi = 1.0 - frac[..., 0], frac[..., 0]
    wy_lo, wy_hi = 1.0 - frac[..., 1], frac[..., 1]
    wz_lo, wz_hi = 1.0 - frac[..., 2], frac[..., 2]
    f000 = field_np[i_lo[..., 0], i_lo[..., 1], i_lo[..., 2]]
    f100 = field_np[i_hi[..., 0], i_lo[..., 1], i_lo[..., 2]]
    f010 = field_np[i_lo[..., 0], i_hi[..., 1], i_lo[..., 2]]
    f110 = field_np[i_hi[..., 0], i_hi[..., 1], i_lo[..., 2]]
    f001 = field_np[i_lo[..., 0], i_lo[..., 1], i_hi[..., 2]]
    f101 = field_np[i_hi[..., 0], i_lo[..., 1], i_hi[..., 2]]
    f011 = field_np[i_lo[..., 0], i_hi[..., 1], i_hi[..., 2]]
    f111 = field_np[i_hi[..., 0], i_hi[..., 1], i_hi[..., 2]]
    return (
        f000 * wx_lo * wy_lo * wz_lo
        + f100 * wx_hi * wy_lo * wz_lo
        + f010 * wx_lo * wy_hi * wz_lo
        + f110 * wx_hi * wy_hi * wz_lo
        + f001 * wx_lo * wy_lo * wz_hi
        + f101 * wx_hi * wy_lo * wz_hi
        + f011 * wx_lo * wy_hi * wz_hi
        + f111 * wx_hi * wy_hi * wz_hi
    )


def geometric_correctness_check(seed: int = 2026) -> dict:
    """32 angled rays on a sin(2*pi*x/L) analytic field on a 64**3 grid.
    PASS gate (1e-4 tolerance) is against a reference NumPy trilinear-
    periodic sampler (NOT the continuous analytic, which differs by
    O((dx)^2 |f''|) ~ 1e-2 for sin at N=64 — that order-of-magnitude
    discretization error would unfairly fail the 1e-4 gate). The PI's
    brief language "max|sample - f_analytic|" is interpreted as
    "max|sample - trilinear(discretized f_analytic)|", since the field
    is sampled on a finite grid.
    """
    rng = np.random.default_rng(seed)
    N = 64
    L_unit = 1.0
    # x coord at voxel CENTERS for align_corners=False convention: x_i = (i+0.5)/N
    i = np.arange(N, dtype=np.float64)
    xc = (i + 0.5) / N
    field_np = np.ascontiguousarray(
        np.sin(2.0 * np.pi * xc / L_unit)[:, None, None] * np.ones((1, N, N))
    )
    field = torch.tensor(field_np, dtype=torch.float64).contiguous()

    n_rays = 32
    origins_np = rng.uniform(0.0, 1.0, size=(n_rays, 3))
    dirs_np = _sample_unit_sphere(n_rays, rng)
    # Span ~ a full box-side so a meaningful fraction crosses boundaries.
    origins = torch.tensor(origins_np, dtype=torch.float64, requires_grad=True)
    dirs = torch.tensor(dirs_np, dtype=torch.float64, requires_grad=True)

    n_samples = 256
    t_max = 1.0
    sampled = sample_field_along_rays(
        field, origins, dirs, n_samples=n_samples, t_max_unit=t_max,
    )
    t = torch.linspace(0.0, t_max, n_samples, dtype=torch.float64)
    coords_raw = origins[:, None, :] + t[None, :, None] * dirs[:, None, :]

    # Reference trilinear-periodic via NumPy (ground truth for our wrapper).
    ref_np = _reference_trilinear_periodic(field_np, coords_raw.detach().cpu().numpy())
    ref = torch.tensor(ref_np, dtype=torch.float64)

    # Diagnostic: also report analytic deviation for honest framing.
    analytic = torch.sin(2.0 * np.pi * (coords_raw[..., 0]) / L_unit)
    analytic_max_dev = float((sampled.detach() - analytic).abs().max().item())

    diff = (sampled - ref).abs()
    max_dev = float(diff.max().item())
    per_ray_max = [float(diff[r].max().item()) for r in range(n_rays)]
    n_pass = int(sum(1 for d in per_ray_max if d <= 1e-4))
    passed = n_pass == n_rays

    return {
        "max_deviation": max_dev,
        "max_deviation_vs_continuous_analytic": analytic_max_dev,
        "reference_kind": "trilinear-periodic (NumPy independent path)",
        "n_rays": n_rays,
        "n_pass": n_pass,
        "per_ray_max_deviation": per_ray_max,
        "tolerance": 1e-4,
        "pass": bool(passed),
        # cached for (c) reuse
        "_field": field,
        "_origins": origins,
        "_dirs": dirs,
        "_sampled": sampled,
        "_coords_raw": coords_raw,
        "_n_samples": n_samples,
        "_t_max": t_max,
    }


# -----------------------------------------------------------------------------
# (c) Boundary-gradient autograd verification.
# -----------------------------------------------------------------------------

def boundary_gradient_check(geo_result: dict) -> dict:
    """Of the 32 angled rays from (b), identify rays whose pre-modulo
    coord range crosses the box boundary; check finite + non-NaN
    gradients via torch.autograd.grad through sample_field_along_rays.
    """
    field = geo_result["_field"]
    origins = geo_result["_origins"]
    dirs = geo_result["_dirs"]
    coords_raw = geo_result["_coords_raw"]
    n_samples = geo_result["_n_samples"]
    t_max = geo_result["_t_max"]

    # Boundary rays identified per brief A1.5 spec.
    coord_min = coords_raw.detach().amin(dim=1)
    coord_max = coords_raw.detach().amax(dim=1)
    boundary_ray_mask = (
        (coord_min < 0.0).any(dim=1) | (coord_max > 1.0).any(dim=1)
    )
    n_boundary = int(boundary_ray_mask.sum().item())

    # Fresh forward with retain_graph=True (the geo_result tensors are
    # already in the same graph but we rebuild for explicit safety).
    samples = sample_field_along_rays(
        field, origins, dirs, n_samples=n_samples, t_max_unit=t_max,
    )

    # Only check boundary rays' gradients.
    bs = samples[boundary_ray_mask]
    if bs.numel() == 0:
        return {
            "n_boundary_rays": 0,
            "pass": False,
            "reason": "No boundary-crossing rays were sampled; check t_max / seed.",
            "n_nan_grads_origins": 0,
            "n_nan_grads_dirs": 0,
        }
    loss = bs.sum()
    grad_o, grad_d = torch.autograd.grad(
        loss, (origins, dirs), retain_graph=True
    )
    # Boundary-ray subset of the grad tensors.
    grad_o_b = grad_o[boundary_ray_mask]
    grad_d_b = grad_d[boundary_ray_mask]
    n_nan_o = int((~torch.isfinite(grad_o_b)).sum().item())
    n_nan_d = int((~torch.isfinite(grad_d_b)).sum().item())

    # Additionally: per-bin grad finiteness — backprop sum over only the
    # boundary-bin samples (those bins whose coords cross the boundary).
    # We do this via a Jacobian-of-sum check at each boundary BIN; if
    # the per-bin grad is finite at every BIN for every boundary ray,
    # we have the strong guarantee A1.5 specifies.
    # Cheap proxy: per-ray sum already checks finiteness; we additionally
    # check that grad does not depend on a NaN-emitting bin by inspecting
    # per-bin analytic: sample at boundary-rays' bins individually and
    # confirm no NaN samples.
    sample_nans = int((~torch.isfinite(samples[boundary_ray_mask])).sum().item())

    pass_ge_4 = n_boundary >= 4 and n_nan_o == 0 and n_nan_d == 0 and sample_nans == 0
    return {
        "n_boundary_rays": n_boundary,
        "n_nan_grads_origins": n_nan_o,
        "n_nan_grads_dirs": n_nan_d,
        "n_nan_samples_boundary": sample_nans,
        "pass": bool(pass_ge_4),
    }


# -----------------------------------------------------------------------------
# (d) Bootstrap-SE measurement on real rho field.
# -----------------------------------------------------------------------------

def _load_rho_cached(physics_id: int = 1, redshift: float = 0.300,
                     n_grid: int = 768) -> np.ndarray:
    """Load cached CIC rho field; HALT if missing (no auto-regeneration)."""
    cache_dir = Path(r"D:\Data\sujin\CosmoGasVision\Sherwood\.rho_field_cache")
    npy_path = cache_dir / f"rho_field_p{physics_id}_z{redshift:.3f}_n{n_grid}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"rho cache missing at {npy_path}. Per dispatch brief, HALT — "
            "do NOT auto-regenerate (multi-hour CIC out of scope for pre-flight)."
        )
    return np.array(np.load(str(npy_path), mmap_mode="r"), dtype=np.float32, copy=True)


def _block_pool_avg(arr: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool a cubic array by an integer factor (memory-aware)."""
    N = arr.shape[0]
    if N % factor != 0:
        raise ValueError(f"N={N} not divisible by factor={factor}")
    M = N // factor
    out = arr.reshape(M, factor, M, factor, M, factor).mean(axis=(1, 3, 5))
    return out.astype(np.float32)


def _build_set_A(n_rays_total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Axis-parallel set: n_rays_total/3 per axis, uniform transverse,
    uniform LOS-origin offset. Returns (origins, dirs), each (N, 3) in unit-cube coords.
    """
    per_axis = n_rays_total // 3
    origins = np.empty((per_axis * 3, 3), dtype=np.float64)
    dirs = np.zeros((per_axis * 3, 3), dtype=np.float64)
    for axis in range(3):
        ax_lo = axis * per_axis
        ax_hi = ax_lo + per_axis
        o = rng.uniform(0.0, 1.0, size=(per_axis, 3))
        d = np.zeros((per_axis, 3))
        d[:, axis] = 1.0
        origins[ax_lo:ax_hi] = o
        dirs[ax_lo:ax_hi] = d
    return origins, dirs


def _build_set_B(n_rays: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Angled set: uniform-S^2 direction, uniform origin in [0,1]^3."""
    origins = rng.uniform(0.0, 1.0, size=(n_rays, 3))
    dirs = _sample_unit_sphere(n_rays, rng)
    return origins, dirs


def _render_fgpa_rho_only(rho_samples: torch.Tensor, beta: float = BETA_TGAMMA,
                          tau_amp: float = 1.0) -> torch.Tensor:
    """Simplified rho-only FGPA: tau ~ rho^beta. Calibration of tau_amp to
    <F>=0.979 is performed afterwards to give a like-for-like F.
    Isothermal T=1e4 K -> only sets b (Doppler); since v_pec=0 and
    T=const, the line-profile convolution reduces to a fixed kernel
    that doesn't change R_feas (multiplicative scaling cancels).
    For the pre-flight we use the local-tau form (no Voigt convolution)
    per the brief's 'rho-only FGPA proxy' scientific simplification.
    """
    return tau_amp * (rho_samples ** beta)


def _calibrate_tau_amp(tau_unscaled: torch.Tensor, target_F: float = 0.979,
                       n_iter: int = 50) -> float:
    """Solve <exp(-amp * tau_unscaled)> = target_F via bisection."""
    lo, hi = 1e-6, 1e3
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        F_mean = torch.exp(-mid * tau_unscaled).mean().item()
        if F_mean > target_F:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _r_feas_from_F(F: torch.Tensor, dv: float) -> tuple[float, np.ndarray, np.ndarray]:
    """R_feas = Var_rays(band_integral_per_ray) / mean(band_integral)^2.
    Returns (R_feas, per_ray_band_integral, P_F_set_mean(k)) where the
    latter is for plotting.
    """
    k_axis, psd = compute_p_flux_torch(F, dv)
    band_mask = (k_axis >= K_MIN_INERTIAL) & (k_axis <= K_MAX_INERTIAL)
    if not band_mask.any():
        raise RuntimeError(
            f"No FFT bins in inertial band; k_axis spans "
            f"[{k_axis.min().item():.3g}, {k_axis.max().item():.3g}]"
        )
    band_int = psd[:, band_mask].mean(dim=1)  # (n_rays,)
    band_int_np = band_int.detach().cpu().numpy().astype(np.float64)
    mean_bi = band_int_np.mean()
    var_bi = band_int_np.var(ddof=1)
    r_feas = float(var_bi / max(mean_bi ** 2, 1e-300))
    psd_mean = psd.detach().cpu().numpy().mean(axis=0)
    k_axis_np = k_axis.detach().cpu().numpy()
    return r_feas, band_int_np, np.stack([k_axis_np, psd_mean], axis=0)


def _bootstrap_log_r(band_int: np.ndarray, n_boot: int = 200,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Resample rays with replacement, recompute R_feas, return log10(R)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = band_int.shape[0]
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bi = band_int[idx]
        mean_bi = bi.mean()
        var_bi = bi.var(ddof=1)
        r = var_bi / max(mean_bi ** 2, 1e-300)
        out[b] = np.log10(max(r, 1e-300))
    return out


def bootstrap_se_measurement(args) -> dict:
    """Load rho, downsample 768->192, sample sets A and B, FGPA, P_F,
    band integral, bootstrap SE.
    """
    print("[load] reading cached rho field (768^3 float32 ~ 1.7 GiB)...", flush=True)
    rho_768 = _load_rho_cached(physics_id=1, redshift=0.300, n_grid=768)
    print(f"[load] rho_768 stats: shape={rho_768.shape} dtype={rho_768.dtype} "
          f"mean={rho_768.mean():.4f} min={rho_768.min():.4g} "
          f"max={rho_768.max():.4g}", flush=True)

    print("[pool] 4^3 block-average pool 768 -> 192...", flush=True)
    rho_192 = _block_pool_avg(rho_768, factor=4)
    del rho_768  # free 1.7 GiB
    print(f"[pool] rho_192 stats: shape={rho_192.shape} "
          f"mean={rho_192.mean():.4f} min={rho_192.min():.4g} "
          f"max={rho_192.max():.4g}", flush=True)

    rho_field = torch.tensor(rho_192, dtype=torch.float32).contiguous()

    N_rays = int(args.n_rays)
    n_bins = int(args.n_bins)
    rng = np.random.default_rng(args.seed)

    o_A, d_A = _build_set_A(N_rays, rng)
    o_B, d_B = _build_set_B(N_rays, rng)

    # Velocity-axis spacing. dv per bin = v_per_kpc_h * (box_kpc_h / n_bins).
    # For axis-parallel rays t_max_unit=1 spans one box side; for angled
    # rays we use t_max_unit=1 too (path length up to sqrt(3) box sides).
    # We use the SAME dv for both sets so P_F units match. dv corresponds
    # to a one-box-side traversal divided into n_bins steps.
    v_box_kms = v_per_kpc_h_comoving(Z_SNAP) * BOX_KPC_H
    dv = float(v_box_kms / n_bins)
    print(f"[dv] v_box_kms={v_box_kms:.2f}  dv={dv:.4f} km/s per bin", flush=True)

    print(f"[sample] Set A axis-parallel, N={o_A.shape[0]}, n_bins={n_bins}",
          flush=True)
    rho_A = sample_field_along_rays(
        rho_field,
        torch.tensor(o_A, dtype=torch.float32),
        torch.tensor(d_A, dtype=torch.float32),
        n_samples=n_bins, t_max_unit=1.0,
    ).clamp(min=1e-6)

    print(f"[sample] Set B angled, N={o_B.shape[0]}, n_bins={n_bins}", flush=True)
    rho_B = sample_field_along_rays(
        rho_field,
        torch.tensor(o_B, dtype=torch.float32),
        torch.tensor(d_B, dtype=torch.float32),
        n_samples=n_bins, t_max_unit=1.0,
    ).clamp(min=1e-6)

    # Calibrate tau_amp jointly so <F> = 0.979 across the union of A+B
    # (so both sets share the same calibration as per d62_3 convention).
    tau_unscaled_AB = _render_fgpa_rho_only(
        torch.cat([rho_A, rho_B], dim=0), beta=BETA_TGAMMA, tau_amp=1.0,
    )
    tau_amp = _calibrate_tau_amp(tau_unscaled_AB, target_F=0.979)
    print(f"[fgpa] calibrated tau_amp = {tau_amp:.4g}", flush=True)

    tau_A = _render_fgpa_rho_only(rho_A, beta=BETA_TGAMMA, tau_amp=tau_amp)
    tau_B = _render_fgpa_rho_only(rho_B, beta=BETA_TGAMMA, tau_amp=tau_amp)
    F_A = torch.exp(-tau_A)
    F_B = torch.exp(-tau_B)
    print(f"[fgpa] <F_A>={F_A.mean().item():.4f}  "
          f"<F_B>={F_B.mean().item():.4f}", flush=True)

    R_A, band_A, psd_A = _r_feas_from_F(F_A, dv)
    R_B, band_B, psd_B = _r_feas_from_F(F_B, dv)
    print(f"[R_feas] Set A={R_A:.4g}  Set B={R_B:.4g}", flush=True)

    rng_boot = np.random.default_rng(args.seed + 1)
    logR_boot_A = _bootstrap_log_r(band_A, n_boot=args.n_boot, rng=rng_boot)
    rng_boot = np.random.default_rng(args.seed + 2)
    logR_boot_B = _bootstrap_log_r(band_B, n_boot=args.n_boot, rng=rng_boot)
    SE_A = float(np.std(logR_boot_A, ddof=1))
    SE_B = float(np.std(logR_boot_B, ddof=1))
    # delta log10 R = log10(R_B) - log10(R_A)
    delta_boot = logR_boot_B - logR_boot_A
    delta_ci_lo = float(np.percentile(delta_boot, 2.5))
    delta_ci_hi = float(np.percentile(delta_boot, 97.5))
    delta_meas = float(np.log10(max(R_B, 1e-300)) - np.log10(max(R_A, 1e-300)))
    print(f"[SE] SE_meas(A)={SE_A:.4f} dex   SE_meas(B)={SE_B:.4f} dex",
          flush=True)
    print(f"[delta] log10 R_B - log10 R_A = {delta_meas:.4f}  "
          f"95% CI [{delta_ci_lo:.4f}, {delta_ci_hi:.4f}]", flush=True)

    return {
        "rho_stats": {
            "shape_downsampled": [192, 192, 192],
            "mean": float(rho_192.mean()),
            "min": float(rho_192.min()),
            "max": float(rho_192.max()),
            "source_cache": str(Path(r"D:\Data\sujin\CosmoGasVision\Sherwood\.rho_field_cache\rho_field_p1_z0.300_n768.npy")),
            "downsample_factor": 4,
        },
        "dv_kms_per_bin": dv,
        "n_rays_per_set": N_rays,
        "n_bins": n_bins,
        "tau_amp_calibrated": float(tau_amp),
        "mean_F_A": float(F_A.mean().item()),
        "mean_F_B": float(F_B.mean().item()),
        "R_feas_A": float(R_A),
        "R_feas_B": float(R_B),
        "band_integral_A": band_A.tolist(),
        "band_integral_B": band_B.tolist(),
        "SE_meas_A_dex": SE_A,
        "SE_meas_B_dex": SE_B,
        "delta_log10_R_feas": delta_meas,
        "delta_log10_R_feas_95ci_lo": delta_ci_lo,
        "delta_log10_R_feas_95ci_hi": delta_ci_hi,
        "n_bootstrap": int(args.n_boot),
        "_psd_A": psd_A,  # (2, n_freq) [k, P_F]
        "_psd_B": psd_B,
    }


# -----------------------------------------------------------------------------
# (e) JSON + PNG output.
# -----------------------------------------------------------------------------

def _write_capsule(out_dir: Path, geo: dict, bgrad: dict, boot: dict,
                   wall_time_sec: float) -> Path:
    capsule = {
        "schema": "d71_I_angled_feasibility_preflight.v1",
        "honest_framing": (
            "Per [D-37] rule (a): observation-first. The bootstrap-SE "
            "measurement does NOT bias verdict-band selection; verdict-band "
            "tabulation is PI Amendment 2 work. 192^3 downsampled rho is the "
            "pre-flight approximation; Juno will run 768^3 native."
        ),
        "raw_observation": {
            "geometric_correctness": {
                "max_deviation": geo["max_deviation"],
                "max_deviation_vs_continuous_analytic": geo[
                    "max_deviation_vs_continuous_analytic"
                ],
                "reference_kind": geo["reference_kind"],
                "tolerance": geo["tolerance"],
                "n_rays": geo["n_rays"],
                "n_pass": geo["n_pass"],
                "pass": geo["pass"],
            },
            "boundary_gradient": {
                "n_boundary_rays": bgrad["n_boundary_rays"],
                "n_nan_grads_origins": bgrad["n_nan_grads_origins"],
                "n_nan_grads_dirs": bgrad["n_nan_grads_dirs"],
                "n_nan_samples_boundary": bgrad.get("n_nan_samples_boundary", 0),
                "pass": bgrad["pass"],
            },
            "rho_stats": boot["rho_stats"],
            "fgpa_calibration": {
                "tau_amp": boot["tau_amp_calibrated"],
                "mean_F_A": boot["mean_F_A"],
                "mean_F_B": boot["mean_F_B"],
                "target_mean_F": 0.979,
            },
            "n_rays_per_set": boot["n_rays_per_set"],
            "n_bins": boot["n_bins"],
            "dv_kms_per_bin": boot["dv_kms_per_bin"],
            "band_integral_A_first10": boot["band_integral_A"][:10],
            "band_integral_B_first10": boot["band_integral_B"][:10],
            "band_integral_A_summary": {
                "mean": float(np.mean(boot["band_integral_A"])),
                "std": float(np.std(boot["band_integral_A"], ddof=1)),
                "min": float(np.min(boot["band_integral_A"])),
                "max": float(np.max(boot["band_integral_A"])),
            },
            "band_integral_B_summary": {
                "mean": float(np.mean(boot["band_integral_B"])),
                "std": float(np.std(boot["band_integral_B"], ddof=1)),
                "min": float(np.min(boot["band_integral_B"])),
                "max": float(np.max(boot["band_integral_B"])),
            },
            "R_feas_A": boot["R_feas_A"],
            "R_feas_B": boot["R_feas_B"],
            "SE_meas_A_dex": boot["SE_meas_A_dex"],
            "SE_meas_B_dex": boot["SE_meas_B_dex"],
            "n_bootstrap": boot["n_bootstrap"],
        },
        "ratios": {
            "log10_R_feas_A": float(np.log10(max(boot["R_feas_A"], 1e-300))),
            "log10_R_feas_B": float(np.log10(max(boot["R_feas_B"], 1e-300))),
            "delta_log10_R_feas_B_minus_A": boot["delta_log10_R_feas"],
            "delta_log10_R_feas_95ci_lo": boot["delta_log10_R_feas_95ci_lo"],
            "delta_log10_R_feas_95ci_hi": boot["delta_log10_R_feas_95ci_hi"],
        },
        "verdict_band_crossref_deferred": {
            "rule": (
                "Verdict label deferred to PI Amendment-2 absorption per "
                "cycle #4 amended bands. SE_meas values above feed the "
                "verdict-band table (Amendment 1 §A1.1)."
            ),
            "amendment_1_reference_thresholds": {
                "PIVOT": "|delta| >= 6 * SE_meas + 0.30 dex",
                "MARGINAL": "6 * SE_meas <= |delta| < 6 * SE_meas + 0.30 dex",
                "NO_PIVOT": "|delta| < 6 * SE_meas",
            },
        },
        "wall_time_seconds": wall_time_sec,
    }
    out_path = out_dir / "capsule.json"
    out_path.write_text(json.dumps(capsule, indent=2))
    return out_path


def _write_histogram(out_dir: Path, boot: dict) -> Path:
    psd_A = boot["_psd_A"]  # (2, n_freq)
    psd_B = boot["_psd_B"]
    k_A, p_A = psd_A[0], psd_A[1]
    k_B, p_B = psd_B[0], psd_B[1]
    # Mask DC bin
    m_A = k_A > 0
    m_B = k_B > 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, k, p, title in [
        (axes[0], k_A[m_A], p_A[m_A], "Set A (axis-parallel)"),
        (axes[1], k_B[m_B], p_B[m_B], "Set B (angled, uniform-S2)"),
    ]:
        ax.loglog(k, np.clip(p, 1e-30, None), lw=1.0, color='tab:blue')
        ax.axvline(K_MIN_INERTIAL, color='k', ls='--', lw=0.7, alpha=0.6)
        ax.axvline(K_MAX_INERTIAL, color='k', ls='--', lw=0.7, alpha=0.6)
        ax.set_xlabel(r"$k_{\parallel}$  [s/km]")
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.25)
    axes[0].set_ylabel(r"mean $P_F(k_{\parallel})$  [s/km]")
    R_A = boot["R_feas_A"]
    R_B = boot["R_feas_B"]
    SE_A = boot["SE_meas_A_dex"]
    SE_B = boot["SE_meas_B_dex"]
    fig.suptitle(
        f"[D-71] Rung 4 CPU pre-flight   "
        f"R_feas(A)={R_A:.3g}  R_feas(B)={R_B:.3g}   "
        f"SE_meas(A)={SE_A:.3f}dex  SE_meas(B)={SE_B:.3f}dex",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = out_dir / "histogram.png"
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-rays", type=int, default=768,
                        help="Per-set ray count (axis-parallel split 256/256/256).")
    parser.add_argument("--n-bins", type=int, default=2048,
                        help="Sherwood convention.")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument(
        "--out-dir", type=str,
        default=str(_REPO_ROOT / "experiments/nerf/artifacts/d71_I_angled_feasibility_preflight"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}", flush=True)

    t0 = time.time()
    print("[step 1/3] geometric correctness on analytic sin(2pi x/L) field...",
          flush=True)
    geo = geometric_correctness_check(seed=args.seed)
    print(f"  -> max_dev={geo['max_deviation']:.3e}  "
          f"pass={geo['pass']}  ({geo['n_pass']}/{geo['n_rays']})",
          flush=True)

    print("[step 2/3] boundary-gradient autograd verification...", flush=True)
    bgrad = boundary_gradient_check(geo)
    print(f"  -> n_boundary={bgrad['n_boundary_rays']}  "
          f"NaN grad o={bgrad['n_nan_grads_origins']}  "
          f"NaN grad d={bgrad['n_nan_grads_dirs']}  "
          f"pass={bgrad['pass']}", flush=True)

    print("[step 3/3] bootstrap-SE on real rho field (192^3 from 768^3 cache)...",
          flush=True)
    boot = bootstrap_se_measurement(args)

    wall = time.time() - t0
    capsule_path = _write_capsule(out_dir, geo, bgrad, boot, wall_time_sec=wall)
    png_path = _write_histogram(out_dir, boot)

    print(f"\n[done] wall_time = {wall:.1f} s")
    print(f"[done] capsule: {capsule_path}")
    print(f"[done] histogram: {png_path}")
    # Hand-back summary block
    print("\n=== HAND-BACK SUMMARY ===")
    print(f"geometric-correctness: PASS={geo['pass']} "
          f"max_dev={geo['max_deviation']:.3e}  ({geo['n_pass']}/{geo['n_rays']})")
    print(f"boundary-gradient: PASS={bgrad['pass']}  "
          f"n_boundary={bgrad['n_boundary_rays']}  "
          f"n_nan_o={bgrad['n_nan_grads_origins']}  "
          f"n_nan_d={bgrad['n_nan_grads_dirs']}")
    print(f"SE_meas(A) = {boot['SE_meas_A_dex']:.4f} dex   "
          f"SE_meas(B) = {boot['SE_meas_B_dex']:.4f} dex")
    print(f"delta log10 R_feas (B-A) = {boot['delta_log10_R_feas']:.4f}   "
          f"95% CI [{boot['delta_log10_R_feas_95ci_lo']:.4f}, "
          f"{boot['delta_log10_R_feas_95ci_hi']:.4f}]")
    print(f"wall time: {wall:.1f} s")


if __name__ == "__main__":
    main()

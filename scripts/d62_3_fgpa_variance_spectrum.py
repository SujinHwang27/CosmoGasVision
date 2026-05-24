"""CPU pre-flight Deliverable A for Sprint-L2 [D-62] candidate (3):
fGPA-truth flux power spectrum vs L1 collapsed-basin variance.

Computes the inertial-band P_F variance of `F_fGPA = exp(-tau_fGPA)` where
`tau_fGPA` is the Voigt-integrated optical depth produced by the standard
[D-41] Hui-Gnedin 1997 power-law substitution

    n_HI_fGPA(x)  s.t.  tau_local ~ (1 + delta(x))**beta * T(x)**gamma

at z=0.3, beta=1.6, gamma=-0.7 (defaults from
``experiments/nerf/pipeline.py``). The variance is compared to the L1
collapsed-basin `var_pf_band_ratio` cluster
`[3.74e-7, 2.93e-6]` (per task spec).

PASS criterion (per task spec): fGPA variance must be >= 30x the upper end
of the L1 cluster (1.5 log10-decades headroom). On PASS, the [D-62] ladder
proceeds to candidate (3) build. On FAIL, eligibility is revoked and the
ladder escalates to (2) density-pretraining.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from src.data.loader import SherwoodLoader  # noqa: E402
from src.models.nerf import tepper_garcia_voigt  # noqa: E402


# [D-13] inertial band edges (same as flux_power_torch._K_*_INERTIAL)
K_MIN_INERTIAL = 10.0 ** -2.5  # s/km
K_MAX_INERTIAL = 10.0 ** -1.5  # s/km

# Task-spec anchors: L1 7-attempt cluster var_pf_band_ratio range
L1_CLUSTER_LO = 3.74e-7
L1_CLUSTER_HI = 2.93e-6
PASS_MULTIPLIER = 30.0  # fGPA must be >= 30x L1_CLUSTER_HI


def _render_tau_chunk(
    density: torch.Tensor, h1_frac: torch.Tensor, temp: torch.Tensor,
    v_pec: torch.Tensor, vel_axis: torch.Tensor, window: int,
) -> torch.Tensor:
    n_obs = vel_axis.shape[0]
    n_rays, _ = density.shape
    device = vel_axis.device
    dtype = density.dtype

    b = 12.85 * torch.sqrt(temp / 10000.0)
    a = 6.063e-3 / b
    n_hi = density * h1_frac

    dv_per_bin = (vel_axis[-1] - vel_axis[0]) / (n_obs - 1)
    v_source = vel_axis[None, :] + v_pec
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

    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))
    contrib = (n_hi[..., None] * H) / (b[..., None] * sqrt_pi)

    tau = torch.zeros((n_rays, n_obs), dtype=dtype, device=device)
    tau.scatter_add_(
        1,
        obs_idx_safe.reshape(n_rays, -1),
        contrib.reshape(n_rays, -1),
    )
    return tau


def render_tau_from_fields(
    density: torch.Tensor,   # rho/<rho>, (n_rays, n_src)
    h1_frac: torch.Tensor,   # X_HI, (n_rays, n_src)
    temp: torch.Tensor,      # K, (n_rays, n_src)
    v_pec: torch.Tensor,     # km/s, (n_rays, n_src)
    vel_axis: torch.Tensor,  # km/s, (n_obs,)
    window: int = 64,
    chunk_size: int = 32,    # rays per micro-batch (CPU memory guard)
) -> torch.Tensor:
    """Strip of the [D-24] Voigt renderer from `volume_render_physics`,
    callable with truth/fGPA-substituted fields directly (no MLP).

    Chunked along the rays axis to keep the intermediate Voigt tensor
    (n_rays, n_src, 2W+1) inside CPU RAM at n_src=2048, W=64.

    Returns
    -------
    tau : (n_rays, n_obs) optical depth.
    """
    n_rays = density.shape[0]
    chunks = []
    for i in range(0, n_rays, chunk_size):
        chunks.append(_render_tau_chunk(
            density[i:i+chunk_size], h1_frac[i:i+chunk_size],
            temp[i:i+chunk_size], v_pec[i:i+chunk_size],
            vel_axis, window,
        ))
    return torch.cat(chunks, dim=0)


def compute_inertial_band_variance_per_kbin(
    F: torch.Tensor,
    dv: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (k_inband, var_per_kbin, var_band_mean).

    var_per_kbin: per-FFT-bin variance of P_F across the sightline ensemble
    (variance taken over the n_rays axis at each k). var_band_mean: mean of
    var_per_kbin across the inertial-band FFT bins — the scalar number that
    enters the L1-comparison ratio.
    """
    with torch.no_grad():
        k_axis, psd = compute_p_flux_torch(F, dv=dv)  # (n_rays, n_freq)
        band_mask = (k_axis >= K_MIN_INERTIAL) & (k_axis <= K_MAX_INERTIAL)
        psd_inband = psd[:, band_mask]            # (n_rays, n_inband)
        k_inband = k_axis[band_mask]              # (n_inband,)
        # variance across sightline ensemble at each k
        var_per_kbin = psd_inband.var(dim=0, unbiased=False)
        var_band_mean = var_per_kbin.mean().item()
    return (
        k_inband.cpu().numpy(),
        var_per_kbin.cpu().numpy(),
        var_band_mean,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str,
        default=str(_REPO_ROOT / "Sherwood"),
        help="Sherwood data root (default: <repo>/Sherwood).",
    )
    parser.add_argument("--physics_id", type=int, default=1)
    parser.add_argument("--redshift", type=float, default=0.300)
    parser.add_argument(
        "--n_rays", type=int, default=512,
        help="Number of sightlines to draw (default 512; CPU-fits).",
    )
    parser.add_argument("--fgpa_beta", type=float, default=1.6)
    parser.add_argument("--fgpa_gamma", type=float, default=-0.7)
    parser.add_argument(
        "--out_png", type=str,
        default=str(_REPO_ROOT / "experiments/nerf/artifacts/"
                    "d62_3_fgpa_variance_spectrum.png"),
    )
    parser.add_argument(
        "--out_json", type=str,
        default=str(_REPO_ROOT / "experiments/nerf/artifacts/"
                    "d62_3_fgpa_variance_spectrum.json"),
    )
    args = parser.parse_args()

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)

    print(f"[D-62/3] Loading P{args.physics_id} z={args.redshift:.3f} from {args.data_root}",
          flush=True)
    loader = SherwoodLoader(args.data_root)
    sl = loader.load_sightlines(
        physics_id=args.physics_id, redshift=args.redshift, nspec=16384,
    )
    n_rays = min(args.n_rays, sl["density"].shape[0])
    print(f"[D-62/3] Using n_rays={n_rays} sightlines (truncated from "
          f"{sl['density'].shape[0]})", flush=True)

    device = torch.device("cpu")
    density_truth = torch.tensor(sl["density"][:n_rays], dtype=torch.float32, device=device)
    h1_frac_truth = torch.tensor(sl["h1_frac"][:n_rays], dtype=torch.float32, device=device)
    temp_truth = torch.tensor(sl["temp"][:n_rays], dtype=torch.float32, device=device)
    v_pec_truth = torch.tensor(sl["v_pec"][:n_rays], dtype=torch.float32, device=device)
    vel_axis = torch.tensor(sl["vel_axis"], dtype=torch.float32, device=device)
    dv_kms = float((vel_axis[1] - vel_axis[0]).item())
    print(f"[D-62/3] dv = {dv_kms:.4f} km/s, n_obs = {vel_axis.shape[0]}", flush=True)

    # ---------------------- fGPA substitution -------------------------
    # The fGPA prediction is for n_HI such that
    #     tau_local_fGPA ~ density^beta * temp^gamma,
    # i.e. n_HI_fGPA = density^beta * temp^gamma / density = density^(beta-1) * temp^gamma
    # since the renderer combines n_hi = density * h1_frac and then
    # tau_local = n_hi * dv / (b * sqrt(pi)) with b ~ sqrt(T). We bypass
    # h1_frac entirely and inject an effective h1_frac_fGPA that yields the
    # Hui-Gnedin scaling at constant tau_amp=1. Concretely we set
    #     h1_frac_fGPA = density^(beta-1) * temp^gamma
    # and pass the truth temp, density, v_pec through the same renderer.
    # This isolates the variance carried by the fGPA *spatial structure*
    # without re-implementing the Voigt path.
    beta = args.fgpa_beta
    gamma = args.fgpa_gamma
    print(f"[D-62/3] fGPA params: beta={beta}, gamma={gamma}", flush=True)
    # Numeric guards: clamp positive
    density_safe = density_truth.clamp_min(1e-30)
    temp_safe = temp_truth.clamp_min(1e-30)
    h1_frac_fgpa = density_safe.pow(beta - 1.0) * temp_safe.pow(gamma)

    print("[D-62/3] Rendering tau_fGPA (truth-side beta/gamma substitution)...",
          flush=True)
    tau_fgpa = render_tau_from_fields(
        density=density_truth,
        h1_frac=h1_frac_fgpa,
        temp=temp_truth,
        v_pec=v_pec_truth,
        vel_axis=vel_axis,
        window=64,
    )
    print(f"[D-62/3] tau_fGPA stats: mean={tau_fgpa.mean().item():.4e} "
          f"max={tau_fgpa.max().item():.4e} median={tau_fgpa.median().item():.4e}",
          flush=True)
    F_fgpa = torch.exp(-tau_fgpa)
    print(f"[D-62/3] F_fGPA stats: mean={F_fgpa.mean().item():.4f} "
          f"min={F_fgpa.min().item():.4e} max={F_fgpa.max().item():.4f}",
          flush=True)

    # Also render the truth flux as an upper-anchor sanity check
    print("[D-62/3] Rendering tau_truth (full Sherwood fields, same renderer)...",
          flush=True)
    tau_truth = render_tau_from_fields(
        density=density_truth,
        h1_frac=h1_frac_truth,
        temp=temp_truth,
        v_pec=v_pec_truth,
        vel_axis=vel_axis,
        window=64,
    )
    F_truth = torch.exp(-tau_truth)
    print(f"[D-62/3] F_truth stats: mean={F_truth.mean().item():.4f}",
          flush=True)

    # ---------------------- inertial-band variance --------------------
    k_in, var_per_k_fgpa, var_mean_fgpa = compute_inertial_band_variance_per_kbin(
        F_fgpa, dv=dv_kms)
    _, var_per_k_truth, var_mean_truth = compute_inertial_band_variance_per_kbin(
        F_truth, dv=dv_kms)

    # Compare to L1 cluster
    ratio_fgpa_vs_L1hi = var_mean_fgpa / L1_CLUSTER_HI
    ratio_fgpa_vs_L1lo = var_mean_fgpa / L1_CLUSTER_LO
    log10_decades_above_L1hi = float(np.log10(max(ratio_fgpa_vs_L1hi, 1e-30)))

    pass_flag = bool(ratio_fgpa_vs_L1hi >= PASS_MULTIPLIER)

    print("", flush=True)
    print("================= [D-62/3] CPU pre-flight A verdict =================",
          flush=True)
    print(f"  fGPA var_pf_band_mean       = {var_mean_fgpa:.4e}", flush=True)
    print(f"  truth var_pf_band_mean      = {var_mean_truth:.4e}", flush=True)
    print(f"  L1 cluster range            = [{L1_CLUSTER_LO:.2e}, "
          f"{L1_CLUSTER_HI:.2e}]", flush=True)
    print(f"  fGPA / L1_HI                = {ratio_fgpa_vs_L1hi:.3f}x "
          f"({log10_decades_above_L1hi:+.3f} log10-decades)", flush=True)
    print(f"  fGPA / L1_LO                = {ratio_fgpa_vs_L1lo:.3f}x", flush=True)
    print(f"  PASS threshold (>= 30x L1_HI): "
          f"{'PASS' if pass_flag else 'FAIL'}", flush=True)
    print("======================================================================",
          flush=True)

    # ---------------------- overlay plot ------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k_in, var_per_k_truth, "-", color="tab:green",
              label=f"truth (mean={var_mean_truth:.2e})")
    ax.loglog(k_in, var_per_k_fgpa, "-", color="tab:blue",
              label=f"fGPA (mean={var_mean_fgpa:.2e})")
    ax.axhspan(L1_CLUSTER_LO, L1_CLUSTER_HI, alpha=0.2, color="tab:red",
               label=f"L1 cluster [{L1_CLUSTER_LO:.1e}, {L1_CLUSTER_HI:.1e}]")
    ax.axhline(PASS_MULTIPLIER * L1_CLUSTER_HI, color="tab:red", ls="--",
               alpha=0.7,
               label=f"PASS threshold = 30x L1_HI = {PASS_MULTIPLIER*L1_CLUSTER_HI:.1e}")
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
    ax.set_ylabel(r"$\mathrm{Var}_{\mathrm{sightlines}}[P_F(k_\parallel)]$")
    ax.set_title(f"[D-62/3] fGPA flux-power variance vs L1 collapsed basin\n"
                 f"P{args.physics_id} z={args.redshift:.3f}, n_rays={n_rays}, "
                 f"verdict = {'PASS' if pass_flag else 'FAIL'}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"[D-62/3] Wrote overlay plot: {args.out_png}", flush=True)

    # ---------------------- numeric artifact --------------------------
    payload = {
        "d_decision": "D-62/3 CPU pre-flight A",
        "physics_id": args.physics_id,
        "redshift": args.redshift,
        "n_rays": n_rays,
        "n_obs": int(vel_axis.shape[0]),
        "dv_kms": dv_kms,
        "fgpa_beta": beta,
        "fgpa_gamma": gamma,
        "k_inband_s_per_km": k_in.tolist(),
        "var_per_kbin_fgpa": var_per_k_fgpa.tolist(),
        "var_per_kbin_truth": var_per_k_truth.tolist(),
        "var_band_mean_fgpa": var_mean_fgpa,
        "var_band_mean_truth": var_mean_truth,
        "L1_cluster_lo": L1_CLUSTER_LO,
        "L1_cluster_hi": L1_CLUSTER_HI,
        "ratio_fgpa_over_L1hi": float(ratio_fgpa_vs_L1hi),
        "ratio_fgpa_over_L1lo": float(ratio_fgpa_vs_L1lo),
        "log10_decades_above_L1hi": log10_decades_above_L1hi,
        "pass_multiplier": PASS_MULTIPLIER,
        "verdict": "PASS" if pass_flag else "FAIL",
    }
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[D-62/3] Wrote numeric artifact: {args.out_json}", flush=True)

    return 0 if pass_flag else 1


if __name__ == "__main__":
    raise SystemExit(main())

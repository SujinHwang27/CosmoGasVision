"""CPU pre-flight for Sprint-L2 [D-62] candidate (3) — REDO per PI v9 [D-66].

Defense panel killed the previous PASS verdict (commit 4b2b699) with:
  K1: rendered fGPA tau WITHOUT [D-10] sigma_0 * path-length prefactor,
      while L1 cluster numbers WERE measured WITH it. Apples-to-oranges.
  K2: var_truth = 3.74e-15 is 13+ OOM below Boera+ 2019 z=0.3 published
      P_F variance O(1e-2 to 1e-1) — FP-noise-floor, not science.
  S3: missing noise-init robustness deliverable (the (3) feasibility
      falsifier: does fGPA prevent the L1 collapse basin when delta is
      drawn from noise, not the truth field?).

Fixes (Stage 1 REDO):
  Fix A (K1): apply [D-10] tau_amp prefactor analytically calibrated so
              <F_truth> = 0.979 (Becker+ 2013 anchor). Same tau_amp then
              applied to fGPA-rendered tau for like-for-like comparison.
  Fix B (K2): hard-gate assertion var_truth in [1e-3, 1e-1] per Boera+
              2019 (arXiv:1809.06980) Table 2 z=0.3, BEFORE PASS verdict.
  Fix C (S3): Deliverable C — render fGPA from delta_noise ~ N(0, sigma_truth)
              (gaussian noise field, truth moments, no spatial structure).
              PASS gate: var_fgpa(noise-delta) >= 30x L1_cluster_hi. This is
              the (3)-vs-(2) feasibility falsifier per PI v9.

Bonus: ||rho_truth - rho_fGPA||_2 / ||rho_fGPA||_2 for the (3) inverse-failure
       threshold (panel S1) — written to d62_3_fgpa_prior_reconstruction_error.json.
"""

from __future__ import annotations

import argparse
import json
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

# Becker+ 2013 mean-flux anchor at z=0.3 (production [D-10] convention)
MEAN_FLUX_TARGET = 0.979

# Boera+ 2019 (arXiv:1809.06980) Table 2 z=0.3 P_F variance physical band
# (truth-side P_F at inertial scales): O(1e-2 to 1e-1) in (km/s)^-1 units
# of P_F. Conservative gate per task spec Fix B.
BOERA_PF_VAR_LO = 1e-3
BOERA_PF_VAR_HI = 1e-1


def _render_tau_unscaled_chunk(
    density: torch.Tensor, h1_frac: torch.Tensor, temp: torch.Tensor,
    v_pec: torch.Tensor, vel_axis: torch.Tensor, window: int,
) -> torch.Tensor:
    """Render UNSCALED tau (tau_amp = 1). The scalar prefactor is applied
    once in the outer scope so calibrated and uncalibrated variants share
    the same expensive Voigt kernel evaluation."""
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


def render_tau_unscaled(
    density: torch.Tensor, h1_frac: torch.Tensor, temp: torch.Tensor,
    v_pec: torch.Tensor, vel_axis: torch.Tensor,
    window: int = 64, chunk_size: int = 32,
) -> torch.Tensor:
    """Chunked unscaled-tau renderer (no tau_amp). Outer scope multiplies
    by the calibrated scalar tau_amp."""
    n_rays = density.shape[0]
    chunks = []
    for i in range(0, n_rays, chunk_size):
        chunks.append(_render_tau_unscaled_chunk(
            density[i:i+chunk_size], h1_frac[i:i+chunk_size],
            temp[i:i+chunk_size], v_pec[i:i+chunk_size],
            vel_axis, window,
        ))
    return torch.cat(chunks, dim=0)


def calibrate_tau_amp(tau_unscaled: torch.Tensor, target_mean_F: float,
                      n_iter: int = 80) -> float:
    """[D-10] anchor: solve <exp(-tau_amp * tau_unscaled)> = target_mean_F
    by bisection in log10(tau_amp). The function <F>(log tau_amp) is
    monotonically decreasing from 1 to 0, so bisection is well-posed.
    """
    with torch.no_grad():
        # Search log10(tau_amp) in [-6, 6]; truth-renderer tau_unscaled is
        # ~1e0 in mean (small h1_frac in physical units inside this script
        # since loader returns X_HI dimensionless), so true tau_amp covers
        # several decades.
        lo, hi = -6.0, 6.0
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            amp = 10.0 ** mid
            mean_F = torch.exp(-amp * tau_unscaled).mean().item()
            if mean_F > target_mean_F:
                # too transparent => need MORE absorption => bigger tau_amp
                lo = mid
            else:
                hi = mid
        return 10.0 ** (0.5 * (lo + hi))


def compute_inertial_band_variance_per_kbin(
    F: torch.Tensor, dv: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (k_inband, var_per_kbin, var_band_mean)."""
    with torch.no_grad():
        k_axis, psd = compute_p_flux_torch(F, dv=dv)  # (n_rays, n_freq)
        band_mask = (k_axis >= K_MIN_INERTIAL) & (k_axis <= K_MAX_INERTIAL)
        psd_inband = psd[:, band_mask]
        k_inband = k_axis[band_mask]
        var_per_kbin = psd_inband.var(dim=0, unbiased=False)
        var_band_mean = var_per_kbin.mean().item()
    return (
        k_inband.cpu().numpy(),
        var_per_kbin.cpu().numpy(),
        var_band_mean,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default=str(_REPO_ROOT / "Sherwood"))
    parser.add_argument("--physics_id", type=int, default=1)
    parser.add_argument("--redshift", type=float, default=0.300)
    parser.add_argument("--n_rays", type=int, default=512)
    parser.add_argument("--fgpa_beta", type=float, default=1.6)
    parser.add_argument("--fgpa_gamma", type=float, default=-0.7)
    parser.add_argument("--noise_seed", type=int, default=2026)
    parser.add_argument("--mean_flux_target", type=float,
                        default=MEAN_FLUX_TARGET,
                        help="[D-10] Becker+2013 anchor for tau_amp calibration.")
    parser.add_argument("--force_bad_truth_var", action="store_true",
                        help="DEBUG: scale truth tau by tiny factor to confirm "
                             "K2 smoke assert fires (test path only).")
    parser.add_argument("--out_png", type=str,
                        default=str(_REPO_ROOT / "experiments/nerf/artifacts/"
                                    "d62_3_fgpa_variance_spectrum.png"))
    parser.add_argument("--out_json", type=str,
                        default=str(_REPO_ROOT / "experiments/nerf/artifacts/"
                                    "d62_3_fgpa_variance_spectrum.json"))
    parser.add_argument("--out_recon_json", type=str,
                        default=str(_REPO_ROOT / "experiments/nerf/artifacts/"
                                    "d62_3_fgpa_prior_reconstruction_error.json"))
    args = parser.parse_args()

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)

    print(f"[D-62/3 REDO] Loading P{args.physics_id} z={args.redshift:.3f} "
          f"from {args.data_root}", flush=True)
    loader = SherwoodLoader(args.data_root)
    sl = loader.load_sightlines(
        physics_id=args.physics_id, redshift=args.redshift, nspec=16384,
    )
    n_rays = min(args.n_rays, sl["density"].shape[0])
    print(f"[D-62/3 REDO] Using n_rays={n_rays}", flush=True)

    device = torch.device("cpu")
    density_truth = torch.tensor(sl["density"][:n_rays], dtype=torch.float32, device=device)
    h1_frac_truth = torch.tensor(sl["h1_frac"][:n_rays], dtype=torch.float32, device=device)
    temp_truth = torch.tensor(sl["temp"][:n_rays], dtype=torch.float32, device=device)
    v_pec_truth = torch.tensor(sl["v_pec"][:n_rays], dtype=torch.float32, device=device)
    vel_axis = torch.tensor(sl["vel_axis"], dtype=torch.float32, device=device)
    dv_kms = float((vel_axis[1] - vel_axis[0]).item())
    print(f"[D-62/3 REDO] dv = {dv_kms:.4f} km/s, n_obs = {vel_axis.shape[0]}",
          flush=True)

    # ============== UNSCALED tau renderer for all three variants ===========
    # 1. TRUTH
    print("[D-62/3 REDO] Rendering tau_unscaled_truth ...", flush=True)
    tau_unscaled_truth = render_tau_unscaled(
        density_truth, h1_frac_truth, temp_truth, v_pec_truth, vel_axis,
    )

    # 2. fGPA with truth-delta spatial structure: h1_frac_fGPA = Delta^(beta-1) * T^gamma
    beta = args.fgpa_beta
    gamma = args.fgpa_gamma
    print(f"[D-62/3 REDO] fGPA params: beta={beta}, gamma={gamma}", flush=True)
    density_safe = density_truth.clamp_min(1e-30)
    temp_safe = temp_truth.clamp_min(1e-30)
    h1_frac_fgpa_truth = density_safe.pow(beta - 1.0) * temp_safe.pow(gamma)

    print("[D-62/3 REDO] Rendering tau_unscaled_fgpa(truth-delta) ...", flush=True)
    tau_unscaled_fgpa_td = render_tau_unscaled(
        density_truth, h1_frac_fgpa_truth, temp_truth, v_pec_truth, vel_axis,
    )

    # 3. Fix C — fGPA with NOISE-delta: gaussian field preserving truth's
    # first/second moments but destroying spatial structure. The renderer
    # uses Delta (density) and T fields to build h1_frac_fgpa; we replace
    # Delta only (T stays truth so the prior's T^gamma term is well-defined).
    rng = np.random.default_rng(args.noise_seed)
    mu_d = float(density_truth.mean().item())
    sigma_d = float(density_truth.std(unbiased=False).item())
    print(f"[D-62/3 REDO] Truth delta moments: mu={mu_d:.4e}, sigma={sigma_d:.4e}",
          flush=True)
    density_noise_np = rng.normal(loc=mu_d, scale=sigma_d, size=density_truth.shape)
    density_noise = torch.tensor(density_noise_np, dtype=torch.float32, device=device)
    density_noise = density_noise.clamp_min(1e-30)  # positivity
    h1_frac_fgpa_noise = density_noise.pow(beta - 1.0) * temp_safe.pow(gamma)
    print("[D-62/3 REDO] Rendering tau_unscaled_fgpa(noise-delta) ...", flush=True)
    tau_unscaled_fgpa_nd = render_tau_unscaled(
        density_noise, h1_frac_fgpa_noise, temp_truth, v_pec_truth, vel_axis,
    )

    # ================ Fix A: [D-10] tau_amp calibration on truth =============
    print(f"[D-62/3 REDO] Calibrating tau_amp such that <F_truth> = "
          f"{args.mean_flux_target:.4f} ...", flush=True)
    tau_amp = calibrate_tau_amp(tau_unscaled_truth, args.mean_flux_target)
    print(f"[D-62/3 REDO] Calibrated tau_amp = {tau_amp:.6e}", flush=True)

    if args.force_bad_truth_var:
        # Debug path to verify Fix B assert fires.
        tau_amp_for_truth = tau_amp * 1e-12
        print("[D-62/3 REDO] WARNING: --force_bad_truth_var active "
              "(testing K2 smoke assert).", flush=True)
    else:
        tau_amp_for_truth = tau_amp

    tau_truth = tau_amp_for_truth * tau_unscaled_truth
    tau_fgpa_td = tau_amp * tau_unscaled_fgpa_td
    tau_fgpa_nd = tau_amp * tau_unscaled_fgpa_nd

    F_truth = torch.exp(-tau_truth)
    F_fgpa_td = torch.exp(-tau_fgpa_td)
    F_fgpa_nd = torch.exp(-tau_fgpa_nd)

    print(f"[D-62/3 REDO] <F_truth>       = {F_truth.mean().item():.4f}", flush=True)
    print(f"[D-62/3 REDO] <F_fgpa(td)>    = {F_fgpa_td.mean().item():.4f}", flush=True)
    print(f"[D-62/3 REDO] <F_fgpa(nd)>    = {F_fgpa_nd.mean().item():.4f}", flush=True)

    # ================ Inertial-band variance =================
    k_in, var_per_k_truth, var_mean_truth = compute_inertial_band_variance_per_kbin(
        F_truth, dv=dv_kms)
    _, var_per_k_fgpa_td, var_mean_fgpa_td = compute_inertial_band_variance_per_kbin(
        F_fgpa_td, dv=dv_kms)
    _, var_per_k_fgpa_nd, var_mean_fgpa_nd = compute_inertial_band_variance_per_kbin(
        F_fgpa_nd, dv=dv_kms)

    # ================ Fix B: K2 SMOKE ASSERT BEFORE VERDICT ===============
    print("", flush=True)
    print(f"[D-62/3 REDO] var_truth_band_mean = {var_mean_truth:.4e} "
          f"(Boera+2019 z=0.3 range [{BOERA_PF_VAR_LO:.1e}, "
          f"{BOERA_PF_VAR_HI:.1e}])", flush=True)
    assert BOERA_PF_VAR_LO <= var_mean_truth <= BOERA_PF_VAR_HI, (
        f"K2 SMOKE FAIL: var_truth_band_mean={var_mean_truth:.3e} is "
        f"outside Boera+ 2019 published range [{BOERA_PF_VAR_LO:.0e}, "
        f"{BOERA_PF_VAR_HI:.0e}] for z=0.3 P_F. Renderer is mis-instrumented; "
        f"PASS verdict invalidated. (Per PI v9 / panel killer K2.)"
    )
    print("[D-62/3 REDO] K2 smoke assert PASSED (truth variance physical).",
          flush=True)

    # ================ Verdict gates ===================
    # Gate A (legacy, truth-delta fGPA): kept for continuity; informational.
    ratio_td_vs_L1hi = var_mean_fgpa_td / L1_CLUSTER_HI
    pass_A = bool(ratio_td_vs_L1hi >= PASS_MULTIPLIER)

    # Gate C (Fix C, noise-delta fGPA): the (3)-vs-(2) feasibility falsifier.
    ratio_nd_vs_L1hi = var_mean_fgpa_nd / L1_CLUSTER_HI
    pass_C = bool(ratio_nd_vs_L1hi >= PASS_MULTIPLIER)
    log10_dec_nd = float(np.log10(max(ratio_nd_vs_L1hi, 1e-30)))
    log10_dec_td = float(np.log10(max(ratio_td_vs_L1hi, 1e-30)))

    # Overall Stage 1 PASS: K2 already asserted; require both gates.
    overall_pass = pass_A and pass_C

    print("", flush=True)
    print("============== [D-62/3 REDO] Stage 1 verdict ===============",
          flush=True)
    print(f"  Fix A tau_amp (calibrated)        = {tau_amp:.4e}", flush=True)
    print(f"  Fix B var_truth                   = {var_mean_truth:.4e}  "
          f"PASS  (in Boera band)", flush=True)
    print(f"  Gate A var_fgpa(truth-delta)      = {var_mean_fgpa_td:.4e}  "
          f"ratio={ratio_td_vs_L1hi:.2f}x  "
          f"({log10_dec_td:+.2f} dec)  "
          f"{'PASS' if pass_A else 'FAIL'}", flush=True)
    print(f"  Gate C var_fgpa(noise-delta)      = {var_mean_fgpa_nd:.4e}  "
          f"ratio={ratio_nd_vs_L1hi:.2f}x  "
          f"({log10_dec_nd:+.2f} dec)  "
          f"{'PASS' if pass_C else 'FAIL'}  <-- (3)-vs-(2) feasibility falsifier",
          flush=True)
    print(f"  L1 cluster anchor                 = "
          f"[{L1_CLUSTER_LO:.2e}, {L1_CLUSTER_HI:.2e}]  "
          f"(>= {PASS_MULTIPLIER}x L1_HI = {PASS_MULTIPLIER*L1_CLUSTER_HI:.2e})",
          flush=True)
    print(f"  OVERALL STAGE 1                   = "
          f"{'PASS' if overall_pass else 'FAIL'}", flush=True)
    if not pass_C:
        print("  >>> Gate C FAIL: (3) architectural-rescue claim FALSIFIED "
              "ex ante.  Stages 2-5 ABORT per PI v9.  Escalate to (2).",
              flush=True)
    print("============================================================",
          flush=True)

    # ================ Bonus: rho-prior reconstruction error =================
    # The (3) prior gives delta_pred from local thermo; if we invert fGPA
    # at the source bins:
    #     n_HI_fGPA = density^(beta-1) * temp^gamma * density = density^beta * temp^gamma
    # but here we report the simpler 'rho_fGPA = density' (trivially identity
    # for truth-delta input). For the inversion-failure threshold per panel S1,
    # PI wants the residual between truth density and the density implied by
    # fitting fGPA to truth n_HI. Solve n_HI_truth = density_implied^beta * T^gamma
    # => density_implied = (n_HI_truth / T^gamma)^(1/beta)
    n_HI_truth = density_truth * h1_frac_truth
    n_HI_safe = n_HI_truth.clamp_min(1e-40)
    density_implied = (n_HI_safe / temp_safe.pow(gamma)).pow(1.0 / beta)
    num = (density_truth - density_implied).pow(2).sum().sqrt().item()
    den = density_implied.pow(2).sum().sqrt().item()
    rel_l2 = num / max(den, 1e-30)
    print(f"[D-62/3 REDO] Bonus: ||rho_truth - rho_fGPA_implied||_2 / "
          f"||rho_fGPA_implied||_2 = {rel_l2:.4e}", flush=True)
    with open(args.out_recon_json, "w") as f:
        json.dump({
            "d_decision": "D-62/3 Stage 1 REDO bonus for PI Stage 3 panel-S1 calibration",
            "physics_id": args.physics_id,
            "redshift": args.redshift,
            "fgpa_beta": beta,
            "fgpa_gamma": gamma,
            "rel_l2_rho_truth_vs_rho_fgpa_implied": rel_l2,
            "n_rays_used": n_rays,
        }, f, indent=2)
    print(f"[D-62/3 REDO] Wrote bonus artifact: {args.out_recon_json}",
          flush=True)

    # ================ Overlay plot ==================
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.loglog(k_in, var_per_k_truth, "-", color="tab:green",
              label=f"truth (mean={var_mean_truth:.2e})")
    ax.loglog(k_in, var_per_k_fgpa_td, "-", color="tab:blue",
              label=f"fGPA(truth-delta) (mean={var_mean_fgpa_td:.2e})")
    ax.loglog(k_in, var_per_k_fgpa_nd, "-", color="tab:purple",
              label=f"fGPA(noise-delta) (mean={var_mean_fgpa_nd:.2e})")
    ax.axhspan(L1_CLUSTER_LO, L1_CLUSTER_HI, alpha=0.2, color="tab:red",
               label=f"L1 cluster [{L1_CLUSTER_LO:.1e}, {L1_CLUSTER_HI:.1e}]")
    ax.axhline(PASS_MULTIPLIER * L1_CLUSTER_HI, color="tab:red", ls="--",
               alpha=0.7,
               label=f"PASS = 30x L1_HI = {PASS_MULTIPLIER*L1_CLUSTER_HI:.1e}")
    ax.axhspan(BOERA_PF_VAR_LO, BOERA_PF_VAR_HI, alpha=0.08, color="tab:green",
               label=f"Boera+2019 z=0.3 band [{BOERA_PF_VAR_LO:.0e}, "
                     f"{BOERA_PF_VAR_HI:.0e}]")
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
    ax.set_ylabel(r"$\mathrm{Var}_{\mathrm{sightlines}}[P_F(k_\parallel)]$")
    ax.set_title(f"[D-62/3 REDO] fGPA flux-power variance vs L1 collapse basin\n"
                 f"P{args.physics_id} z={args.redshift:.3f}, n_rays={n_rays}, "
                 f"tau_amp={tau_amp:.2e}, verdict="
                 f"{'PASS' if overall_pass else 'FAIL'}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"[D-62/3 REDO] Wrote overlay plot: {args.out_png}", flush=True)

    # ================ Numeric artifact ================
    payload = {
        "d_decision": "D-62/3 CPU pre-flight Stage 1 REDO per PI v9 [D-66]",
        "redo_fixes_applied": {
            "FixA_K1_tau_amp_prefactor": True,
            "FixB_K2_Boera_truth_anchor_assert": True,
            "FixC_S3_noise_delta_robustness": True,
        },
        "physics_id": args.physics_id,
        "redshift": args.redshift,
        "n_rays": n_rays,
        "n_obs": int(vel_axis.shape[0]),
        "dv_kms": dv_kms,
        "fgpa_beta": beta,
        "fgpa_gamma": gamma,
        "noise_seed": args.noise_seed,
        "tau_amp_calibrated": float(tau_amp),
        "mean_flux_target": args.mean_flux_target,
        "mean_F_truth": float(F_truth.mean().item()),
        "mean_F_fgpa_truth_delta": float(F_fgpa_td.mean().item()),
        "mean_F_fgpa_noise_delta": float(F_fgpa_nd.mean().item()),
        "k_inband_s_per_km": k_in.tolist(),
        "var_per_kbin_truth": var_per_k_truth.tolist(),
        "var_per_kbin_fgpa_truth_delta": var_per_k_fgpa_td.tolist(),
        "var_per_kbin_fgpa_noise_delta": var_per_k_fgpa_nd.tolist(),
        "var_band_mean_truth": var_mean_truth,
        "var_band_mean_fgpa_truth_delta": var_mean_fgpa_td,
        "var_band_mean_fgpa_noise_delta": var_mean_fgpa_nd,
        "boera2019_z03_PF_band_lo": BOERA_PF_VAR_LO,
        "boera2019_z03_PF_band_hi": BOERA_PF_VAR_HI,
        "L1_cluster_lo": L1_CLUSTER_LO,
        "L1_cluster_hi": L1_CLUSTER_HI,
        "pass_multiplier": PASS_MULTIPLIER,
        "ratio_fgpa_td_over_L1hi": float(ratio_td_vs_L1hi),
        "ratio_fgpa_nd_over_L1hi": float(ratio_nd_vs_L1hi),
        "log10_decades_above_L1hi_td": log10_dec_td,
        "log10_decades_above_L1hi_nd": log10_dec_nd,
        "gate_A_truth_delta_pass": pass_A,
        "gate_C_noise_delta_pass": pass_C,
        "gate_K2_boera_pass": True,  # would have raised AssertionError otherwise
        "verdict_overall": "PASS" if overall_pass else "FAIL",
        "stages_2_5_routing": (
            "PROCEED to Stage 2" if overall_pass
            else ("ABORT Stages 2-5; escalate to (2) per [D-62] ladder"
                  if not pass_C else "FAIL but Gate C passed (anomaly)")
        ),
    }
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[D-62/3 REDO] Wrote numeric artifact: {args.out_json}", flush=True)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

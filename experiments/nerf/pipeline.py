"""Stage 2b training entry-point for the NeRF track.

CLI-driven training loop with AdamW + warmup-cosine LR schedule, microbatched
gradient accumulation, mean-flux soft constraint ([D-11]), checkpointing with
RNG-state restoration, and MLflow tagging compatible with the 4x4 ablation
matrix ([D-12], [D-13]).

See LEDGER §3 (D-10..D-17) for every constant and design choice. This file is
the C1+C2+C3 deliverable of the Stage 2b dispatch.
"""

import argparse
import math
import os
import random
import sys

# Add src to path for imports
sys.path.append(os.path.abspath('.'))

# Force UTF-8 stdout so MLflow's run-link emoji doesn't trigger a cp949 codec
# error on Korean-locale Windows consoles (encountered during Stage 2a smoke).
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from src.data.loader import SherwoodLoader
from src.models.nerf import IGMNeRF, volume_render_physics

import mlflow
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 2b NeRF trainer.")

    # Matrix axes (required) -------------------------------------------------
    p.add_argument("--n_rays", type=int, required=True,
                   choices=[16384, 1024, 256, 64],
                   help="Sightline-density ablation axis [D-13].")
    p.add_argument("--physics", type=int, required=True,
                   choices=[1, 2, 3, 4],
                   help="Sherwood physics variant [D-12].")
    p.add_argument("--seed", type=int, required=True,
                   help="RNG seed; logged as MLflow tag.")

    # Memory / accumulation --------------------------------------------------
    p.add_argument("--microbatch", type=int, default=1024,
                   help="Rays per forward pass [D-14].")
    p.add_argument("--accum_steps", type=int, default=None,
                   help="Gradient accumulation factor; defaults to "
                        "ceil(n_rays/microbatch) per [D-14].")

    # Schedule ---------------------------------------------------------------
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--lr_max", type=float, default=5e-4)
    p.add_argument("--lr_min", type=float, default=5e-6)
    p.add_argument("--warmup_steps", type=int, default=1000)

    # Checkpointing ----------------------------------------------------------
    p.add_argument("--checkpoint_dir", type=str,
                   default="experiments/nerf/artifacts/checkpoints/")
    p.add_argument("--checkpoint_interval", type=int, default=5000)
    p.add_argument("--resume_from", type=str, default=None,
                   help="Optional path to a step_*.pt checkpoint to resume.")

    # Run identification -----------------------------------------------------
    p.add_argument("--run_name", type=str, default=None,
                   help="MLflow run name; auto-built if absent.")

    # Loss config ------------------------------------------------------------
    p.add_argument("--use_log_prior", action="store_true",
                   help="Retain the [D-10] generic Gaussian log-prior on "
                        "log(tau_amp). Off by default; on only for fiducial "
                        "comparison runs.")
    p.add_argument("--mean_flux_obs", type=float, default=0.877,
                   help="Observed mean flux <F> at z=0.3 [D-11].")
    p.add_argument("--lambda_F", type=float, default=1.0,
                   help="Weight on the mean-flux soft constraint [D-11].")
    p.add_argument("--tau_max", type=float, default=10.0,
                   help="Forest cap for the [D-24] log1p+cap+mask data loss. "
                        "Default 10.0 per [D-24] item (2). PI mandated a "
                        "sensitivity test at tau_max in {5, 10, 20}; this "
                        "flag exposes the cap as a CLI override so the "
                        "sweep can run without rebuilding the image. If the "
                        "sensitivity test exceeds 2%% in the [D-13] inertial "
                        "range, the cap is re-pinned with the measured anchor.")
    p.add_argument("--disable_dla_mask", action="store_true",
                   help="[ablation S5/S7] Force the [D-24] saturated-absorber "
                        "mask to all-True, i.e. include every bin in the data "
                        "MSE and mean-F surrogate. Reproduces the pre-[D-24] "
                        "supervision regime; expected to re-open the "
                        "cross-physics scale spread per the panel S7 attack. "
                        "The [D-21] two-pass gradient identity holds "
                        "regardless because the same mask tensor is reused "
                        "in Pass 1 and Pass 2 by construction.")

    # [D-39] saturation-aware P_F loss components ----------------------------
    # All three default to OFF (preserves cost-survey / pub-t1 byte-identical
    # behavior). They add ADDITIVELY on top of the [D-24] log1p data loss +
    # [D-11] mean-F soft constraint; they do NOT modify the [D-21] two-pass
    # mean-F gradient identity. See LEDGER §3 [D-39] Addendum 4 for the
    # mechanism-(c) saturation-regime hypothesis these flags are designed to
    # test. The sat-band/FGPA-tail PI design specs they implement are in
    # LEDGER §3 [D-40] (sat-aware) / [D-41] (FGPA-tail); both retired
    # empirically. See also ``papers/shared/sec/4_next_steps_main.tex``
    # §4.1 (Three retired counterfactual interventions).
    p.add_argument("--sat_band_weight", type=float, default=1.0,
                   help="[D-39] saturation-band data-MSE up-weight. Bins with "
                        "F_gt in [0.05, 0.30] (truth flux, computed once, "
                        "static under training) get this multiplicative weight "
                        "in the [D-24] log1p MSE; linear-regime bins keep "
                        "weight 1.0. Default 1.0 = OFF (uniform weights).")
    p.add_argument("--rank_order_weight", type=float, default=0.0,
                   help="[D-39] saturation-band rank-order penalty weight. "
                        "Adds a pairwise-margin Spearman surrogate over "
                        "saturation-band bins per sightline (see "
                        "_sat_band_rank_loss). Default 0.0 = OFF.")
    p.add_argument("--rank_order_pairs", type=int, default=64,
                   help="Random pairs per sightline for the rank-order term. "
                        "Higher = lower variance estimator; 64 is the smoke "
                        "default, 256 is the per-sightline-saturated "
                        "publication default.")
    p.add_argument("--pf_loss_weight", type=float, default=0.0,
                   help="[D-39] band-integrated P_F-residual weight. "
                        "Adds mean_sightlines ((P_F_pred - P_F_truth)/P_F_truth)^2 "
                        "evaluated on the inertial band k_|| in "
                        "[10^-2.5, 10^-1.5] s/km (Walther+ 2018 convention; "
                        "see src/analysis/flux_power_torch.py). The truth-side "
                        "PSD is cached once at startup since the GT flux is "
                        "static. Default 0.0 = OFF.")

    # [D-41] FGPA-tail regularizer: per-voxel physics prior on the
    # photoionized diffuse-IGM regime. tau_local = tau_amp * n_HI * dv /
    # (b * sqrt(pi)) should obey Hui-Gnedin 1997 scaling tau ~ Delta^beta T^gamma
    # at FGPA-valid voxels (tau_truth_local < fgpa_valid_tau_max). The truth-
    # anchored offset C is computed once at startup. Structurally immune to
    # the [D-40] amplitude-shrink degeneracy because C is frozen and the
    # residual is per-voxel.
    p.add_argument("--fgpa_tail_weight", type=float, default=0.0,
                   help="[D-41] FGPA-tail regularizer weight. Penalizes voxel-wise "
                        "deviation of log(tau_local) from beta*log(Delta) + gamma*log(T) + C "
                        "at FGPA-valid source bins. Huber loss in log-units. "
                        "Default 0.0 = OFF.")
    p.add_argument("--fgpa_beta", type=float, default=1.6,
                   help="[D-41] FGPA exponent on Delta. Hui-Gnedin 1997 default 1.6.")
    p.add_argument("--fgpa_gamma", type=float, default=-0.7,
                   help="[D-41] FGPA exponent on T. Hui-Gnedin 1997 default -0.7.")
    p.add_argument("--fgpa_valid_tau_max", type=float, default=0.5,
                   help="[D-41] FGPA-valid mask threshold: regularizer applies "
                        "only at source bins where tau_truth_local < this. "
                        "Default 0.5 (diffuse-IGM regime).")
    p.add_argument("--fgpa_huber_delta", type=float, default=0.5,
                   help="[D-41] Huber loss delta in log-units. Default 0.5.")

    # [D-42] velocity-gradient conditioning ---------------------------------
    p.add_argument("--use_velocity_gradient_conditioning",
                   action="store_true",
                   help="[D-42] velocity-gradient conditioning. Concatenates "
                        "Sherwood-truth dv_pec/dchi (z-scored, detached) as a "
                        "1D input feature to the density head. Disabled by "
                        "default. See experiments/nerf/LEDGER.md [D-42] for "
                        "the spec.")

    # [D-46] joint-physics conditional MLP with physics_id embedding ---------
    p.add_argument("--use_physics_embedding",
                   action="store_true",
                   help="[D-46] joint-physics conditional MLP. Loads all 4 "
                        "Sherwood physics variants (P1..P4) into a single "
                        "dataset, attaches a per-ray physics_id, and trains "
                        "the IGMNeRF with a learned nn.Embedding(4, 16) "
                        "concatenated onto the layer-1 input. Microbatches "
                        "are composed by interleaving (no physics-blocking) "
                        "so each gradient step mixes all 4 physics. When ON, "
                        "--physics is interpreted as a logging hint only — "
                        "the dataset draws from all 4. Disabled by default. "
                        "See experiments/nerf/LEDGER.md [D-46] for the spec.")
    p.add_argument("--lambda_inv", type=float, default=0.0,
                   help="[D-46] Tier-1 conditional cross-physics invariance "
                        "penalty weight. Per spec, code path is stubbed and "
                        "disabled (0.0) at smoke; wired forward if Tier-1 "
                        "lands. Default 0.0 = OFF.")

    # Sprint-L1 (direct P_F MSE loss; design v2 §2) -------------------------
    # OPT-IN flag — default OFF preserves bit-identical [D-24] behavior on
    # every existing run. When ON, the training step computes the predicted
    # flux F = exp(-cap(tau_pred, tau_max)), runs the differentiable torch
    # P_F estimator (src/training/p_flux_loss.py), evaluates the log-MSE loss
    # over the [D-13] inertial range, and combines it with the [D-24]
    # tau-MSE loss via GradNorm (Chen+ 2018, alpha=0.12). The 5 retire
    # conditions R-a..R-h are evaluated per step; on trigger the process
    # logs retire-reason and exits 0 (PCV-pattern per [D-37]-Ext rule-7).
    p.add_argument("--enable-l1-pf-loss", dest="enable_l1_pf_loss",
                   action="store_true",
                   help="[sprint-L1] Enable the direct P_F MSE loss test "
                        "(log-MSE over [D-13] inertial range, GradNorm-balanced "
                        "with [D-24] tau-MSE). Default OFF.")
    p.add_argument("--l1-gradnorm-alpha", dest="l1_gradnorm_alpha",
                   type=float, default=0.12,
                   help="[sprint-L1] GradNorm alpha (Chen+ 2018 default 0.12).")
    p.add_argument("--l1-burnin-tau-mse", dest="l1_burnin_tau_mse",
                   type=int, default=1000,
                   help="[sprint-L1] Burn-in before R-c (val tau-MSE) retire check.")
    p.add_argument("--l1-burnin-var-f", dest="l1_burnin_var_f",
                   type=int, default=500,
                   help="[sprint-L1] Burn-in before R-d (Var(F_pred)) retire check.")
    p.add_argument("--l1-d24-baseline-tau-mse", dest="l1_d24_baseline_tau_mse",
                   type=float, default=float("inf"),
                   help="[sprint-L1] [D-24] baseline val tau-MSE; used for R-c "
                        "(>2.0x ratio retire). Default inf disables R-c so the "
                        "smoke runs without an anchor. Production runs pass "
                        "the measured baseline.")
    p.add_argument("--l1-retire-dir", dest="l1_retire_dir", type=str,
                   default=None,
                   help="[sprint-L1] Directory to drop retire.json on a "
                        "retire-trigger. Defaults to checkpoint_dir.")
    p.add_argument("--gradnorm-full", dest="gradnorm_full",
                   action="store_true",
                   help="[sprint-L1 gate-8 Option A(b) rescue] Instantiate the "
                        "GradNormWrapper with simplified=False (full Chen+ 2018 "
                        "second-order autograd path). Default OFF — the "
                        "simplified loss-magnitude proxy (G_i = w_i * |L_i|) is "
                        "the NON-PROVISIONAL default per job 201587. This flag "
                        "is the fallback knob added after the proxy was "
                        "falsified at T3 scale (gradnorm_runaway retire at "
                        "step 2271, w_ratio=0.000867).")

    # Data root --------------------------------------------------------------
    p.add_argument("--data_root", type=str, default="Sherwood")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_global_seed(seed: int):
    """Seed every RNG that the training loop touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lr_lambda(warmup_steps: int, max_steps: int,
                    lr_max: float, lr_min: float):
    """Linear warmup 0 -> lr_max, then cosine decay lr_max -> lr_min."""
    decay_steps = max(1, max_steps - warmup_steps)
    min_ratio = lr_min / lr_max if lr_max > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup. Returns multiplier on lr_max (the AdamW base lr).
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


# ---------------------------------------------------------------------------
# [D-39] saturation-aware loss helpers
# ---------------------------------------------------------------------------

def _sat_band_rank_loss(
    tau_pred,
    tau_gt,
    sat_mask,
    *,
    n_pairs,
    generator=None,
):
    """Pairwise-margin rank-order penalty over saturation-band bins.

    For each sightline we sample ``n_pairs`` index pairs from the bins flagged
    in ``sat_mask`` and penalize pairs whose predicted ordering disagrees with
    the truth ordering:

        L_pair = mean_pairs( relu( -(tau_pred_i - tau_pred_j)
                                   * sign(tau_gt_i  - tau_gt_j) ) )

    This is the pairwise-margin surrogate to Spearman's $\\rho$ — it is
    differentiable in ``tau_pred`` (the only learnable side) and has no
    ``argsort`` / soft-rank approximation knobs. Pairs whose GT values are
    equal contribute zero gradient (``sign = 0``) so the loss is well-defined
    even when the band collapses.

    Sightlines with fewer than 2 saturation-band bins contribute zero (their
    rank order is structurally undefined). The reduction is a mean over the
    valid sightlines so the loss scale is independent of how saturated the
    microbatch happens to be.

    Parameters
    ----------
    tau_pred, tau_gt : (n_rays, n_bins) tensor
        Predicted and truth optical depths. ``tau_pred`` must carry autograd.
    sat_mask : (n_rays, n_bins) bool tensor
        True wherever a bin is in the saturation band AND not DLA-cored
        (caller combines ``F_gt in [0.05, 0.30]`` with ``mask_no_dla``).
    n_pairs : int
        Random pairs per sightline.
    generator : torch.Generator, optional
        RNG for the pair sampling. Pass the run's generator for determinism.

    Returns
    -------
    loss : (,) tensor
        Scalar loss; zero (with autograd-compatible gradient zero) if no
        sightline has >=2 saturation-band bins.
    """
    n_rays, n_bins = tau_pred.shape
    device = tau_pred.device

    # Per-sightline saturation-band counts; sightlines with <2 are skipped.
    counts = sat_mask.sum(dim=1)  # (n_rays,)
    valid = counts >= 2
    if not bool(valid.any()):
        # Anchor to the autograd graph so backward() is well-defined.
        return tau_pred.sum() * 0.0

    # We sample pairs from the entire (n_bins,) index range and then mask
    # pairs where either endpoint is not in the saturation band. This avoids
    # the variable-length per-sightline indexing that would otherwise need a
    # Python loop. With n_pairs >> 1 and band fraction f, the expected useful
    # pairs per sightline is n_pairs * f^2.
    i_idx = torch.randint(
        0, n_bins, (n_rays, n_pairs), generator=generator, device=device
    )
    j_idx = torch.randint(
        0, n_bins, (n_rays, n_pairs), generator=generator, device=device
    )

    # Gather both endpoints, autograd-preserving on tau_pred.
    tp_i = torch.gather(tau_pred, 1, i_idx)
    tp_j = torch.gather(tau_pred, 1, j_idx)
    tg_i = torch.gather(tau_gt, 1, i_idx)
    tg_j = torch.gather(tau_gt, 1, j_idx)
    in_band_i = torch.gather(sat_mask, 1, i_idx)
    in_band_j = torch.gather(sat_mask, 1, j_idx)

    pair_mask = in_band_i & in_band_j & (i_idx != j_idx)  # (n_rays, n_pairs)
    sign_gt = torch.sign(tg_i - tg_j)  # 0 when tied
    diff_pred = tp_i - tp_j

    # Margin: positive when prediction disagrees with truth ordering.
    margin = torch.relu(-diff_pred * sign_gt)

    # Mask invalid pairs to zero (so they contribute neither magnitude nor
    # gradient). Reduce over (rays, pairs) jointly with a denominator that
    # counts only the valid pairs (clamped to avoid 0/0).
    pair_weight = pair_mask.to(margin.dtype) * (sign_gt != 0).to(margin.dtype)
    total = (margin * pair_weight).sum()
    denom = pair_weight.sum().clamp(min=1.0)
    return total / denom


def load_dataset(args):
    """Load sightlines or fall back to dummy data for smoke runs.

    Returns ``(coords, vel_axis, tau_gt_profile, mask_no_dla_profile, box_max,
    v_pec_grad_truth, physics_id_per_ray)`` as float32/bool/long tensors.
    Coords are already normalized to the unit cube.

    The ``mask_no_dla_profile`` (bool, ``True`` = include in loss/mean-flux
    reductions) is the [D-24] DLA exclusion mask emitted by the loader.

    ``v_pec_grad_truth`` is returned when ``--use_velocity_gradient_conditioning``
    is set (z-scored, shape ``(n_rays, n_bins)``, float32). Otherwise ``None``.

    ``physics_id_per_ray`` is a (n_rays,) long tensor of zero-indexed physics
    ids (P1->0, P2->1, P3->2, P4->3). When ``--use_physics_embedding`` is OFF,
    every entry equals ``args.physics - 1`` (passthrough only; not used in
    forward because the model is in flag-off mode). When ON, the function
    loads sightlines from ALL 4 physics, stacks them along the ray axis, and
    fills ``physics_id_per_ray`` with the per-ray label (per [D-46] spec).
    """
    if not os.path.exists(args.data_root):
        print(f"Warning: Data root {args.data_root} missing. Using dummy data.")
        box_max = 60000.0
        nbins_dummy = 256
        # Use a generator seeded from args.seed so dummy data is deterministic
        gen = torch.Generator().manual_seed(args.seed)
        coords = torch.rand(args.n_rays, nbins_dummy, 3, generator=gen)
        vel_axis = torch.linspace(0, 6000.0, nbins_dummy)
        tau_gt_profile = torch.rand(args.n_rays, nbins_dummy, generator=gen)
        # Synthetic data has no DLAs by construction; include every bin.
        mask_no_dla_profile = torch.ones_like(tau_gt_profile, dtype=torch.bool)
        # [D-42] dummy-data fallback: synthesize a z-scored N(0,1) grad
        # tensor of the right shape so the smoke path works on machines
        # without Sherwood. Deterministic per --seed.
        v_pec_grad_truth = None
        if args.use_velocity_gradient_conditioning:
            v_pec_grad_truth = torch.randn(
                args.n_rays, nbins_dummy, generator=gen, dtype=torch.float32,
            )
        # [D-46] dummy-data physics_id_per_ray. When the joint-physics flag
        # is ON we *may* still want a deterministic smoke without real
        # Sherwood data: assign physics_ids in 4 contiguous blocks so the
        # microbatch interleaver has 4 distinct labels to draw from.
        if args.use_physics_embedding:
            n_rays = args.n_rays
            # Equal-split blocks; tail rays land in P4 if n_rays % 4 != 0.
            block = max(1, n_rays // 4)
            physics_id_per_ray = torch.full((n_rays,), 3, dtype=torch.long)
            for p_idx in range(4):
                s = p_idx * block
                e = (p_idx + 1) * block if p_idx < 3 else n_rays
                physics_id_per_ray[s:e] = p_idx
        else:
            physics_id_per_ray = torch.full(
                (args.n_rays,), args.physics - 1, dtype=torch.long,
            )
        return (coords, vel_axis, tau_gt_profile, mask_no_dla_profile,
                box_max, v_pec_grad_truth, physics_id_per_ray)

    loader = SherwoodLoader(args.data_root)

    # [D-46] joint-physics path: load all 4 physics, stack, label per-ray.
    # Default-OFF preserves the pre-[D-46] single-physics behavior (and the
    # bit-equivalent regression contract — the model under flag-off is
    # constructed without nn.Embedding so it does not consume from the RNG
    # stream).
    if args.use_physics_embedding:
        n_rays = args.n_rays
        n_rays_per_physics = n_rays // 4
        if n_rays % 4 != 0:
            print(
                f"[warn] [D-46] --n_rays={n_rays} not divisible by 4; "
                f"truncating to {n_rays_per_physics * 4} rays "
                f"({n_rays_per_physics} per physics)."
            )
        coords_list, tau_list, mask_list = [], [], []
        v_grad_list = [] if args.use_velocity_gradient_conditioning else None
        physics_id_list = []
        box_max = None
        vel_axis = None
        for p_idx, physics_label in enumerate([1, 2, 3, 4]):
            sl = loader.load_sightlines(physics_label, 0.3)
            coords_raw = loader.get_world_coordinates(sl)
            if box_max is None:
                box_max = sl['header']['box_kpc_h']
            else:
                # All 4 physics share the 60 Mpc/h box; sanity-check.
                assert sl['header']['box_kpc_h'] == box_max, (
                    f"[D-46] physics {physics_label} box_kpc_h mismatch."
                )
            if vel_axis is None:
                vel_axis = torch.tensor(sl['vel_axis'], dtype=torch.float32)
            coords_p = torch.tensor(
                coords_raw[:n_rays_per_physics], dtype=torch.float32,
            ) / box_max
            tau_p = torch.tensor(
                sl['tau_h1'][:n_rays_per_physics], dtype=torch.float32,
            )
            mask_p = torch.tensor(
                sl['mask_no_dla'][:n_rays_per_physics], dtype=torch.bool,
            )
            coords_list.append(coords_p)
            tau_list.append(tau_p)
            mask_list.append(mask_p)
            physics_id_list.append(
                torch.full((coords_p.shape[0],), p_idx, dtype=torch.long)
            )
            if args.use_velocity_gradient_conditioning:
                v_grad_p = torch.tensor(
                    sl['v_pec_grad_truth'][:n_rays_per_physics],
                    dtype=torch.float32,
                )
                v_grad_list.append(v_grad_p)
            print(
                f"[D-46] loaded P{physics_label}: "
                f"{coords_p.shape[0]} rays x {coords_p.shape[1]} bins "
                f"(physics_id={p_idx}).",
                flush=True,
            )
        coords = torch.cat(coords_list, dim=0)
        tau_gt_profile = torch.cat(tau_list, dim=0)
        mask_no_dla_profile = torch.cat(mask_list, dim=0)
        physics_id_per_ray = torch.cat(physics_id_list, dim=0)
        v_pec_grad_truth = (
            torch.cat(v_grad_list, dim=0) if v_grad_list is not None else None
        )
        n_dla_bins = int((~mask_no_dla_profile).sum().item())
        n_total_bins = int(mask_no_dla_profile.numel())
        print(
            f"[D-46] joint dataset: {coords.shape[0]} rays x "
            f"{coords.shape[1]} bins (4 physics interleaved at microbatch "
            f"level). DLA mask: {n_dla_bins}/{n_total_bins} bins excluded "
            f"({100.0 * n_dla_bins / max(1, n_total_bins):.3f}%).",
            flush=True,
        )
        return (coords, vel_axis, tau_gt_profile, mask_no_dla_profile,
                box_max, v_pec_grad_truth, physics_id_per_ray)

    # Default-OFF path: single-physics, baseline behavior, byte-equivalent.
    sightlines = loader.load_sightlines(args.physics, 0.3)
    coords_raw = loader.get_world_coordinates(sightlines)

    box_max = sightlines['header']['box_kpc_h']
    print(f"Loaded {coords_raw.shape[0]} rays. Normalizing to box {box_max} kpc/h")

    n_rays = args.n_rays
    coords = torch.tensor(coords_raw[:n_rays], dtype=torch.float32) / box_max
    vel_axis = torch.tensor(sightlines['vel_axis'], dtype=torch.float32)
    tau_gt_profile = torch.tensor(sightlines['tau_h1'][:n_rays], dtype=torch.float32)
    # [D-24]: per-bin DLA exclusion mask from the loader. True = include.
    mask_no_dla_profile = torch.tensor(
        sightlines['mask_no_dla'][:n_rays], dtype=torch.bool,
    )

    # [D-42]: velocity-gradient sidecar. Loader already z-scored and validated
    # the tensor (post-zscore std in [0.9, 1.1] per _validate_data). The
    # gradient is truth-derived and detached; .requires_grad stays False.
    v_pec_grad_truth = None
    if args.use_velocity_gradient_conditioning:
        v_pec_grad_truth = torch.tensor(
            sightlines['v_pec_grad_truth'][:n_rays], dtype=torch.float32,
        )
        g_stats = sightlines['v_pec_grad_stats']
        print(
            f"[D-42] v_pec_grad_truth: shape={tuple(v_pec_grad_truth.shape)} "
            f"mean={float(v_pec_grad_truth.mean()):+.4f} "
            f"std={float(v_pec_grad_truth.std()):.4f} "
            f"(loader cache: mean={g_stats['mean']:+.4f} std={g_stats['std']:.4f} "
            f"dchi_mpc_h={g_stats['dchi_mpc_h']:.6f}).",
            flush=True,
        )

    n_dla_bins = int((~mask_no_dla_profile).sum().item())
    n_total_bins = int(mask_no_dla_profile.numel())
    print(f"Normalized coord range: [{coords.min().item():.4f}, "
          f"{coords.max().item():.4f}]")
    print(f"Run scope: {n_rays} rays x {coords.shape[1]} bins (full grid).")
    print(f"[D-24] DLA mask: {n_dla_bins}/{n_total_bins} bins excluded "
          f"({100.0 * n_dla_bins / max(1, n_total_bins):.3f}%).")
    # [D-46] passthrough: physics_id is logged-only when the flag is OFF.
    # The model is constructed without nn.Embedding so this tensor is never
    # passed to forward(); it's surfaced for downstream MLflow tagging only.
    physics_id_per_ray = torch.full(
        (coords.shape[0],), args.physics - 1, dtype=torch.long,
    )
    return (coords, vel_axis, tau_gt_profile, mask_no_dla_profile,
            box_max, v_pec_grad_truth, physics_id_per_ray)


def save_checkpoint(path, *, model, optimizer, scheduler, log_tau_amp,
                    step, mlflow_run_id):
    """Write a checkpoint with full RNG state for bit-identical resume."""
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "log_tau_amp": log_tau_amp.detach().clone(),
        "step": step,
        "mlflow_run_id": mlflow_run_id,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[checkpoint] step={step} -> {path}")


def load_checkpoint(path, *, model, optimizer, scheduler, log_tau_amp):
    """Restore model + optimizer + scheduler + log_tau_amp + RNG state.

    Returns ``(start_step, mlflow_run_id)``.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    with torch.no_grad():
        log_tau_amp.copy_(state["log_tau_amp"])

    rng = state["rng_state"]
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])

    start_step = int(state["step"])
    print(f"[resume] loaded {path} at step={start_step}")
    return start_step, state.get("mlflow_run_id")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_global_seed(args.seed)

    # Auto-fill accum_steps per [D-14] -------------------------------------
    if args.accum_steps is None:
        args.accum_steps = max(1, math.ceil(args.n_rays / args.microbatch))
    # The microbatched loop implicitly pads with full chunks; warn if the user
    # supplied a value that doesn't tile n_rays exactly.
    if args.accum_steps * args.microbatch < args.n_rays:
        print(f"[warn] accum_steps*microbatch ({args.accum_steps*args.microbatch}) "
              f"< n_rays ({args.n_rays}); tail rays will be skipped.")

    # Auto-build run name --------------------------------------------------
    if args.run_name is None:
        args.run_name = (
            f"Stage2b-Ablation-P{args.physics}-N{args.n_rays}-S{args.seed}"
        )

    # Device --------------------------------------------------------------
    # Without explicit device placement the entire training loop runs on CPU
    # even on a GPU instance — silent ~7-min hang surfaced on the B-2 cloud
    # smoke (33M MLP points/step on g5.xlarge's 4 vCPU). The renderer in
    # src/models/nerf.py already follows the input tensor's device, so moving
    # the model + parameters + dataset tensors here cascades correctly.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Dataset --------------------------------------------------------------
    (coords, vel_axis, tau_gt_profile, mask_no_dla_profile,
     box_max, v_pec_grad_profile, physics_id_per_ray) = load_dataset(args)
    coords = coords.to(device)
    vel_axis = vel_axis.to(device)
    tau_gt_profile = tau_gt_profile.to(device)
    # [D-24] mask: bool, True = include in loss + mean-flux reductions.
    mask_no_dla_profile = mask_no_dla_profile.to(device)
    # [D-42]: velocity-gradient feature is detached truth-side data, used as
    # MLP input only. No autograd link back to it.
    if v_pec_grad_profile is not None:
        v_pec_grad_profile = v_pec_grad_profile.to(device)
    # [D-46]: per-ray physics_id (zero-indexed). Used by the IGMNeRF forward
    # only when args.use_physics_embedding is True; otherwise it's a
    # passthrough tensor whose only consumer is the per-ray microbatch
    # composition logic (which falls back to a single contiguous block when
    # the flag is off).
    physics_id_per_ray = physics_id_per_ray.to(device)
    if args.disable_dla_mask:
        mask_no_dla_profile = torch.ones_like(mask_no_dla_profile, dtype=torch.bool)
        print(
            f"[ablation] --disable_dla_mask: mask forced all-True "
            f"(0/{mask_no_dla_profile.numel()} bins excluded).",
            flush=True,
        )

    # Available rays after potential truncation
    n_rays_actual = coords.shape[0]
    n_bins = coords.shape[1]

    # [D-39] saturation-aware loss precomputes ----------------------------
    # All three new components key off the truth-side flux F_gt = exp(-tau_gt).
    # Using the *truth* flux (not the moving prediction) for the saturation
    # band keeps the per-bin pixel weight schedule static across training,
    # which is what the PI spec calls for and what keeps the [D-21] two-pass
    # mean-F gradient identity intact (the schedule does not depend on
    # tau_pred). The truth-side P_F is also static, so we cache it once.
    tau_gt_for_band = tau_gt_profile.clamp_max(args.tau_max)
    F_gt_profile = torch.exp(-tau_gt_for_band)  # (n_rays_actual, n_bins)
    sat_band_profile = (
        (F_gt_profile > 0.05) & (F_gt_profile < 0.30) & mask_no_dla_profile
    )
    sat_band_active = (
        args.sat_band_weight != 1.0 or args.rank_order_weight > 0.0
    )
    n_sat_total = int(sat_band_profile.sum().item())
    n_mask_total = int(mask_no_dla_profile.sum().item())
    print(
        f"[D-39] saturation band: {n_sat_total}/{n_mask_total} non-DLA bins "
        f"in F_gt in (0.05, 0.30) "
        f"({100.0 * n_sat_total / max(1, n_mask_total):.3f}%).",
        flush=True,
    )

    # [sprint-L1] Truth-side flux cache for the direct P_F MSE loss path.
    # F_truth_l1 = exp(-cap(tau_truth, tau_max)) is the GROUND TRUTH whose
    # log-binned P_F the network's predicted flux must match. We cache the
    # tensor (NOT its P_F — the per-step loss recomputes the full estimator
    # so a graph-attached comparison on the matched bin grid is available).
    F_truth_l1 = None
    if args.enable_l1_pf_loss:
        with torch.no_grad():
            tau_truth_capped = tau_gt_profile.clamp_max(args.tau_max)
            F_truth_l1 = torch.exp(-tau_truth_capped)
        print(
            f"[sprint-L1] truth flux cache: shape={tuple(F_truth_l1.shape)} "
            f"mean={float(F_truth_l1.mean()):.4f} "
            f"Var={float(F_truth_l1.var()):.4e}",
            flush=True,
        )

    # Truth-side P_F band-mean cache (only when --pf_loss_weight > 0).
    # Static across training. We compute it once with the autograd-free Torch
    # path so the cache is on the right device and shares the FFT convention
    # with the prediction-side P_F.
    P_F_truth_band = None
    if args.pf_loss_weight > 0.0:
        from src.analysis.flux_power_torch import (
            compute_p_flux_torch, band_mean_inertial,
        )
        with torch.no_grad():
            dv_kms = float((vel_axis[1] - vel_axis[0]).item())
            k_axis_cached, psd_truth = compute_p_flux_torch(
                F_gt_profile, dv_kms,
            )
            P_F_truth_band = band_mean_inertial(psd_truth, k_axis_cached)
        # Sanity guard: a zero P_F bin would NaN-blow the relative residual.
        # In practice the [D-13] inertial band is well-populated, but we
        # surface this here rather than mid-loop.
        if not bool(torch.isfinite(P_F_truth_band).all()):
            raise RuntimeError(
                "[D-39] P_F truth-side cache is not finite; cannot use as "
                "denominator. Inspect tau_gt / vel_axis."
            )
        if bool((P_F_truth_band <= 0.0).any()):
            raise RuntimeError(
                "[D-39] P_F truth-side cache has non-positive entries; "
                "relative residual is undefined."
            )
        print(
            f"[D-39] P_F truth-side cache: n_rays={P_F_truth_band.shape[0]} "
            f"P_F_truth_band mean={P_F_truth_band.mean().item():.4e} "
            f"min={P_F_truth_band.min().item():.4e} "
            f"max={P_F_truth_band.max().item():.4e} s/km",
            flush=True,
        )

    # [D-41] FGPA-tail regularizer truth-anchor cache. We compute the truth-
    # side tau_local from the loader's per-bin (density, h1_frac, temp) fields,
    # build the FGPA-valid mask (tau_truth_local < fgpa_valid_tau_max), and
    # fit the offset C = mean(log(tau_truth_local) - beta*log(Delta) - gamma*log(T))
    # over the mask. C is FROZEN — this is what makes the regularizer immune
    # to the [D-40] amplitude-shrink degeneracy: a uniform pred-side shrink
    # cannot move C and therefore gets caught voxel-wise.
    fgpa_C = None
    fgpa_valid_mask = None
    fgpa_truth_density = None
    fgpa_truth_temp = None
    if args.fgpa_tail_weight > 0.0:
        # Re-load sightlines to get truth source-space (density, h1_frac, temp).
        # Cheap (~seconds); one-time startup cost. Dummy-data fallback path
        # is not supported (Sherwood-only).
        _fgpa_loader = SherwoodLoader(args.data_root)
        _fgpa_sl = _fgpa_loader.load_sightlines(args.physics, 0.3)
        _n_rays_fgpa = args.n_rays
        density_truth = torch.tensor(_fgpa_sl['density'][:_n_rays_fgpa], dtype=torch.float32, device=device)
        h1_frac_truth = torch.tensor(_fgpa_sl['h1_frac'][:_n_rays_fgpa], dtype=torch.float32, device=device)
        temp_truth = torch.tensor(_fgpa_sl['temp'][:_n_rays_fgpa], dtype=torch.float32, device=device)
        with torch.no_grad():
            n_hi_truth = density_truth * h1_frac_truth                    # (n_rays, n_src)
            b_truth = 12.85 * torch.sqrt(temp_truth / 10000.0)            # km/s
            sqrt_pi_t = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=torch.float32))
            dv_kms_t = float((vel_axis[1] - vel_axis[0]).item())
            # tau_amp_init = exp(log_tau_amp_init=0) = 1.0; same convention as
            # the renderer at step 0. The amp will drift; the --use_log_prior
            # guard keeps it near 1.0 to prevent the Huber tau_amp -> 0 escape.
            tau_truth_local = 1.0 * n_hi_truth * dv_kms_t / (b_truth * sqrt_pi_t)
            # FGPA-valid mask + numeric guards. log(0) is -inf; mask out zero
            # density / h1_frac / temp bins (rare but possible at extreme voids).
            finite_mask = (
                (tau_truth_local > 0)
                & (density_truth > 0)
                & (temp_truth > 0)
            )
            fgpa_valid_mask = finite_mask & (tau_truth_local < args.fgpa_valid_tau_max)
            log_tau_t = torch.log(tau_truth_local.clamp_min(1e-30))
            log_d_t = torch.log(density_truth.clamp_min(1e-30))
            log_T_t = torch.log(temp_truth.clamp_min(1e-30))
            fgpa_residual_unanchored = (
                log_tau_t - args.fgpa_beta * log_d_t - args.fgpa_gamma * log_T_t
            )
            n_valid = int(fgpa_valid_mask.sum().item())
            if n_valid == 0:
                raise RuntimeError(
                    "[D-41] FGPA-valid mask is empty; no source bins satisfy "
                    f"tau_truth_local < {args.fgpa_valid_tau_max}. Check anchor "
                    "or raise --fgpa_valid_tau_max."
                )
            fgpa_C = fgpa_residual_unanchored[fgpa_valid_mask].mean().item()
            # Persist truth-side density/temp at source-bin grid for the loss
            # term (the network's own density/temp at predict time will be
            # different; the C anchor only uses truth-side values).
            fgpa_truth_density = density_truth   # (n_rays, n_src)
            fgpa_truth_temp = temp_truth
        n_total_src = int(fgpa_valid_mask.numel())
        print(
            f"[D-41] FGPA-valid mask: {n_valid}/{n_total_src} source bins "
            f"({100.0 * n_valid / max(1, n_total_src):.3f}%); "
            f"C={fgpa_C:.4f} log-units (beta={args.fgpa_beta}, gamma={args.fgpa_gamma}, "
            f"tau_max={args.fgpa_valid_tau_max}).",
            flush=True,
        )

    # Model ---------------------------------------------------------------
    # [D-46] use_physics_embedding gates the learned nn.Embedding(4, 16) head.
    # Default-OFF preserves byte-equivalent baseline: the embedding is not
    # instantiated and does not consume from the RNG stream, so layer init
    # is unchanged from the pre-[D-46] code.
    model = IGMNeRF(
        hidden_dim=256, num_layers=8, L=10,
        use_velocity_gradient_conditioning=args.use_velocity_gradient_conditioning,
        use_physics_embedding=args.use_physics_embedding,
    ).to(device)
    log_tau_amp = torch.nn.Parameter(torch.tensor(0.0, device=device))
    sigma_log = 0.5
    tau_amp_prior_weight = 1e-3

    params = list(model.parameters()) + [log_tau_amp]
    optimizer = optim.AdamW(
        params,
        lr=args.lr_max,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )

    # [sprint-L1] GradNorm wrapper + separate optimizer for w_tau / w_pf.
    # Default-OFF path: l1_gn / l1_gn_opt remain None and nothing in the
    # training loop changes. ON path: wrapper holds the 2 task weights, a
    # small Adam optimizer at lr=1e-3 updates them, separate backward pass
    # per Chen+ 2018 Algorithm 1.
    l1_gn = None
    l1_gn_opt = None
    if args.enable_l1_pf_loss:
        from src.training.p_flux_loss import GradNormWrapper
        # Default: simplified=True (G_i = w_i * |L_i|, loss-magnitude proxy) to
        # avoid the second-order ``torch.autograd.grad(create_graph=True)``
        # path through ``volume_render_physics``. The double-backward path
        # is the technically correct Chen+ 2018 formulation but it segfaults
        # on Windows CPU pytorch in practice (host smoke 2026-05-16 returned
        # exit code 0xC0000005 before step 1). The loss-magnitude proxy is
        # a known practical approximation that preserves the GradNorm
        # balance dynamics under well-conditioned per-task loss scales.
        #
        # [gate-8 Option A(b) rescue] --gradnorm-full opts back into the full
        # second-order Chen+ 2018 path. Triggered after job 201587 retired at
        # step 2271 on R-g gradnorm_runaway (w_ratio=0.000867, w_pf saturated
        # at 1.998), falsifying the simplified proxy at T3 scale per Chen+ 2018
        # §3 motivation. Default remains simplified=True (NON-PROVISIONAL).
        simplified_flag = not bool(args.gradnorm_full)
        l1_gn = GradNormWrapper(
            initial_w=(1.0, 1.0),
            alpha=args.l1_gradnorm_alpha,
            simplified=simplified_flag,
        ).to(device)
        l1_gn_opt = torch.optim.Adam(l1_gn.parameters(), lr=1e-3)
        print(
            f"[sprint-L1] GradNorm wrapper active (alpha={args.l1_gradnorm_alpha}, "
            f"simplified={simplified_flag}); w_tau=w_pf=1.0 at init.",
            flush=True,
        )
    lr_lambda = build_lr_lambda(
        args.warmup_steps, args.max_steps, args.lr_max, args.lr_min,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # Note: pre-[D-24] used torch.nn.MSELoss() for the data term. The [D-24]
    # log1p MSE is now computed inline below to apply the DLA mask correctly.

    # Resume --------------------------------------------------------------
    start_step = 0
    resume_run_id = None
    if args.resume_from:
        start_step, resume_run_id = load_checkpoint(
            args.resume_from,
            model=model, optimizer=optimizer, scheduler=scheduler,
            log_tau_amp=log_tau_amp,
        )

    # MLflow --------------------------------------------------------------
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow_active = True
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("CosmoGasVision/NeRF")
        print(f"Connected to MLflow at {mlflow_uri} [Experiment: CosmoGasVision/NeRF]")
    except Exception as e:
        print(f"MLflow connection issue: {e}")
        mlflow_active = False

    # Open run (fresh or resumed) -----------------------------------------
    if resume_run_id is not None and mlflow_active:
        run_ctx = mlflow.start_run(run_id=resume_run_id)
    elif mlflow_active:
        run_ctx = mlflow.start_run(run_name=args.run_name)
    else:
        from contextlib import nullcontext
        run_ctx = nullcontext()

    with run_ctx:
        if mlflow_active:
            mlflow.set_tags({
                "model_type": "nerf",
                "stage": "2b",
                # [D-46] When the joint-physics flag is ON, physics_id is no
                # longer a single scalar per run — the dataset contains all 4.
                # We tag the run as "P-mixed" so downstream slicing recognizes
                # this and falls back to the per-cell physics decomposition
                # exposed elsewhere (e.g., 4-cell pub-t1 eval).
                "physics_id": (
                    "P-mixed" if args.use_physics_embedding else str(args.physics)
                ),
                "redshift": "0.3",
                "n_rays": str(args.n_rays),
                "seed": str(args.seed),
                "ablation_matrix": "stage2b-4x4",
                # [D-42] tag — "on" when the velocity-gradient feature is
                # concatenated to the MLP input, "off" otherwise. Read by
                # downstream eval scripts to slice the run table.
                "velocity_gradient_conditioning": (
                    "on" if args.use_velocity_gradient_conditioning else "off"
                ),
                # [D-46] tag set: --feature physics_embedding when ON; the
                # lambda_inv weight is logged as a param below.
                "feature": (
                    "physics_embedding"
                    if args.use_physics_embedding else "baseline"
                ),
                # [sprint-L1] loss-variant tag — distinguishes L1 runs from
                # the [D-24] baseline + [D-39] band-residual runs.
                "loss_variant": (
                    "L1_direct_pf" if args.enable_l1_pf_loss else "D24_baseline"
                ),
            })
            if resume_run_id is None:
                mlflow.log_params({
                    "num_layers": 8,
                    "hidden_dim": 256,
                    "L_fourier": 10,
                    "n_rays": args.n_rays,
                    "n_bins": n_bins,
                    "microbatch": args.microbatch,
                    "accum_steps": args.accum_steps,
                    "lr_max": args.lr_max,
                    "lr_min": args.lr_min,
                    "warmup_steps": args.warmup_steps,
                    "max_steps": args.max_steps,
                    "weight_decay": 1e-6,
                    "grad_clip": 1.0,
                    "mean_flux_obs": args.mean_flux_obs,
                    "lambda_F": args.lambda_F,
                    "use_log_prior": args.use_log_prior,
                    "log_tau_amp_sigma": sigma_log,
                    "tau_amp_prior_weight": tau_amp_prior_weight,
                    "tau_max": args.tau_max,
                    "disable_dla_mask": args.disable_dla_mask,
                    "sat_band_weight": args.sat_band_weight,
                    "rank_order_weight": args.rank_order_weight,
                    "rank_order_pairs": args.rank_order_pairs,
                    "pf_loss_weight": args.pf_loss_weight,
                    "use_velocity_gradient_conditioning": (
                        args.use_velocity_gradient_conditioning
                    ),
                    # [D-46] joint-physics conditional MLP parameters.
                    "use_physics_embedding": args.use_physics_embedding,
                    "lambda_inv": args.lambda_inv,
                    "loss_form": (
                        ("log1p_mse_capped" if args.tau_max < 1e8 else "log1p_mse_uncapped")
                        + ("_masked" if not args.disable_dla_mask else "_unmasked")
                        + " + meanF_soft"
                        + ("_masked" if not args.disable_dla_mask else "_unmasked")
                        + (" + log_tau_amp_prior" if args.use_log_prior else "")
                        + (f" + sat_band(w={args.sat_band_weight})"
                           if args.sat_band_weight != 1.0 else "")
                        + (f" + rank_order(w={args.rank_order_weight},"
                           f"p={args.rank_order_pairs})"
                           if args.rank_order_weight > 0.0 else "")
                        + (f" + pf_band(w={args.pf_loss_weight})"
                           if args.pf_loss_weight > 0.0 else "")
                    ),
                })

        active_run_id = (mlflow.active_run().info.run_id
                         if mlflow_active else None)
        print(f"Run id: {active_run_id}", flush=True)
        print(f"Model: {sum(p.numel() for p in model.parameters())} params + "
              f"log_tau_amp scalar.")

        # [D-39] rank-order pair-sampling RNG. Seeded off args.seed so the
        # sequence is deterministic per (seed, step). Lives on the training
        # device so the randint calls match the data tensors.
        rank_rng = None
        if args.rank_order_weight > 0.0:
            rank_rng = torch.Generator(device=device)
            rank_rng.manual_seed(args.seed + 9173)

        # [D-46] microbatch composition RNG. Used only when the joint-physics
        # path is active. Seeded deterministically off args.seed so two runs
        # at the same seed produce the same interleaved index sequence.
        d46_rng = None
        d46_per_physics_indices = None
        if args.use_physics_embedding:
            d46_rng = torch.Generator(device=device)
            d46_rng.manual_seed(args.seed + 4646)
            # Pre-bucket ray indices by physics_id so the per-step interleaver
            # only needs to draw 16 indices per bucket. This assumes (and the
            # loader guarantees) that physics_id_per_ray is contiguous in
            # blocks of n_rays_per_physics, but we don't rely on it — we
            # derive the buckets from the tensor itself.
            d46_per_physics_indices = [
                torch.nonzero(physics_id_per_ray == p_idx, as_tuple=True)[0]
                for p_idx in range(4)
            ]

        # Helper: produce iterable of (ray_indices, mb_size) microbatches.
        # In the default-OFF path this is byte-equivalent to the prior
        # contiguous (start, end) iteration — torch.arange(s, e) is the
        # same as the slice `s:e` on every consumer below.
        # In the [D-46] flag-ON path this yields interleaved indices: each
        # microbatch contains microbatch//4 rays drawn (with replacement
        # across steps; without within-microbatch) from each of P1/P2/P3/P4
        # so the gradient step mixes all 4 physics every step.
        def microbatch_index_iter():
            if args.use_physics_embedding:
                per_physics_quota = args.microbatch // 4
                for chunk_i in range(args.accum_steps):
                    chunks = []
                    for p_idx in range(4):
                        pool = d46_per_physics_indices[p_idx]
                        if pool.numel() == 0:
                            continue
                        # Random subset with replacement across the dataset
                        # cycle (NOT within a microbatch — we draw distinct
                        # indices per draw via torch.randperm on the pool).
                        perm = torch.randperm(
                            pool.numel(), generator=d46_rng, device=device,
                        )[:per_physics_quota]
                        chunks.append(pool[perm])
                    if not chunks:
                        return
                    idx = torch.cat(chunks, dim=0)
                    yield idx, idx.numel()
            else:
                for chunk_i in range(args.accum_steps):
                    s = chunk_i * args.microbatch
                    e = min(s + args.microbatch, n_rays_actual)
                    if s >= e:
                        return
                    idx = torch.arange(s, e, device=device)
                    yield idx, idx.numel()

        # Training loop ----------------------------------------------------
        for step in range(start_step + 1, args.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            # Pass 1 runs under no_grad, so a single tau_amp tensor is fine
            # here. Pass 2 must recompute tau_amp inside the chunk loop so
            # each microbatch's backward owns its own autograd subgraph
            # rooted at log_tau_amp (otherwise chunk 1's backward would try
            # to traverse the torch.exp node that chunk 0 already freed —
            # the latent D-14 bug surfaced by the B-2 cloud smoke).
            tau_amp = torch.exp(log_tau_amp)

            # ---- Pass 1: compute the cycle mean of exp(-tau) (no grad). ----
            # The mean-flux soft constraint per [D-11] is a global anchor over
            # all rays * bins in the accumulation cycle. We need its current
            # value to linearize the squared loss for the per-microbatch
            # gradient pass. Pass 1 is grad-free, so memory stays bounded.
            #
            # [D-24]: reduce only over non-DLA bins. The mask is constant per
            # microbatch (a per-bin attribute of the GT data, independent of
            # tau_pred), so the [D-21] chain-rule identity still holds — the
            # only change is that F_cycle is now the *masked* cycle mean.
            # [D-46] Cache the per-step microbatch index list so Pass 1 and
            # Pass 2 see the exact same composition. Without this, Pass 1
            # would compute mean_F over one interleaved draw and Pass 2 would
            # backward against a different draw — the [D-21] chain-rule
            # identity would no longer hold. We materialize the iterator once
            # per step and reuse it in both passes.
            microbatch_indices = list(microbatch_index_iter())

            with torch.no_grad():
                weighted_F_sum = 0.0
                total_F_count = 0
                for idx_mb, mb_size in microbatch_indices:
                    # [D-42]: slice the per-(ray, bin) gradient feature in lockstep
                    # with `coords`. Loader-side data is 1:1 aligned with the
                    # source-bin grid (same n_bins, same ray ordering) so direct
                    # index gather is correct — no interpolation needed.
                    g_mb = None
                    if v_pec_grad_profile is not None:
                        g_mb = v_pec_grad_profile[idx_mb].unsqueeze(-1)  # (mb, n_bins, 1)
                    # [D-46]: pass physics_id per-microbatch only when the
                    # embedding flag is on — otherwise the model rejects it.
                    pid_mb = (
                        physics_id_per_ray[idx_mb]
                        if args.use_physics_embedding else None
                    )
                    tau_pred_mb = volume_render_physics(
                        model, coords[idx_mb], vel_axis=vel_axis, tau_amp=tau_amp,
                        g=g_mb, physics_id=pid_mb,
                    )
                    mask_mb = mask_no_dla_profile[idx_mb]
                    F_pred_mb = torch.exp(-tau_pred_mb)
                    weighted_F_sum += (F_pred_mb * mask_mb).sum().item()
                    total_F_count += int(mask_mb.sum().item())
                mean_F_pred_val = weighted_F_sum / max(1, total_F_count)
            # Linearization coefficient: d/dF [lambda_F * (F - T)^2]
            #     = 2 * lambda_F * (F_cycle - T)
            # Surrogate per-microbatch loss to inject this gradient is just
            # `c * mean_F_mb`; backwarding it produces the same gradient as
            # the true squared loss at the linearization point F_cycle.
            mean_F_grad_coef = 2.0 * args.lambda_F * (
                mean_F_pred_val - args.mean_flux_obs
            )

            # ---- Pass 2: per-microbatch combined backward pass. ----
            # For each microbatch we backward
            #     (loss_data_mb + c * mean_F_mb) * (1 / accum_steps)
            # plus, on the *first* microbatch only, the optional log-prior
            # (it has no microbatch dependence). One backward per microbatch
            # frees the graph immediately, keeping peak memory at one chunk.
            # [D-24] forest cap: optical depths above this are numerically
            # saturated (F = exp(-tau) is indistinguishable from zero) and
            # the loss should not chase exact tau values in that regime.
            # Default 10.0 per [D-24]; CLI-configurable via --tau_max so the
            # sensitivity sweep at tau_max in {5, 10, 20} can run without
            # rebuilding the image (per PI cost-survey verdict, 2026-05-04).
            TAU_MAX = args.tau_max

            data_loss_chunks = []
            sat_band_loss_chunks = []
            rank_order_loss_chunks = []
            pf_loss_chunks = []
            fgpa_loss_chunks = []
            # [sprint-L1] per-chunk accumulators. L1 path computes a single
            # log-MSE loss over the whole step (all chunks pooled) — but
            # within the existing chunked-backward loop we need the per-chunk
            # F_pred / F_truth slices to feed the estimator. We accumulate
            # both detached scalars (for logging) and the live-graph loss
            # contributions (added to loss_mb so the standard backward
            # absorbs it). With GradNorm, the per-chunk loss contribution
            # is w_pf * loss_pf_mb where w_pf is the current task weight.
            l1_loss_chunks = []           # per-chunk pf log-MSE loss (detached for log)
            l1_F_pred_chunks = []         # for inertial_rel_residual / coherence diagnostics
            l1_F_truth_chunks = []
            l1_loss_pf_step = None        # accumulated graph-attached loss for GradNorm
            for chunk_i, (idx_mb, mb_size) in enumerate(microbatch_indices):
                # Recompute tau_amp per chunk so its torch.exp autograd node
                # is local to this microbatch's backward pass. Without this,
                # the second chunk's .backward() raises "Trying to backward
                # through the graph a second time" because chunk 0 already
                # freed the shared exp node. log_tau_amp.grad still
                # accumulates correctly (sum over chunks).
                tau_amp_chunk = torch.exp(log_tau_amp)
                # [D-42]: per-microbatch gather of the velocity-gradient feature
                # (direct ray index on the same source-bin grid; no resampling).
                g_mb = None
                if v_pec_grad_profile is not None:
                    g_mb = v_pec_grad_profile[idx_mb].unsqueeze(-1)  # (mb, n_bins, 1)
                # [D-46]: physics_id per-microbatch only when the embedding
                # flag is on (model rejects the kwarg otherwise).
                pid_mb = (
                    physics_id_per_ray[idx_mb]
                    if args.use_physics_embedding else None
                )
                # [D-41] expose per-source-bin (tau_local, density, temp)
                # only when the FGPA-tail regularizer is active. Default-OFF
                # path is byte-equivalent: same renderer, same return type.
                if args.fgpa_tail_weight > 0.0:
                    tau_pred_mb, fgpa_fields_mb = volume_render_physics(
                        model, coords[idx_mb], vel_axis=vel_axis, tau_amp=tau_amp_chunk,
                        return_tau_local=True, g=g_mb, physics_id=pid_mb,
                    )
                else:
                    tau_pred_mb = volume_render_physics(
                        model, coords[idx_mb], vel_axis=vel_axis, tau_amp=tau_amp_chunk,
                        g=g_mb, physics_id=pid_mb,
                    )
                    fgpa_fields_mb = None
                # ---- [D-24] data loss: log1p MSE, capped at TAU_MAX, masked. ----
                # log1p(tau) compresses the long Lyman-alpha tail; cap at
                # TAU_MAX = 10 so saturated bins don't dominate; mask out DLA
                # bins entirely. Masked-mean form keeps loss finite even on
                # the pathological microbatch where every bin is DLA-cored:
                # zero-weight bins contribute zero gradient, exactly the
                # supervision behavior PI specified.
                tau_pred_eff = tau_pred_mb.clamp_max(TAU_MAX)
                tau_gt_eff = tau_gt_profile[idx_mb].clamp_max(TAU_MAX)
                mask_mb = mask_no_dla_profile[idx_mb]   # True = include
                diff = torch.log1p(tau_pred_eff) - torch.log1p(tau_gt_eff)
                diff_sq = diff * diff

                # ---- [D-39] saturation-band up-weighting ----
                # Per-bin weight = 1.0 (linear regime) + (w - 1.0) * sat_mask
                # so when --sat_band_weight=1.0 the schedule reduces exactly
                # to the existing uniform-mask form (backward-compatible).
                # The schedule depends only on truth-side flux (cached at
                # startup), so it does NOT introduce any tau_pred dependence
                # in the weights -> [D-21] mean-F identity is unaffected.
                sat_mask_mb = sat_band_profile[idx_mb]   # (mb, n_bins) bool
                if args.sat_band_weight != 1.0:
                    weight_mb = mask_mb.to(diff_sq.dtype) + (
                        args.sat_band_weight - 1.0
                    ) * sat_mask_mb.to(diff_sq.dtype)
                else:
                    weight_mb = mask_mb.to(diff_sq.dtype)
                loss_data_mb = (
                    (diff_sq * weight_mb).sum() / weight_mb.sum().clamp(min=1.0)
                )

                # Detached, per-chunk saturation-band-only MSE for logging
                # (always computed for diagnosis; not added to the loss
                # unless sat_band_weight != 1.0, in which case its
                # contribution is already inside loss_data_mb).
                if sat_band_active:
                    sat_w = sat_mask_mb.to(diff_sq.dtype)
                    sat_band_loss_chunks.append(
                        (
                            (diff_sq.detach() * sat_w).sum()
                            / sat_w.sum().clamp(min=1.0)
                        )
                    )

                # ---- [D-24] mean-F surrogate: same uniform mask. ----
                # Mask is constant per microbatch (data attribute, not a
                # function of tau_pred), so the [D-21] gradient identity
                # ∂L_meanF/∂θ = 2 λ_F (F_cycle - F_obs) · ∂F_cycle/∂θ
                # still holds — F_cycle is the masked cycle mean using the
                # ORIGINAL DLA mask (NOT the saturation-band up-weighted
                # one). The sat-band weighting belongs to the data term
                # only; the mean-F anchor remains the uniform Lyα <F>.
                F_pred_mb = torch.exp(-tau_pred_mb)
                mean_F_mb = (F_pred_mb * mask_mb).sum() / mask_mb.sum().clamp(min=1)

                loss_mb = loss_data_mb + mean_F_grad_coef * mean_F_mb

                # ---- [sprint-L1] direct P_F MSE loss (design v2 §2) ----
                # Compute the per-chunk F_pred = exp(-cap(tau_pred, tau_max))
                # and the matching F_truth slice; the log-MSE loss is built
                # over the WHOLE step (not per-chunk) because the K1-absorbing
                # ray-averaging-inside-the-log semantic operates on all rays
                # in the batch. We therefore accumulate the F_pred / F_truth
                # tensors and emit a single graph-attached loss at the END of
                # the chunk loop. Per-chunk contribution to loss_mb here is
                # the SLICED log-MSE so backward flows through every chunk's
                # tau_pred; the full-batch loss is also computed for logging
                # (the two are equal in expectation, equal in practice when
                # the batch is divided into equal chunks).
                if args.enable_l1_pf_loss:
                    from src.training.p_flux_loss import pf_log_mse_loss as _l1_loss
                    tau_pred_capped = tau_pred_mb.clamp_max(args.tau_max)
                    F_pred_l1_mb = torch.exp(-tau_pred_capped)
                    F_truth_l1_mb = F_truth_l1[idx_mb]
                    loss_pf_mb = _l1_loss(
                        F_pred_l1_mb, F_truth_l1_mb, vel_axis,
                    )
                    # Accumulate detached F tensors for diagnostics.
                    l1_F_pred_chunks.append(F_pred_l1_mb.detach())
                    l1_F_truth_chunks.append(F_truth_l1_mb.detach())
                    l1_loss_chunks.append(loss_pf_mb.detach())
                    # The GradNorm wrapper combines the [D-24] tau-MSE loss
                    # (loss_data_mb) and the L1 P_F loss with task weights.
                    # We REPLACE the additive loss_mb assembly with the
                    # weighted combination so backward applies the current
                    # w_tau / w_pf scaling. The mean-F surrogate term keeps
                    # its [D-21] gradient identity (it is not a task in the
                    # GradNorm formulation — it's a soft constraint).
                    w_t, w_p = l1_gn.weights_clamped
                    # Detach the linearized mean-F gradient coef path; it's
                    # already a per-step constant. The L1 weighted recombo
                    # only re-weights the data + P_F task losses.
                    loss_mb = (
                        w_t * loss_data_mb
                        + w_p * loss_pf_mb
                        + mean_F_grad_coef * mean_F_mb
                    )
                    if l1_loss_pf_step is None:
                        l1_loss_pf_step = loss_pf_mb
                    else:
                        l1_loss_pf_step = l1_loss_pf_step + loss_pf_mb

                # ---- [D-39] rank-order penalty in the saturation band ----
                # Pairwise-margin Spearman surrogate; see _sat_band_rank_loss.
                # Skip the call entirely when the weight is 0 to avoid the
                # ~O(n_mb * n_pairs) gather overhead in the OFF default path.
                if args.rank_order_weight > 0.0:
                    loss_rank_mb = _sat_band_rank_loss(
                        tau_pred_mb,
                        tau_gt_profile[idx_mb],
                        sat_mask_mb,
                        n_pairs=args.rank_order_pairs,
                        generator=rank_rng,
                    )
                    loss_mb = loss_mb + args.rank_order_weight * loss_rank_mb
                    rank_order_loss_chunks.append(loss_rank_mb.detach())

                # ---- [D-39] band-integrated P_F residual ----
                # Path: tau_pred -> F_pred = exp(-tau_pred)
                #       -> torch.fft.rfft on the Hann-windowed contrast
                #       -> |.|^2 PSD -> mean over inertial-band k bins
                #       -> ((P_F_pred - P_F_truth) / P_F_truth)^2 mean over rays.
                # The whole path is autograd-live; the rfft node sits
                # downstream of the renderer's tau_pred so backward through
                # the FFT just adds to the existing tau_pred gradient. The
                # [D-21] mean-F identity is unaffected because this term
                # does not touch mean_F_grad_coef.
                if args.pf_loss_weight > 0.0:
                    from src.analysis.flux_power_torch import (
                        compute_p_flux_torch, band_mean_inertial,
                    )
                    dv_kms = float((vel_axis[1] - vel_axis[0]).item())
                    k_ax_mb, psd_pred_mb = compute_p_flux_torch(
                        F_pred_mb, dv_kms,
                    )
                    P_pred_band_mb = band_mean_inertial(psd_pred_mb, k_ax_mb)
                    P_truth_band_mb = P_F_truth_band[idx_mb]
                    rel_residual = (
                        (P_pred_band_mb - P_truth_band_mb) / P_truth_band_mb
                    )
                    loss_pf_mb = (rel_residual ** 2).mean()
                    loss_mb = loss_mb + args.pf_loss_weight * loss_pf_mb
                    pf_loss_chunks.append(loss_pf_mb.detach())

                # ---- [D-41] FGPA-tail regularizer ----
                # Per-source-bin Huber residual on
                #   r = log(tau_local) - beta*log(Delta) - gamma*log(T) - C
                # where (tau_local, Delta, T) come from the network's forward
                # pass (fgpa_fields_mb) and C is the truth-anchored offset
                # cached at startup. The FGPA-valid mask was also computed
                # from truth at startup and is sliced by ray-index here. The
                # constraint is per-voxel and absolute (C is frozen), which
                # is the structural property that makes #1 immune to the
                # [D-40] amplitude-shrink degeneracy.
                if args.fgpa_tail_weight > 0.0:
                    tau_local_mb = fgpa_fields_mb["tau_local"]
                    density_pred_mb = fgpa_fields_mb["density"]
                    temp_pred_mb = fgpa_fields_mb["temp"]
                    # Numeric guards. Network outputs are bounded-physics
                    # (positive density, temp > 0 via softplus); clamp_min
                    # is belt-and-braces against random-init edge cases.
                    log_tau_p = torch.log(tau_local_mb.clamp_min(1e-30))
                    log_d_p = torch.log(density_pred_mb.clamp_min(1e-30))
                    log_T_p = torch.log(temp_pred_mb.clamp_min(1e-30))
                    r_mb = (
                        log_tau_p
                        - args.fgpa_beta * log_d_p
                        - args.fgpa_gamma * log_T_p
                        - fgpa_C
                    )
                    # Huber per element, then mask + mean.
                    fgpa_delta = args.fgpa_huber_delta
                    huber = torch.where(
                        r_mb.abs() < fgpa_delta,
                        0.5 * r_mb * r_mb,
                        fgpa_delta * (r_mb.abs() - 0.5 * fgpa_delta),
                    )
                    mask_fgpa_mb = fgpa_valid_mask[idx_mb]
                    if mask_fgpa_mb.any():
                        loss_fgpa_mb = (
                            (huber * mask_fgpa_mb.to(huber.dtype)).sum()
                            / mask_fgpa_mb.sum().clamp(min=1.0)
                        )
                    else:
                        # All bins in this microbatch are saturation-side
                        # (rare; happens on degenerate ray selections).
                        loss_fgpa_mb = torch.zeros((), dtype=huber.dtype, device=huber.device)
                    loss_mb = loss_mb + args.fgpa_tail_weight * loss_fgpa_mb
                    fgpa_loss_chunks.append(loss_fgpa_mb.detach())

                if chunk_i == 0 and args.use_log_prior:
                    loss_prior_term = (
                        tau_amp_prior_weight
                        * (log_tau_amp ** 2) / (2 * sigma_log ** 2)
                    )
                    loss_mb = loss_mb + loss_prior_term

                # [D-46] Tier-1 conditional invariance penalty stub.
                #
                #   L_inv = lambda_inv * sum_{p != p'} ||f_theta(x; e_p) -
                #                                       f_theta(x; e_p')||^2
                #
                # restricted to the saturated-mask bins, per LEDGER §3 [D-46]
                # Math contract. Default-OFF (lambda_inv=0.0) at smoke per
                # the spec's "Loss: unchanged from [D-24]/[D-11]/[D-21] base"
                # discipline constraint — we wire it forward only if Tier-1
                # lands and the smoke gates pass.
                if args.lambda_inv > 0.0:
                    raise NotImplementedError(
                        "[D-46] lambda_inv > 0 is a Tier-1 conditional code "
                        "path; the spec specifies this stays stubbed and "
                        "disabled at smoke. Re-enable after Tier-1 lands."
                    )

                (loss_mb / args.accum_steps).backward()
                data_loss_chunks.append(loss_data_mb.detach())

            # Loss values for logging (computed analytically from cycle mean)
            loss_meanF_val = args.lambda_F * (
                mean_F_pred_val - args.mean_flux_obs
            ) ** 2
            if args.use_log_prior:
                loss_prior = (log_tau_amp ** 2) / (2 * sigma_log ** 2)
            else:
                loss_prior = torch.tensor(0.0)

            # [sprint-L1] GradNorm task-weight update (simplified path).
            # The wrapper is in simplified=True mode (G_i = w_i * |L_i|);
            # no second-order autograd through the model. We feed the
            # already-accumulated per-step task losses (mean over chunks) as
            # detached scalars, which are autograd-LIVE in the w_t, w_p
            # parameters via the wrapper's internal multiply. The main
            # backward through the model has already been frozen at this
            # point so the GradNorm step does not perturb the model graph.
            l1_gn_metrics = {}
            if args.enable_l1_pf_loss and l1_gn is not None:
                try:
                    loss_tau_scalar = torch.tensor(
                        float(np.mean([c.item() for c in data_loss_chunks])),
                        dtype=torch.float32, device=device,
                    )
                    loss_pf_scalar = torch.tensor(
                        float(np.mean([c.item() for c in l1_loss_chunks])),
                        dtype=torch.float32, device=device,
                    ) if l1_loss_chunks else torch.tensor(0.0, device=device)
                    gn_loss = l1_gn.compute_gradnorm_loss(
                        loss_tau_scalar, loss_pf_scalar,
                        shared_params=[l1_gn.w_tau],  # placeholder; ignored in simplified path
                    )
                    l1_gn_opt.zero_grad()
                    gn_loss.backward()
                    l1_gn_opt.step()
                    l1_gn.renormalize_weights()
                except Exception as gn_e:
                    print(f"[sprint-L1] GradNorm update skipped: {gn_e}",
                          flush=True)
                if step == 100 and args.enable_l1_pf_loss and l1_gn is not None:
                    assert abs(l1_gn.w_tau.item() - 1.0) >= 0.01 and abs(l1_gn.w_pf.item() - 1.0) >= 0.01, (
                        f"[sprint-L1] GradNorm degeneracy contract violated at step 100: "
                        f"w_tau={l1_gn.w_tau.item():.6f}, w_pf={l1_gn.w_pf.item():.6f}. "
                        f"Weights pinned at init -> GradNorm dead. See gate-pilot-bug-prevention contract."
                    )
                l1_gn_metrics = {
                    "w_tau": float(l1_gn.w_tau.detach().item()),
                    "w_pf": float(l1_gn.w_pf.detach().item()),
                    "w_ratio": float(l1_gn.weight_ratio),
                }

            # Gradient clip + step
            grad_norm_clip = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            # Per-step metrics (post-accumulation) -----------------------
            loss_data = torch.stack(data_loss_chunks).mean().item()
            # [D-39] component diagnostics: per-chunk means -> step mean.
            # NaN-safe defaults when the component is OFF this run.
            loss_sat_band_val = (
                torch.stack(sat_band_loss_chunks).mean().item()
                if sat_band_loss_chunks else 0.0
            )
            loss_rank_order_val = (
                torch.stack(rank_order_loss_chunks).mean().item()
                if rank_order_loss_chunks else 0.0
            )
            loss_pf_band_val = (
                torch.stack(pf_loss_chunks).mean().item()
                if pf_loss_chunks else 0.0
            )
            loss_fgpa_val = (
                torch.stack(fgpa_loss_chunks).mean().item()
                if fgpa_loss_chunks else 0.0
            )
            loss_total = loss_data + loss_meanF_val + (
                tau_amp_prior_weight * loss_prior.item() if args.use_log_prior else 0.0
            )
            # Add the [D-39]/[D-41] components' weighted contribution to the
            # printed total. loss_data already absorbs the saturation-band
            # weighting (in-place re-weight of the per-bin schedule); the
            # rank-order, P_F, and FGPA-tail terms enter additively with
            # their CLI weights.
            loss_total = (
                loss_total
                + args.rank_order_weight * loss_rank_order_val
                + args.pf_loss_weight * loss_pf_band_val
                + args.fgpa_tail_weight * loss_fgpa_val
            )
            grad_norm = model.out_layer.weight.grad.norm().item() \
                if model.out_layer.weight.grad is not None else 0.0
            cur_lr = scheduler.get_last_lr()[0]

            if step <= 10 or step % 50 == 0 or step == args.max_steps:
                extras = ""
                if sat_band_active:
                    extras += f" satMSE={loss_sat_band_val:.4f}"
                if args.rank_order_weight > 0.0:
                    extras += f" rank={loss_rank_order_val:.4e}"
                if args.pf_loss_weight > 0.0:
                    extras += f" pf={loss_pf_band_val:.4e}"
                if args.fgpa_tail_weight > 0.0:
                    extras += f" fgpa={loss_fgpa_val:.4e}"
                print(
                    f"Step {step}/{args.max_steps} | loss={loss_total:.4f} "
                    f"(data={loss_data:.4f}, meanF={loss_meanF_val:.4e}, "
                    f"prior={loss_prior.item():.4f}){extras} | "
                    f"<F>={mean_F_pred_val:.4f} | grad={grad_norm:.4f} | "
                    f"clip={grad_norm_clip:.3f} | lr={cur_lr:.2e} | "
                    f"tau_amp={tau_amp.item():.4f}",
                    flush=True,
                )

            if mlflow_active:
                mlflow.log_metric("loss", loss_total, step=step)
                mlflow.log_metric("loss_data", loss_data, step=step)
                mlflow.log_metric("loss_meanF", loss_meanF_val, step=step)
                mlflow.log_metric("mean_flux_pred", mean_F_pred_val, step=step)
                mlflow.log_metric("grad_norm", grad_norm, step=step)
                mlflow.log_metric("grad_norm_clipped",
                                  float(grad_norm_clip), step=step)
                mlflow.log_metric("tau_amp", tau_amp.item(), step=step)
                mlflow.log_metric("lr", cur_lr, step=step)
                if args.use_log_prior:
                    mlflow.log_metric("loss_prior", loss_prior.item(), step=step)
                # [D-39] component-level metrics. Only logged when the
                # respective component is active so we don't pollute the
                # baseline-run MLflow scalar set.
                if sat_band_active:
                    mlflow.log_metric("loss_sat_band", loss_sat_band_val, step=step)
                if args.rank_order_weight > 0.0:
                    mlflow.log_metric("loss_rank_order", loss_rank_order_val, step=step)
                if args.pf_loss_weight > 0.0:
                    mlflow.log_metric("loss_pf_band", loss_pf_band_val, step=step)
                if args.fgpa_tail_weight > 0.0:
                    mlflow.log_metric("loss_fgpa_tail", loss_fgpa_val, step=step)

            # [sprint-L1] per-step metrics + retire-condition evaluation.
            if args.enable_l1_pf_loss:
                from src.training.p_flux_loss import (
                    inertial_rel_residual as _l1_irr,
                    cross_coherence_per_bin as _l1_coh,
                )
                # Pool detached F tensors across chunks for full-batch
                # diagnostic metrics. With per-physics microbatch composition
                # ([D-46] path) the pool sums up to the full step batch.
                with torch.no_grad():
                    F_pred_pool = torch.cat(l1_F_pred_chunks, dim=0)
                    F_truth_pool = torch.cat(l1_F_truth_chunks, dim=0)
                    irr = float(_l1_irr(F_pred_pool, F_truth_pool, vel_axis).item())
                    coh_per_bin = _l1_coh(F_pred_pool, F_truth_pool, vel_axis)
                    finite_coh = coh_per_bin[torch.isfinite(coh_per_bin)]
                    coh_median = (
                        float(finite_coh.median().item())
                        if finite_coh.numel() > 0 else float("nan")
                    )
                    n_inertial_above_0p5 = int(
                        (finite_coh >= 0.5).sum().item()
                    ) if finite_coh.numel() > 0 else 0
                    var_F_pred = float(F_pred_pool.var().item())
                    var_F_truth = float(F_truth_pool.var().item())
                    var_F_ratio = var_F_pred / max(var_F_truth, 1e-30)
                    # P_F-pred Var_k over inertial range (R-b backstop).
                    from src.training.p_flux_loss import (
                        torch_p_flux as _l1_pf,
                        K_MIN_INERTIAL as _K_MIN, K_MAX_INERTIAL as _K_MAX,
                    )
                    centers_l1, P_pred_l1 = _l1_pf(F_pred_pool, vel_axis)
                    _, P_truth_l1 = _l1_pf(F_truth_pool, vel_axis)
                    band_l1 = (centers_l1 >= _K_MIN) & (centers_l1 <= _K_MAX)
                    if bool(band_l1.any()):
                        Pp_ravg = P_pred_l1.to(torch.float64).mean(dim=0)[band_l1]
                        Pt_ravg = P_truth_l1.to(torch.float64).mean(dim=0)[band_l1]
                        var_pf_pred_band = float(Pp_ravg.var().item())
                        var_pf_truth_band = float(Pt_ravg.var().item())
                        var_pf_ratio = var_pf_pred_band / max(var_pf_truth_band, 1e-30)
                    else:
                        var_pf_ratio = float("nan")
                loss_pf_step_val = (
                    float(torch.stack(l1_loss_chunks).mean().item())
                    if l1_loss_chunks else 0.0
                )
                # Log per-step metrics.
                if mlflow_active:
                    mlflow.log_metric("loss_tau", loss_data, step=step)
                    mlflow.log_metric("loss_pf", loss_pf_step_val, step=step)
                    mlflow.log_metric("l1_inertial_rel_residual", irr, step=step)
                    mlflow.log_metric("l1_var_F_ratio", var_F_ratio, step=step)
                    mlflow.log_metric("l1_var_pf_band_ratio", var_pf_ratio, step=step)
                    mlflow.log_metric("l1_coh_median", coh_median, step=step)
                    mlflow.log_metric("l1_n_coh_above_0p5", n_inertial_above_0p5, step=step)
                    if l1_gn_metrics:
                        mlflow.log_metric("w_tau", l1_gn_metrics["w_tau"], step=step)
                        mlflow.log_metric("w_pf", l1_gn_metrics["w_pf"], step=step)
                        mlflow.log_metric("w_ratio", l1_gn_metrics["w_ratio"], step=step)

                # ---- Retire-condition checks (R-a..R-h per design v2 §4) ----
                retire_reason = None
                # R-a: loss NaN/Inf or training-loss divergence within 1k steps.
                if not np.isfinite(loss_total):
                    retire_reason = "R-a:loss_nan_or_inf"
                # R-b: P_F^pred collapses to flat-zero/flat-constant within
                # 5k steps. Apply a 200-step initialization burn-in so the
                # random-init period (where the network has not yet learned
                # any flux structure) does not trip the check.
                if (
                    retire_reason is None and 200 <= step <= 5000
                    and np.isfinite(var_pf_ratio) and var_pf_ratio < 0.1
                ):
                    retire_reason = "R-b:pf_pred_variance_collapse"
                # R-c: val tau-MSE > 2.0x [D-24] baseline. We use the
                # training-side loss_data as a proxy since we don't have a
                # separate val split mid-loop; CLI lets the user pin the
                # baseline anchor. R-c is the SHARED-RISK backstop with D1.
                if (
                    retire_reason is None
                    and step >= args.l1_burnin_tau_mse
                    and np.isfinite(args.l1_d24_baseline_tau_mse)
                    and loss_data > 2.0 * args.l1_d24_baseline_tau_mse
                ):
                    retire_reason = "R-c:tau_mse_doubled_vs_d24"
                # R-d: Var(F_pred) < 0.5x Var(F_truth) after burn-in 500.
                if (
                    retire_reason is None
                    and step >= args.l1_burnin_var_f
                    and var_F_ratio < 0.5
                ):
                    retire_reason = "R-d:flux_variance_collapse"
                # R-e: median coherence > 0.5 in < 4/6 inertial k-bins after 5k.
                # n_inertial_above_0p5 counts the bins (>=0.5); retire if <4 after 5k.
                if (
                    retire_reason is None and step > 5000
                    and finite_coh.numel() > 0
                    and n_inertial_above_0p5 < 4
                ):
                    retire_reason = "R-e:coherence_below_threshold"
                # R-g: GradNorm weight ratio exceeds 1000:1 either direction.
                if retire_reason is None and l1_gn_metrics:
                    wr = l1_gn_metrics["w_ratio"]
                    if wr > 1000.0 or wr < 1e-3:
                        retire_reason = "R-g:gradnorm_runaway"
                # R-f (step 25k) and R-h (wallclock) are NOT checked here —
                # R-f needs a 5k-step trailing slope window which is too
                # heavy for the per-step loop without a metric ring buffer;
                # the gate-6 Juno job picks it up via MLflow post-hoc.
                # R-h (wallclock) is owned by the dispatcher.

                if retire_reason is not None:
                    import json
                    retire_dir = args.l1_retire_dir or args.checkpoint_dir
                    os.makedirs(retire_dir, exist_ok=True)
                    retire_path = os.path.join(retire_dir, "retire.json")
                    retire_payload = {
                        "retire_reason": retire_reason,
                        "step": int(step),
                        "loss_total": float(loss_total),
                        "loss_tau": float(loss_data),
                        "loss_pf": float(loss_pf_step_val),
                        "inertial_rel_residual": float(irr),
                        "var_F_ratio": float(var_F_ratio),
                        "var_pf_band_ratio": float(var_pf_ratio),
                        "coh_median": float(coh_median),
                        "n_coh_above_0p5": int(n_inertial_above_0p5),
                        "w_tau": l1_gn_metrics.get("w_tau"),
                        "w_pf": l1_gn_metrics.get("w_pf"),
                        "w_ratio": l1_gn_metrics.get("w_ratio"),
                        "mlflow_run_id": active_run_id,
                    }
                    with open(retire_path, "w", encoding="utf-8") as fh:
                        json.dump(retire_payload, fh, indent=2)
                    if mlflow_active:
                        mlflow.set_tag("l1_retire_reason", retire_reason)
                        mlflow.log_metric("l1_retire_step", float(step), step=step)
                    print(
                        f"[sprint-L1] RETIRE @ step {step}: {retire_reason}. "
                        f"Wrote {retire_path}. Exiting 0 (PCV pattern).",
                        flush=True,
                    )
                    sys.exit(0)

            # Checkpoint ---------------------------------------------------
            if (args.checkpoint_interval > 0
                    and step % args.checkpoint_interval == 0):
                ckpt_path = os.path.join(
                    args.checkpoint_dir, f"step_{step:06d}.pt"
                )
                save_checkpoint(
                    ckpt_path,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    log_tau_amp=log_tau_amp, step=step,
                    mlflow_run_id=active_run_id,
                )

        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"Peak VRAM: {peak_vram_gb:.2f} GB", flush=True)
            if mlflow_active:
                mlflow.log_metric("peak_vram_gb", peak_vram_gb)

        print("Training finished.", flush=True)


def main(argv=None):
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()

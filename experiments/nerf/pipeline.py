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


def load_dataset(args):
    """Load sightlines or fall back to dummy data for smoke runs.

    Returns ``(coords, vel_axis, tau_gt_profile, mask_no_dla_profile, box_max)``
    as float32/bool tensors. The first axis of ``coords``, ``tau_gt_profile``,
    and ``mask_no_dla_profile`` has been truncated to ``args.n_rays``. Coords
    are already normalized to the unit cube.

    The ``mask_no_dla_profile`` (bool, ``True`` = include in loss/mean-flux
    reductions) is the [D-24] DLA exclusion mask emitted by the loader.
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
        return coords, vel_axis, tau_gt_profile, mask_no_dla_profile, box_max

    loader = SherwoodLoader(args.data_root)
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

    n_dla_bins = int((~mask_no_dla_profile).sum().item())
    n_total_bins = int(mask_no_dla_profile.numel())
    print(f"Normalized coord range: [{coords.min().item():.4f}, "
          f"{coords.max().item():.4f}]")
    print(f"Run scope: {n_rays} rays x {coords.shape[1]} bins (full grid).")
    print(f"[D-24] DLA mask: {n_dla_bins}/{n_total_bins} bins excluded "
          f"({100.0 * n_dla_bins / max(1, n_total_bins):.3f}%).")
    return coords, vel_axis, tau_gt_profile, mask_no_dla_profile, box_max


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
    coords, vel_axis, tau_gt_profile, mask_no_dla_profile, box_max = load_dataset(args)
    coords = coords.to(device)
    vel_axis = vel_axis.to(device)
    tau_gt_profile = tau_gt_profile.to(device)
    # [D-24] mask: bool, True = include in loss + mean-flux reductions.
    mask_no_dla_profile = mask_no_dla_profile.to(device)

    # Available rays after potential truncation
    n_rays_actual = coords.shape[0]
    n_bins = coords.shape[1]

    # Model ---------------------------------------------------------------
    model = IGMNeRF(hidden_dim=256, num_layers=8, L=10).to(device)
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
                "physics_id": str(args.physics),
                "redshift": "0.3",
                "n_rays": str(args.n_rays),
                "seed": str(args.seed),
                "ablation_matrix": "stage2b-4x4",
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
                    "loss_form": (
                        "log1p_mse_capped_masked + meanF_soft_masked"  # [D-24]
                        + (" + log_tau_amp_prior" if args.use_log_prior else "")
                    ),
                })

        active_run_id = (mlflow.active_run().info.run_id
                         if mlflow_active else None)
        print(f"Run id: {active_run_id}", flush=True)
        print(f"Model: {sum(p.numel() for p in model.parameters())} params + "
              f"log_tau_amp scalar.")

        # Helper: produce iterable of (start, end) microbatch slices.
        def microbatch_slices():
            for chunk_i in range(args.accum_steps):
                s = chunk_i * args.microbatch
                e = min(s + args.microbatch, n_rays_actual)
                if s >= e:
                    return
                yield s, e

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
            with torch.no_grad():
                weighted_F_sum = 0.0
                total_F_count = 0
                for s, e in microbatch_slices():
                    tau_pred_mb = volume_render_physics(
                        model, coords[s:e], vel_axis=vel_axis, tau_amp=tau_amp,
                    )
                    mask_mb = mask_no_dla_profile[s:e]
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
            # [D-24] Bolton+ 2017 forest cap: optical depths above this are
            # numerically saturated (F = exp(-tau) is indistinguishable from
            # zero) and the loss should not chase exact tau values in that
            # regime. Hard-coded; PI re-tunes via a new D-XX, not a CLI flag.
            TAU_MAX = 10.0

            data_loss_chunks = []
            for chunk_i, (s, e) in enumerate(microbatch_slices()):
                # Recompute tau_amp per chunk so its torch.exp autograd node
                # is local to this microbatch's backward pass. Without this,
                # the second chunk's .backward() raises "Trying to backward
                # through the graph a second time" because chunk 0 already
                # freed the shared exp node. log_tau_amp.grad still
                # accumulates correctly (sum over chunks).
                tau_amp_chunk = torch.exp(log_tau_amp)
                tau_pred_mb = volume_render_physics(
                    model, coords[s:e], vel_axis=vel_axis, tau_amp=tau_amp_chunk,
                )
                # ---- [D-24] data loss: log1p MSE, capped at TAU_MAX, masked. ----
                # log1p(tau) compresses the long Lyman-alpha tail; cap at
                # TAU_MAX = 10 so saturated bins don't dominate; mask out DLA
                # bins entirely. Masked-mean form keeps loss finite even on
                # the pathological microbatch where every bin is DLA-cored:
                # zero-weight bins contribute zero gradient, exactly the
                # supervision behavior PI specified.
                tau_pred_eff = tau_pred_mb.clamp_max(TAU_MAX)
                tau_gt_eff = tau_gt_profile[s:e].clamp_max(TAU_MAX)
                mask_mb = mask_no_dla_profile[s:e]   # True = include
                diff = torch.log1p(tau_pred_eff) - torch.log1p(tau_gt_eff)
                diff_sq = diff * diff
                loss_data_mb = (
                    (diff_sq * mask_mb).sum() / mask_mb.sum().clamp(min=1)
                )

                # ---- [D-24] mean-F surrogate: same masked reduction. ----
                # Mask is constant per microbatch (data attribute, not a
                # function of tau_pred), so the [D-21] gradient identity
                # ∂L_meanF/∂θ = 2 λ_F (F_cycle - F_obs) · ∂F_cycle/∂θ
                # still holds — F_cycle is now the masked cycle mean.
                F_pred_mb = torch.exp(-tau_pred_mb)
                mean_F_mb = (F_pred_mb * mask_mb).sum() / mask_mb.sum().clamp(min=1)

                loss_mb = loss_data_mb + mean_F_grad_coef * mean_F_mb
                if chunk_i == 0 and args.use_log_prior:
                    loss_prior_term = (
                        tau_amp_prior_weight
                        * (log_tau_amp ** 2) / (2 * sigma_log ** 2)
                    )
                    loss_mb = loss_mb + loss_prior_term

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

            # Gradient clip + step
            grad_norm_clip = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            # Per-step metrics (post-accumulation) -----------------------
            loss_data = torch.stack(data_loss_chunks).mean().item()
            loss_total = loss_data + loss_meanF_val + (
                tau_amp_prior_weight * loss_prior.item() if args.use_log_prior else 0.0
            )
            grad_norm = model.out_layer.weight.grad.norm().item() \
                if model.out_layer.weight.grad is not None else 0.0
            cur_lr = scheduler.get_last_lr()[0]

            if step <= 10 or step % 50 == 0 or step == args.max_steps:
                print(
                    f"Step {step}/{args.max_steps} | loss={loss_total:.4f} "
                    f"(data={loss_data:.4f}, meanF={loss_meanF_val:.4e}, "
                    f"prior={loss_prior.item():.4f}) | "
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

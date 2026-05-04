"""[D-24] item (2): tau_max sensitivity gate for cost-survey Batch 1b.

PI mandate (LEDGER §3 [D-24]): if max(|Delta P_F / P_F|) over k_|| in
[10^-2.5, 10^-1.5] s/km between tau_max=10 and {tau_max=5, tau_max=20}
exceeds 2%, BLOCK Batch 2 of the cost-survey, re-pin tau_max with the
measured anchor, and amend [D-24].

Compares three Stage 2b ablation runs at P1, n_rays=64, max_steps=50000,
identical except for the forest-cap tau_max:
    - Stage2b-Ablation-P1-N64-S0-1777912895-49b2cd  (tau_max=5)
    - Stage2b-Ablation-P1-N64-S0-1777903763-8d162e  (tau_max=10, baseline)
    - Stage2b-Ablation-P1-N64-S0-1777912902-05956d  (tau_max=20)

Pipeline:
  1. Download step_050000.pt for each run from S3.
  2. Load IGMNeRF(hidden_dim=256, num_layers=8, L=10) + log_tau_amp.
  3. Forward-pass on the P1 sightline grid (n_rays=64, seed=0 deterministic
     ordering -> first 64 sightlines, matching the trainer's load_dataset).
  4. F_pred = exp(-tau_pred); shape (64, 2048).
  5. P_F(k_||) via src.analysis.p_flux.compute_p_flux.
  6. Restrict to inertial range k_|| in [10^-2.5, 10^-1.5] s/km.
  7. Gate metric: max(|P_F_cell - P_F_baseline| / P_F_baseline) over the
     inertial-range k bins, for cell in {tau_max=5, tau_max=20}.
  8. PASS if both <= 2%, else FAIL.

Usage:
    uv run python -u scripts/diag_tau_max_sensitivity.py

Outputs:
    - paper_cvpr/figures/tau_max_sensitivity.png  (DVC-tracked)
    - prints the verdict + per-bin table to stdout
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
from dotenv import load_dotenv

# Make src.* importable from repo root.
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.analysis.p_flux import compute_p_flux  # noqa: E402
from src.data.loader import SherwoodLoader  # noqa: E402
from src.models.nerf import IGMNeRF, volume_render_physics  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "Sherwood")
FIG_PATH = os.path.join(REPO_ROOT, "paper_cvpr", "figures",
                        "tau_max_sensitivity.png")

S3_BUCKET = "cosmo-gas-vision-storage"
S3_PREFIX = "stage2b-checkpoints"

RUNS = [
    {
        "label": r"$\tau_{\max}=5$",
        "tau_max": 5,
        "run_dir": "Stage2b-Ablation-P1-N64-S0-1777912895-49b2cd",
        "color": "#3b82f6",  # blue
    },
    {
        "label": r"$\tau_{\max}=10$ (baseline)",
        "tau_max": 10,
        "run_dir": "Stage2b-Ablation-P1-N64-S0-1777903763-8d162e",
        "color": "#111827",  # near-black
    },
    {
        "label": r"$\tau_{\max}=20$",
        "tau_max": 20,
        "run_dir": "Stage2b-Ablation-P1-N64-S0-1777912902-05956d",
        "color": "#dc2626",  # red
    },
]
CHECKPOINT_NAME = "step_050000.pt"

# Inertial range per [D-13] / [D-24]: log10(k) in (-2.5, -1.5) -> k in s/km.
K_MIN_GATE = 10 ** -2.5  # ~3.162e-3
K_MAX_GATE = 10 ** -1.5  # ~3.162e-2

# P_F binning: cover a couple of decades around the inertial range so the
# log-bin centers fall comfortably inside [10^-2.5, 10^-1.5]. Use the
# compute_p_flux defaults (k_min=1e-3, k_max=1e-1, n_kbins=20) which give
# 5 dex / 20 = 0.25 dex per bin -> 4 bins inside the [D-13] window.
K_MIN_PF = 10 ** -3
K_MAX_PF = 10 ** -1
N_KBINS = 20

GATE_THRESHOLD = 0.02  # 2 percent


# ---------------------------------------------------------------------------
# Step 1: download checkpoints from S3
# ---------------------------------------------------------------------------

def download_checkpoint(run_dir: str, dest_dir: str) -> str:
    """Download step_050000.pt for one run via boto3. Returns local path."""
    import boto3
    s3 = boto3.client("s3")
    key = f"{S3_PREFIX}/{run_dir}/{CHECKPOINT_NAME}"
    local = os.path.join(dest_dir, f"{run_dir}_{CHECKPOINT_NAME}")
    if os.path.exists(local):
        print(f"[s3] cached {local} ({os.path.getsize(local)} bytes)")
        return local
    print(f"[s3] downloading s3://{S3_BUCKET}/{key} -> {local}")
    s3.download_file(S3_BUCKET, key, local)
    print(f"[s3] done ({os.path.getsize(local)} bytes)")
    return local


# ---------------------------------------------------------------------------
# Step 2 & 3: load model + forward-pass on the P1 sightline grid
# ---------------------------------------------------------------------------

def load_p1_sightlines(n_rays: int = 64) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Mirror experiments/nerf/pipeline.py:load_dataset for P1 z=0.300, n_rays=64.

    With seed=0 and n_rays=64 the trainer does NOT shuffle: it uses the first
    n_rays rows of the loaded sightlines (see load_dataset in pipeline.py:170-171).
    Here we replicate that exact slice so the forward pass coincides with what
    the model trained on.
    """
    loader = SherwoodLoader(DATA_ROOT)
    sightlines = loader.load_sightlines(1, 0.3)
    box_max = float(sightlines["header"]["box_kpc_h"])

    # The full (16384, 2048, 3) coord array won't fit in memory on the local
    # CPU box (~768 MiB float64). Build only the first n_rays rows here,
    # replicating the iaxis dispatch in SherwoodLoader.get_world_coordinates
    # (axis=1 -> x runs, axis=2 -> y runs, axis=3 -> z runs). The loader
    # itself is out of scope to modify (per dispatch).
    pos_axis = sightlines["pos_axis"]              # kpc/h, shape (nbins,)
    nbins = int(sightlines["header"]["nbins"])
    coords_raw = np.zeros((n_rays, nbins, 3), dtype=np.float64)
    for i in range(n_rays):
        axis = int(sightlines["iaxis"][i])
        x = float(sightlines["xaxis"][i])
        y = float(sightlines["yaxis"][i])
        z = float(sightlines["zaxis"][i])
        if axis == 1:
            coords_raw[i, :, 0] = pos_axis
            coords_raw[i, :, 1] = y
            coords_raw[i, :, 2] = z
        elif axis == 2:
            coords_raw[i, :, 0] = x
            coords_raw[i, :, 1] = pos_axis
            coords_raw[i, :, 2] = z
        elif axis == 3:
            coords_raw[i, :, 0] = x
            coords_raw[i, :, 1] = y
            coords_raw[i, :, 2] = pos_axis
        else:
            raise ValueError(f"unexpected iaxis {axis}")

    coords = (torch.tensor(coords_raw, dtype=torch.float32) / box_max)
    vel_axis = torch.tensor(sightlines["vel_axis"], dtype=torch.float32)
    print(f"[data] coords {tuple(coords.shape)}, vel_axis {tuple(vel_axis.shape)}, "
          f"box_max={box_max:.1f} kpc/h")
    return coords, vel_axis, box_max


def predict_flux(checkpoint_path: str,
                 coords: torch.Tensor,
                 vel_axis: torch.Tensor) -> np.ndarray:
    """Load IGMNeRF + log_tau_amp from checkpoint, forward-pass, return F_pred."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = IGMNeRF(hidden_dim=256, num_layers=8, L=10)
    model.load_state_dict(state["model_state"])
    model.eval()
    log_tau_amp = state["log_tau_amp"].detach().clone()
    tau_amp = torch.exp(log_tau_amp)
    print(f"[ckpt] step={state['step']}  log_tau_amp={float(log_tau_amp):.4f}  "
          f"tau_amp={float(tau_amp):.4f}")

    with torch.no_grad():
        tau = volume_render_physics(model, coords, vel_axis, tau_amp=tau_amp)
    tau_np = tau.detach().cpu().numpy()
    F = np.exp(-tau_np)
    print(f"[forward] tau range [{tau_np.min():.4g}, {tau_np.max():.4g}]  "
          f"<F>={F.mean():.4f}  shape={F.shape}")
    return F


# ---------------------------------------------------------------------------
# Step 5-7: P_F + gate metric
# ---------------------------------------------------------------------------

def compute_pf(F: np.ndarray, vel_axis_np: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    k_axis, P_F = compute_p_flux(
        F, vel_axis_np,
        k_min=K_MIN_PF, k_max=K_MAX_PF, n_kbins=N_KBINS,
    )
    return k_axis, P_F


def gate_metric(P_F_cell: np.ndarray,
                P_F_base: np.ndarray,
                k_axis: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return (max_rel_diff, k_at_max, mask_of_inertial_range_bins)."""
    in_range = (k_axis >= K_MIN_GATE) & (k_axis <= K_MAX_GATE)
    valid = in_range & np.isfinite(P_F_cell) & np.isfinite(P_F_base) & (P_F_base > 0)
    rel = np.abs(P_F_cell - P_F_base) / P_F_base
    rel_in = np.where(valid, rel, np.nan)
    if not np.any(np.isfinite(rel_in)):
        return float("nan"), float("nan"), in_range
    idx = int(np.nanargmax(rel_in))
    return float(rel_in[idx]), float(k_axis[idx]), in_range


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure(k_axis: np.ndarray,
                P_F_per_run: dict,
                fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.05},
        dpi=200,
    )

    # Top: P_F(k_||)
    for run in RUNS:
        P_F = P_F_per_run[run["tau_max"]]
        ax_top.loglog(
            k_axis, P_F,
            color=run["color"], lw=1.8, marker="o", ms=4.0,
            label=run["label"],
        )

    # Inertial range shading
    for ax in (ax_top, ax_bot):
        ax.axvspan(K_MIN_GATE, K_MAX_GATE, color="#fde68a", alpha=0.35,
                   zorder=0, label=None)

    ax_top.set_ylabel(r"$P_F(k_\parallel)\ \mathrm{[s\,km^{-1}]}$")
    ax_top.set_title(
        r"$\tau_{\max}$ sensitivity gate (P1, $n_{\rm rays}{=}64$, step 50k)"
    )
    ax_top.legend(loc="lower left", frameon=True, fontsize=9)
    ax_top.grid(True, which="both", alpha=0.25)

    # Bottom: ratio against tau_max=10 baseline
    P_F_base = P_F_per_run[10]
    for run in RUNS:
        if run["tau_max"] == 10:
            continue
        ratio = (P_F_per_run[run["tau_max"]] - P_F_base) / P_F_base
        ax_bot.semilogx(
            k_axis, 100.0 * ratio,
            color=run["color"], lw=1.6, marker="o", ms=4.0,
            label=rf"$\Delta P_F / P_F$ vs baseline ({run['label']})",
        )
    ax_bot.axhline(0.0, color="#111827", lw=0.8, ls="--", alpha=0.7)
    ax_bot.axhline(2.0, color="#dc2626", lw=0.8, ls=":", alpha=0.7)
    ax_bot.axhline(-2.0, color="#dc2626", lw=0.8, ls=":", alpha=0.7,
                   label=r"$\pm 2\%$ gate")
    ax_bot.set_xlabel(r"$k_\parallel\ \mathrm{[s\,km^{-1}]}$")
    ax_bot.set_ylabel(r"$\Delta P_F / P_F\ [\%]$")
    ax_bot.legend(loc="lower left", frameon=True, fontsize=8)
    ax_bot.grid(True, which="both", alpha=0.25)

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    sz = os.path.getsize(fig_path)
    print(f"[fig] wrote {fig_path} ({sz} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    torch.manual_seed(0)
    np.random.seed(0)

    # P1 sightline grid (shared across all 3 runs)
    coords, vel_axis, _box_max = load_p1_sightlines(n_rays=64)
    vel_axis_np = vel_axis.cpu().numpy().astype(np.float64)

    # Download + forward-pass for each cell
    cache_dir = os.path.join(REPO_ROOT, "experiments", "nerf", "artifacts",
                             "_tau_max_gate_ckpts")
    os.makedirs(cache_dir, exist_ok=True)

    F_per_run = {}
    for run in RUNS:
        print(f"\n=== {run['label']} (run_dir={run['run_dir']}) ===")
        ckpt_path = download_checkpoint(run["run_dir"], cache_dir)
        F = predict_flux(ckpt_path, coords, vel_axis)
        F_per_run[run["tau_max"]] = F

    # P_F per run
    print("\n=== P_F(k_||) ===")
    P_F_per_run = {}
    k_axis = None
    for run in RUNS:
        k, P_F = compute_pf(F_per_run[run["tau_max"]], vel_axis_np)
        if k_axis is None:
            k_axis = k
        else:
            assert np.allclose(k, k_axis), "k_axis mismatch across runs"
        P_F_per_run[run["tau_max"]] = P_F
        n_finite = int(np.isfinite(P_F).sum())
        print(f"  tau_max={run['tau_max']:>2}: {n_finite}/{len(P_F)} finite "
              f"bins; sample P_F[k~1e-2] = {P_F[np.argmin(np.abs(k_axis-1e-2))]:.4g}")

    # Gate metrics
    print("\n=== Gate (k_|| in [10^-2.5, 10^-1.5] s/km) ===")
    P_F_base = P_F_per_run[10]
    gate_results = {}
    for run in RUNS:
        if run["tau_max"] == 10:
            continue
        max_rel, k_at, in_range = gate_metric(
            P_F_per_run[run["tau_max"]], P_F_base, k_axis,
        )
        gate_results[run["tau_max"]] = {
            "max_rel": max_rel,
            "k_at_max": k_at,
            "pass": max_rel <= GATE_THRESHOLD,
        }
        verdict = "PASS" if max_rel <= GATE_THRESHOLD else "FAIL"
        print(f"  tau_max={run['tau_max']:>2}: "
              f"max|dP_F/P_F| = {100*max_rel:.4f}%  at k = {k_at:.4g} s/km  "
              f"-> {verdict}")

    overall_pass = all(r["pass"] for r in gate_results.values())
    print(f"\n=== VERDICT: {'PASS' if overall_pass else 'FAIL'} ===")

    # Per-bin table (inertial range only)
    in_range = (k_axis >= K_MIN_GATE) & (k_axis <= K_MAX_GATE)
    print("\n=== Per-bin table (inertial range) ===")
    print(f"{'k_||':>12} {'P_F(t=5)':>12} {'P_F(t=10)':>12} "
          f"{'P_F(t=20)':>12} {'|d5/10|%':>10} {'|d20/10|%':>10}")
    for i in np.where(in_range)[0]:
        k = k_axis[i]
        p5  = P_F_per_run[5][i]
        p10 = P_F_per_run[10][i]
        p20 = P_F_per_run[20][i]
        if not (np.isfinite(p5) and np.isfinite(p10) and np.isfinite(p20)
                and p10 > 0):
            print(f"{k:12.4g} {p5:12.4g} {p10:12.4g} {p20:12.4g} "
                  f"{'NaN':>10} {'NaN':>10}")
            continue
        r5 = 100.0 * abs(p5 - p10) / p10
        r20 = 100.0 * abs(p20 - p10) / p10
        print(f"{k:12.4g} {p5:12.4g} {p10:12.4g} {p20:12.4g} "
              f"{r5:10.4f} {r20:10.4f}")

    # Figure
    print()
    make_figure(k_axis, P_F_per_run, FIG_PATH)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())

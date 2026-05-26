"""
[D-70] Rev 2 M0 frozen-init variance-ratio baseline.

Computes Var(rho_theta)/Var(rho_truth) over 100 random 48^3 crops, averaged
over 10 frozen-init seeds of the *current* IGMNeRF architecture (no training,
just init + forward pass). Outputs mu_frozen, sigma_frozen, and the suggested
M0_PASS_BAR = mu_frozen + 2*sigma_frozen for [D-70] §2 gate definition.

Per PI dispatch (2026-05-25): K3 audit calibrates baseline against which the
variance ratio is judged; ResMLP-variant wiring lands in a later dispatch.
This baseline characterizes the current 8x256 + skip vanilla IGMNeRF; the
(1b) variant should be re-baselined after wiring if its init distribution
shifts meaningfully. For now this bounds the variant from one side.

CPU-only; should take ~minutes. No MLflow, no DVC.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.nerf import IGMNeRF  # noqa: E402


RHO_CACHE = REPO_ROOT / "Sherwood" / ".rho_field_cache" / "rho_field_p1_z0.300_n768.npy"
OUT_DIR = REPO_ROOT / "experiments" / "nerf" / "artifacts" / "d70_m0_baseline"
OUT_JSON = OUT_DIR / "baseline.json"

N_SEEDS = 10
N_CROPS = 100
CROP = 48
BOX_N = 768  # cache grid size


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception as exc:  # pragma: no cover
        return f"unknown ({exc})"


def make_unit_cube_coords(origin_idx: tuple[int, int, int], crop: int, box_n: int) -> torch.Tensor:
    """Build (crop, crop, crop, 3) coordinates in [0,1] for the requested sub-cube.

    The MLP is parameterized over the unit cube of the full simulation; a crop
    spans (origin_idx .. origin_idx+crop) cell indices, mapped to
    [origin/box_n, (origin+crop)/box_n] along each axis.
    """
    ox, oy, oz = origin_idx
    # cell-centered coordinates
    xs = (torch.arange(ox, ox + crop, dtype=torch.float32) + 0.5) / box_n
    ys = (torch.arange(oy, oy + crop, dtype=torch.float32) + 0.5) / box_n
    zs = (torch.arange(oz, oz + crop, dtype=torch.float32) + 0.5) / box_n
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)  # (crop, crop, crop, 3)
    return coords


BIN_EDGES = [(-np.inf, -1.0, "bin_A"),
             (-1.0, 0.3, "bin_B"),
             (0.3, 1.0, "bin_C"),
             (1.0, np.inf, "bin_D")]
MIN_BIN_SAMPLES = 5


def _per_bin_ratios(rho_theta: np.ndarray, rho_truth: np.ndarray, mean_rho: float) -> dict:
    """Per-bin Var(rho_theta|bin)/Var(rho_truth|bin) keyed on log10(truth/mean)."""
    log_overdensity = np.log10(np.maximum(rho_truth, 1e-30) / mean_rho)
    out = {}
    for lo, hi, name in BIN_EDGES:
        mask = (log_overdensity > lo) & (log_overdensity <= hi)
        n = int(mask.sum())
        if n < MIN_BIN_SAMPLES:
            out[name] = {"ratio": float("nan"), "n_samples": n}
            continue
        v_truth = float(np.var(rho_truth[mask]))
        v_theta = float(np.var(rho_theta[mask]))
        ratio = v_theta / v_truth if v_truth > 0 else float("nan")
        out[name] = {"ratio": ratio, "n_samples": n}
    return out


def variance_ratio_for_crop(
    model: IGMNeRF, rho_truth_field: np.ndarray, rng: np.random.Generator, mean_rho: float
) -> tuple[float, dict]:
    ox = int(rng.integers(0, BOX_N - CROP + 1))
    oy = int(rng.integers(0, BOX_N - CROP + 1))
    oz = int(rng.integers(0, BOX_N - CROP + 1))

    coords = make_unit_cube_coords((ox, oy, oz), CROP, BOX_N)
    # IGMNeRF.forward expects (..., 3); we flatten cube to (n_pts, 1, 3) so the
    # internal n_rays/n_bins shape pattern is preserved (n_rays=n_pts, n_bins=1).
    n_pts = CROP ** 3
    coords_flat = coords.reshape(n_pts, 1, 3)

    with torch.no_grad():
        fields = model(coords_flat)  # (n_pts, 1, 4)
    rho_theta = fields[..., 0].reshape(CROP, CROP, CROP).cpu().numpy()

    rho_truth = rho_truth_field[ox:ox + CROP, oy:oy + CROP, oz:oz + CROP]

    v_theta = float(np.var(rho_theta))
    v_truth = float(np.var(rho_truth))
    bin_breakdown = _per_bin_ratios(rho_theta, rho_truth, mean_rho)
    if v_truth <= 0:
        return float("nan"), bin_breakdown
    return v_theta / v_truth, bin_breakdown


def main() -> int:
    if not RHO_CACHE.exists():
        print(f"FATAL: rho cache missing at {RHO_CACHE}", file=sys.stderr)
        return 2

    print(f"[d70-m0-baseline] loading rho cache: {RHO_CACHE}", flush=True)
    rho = np.load(RHO_CACHE)
    if rho.shape != (BOX_N, BOX_N, BOX_N):
        print(f"FATAL: unexpected rho shape {rho.shape}, expected ({BOX_N},{BOX_N},{BOX_N})", file=sys.stderr)
        return 2
    print(f"[d70-m0-baseline] rho shape={rho.shape} min={rho.min():.3e} max={rho.max():.3e} mean={rho.mean():.3e}", flush=True)

    mean_rho = float(rho.mean())

    ratios_per_seed: list[float] = []
    crop_ratios_per_seed: dict[str, list[float]] = {}
    sigma_within_per_seed: list[float] = []
    per_bin_ratios_per_seed: dict[str, dict[str, float]] = {}
    per_bin_collected: dict[str, list[float]] = {name: [] for _, _, name in BIN_EDGES}
    per_bin_sample_counts: dict[str, list[int]] = {name: [] for _, _, name in BIN_EDGES}

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = IGMNeRF()  # current architecture, defaults: 8x256, L=10, no g, no e_p
        model.eval()
        # Crop sampling uses a separate RNG seeded the same way so each seed
        # sees the same set of 100 crop origins -> seed-to-seed comparison is
        # only over weight-init draws, not over crop locations.
        rng = np.random.default_rng(seed)

        crop_ratios: list[float] = []
        # accumulate per-bin ratios across this seed's crops, keyed by bin
        bin_accum: dict[str, list[float]] = {name: [] for _, _, name in BIN_EDGES}
        bin_n_accum: dict[str, list[int]] = {name: [] for _, _, name in BIN_EDGES}
        for k in range(N_CROPS):
            r, bin_break = variance_ratio_for_crop(model, rho, rng, mean_rho)
            crop_ratios.append(r)
            for name, entry in bin_break.items():
                bin_accum[name].append(entry["ratio"])
                bin_n_accum[name].append(entry["n_samples"])

        seed_mean = float(np.nanmean(crop_ratios))
        seed_within_sd = float(np.nanstd(crop_ratios, ddof=1))
        ratios_per_seed.append(seed_mean)
        sigma_within_per_seed.append(seed_within_sd)
        crop_ratios_per_seed[f"seed_{seed}"] = [float(x) for x in crop_ratios]

        # per-seed per-bin aggregate (mean across crops with valid samples)
        seed_bin = {}
        for name in bin_accum:
            arr = np.array(bin_accum[name], dtype=float)
            n_valid = int(np.sum(~np.isnan(arr)))
            mean_ratio = float(np.nanmean(arr)) if n_valid > 0 else float("nan")
            seed_bin[name] = mean_ratio
            if not np.isnan(mean_ratio):
                per_bin_collected[name].append(mean_ratio)
            per_bin_sample_counts[name].extend(bin_n_accum[name])
        per_bin_ratios_per_seed[f"seed_{seed}"] = seed_bin

        print(
            f"[d70-m0-baseline] seed={seed} mean_ratio={seed_mean:.6e} "
            f"within_sd={seed_within_sd:.3e} "
            f"min={float(np.nanmin(crop_ratios)):.3e} max={float(np.nanmax(crop_ratios)):.3e} "
            f"bins[A={seed_bin['bin_A']:.3e} B={seed_bin['bin_B']:.3e} "
            f"C={seed_bin['bin_C']:.3e} D={seed_bin['bin_D']:.3e}]",
            flush=True,
        )

    mu = float(np.mean(ratios_per_seed))
    sigma = float(np.std(ratios_per_seed, ddof=1))
    bar = mu + 2.0 * sigma

    sigma_within_median = float(np.median(sigma_within_per_seed))
    sigma_within_worst = float(np.max(sigma_within_per_seed))

    per_bin_aggregate = {}
    for name in per_bin_collected:
        arr = np.array(per_bin_collected[name], dtype=float)
        sample_counts = np.array(per_bin_sample_counts[name], dtype=float)
        per_bin_aggregate[name] = {
            "mean": float(np.mean(arr)) if arr.size > 0 else float("nan"),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
            "n_seeds_valid": int(arr.size),
            "median_samples_per_crop": float(np.median(sample_counts)) if sample_counts.size > 0 else 0.0,
            "min_samples_per_crop": int(sample_counts.min()) if sample_counts.size > 0 else 0,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_seeds": N_SEEDS,
        "n_crops_per_seed": N_CROPS,
        "crop_size": CROP,
        "ratios_per_seed": ratios_per_seed,
        "mu_frozen": mu,
        "sigma_frozen": sigma,
        "m0_pass_bar": bar,
        # ---- Rev 4 augmentation (R28 PROVISIONAL) ---------------------------
        "crop_ratios_per_seed": crop_ratios_per_seed,
        "sigma_within_per_seed": sigma_within_per_seed,
        "sigma_within_median": sigma_within_median,
        "sigma_within_worst": sigma_within_worst,
        "per_bin_ratios": per_bin_ratios_per_seed,
        "per_bin_aggregate": per_bin_aggregate,
        "bin_definition": {
            "axis": "log10(rho_truth / mean(rho))",
            "bin_A_void": "log10 <= -1",
            "bin_B_mean": "-1 < log10 <= +0.3",
            "bin_C_mid":  "+0.3 < log10 <= +1.0",
            "bin_D_tail": "log10 > +1.0",
            "min_samples_per_crop_required": MIN_BIN_SAMPLES,
        },
        "mean_rho_truth": mean_rho,
        # ----------------------------------------------------------------------
        "architecture": "current IGMNeRF (8x256, L=10, single mid-skip, ReLU, Softplus head)",
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rho_cache": str(RHO_CACHE.relative_to(REPO_ROOT)),
        "notes": (
            "Frozen-init forward-pass-only baseline; no training. Variance ratio "
            "Var(rho_theta)/Var(rho_truth) computed per 48^3 crop, averaged over "
            "100 random crops per seed, then mean+std across 10 seeds. "
            "PI-authorized current-architecture baseline; rerun on (1b) ResMLP "
            "variant after wiring lands if its init distribution shifts. "
            "Rev 4 augmentation adds (a) per-seed within-seed crop SD as honest "
            "disclosure (NOT a gate knob — PI ruled Spearman/Wilcoxon path), and "
            "(b) per-bin variance-ratio breakdown for the S-D pre-commit per-bin "
            "log-MSE flatness check."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[d70-m0-baseline] wrote {OUT_JSON}", flush=True)
    print(
        f"M0_PASS_BAR={mu:.6e}+2*{sigma:.6e}={bar:.6e} | "
        f"n_seeds={N_SEEDS} n_crops={N_CROPS} crop={CROP}^3",
        flush=True,
    )
    bin_d_mean = per_bin_aggregate["bin_D"]["mean"]
    print(
        f"M0 re-bake: sigma_within_median={sigma_within_median:.2e}, "
        f"sigma_within_worst={sigma_within_worst:.2e}; "
        f"Bin-D ratio aggregate mean={bin_d_mean:.2e} | "
        f"n_seeds={N_SEEDS} n_crops={N_CROPS}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
[D-69] M3 PASS-band bootstrap calibration.

Defense-panel KILLER K3 absorbed by PI (2026-05-24): the original [0.7, 1.3]
PASS band on R_real = Var[rho_theta] / Var[rho_truth] was an informal expert
prior with no calibration to the sampling-noise floor of the 100-crop
estimator that scores the gate. This script produces that calibration.

Procedure (single local CPU invocation):
  1. mmap-load the canonical P1 z=0.3 n_grid=768 rho/<rho> field.
  2. Use the production loader.extract_rho_crops API to draw 100 crops at
     crop_size in {32, 48} (both precedented per sprint4 / sprint5_cprime).
  3. Compute the LOG-space variance (matches L = MSE(log10(rho+1e-3))) and
     LINEAR variance, both aggregated over all voxels in the 100 crops.
  4. Bootstrap: 1000 independent resamples (fresh seed per resample).
     -> Report empirical (mean, std, [2.5, 16, 50, 84, 97.5] percentiles)
        for var_truth.
  5. Null-ratio: 1000 paired resamples; each draws TWO independent 100-crop
     samples from the same rho-field, computes R = Var_a / Var_b.
     Under the null (numerator and denominator drawn from the same
     distribution), R ~ 1 in expectation; the empirical (mean, std,
     percentiles) gives the sampling-noise floor of the gate metric.
  6. PASS = R in [mu - 1*sigma, mu + 1*sigma] of the null ratio distribution.

Sanity:
  - Confirm _RHO_CROP_LO semantics (it is documentation only; NOT applied as
    a per-voxel clamp; the validator enforces only non-negativity).

Folded-in panel asks:
  - S2 compute estimate: 10-step CPU dry-run of the pretrain inner loss
    (log10-MSE) on the existing IGMNeRF MLP at microbatch 1024 from a
    single 48^3 crop; report per-step wall-clock and extrapolate to 5000
    steps on Juno A30 via a 30-100x speedup band.

CONSTRAINT: CPU only. No CUDA, no Juno, no sbatch.

Outputs:
  experiments/nerf/artifacts/d69_m3_band_calibration.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Repo-root onto sys.path so `src.` imports resolve when run from anywhere.
REPO_ROOT = Path(r"D:\Data\sujin\CosmoGasVision")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import (
    SherwoodLoader,
    _RHO_CROP_LO,
    _RHO_FIELD_CACHE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_NPY = REPO_ROOT / "Sherwood" / ".rho_field_cache" / "rho_field_p1_z0.300_n768.npy"
CACHE_JSON = REPO_ROOT / "Sherwood" / ".rho_field_cache" / "rho_field_p1_z0.300_n768.json"
OUT_JSON = REPO_ROOT / "experiments" / "nerf" / "artifacts" / "d69_m3_band_calibration.json"

PHYSICS_ID = 1
REDSHIFT = 0.300
N_GRID = 768
N_CROPS_PER_SAMPLE = 100      # per the PI's R_real estimator
N_BOOTSTRAP = 1000             # outer bootstrap count
LOG_EPS = 1e-3                 # matches the PI's L = MSE(log10(rho + 1e-3))
CROP_SIZES = (32, 48)          # sprint4 / sprint5_cprime precedent

# Per-bootstrap seed offsets — fixed so the calibration is reproducible.
SEED_BASE_SINGLE = 100000      # var_truth resamples
SEED_BASE_RATIO_NUM = 200000   # numerator of R_real
SEED_BASE_RATIO_DEN = 300000   # denominator of R_real

# Microbatch dry-run config
DRYRUN_MICROBATCH = 1024
DRYRUN_STEPS = 10
DRYRUN_CROP_SIZE = 48


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def md5_first_1MB(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()


def summarize(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p16": float(np.percentile(arr, 16)),
        "p50": float(np.percentile(arr, 50)),
        "p84": float(np.percentile(arr, 84)),
        "p97_5": float(np.percentile(arr, 97.5)),
        "n": int(arr.size),
    }


def derive_pass_band(ratio_summary: dict) -> dict:
    mu = ratio_summary["mean"]
    sd = ratio_summary["std"]
    return {
        "pass_lo": mu - 1.0 * sd,
        "pass_hi": mu + 1.0 * sd,
        "marginal_lo_outer": mu - 2.0 * sd,
        "marginal_hi_outer": mu + 2.0 * sd,
        "marginal_lo_inner": mu - 1.0 * sd,
        "marginal_hi_inner": mu + 1.0 * sd,
        "rule": "PASS = R in [mu-1s, mu+1s]; MARGINAL = [mu-2s, mu-1s] U [mu+1s, mu+2s]; FAIL = outside [mu-2s, mu+2s]",
        "mu": mu,
        "sigma": sd,
    }


# ---------------------------------------------------------------------------
# Crop assembly (uses the production loader path)
# ---------------------------------------------------------------------------
def make_loader() -> SherwoodLoader:
    """SherwoodLoader instance configured for sightline-free 3D-only use.

    The data_root argument is irrelevant for `extract_rho_crops` (that path
    uses SherwoodIGMGalLoader internally for the underlying CIC field), but
    the loader's __init__ may still try to populate sightline metadata.
    Use an arbitrary existing dir.
    """
    # SherwoodLoader signature: (data_root, ...). Use repo root as a benign
    # placeholder — we never call sightline methods.
    return SherwoodLoader(data_root=str(REPO_ROOT / "Sherwood"))


def draw_var_streaming(
    rho_field: np.ndarray, crop_size: int, seed: int
) -> tuple[float, float]:
    """Sample N_CROPS_PER_SAMPLE crops with the production-equivalent RNG
    (np.random.default_rng(seed).integers as in loader.extract_rho_crops)
    and accumulate first/second moments of log10(rho + eps) AND (rho + eps)
    across all crop voxels in float64. Returns (var_log10, var_linear).

    This bypasses the loader's float32 tensor materialization + validator
    (each call to extract_rho_crops with N_CROPS_PER_SAMPLE=100 is ~5s at
    L=32 / ~14s at L=48 dominated by Python overhead — uneconomic for 4000+
    resamples). The sampling logic (seed, corners, periodic-BC np.ix_
    indexing) is BYTE-IDENTICAL to the loader path; only the post-indexing
    accumulation differs (streaming Welford-equivalent in float64 rather
    than materialize-then-var-in-float64).
    """
    N = rho_field.shape[0]
    rng = np.random.default_rng(int(seed))
    corners = rng.integers(low=0, high=N, size=(N_CROPS_PER_SAMPLE, 3), dtype=np.int64)
    offset = np.arange(crop_size, dtype=np.int64)

    n_total = 0
    sum_log = 0.0
    sumsq_log = 0.0
    sum_lin = 0.0
    sumsq_lin = 0.0
    L = int(crop_size)
    for c in range(N_CROPS_PER_SAMPLE):
        i0, j0, k0 = int(corners[c, 0]), int(corners[c, 1]), int(corners[c, 2])
        # Fast path: contiguous slice when no axis wraps.
        if (i0 + L) <= N and (j0 + L) <= N and (k0 + L) <= N:
            crop = rho_field[i0:i0+L, j0:j0+L, k0:k0+L]
        else:
            # Slow path: periodic wrap via np.ix_. Same byte values as the
            # loader's path; preserved for the ~10-20% of corners near the
            # box edge.
            ii = (i0 + offset) % N
            jj = (j0 + offset) % N
            kk = (k0 + offset) % N
            crop = rho_field[np.ix_(ii, jj, kk)]
        rho64 = crop.astype(np.float64, copy=False).ravel()
        lin = rho64 + LOG_EPS
        log10 = np.log10(lin)
        n_total += rho64.size
        sum_log += log10.sum()
        sumsq_log += (log10 * log10).sum()
        sum_lin += lin.sum()
        sumsq_lin += (lin * lin).sum()

    mean_log = sum_log / n_total
    mean_lin = sum_lin / n_total
    var_log = sumsq_log / n_total - mean_log * mean_log
    var_lin = sumsq_lin / n_total - mean_lin * mean_lin
    return float(var_log), float(var_lin)


# ---------------------------------------------------------------------------
# Bootstrap loops
# ---------------------------------------------------------------------------
def bootstrap_single(rho_field: np.ndarray, crop_size: int) -> dict:
    """Empirical distribution of var_truth (log + linear) over N_BOOTSTRAP
    independent 100-crop resamples."""
    var_log = np.empty(N_BOOTSTRAP, dtype=np.float64)
    var_lin = np.empty(N_BOOTSTRAP, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(N_BOOTSTRAP):
        vl, vli = draw_var_streaming(rho_field, crop_size=crop_size,
                                     seed=SEED_BASE_SINGLE + i)
        var_log[i] = vl
        var_lin[i] = vli
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [single L={crop_size}] {i+1}/{N_BOOTSTRAP} resamples "
                f"({elapsed:.1f}s, mean log-var so far = {var_log[:i+1].mean():.4f})",
                flush=True,
            )
    wall = time.perf_counter() - t0
    return {
        "log10_var": summarize(var_log),
        "linear_var": summarize(var_lin),
        "wall_clock_sec": wall,
    }


def bootstrap_ratio(rho_field: np.ndarray, crop_size: int) -> dict:
    """Empirical null distribution of R_real = Var_num / Var_den with two
    independent 100-crop samples per resample."""
    ratio_log = np.empty(N_BOOTSTRAP, dtype=np.float64)
    ratio_lin = np.empty(N_BOOTSTRAP, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(N_BOOTSTRAP):
        vl_n, vli_n = draw_var_streaming(rho_field, crop_size=crop_size,
                                         seed=SEED_BASE_RATIO_NUM + i)
        vl_d, vli_d = draw_var_streaming(rho_field, crop_size=crop_size,
                                         seed=SEED_BASE_RATIO_DEN + i)
        ratio_log[i] = vl_n / vl_d
        ratio_lin[i] = vli_n / vli_d
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [ratio  L={crop_size}] {i+1}/{N_BOOTSTRAP} resamples "
                f"({elapsed:.1f}s, mean log-ratio so far = {ratio_log[:i+1].mean():.4f})",
                flush=True,
            )
    wall = time.perf_counter() - t0
    return {
        "log10_ratio": summarize(ratio_log),
        "linear_ratio": summarize(ratio_lin),
        "wall_clock_sec": wall,
    }


# ---------------------------------------------------------------------------
# S2 compute dry-run (10-step CPU forward+backward with the production MLP)
# ---------------------------------------------------------------------------
def s2_dry_run(loader: SherwoodLoader) -> dict:
    """Time 10 forward+backward steps of L = MSE(log10(rho_theta+eps), log10(rho_truth+eps))
    with microbatch=1024 voxels drawn from a single 48^3 crop. Random-init
    IGMNeRF; CPU; no Voigt; no MLflow."""
    from src.models.nerf import IGMNeRF

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    torch.manual_seed(0)
    model = IGMNeRF(hidden_dim=256, num_layers=8, L=10)
    model = model.to(torch.float32)
    model.train()

    # Single 48^3 crop, drawn deterministically.
    crops_t, _ = loader.extract_rho_crops(
        physics_id=PHYSICS_ID,
        redshift=REDSHIFT,
        crop_size=DRYRUN_CROP_SIZE,
        n_crops=1,
        seed=42,
        n_grid=N_GRID,
    )
    rho_truth_grid = crops_t[0, 0].numpy()           # (48, 48, 48)
    L = rho_truth_grid.shape[0]

    # Build a coords lookup: voxel-center coords in [0, 1] normalized to the
    # crop (matches NeRF input convention).
    rng = np.random.default_rng(0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    per_step_wall = []
    for step in range(DRYRUN_STEPS):
        # Sample microbatch voxel indices uniformly with replacement.
        idx = rng.integers(low=0, high=L, size=(DRYRUN_MICROBATCH, 3))
        coords_np = (idx.astype(np.float32) + 0.5) / float(L)      # (B, 3)
        rho_truth_np = rho_truth_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        coords = torch.from_numpy(coords_np)                       # (B, 3)
        # NeRF forward expects (..., 3) and returns (..., 4); add a fake
        # ray axis of length 1 to be unambiguous.
        coords_in = coords.unsqueeze(0)                            # (1, B, 3)
        rho_truth = torch.from_numpy(rho_truth_np.astype(np.float32))  # (B,)

        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(coords_in)                                     # (1, B, 4)
        rho_theta = out[0, :, 0]                                   # softplus density
        loss = ((torch.log10(rho_theta + LOG_EPS)
                 - torch.log10(rho_truth + LOG_EPS)) ** 2).mean()
        loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        per_step_wall.append(dt)
        print(f"  [s2 dry-run] step {step+1}/{DRYRUN_STEPS}  loss={loss.item():.4f}  wall={dt*1000:.1f} ms", flush=True)

    arr = np.array(per_step_wall[1:], dtype=np.float64)  # exclude warmup step
    mean_step = float(arr.mean()) if arr.size > 0 else float(per_step_wall[0])
    # Juno A30 vs local CPU back-of-envelope: 30-100x speedup band.
    juno_5k_low = (mean_step / 100.0) * 5000.0
    juno_5k_high = (mean_step / 30.0) * 5000.0
    return {
        "microbatch": DRYRUN_MICROBATCH,
        "crop_size_for_voxels": DRYRUN_CROP_SIZE,
        "n_steps_timed": DRYRUN_STEPS,
        "per_step_wall_sec_all": [float(x) for x in per_step_wall],
        "per_step_wall_sec_mean_excl_warmup": mean_step,
        "cpu_5000_step_extrapolation_sec": mean_step * 5000.0,
        "juno_a30_5000_step_estimate_sec_lo_100x": juno_5k_low,
        "juno_a30_5000_step_estimate_sec_hi_30x": juno_5k_high,
        "speedup_band_used": "30x-100x (back-of-envelope CPU-thread -> A30 fp32 MLP)",
        "model": "IGMNeRF(hidden_dim=256, num_layers=8, L=10)",
        "torch_num_threads": torch.get_num_threads(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[D-69] M3 PASS-band calibration — started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cache npy: {CACHE_NPY}")
    print(f"Cache npy size: {CACHE_NPY.stat().st_size} bytes")
    print(f"Output:    {OUT_JSON}")

    npy_md5 = md5_first_1MB(CACHE_NPY)
    print(f"npy first-1MB md5: {npy_md5}")
    with open(CACHE_JSON, "r") as f:
        cache_meta = json.load(f)

    loader = make_loader()

    # Force materialization (and cache hydration) up front so the bootstrap
    # loop wall-clock isn't polluted by the one-shot disk load.
    print("\nWarming rho-field cache via single crop draw ...")
    t_warm = time.perf_counter()
    _ = loader.extract_rho_crops(
        physics_id=PHYSICS_ID, redshift=REDSHIFT,
        crop_size=32, n_crops=1, seed=0, n_grid=N_GRID,
    )
    print(f"  warm-up wall {time.perf_counter() - t_warm:.2f}s; "
          f"_RHO_FIELD_CACHE keys: {list(_RHO_FIELD_CACHE.keys())}")

    rho_field = _RHO_FIELD_CACHE[(PHYSICS_ID, REDSHIFT, N_GRID)]
    # Page-fault every page of the 1.8GB float32 field into hot DRAM/L3
    # before bootstrap timing — without this, the random-access fancy-index
    # pattern in draw_var_streaming hits cold-cache stalls (~5s/draw at
    # L=32, vs 0.15s warm). Single .sum() over the array forces a
    # contiguous pass and brings every page hot.
    t_warm = time.perf_counter()
    _field_sum_for_warmup = float(rho_field.sum())
    print(f"  full-array warm-up touch (sum={_field_sum_for_warmup:.6e}): "
          f"{time.perf_counter() - t_warm:.2f}s")

    field_min = float(rho_field.min())
    field_max = float(rho_field.max())
    field_mean = float(rho_field.mean())
    n_zero = int((rho_field == 0.0).sum())
    print(f"  rho field: shape={rho_field.shape} dtype={rho_field.dtype}  "
          f"min={field_min:.3e} max={field_max:.3e} mean={field_mean:.6f} "
          f"zero_cells={n_zero}/{rho_field.size} ({100*n_zero/rho_field.size:.1f}%)")

    # P3 verification: _RHO_CROP_LO is documented heuristic only — the
    # validator (loader._validate_rho_crops, lines 1254-1290) enforces
    # non-negativity and the 1e6 upper ceiling but does NOT apply a 1e-3
    # floor. Zero cells survive through to the loss layer; the LOG_EPS
    # additive is what stabilizes log10(0) downstream.
    rho_crop_lo_semantics = "not-enforced (documentation only; loader._validate_rho_crops asserts only non-negativity + max < 1e6; zero cells pass through unchanged)"
    print(f"\n_RHO_CROP_LO ({_RHO_CROP_LO}) semantics: {rho_crop_lo_semantics}")

    results = {
        "meta": {
            "script": str(Path(__file__).resolve()),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cache_npy_path": str(CACHE_NPY),
            "cache_npy_size_bytes": CACHE_NPY.stat().st_size,
            "cache_npy_first_1MB_md5": npy_md5,
            "cache_json": cache_meta,
            "physics_id": PHYSICS_ID,
            "redshift": REDSHIFT,
            "n_grid": N_GRID,
            "n_crops_per_sample": N_CROPS_PER_SAMPLE,
            "n_bootstrap": N_BOOTSTRAP,
            "log_eps": LOG_EPS,
            "crop_sizes": list(CROP_SIZES),
            "seed_base_single": SEED_BASE_SINGLE,
            "seed_base_ratio_num": SEED_BASE_RATIO_NUM,
            "seed_base_ratio_den": SEED_BASE_RATIO_DEN,
            "_RHO_CROP_LO_value_in_loader": _RHO_CROP_LO,
            "rho_crop_lo_semantics": "per-voxel-not-enforced",
            "rho_crop_lo_semantics_long": rho_crop_lo_semantics,
            "rho_field_stats": {
                "shape": list(rho_field.shape),
                "dtype": str(rho_field.dtype),
                "min": field_min,
                "max": field_max,
                "mean": field_mean,
                "zero_cells_frac": float(n_zero) / float(rho_field.size),
            },
        },
        "var_truth": {},
        "ratio_under_null": {},
        "pass_band": {},
        "s2_dry_run": None,
        "wall_clock_total_sec": None,
    }

    t_total = time.perf_counter()

    for L in CROP_SIZES:
        print(f"\n=== crop_size = {L}^3 ===")
        print(f"Bootstrap var_truth at L={L}")
        single = bootstrap_single(rho_field, crop_size=L)
        print(f"Bootstrap null-ratio at L={L}")
        ratio = bootstrap_ratio(rho_field, crop_size=L)
        results["var_truth"][f"L{L}"] = single
        results["ratio_under_null"][f"L{L}"] = ratio
        results["pass_band"][f"L{L}"] = {
            "log10_framing": derive_pass_band(ratio["log10_ratio"]),
            "linear_framing": derive_pass_band(ratio["linear_ratio"]),
        }
        pb = results["pass_band"][f"L{L}"]["log10_framing"]
        print(f"  -> PASS (log10) at L={L}: "
              f"[{pb['pass_lo']:.4f}, {pb['pass_hi']:.4f}]  "
              f"(mu={pb['mu']:.4f}, 1sigma={pb['sigma']:.4f})")

    print("\n=== S2 compute dry-run (CPU, microbatch=1024, L=48 sample) ===")
    results["s2_dry_run"] = s2_dry_run(loader)

    results["wall_clock_total_sec"] = time.perf_counter() - t_total
    results["meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON}")
    print(f"Total wall: {results['wall_clock_total_sec']:.1f}s")


if __name__ == "__main__":
    main()

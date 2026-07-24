"""Pipeline for the unet-inversion track — [U-06] Stage-2 A2.

Spec of record: ``experiments/unet-inversion/design/u06_stage2_spec.md``
S(a)/(b)/(d), commit 489a0d3. Subcommands:

* ``s2`` — overfit-one-batch smoke (2 examples, seed 42, lr 1e-3, 50 steps;
  gate loss(50) <= 0.1 x loss(0)).
* ``s3`` — 500-step mini-run (P1+P2, batch 4 MPS / grad-accum CPU, lr 3e-4,
  aug on, length 4096) + step-100 contract assertion + loss-trend gate +
  quick masked eval with U-G controls and descriptive columns.

MLflow contract per spec S(b) with nullcontext fallback (tracker-403
precedent); ALL metrics mirrored to a local CSV under
``experiments/unet-inversion/artifacts/stage2/`` regardless of tracker
availability. fp32 throughout (MPS bf16 not trusted). Seed 42 primary.

Run from repo root: ``PYTHONPATH=. .venv/bin/python -u
experiments/unet-inversion/pipeline.py {s2,s3}``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.sightline_rasterizer import DELTA_F_SCALE, rasterize_crop  # noqa: E402
from src.data.unet_crop_sampler import intersecting_rays  # noqa: E402
from src.data.unet_pair_dataset import (  # noqa: E402
    PhysicsSource,
    UNetPairDataset,
    build_physics_source,
)
from src.models.unet3d import UNet3D  # noqa: E402

EXPERIMENT_NAME = "CosmoGasVision/unet-inversion"
STAGE2_DIR = REPO / "experiments" / "unet-inversion" / "artifacts" / "stage2"
NULL_BAND = STAGE2_DIR / "null_band_n200.json"
CUBES = {
    1: REPO / "experiments/nerf/artifacts/d75_rescore/cubes/truth_real_192.npy",
    2: REPO / "experiments/unet-inversion/artifacts/stage1/cubes/truth_real_192_p2.npy",
}
SEED = 42
VAL_SEED = 4242
CROP = 64
STRIDE = 32
N_GRID = 192

# ----------------------------------------------------------------- utilities


def git_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CSVMirror:
    """Local CSV metric mirror (spec S(b): mandatory alongside MLflow)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(["step", "metric", "value"])

    def log(self, step: int, metric: str, value: float) -> None:
        self._w.writerow([step, metric, f"{value:.10g}"])
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class Tracker:
    """MLflow wrapper with the mandatory nullcontext fallback + CSV mirror."""

    def __init__(self, run_name: str, tags: Dict[str, str]) -> None:
        self.csv = CSVMirror(STAGE2_DIR / f"{run_name}_metrics.csv")
        self._mlflow = None
        self._ctx = nullcontext()
        try:
            import mlflow

            mlflow.set_tracking_uri(
                os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
            )
            mlflow.set_experiment(EXPERIMENT_NAME)
            self._ctx = mlflow.start_run(run_name=run_name, tags=tags)
            self._mlflow = mlflow
        except Exception as exc:  # noqa: BLE001 — degrade to local CSV only
            print(f"[pipeline] MLflow unavailable ({exc!r}); CSV mirror only.",
                  flush=True)

    def __enter__(self):
        self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        self.csv.close()
        return self._ctx.__exit__(*a)

    def log_metric(self, step: int, metric: str, value: float) -> None:
        self.csv.log(step, metric, value)
        if self._mlflow is not None:
            try:
                self._mlflow.log_metric(metric, value, step=step)
            except Exception:  # noqa: BLE001
                self._mlflow = None
                print("[pipeline] MLflow dropped mid-run; CSV mirror only.",
                      flush=True)

    def log_params(self, params: Dict) -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.log_params(params)
            except Exception:  # noqa: BLE001
                pass


def mandatory_tags(physics_id: str, seed: int) -> Dict[str, str]:
    return {
        "model_type": "unet3d",
        "stage": "2",
        "physics_id": physics_id,
        "redshift": "0.3",
        "seed": str(seed),
        "delta_f_scale": str(DELTA_F_SCALE),
        "crop_size": str(CROP),
        "base_channels": "32",
        "git_commit": git_commit_hash(),
    }


def build_sources(physics_ids: Sequence[int]) -> List[PhysicsSource]:
    out = []
    for pid in physics_ids:
        print(f"[pipeline] building PhysicsSource P{pid} ...", flush=True)
        out.append(build_physics_source(
            str(REPO / "Sherwood"), pid, str(CUBES[pid]), redshift=0.3,
        ))
    return out


def stack_batch(ds: UNetPairDataset, indices: Sequence[int], device):
    xs, ys = [], []
    for i in indices:
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    return (torch.stack(xs).to(device), torch.stack(ys).to(device))


def val_pred_std(model: UNet3D, val_batch) -> float:
    """Prediction std on the fixed val batch (spec S(b) s3 contract)."""
    model.eval()
    with torch.no_grad():
        pred = model(val_batch[0])
    model.train()
    return float(pred.float().std().item())


# ------------------------------------------------------------------ s2 smoke


def run_s2(steps: int = 50, lr: float = 1.0e-3) -> Dict:
    """Overfit-one-batch (spec S(b) s2). Gate: loss(50) <= 0.1 x loss(0),
    where loss(0) is the pre-update loss and loss(k) follows k updates."""
    t0 = time.time()
    device = pick_device()
    torch.manual_seed(SEED)
    sources = build_sources([1])
    ds = UNetPairDataset(sources, length=2, seed=SEED, augment=True)
    batch = stack_batch(ds, [0, 1], device)
    model = UNet3D().to(device)
    n_params = model.n_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    losses: List[float] = []
    tags = mandatory_tags("1", SEED)
    with Tracker("Stage2-OverfitOneBatch", tags) as trk:
        trk.log_params({"lr": lr, "steps": steps, "batch": 2,
                        "n_params": n_params, "device": str(device),
                        "optimizer": "AdamW", "weight_decay": 1e-4,
                        "grad_clip": 1.0})
        model.train()
        for step in range(steps + 1):           # loss(0) .. loss(steps)
            pred = model(batch[0])
            loss = torch.nn.functional.mse_loss(pred, batch[1])
            losses.append(float(loss.item()))
            trk.log_metric(step, "train_mse", losses[-1])
            if step == steps:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        gate_pass = losses[steps] <= 0.1 * losses[0]
        trk.log_metric(steps, "s2_gate_pass", float(gate_pass))
    record = {
        "rung": "A6 (s2) — overfit-one-batch smoke",
        "spec": "experiments/unet-inversion/design/u06_stage2_spec.md S(b)/(d)",
        "session_utc": "2026-07-24",
        "config": {"n_examples": 2, "seed": SEED, "lr": lr, "steps": steps,
                   "optimizer": "AdamW", "weight_decay": 1e-4,
                   "grad_clip": 1.0, "augment": True, "physics": ["P1"],
                   "loss": "MSE on x over ALL crop voxels",
                   "loss_indexing": "loss(k) = MSE after k updates; "
                                    "loss(0) = pre-update"},
        "model": {"arch": "UNet3D 4-level base-32 GN8+SiLU",
                  "n_params_exact": n_params},
        "device": str(device),
        "dtype": "float32",
        "losses": losses,
        "gate": {"rule": "loss(50) <= 0.1 * loss(0)",
                 "loss_0": losses[0], "loss_50": losses[steps],
                 "ratio": losses[steps] / losses[0],
                 "verdict": "PASS" if gate_pass else "FAIL"},
        "wall_clock_s": time.time() - t0,
        "git_commit": git_commit_hash(),
    }
    out = STAGE2_DIR / "s2_overfit_record.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"[s2] loss0={losses[0]:.4f} loss{steps}={losses[steps]:.4f} "
          f"ratio={losses[steps]/losses[0]:.4f} "
          f"gate={'PASS' if gate_pass else 'FAIL'} -> {out}", flush=True)
    return record


# ----------------------------------------------------------- s3 training core


def train_s3(model: UNet3D, ds: UNetPairDataset, val_batch, trk: Tracker,
             device, steps: int, lr: float, batch_size: int,
             accum: int) -> List[float]:
    """500-step mini-run loop. Returns per-step losses (1-indexed list;
    losses[k-1] = loss at step k, averaged over accumulation micro-batches).

    The step-100 contract assertion lives INSIDE this loop, OUTSIDE any
    try/except (spec S(b), raising loud).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    model.train()
    losses: List[float] = []
    idx = 0
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            xb, yb = stack_batch(
                ds, range(idx, idx + batch_size), device)
            idx += batch_size
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            (loss / accum).backward()
            step_loss += float(loss.item()) / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(step_loss)
        trk.log_metric(step, "train_mse", step_loss)
        if step % 25 == 0 or step == 1:
            print(f"[s3] step {step}/{steps} mse={step_loss:.4f}", flush=True)

        # R20(ii) contract assertion — inside the loop, OUTSIDE try/except.
        if step == 100:
            pstd = val_pred_std(model, val_batch)
            trk.log_metric(step, "val_pred_std", pstd)
            assert pstd > 0.01, (
                f"CONTRACT FAIL (R20(ii)): val prediction std {pstd:.6f} "
                f"<= 0.01 x-units at step 100")
            assert losses[99] < losses[0], (
                f"CONTRACT FAIL (R20(ii)): loss(100)={losses[99]:.6f} "
                f">= loss(1)={losses[0]:.6f}")
            print(f"[s3] step-100 contract PASS: pred_std={pstd:.4f}, "
                  f"loss100={losses[99]:.4f} < loss1={losses[0]:.4f}",
                  flush=True)
    return losses


# --------------------------------------------- sliding-window inference util
# Minimal version (spec S(c) inference clause); the full eval harness is A4
# (another agent's artifact). 64^3 windows, stride 32, uniform overlap
# averaging; each window rasterizes its intersecting subset of a FIXED
# file-order eval ray pattern ([0,1024) primary / [0,64) secondary).


def hann3_weight(crop: int = CROP) -> np.ndarray:
    """Periodic Hann^3 window weight (micro-cycle A1, 50de53a). With hop =
    crop/2 the periodic Hann satisfies COLA exactly: w[n] + w[n + crop/2]
    = 1 per axis, so the 3D accumulated weight is uniformly 1."""
    n = np.arange(crop, dtype=np.float64)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / crop))   # periodic Hann
    return w[:, None, None] * w[None, :, None] * w[None, None, :]


def sliding_window_predict(
    model: UNet3D,
    source: PhysicsSource,
    ray_ids: np.ndarray,
    device: torch.device,
    delta_f: Optional[np.ndarray] = None,
    input_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    taper: str = "hann",
) -> np.ndarray:
    """Predict the full n_grid^3 x-field by overlapping 64^3 windows.

    taper='hann' (inference-of-record per micro-cycle A1): per-window
    Hann^3 weighting, normalized by accumulated weight; runtime assertion
    accumulated-weight-uniform-to-<1e-6. taper='uniform' retained as a
    diagnostic co-report column only (8x-count assert).
    """
    geom = source.geometry
    n = geom.n_grid
    df = source.delta_f if delta_f is None else delta_f
    ray_ids = np.asarray(ray_ids, dtype=np.int64)
    if taper not in ("hann", "uniform"):
        raise ValueError(f"unknown taper {taper!r}")
    W = hann3_weight(CROP) if taper == "hann" else np.ones((CROP,) * 3)
    pred_sum = np.zeros((n, n, n), dtype=np.float64)
    wsum = np.zeros((n, n, n), dtype=np.float64)
    # S7 (spec v2): PERIODIC window placement — stride 32 on the periodic
    # box, 6 positions/axis, 216 windows, exact 8x per-voxel coverage.
    positions = list(range(0, n, STRIDE))
    model.eval()
    with torch.no_grad():
        for ca, cb, cc in itertools.product(positions, repeat=3):
            corner = np.array([ca, cb, cc], dtype=np.int64)
            pool = intersecting_rays(geom, corner, CROP)
            take = np.intersect1d(pool, ray_ids)
            inp = rasterize_crop(df, geom, take, corner, crop_size=CROP,
                                 scale=DELTA_F_SCALE)
            if input_transform is not None:
                inp = input_transform(inp)
            xb = torch.from_numpy(inp[None]).to(device)
            out = model(xb)[0, 0].float().cpu().numpy().astype(np.float64)
            ix = np.ix_(*[(np.arange(c, c + CROP) % n) for c in
                          (ca, cb, cc)])
            pred_sum[ix] += out * W
            wsum[ix] += W
    if taper == "hann":
        # A1 runtime assertion: accumulated weight uniform to < 1e-6
        assert np.abs(wsum - 1.0).max() < 1e-6, (
            f"Hann COLA FAIL: accumulated weight in "
            f"[{wsum.min():.8f}, {wsum.max():.8f}]")
    else:
        # uniform diagnostic path keeps the 8x-count assert
        assert (wsum == 8.0).all(), (
            f"S7 coverage FAIL: expected uniform 8x, got "
            f"[{wsum.min():g}, {wsum.max():g}]")
    return pred_sum / wsum


def window_interior_mask(n: int = N_GRID) -> np.ndarray:
    """S7 seam diagnostic support: voxels >= 8 from EVERY face of EVERY
    covering window. With stride 32 / crop 64, a voxel's local coord along
    an axis in its two covering windows is (c mod 32) and (c mod 32) + 32;
    both are >= 8 from the faces iff (c mod 32) in [8, 24)."""
    ax = (np.arange(n) % STRIDE >= 8) & (np.arange(n) % STRIDE < 24)
    return ax[:, None, None] & ax[None, :, None] & ax[None, None, :]


def read_val_band_edge():
    """Governing VAL null97.5(sigma=2, real, Pearson) edge + m (K2).

    K2 N-selection rule: if the N=200 edge bootstrap CI95 half-width
    > ~0.02, the N=1000 extension governs. m = CI95 half-width of the
    governing edge. Returns (edge, m, provenance_block); (None, None, ...)
    if the artifact or required fields are absent (-> PENDING-cell).
    """
    f = STAGE2_DIR / "null_band_val_n200.json"
    if not f.exists():
        return None, None, {"file": None, "status": "PENDING-val-band"}
    vb = json.loads(f.read_text())
    try:
        band = vb["band"]["real"]
        null975_band = {s: band[s]["pearson"]["pct_97p5"]
                        for s in ("1", "2", "4")}
        n200 = vb["mc_error_97p5_edge"]["real"]["2"]["pearson"]
        ci200 = n200["boot_ci95"]
        hw200 = (ci200[1] - ci200[0]) / 2.0
        if hw200 > 0.02 and "n1000_extension_sigma2" in vb:
            n1k = vb["n1000_extension_sigma2"]["real"]["pearson"]
            ci1k = n1k["boot_ci95_edge_n1000"]
            edge = float(n1k["pct_97p5_n1000"])
            m_edge = (ci1k[1] - ci1k[0]) / 2.0
            prov = ("N=1000 governs (N=200 edge CI95 half-width "
                    f"{hw200:.5f} > 0.02)")
        else:
            edge = float(n200["edge"])
            m_edge = hw200
            prov = f"N=200 governs (edge CI95 half-width {hw200:.5f})"
    except (KeyError, TypeError):
        return None, None, {"file": str(f.relative_to(REPO)),
                            "status": "PENDING-m (schema fields missing)"}
    return edge, m_edge, {
        "file": str(f.relative_to(REPO)),
        "pct_97p5_real_pearson_n200_band": null975_band,
        "governing_edge_sigma2": edge,
        "m_ci95_half_width": m_edge,
        "edge_provenance": prov,
        "green_floor": edge + m_edge,
        "red_ceiling": edge - m_edge,
    }


# ------------------------------------------------------- quick masked eval


def val_block_slices():
    """8 congruent sub-blocks of the VAL slab [134,163) — R9 geometry
    transplanted: 29 voxels is odd -> trim ONE voxel at the high-index end
    ([134,162), two 14-voxel halves) x 2x2 transverse (96)."""
    lo, hi = 134, 162
    half = (hi - lo) // 2
    ht = N_GRID // 2
    out = []
    for a0, a1 in ((lo, lo + half), (lo + half, hi)):
        for j in (0, 1):
            for k in (0, 1):
                out.append((slice(a0, a1), slice(j * ht, (j + 1) * ht),
                            slice(k * ht, (k + 1) * ht)))
    return out


def quick_masked_eval(model: UNet3D, source: PhysicsSource,
                      device: torch.device,
                      source_p2: Optional[PhysicsSource] = None,
                      untrained_swap: Optional[Dict] = None) -> Dict:
    """s3 quick eval: r_s on the [D-49] VAL mask, real frame, sigma {1,2,4};
    U-G controls z1/z2/z3; descriptive columns. Scoring conventions IMPORTED
    from the R9 machinery, not re-implemented.

    Spec v2 (74684ad) + s2 ruling (7a7f251) shape: K1 VAL-slab reads only
    (region_voxel_interval('val', 192) = [134,163); test mask touched
    exactly twice ever: G2 + G3); S5 controls z1-z4 (z3 = pinned seed-
    20260729 derangement; z4 = P2 flux into the P1 pattern); S4 U-G
    trigger; S7 periodic windows + coverage assert + seam column; K2
    re-banded cell, evaluated ONLY if the VAL band AND its m value exist —
    otherwise raw numbers + PENDING-cell.
    """
    from scripts.d75_corrected_metric_rescore import BOX_MPC_H
    from scripts.u04_r9_heldout_rescore import masked_pearson, masked_spearman
    from src.analysis import nccf as NC
    from src.data.loader import DEFAULT_SCHEME, region_voxel_interval

    EVAL_REGION = "val"                       # pre-G2 smoke reads: VAL only
    assert EVAL_REGION == "val", "pre-G2 eval must never read the test slab"
    lo, hi = region_voxel_interval(EVAL_REGION, N_GRID, DEFAULT_SCHEME)
    assert (lo, hi) == (134, 163), f"unexpected val interval [{lo},{hi})"
    assert DEFAULT_SCHEME.axis == 0
    mask = np.zeros((N_GRID,) * 3, dtype=bool)
    mask[lo:hi] = True

    x_truth = source.provider.x_cube               # exact [D-75] x, float64
    sigmas = (1.0, 2.0, 4.0)
    truth_s = {s: NC.gaussian_smooth_periodic(x_truth, BOX_MPC_H, s)
               for s in sigmas}

    rays_primary = np.arange(1024, dtype=np.int64)   # file-order [0,1024)
    rays_secondary = np.arange(64, dtype=np.int64)   # file-order [0,64)

    # z3 (S5, pinned): GLOBAL seeded DERANGEMENT of whole-ray flux profiles
    # across the eval pattern — geometry fixed, seed 20260729, no ray maps
    # to itself (rejection-resampled until derangement).
    z3_rng = np.random.default_rng(20260729)
    m = rays_primary.size
    perm = z3_rng.permutation(m)
    n_redraws = 0
    while (perm == np.arange(m)).any():
        perm = z3_rng.permutation(m)
        n_redraws += 1
    df_derange = source.delta_f.copy()
    df_derange[rays_primary] = source.delta_f[rays_primary[perm]]

    conditions = {
        "actual": dict(),
        "z1_all_zero_input": dict(
            input_transform=lambda a: np.zeros_like(a)),
        "z2_mask_only": dict(
            input_transform=lambda a: np.stack(
                [np.zeros_like(a[0]), a[1]])),
        "z3_deranged_ray": dict(delta_f=df_derange),
    }
    # z4 (S5, reclassified 3ccf6a2): cross-physics flux swap — P2 flux into
    # the P1 eval pattern. Diagnostic Delta_phys column, NOT a U-G trigger.
    if source_p2 is not None:
        assert np.array_equal(source.geometry.voxel3,
                              source_p2.geometry.voxel3)
        assert np.array_equal(source.geometry.axis, source_p2.geometry.axis)
        df_swap = source.delta_f.copy()
        df_swap[rays_primary] = source_p2.delta_f[rays_primary]
        conditions["z4_cross_physics_swap"] = dict(delta_f=df_swap)
    # R2 (50de53a): untrained-physics swap column (diagnostic, not trigger)
    # untrained_swap = {"label": "z5_untrained_swap_P3", "delta_f": arr}
    if untrained_swap is not None:
        df_u = source.delta_f.copy()
        df_u[rays_primary] = untrained_swap["delta_f"][rays_primary]
        conditions[untrained_swap["label"]] = dict(delta_f=df_u)

    cubes, scores = {}, {}
    for name, kw in conditions.items():
        t0 = time.time()
        pred = sliding_window_predict(model, source, rays_primary, device,
                                      **kw)
        cubes[name] = pred
        entry = {}
        for s in sigmas:
            ps = NC.gaussian_smooth_periodic(pred, BOX_MPC_H, s)
            entry[f"{s:g}"] = {
                "pearson_masked": masked_pearson(truth_s[s], ps, mask),
                "spearman_masked": masked_spearman(truth_s[s], ps, mask),
            }
        scores[name] = entry
        print(f"[eval] {name}: r_s(2,real,masked)="
              f"{entry['2']['pearson_masked']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # uniform-taper diagnostic co-report (A1: hann governs; uniform demoted)
    pred_unif = sliding_window_predict(model, source, rays_primary, device,
                                       taper="uniform")
    unif_s2 = NC.gaussian_smooth_periodic(pred_unif, BOX_MPC_H, 2.0)
    uniform_co_report = {
        "r_s2_val_mask_uniform": masked_pearson(truth_s[2.0], unif_s2, mask),
        "note": "diagnostic co-report only; hann column governs (A1)",
    }

    # secondary pattern (descriptive; no controls, no gate)
    pred64 = sliding_window_predict(model, source, rays_secondary, device)
    p64 = masked_pearson(
        truth_s[2.0], NC.gaussian_smooth_periodic(pred64, BOX_MPC_H, 2.0),
        mask)
    print(f"[eval] secondary [0,64): r_s(2,real,masked)={p64:.4f}",
          flush=True)

    # descriptive anti-degeneracy columns (spec S(c)), on the actual cube
    pred = cubes["actual"]
    pcts = [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
    pred_s2 = NC.gaussian_smooth_periodic(pred, BOX_MPC_H, 2.0)
    descriptive = {
        "var_ratio_unsmoothed_mask": float(np.var(pred[mask])
                                           / np.var(x_truth[mask])),
        "var_ratio_sigma2_mask": float(np.var(pred_s2[mask])
                                       / np.var(truth_s[2.0][mask])),
        "x_pdf_percentiles": pcts,
        "x_pdf_pred_mask": [float(v) for v in
                            np.percentile(pred[mask], pcts)],
        "x_pdf_truth_mask": [float(v) for v in
                             np.percentile(x_truth[mask], pcts)],
    }

    # S7 seam-diagnostic column (actual cube, sigma=2, VAL-mask cut to the
    # all-window-interior voxel set; divergence > 0.02 raises a seam flag)
    interior = window_interior_mask(N_GRID)
    r2 = scores["actual"]["2"]["pearson_masked"]
    r2_interior = masked_pearson(truth_s[2.0], pred_s2, mask & interior)
    seam = {"r_s2_val_mask": r2,
            "r_s2_val_mask_window_interior": r2_interior,
            "interior_definition": "(coord mod 32) in [8,24) on all axes "
                                   "— >=8 from every face of every "
                                   "covering window",
            "n_interior_val_voxels": int((mask & interior).sum()),
            "divergence": abs(r2_interior - r2),
            "seam_flag": bool(abs(r2_interior - r2) > 0.02)}

    # U-G trigger set = z1-z3 ONLY (z4 reclassified, 3ccf6a2/50de53a)
    trigger_keys = ("z1_all_zero_input", "z2_mask_only", "z3_deranged_ray")
    controls_r2 = {k: scores[k]["2"]["pearson_masked"]
                   for k in conditions if k != "actual"}
    trigger_r2 = {k: controls_r2[k] for k in trigger_keys}
    collapse = descriptive["var_ratio_unsmoothed_mask"] < 0.01

    # Delta_phys column (mandatory on every read) + R1 two-sided flag
    delta_phys = None
    if "z4_cross_physics_swap" in scores:
        blocks = val_block_slices()
        z4_s2 = NC.gaussian_smooth_periodic(
            cubes["z4_cross_physics_swap"], BOX_MPC_H, 2.0)
        d_blk = np.array([
            NC.pearson(truth_s[2.0][b], pred_s2[b])
            - NC.pearson(truth_s[2.0][b], z4_s2[b]) for b in blocks])
        d_se = float(np.std(d_blk, ddof=1) / np.sqrt(len(d_blk)))
        d_val = r2 - scores["z4_cross_physics_swap"]["2"]["pearson_masked"]
        delta_phys = {
            "delta_phys_sigma2": d_val,
            "block_deltas": [float(v) for v in d_blk],
            "block_se": d_se,
            "r1_anomaly_flag_two_sided": bool(d_val < -2.0 * d_se),
            "scope_note": "Delta_phys ~ 0 => recovery scoped 'suite-common "
                          "(shared-IC) structure recovered from flux' "
                          "(3ccf6a2)",
        }
    # VAL null band + m (K2): cell evaluates ONLY with both present
    edge, m_edge, null_block = read_val_band_edge()
    # S4 trigger over z1-z3: any >= 0.5 x actual OR any > null97.5(VAL)
    u_g_fired = any(v >= 0.5 * r2 for v in trigger_r2.values())
    if edge is not None:
        u_g_fired = u_g_fired or any(v > edge for v in trigger_r2.values())
    # K2 re-banded cell (VAL slab; requires the governing edge AND m)
    if edge is not None and m_edge is not None:
        if u_g_fired or collapse or r2 <= edge - m_edge:
            cell = "RED"
        elif r2 > edge + m_edge:
            cell = "GREEN"
        else:
            cell = "AMBER"
    else:
        cell = "PENDING-cell (VAL band or m absent)"
    return {
        "mask": {"interval_right_open": [lo, hi], "axis": 0,
                 "region": "val",
                 "source": "region_voxel_interval('val', 192) — "
                           "runtime-asserted; pre-G2 reads never touch "
                           "the test slab (B1 KILLER-1)"},
        "ray_patterns": {"primary": "[0, 1024) file-order",
                         "secondary": "[0, 64) file-order"},
        "scores_real_frame": scores,
        "secondary_pattern_r_s2": p64,
        "descriptive": descriptive,
        "taper": "hann (inference-of-record, micro-cycle A1; uniform = "
                 "diagnostic co-report)",
        "uniform_co_report": uniform_co_report,
        "seam_diagnostic": seam,
        "controls_r_s2": controls_r2,
        "delta_phys": delta_phys,
        "u_g_fired_S4_rule": u_g_fired,
        "u_g_rule": "over z1-z3 ONLY (z4 reclassified 3ccf6a2): any >= "
                    "0.5 x actual OR any > null97.5(VAL)",
        "variance_collapse": collapse,
        "z3_derangement": {"seed": 20260729, "n_redraws": n_redraws,
                           "rule": "global derangement of whole-ray flux "
                                   "profiles, geometry fixed, no self-maps"},
        "null_band": null_block,
        "cell": cell,
        "cell_rule": "K2 re-band (spec v2): GREEN r>null975(VAL)+m & U-G "
                     "clean; AMBER in (null975-m, null975+m]; RED "
                     "r<=null975-m or var<0.01 or U-G",
    }


# ------------------------------------------------------------------ s3 run


def run_s3(steps: int = 500, lr: float = 3.0e-4) -> Dict:
    t0 = time.time()
    device = pick_device()
    torch.manual_seed(SEED)
    if device.type == "mps":
        batch_size, accum = 4, 1
    else:
        batch_size, accum = 2, 2               # CPU fallback, effective 4
    sources = build_sources([1, 2])
    ds = UNetPairDataset(sources, length=4096, seed=SEED, augment=True)
    val_ds = UNetPairDataset(sources, length=8, seed=VAL_SEED, augment=False)
    val_batch = stack_batch(val_ds, range(8), device)
    model = UNet3D().to(device)
    n_params = model.n_parameters()
    tags = mandatory_tags("1+2", SEED)
    with Tracker("Stage2-MiniRun500", tags) as trk:
        trk.log_params({"lr": lr, "steps": steps, "batch": batch_size,
                        "grad_accum": accum, "n_params": n_params,
                        "device": str(device), "optimizer": "AdamW",
                        "weight_decay": 1e-4, "grad_clip": 1.0,
                        "dataset_length": 4096, "augment": True,
                        "n_rays": "log-uniform [64, 1024]",
                        "lr_schedule": "constant (smoke)"})
        losses = train_s3(model, ds, val_batch, trk, device, steps, lr,
                          batch_size, accum)
        # loss-trend gate (spec S(d)): smoothed = trailing-10-step mean
        mse10 = float(np.mean(losses[0:10]))
        mse500 = float(np.mean(losses[steps - 10:steps]))
        trend_pass = mse500 <= 0.7 * mse10
        trk.log_metric(steps, "loss_trend_ratio", mse500 / mse10)
        print(f"[s3] trend gate: mse(500)={mse500:.4f} vs 0.7*mse(10)="
              f"{0.7 * mse10:.4f} -> {'PASS' if trend_pass else 'FAIL'}",
              flush=True)
        pstd_500 = val_pred_std(model, val_batch)
        trk.log_metric(steps, "val_pred_std", pstd_500)
        ckpt = STAGE2_DIR / "s3_model_step500.pt"
        torch.save(model.state_dict(), ckpt)
        # quick masked eval on P1 (trained physics), primary ray pattern
        eval_block = quick_masked_eval(model, sources[0], device,
                                       source_p2=sources[1])
        trk.log_metric(steps, "eval_r_s2_real_masked",
                       eval_block["scores_real_frame"]["actual"]["2"]
                       ["pearson_masked"])
    record = {
        "rung": "A7 (s3) — 500-step mini-run + quick masked eval",
        "spec": "experiments/unet-inversion/design/u06_stage2_spec.md "
                "S(b)/(c)/(d)",
        "session_utc": "2026-07-24",
        "config": {"physics": ["P1", "P2"], "seed": SEED, "lr": lr,
                   "steps": steps, "batch": batch_size, "grad_accum": accum,
                   "effective_batch": batch_size * accum,
                   "dataset_length": 4096, "augment": True,
                   "n_rays": "log-uniform [64, 1024]",
                   "optimizer": "AdamW", "weight_decay": 1e-4,
                   "grad_clip": 1.0, "lr_schedule": "constant (smoke)",
                   "val_batch": {"n": 8, "seed": VAL_SEED, "augment": False}},
        "model": {"arch": "UNet3D 4-level base-32 GN8+SiLU",
                  "n_params_exact": n_params,
                  "checkpoint": str(ckpt.relative_to(REPO))},
        "device": str(device),
        "dtype": "float32",
        "losses": losses,
        "contract_step100": {
            "rule": "val pred std > 0.01 AND loss(100) < loss(1); "
                    "asserted inside the loop, outside try/except",
            "verdict": "PASS (run reached this line — assertion raises loud "
                       "on fail)",
            "loss_1": losses[0], "loss_100": losses[99]},
        "val_pred_std_step500": pstd_500,
        "loss_trend_gate": {
            "rule": "smoothed MSE(500) <= 0.7 * MSE(10); smoothed = "
                    "trailing-10-step mean",
            "mse_10": mse10, "mse_500": mse500, "ratio": mse500 / mse10,
            "verdict": "PASS" if trend_pass else "FAIL"},
        "quick_masked_eval": eval_block,
        "wall_clock_s": time.time() - t0,
        "git_commit": git_commit_hash(),
    }
    out = STAGE2_DIR / "s3_minirun_record.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"[s3] cell={eval_block['cell']} -> {out} "
          f"({record['wall_clock_s']:.0f}s)", flush=True)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["s2", "s3"])
    args = ap.parse_args()
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "s2":
        rec = run_s2()
        return 0 if rec["gate"]["verdict"] == "PASS" else 1
    rec = run_s3()
    return 0 if rec["loss_trend_gate"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

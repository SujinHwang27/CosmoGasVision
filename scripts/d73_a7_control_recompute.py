"""[D-73] A7 — healthy-run var_pf control + production xi recompute (read-only).

Two deliverables, recomputed from LOCAL production checkpoints because the
production pub-t1 MLflow run (31acdf9d900e447081e6d051f7d42c0e) predates the
`l1_var_pf_band_ratio` instrumentation at experiments/nerf/pipeline.py:3073
(that metric is gated behind `args.enable_l1_pf_loss`, OFF for the pub-t1
ablation matrix). Confirmed ABSENT by enumerating the run's on-disk metrics
file-store. Fallback per [D-73] §A item A7 = recompute from the checkpoint.

(a) var_pf_band_ratio: replicates the pipeline definition EXACTLY
    (pipeline.py:3047-3060):
        F_pred  = exp(-clamp_max(tau_pred, tau_max))        # rendered from ckpt
        F_truth = exp(-clamp_max(tau_gt,   tau_max))
        _, P_pred  = torch_p_flux(F_pred,  vel_axis)         # src.training.p_flux_loss
        _, P_truth = torch_p_flux(F_truth, vel_axis)
        band = (centers >= K_MIN_INERTIAL) & (centers <= K_MAX_INERTIAL)
        Pp = P_pred.float64.mean(dim=0)[band]                # ray-average inside
        Pt = P_truth.float64.mean(dim=0)[band]
        var_pf_ratio = Var_k(Pp) / max(Var_k(Pt), 1e-30)
    tau_max from the run params (10.0). Evaluated at every available
    production checkpoint step.

(b) production xi surrogate r_rho^log: replicates scripts/proxy_xi_1d_sample.py
    EXACTLY (per-row Pearson r between predicted and GT log10(rho/<rho>)
    sampled at simulator bin centers; median across rays). This is the
    [D-58] production metric (1D-along-ray r_rho^log surrogate).

Read-only. No training. Writes JSON with provenance to
experiments/nerf/artifacts/d73_a7_control/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import SherwoodLoader  # noqa: E402
from src.models.nerf import IGMNeRF, volume_render_physics  # noqa: E402
from src.training.p_flux_loss import (  # noqa: E402
    torch_p_flux,
    K_MIN_INERTIAL,
    K_MAX_INERTIAL,
)

# Production pub-t1 P1 run geometry, from the run params/tags file-store
# (cloud_runs/pub-t1-extracted/.../31acdf9d.../params,tags).
RUN_ID = "31acdf9d900e447081e6d051f7d42c0e"
RUN_DIR = REPO_ROOT / "cloud_runs" / "pub-t1-extracted" / "P1-N64-S0-1778430089-7f65fe"
CKPT_DIR = RUN_DIR / "checkpoints"
TAU_MAX = 10.0          # params/tau_max
N_RAYS_RUN = 64         # params/n_rays  (this run is the N64 ablation cell)
HIDDEN_DIM = 256        # params/hidden_dim
NUM_LAYERS = 8          # params/num_layers
L_FOURIER = 10          # params/L_fourier
PHYSICS_ID = 1          # tags/physics_id
REDSHIFT = 0.3          # tags/redshift

OUT_DIR = REPO_ROOT / "experiments" / "nerf" / "artifacts" / "d73_a7_control"

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def _provenance(extra: dict) -> dict:
    base = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/d73_a7_control_recompute.py",
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "data_root": "Sherwood",
        "geometry": {
            "physics_id": PHYSICS_ID,
            "redshift": REDSHIFT,
            "tau_max": TAU_MAX,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "L_fourier": L_FOURIER,
        },
        "metric_def_var_pf": {
            "code": "experiments/nerf/pipeline.py:3047-3060 "
                    "+ src/training/p_flux_loss.py:torch_p_flux",
            "K_MIN_INERTIAL": float(K_MIN_INERTIAL),
            "K_MAX_INERTIAL": float(K_MAX_INERTIAL),
            "definition": "Var_k(mean_rays(P_F_pred)[band]) / "
                          "max(Var_k(mean_rays(P_F_truth)[band]), 1e-30)",
        },
        "metric_def_xi": {
            "code": "scripts/proxy_xi_1d_sample.py:_pearson_per_row (median over rays)",
            "name": "r_rho^log (1D-along-ray Pearson on log10(rho/<rho>), [D-58])",
        },
    }
    base.update(extra)
    return base


def _load_model(ckpt_path: Path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = IGMNeRF(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, L=L_FOURIER)
    model.load_state_dict(state["model_state"])
    model.eval()
    log_tau_amp = torch.tensor(float(state["log_tau_amp"]), dtype=torch.float32)
    return model, log_tau_amp, int(state.get("step", -1)), state.get("mlflow_run_id")


def _load_geometry(n_rays: int):
    """Load P1 z=0.3 sightlines; first n_rays rays (= production slice order).

    The pub-t1 default-OFF path takes `coords_raw[:n_rays]` (pipeline.py:1518),
    NOT a random subsample. We mirror that exact slice so the var_pf is on the
    same rays the run trained on.
    """
    loader = SherwoodLoader(str(REPO_ROOT / "Sherwood"))
    sl = loader.load_sightlines(PHYSICS_ID, REDSHIFT)
    box_max = float(sl["header"]["box_kpc_h"])
    coords_raw = loader.get_world_coordinates(sl)  # (N,n_bins,3) world kpc/h
    coords = torch.tensor(coords_raw[:n_rays], dtype=torch.float32) / box_max
    vel_axis = torch.tensor(sl["vel_axis"], dtype=torch.float32)
    tau_gt = torch.tensor(sl["tau_h1"][:n_rays], dtype=torch.float32)
    rho_gt = np.asarray(sl["density"][:n_rays], dtype=np.float64)
    return coords, vel_axis, tau_gt, rho_gt, box_max


def _render_F_pred(model, coords, vel_axis, log_tau_amp):
    with torch.no_grad():
        tau_amp = torch.exp(log_tau_amp)
        tau_pred = volume_render_physics(
            model, coords, vel_axis=vel_axis, tau_amp=tau_amp,
        )
        tau_pred_capped = tau_pred.clamp_max(TAU_MAX)
        F_pred = torch.exp(-tau_pred_capped)
    return F_pred, tau_pred


def _var_pf_band_ratio(F_pred, F_truth, vel_axis):
    """EXACT replication of pipeline.py:3052-3060."""
    centers, P_pred = torch_p_flux(F_pred, vel_axis)
    _, P_truth = torch_p_flux(F_truth, vel_axis)
    band = (centers >= K_MIN_INERTIAL) & (centers <= K_MAX_INERTIAL)
    if not bool(band.any()):
        return float("nan"), {}
    Pp = P_pred.to(torch.float64).mean(dim=0)[band]
    Pt = P_truth.to(torch.float64).mean(dim=0)[band]
    var_pred = float(Pp.var().item())
    var_truth = float(Pt.var().item())
    ratio = var_pred / max(var_truth, 1e-30)
    diag = {
        "var_pf_pred_band": var_pred,
        "var_pf_truth_band": var_truth,
        "n_band_bins": int(band.sum().item()),
        "P_pred_band_mean": [float(v) for v in Pp.tolist()],
        "P_truth_band_mean": [float(v) for v in Pt.tolist()],
    }
    return ratio, diag


def _pearson_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64); b = b.astype(np.float64)
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def _predict_rho(model, coords) -> np.ndarray:
    n_rays = coords.shape[0]
    out = np.empty((n_rays, coords.shape[1]), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n_rays, 64):
            sl = slice(i, min(i + 64, n_rays))
            fields = model(coords[sl])
            out[sl] = fields[..., 0].cpu().numpy().astype(np.float64)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xi-n-rays", type=int, default=1024,
                   help="N rays for the xi surrogate (production fiducial=1024).")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- enumerate available checkpoints (skip ExFAT ._ junk) ----
    ckpts = sorted(
        g for g in glob.glob(str(CKPT_DIR / "step_*.pt"))
        if not os.path.basename(g).startswith("._")
    )
    avail_steps = []
    for c in ckpts:
        m = re.search(r"step_(\d+)\.pt$", os.path.basename(c))
        if m:
            avail_steps.append(int(m.group(1)))
    avail_steps.sort()
    print(f"[A7] available production checkpoint steps: {avail_steps}", flush=True)

    requested_steps = [200, 1000, 5000, 50000]
    missing_steps = [s for s in requested_steps if s not in avail_steps]

    # ============ (a) var_pf_band_ratio control ============
    coords, vel_axis, tau_gt, rho_gt_n64, _ = _load_geometry(N_RAYS_RUN)
    F_truth = torch.exp(-tau_gt.clamp_max(TAU_MAX))

    var_pf_rows = []
    for step in avail_steps:
        ckpt = CKPT_DIR / f"step_{step:06d}.pt"
        model, log_tau_amp, ck_step, ck_run = _load_model(ckpt)
        F_pred, tau_pred = _render_F_pred(model, coords, vel_axis, log_tau_amp)
        ratio, diag = _var_pf_band_ratio(F_pred, F_truth, vel_axis)
        row = {
            "step": step,
            "ckpt": str(ckpt.relative_to(REPO_ROOT)),
            "ckpt_internal_step": ck_step,
            "ckpt_mlflow_run_id": ck_run,
            "log_tau_amp": float(log_tau_amp.item()),
            "tau_amp": float(torch.exp(log_tau_amp).item()),
            "var_pf_band_ratio": ratio,
            "mean_F_pred": float(F_pred.mean().item()),
            "var_F_pred": float(F_pred.var().item()),
            "var_F_truth": float(F_truth.var().item()),
            **diag,
        }
        var_pf_rows.append(row)
        print(f"[A7-a] step {step}: var_pf_band_ratio={ratio:.4e} "
              f"<F_pred>={row['mean_F_pred']:.4f} "
              f"(run_id match={ck_run == RUN_ID})", flush=True)

    var_pf_out = {
        "deliverable": "[D-73] A7(a) healthy-run var_pf_band_ratio control",
        "provenance": _provenance({
            "n_rays_used": N_RAYS_RUN,
            "F_truth_var": float(F_truth.var().item()),
            "command": "PYTHONPATH=. ~/.venvs/cosmogasvision/bin/python -u "
                       "scripts/d73_a7_control_recompute.py",
        }),
        "mlflow_metric_status": {
            "metric": "l1_var_pf_band_ratio",
            "present_in_run_filestore": False,
            "reason": "production run predates sprint-L1 instrumentation; "
                      "metric gated behind args.enable_l1_pf_loss (OFF for pub-t1). "
                      "Verified absent by listing the run metrics/ dir.",
            "run_metrics_present": [
                "grad_norm", "grad_norm_clipped", "loss", "loss_data",
                "loss_meanF", "lr", "mean_flux_pred", "peak_vram_gb", "tau_amp",
            ],
        },
        "requested_steps": requested_steps,
        "available_checkpoint_steps": avail_steps,
        "missing_requested_steps": missing_steps,
        "missing_steps_note": (
            "Production checkpoints are emitted every 5000 steps starting at "
            "5000; steps 200 and 1000 have NO local checkpoint -> not "
            "recomputable. The 7-lever-table retired runs at step 200; this "
            "control therefore supplies the var_pf trajectory at the production "
            "checkpoint cadence (5000..50000), NOT a step-200 head-to-head. "
            "Any 7-lever-table citation must carry this cadence caveat."
        ),
        "var_pf_trajectory": var_pf_rows,
    }
    out_a = OUT_DIR / "a7a_var_pf_control.json"
    out_a.write_text(json.dumps(var_pf_out, indent=2))
    print(f"[A7-a] wrote {out_a}", flush=True)

    # ============ (b) production xi surrogate r_rho^log ============
    # Production fiducial xi geometry = N1024 (per [D-13] fiducial n_rays=1024).
    # We evaluate the A7-named run's step_050000 checkpoint on n_rays=1024
    # rays, replicating proxy_xi_1d_sample.py (first-N slice for determinism,
    # NOT the random subsample the throwaway used; we record both for honesty).
    final_step = max(avail_steps)
    ckpt_final = CKPT_DIR / f"step_{final_step:06d}.pt"
    model, log_tau_amp, _, ck_run = _load_model(ckpt_final)

    xi_n = args.xi_n_rays
    coords_xi, _, _, rho_gt_xi, box_max = _load_geometry(xi_n)
    rho_pred_xi = _predict_rho(model, coords_xi)
    floor = 1e-6
    rho_pred_pos = np.maximum(rho_pred_xi, floor)
    rho_gt_pos = np.maximum(rho_gt_xi, floor)
    r_lin = _pearson_per_row(rho_pred_xi, rho_gt_xi)
    r_log = _pearson_per_row(np.log10(rho_pred_pos), np.log10(rho_gt_pos))

    def _summ(r):
        f = np.isfinite(r)
        n = int(f.sum())
        return {
            "n_valid": n,
            "median": float(np.median(r[f])) if n else float("nan"),
            "q16": float(np.quantile(r[f], 0.16)) if n else float("nan"),
            "q25": float(np.quantile(r[f], 0.25)) if n else float("nan"),
            "q75": float(np.quantile(r[f], 0.75)) if n else float("nan"),
            "q84": float(np.quantile(r[f], 0.84)) if n else float("nan"),
        }

    s_lin, s_log = _summ(r_lin), _summ(r_log)
    print(f"[A7-b] r_rho^log median={s_log['median']:+.4f} "
          f"IQR=[{s_log['q25']:+.4f},{s_log['q75']:+.4f}] N={s_log['n_valid']}",
          flush=True)
    print(f"[A7-b] r_rho^lin median={s_lin['median']:+.4f}", flush=True)

    xi_out = {
        "deliverable": "[D-73] A7(b) production xi(r=2 Mpc/h) surrogate",
        "provenance": _provenance({
            "ckpt": str(ckpt_final.relative_to(REPO_ROOT)),
            "ckpt_mlflow_run_id": ck_run,
            "xi_n_rays": xi_n,
            "rho_pred_floor": floor,
            "ray_selection": "first-N (coords_raw[:n]); deterministic, matches "
                             "pub-t1 default-OFF slice order. The [D-58] "
                             "throwaway used a seeded random subsample; "
                             "estimator identical, ray set differs.",
            "command": "PYTHONPATH=. ~/.venvs/cosmogasvision/bin/python -u "
                       "scripts/d73_a7_control_recompute.py",
        }),
        "estimator_name": "r_rho^log (1D-along-ray Pearson on log10(rho/<rho>))",
        "estimator_note": (
            "This is the production metric per [D-58]: a 1D-along-ray "
            "zero-lag Pearson surrogate, NOT the [D-13] 3D "
            "xi_{rho_hat,rho}(r=2 h^-1 Mpc) FFT-shell estimator "
            "(src/analysis/cross_corr.py:compute_xi_pearson). The 3D measure "
            "requires a reconstructed 3D rho cube; the production model is "
            "supervised on 1D flux sightlines only, so no 3D cube exists to "
            "feed the [D-13] gate evaluator. The 3D measurement was deferred "
            "per A2/[D-23] (pending the ~40 GB SherwoodIGM_gal extraction)."
        ),
        "r_rho_log": s_log,
        "r_rho_lin": s_lin,
        "comparison_to_published": (
            "LEDGER cites production r_rho^log=+0.077 from the T3 N1024 "
            "checkpoint (cloud_runs/prong3-p1-t3/.../step_010000.pt, run "
            "f74dbb669c9641568ab883023a84d1fa). THIS number is from the A7-named "
            "pub-t1 P1-N64 run 31acdf9d step_050000 on n_rays=1024 -- a "
            "DIFFERENT checkpoint; values are not expected to be identical."
        ),
    }
    out_b = OUT_DIR / "a7b_production_xi.json"
    out_b.write_text(json.dumps(xi_out, indent=2))
    print(f"[A7-b] wrote {out_b}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

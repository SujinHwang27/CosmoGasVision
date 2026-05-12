"""[D-35] KS-shift overlay: predicted vs truth flux PDF, as-is vs anchor-rescaled.

For each physics in {1, 2, 4} at the T3 fiducial cost-survey schedule:
  - render predicted F = exp(-tau_pred) on N=1024 held-out sightlines
  - load truth F from Sherwood tauH1 file
  - compute F_rescaled = (0.979/<F_pred>) * F_pred (clipped to (eps, 1-eps))
  - histogram F restricted to [0.05, 0.95] gate window
  - report KS distance (as-is) and KS distance (rescaled) per panel

Produces a 1x3 figure showing the histogram shift visualization that PI
ruling 4 / [D-35] commissioned for the appendix. Numerics are cross-checked
against the eval_anchor_invariance_d34.py table.

PCV: refuses non-32-hex run-id; refuses missing checkpoints.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import SherwoodLoader  # noqa: E402
from src.analysis.stage2b_report import (  # noqa: E402
    _build_model_from_run,
    _load_mlflow_run,
    _render_tau_for_model,
)

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class CellSpec:
    physics_id: int
    label: str
    run_id: str
    ckpt_path: str


# Defaults: T3 fiducials at step 10000/12500, the same checkpoints
# eval_anchor_invariance_d34.py used. Source: LEDGER [D-35] cross-physics table.
DEFAULTS = [
    CellSpec(
        physics_id=1,
        label="P1 (no feedback)",
        run_id="f74dbb669c9641568ab883023a84d1fa",
        ckpt_path=str(
            REPO_ROOT / "cloud_runs" / "prong3-p1-t3"
            / "P1-N1024-S0-1778229084-c08848" / "checkpoints" / "step_010000.pt"
        ),
    ),
    CellSpec(
        physics_id=2,
        label="P2 (stellar wind)",
        run_id="240f2ea4502146a58ae4d038ad8e067f",
        ckpt_path=str(
            REPO_ROOT / "cloud_runs" / "batch3b-rescue" / "P2-S1" / "step_010000.pt"
        ),
    ),
    CellSpec(
        physics_id=4,
        label="P4 (strong AGN)",
        run_id="11452b3925e04e75af4bc7e04b37f1a9",
        ckpt_path=str(
            REPO_ROOT / "cloud_runs" / "batch3b-rescue" / "P4-S1" / "step_010000.pt"
        ),
    ),
]


def _assert_run_id(run_id: str) -> None:
    if not _HEX32.match(run_id):
        sys.exit(
            f"FATAL [PCV]: run-id {run_id!r} is not a 32-hex MLflow id."
        )


def _ks_distance_window(
    F_pred: np.ndarray, F_truth: np.ndarray, lo: float = 0.05, hi: float = 0.95,
) -> float:
    """Two-sample KS distance on flux samples restricted to [lo, hi]."""
    p = F_pred.ravel()
    t = F_truth.ravel()
    p = p[(p >= lo) & (p <= hi)]
    t = t[(t >= lo) & (t <= hi)]
    if p.size == 0 or t.size == 0:
        return float("nan")
    p_sorted = np.sort(p)
    t_sorted = np.sort(t)
    grid = np.union1d(p_sorted, t_sorted)
    cdf_p = np.searchsorted(p_sorted, grid, side="right") / p_sorted.size
    cdf_t = np.searchsorted(t_sorted, grid, side="right") / t_sorted.size
    return float(np.max(np.abs(cdf_p - cdf_t)))


def _render_chunked(
    model, coords: torch.Tensor, vel_axis_t: torch.Tensor, chunk_rays: int = 32,
) -> np.ndarray:
    chunks = []
    for i in range(0, coords.shape[0], chunk_rays):
        sl = slice(i, min(i + chunk_rays, coords.shape[0]))
        chunks.append(_render_tau_for_model(model, coords[sl], vel_axis_t))
    return np.concatenate(chunks, axis=0)


def _process_cell(
    spec: CellSpec, n_rays_eval: int, seed: int, target_mean_F: float,
    chunk_rays: int,
) -> dict:
    _assert_run_id(spec.run_id)
    if not os.path.exists(spec.ckpt_path):
        sys.exit(f"FATAL: checkpoint {spec.ckpt_path!r} does not exist.")

    sherwood = SherwoodLoader(str(REPO_ROOT / "Sherwood"))
    sl = sherwood.load_sightlines(spec.physics_id, 0.3)
    tau_truth_full = np.asarray(sl["tau_h1"], dtype=np.float64)
    pos_axis = np.asarray(sl["pos_axis"], dtype=np.float64)
    box_kpc_h = float(sl["header"]["box_kpc_h"])
    vel_axis = np.asarray(sl["vel_axis"], dtype=np.float64)

    n_rays_avail = tau_truth_full.shape[0]
    n_rays = min(n_rays_eval, n_rays_avail)
    rng = np.random.default_rng(seed=seed)
    sel = rng.choice(n_rays_avail, size=n_rays, replace=False)
    sel.sort()
    tau_truth = tau_truth_full[sel]

    run, _ = _load_mlflow_run(spec.run_id)
    model = _build_model_from_run(run, spec.ckpt_path)
    coords_world = sherwood.get_world_coordinates(sl)
    coords_unit = (coords_world[sel] / box_kpc_h).astype(np.float64)
    coords = torch.tensor(coords_unit, dtype=torch.float32)
    vel_axis_t = torch.tensor(vel_axis, dtype=torch.float32)

    print(f"[d35-overlay] rendering {spec.label} ({n_rays} rays) ...")
    tau_pred = _render_chunked(model, coords, vel_axis_t, chunk_rays=chunk_rays)
    assert tau_pred.shape == tau_truth.shape

    F_truth = np.exp(-tau_truth)
    F_pred = np.exp(-tau_pred)
    mean_F_pred = float(F_pred.mean())
    r = float(target_mean_F) / mean_F_pred
    eps = 1e-9
    F_rescaled = np.clip(r * F_pred, eps, 1.0 - eps)

    ks_asis = _ks_distance_window(F_pred, F_truth)
    ks_resc = _ks_distance_window(F_rescaled, F_truth)

    print(
        f"[d35-overlay] {spec.label}: <F_pred>={mean_F_pred:.4f}  "
        f"r={r:.4f}  KS_as-is={ks_asis:.4f}  KS_rescaled={ks_resc:.4f}"
    )
    return {
        "spec": spec,
        "F_truth": F_truth.ravel(),
        "F_pred": F_pred.ravel(),
        "F_rescaled": F_rescaled.ravel(),
        "mean_F_pred": mean_F_pred,
        "rescale_factor": r,
        "ks_asis": ks_asis,
        "ks_rescaled": ks_resc,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--chunk-rays", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-mean-flux", type=float, default=0.979)
    p.add_argument(
        "--out-png",
        default=str(REPO_ROOT / "papers" / "shared" / "figures" / "ks_shift_overlay_d35.png"),
    )
    args = p.parse_args()

    results = []
    for spec in DEFAULTS:
        results.append(
            _process_cell(
                spec,
                n_rays_eval=args.n_rays_eval,
                seed=args.seed,
                target_mean_F=args.target_mean_flux,
                chunk_rays=args.chunk_rays,
            )
        )

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    bins = np.linspace(0.05, 0.95, 60)
    for ax, res in zip(axes, results):
        spec = res["spec"]
        ax.hist(
            res["F_truth"], bins=bins, density=True, histtype="step",
            color="k", lw=1.6, label="GT (truth)",
        )
        ax.hist(
            res["F_pred"], bins=bins, density=True, histtype="step",
            color="C3", lw=1.4, ls="--", label=f"Pred as-is (KS={res['ks_asis']:.3f})",
        )
        ax.hist(
            res["F_rescaled"], bins=bins, density=True, histtype="step",
            color="C0", lw=1.4, ls="-", label=f"Pred rescaled (KS={res['ks_rescaled']:.3f})",
        )
        verdict = "PASS" if res["ks_rescaled"] < 0.05 else "fail"
        ax.set_title(
            f"{spec.label}\n"
            f"$\\langle\\hat F\\rangle$={res['mean_F_pred']:.3f}, "
            f"$r$={res['rescale_factor']:.3f}, rescaled-KS {verdict}",
            fontsize=9.5,
        )
        ax.set_xlabel(r"$F = e^{-\tau}$")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.05, 0.95)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel(r"density (per unit $F$)")

    fig.suptitle(
        r"[D-35] Flux-PDF shift under uniform anchor rescale "
        r"$\hat F \to (0.979/\langle\hat F\rangle)\cdot \hat F$ "
        f"(T3 fiducial, $z=0.3$, $N={args.n_rays_eval}$ sightlines, $F\\in[0.05,0.95]$)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[d35-overlay] figure -> {out_png}")

    print()
    print("=== [D-35] KS-shift overlay summary ===")
    print(f"  {'physics':<20} {'KS as-is':>10} {'KS rescaled':>14} {'rescaled gate (0.05)':>22}")
    print(f"  {'-'*20} {'-'*10} {'-'*14} {'-'*22}")
    for res in results:
        verdict = "PASS" if res["ks_rescaled"] < 0.05 else "fail"
        print(
            f"  {res['spec'].label:<20} {res['ks_asis']:>10.4f} "
            f"{res['ks_rescaled']:>14.4f} {verdict:>22}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

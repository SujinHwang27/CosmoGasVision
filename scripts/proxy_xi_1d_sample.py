"""Phase A.2 throwaway: 1D density-along-sightline proxy [D-33] sample.

Per-sightline Pearson r between predicted and GT rho/<rho> sampled at the
simulator bin centers (= integrator quadrature points; no resampling).
Honest framing: necessary but not sufficient for full 3D xi_{rho_hat,rho}.

PCV: refuses to run if --run-id is not a 32-hex MLflow run id, and refuses
to evaluate on a missing checkpoint. Single-use evidence for Phase A.2 memo.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import SherwoodLoader  # noqa: E402
from src.models.nerf import IGMNeRF  # noqa: E402

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def _assert_run_id(run_id: str) -> None:
    if not _HEX32.match(run_id):
        sys.exit(
            f"FATAL [PCV]: --run-id {run_id!r} is not a 32-hex MLflow run id. "
            f"Refusing to evaluate -- the [D-33] proxy must be tied to a real "
            f"banked checkpoint, not a smoke / random-init."
        )


def _load_model(ckpt_path: str, hidden_dim: int, num_layers: int, L: int) -> IGMNeRF:
    if not os.path.exists(ckpt_path):
        sys.exit(f"FATAL: checkpoint {ckpt_path!r} does not exist on disk.")
    model = IGMNeRF(hidden_dim=hidden_dim, num_layers=num_layers, L=L)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state", "model_state_dict"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    model.eval()
    print(f"[proxy-xi-1d] loaded weights from {ckpt_path}")
    return model


def _predict_rho_along_rays(
    model: IGMNeRF,
    coords_unit: np.ndarray,
    chunk_rays: int = 64,
) -> np.ndarray:
    n_rays, n_bins, _ = coords_unit.shape
    out = np.empty((n_rays, n_bins), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n_rays, chunk_rays):
            sl = slice(i, min(i + chunk_rays, n_rays))
            x = torch.from_numpy(coords_unit[sl]).to(torch.float32)
            fields = model(x)
            out[sl] = fields[..., 0].cpu().numpy().astype(np.float64)
    return out


def _pearson_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    a = a.astype(np.float64); b = b.astype(np.float64)
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--run-id",
        default="f74dbb669c9641568ab883023a84d1fa",
        help="MLflow run id (32 hex chars). Default = T3-P1 fiducial seed=0.",
    )
    p.add_argument(
        "--ckpt-path",
        default=str(
            REPO_ROOT / "cloud_runs" / "prong3-p1-t3"
            / "P1-N1024-S0-1778229084-c08848" / "checkpoints" / "step_010000.pt"
        ),
    )
    p.add_argument("--physics-id", type=int, default=1)
    p.add_argument("--redshift", type=float, default=0.3)
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--L-fourier", type=int, default=10)
    p.add_argument(
        "--out-png",
        default=str(REPO_ROOT / "paper_cvpr" / "figures" / "proxy_xi_1d_sample.png"),
    )
    args = p.parse_args()

    _assert_run_id(args.run_id)

    sherwood = SherwoodLoader(str(REPO_ROOT / "Sherwood"))
    sl = sherwood.load_sightlines(args.physics_id, args.redshift)
    rho_gt_full = np.asarray(sl["density"], dtype=np.float64)
    pos_axis = np.asarray(sl["pos_axis"], dtype=np.float64)
    box_kpc_h = float(sl["header"]["box_kpc_h"])

    n_rays_avail, n_bins = rho_gt_full.shape
    n_rays = min(args.n_rays_eval, n_rays_avail)

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(n_rays_avail, size=n_rays, replace=False)
    sel.sort()
    rho_gt = rho_gt_full[sel]

    assert np.all(np.isfinite(rho_gt)), "GT rho contains NaN/inf"
    assert np.all(rho_gt >= 0), "GT rho has negative entries"

    coords_world = sherwood.get_world_coordinates(sl)
    coords_unit = (coords_world[sel] / box_kpc_h).astype(np.float64)
    assert coords_unit.shape == (n_rays, n_bins, 3)
    assert (coords_unit.min() >= 0.0) and (coords_unit.max() <= 1.0 + 1e-9), \
        f"coords_unit out of [0,1]: [{coords_unit.min()}, {coords_unit.max()}]"

    model = _load_model(args.ckpt_path, args.hidden_dim, args.num_layers, args.L_fourier)
    rho_pred = _predict_rho_along_rays(model, coords_unit)
    assert rho_pred.shape == rho_gt.shape, \
        f"shape mismatch: pred {rho_pred.shape} vs GT {rho_gt.shape}"
    assert np.all(np.isfinite(rho_pred)), "Predicted rho contains NaN/inf"

    floor = 1e-6
    rho_pred_pos = np.maximum(rho_pred, floor)
    rho_gt_pos = np.maximum(rho_gt, floor)

    r_lin = _pearson_per_row(rho_pred, rho_gt)
    r_log = _pearson_per_row(np.log10(rho_pred_pos), np.log10(rho_gt_pos))

    def _summary(r: np.ndarray, label: str) -> dict:
        finite = np.isfinite(r)
        n_valid = int(finite.sum())
        med = float(np.median(r[finite])) if n_valid else float("nan")
        q16 = float(np.quantile(r[finite], 0.16)) if n_valid else float("nan")
        q84 = float(np.quantile(r[finite], 0.84)) if n_valid else float("nan")
        q25 = float(np.quantile(r[finite], 0.25)) if n_valid else float("nan")
        q75 = float(np.quantile(r[finite], 0.75)) if n_valid else float("nan")
        print(
            f"[proxy-xi-1d] {label}: N={n_valid}/{r.size}  median r_rho={med:+.4f}  "
            f"IQR=[{q25:+.4f}, {q75:+.4f}]  16/84=[{q16:+.4f}, {q84:+.4f}]"
        )
        return {"n": n_valid, "median": med, "q16": q16, "q25": q25,
                "q75": q75, "q84": q84}

    s_lin = _summary(r_lin, "linear rho/<rho>")
    s_log = _summary(r_log, "log10 rho/<rho>")

    mean_log = np.log10(rho_gt_pos).mean(axis=1)
    var_log = np.log10(rho_gt_pos).var(axis=1)
    idx_low = int(np.argsort(mean_log)[max(1, n_rays // 20)])
    idx_mid = int(np.argsort(np.abs(mean_log - np.median(mean_log)))[0])
    idx_high = int(np.argsort(mean_log)[-max(1, n_rays // 20)])
    mixed_order = np.argsort(-var_log)
    picked = {idx_low, idx_mid, idx_high}
    idx_mixed = next(int(i) for i in mixed_order if int(i) not in picked)

    panels = [
        ("low-density",   idx_low),
        ("mean-density",  idx_mid),
        ("high-density",  idx_high),
        ("mixed (high-var)", idx_mixed),
    ]

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    ell_mpc_h = pos_axis / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True)
    axes = axes.ravel()
    for ax, (label, i) in zip(axes, panels):
        ax.semilogy(ell_mpc_h, rho_gt_pos[i], "k-", lw=1.4, label="GT")
        ax.semilogy(ell_mpc_h, rho_pred_pos[i], "C0--", lw=1.2, label="Predicted")
        title = (
            f"{label} (ray {i})  |  "
            f"$r_\\rho^{{\\rm lin}}={r_lin[i]:+.3f}$,  "
            f"$r_\\rho^{{\\rm log}}={r_log[i]:+.3f}$"
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(r"$\rho/\langle\rho\rangle$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    for ax in axes[-2:]:
        ax.set_xlabel(r"$\ell$ along sightline [cMpc/h]")

    fig.suptitle(
        "[D-33] 1D density-along-sightline proxy: predicted vs GT $\\rho/\\langle\\rho\\rangle$\n"
        f"P{args.physics_id}, $z={args.redshift}$, T3 fiducial, step 10000/12500, "
        f"run_id {args.run_id[:8]}...  |  "
        f"all-rays median $r_\\rho^{{\\rm log}}$={s_log['median']:+.3f} "
        f"(IQR=[{s_log['q25']:+.3f}, {s_log['q75']:+.3f}], N={s_log['n']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[proxy-xi-1d] figure -> {out_png}")

    print()
    print("=== [D-33] 1D density-along-sightline proxy summary ===")
    print(f"  run_id            : {args.run_id}")
    print(f"  physics_id        : P{args.physics_id}")
    print(f"  redshift          : {args.redshift}")
    print(f"  n_rays_evaluated  : {n_rays}")
    print(f"  median r_rho_lin  : {s_lin['median']:+.4f}  IQR=[{s_lin['q25']:+.4f}, {s_lin['q75']:+.4f}]")
    print(f"  median r_rho_log  : {s_log['median']:+.4f}  IQR=[{s_log['q25']:+.4f}, {s_log['q75']:+.4f}]")
    print(f"  16/84 r_rho_log   : [{s_log['q16']:+.4f}, {s_log['q84']:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

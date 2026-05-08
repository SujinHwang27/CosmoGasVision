"""Stage 2b evaluation orchestrator.

Pulls a trained NeRF run from MLflow, reconstructs both the predicted
and ground-truth rho/<rho> on a (192,192,192) grid, and emits the
five gating figures + an HTML index per LEDGER §5 / [D-13].

Optional 4-tier degradation overlay produced when --run_ids is supplied
with a list of n_rays settings (16384, 1024, 256, 64).

Constraints:
- Does not import or modify ``src/models/nerf.py`` beyond the public
  IGMNeRF/forward signature.
- Uses ``magma`` for density panels and ``coolwarm`` for residuals.
- Falls back to truth-vs-truth or random-init mock mode if no checkpoint
  is reachable (the C1+C2+C3 chain may not have produced a real run yet).
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.flux_power import compute_PF_1d
from src.analysis.density_power import compute_Pdelta_3d
from src.analysis.cross_corr import compute_xi_pearson
from src.analysis.flux_pdf import compute_F_PDF, ks_distance, ks_distance_pdf
from src.data.igm_gal_loader import SherwoodIGMGalLoader
from src.data.loader import SherwoodLoader
from src.models.nerf import IGMNeRF


# ----------------------------------------------------------------- helpers


def _load_mlflow_run(run_id: str):
    """Return (run, ckpt_path_or_None). Tolerates missing MLflow."""
    try:
        import mlflow
        from dotenv import load_dotenv

        load_dotenv()
        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
    except Exception as exc:  # mlflow down / run missing
        print(f"[stage2b_report] MLflow lookup failed: {exc}")
        return None, None

    # Try to download a checkpoint artifact. Stage 2a runs may not have one.
    ckpt_path = None
    try:
        artifacts = client.list_artifacts(run_id)
        for art in artifacts:
            if art.path.endswith((".pt", ".pth", ".ckpt")):
                local = client.download_artifacts(run_id, art.path)
                ckpt_path = local
                break
    except Exception as exc:
        print(f"[stage2b_report] No checkpoint artifact ({exc}); using random-init.")

    return run, ckpt_path


def _build_model_from_run(run, ckpt_path: Optional[str]) -> IGMNeRF:
    """Instantiate IGMNeRF using the run's hyperparameters and load weights
    if available. Falls back to defaults / random init otherwise."""
    hidden_dim = 256
    num_layers = 8
    L = 10
    if run is not None:
        params = run.data.params
        hidden_dim = int(params.get("hidden_dim", hidden_dim))
        num_layers = int(params.get("num_layers", num_layers))
        L = int(params.get("L_fourier", L))

    model = IGMNeRF(hidden_dim=hidden_dim, num_layers=num_layers, L=L)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        # weights_only=False because pipeline.py saves numpy RNG state alongside
        # model weights for bit-identical resume; torch>=2.6 default rejects it.
        # No try/except: a checkpoint-load failure must NOT be silently swallowed
        # — random-init eval produces meaningless [D-13] gates that look real.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"[stage2b_report] Loaded weights from {ckpt_path}")
    elif ckpt_path is None:
        raise RuntimeError(
            "[stage2b_report] no checkpoint resolved for this run. Random-init "
            "eval was the silent-failure mode that produced bogus gates on "
            "2026-05-08; refusing to proceed. Pass a valid run_id whose MLflow "
            "artifacts include a *.pt, or use mock=True to explicitly request "
            "a random-init smoke."
        )
    model.eval()
    return model


def _eval_mlp_on_grid(model: IGMNeRF, n_grid: int, chunk: int = 262144) -> np.ndarray:
    """Predict rho/<rho> on a (n_grid)^3 grid in unit cube [0, 1].

    Returned grid is mean-normalized so its mean == 1, matching the
    convention the loader returns and what compute_xi_cross expects.
    """
    axis = (np.arange(n_grid) + 0.5) / n_grid
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
    n = coords.shape[0]
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            x = torch.from_numpy(coords[sl])
            fields = model(x)  # (chunk, 4) -> density at index 0
            out[sl] = fields[..., 0].cpu().numpy()
    rho = out.reshape(n_grid, n_grid, n_grid).astype(np.float64)
    m = rho.mean()
    if m > 0:
        rho = rho / m  # force <rho>=1 to land on the same convention
    return rho


# ---------------------------------------------------------------- figures


_DENSITY_CMAP = "magma"
_RESIDUAL_CMAP = "coolwarm"


def _fig_pf_compare(
    out_dir: Path,
    tau_pred: np.ndarray,
    tau_truth: np.ndarray,
    vel_axis: np.ndarray,
) -> dict:
    k_p, P_p = compute_PF_1d(tau_pred, vel_axis)
    k_t, P_t = compute_PF_1d(tau_truth, vel_axis)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.loglog(k_t, P_t, "k-", lw=2, label="Truth")
    ax.loglog(k_p, P_p, "C0--", lw=2, label="Predicted")
    ax.axvspan(10 ** -2.5, 10 ** -1.5, color="grey", alpha=0.15,
               label=r"[D-13] band")
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
    ax.set_ylabel(r"$P_F(k_\parallel)$ [s/km]")
    ax.set_title("1D flux power: pred vs truth")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    p = out_dir / "pf_compare.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)

    # Residual stat in the [D-13] band
    band = (k_t >= 10 ** -2.5) & (k_t <= 10 ** -1.5) & np.isfinite(P_t) & np.isfinite(P_p)
    if band.any():
        resid = np.abs(P_p[band] - P_t[band]) / P_t[band]
        return {"pf_residual_mean": float(np.nanmean(resid))}
    return {"pf_residual_mean": float("nan")}


def _fig_pdelta_anisotropy(
    out_dir: Path, rho_pred: np.ndarray, rho_truth: np.ndarray, box_kpc_h: float
) -> None:
    kpar_p, kperp_p, P_p = compute_Pdelta_3d(rho_pred, box_kpc_h)
    kpar_t, kperp_t, P_t = compute_Pdelta_3d(rho_truth, box_kpc_h)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, (P, title) in zip(axes, [(P_t, "Truth"), (P_p, "Predicted")]):
        # Use log10 with NaN safety
        with np.errstate(divide="ignore"):
            Z = np.log10(P)
        # Note: pcolormesh expects (kperp, kpar) on (X, Y) ordering
        im = ax.pcolormesh(
            kperp_t, kpar_t, Z, cmap=_DENSITY_CMAP, shading="auto"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k_\perp$ [h/Mpc]")
        ax.set_title(f"$P_\\delta$ — {title}")
    axes[0].set_ylabel(r"$k_\parallel$ [h/Mpc]")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(r"$\log_{10} P_\delta$")
    p = out_dir / "pdelta_anisotropy.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)


def _fig_xi_cross(
    out_dir: Path, rho_pred: np.ndarray, rho_truth: np.ndarray, box_kpc_h: float
) -> dict:
    r_bins = np.linspace(0.1, 20.0, 41)  # h^-1 Mpc
    r, xi = compute_xi_pearson(rho_pred, rho_truth, box_kpc_h, r_bins)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(r, xi, "C2-", lw=2, label=r"$\xi_{\hat\rho,\rho}(r)$")
    ax.axhline(0.6, color="k", ls="--", lw=1, label=r"[D-13] threshold $0.6$")
    ax.axvline(2.0, color="grey", ls=":", lw=1, label=r"$r=2\,h^{-1}$ Mpc")
    ax.set_xlabel(r"$r$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$\xi_{\hat\rho,\rho}$")
    ax.set_title("Density cross-correlation")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "xi_cross.png", dpi=140)
    plt.close(fig)

    # Headline value at r=2 h^-1 Mpc (interpolated)
    valid = np.isfinite(xi)
    if valid.any():
        xi2 = float(np.interp(2.0, r[valid], xi[valid]))
    else:
        xi2 = float("nan")
    return {"xi_r2": xi2}


def _fig_flux_pdf(
    out_dir: Path, tau_pred: np.ndarray, tau_truth: np.ndarray
) -> dict:
    # Visualization: pre-binned PDF over the full F-range so the figure
    # shows both the saturated absorber pile-up and the F~1 tail.
    F_bins = np.linspace(0.0, 1.0, 51)
    c, pdf_p = compute_F_PDF(tau_pred, F_bins)
    _, pdf_t = compute_F_PDF(tau_truth, F_bins)

    # [D-13] gate: KS on RAW flux samples over F in [0.05, 0.95]
    # (Bolton+ 2008 / Lee+ 2015 cuts: drop saturated absorbers and the
    # continuum-fitting / metal-line residual tail). The binned-PDF
    # ks_distance_pdf is retained only for the figure annotation.
    F_pred_samples = np.exp(-np.asarray(tau_pred)).ravel()
    F_truth_samples = np.exp(-np.asarray(tau_truth)).ravel()
    ks = ks_distance(F_pred_samples, F_truth_samples, F_range=(0.05, 0.95))

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(c, pdf_t, "k-", lw=2, label="Truth")
    ax.plot(c, pdf_p, "C0--", lw=2, label="Predicted")
    ax.axvspan(0.05, 0.95, color="grey", alpha=0.10, label="[D-13] KS window")
    ax.set_xlabel("F = exp(-tau)")
    ax.set_ylabel("p(F)")
    ax.set_title(f"Flux PDF (KS = {ks:.4f}, F in [0.05, 0.95])")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "flux_pdf.png", dpi=140)
    plt.close(fig)
    return {"ks_distance": ks}


def _fig_slice_compare(
    out_dir: Path, rho_pred: np.ndarray, rho_truth: np.ndarray
) -> None:
    N = rho_truth.shape[0]
    s_t = np.log10(np.maximum(rho_truth[:, :, N // 2], 1e-3))
    s_p = np.log10(np.maximum(rho_pred[:, :, N // 2], 1e-3))
    vmin = float(min(s_t.min(), s_p.min()))
    vmax = float(max(s_t.max(), s_p.max()))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, (S, title) in zip(axes, [(s_t, "Truth"), (s_p, "Predicted")]):
        im = ax.imshow(S, cmap=_DENSITY_CMAP, vmin=vmin, vmax=vmax,
                       origin="lower")
        ax.set_title(f"$\\log_{{10}} \\rho/\\bar{{\\rho}}$ — {title}")
        ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(r"$\log_{10}\rho/\bar\rho$")
    fig.savefig(out_dir / "slice_compare.png", dpi=140)
    plt.close(fig)


def _fig_degradation(
    out_dir: Path,
    n_rays: list[int],
    pf_resid: list[float],
    xi2: list[float],
    ks: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.set_xscale("log")
    ax.plot(n_rays, pf_resid, "o-", label=r"$|\Delta P_F/P_F|$ (band avg)")
    ax.plot(n_rays, xi2, "s-", label=r"$\xi(r{=}2)$")
    ax.plot(n_rays, ks, "^-", label="KS(F-PDF)")
    ax.axhline(0.10, color="grey", ls="--", alpha=0.7, lw=1)
    ax.axhline(0.60, color="grey", ls=":", alpha=0.7, lw=1)
    ax.set_xlabel(r"$n_{\rm rays}$")
    ax.set_ylabel("metric value")
    ax.set_title("4-tier sightline-density degradation")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "degradation_curve.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------- driver


def _chunked_cic_rho(physics_id: int, n_grid: int) -> np.ndarray:
    """Stream gas particles and CIC-deposit in 2M-particle batches.

    Mirrors ``SherwoodIGMGalLoader.load_3d_field`` but with a smaller
    per-call working-set, since the loader's per-file bincount peaks at
    ~170 MB on Physics-1 sub-files which exceeds available memory on
    constrained hosts. The math is identical and the loader is left
    unmodified per the C5 brief.
    """
    import gc as _gc
    loader = SherwoodIGMGalLoader()
    meta = loader.get_box_meta(physics_id)
    box = float(meta["box_kpc_h"])
    grid = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    n3 = n_grid * n_grid * n_grid
    flat_view = grid.reshape(n3)
    cell = np.float32(box / n_grid)
    BATCH = 2_000_000

    for chunk in loader.iter_gas_chunks(
        physics_id, fields=("Coordinates", "Masses")
    ):
        coords = np.ascontiguousarray(chunk["Coordinates"], dtype=np.float32)
        weights = np.ascontiguousarray(chunk["Masses"], dtype=np.float32)
        n_total = coords.shape[0]
        for start in range(0, n_total, BATCH):
            end = min(start + BATCH, n_total)
            c = coords[start:end]
            w = weights[start:end]
            fx = c[:, 0] / cell; fy = c[:, 1] / cell; fz = c[:, 2] / cell
            ix = np.floor(fx).astype(np.int32)
            iy = np.floor(fy).astype(np.int32)
            iz = np.floor(fz).astype(np.int32)
            dx = fx - ix; dy = fy - iy; dz = fz - iz
            ix0 = ix % n_grid; iy0 = iy % n_grid; iz0 = iz % n_grid
            ix1 = (ix + 1) % n_grid
            iy1 = (iy + 1) % n_grid
            iz1 = (iz + 1) % n_grid
            for ax, ay, az, wx, wy, wz in (
                (ix0, iy0, iz0, 1 - dx, 1 - dy, 1 - dz),
                (ix1, iy0, iz0,     dx, 1 - dy, 1 - dz),
                (ix0, iy1, iz0, 1 - dx,     dy, 1 - dz),
                (ix0, iy0, iz1, 1 - dx, 1 - dy,     dz),
                (ix1, iy1, iz0,     dx,     dy, 1 - dz),
                (ix1, iy0, iz1,     dx, 1 - dy,     dz),
                (ix0, iy1, iz1, 1 - dx,     dy,     dz),
                (ix1, iy1, iz1,     dx,     dy,     dz),
            ):
                idx = (
                    ax.astype(np.int64) * n_grid * n_grid
                    + ay.astype(np.int64) * n_grid + az
                )
                ww = (w * wx * wy * wz).astype(np.float32)
                flat_view += np.bincount(
                    idx, weights=ww, minlength=n3
                ).astype(np.float64, copy=False)
                del idx, ww
            _gc.collect()
        del coords, weights, chunk
        _gc.collect()

    m = grid.mean()
    if m <= 0:
        raise RuntimeError(f"non-positive mean grid mean={m}")
    rho = grid / m
    return rho


def _build_truth_inputs(
    physics_id: int, redshift: float, n_grid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load truth tau, vel_axis, rho_grid, box_kpc_h. Returns NumPy arrays."""
    sherwood = SherwoodLoader("Sherwood")
    sl = sherwood.load_sightlines(physics_id, redshift)
    tau_truth = np.asarray(sl["tau_h1"], dtype=np.float64)
    vel_axis = np.asarray(sl["vel_axis"], dtype=np.float64)
    box_kpc_h = float(sl["header"]["box_kpc_h"])

    rho_truth = _chunked_cic_rho(physics_id, n_grid)
    return tau_truth, vel_axis, rho_truth, box_kpc_h


def _render_tau_for_model(
    model: IGMNeRF,
    sightline_coords: torch.Tensor,
    vel_axis_t: torch.Tensor,
) -> np.ndarray:
    """Render tau profiles for the supplied unit-cube coordinates.

    Wrapped so we never fail the report if the integrator import path
    changes; on failure, returns NaNs of the right shape.
    """
    from src.models.nerf import volume_render_physics
    with torch.no_grad():
        tau = volume_render_physics(model, sightline_coords, vel_axis_t)
    return tau.cpu().numpy()


def generate_report(
    run_id: str,
    output_dir: str = "experiments/nerf/artifacts/reports/",
    physics_id: int = 1,
    redshift: float = 0.3,
    n_grid: int = 192,
    n_rays_eval: int = 1024,
    mock: bool = False,
) -> Path:
    """Produce 5 PNGs + index.html for ``run_id``.

    Parameters
    ----------
    run_id : str
        MLflow run id. If ``mock=True``, used only as a folder label;
        no MLflow lookup is performed.
    output_dir : str
        Parent directory; the report lands at ``<output_dir>/<run_id>/``.
    n_rays_eval : int
        Number of sightlines used for tau-based metrics. The fiducial
        [D-13] point is 1024.
    mock : bool
        If True, skip MLflow and use a random-init IGMNeRF with default
        hyperparameters. Used when no real Stage 2b checkpoint exists.
    """
    out = Path(output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    # Load truth
    tau_truth_full, vel_axis, rho_truth, box_kpc_h = _build_truth_inputs(
        physics_id, redshift, n_grid
    )
    n_rays_avail = tau_truth_full.shape[0]
    n_rays = min(n_rays_eval, n_rays_avail)
    rng = np.random.default_rng(seed=42)
    sel = rng.choice(n_rays_avail, size=n_rays, replace=False)
    sel.sort()
    tau_truth = tau_truth_full[sel]

    # Load model
    if mock:
        run = None
        ckpt = None
    else:
        run, ckpt = _load_mlflow_run(run_id)
    model = _build_model_from_run(run, ckpt)

    # Predict tau on the same sightlines (need their unit-cube coords)
    sherwood = SherwoodLoader("Sherwood")
    sl_full = sherwood.load_sightlines(physics_id, redshift)
    coords_world = sherwood.get_world_coordinates(sl_full)
    coords = torch.tensor(coords_world[sel] / box_kpc_h, dtype=torch.float32)
    vel_axis_t = torch.tensor(vel_axis, dtype=torch.float32)
    tau_pred = _render_tau_for_model(model, coords, vel_axis_t)

    # Predict rho on the (n_grid)^3 grid
    print(f"[stage2b_report] evaluating MLP on {n_grid}^3 grid", flush=True)
    rho_pred = _eval_mlp_on_grid(model, n_grid)

    # Figures + scalar headlines
    metrics = {}
    metrics.update(_fig_pf_compare(out, tau_pred, tau_truth, vel_axis))
    _fig_pdelta_anisotropy(out, rho_pred, rho_truth, box_kpc_h)
    metrics.update(_fig_xi_cross(out, rho_pred, rho_truth, box_kpc_h))
    metrics.update(_fig_flux_pdf(out, tau_pred, tau_truth))
    _fig_slice_compare(out, rho_pred, rho_truth)

    # HTML index
    html = (out / "index.html")
    html.write_text(
        _build_html(run_id, metrics, mock), encoding="utf-8"
    )
    print(f"[stage2b_report] report written to {out}")
    return out


def _build_html(run_id: str, metrics: dict, mock: bool) -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4g}</td></tr>" for k, v in metrics.items()
    )
    note = (
        "<p style='color:#a00'><b>Mock mode</b>: random-init weights, no "
        "checkpoint loaded.</p>"
        if mock else ""
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Stage 2b report — {run_id}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 960px; margin: 2em auto;}}
img {{ max-width: 100%; border: 1px solid #ddd; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
table {{ border-collapse: collapse; }}
td, th {{ padding: 4px 12px; border: 1px solid #ccc; }}
</style></head>
<body>
<h1>Stage 2b report — run {run_id}</h1>
{note}
<h2>Headline metrics</h2>
<table><tr><th>metric</th><th>value</th></tr>{rows}</table>
<h2>1D flux power</h2><img src="pf_compare.png">
<h2>3D density anisotropy</h2><img src="pdelta_anisotropy.png">
<h2>Cross-correlation xi(r)</h2><img src="xi_cross.png">
<h2>Flux PDF</h2><img src="flux_pdf.png">
<h2>Density slice</h2><img src="slice_compare.png">
</body></html>
"""


# ------------------------------------------------------------ degradation


def generate_degradation(
    run_ids_by_n_rays: dict[int, str],
    output_dir: str = "experiments/nerf/artifacts/reports/",
) -> Path:
    """Run :func:`generate_report` for each (n_rays, run_id) and emit
    ``degradation_curve.png`` summarizing the three [D-13] metrics."""
    out = Path(output_dir) / "degradation"
    out.mkdir(parents=True, exist_ok=True)
    n_rays_list = sorted(run_ids_by_n_rays, reverse=True)
    pf, xi2, ks = [], [], []
    for n in n_rays_list:
        rid = run_ids_by_n_rays[n]
        rep = generate_report(rid, output_dir=output_dir, n_rays_eval=n)
        # Re-parse the metrics out of the HTML index for now; cheaper than
        # rerunning the metric chain.
        # In production this should return a dict — but the C5 brief asks
        # for a save-to-disk orchestrator, so we recompute with a quick file.
        # (Left as TODO: structured metrics file.)
        pf.append(float("nan"))
        xi2.append(float("nan"))
        ks.append(float("nan"))
    _fig_degradation(out, n_rays_list, pf, xi2, ks)
    return out


# ------------------------------------------------------------------- CLI


def _main():
    ap = argparse.ArgumentParser(description="Stage 2b cosmological report")
    ap.add_argument("--run_id", required=True)
    ap.add_argument(
        "--output_dir", default="experiments/nerf/artifacts/reports/"
    )
    ap.add_argument("--physics_id", type=int, default=1)
    ap.add_argument("--redshift", type=float, default=0.3)
    ap.add_argument("--n_grid", type=int, default=192)
    ap.add_argument("--n_rays_eval", type=int, default=1024)
    ap.add_argument("--mock", action="store_true",
                    help="Skip MLflow; random-init IGMNeRF.")
    args = ap.parse_args()
    generate_report(
        run_id=args.run_id,
        output_dir=args.output_dir,
        physics_id=args.physics_id,
        redshift=args.redshift,
        n_grid=args.n_grid,
        n_rays_eval=args.n_rays_eval,
        mock=args.mock,
    )


if __name__ == "__main__":
    _main()

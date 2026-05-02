"""Sherwood IGM-gal 2D slice render (smoke gallery, C4 success-check).

Loads the Physics-1 (no-feedback) overdensity field at z=0.3 onto a
192^3 mesh by streaming gas particles via SherwoodIGMGalLoader's
``iter_gas_chunks`` and CIC-depositing in particle-count sub-chunks
to keep peak memory low (the in-loader implementation processes whole
HDF5 sub-files at once, which forces a ~170 MB float64 bincount weight
allocation that exceeds available RAM on the current host). The final
output matches what ``load_3d_field(1, 'rho', n_grid=192)`` would have
returned: a (192, 192, 192) ``rho/<rho>`` grid.

We then take the central z-slice (index 96) and save log10(rho/<rho>)
with the matplotlib 'viridis' colormap. The output is the visual
sanity proof that the CIC deposition produces a recognizable cosmic
web with filament/void contrast spanning roughly 10^-2 to 10^2.

Visual comparison note (per C5 brief): Bolton et al. (2017, MNRAS,
"The Sherwood simulation suite", Fig. 1) shows the planck1 60-Mpc/h
volume at z=2 with characteristic filaments and voids. Our z=0.3
output will look smoother than the published z=2 panel because
non-linear structure has continued to evacuate voids and pull material
into filaments by z=0.3 (so peaks are sharper, void floors a bit
deeper) but the overall visual web pattern should match -- this is
the expected behavior at the lower redshift, not a bug.
"""

from __future__ import annotations

import gc
import os
import sys

# Ensure repo-root imports work when invoked as `python scripts/...`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.igm_gal_loader import SherwoodIGMGalLoader  # noqa: E402


OUT_DIR = os.path.join(
    _REPO_ROOT, "experiments", "nerf", "artifacts", "visualizations"
)
OUT_PATH = os.path.join(OUT_DIR, "igm_gal_smoke_slice_z0.3_P1.png")


def _cic_chunk(
    grid: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray,
    box: float,
    n_grid: int,
) -> None:
    """Memory-friendly cloud-in-cell deposit. Processes particles in
    batches and uses float32 throughout. Identical math to the inline
    deposition in ``SherwoodIGMGalLoader``; we re-implement here only
    to keep the per-call temporary allocation small enough to fit in
    the constrained host budget (loader is on the no-edit list)."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3); got {coords.shape}")
    if weights.shape[0] != coords.shape[0]:
        raise ValueError("coords/weights length mismatch")
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)

    cell = np.float32(box / n_grid)
    n_total = coords.shape[0]
    BATCH = 2_000_000
    n3 = n_grid * n_grid * n_grid
    flat_view = grid.reshape(n3)

    for start in range(0, n_total, BATCH):
        end = min(start + BATCH, n_total)
        c = coords[start:end]
        w = weights[start:end]
        fx = c[:, 0] / cell
        fy = c[:, 1] / cell
        fz = c[:, 2] / cell
        ix = np.floor(fx).astype(np.int32)
        iy = np.floor(fy).astype(np.int32)
        iz = np.floor(fz).astype(np.int32)
        dx = fx - ix
        dy = fy - iy
        dz = fz - iz
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
                + ay.astype(np.int64) * n_grid
                + az
            )
            ww = (w * wx * wy * wz).astype(np.float32)
            flat_view += np.bincount(
                idx, weights=ww, minlength=n3
            ).astype(np.float64, copy=False)
            del idx, ww
            gc.collect()


def _build_rho_grid(n_grid: int = 192, physics_id: int = 1) -> np.ndarray:
    loader = SherwoodIGMGalLoader()
    meta = loader.get_box_meta(physics_id)
    box = meta["box_kpc_h"]
    print(f"box = {box} kpc/h, expect {meta['ngas_total']} gas particles total")
    grid = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    n_total = 0
    for k, chunk in enumerate(
        loader.iter_gas_chunks(physics_id, fields=("Coordinates", "Masses"))
    ):
        coords = chunk["Coordinates"]
        masses = chunk["Masses"]
        n_total += coords.shape[0]
        _cic_chunk(grid, coords, masses, box=box, n_grid=n_grid)
        del coords, masses, chunk
        gc.collect()
        print(f"  file {k}: cumulative {n_total} particles deposited")
    if n_total == 0:
        raise RuntimeError("no gas particles found")
    mean = grid.mean()
    if mean <= 0:
        raise RuntimeError(f"non-positive mean grid {mean}")
    rho = grid / mean
    if not np.isfinite(rho).all():
        raise RuntimeError("non-finite cells in rho")
    return rho


def main() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    rho = _build_rho_grid(n_grid=192, physics_id=1)
    print(
        f"rho/<rho> grid: shape={rho.shape}, "
        f"min={rho.min():.3e}, mean={rho.mean():.4f}, max={rho.max():.3e}"
    )

    z_idx = 96
    slab = rho[:, :, z_idx]
    log_slab = np.log10(np.maximum(slab, 1e-3))

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    im = ax.imshow(log_slab, cmap="viridis", origin="lower",
                   vmin=-2.0, vmax=2.0)
    ax.set_title(
        r"Sherwood IGM-gal P1, $z=0.3$ — $\log_{10}\rho/\bar\rho$"
        f" (slice z={z_idx})"
    )
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$\log_{10}\rho/\bar\rho$")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    plt.close(fig)
    sz = os.path.getsize(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({sz} bytes)")
    return OUT_PATH


if __name__ == "__main__":
    main()

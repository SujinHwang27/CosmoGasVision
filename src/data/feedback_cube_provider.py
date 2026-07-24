"""Truth-cube provider for the exp/feedback-latent Stage-1 conditioned field.

Serves coordinate -> x samples from the rho/<rho> truth cubes of the four
Sherwood physics variants (P1..P4) at 192^3, z=0.3, with the [D-49] spatial
split, for the SIREN auto-decoder trunk ([F-02] §4 / §5 R28 rung R2, artifact A3).

Provenance / reuse (do NOT re-derive these):
  * [D-49] split: reused from ``src.data.loader.HeldoutSplitScheme`` /
    ``DEFAULT_SCHEME`` (train_x_max=0.7, val_x_max=0.85, axis=0). At n_grid=192
    the voxel boundaries are int(0.7*192)=134 and int(0.85*192)=163, i.e.
    train [0,134) / val [134,163) / test [163,192) along axis 0. The
    ``int(frac*n_grid)`` floor rule is mirrored exactly from
    ``loader.extract_rho_crops`` (loader.py:1143-1144).
  * [D-75] scoring transform: imported verbatim from
    ``scripts.d75_corrected_metric_rescore.x_transform`` —
    x = log10(max(rho/<rho>, 1e-3)), float64. NOT re-implemented here.

Coordinates: comoving box [0, 60000] kpc/h (60 Mpc/h). Cell centers are mapped
to SIREN's [-1, 1] convention. The box scale cancels in the normalization but is
carried explicitly (``BOX_KPC_H``) so the kpc/h round-trip is testable and the
lineage is documented. Returned coords are torch tensors that can carry
``requires_grad`` downstream (autograd-friendly); no detached-NumPy in any path
the model would differentiate through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.loader import DEFAULT_SCHEME, HeldoutSplitScheme
from scripts.d75_corrected_metric_rescore import x_transform as _d75_x_transform

# --------------------------------------------------------------------------- #
# Constants of record
# --------------------------------------------------------------------------- #
N_GRID = 192
BOX_KPC_H = 60000.0            # comoving box, 60 Mpc/h (Astrophysical conventions)
REDSHIFT = 0.3
Region = str                  # "train" | "val" | "test"
REGIONS: Tuple[str, ...] = ("train", "val", "test")

# Repo root = three parents up from this file (src/data/feedback_cube_provider.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Cube paths per the f01c enumeration (P1 nerf/d75; P2-P4 unet stage1).
CUBE_PATHS: Dict[int, Path] = {
    1: _REPO_ROOT / "experiments/nerf/artifacts/d75_rescore/cubes/truth_real_192.npy",
    2: _REPO_ROOT / "experiments/unet-inversion/artifacts/stage1/cubes/truth_real_192_p2.npy",
    3: _REPO_ROOT / "experiments/unet-inversion/artifacts/stage1/cubes/truth_real_192_p3.npy",
    4: _REPO_ROOT / "experiments/unet-inversion/artifacts/stage1/cubes/truth_real_192_p4.npy",
}

# x-range sanity window. floor is log10(1e-3) = -3; P1 max overdensity ~4115 ->
# log10 ~= 3.61. Allow a small margin.
_X_MIN_OK = -3.0
_X_MAX_OK = 3.7


# --------------------------------------------------------------------------- #
# [D-49] split geometry (mirrors loader.extract_rho_crops voxel-bound rule)
# --------------------------------------------------------------------------- #
def region_voxel_bounds(
    region: Region,
    n_grid: int = N_GRID,
    scheme: HeldoutSplitScheme = DEFAULT_SCHEME,
) -> Tuple[int, int]:
    """Right-open [start, end) voxel index interval on the split axis.

    Uses ``int(frac * n_grid)`` exactly as loader.py:1143-1144, so at n_grid=192
    train=[0,134), val=[134,163), test=[163,192).
    """
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}; got {region!r}")
    train_end = int(scheme.train_x_max * n_grid)
    val_end = int(scheme.val_x_max * n_grid)
    start = {"train": 0, "val": train_end, "test": val_end}[region]
    end = {"train": train_end, "val": val_end, "test": n_grid}[region]
    return start, end


# --------------------------------------------------------------------------- #
# coordinate normalization  (voxel index <-> kpc/h <-> SIREN [-1,1])
# --------------------------------------------------------------------------- #
def _voxel_center_kpc_h(idx: np.ndarray) -> np.ndarray:
    """Cell-center comoving coordinate in kpc/h for integer voxel indices."""
    return (idx.astype(np.float64) + 0.5) / N_GRID * BOX_KPC_H


def _kpc_h_to_siren(coord_kpc_h: np.ndarray) -> np.ndarray:
    """Map box [0, BOX_KPC_H] kpc/h onto SIREN [-1, 1]."""
    u = coord_kpc_h / BOX_KPC_H          # -> [0, 1]
    return 2.0 * u - 1.0                 # -> [-1, 1]


def _siren_to_kpc_h(coord_siren: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_kpc_h_to_siren` (round-trip helper)."""
    u = (np.asarray(coord_siren, dtype=np.float64) + 1.0) / 2.0
    return u * BOX_KPC_H


def voxel_index_to_siren(idx_xyz: np.ndarray) -> np.ndarray:
    """(N,3) integer voxel indices -> (N,3) SIREN [-1,1] cell-center coords."""
    return _kpc_h_to_siren(_voxel_center_kpc_h(np.asarray(idx_xyz)))


# --------------------------------------------------------------------------- #
# x-transform (imported verbatim from the d75 script; thin wrapper)
# --------------------------------------------------------------------------- #
def apply_x_transform(rho_overdensity: np.ndarray) -> Tuple[np.ndarray, float]:
    """[D-75] scoring variable x = log10(max(rho/<rho>, 1e-3)).

    The truth cubes are already stored as rho/<rho> (overdensity), so they are
    passed straight through. Returns (x, clamped_fraction) exactly as the d75
    reference does.
    """
    return _d75_x_transform(rho_overdensity)


# --------------------------------------------------------------------------- #
# provider
# --------------------------------------------------------------------------- #
@dataclass
class _VariantCube:
    p: int
    rho: np.ndarray          # (192,192,192) float64 overdensity rho/<rho>
    x: np.ndarray            # (192,192,192) float64 x = log10(max(rho,1e-3))
    clamped_fraction: float
    path: Path


class FeedbackCubeProvider:
    """Serves coord->x samples from the 4 rho truth cubes for the trunk.

    Parameters
    ----------
    variants : sequence of int, optional
        Which physics variants to attempt to load (default 1..4). A cube that
        is not on disk is skipped (graceful degradation); ``loaded_variants``
        reports what actually loaded and ``missing_variants`` what did not.
    dtype : torch.dtype
        dtype of returned coord/target tensors (default float32).
    strict : bool
        If True, raise on any missing cube instead of degrading.
    """

    def __init__(
        self,
        variants: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
        strict: bool = False,
    ) -> None:
        self.dtype = dtype
        self.scheme = DEFAULT_SCHEME
        want = list(variants) if variants is not None else [1, 2, 3, 4]
        self._cubes: Dict[int, _VariantCube] = {}
        self.missing_variants: List[int] = []
        for p in want:
            path = CUBE_PATHS[p]
            if not path.exists():
                if strict:
                    raise FileNotFoundError(f"P{p} cube missing: {path}")
                self.missing_variants.append(p)
                continue
            rho = np.load(path).astype(np.float64)
            x, clamped = apply_x_transform(rho)
            cube = _VariantCube(p=p, rho=rho, x=x, clamped_fraction=clamped, path=path)
            self._validate_data(cube)
            self._cubes[p] = cube

    # -- properties -------------------------------------------------------- #
    @property
    def loaded_variants(self) -> List[int]:
        return sorted(self._cubes.keys())

    # -- validation contract ---------------------------------------------- #
    def _validate_data(self, cube: _VariantCube) -> None:
        """Mandatory sanity checks (project convention: every provider)."""
        if cube.rho.shape != (N_GRID, N_GRID, N_GRID):
            raise ValueError(
                f"P{cube.p} rho shape {cube.rho.shape} != {(N_GRID,)*3}"
            )
        if cube.x.shape != (N_GRID, N_GRID, N_GRID):
            raise ValueError(f"P{cube.p} x shape {cube.x.shape} != {(N_GRID,)*3}")
        if not np.all(np.isfinite(cube.rho)):
            raise ValueError(f"P{cube.p} rho has non-finite values")
        if not np.all(np.isfinite(cube.x)):
            raise ValueError(f"P{cube.p} x has non-finite values")
        if np.any(cube.rho < 0.0):
            raise ValueError(f"P{cube.p} rho has negative (unphysical) values")
        xmin, xmax = float(cube.x.min()), float(cube.x.max())
        if xmin < _X_MIN_OK - 1e-9 or xmax > _X_MAX_OK:
            raise ValueError(
                f"P{cube.p} x range [{xmin:.4f}, {xmax:.4f}] outside sane "
                f"[{_X_MIN_OK}, {_X_MAX_OK}]"
            )

    def _require(self, p: int) -> _VariantCube:
        if p not in self._cubes:
            raise KeyError(
                f"P{p} not loaded (loaded={self.loaded_variants}, "
                f"missing={self.missing_variants})"
            )
        return self._cubes[p]

    # -- whole-region-cube accessor (for scoring) ------------------------- #
    def region_cube(
        self, p: int, region: Region
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_slab, coords_siren) for the whole region.

        x_slab : (n_axis0, 192, 192) float64 x-values for the region's axis-0
                 slab. coords_siren : (n_axis0, 192, 192, 3) float64 SIREN
                 cell-center coords, aligned voxel-for-voxel with x_slab.
        """
        cube = self._require(p)
        start, end = region_voxel_bounds(region)
        x_slab = cube.x[start:end]
        ii, jj, kk = np.meshgrid(
            np.arange(start, end), np.arange(N_GRID), np.arange(N_GRID),
            indexing="ij",
        )
        idx = np.stack([ii, jj, kk], axis=-1)          # (na,192,192,3)
        coords = voxel_index_to_siren(idx.reshape(-1, 3)).reshape(idx.shape)
        return x_slab, coords

    def region_flat(
        self, p: int, region: Region
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Whole region flattened: (coords (M,3) SIREN, x_targets (M,)) tensors."""
        x_slab, coords = self.region_cube(p, region)
        coords_t = torch.as_tensor(coords.reshape(-1, 3), dtype=self.dtype)
        x_t = torch.as_tensor(x_slab.reshape(-1), dtype=self.dtype)
        return coords_t, x_t

    # -- random point sampler --------------------------------------------- #
    def sample(
        self,
        p: int,
        region: Region,
        n: int,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n`` random voxel-center samples from ``region`` of variant ``p``.

        Returns
        -------
        coords : (n, 3) float tensor in SIREN [-1, 1] (cell centers).
        x_targets : (n,) float tensor, x = log10(max(rho/<rho>, 1e-3)).

        Deterministic given ``seed``. Sampling is with replacement over the
        region's voxels (Stage-1 smoke does not require without-replacement).
        Coords carry no grad by default; the caller may set requires_grad.
        """
        cube = self._require(p)
        start, end = region_voxel_bounds(region)
        rng = np.random.default_rng(seed)
        i0 = rng.integers(start, end, size=n)
        i1 = rng.integers(0, N_GRID, size=n)
        i2 = rng.integers(0, N_GRID, size=n)
        idx = np.stack([i0, i1, i2], axis=-1)          # (n,3)
        coords = voxel_index_to_siren(idx)
        x_vals = cube.x[i0, i1, i2]
        coords_t = torch.as_tensor(coords, dtype=self.dtype)
        x_t = torch.as_tensor(x_vals, dtype=self.dtype)
        return coords_t, x_t

    # -- lineage summary --------------------------------------------------- #
    def summary(self) -> dict:
        return {
            "n_grid": N_GRID,
            "box_kpc_h": BOX_KPC_H,
            "redshift": REDSHIFT,
            "split_scheme": {
                "train_x_max": self.scheme.train_x_max,
                "val_x_max": self.scheme.val_x_max,
                "axis": self.scheme.axis,
                "voxel_bounds": {r: region_voxel_bounds(r) for r in REGIONS},
                "source": "src.data.loader.DEFAULT_SCHEME ([D-49])",
            },
            "x_transform": "scripts.d75_corrected_metric_rescore.x_transform "
                           "([D-75]: log10(max(rho/<rho>, 1e-3)))",
            "loaded_variants": self.loaded_variants,
            "missing_variants": self.missing_variants,
            "clamped_fraction": {
                p: self._cubes[p].clamped_fraction for p in self.loaded_variants
            },
            "cube_paths": {p: str(CUBE_PATHS[p]) for p in CUBE_PATHS},
        }

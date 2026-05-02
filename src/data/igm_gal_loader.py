"""Loader for SherwoodIGM_gal 3D snapshot data at z = 0.3.

Each physics variant is distributed as a GADGET multi-part HDF5 snapshot
(`snapdir_012/snap_012.{0..N-1}.hdf5`). Particle data is split across files
by the simulation's domain decomposition; the canonical reading pattern
concatenates `PartType0` (gas) groups across all files.

This loader exposes the contract the PI pinned in LEDGER [D-15]:

    load_3d_field(physics_id, field) -> np.ndarray of shape (n_grid,)*3

The `'rho'` field is mass-weighted CIC-deposited from gas particles to the
target grid. Other fields (`'T'`, `'xHI'`, `'vlos'`) are stubbed with
NotImplementedError; they require SPH-kernel weighting against `Density`
and `SmoothingLength` to match the simulation's effective resolution and
will be filled when support-researcher's metric work needs them.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Literal

import h5py
import numpy as np


# Physics-id -> upstream snapshot name. The integer ids match the convention
# used by `src/data/loader.py` for the 1D sightline data.
PHYSICS_DIRS: Dict[int, str] = {
    1: "planck1_60_768_z0.300",                 # no feedback (fiducial)
    2: "planck1_60_768_ps13_z0.300",            # ps13 stellar wind
    3: "planck1_60_768_ps13agn_z0.300",         # ps13 + AGN
    4: "planck1_60_768_ps13agn_strong_z0.300",  # ps13 + strong AGN
}


class SherwoodIGMGalLoader:
    """Read GADGET multi-part HDF5 snapshots and project particles onto a grid.

    Parameters
    ----------
    data_root : str
        Path to the directory containing the per-physics extracted snapshot
        trees, e.g. ``SherwoodIGM_gal/extracted/``.
    """

    def __init__(self, data_root: str = "SherwoodIGM_gal/extracted") -> None:
        self.data_root = data_root

    # ------------------------------------------------------------------ meta

    def _snap_files(self, physics_id: int) -> list[str]:
        if physics_id not in PHYSICS_DIRS:
            raise ValueError(f"Invalid physics_id {physics_id}; expected 1..4")
        snap_dir = os.path.join(
            self.data_root, PHYSICS_DIRS[physics_id], "snapdir_012"
        )
        if not os.path.isdir(snap_dir):
            raise FileNotFoundError(
                f"Snapshot directory missing: {snap_dir}. "
                f"Run scripts/extract_sherwood_igm_gal.sh for physics {physics_id}."
            )
        files = sorted(glob.glob(os.path.join(snap_dir, "snap_012.*.hdf5")))
        if not files:
            raise FileNotFoundError(f"No snap_012.*.hdf5 files in {snap_dir}")
        return files

    def get_box_meta(self, physics_id: int) -> dict:
        """Header values from the first file. The GADGET split-file convention
        replicates the cosmology / box / particle-count header in every
        sub-file, so reading file 0 is sufficient.
        """
        f0 = self._snap_files(physics_id)[0]
        with h5py.File(f0, "r") as h:
            hdr = h["Header"].attrs
            box_kpc_h = float(hdr["BoxSize"])  # GADGET stores BoxSize in kpc/h
            redshift = float(hdr["Redshift"])
            num_files = int(hdr["NumFilesPerSnapshot"])
            ngas_total = int(hdr["NumPart_Total"][0])
            hubble = float(hdr["HubbleParam"])
            omega0 = float(hdr["Omega0"])
        return {
            "box_kpc_h": box_kpc_h,
            "redshift": redshift,
            "num_files": num_files,
            "ngas_total": ngas_total,
            "hubble": hubble,
            "omega0": omega0,
            "physics_id": physics_id,
            "physics_name": PHYSICS_DIRS[physics_id],
        }

    # -------------------------------------------------------------- particles

    def iter_gas_chunks(
        self,
        physics_id: int,
        fields: tuple[str, ...] = ("Coordinates", "Masses"),
    ):
        """Yield per-file ``PartType0`` arrays. Generator form so callers do
        not have to hold all 16 sub-files' worth of particles in RAM at once.

        Coordinates are in the simulation's native units (comoving kpc/h,
        matching [D-08]). Masses are in code units (1e10 Msun/h). Field-specific
        GADGET names: ``Coordinates``, ``Masses``, ``Density``,
        ``InternalEnergy``, ``NeutralHydrogenAbundance``, ``Velocities``,
        ``SmoothingLength``.
        """
        for path in self._snap_files(physics_id):
            with h5py.File(path, "r") as h:
                if "PartType0" not in h:
                    continue
                p0 = h["PartType0"]
                chunk = {f: p0[f][...] for f in fields if f in p0}
                if chunk:
                    yield chunk

    def load_gas_particles(
        self,
        physics_id: int,
        fields: tuple[str, ...] = ("Coordinates", "Masses"),
    ) -> Dict[str, np.ndarray]:
        """Concatenated form of :meth:`iter_gas_chunks`. Allocates the full
        particle table — only safe for small fields (e.g., ``Masses`` alone)
        or for small physics variants. For ``Coordinates``, prefer streaming.
        """
        out: Dict[str, list[np.ndarray]] = {f: [] for f in fields}
        for chunk in self.iter_gas_chunks(physics_id, fields):
            for f, arr in chunk.items():
                out[f].append(arr)
        return {f: np.concatenate(arr, axis=0) for f, arr in out.items() if arr}

    # -------------------------------------------------------------- 3D fields

    def load_3d_field(
        self,
        physics_id: int,
        field: Literal["rho", "T", "xHI", "vlos"] = "rho",
        n_grid: int = 768,
    ) -> np.ndarray:
        """Return a (n_grid, n_grid, n_grid) field on the simulation native mesh.

        ``'rho'`` returns ρ/⟨ρ⟩ (overdensity) via mass-weighted CIC deposition.
        Other fields are not yet implemented; they require SPH-kernel weighting
        against per-particle ``Density`` and ``SmoothingLength`` to recover the
        mass-weighted intensive quantity correctly.
        """
        if field == "rho":
            meta = self.get_box_meta(physics_id)
            grid = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
            n_total = 0
            for chunk in self.iter_gas_chunks(
                physics_id, fields=("Coordinates", "Masses")
            ):
                _cic_deposit_inplace(
                    grid,
                    chunk["Coordinates"],
                    chunk["Masses"],
                    box=meta["box_kpc_h"],
                    n_grid=n_grid,
                )
                n_total += chunk["Coordinates"].shape[0]
            if n_total == 0:
                raise ValueError("No gas particles found across snap files")
            mean = grid.mean()
            if mean <= 0:
                raise ValueError(
                    f"CIC-deposited grid has non-positive mean ({mean}); "
                    "check that Coordinates and Masses are populated."
                )
            rho_over_mean = grid / mean
            self._validate_data("rho", rho_over_mean)
            return rho_over_mean

        if field in ("T", "xHI", "vlos"):
            raise NotImplementedError(
                f"Field '{field}' requires SPH-kernel weighting against "
                "PartType0/Density and PartType0/SmoothingLength. Track in "
                "the next data-engineer dispatch when the metric module needs it."
            )

        raise ValueError(f"Unknown field {field!r}; expected one of "
                         "'rho', 'T', 'xHI', 'vlos'.")

    # -------------------------------------------------------------- validation

    def _validate_data(self, field_name: str, arr: np.ndarray) -> None:
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(f"{field_name}: {n_bad} non-finite cells")
        if field_name == "rho":
            if (arr < 0).any():
                raise ValueError(f"rho: negative cells, min = {arr.min()}")
            mean = arr.mean()
            if not (0.95 < mean < 1.05):
                raise ValueError(
                    f"rho/<rho>: mean = {mean:.4f}, expected ~1.0; "
                    "deposition may be incorrect."
                )
        elif field_name == "xHI":
            if not ((arr >= 0).all() and (arr <= 1.0001).all()):
                raise ValueError(
                    f"xHI: range [{arr.min()}, {arr.max()}], expected [0, 1]"
                )
        elif field_name == "T":
            if (arr <= 0).any():
                raise ValueError(f"T: non-positive cells, min = {arr.min()}")


# ----------------------------------------------------------------------- CIC


def _cic_deposit_inplace(
    grid: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray,
    box: float,
    n_grid: int,
) -> None:
    """Cloud-in-cell deposition of ``weights`` at ``coords`` accumulated
    *in place* into ``grid`` (shape ``(n_grid,)*3``, dtype float64).

    Stays in float32 for the per-particle arithmetic (~½ the memory) and
    promotes to float64 only at the per-cell accumulation step. Designed
    to be called once per HDF5 sub-file so the full particle table is
    never materialized.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")
    if weights.shape[0] != coords.shape[0]:
        raise ValueError(
            f"weights length {weights.shape[0]} != coords length {coords.shape[0]}"
        )

    coords = np.ascontiguousarray(coords, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)

    cell = np.float32(box / n_grid)
    fx = coords[:, 0] / cell
    fy = coords[:, 1] / cell
    fz = coords[:, 2] / cell

    ix = np.floor(fx).astype(np.int32)
    iy = np.floor(fy).astype(np.int32)
    iz = np.floor(fz).astype(np.int32)
    dx = fx - ix
    dy = fy - iy
    dz = fz - iz

    ix0 = ix % n_grid
    iy0 = iy % n_grid
    iz0 = iz % n_grid
    ix1 = (ix + 1) % n_grid
    iy1 = (iy + 1) % n_grid
    iz1 = (iz + 1) % n_grid

    n3 = n_grid * n_grid * n_grid
    flat_view = grid.reshape(n3)
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
        # int64 promotion is needed because n3 can exceed int32 max for n_grid>=1290;
        # for n_grid=768 we are well under, but cheap insurance.
        idx = ax.astype(np.int64) * n_grid * n_grid + ay.astype(np.int64) * n_grid + az
        flat_view += np.bincount(idx, weights=weights * wx * wy * wz, minlength=n3)

import hashlib
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch  # kept for downstream import-compatibility
from scipy.ndimage import label as _scipy_label

# ---------------------------------------------------------------------------
# [D-42] milestone 1: velocity-gradient ground-truth sidecar
# ---------------------------------------------------------------------------
# Cache of (mean, std) of the centered-finite-difference of `v_pec` along the
# LOS axis, keyed by (physics_id, redshift_rounded_3dp). Computed ONCE per
# (physics, redshift) pair so the same normalization is used regardless of
# whether the caller is a smoke run, a pub run, or a downstream diagnostic.
# In-memory dict (matches the project's convention — no joblib pickles or
# .npy sidecars elsewhere in this loader; pipeline.py instantiates a fresh
# SherwoodLoader per run so this dict is per-process).
_VPEC_GRAD_STATS_CACHE: Dict[Tuple[int, float], Dict[str, float]] = {}

# ---------------------------------------------------------------------------
# Stage 3 infrastructure: 3D overdensity crop extraction for the feedback
# classifier (parallel to [D-46]).
# ---------------------------------------------------------------------------
# In-memory cache of full 3D overdensity grids (rho / <rho>) keyed by
# (physics_id, redshift_rounded_3dp, n_grid). The CIC deposition of ~10^9
# particles to a 768^3 grid is expensive (minutes per physics, ~3.4 GB
# working set), so we hold the result for the lifetime of the loader's
# process. The cache is bounded by the host's RAM — caller is responsible
# for not loading more physics variants than the host can hold.
_RHO_FIELD_CACHE: Dict[Tuple[int, float, int], np.ndarray] = {}

# Upper-bound sanity check for the overdensity field. CIC deposition is
# mathematically non-negative (zero in cells whose 8-corner support carries
# no particles — common at high n_grid where the mean per-cell occupancy
# drops below 1); the validator therefore enforces only the upper bound
# plus non-negativity, not a positive floor. _RHO_CROP_HI = 1e3 is above
# a virialized halo and would indicate a deposition bug. _RHO_CROP_LO is
# retained as a documentation anchor for what "suspicious near-zero" looks
# like but is no longer used as an assertion floor.
_RHO_CROP_LO = 1.0e-3  # heuristic only; not enforced
_RHO_CROP_HI = 1.0e3

# ---------------------------------------------------------------------------
# Disk-cache for CIC-deposited rho/<rho> fields (Sprint-1 of [D-46]/[D-47]
# Stage 3 infra prep). Persists the result of `load_3d_field` between Python
# processes so the ~9 min/process CIC cost is paid once-ever per
# (physics_id, redshift, n_grid). On a cache hit the disk read is mmap'd and
# the in-memory `_RHO_FIELD_CACHE` is repopulated; target <= 15 s round-trip.
#
# Default location is D:\...\Sherwood\.rho_field_cache\ per the project's
# C:-drive-constrained storage layout. Override via COSMOGAS_RHO_CACHE_DIR.
# ---------------------------------------------------------------------------
_RHO_CACHE_SCHEMA_VERSION = 1

# Sentinel for the default cache location. Resolved at call time (not import
# time) so a test can `monkeypatch.setenv("COSMOGAS_RHO_CACHE_DIR", ...)`
# without having to reload the module.
_RHO_CACHE_DEFAULT_DIR = Path(
    r"D:\Data\sujin\CosmoGasVision\Sherwood\.rho_field_cache"
)


def _resolve_rho_cache_dir() -> Path:
    """Return the directory where rho-field cache entries are stored.

    Honors the ``COSMOGAS_RHO_CACHE_DIR`` env var when set; otherwise falls
    back to the D:-drive default. Per the storage directive we deliberately
    do NOT fall back to ``~/.cache`` or ``tempfile.gettempdir()`` on Windows
    (both can resolve to C:); a manual override is required to relocate.
    """
    override = os.environ.get("COSMOGAS_RHO_CACHE_DIR")
    if override:
        return Path(override)
    return _RHO_CACHE_DEFAULT_DIR


def _rho_cache_basename(physics_id: int, redshift: float, n_grid: int) -> str:
    return (
        f"rho_field_p{int(physics_id)}"
        f"_z{float(redshift):.3f}"
        f"_n{int(n_grid)}"
    )


def _rho_cache_paths(
    physics_id: int, redshift: float, n_grid: int, cache_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    """Return ``(npy_path, json_path)`` for this cache key."""
    base = _rho_cache_basename(physics_id, redshift, n_grid)
    root = cache_dir if cache_dir is not None else _resolve_rho_cache_dir()
    return root / f"{base}.npy", root / f"{base}.json"


def _sherwood_snapshot_mtime_utc(physics_id: int) -> Optional[str]:
    """ISO-8601 UTC mtime of the upstream ``snap_012.0.hdf5`` for this
    physics variant, or ``None`` if the snapshot is not locally available.

    The mtime is what we use to detect that the underlying simulation has
    been re-extracted / changed; if mtime drifts the cache entry is stale
    and must be regenerated.
    """
    from src.data.igm_gal_loader import PHYSICS_DIRS as _PHYSICS_DIRS

    if physics_id not in _PHYSICS_DIRS:
        return None
    # The IGM_gal loader's default root is `SherwoodIGM_gal/extracted`,
    # resolved relative to CWD. Mirror that here. A non-default root is
    # exotic enough that callers using one are responsible for cache hygiene.
    snap = (
        Path("SherwoodIGM_gal")
        / "extracted"
        / _PHYSICS_DIRS[physics_id]
        / "snapdir_012"
        / "snap_012.0.hdf5"
    )
    if not snap.exists():
        return None
    ts = datetime.fromtimestamp(snap.stat().st_mtime, tz=timezone.utc)
    # Microsecond precision is overkill; second-precision matches the FS
    # granularity we care about and avoids spurious mismatches from clock
    # rounding.
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_first_1mb(path: Path) -> str:
    """SHA-256 of the first 1 MB of ``path``. Cheap fingerprint to detect
    truncation / partial overwrite without re-hashing a ~17 GB .npy."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1024 * 1024))
    return h.hexdigest()


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Atomic write via temp-file + ``os.replace``. Used for the manifest.
    The .npy is written via ``np.save`` to a temp path then replaced."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp_name, target)
    except Exception:
        # Best-effort cleanup on failure; never bubble the cleanup error.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_write_npy(target: Path, array: np.ndarray) -> None:
    """Atomic ``np.save`` via a sibling ``.tmp`` then ``os.replace``.

    We pass an *open file handle* to ``np.save`` (not a string path), which
    bypasses np.save's "auto-append .npy if missing" behavior — that
    behavior creates a second file at ``tmp_name + ".npy"`` while leaving
    the empty mkstemp-created file at ``tmp_name``, which would then be
    the (empty!) file we move into place. Writing to the fd directly keeps
    the data and the path we control in sync.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, array, allow_pickle=False)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _validate_rho_disk_cache(
    npy_path: Path,
    json_path: Path,
    physics_id: int,
    redshift: float,
    n_grid: int,
) -> Optional[np.ndarray]:
    """Validate the on-disk cache entry for this key.

    Returns the mmap'd array on success, or ``None`` on any validation
    failure. On failure, the corrupt cache files are removed and a
    ``warnings.warn`` is emitted describing the failure mode. No exception
    is raised — the caller falls through to fresh CIC deposition. This is
    by spec: cache-corruption must not bubble; only fresh-CIC failure does.
    """
    if not (npy_path.exists() and json_path.exists()):
        return None

    failure_mode: Optional[str] = None
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        failure_mode = f"manifest unreadable: {exc!r}"
        manifest = None

    if manifest is not None:
        # Schema version gate
        if manifest.get("schema_version") != _RHO_CACHE_SCHEMA_VERSION:
            failure_mode = (
                f"schema_version {manifest.get('schema_version')!r} != "
                f"{_RHO_CACHE_SCHEMA_VERSION}"
            )

        # Key fields
        elif int(manifest.get("physics_id", -1)) != int(physics_id):
            failure_mode = (
                f"physics_id mismatch: manifest={manifest.get('physics_id')} "
                f"requested={physics_id}"
            )
        elif round(float(manifest.get("redshift", -1.0)), 3) != round(
            float(redshift), 3
        ):
            failure_mode = (
                f"redshift mismatch: manifest={manifest.get('redshift')} "
                f"requested={redshift}"
            )
        elif int(manifest.get("n_grid", -1)) != int(n_grid):
            failure_mode = (
                f"n_grid mismatch: manifest={manifest.get('n_grid')} "
                f"requested={n_grid}"
            )

        # mtime check
        elif (current_mtime := _sherwood_snapshot_mtime_utc(physics_id)) is not None:
            if manifest.get("sherwood_snapshot_mtime_utc") != current_mtime:
                failure_mode = (
                    f"sherwood_snapshot_mtime_utc mismatch: manifest="
                    f"{manifest.get('sherwood_snapshot_mtime_utc')!r} "
                    f"current={current_mtime!r}"
                )

        # NaN in stats (degenerate manifest)
        elif any(
            (not isinstance(manifest.get(k), (int, float)))
            or not np.isfinite(manifest.get(k))
            for k in ("mean", "min", "max")
        ):
            failure_mode = "manifest mean/min/max non-finite or absent"

        # Loose physical-mean bound for rho/<rho>
        elif not (0.95 <= float(manifest["mean"]) <= 1.05):
            failure_mode = (
                f"manifest mean {manifest['mean']:.4f} outside [0.95, 1.05] "
                "for rho/<rho>"
            )

        # First-MB hash (detects partial truncation / overwrite)
        else:
            try:
                actual_hash = _sha256_first_1mb(npy_path)
            except OSError as exc:
                failure_mode = f"sha256 read failed: {exc!r}"
                actual_hash = None
            if (
                failure_mode is None
                and manifest.get("sha256_first_1MB") != actual_hash
            ):
                failure_mode = (
                    f"sha256_first_1MB mismatch: manifest="
                    f"{manifest.get('sha256_first_1MB')!r} "
                    f"actual={actual_hash!r}"
                )

    if failure_mode is None:
        # Open as mmap; shape/dtype assertion last (np.load surfaces I/O
        # errors and pickle issues before we touch the data).
        try:
            arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            failure_mode = f"np.load failed: {exc!r}"
            arr = None
        if failure_mode is None:
            expected_shape = (int(n_grid),) * 3
            if arr.shape != expected_shape:
                failure_mode = (
                    f"shape mismatch: on-disk {arr.shape} != expected "
                    f"{expected_shape}"
                )
            elif arr.dtype != np.dtype(manifest["dtype"]):
                failure_mode = (
                    f"dtype mismatch: on-disk {arr.dtype} != manifest "
                    f"{manifest['dtype']!r}"
                )

    if failure_mode is not None:
        warnings.warn(
            f"rho-field disk cache invalid for "
            f"(physics_id={physics_id}, redshift={redshift}, n_grid={n_grid}): "
            f"{failure_mode}. Removing and falling through to CIC deposition.",
            RuntimeWarning,
            stacklevel=2,
        )
        for p in (npy_path, json_path):
            try:
                p.unlink()
            except OSError:
                pass
        return None

    return arr


def _write_rho_disk_cache(
    rho_field: np.ndarray,
    physics_id: int,
    redshift: float,
    n_grid: int,
) -> None:
    """Persist ``rho_field`` to the disk cache atomically (npy first, then
    manifest). On any failure, best-effort clean up partial files and emit
    a warning; never bubble (cache write failure must not break the call).
    """
    npy_path, json_path = _rho_cache_paths(physics_id, redshift, n_grid)
    try:
        _atomic_write_npy(npy_path, rho_field)
        sha = _sha256_first_1mb(npy_path)
        manifest = {
            "physics_id": int(physics_id),
            "redshift": float(redshift),
            "n_grid": int(n_grid),
            "sherwood_snapshot_mtime_utc": _sherwood_snapshot_mtime_utc(physics_id),
            "schema_version": _RHO_CACHE_SCHEMA_VERSION,
            "sha256_first_1MB": sha,
            "shape": list(rho_field.shape),
            "dtype": str(rho_field.dtype),
            "mean": float(rho_field.mean()),
            "min": float(rho_field.min()),
            "max": float(rho_field.max()),
        }
        _atomic_write_bytes(
            json_path,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
    except Exception as exc:  # noqa: BLE001 — we intentionally don't bubble
        warnings.warn(
            f"Failed to write rho-field disk cache for "
            f"(physics_id={physics_id}, redshift={redshift}, n_grid={n_grid}): "
            f"{exc!r}. The in-memory cache is still populated; this run is "
            "not affected, but a future process will re-run CIC.",
            RuntimeWarning,
            stacklevel=2,
        )
        for p in (npy_path, json_path):
            try:
                p.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Sprint-2 [D-49]: held-out region spatial split for the Stage 3 classifier.
# Mandatory mitigation for the [D-12] anti-leakage rule under the [D-46]
# physics_id-embedding walk-back, per [D-47] option-C hybrid framing.
# See experiments/nerf/design/sprint2_heldout_split.md for the design.
# ---------------------------------------------------------------------------

Region = Literal["train", "val", "test", "heldout"]


@dataclass(frozen=True)
class HeldoutSplitScheme:
    """Geometry of the train / val / test partition of the periodic box.

    Contiguous-slab partition along one axis (x by default); other axes are
    unconstrained. train + val + test cover [0, 1] exactly on the split axis;
    train is [0, train_x_max), val is [train_x_max, val_x_max), test is
    [val_x_max, 1.0). All intervals right-open in normalized box coords.
    """

    train_x_max: float = 0.7
    val_x_max: float = 0.85
    axis: int = 0  # 0=x, 1=y, 2=z


DEFAULT_SCHEME = HeldoutSplitScheme()


def _coord_to_array(
    coord: Union[float, np.ndarray, "torch.Tensor"]
) -> np.ndarray:
    """Convert region-helper input to a float64 np.ndarray. Accepts:
    - float (single scalar coord on the split axis)
    - np.ndarray shape (3,) (single 3D coord)
    - np.ndarray shape (N, 3) (batch of 3D coords)
    - torch.Tensor of any of the above shapes
    """
    if hasattr(coord, "detach"):
        coord = coord.detach().cpu().numpy()
    return np.asarray(coord, dtype=np.float64)


def _region_label_scalar(x: float, scheme: HeldoutSplitScheme) -> str:
    if x < scheme.train_x_max:
        return "train"
    if x < scheme.val_x_max:
        return "val"
    return "test"


def region_mask(
    coord_normalized: Union[float, np.ndarray, "torch.Tensor"],
    scheme: HeldoutSplitScheme = DEFAULT_SCHEME,
) -> Union[str, np.ndarray]:
    """Discrete region label given normalized coord(s).

    Parameters
    ----------
    coord_normalized : float, np.ndarray (3,), or np.ndarray (N, 3)
        Coord in [0, 1). For 3D input only ``coord_normalized[..., scheme.axis]``
        is consulted; scalar input is interpreted as already on the split axis.

    Returns
    -------
    str ("train" | "val" | "test") for single-coord input;
    np.ndarray of dtype <U5 for batched (N, 3) input.
    """
    arr = _coord_to_array(coord_normalized)
    if arr.ndim == 0:
        return _region_label_scalar(float(arr), scheme)
    if arr.ndim == 1 and arr.shape == (3,):
        return _region_label_scalar(float(arr[scheme.axis]), scheme)
    if arr.ndim == 2 and arr.shape[1] == 3:
        xs = arr[:, scheme.axis]
        out = np.empty(xs.shape, dtype="<U5")
        out[xs < scheme.train_x_max] = "train"
        out[(xs >= scheme.train_x_max) & (xs < scheme.val_x_max)] = "val"
        out[xs >= scheme.val_x_max] = "test"
        return out
    raise ValueError(
        f"coord_normalized must be float, (3,), or (N, 3); got shape {arr.shape}"
    )


def _dist_to_train_scalar(x: float, scheme: HeldoutSplitScheme) -> float:
    if x < scheme.train_x_max:
        return 0.0
    return min(x - scheme.train_x_max, 1.0 - x)


def distance_to_train_region(
    coord_normalized: Union[float, np.ndarray, "torch.Tensor"],
    scheme: HeldoutSplitScheme = DEFAULT_SCHEME,
) -> Union[float, np.ndarray]:
    """Shortest periodic 1D distance along ``scheme.axis`` from the query
    coord to the train interval ``[0, scheme.train_x_max]``.

    0 inside train; in (0, 0.5*(1 - train_x_max)] outside train. Periodic
    wraparound: with train_x_max=0.7, a point at x=0.99 is distance 0.01
    (close via wraparound back to x=0), NOT 0.29.
    """
    arr = _coord_to_array(coord_normalized)
    if arr.ndim == 0:
        return _dist_to_train_scalar(float(arr), scheme)
    if arr.ndim == 1 and arr.shape == (3,):
        return _dist_to_train_scalar(float(arr[scheme.axis]), scheme)
    if arr.ndim == 2 and arr.shape[1] == 3:
        xs = arr[:, scheme.axis]
        outside = xs >= scheme.train_x_max
        below = xs - scheme.train_x_max
        above = 1.0 - xs
        dist = np.zeros_like(xs)
        dist[outside] = np.minimum(below[outside], above[outside])
        return dist
    raise ValueError(
        f"coord_normalized must be float, (3,), or (N, 3); got shape {arr.shape}"
    )


# Header byte layout matches Sherwood/src/utils.py:
#   7 doubles (z, Om, OL, Ob, h100, box, Xh) + 2 int32 (nbins, num_los)
_N_HEADER_BYTES = 7 * 8 + 2 * 4

# DLA detection thresholds, per [D-24]:
#   - DLA "core" bin: tau_gt > 1e5 (raw simulator units)
#   - DLA region: connected component of tau_gt > 10 around each core
#     (the damping wing). Forest-bin clamp at tau=10 lives in the loss, not
#     in the loader.
_DLA_CORE_TAU = 1.0e5
_DLA_WING_TAU = 1.0e1


class SherwoodLoader:
    """
    Loader for Sherwood simulation binary data.

    Ported and enhanced from Sherwood/src/utils.py. Handles per-snapshot
    sightline binaries (`los2048_*.dat`), the redshift-space optical-depth
    file (`tauH1_*.dat`), per-bin DLA detection per [D-24], and exposes the
    second half of the tau file as `tau_h1_real` (real-space companion;
    diagnostic-only, not used for training).

    File-half convention (confirmed numerically by
    `scripts/diag_tau_filehalf.py`, P1 z=0.300, sightline 0):
      - first  nbins*num_los doubles -> redshift-space tau (`tau_h1`),
      - second nbins*num_los doubles -> real-space   tau (`tau_h1_real`).
    The redshift-space half is the training target per [D-24]; the real-space
    half is exposed for diagnostics only (e.g. RSD-shift studies).
    """

    def __init__(self, data_root: str):
        self.data_root = data_root
        # Mapping physics IDs to directory names
        self.physics_models = {
            1: 'Physics1_nofeedback',
            2: 'Physics2_stellarwind',
            3: 'Physics3_windAGN',
            4: 'Physics4_windstrongAGN'
        }

    def load_sightlines(
        self,
        physics_id: int,
        redshift: float,
        nspec: int = 16384,
        dla_threshold_log_nhi: float = 20.3,
    ) -> Dict[str, np.ndarray]:
        """
        Load Sherwood sightline data + redshift-space tau + real-space
        companion + DLA mask per [D-24].

        Parameters
        ----------
        physics_id : int
            One of {1, 2, 3, 4}.
        redshift : float
            Snapshot redshift (e.g. 0.300).
        nspec : int
            Number of sightlines in the file (default 16384).
        dla_threshold_log_nhi : float
            Documentation-only argument carrying the literature DLA column-
            density threshold log10(N_HI) >= 20.3 (Wolfe+ 2005). The actual
            mask is built from the on-disk tau via `_detect_dla_mask` since
            N_HI is not materialized in `tauH1_*.dat`. Reserved for a future
            N_HI-based detection path.

        Returns
        -------
        dict with keys:
            'header', 'iaxis', 'xaxis', 'yaxis', 'zaxis',
            'pos_axis', 'vel_axis',
            'density', 'h1_frac', 'temp', 'v_pec',
            'tau_h1'         : (num_los, nbins) redshift-space tau (training target)
            'tau_h1_real'    : (num_los, nbins) real-space tau (diagnostic only)
            'mask_no_dla'    : (num_los, nbins) bool, True on bins to *include*
                               in loss/mean-flux reductions (per [D-24])
            'dla_threshold_log_nhi' : float (echoed from the argument)
            'v_pec_grad_truth' : (num_los, nbins) float32, [D-42] sidecar —
                               centered finite-difference of `v_pec` along
                               the LOS axis with periodic BCs, z-scored across
                               the full (physics_id, redshift) dataset.
            'v_pec_grad_stats' : dict {'mean': float, 'std': float, 'dchi_mpc_h': float}
                               — raw-units mean/std used for the z-score (cached
                               across calls with matching (physics_id, redshift)).
        """
        if physics_id not in self.physics_models:
            raise ValueError(f"Invalid physics_id {physics_id}. Must be 1-4.")

        sim_name = self.physics_models[physics_id]
        base_path = os.path.join(self.data_root, sim_name)

        # File names as per Sherwood naming convention
        los_file = os.path.join(base_path, f"los2048_n{nspec}_z{redshift:.3f}.dat")
        tau_file = os.path.join(base_path, f"tauH1_2048_n{nspec}_z{redshift:.3f}.dat")

        if not os.path.exists(los_file):
            raise FileNotFoundError(f"LOS file not found: {los_file}")
        if not os.path.exists(tau_file):
            raise FileNotFoundError(f"Tau file not found: {tau_file}")

        # ---------------------------------------------------------------- LOS
        with open(los_file, "rb") as f:
            # Header
            header = {
                'redshift': np.fromfile(f, dtype=np.double, count=1)[0],
                'omega_m': np.fromfile(f, dtype=np.double, count=1)[0],
                'omega_l': np.fromfile(f, dtype=np.double, count=1)[0],
                'omega_b': np.fromfile(f, dtype=np.double, count=1)[0],
                'h100': np.fromfile(f, dtype=np.double, count=1)[0],
                'box_kpc_h': np.fromfile(f, dtype=np.double, count=1)[0],
                'Xh': np.fromfile(f, dtype=np.double, count=1)[0],
                'nbins': np.fromfile(f, dtype=np.int32, count=1)[0],
                'num_los': np.fromfile(f, dtype=np.int32, count=1)[0]
            }

            nbins = int(header['nbins'])
            num_los = int(header['num_los'])

            # Coordinates
            iaxis = np.fromfile(f, dtype=np.int32, count=num_los)      # 1=x, 2=y, 3=z
            xaxis = np.fromfile(f, dtype=np.double, count=num_los)     # kpc/h
            yaxis = np.fromfile(f, dtype=np.double, count=num_los)     # kpc/h
            zaxis = np.fromfile(f, dtype=np.double, count=num_los)     # kpc/h

            # Axes
            pos_axis = np.fromfile(f, dtype=np.double, count=nbins)  # kpc/h
            vel_axis = np.fromfile(f, dtype=np.double, count=nbins)  # km/s

            # Physical fields
            density = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
            h1_frac = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
            temp    = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
            v_pec   = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))

        # ---------------------------------------------------------------- tau
        # File contains TWO contiguous (num_los, nbins) double blocks:
        #   block 1: redshift-space tau (training target, per [D-24]/[D-06])
        #   block 2: real-space tau     (diagnostic only)
        # Confirmed by scripts/diag_tau_filehalf.py on P1 z=0.300, sightline 0:
        #   half1 vs RSD-surrogate corr = +0.582,  vs real-surrogate = +0.034
        #   half2 vs RSD-surrogate corr = +0.025,  vs real-surrogate = +0.670
        n_per_block = nbins * num_los
        expected_two_block_bytes = 2 * n_per_block * 8 + 0  # tau file has no header
        actual_size = os.path.getsize(tau_file)

        # Sidecar sanity warn (non-fatal): the on-disk file is exactly 2x the
        # single-block size. If it ever grows beyond that, surface it.
        if actual_size > expected_two_block_bytes:
            warnings.warn(
                f"tau file {tau_file} is {actual_size} bytes; expected "
                f"<= {expected_two_block_bytes} (2 x num_los x nbins x 8). "
                f"Extra bytes are not parsed.",
                stacklevel=2,
            )

        tau_h1_real: Optional[np.ndarray] = None
        with open(tau_file, "rb") as f:
            tau_h1 = np.fromfile(f, dtype=np.double, count=n_per_block).reshape((num_los, nbins))
            if actual_size >= expected_two_block_bytes:
                tau_h1_real = np.fromfile(
                    f, dtype=np.double, count=n_per_block
                ).reshape((num_los, nbins))
            else:
                # Older / truncated file: emit a real-space sentinel of NaNs so
                # downstream consumers can detect absence without a KeyError.
                warnings.warn(
                    f"tau file {tau_file} has only one block "
                    f"({actual_size} bytes); `tau_h1_real` returned as NaNs.",
                    stacklevel=2,
                )
                tau_h1_real = np.full((num_los, nbins), np.nan, dtype=np.double)

        # ---------------------------------------------------------- sanitize
        density     = np.nan_to_num(density,     nan=0.0)
        h1_frac     = np.nan_to_num(h1_frac,     nan=0.0)
        temp        = np.nan_to_num(temp,        nan=1.0e4)  # warm IGM default
        v_pec       = np.nan_to_num(v_pec,       nan=0.0)
        tau_h1      = np.nan_to_num(tau_h1,      nan=0.0)
        # tau_h1_real is allowed to carry NaNs (diagnostic only); only sanitize
        # if it was actually read from disk.
        if not np.isnan(tau_h1_real).all():
            tau_h1_real = np.nan_to_num(tau_h1_real, nan=0.0)

        # ----------------------------------------------------- velocity-gradient sidecar
        # [D-42] milestone 1: centered finite-difference of v_pec along the
        # LOS axis with periodic BCs, z-scored once per (physics_id, redshift)
        # across the FULL dataset (no train/eval split — the spec calls for
        # stable global normalization so smoke and pub runs see identical
        # inputs).
        #
        # Δχ recovery: header['box_kpc_h'] is the cosmological box size in
        # comoving kpc/h. Convert to Mpc/h (/1000), divide by nbins to get
        # the per-bin comoving spacing.
        dchi_mpc_h = float(header['box_kpc_h']) / 1000.0 / float(nbins)
        v_pec_grad_raw = self.compute_vpec_grad(v_pec, dchi_mpc_h)

        cache_key = (int(physics_id), round(float(redshift), 3))
        cached = _VPEC_GRAD_STATS_CACHE.get(cache_key)
        if cached is None:
            g_mean = float(v_pec_grad_raw.mean())
            g_std = float(v_pec_grad_raw.std())
            if not np.isfinite(g_std) or g_std <= 0.0:
                # Degenerate dataset (e.g. zero v_pec everywhere): bail to a
                # safe default rather than divide-by-zero. Documented choice:
                # std → 1.0 leaves the z-scored field equal to the raw field
                # with mean subtracted, which is the closest non-divergent
                # behavior. Warn so this doesn't pass silently.
                warnings.warn(
                    f"v_pec_grad has non-positive std ({g_std}) for "
                    f"physics={physics_id} z={redshift}; using std=1.0 fallback.",
                    stacklevel=2,
                )
                g_std = 1.0
            _VPEC_GRAD_STATS_CACHE[cache_key] = {
                'mean': g_mean,
                'std': g_std,
                'dchi_mpc_h': dchi_mpc_h,
            }
        g_mean = _VPEC_GRAD_STATS_CACHE[cache_key]['mean']
        g_std = _VPEC_GRAD_STATS_CACHE[cache_key]['std']
        v_pec_grad_truth = ((v_pec_grad_raw - g_mean) / g_std).astype(np.float32, copy=False)

        # ----------------------------------------------------- sanity checks
        self._validate_data(density, h1_frac, temp, tau_h1, v_pec_grad_truth)

        # ------------------------------------------------------ DLA masking
        mask_no_dla = self._detect_dla_mask(tau_h1)

        return {
            'header': header,
            'iaxis': iaxis,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'zaxis': zaxis,
            'pos_axis': pos_axis,
            'vel_axis': vel_axis,
            'density': density,
            'h1_frac': h1_frac,
            'temp': temp,
            'v_pec': v_pec,
            'tau_h1': tau_h1,
            'tau_h1_real': tau_h1_real,
            'mask_no_dla': mask_no_dla,
            'dla_threshold_log_nhi': float(dla_threshold_log_nhi),
            'v_pec_grad_truth': v_pec_grad_truth,
            'v_pec_grad_stats': dict(_VPEC_GRAD_STATS_CACHE[cache_key]),
        }

    # ------------------------------------------------------------- internals
    @staticmethod
    def compute_vpec_grad(v_pec: np.ndarray, dchi: float) -> np.ndarray:
        """
        Centered finite-difference of `v_pec` along the LOS (last) axis with
        periodic boundary conditions, per [D-42] milestone 1.

            g[..., i] = (v_pec[..., i+1] - v_pec[..., i-1]) / (2 * dchi)

        Implemented via `torch.roll` (avoids np-pad-and-slice complexity);
        the spec mandates "matches a reference NumPy implementation to 1e-12".
        Operates in float32 to match the project's training dtype, but the
        roll-and-subtract is dtype-agnostic.

        Parameters
        ----------
        v_pec : np.ndarray, shape (..., nbins)
            Peculiar velocity field in km/s.
        dchi : float
            Comoving bin spacing in Mpc/h (= box_kpc_h / 1000 / nbins). The
            constant is recovered from the on-disk header — NOT hard-coded.

        Returns
        -------
        g : np.ndarray, same shape as v_pec, dtype float32
            Centered-difference gradient in km/s per (comoving Mpc/h),
            detached from any grad graph (input was numpy).
        """
        v_pec_t = torch.from_numpy(np.asarray(v_pec, dtype=np.float32))
        # Periodic boundary: roll along the last axis
        g_t = (
            torch.roll(v_pec_t, shifts=-1, dims=-1)
            - torch.roll(v_pec_t, shifts=+1, dims=-1)
        ) / (2.0 * float(dchi))
        return g_t.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _detect_dla_mask(
        tau: np.ndarray,
        core_threshold: float = _DLA_CORE_TAU,
        wing_threshold: float = _DLA_WING_TAU,
    ) -> np.ndarray:
        """
        Per-sightline DLA detection per [D-24].

        Algorithm (per row of `tau`, axis=-1 = velocity bins):
          1. Cores: bins with tau > `core_threshold` (default 1e5).
          2. Region: connected component of bins with tau > `wing_threshold`
             (default 10) containing each core. The whole connected component
             of the over-10 mask containing a core bin is masked out.
          3. `mask_no_dla[i, j] = True` on bins that are NOT inside any
             DLA region of sightline `i`.

        We use `scipy.ndimage.label` per-sightline (1D structure). Per-
        sightline rather than 2D-cross-sightline labelling per the PI dispatch.

        Parameters
        ----------
        tau : np.ndarray, shape (num_los, nbins)
            Redshift-space optical depth.

        Returns
        -------
        mask_no_dla : np.ndarray of bool, shape (num_los, nbins)
            True on bins to *include* in loss/mean-flux reductions.
        """
        if tau.ndim == 1:
            tau = tau[None, :]
            squeeze = True
        else:
            squeeze = False

        num_los, nbins = tau.shape
        mask_no_dla = np.ones_like(tau, dtype=bool)

        wing_mask_full = tau > wing_threshold
        core_mask_full = tau > core_threshold

        for i in range(num_los):
            wing_row = wing_mask_full[i]
            if not wing_row.any():
                continue  # nothing to do
            core_row = core_mask_full[i]
            if not core_row.any():
                # Tau peaks above the wing threshold (>10) but never reaches
                # a DLA core (>1e5): a strong forest absorber, not a DLA.
                # Per [D-24] forest cutoff lives in the loss; do not mask.
                continue

            # 1D connected-component labelling on the wing mask
            labels, n_comp = _scipy_label(wing_row)
            if n_comp == 0:
                continue

            # Components touching at least one core bin become the DLA region
            core_labels = np.unique(labels[core_row])
            core_labels = core_labels[core_labels > 0]
            if core_labels.size == 0:
                continue

            dla_region = np.isin(labels, core_labels)
            mask_no_dla[i, dla_region] = False

        if squeeze:
            mask_no_dla = mask_no_dla[0]
        return mask_no_dla

    def _validate_data(
        self,
        density: np.ndarray,
        h1_frac: np.ndarray,
        temp: np.ndarray,
        tau: np.ndarray,
        v_pec_grad_truth: Optional[np.ndarray] = None,
    ):
        """
        Validates astrophysical data ranges.
        """
        assert (density >= 0).all(), f"Negative density found: {density.min()}"
        assert (h1_frac >= 0).all() and (h1_frac <= 1.0001).all(), f"Invalid H1 fraction: {h1_frac.min()} - {h1_frac.max()}"
        assert (temp > 0).all(), f"Non-positive temperature found: {temp.min()}"
        assert (tau >= 0).all(), f"Negative optical depth found: {tau.min()}"
        if v_pec_grad_truth is not None:
            # [D-42] milestone 1 — sidecar validation contract.
            assert v_pec_grad_truth.ndim == 2, (
                f"v_pec_grad_truth must be 2D (num_los, nbins); got shape "
                f"{v_pec_grad_truth.shape}"
            )
            assert density.shape == v_pec_grad_truth.shape, (
                f"v_pec_grad_truth shape {v_pec_grad_truth.shape} does not match "
                f"density shape {density.shape}"
            )
            assert np.isfinite(v_pec_grad_truth).all(), (
                "v_pec_grad_truth contains non-finite values (NaN or inf)"
            )
            g_mean = float(v_pec_grad_truth.mean())
            g_std = float(v_pec_grad_truth.std())
            assert -0.05 <= g_mean <= 0.05, (
                f"v_pec_grad_truth post-zscore mean {g_mean:.6f} outside [-0.05, 0.05]"
            )
            assert 0.9 <= g_std <= 1.1, (
                f"v_pec_grad_truth post-zscore std {g_std:.6f} outside [0.9, 1.1]"
            )
        print("Data sanity check passed.")

    # ----------------------------------------------------------------------
    # Stage 3 infrastructure: 3D overdensity crop extraction (parallel to
    # [D-46]). Used by the feedback classifier (next dispatch).
    # ----------------------------------------------------------------------
    def extract_rho_crops(
        self,
        physics_id: int,
        redshift: float,
        crop_size: int,
        n_crops: int,
        seed: int = 42,
        n_grid: int = 768,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Extract random 3D overdensity (rho / <rho>) cubes from the native
        simulation grid for the feedback classifier.

        The full 3D field is materialized once per (physics_id, redshift,
        n_grid) via `SherwoodIGMGalLoader.load_3d_field('rho', ...)` —
        CIC-deposited from the GADGET HDF5 snapshot — and cached in module
        memory for the lifetime of the process. Crops are then carved from
        the cached field via periodic-BC slicing.

        Parameters
        ----------
        physics_id : int
            One of {1, 2, 3, 4}.
        redshift : float
            Snapshot redshift (currently only z=0.300 is materialized in
            `SherwoodIGM_gal/extracted/`; other redshifts will require
            extracting the corresponding `snapdir_NNN`).
        crop_size : int
            Side length of each cubic crop in grid cells. Must satisfy
            `0 < crop_size <= n_grid`.
        n_crops : int
            Number of crops to extract per call.
        seed : int, default 42
            Seed for the RNG used to pick crop corners. Same seed produces
            byte-identical crops.
        n_grid : int, default 768
            Side length of the native simulation grid. The Sherwood IGM_gal
            snapshots are 768^3 native; smaller values trigger a CIC
            deposition at that lower resolution (useful for fast smokes).

        Returns
        -------
        crops : torch.Tensor of shape (n_crops, 1, crop_size, crop_size, crop_size)
            Float32 overdensity. Channel axis inserted for 3D-CNN consumption.
        labels : torch.Tensor of shape (n_crops,) and dtype long
            All entries equal `physics_id` (broadcast). Class label for
            the feedback classifier.

        Notes
        -----
        * Periodic BC: crop corners are drawn uniformly from `[0, n_grid)^3`;
          a crop that runs off the box wraps around via modular indexing
          (`np.take` with `mode='wrap'`). This matches the simulation's
          periodic boundary conditions.
        * Reproducibility: a fresh `np.random.default_rng(seed)` is used per
          call (no global RNG mutation).
        * Validation: `_validate_rho_crops` (positivity, NaN, range bound
          `[1e-3, 1e3]`) is asserted on the assembled tensor before return.
        * Cross-track: 3D fields live in `SherwoodIGM_gal/extracted/` and
          are loaded via `SherwoodIGMGalLoader`, NOT from the sightline
          binaries the rest of this class consumes. The `self.data_root`
          attribute is unrelated to the 3D field path; the IGM_gal loader
          uses its own default of `SherwoodIGM_gal/extracted/`.
        """
        # ---- shape / argument validation
        if physics_id not in self.physics_models:
            raise ValueError(f"Invalid physics_id {physics_id}. Must be 1-4.")
        if not isinstance(crop_size, int) or crop_size <= 0:
            raise ValueError(f"crop_size must be a positive int; got {crop_size!r}")
        if crop_size > n_grid:
            raise ValueError(
                f"crop_size {crop_size} exceeds n_grid {n_grid}; not supported."
            )
        if not isinstance(n_crops, int) or n_crops <= 0:
            raise ValueError(f"n_crops must be a positive int; got {n_crops!r}")

        # ---- materialize (or fetch from cache) the full rho / <rho> field
        cache_key = (int(physics_id), round(float(redshift), 3), int(n_grid))
        rho_field = _RHO_FIELD_CACHE.get(cache_key)
        if rho_field is None:
            # Tier-2: disk cache (Sprint-1 [D-46]/[D-47] Stage 3 infra). Try
            # to read a previously CIC-deposited field from
            # `Sherwood/.rho_field_cache/` before falling back to the ~9 min
            # CIC deposition. Validation failures are non-fatal — the disk
            # entry is cleaned up and we fall through to fresh deposition.
            npy_path, json_path = _rho_cache_paths(
                physics_id, redshift, n_grid
            )
            disk_arr = _validate_rho_disk_cache(
                npy_path, json_path, physics_id, redshift, n_grid
            )
            if disk_arr is not None:
                # mmap'd, dtype already matches what we'd produce. Cast to
                # float32 only if the on-disk dtype differs (it won't, given
                # we write float32, but defensively keep the invariant).
                if disk_arr.dtype != np.float32:
                    rho_field = np.asarray(disk_arr, dtype=np.float32)
                else:
                    # Materialize into RAM (the per-call crop loop does
                    # fancy-indexing which we don't want to thrash from
                    # an mmap; for production n_grid=768 this is ~1.7 GB
                    # float32, well within RAM budget).
                    rho_field = np.array(disk_arr, dtype=np.float32, copy=True)
                _RHO_FIELD_CACHE[cache_key] = rho_field
            else:
                # Tier-3: fresh CIC deposition (cost: ~9 min for n_grid=768).
                # Import here so SherwoodLoader has no hard h5py dependency
                # for callers that only consume sightlines.
                from src.data.igm_gal_loader import SherwoodIGMGalLoader

                igm_loader = SherwoodIGMGalLoader()
                # The IGM_gal loader does not currently key by redshift in
                # its path (it pins z=0.300); the `redshift` argument
                # participates in the cache key so a future multi-redshift
                # loader can drop in without API churn.
                rho_field = igm_loader.load_3d_field(
                    physics_id=physics_id, field="rho", n_grid=n_grid
                ).astype(np.float32, copy=False)
                _RHO_FIELD_CACHE[cache_key] = rho_field
                # Persist for the next process. Failures here warn but do
                # not bubble (current run already has the in-memory copy).
                _write_rho_disk_cache(rho_field, physics_id, redshift, n_grid)

        N = rho_field.shape[0]
        if rho_field.ndim != 3 or rho_field.shape != (N, N, N):
            raise ValueError(
                f"Expected cubic 3D rho field, got shape {rho_field.shape}"
            )

        # ---- deterministic corner sampling
        rng = np.random.default_rng(int(seed))
        corners = rng.integers(low=0, high=N, size=(n_crops, 3), dtype=np.int64)

        # ---- periodic-BC crop assembly
        # Precompute the per-axis crop offset vector [0, 1, ..., L-1].
        offset = np.arange(crop_size, dtype=np.int64)
        crops = np.empty(
            (n_crops, 1, crop_size, crop_size, crop_size), dtype=np.float32
        )
        for c in range(n_crops):
            i0, j0, k0 = corners[c]
            ii = (i0 + offset) % N
            jj = (j0 + offset) % N
            kk = (k0 + offset) % N
            # Outer-product 3D indexing — picks up the wrapped block in one shot.
            crop = rho_field[np.ix_(ii, jj, kk)]
            crops[c, 0] = crop  # float32 already

        # ---- output sanity check (positivity / NaN / loose physical bound)
        self._validate_rho_crops(crops)

        # ---- torch handoff. `from_numpy` shares memory but the upstream
        # array is a fresh np.empty so no autograd-graph contamination.
        # The tensor is leaf and `requires_grad=False`; downstream callers
        # can `.to(device)` or set `requires_grad_(True)` as needed.
        crops_t = torch.from_numpy(crops)
        labels_t = torch.full(
            (n_crops,), fill_value=int(physics_id), dtype=torch.long
        )
        return crops_t, labels_t

    def extract_rho_crops_split(
        self,
        physics_id: int,
        redshift: float,
        crop_size: int,
        n_crops: int,
        region: Region,
        scheme: HeldoutSplitScheme = DEFAULT_SCHEME,
        seed: int = 42,
        n_grid: int = 768,
        max_rejections: int = 100_000,
    ) -> Tuple["torch.Tensor", "torch.Tensor", np.ndarray]:
        """Extract crops whose voxel support is wholly within ``region``.

        Sprint-2 [D-49] companion to :meth:`extract_rho_crops`. Straddling
        crops (those whose split-axis voxel range crosses a region boundary
        or wraps around the periodic seam) are REJECTED, not relabelled,
        per the strict-rejection policy in
        ``experiments/nerf/design/sprint2_heldout_split.md`` §5.

        Parameters
        ----------
        physics_id, redshift, crop_size, n_crops, seed, n_grid
            As in :meth:`extract_rho_crops`.
        region : {"train", "val", "test", "heldout"}
            Target partition. "heldout" = val ∪ test.
        scheme : HeldoutSplitScheme
            Geometry of the partition. Defaults to 70/15/15 along axis-0.
        max_rejections : int
            Upper bound on rejected draws before raising ``RuntimeError``.
            Catches pathological combinations of ``region`` and ``crop_size``.

        Returns
        -------
        crops : torch.Tensor (n_crops, 1, crop_size, crop_size, crop_size)
        labels : torch.Tensor (n_crops,) long — physics_id (broadcast)
        distances : np.ndarray (n_crops,) float32 — per-crop
            distance_to_train_region evaluated at the crop CENTER in
            normalized box coords. 0 for ``region="train"``; > 0 otherwise.
        """
        # ---- argument validation
        if physics_id not in self.physics_models:
            raise ValueError(f"Invalid physics_id {physics_id}. Must be 1-4.")
        if not isinstance(crop_size, int) or crop_size <= 0:
            raise ValueError(f"crop_size must be a positive int; got {crop_size!r}")
        if crop_size > n_grid:
            raise ValueError(
                f"crop_size {crop_size} exceeds n_grid {n_grid}; not supported."
            )
        if not isinstance(n_crops, int) or n_crops <= 0:
            raise ValueError(f"n_crops must be a positive int; got {n_crops!r}")
        valid_regions = ("train", "val", "test", "heldout")
        if region not in valid_regions:
            raise ValueError(
                f"region must be one of {valid_regions}; got {region!r}"
            )
        if scheme.axis not in (0, 1, 2):
            raise ValueError(f"scheme.axis must be in {{0, 1, 2}}; got {scheme.axis!r}")
        if not (0.0 < scheme.train_x_max < scheme.val_x_max < 1.0):
            raise ValueError(
                "scheme must satisfy 0 < train_x_max < val_x_max < 1; got "
                f"train_x_max={scheme.train_x_max}, val_x_max={scheme.val_x_max}"
            )

        # ---- voxel-index intervals per region (right-open).
        # Use floor() implicitly via int() so the split is robust against
        # non-multiple-of-N fractions (e.g., 0.7 * 32 = 22.4 -> 22).
        train_end = int(scheme.train_x_max * n_grid)
        val_end = int(scheme.val_x_max * n_grid)
        region_voxel_start = {
            "train": 0,
            "val": train_end,
            "test": val_end,
            "heldout": train_end,
        }[region]
        region_voxel_end = {
            "train": train_end,
            "val": val_end,
            "test": n_grid,
            "heldout": n_grid,
        }[region]
        # Acceptance interval on the split axis for a corner c: a crop is
        # wholly in region iff c >= region_voxel_start AND
        # c + crop_size <= region_voxel_end. Wraparound is automatically
        # impossible because region_voxel_end <= n_grid.
        corner_min = region_voxel_start
        corner_max_inclusive = region_voxel_end - crop_size
        if corner_max_inclusive < corner_min:
            raise ValueError(
                f"crop_size {crop_size} too large for region {region!r}: "
                f"region width {region_voxel_end - region_voxel_start} voxels at "
                f"n_grid={n_grid}, scheme={scheme!r}."
            )

        # ---- materialize the rho field via the three-tier cache (sprint-1 [D-48]).
        cache_key = (int(physics_id), round(float(redshift), 3), int(n_grid))
        rho_field = _RHO_FIELD_CACHE.get(cache_key)
        if rho_field is None:
            npy_path, json_path = _rho_cache_paths(
                physics_id, redshift, n_grid
            )
            disk_arr = _validate_rho_disk_cache(
                npy_path, json_path, physics_id, redshift, n_grid
            )
            if disk_arr is not None:
                rho_field = np.array(disk_arr, dtype=np.float32, copy=True)
                _RHO_FIELD_CACHE[cache_key] = rho_field
            else:
                from src.data.igm_gal_loader import SherwoodIGMGalLoader

                igm_loader = SherwoodIGMGalLoader()
                rho_field = igm_loader.load_3d_field(
                    physics_id=physics_id, field="rho", n_grid=n_grid
                ).astype(np.float32, copy=False)
                _RHO_FIELD_CACHE[cache_key] = rho_field
                _write_rho_disk_cache(rho_field, physics_id, redshift, n_grid)

        N = rho_field.shape[0]
        if rho_field.ndim != 3 or rho_field.shape != (N, N, N):
            raise ValueError(
                f"Expected cubic 3D rho field, got shape {rho_field.shape}"
            )

        # ---- rejection sampling on the split axis. y, z are unconstrained
        # and drawn uniformly from [0, N). One RNG, advanced through both
        # accepts and rejects so determinism is preserved.
        rng = np.random.default_rng(int(seed))
        accepted = np.empty((n_crops, 3), dtype=np.int64)
        n_accept = 0
        n_rej = 0
        while n_accept < n_crops:
            c = rng.integers(low=0, high=N, size=3, dtype=np.int64)
            split_c = int(c[scheme.axis])
            if corner_min <= split_c <= corner_max_inclusive:
                accepted[n_accept] = c
                n_accept += 1
            else:
                n_rej += 1
                if n_rej > max_rejections:
                    raise RuntimeError(
                        f"Held-out split rejection sampling exceeded "
                        f"max_rejections={max_rejections}. "
                        f"region={region!r}, crop_size={crop_size}, "
                        f"n_grid={n_grid}, scheme={scheme!r}."
                    )

        # ---- crop assembly. Split-axis is non-wrapping by acceptance; y, z
        # may wrap (periodic on non-split axes). Use the same np.ix_ pattern
        # as extract_rho_crops.
        offset = np.arange(crop_size, dtype=np.int64)
        crops = np.empty(
            (n_crops, 1, crop_size, crop_size, crop_size), dtype=np.float32
        )
        for c_idx in range(n_crops):
            i0, j0, k0 = accepted[c_idx]
            ii = (i0 + offset) % N
            jj = (j0 + offset) % N
            kk = (k0 + offset) % N
            crops[c_idx, 0] = rho_field[np.ix_(ii, jj, kk)]

        # ---- output sanity check (re-use _validate_rho_crops).
        self._validate_rho_crops(crops)

        # ---- per-crop distance_to_train_region at the crop CENTER, in
        # normalized box coords.
        centers = (accepted.astype(np.float64) + (crop_size - 1) / 2.0) / float(N)
        distances = distance_to_train_region(centers, scheme).astype(np.float32)

        # ---- torch handoff
        crops_t = torch.from_numpy(crops)
        labels_t = torch.full(
            (n_crops,), fill_value=int(physics_id), dtype=torch.long
        )
        return crops_t, labels_t, distances

    @staticmethod
    def _validate_rho_crops(crops: np.ndarray) -> None:
        """
        Sanity check on assembled overdensity crops.

        Contract per the Stage 3 dispatch:
          * No NaN / inf cells.
          * All cells non-negative. CIC deposition is mathematically
            non-negative; a negative cell would indicate a deposition bug.
            Zero is permitted: at high ``n_grid`` (e.g. 768) the mean
            per-cell occupancy drops below 1 and many cells legitimately
            receive no contribution (~25% empty at the production
            n_grid=768 P1 setting; surfaced by [D-50] sprint-3 once the
            CIC OOM was lifted).
          * All cells below ``_RHO_CROP_HI`` (= 1e3). Above this is
            denser than a virialized halo and indicates a deposition bug.
        """
        if not np.isfinite(crops).all():
            n_bad = int((~np.isfinite(crops)).sum())
            raise AssertionError(
                f"rho_crops contain {n_bad} non-finite cells (NaN or inf)."
            )
        cmin = float(crops.min())
        cmax = float(crops.max())
        if cmin < 0.0:
            raise AssertionError(
                f"rho_crops min {cmin:.3e} is negative; CIC deposition "
                f"must be non-negative — indicates a deposition bug."
            )
        if cmax > _RHO_CROP_HI:
            raise AssertionError(
                f"rho_crops max {cmax:.3e} above physical ceiling "
                f"{_RHO_CROP_HI:.0e}."
            )

    def get_world_coordinates(self, data: Dict) -> np.ndarray:
        """
        Converts los indices and pos_axis into full 3D coordinates for every bin.
        Returns array of shape (num_los, nbins, 3)
        """
        num_los = data['header']['num_los']
        nbins = data['header']['nbins']

        coords = np.zeros((num_los, nbins, 3))

        # iaxis: 1=x, 2=y, 3=z. This is the axis ALONG which the sightline runs.
        # xaxis, yaxis, zaxis: the coordinates of the sightline in the other two axes.
        for i in range(num_los):
            axis = data['iaxis'][i]
            x, y, z = data['xaxis'][i], data['yaxis'][i], data['zaxis'][i]

            if axis == 1:  # Runs along x
                coords[i, :, 0] = data['pos_axis']
                coords[i, :, 1] = y
                coords[i, :, 2] = z
            elif axis == 2:  # Runs along y
                coords[i, :, 0] = x
                coords[i, :, 1] = data['pos_axis']
                coords[i, :, 2] = z
            elif axis == 3:  # Runs along z
                coords[i, :, 0] = x
                coords[i, :, 1] = y
                coords[i, :, 2] = data['pos_axis']

        return coords

import os
import warnings
import numpy as np
from typing import Dict, Optional
import torch  # kept for downstream import-compatibility
from scipy.ndimage import label as _scipy_label

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

        # ----------------------------------------------------- sanity checks
        self._validate_data(density, h1_frac, temp, tau_h1)

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
        }

    # ------------------------------------------------------------- internals
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

    def _validate_data(self, density: np.ndarray, h1_frac: np.ndarray, temp: np.ndarray, tau: np.ndarray):
        """
        Validates astrophysical data ranges.
        """
        assert (density >= 0).all(), f"Negative density found: {density.min()}"
        assert (h1_frac >= 0).all() and (h1_frac <= 1.0001).all(), f"Invalid H1 fraction: {h1_frac.min()} - {h1_frac.max()}"
        assert (temp > 0).all(), f"Non-positive temperature found: {temp.min()}"
        assert (tau >= 0).all(), f"Negative optical depth found: {tau.min()}"
        print("Data sanity check passed.")

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

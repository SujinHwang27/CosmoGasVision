"""
Diagnostic [D-24]: confirm which half of `tauH1_*.dat` is the redshift-space tau.

Per PI's tentative ruling in [D-06] amendment / [D-24], the upstream Sherwood
`utils.py:67` reads only the first `nbins*num_los` doubles of `tauH1_*.dat` and
treats it as redshift-space tau. The file on disk is exactly 2x that size; the
second half is presumed to be the real-space companion.

Numerical test (Item 1 in PI dispatch):
  1. Load both halves of `tauH1_2048_n16384_z0.300.dat` for Physics 1 (DLA-free).
  2. For sightline index 0:
     a. Build a no-Voigt no-thermal-broadening surrogate tau from
        density * h1_frac integrated against `vel_axis + v_pec`
        (real-space -> redshift-space via the peculiar-velocity offset).
     b. Build the same surrogate against `vel_axis` alone (no v_pec, real-space).
  3. Whichever file half has its strong-absorption peaks aligned with the
     `vel_axis + v_pec` surrogate is the redshift-space half.

A "diagnostic number" is the Pearson correlation coefficient between the
surrogate and the file half over the full sightline. Higher = better match.
The discriminating signal is:
    corr(half_RSD, surrogate_with_vpec) > corr(half_RSD, surrogate_no_vpec)
combined with the comparison across halves.

Run:
    PYTHONPATH=. uv run python -u scripts/diag_tau_filehalf.py
"""
from __future__ import annotations

import os
import numpy as np

# --------------------------------------------------------------------- config
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Sherwood")
PHYSICS_DIR = "Physics1_nofeedback"
REDSHIFT = 0.300
NSPEC = 16384
NBINS = 2048
LOS_FILE = os.path.join(ROOT, PHYSICS_DIR, f"los2048_n{NSPEC}_z{REDSHIFT:.3f}.dat")
TAU_FILE = os.path.join(ROOT, PHYSICS_DIR, f"tauH1_2048_n{NSPEC}_z{REDSHIFT:.3f}.dat")

SIGHTLINE_IDX = 0  # the chosen sightline (PI dispatch suggested 0)


def read_los_minimal(path: str):
    """Stream the LOS file once and return only what we need for the test:
    vel_axis (nbins,), and per-sightline density/h1_frac/v_pec arrays
    for SIGHTLINE_IDX. We avoid loading the entire (num_los, nbins) cubes.
    """
    with open(path, "rb") as f:
        # Header (mirror SherwoodLoader.load_sightlines)
        np.fromfile(f, dtype=np.double, count=7)  # ztime..Xh
        nbins = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        num_los = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        assert nbins == NBINS, f"nbins mismatch: {nbins} vs {NBINS}"
        assert num_los == NSPEC, f"num_los mismatch: {num_los} vs {NSPEC}"

        # Coordinates
        np.fromfile(f, dtype=np.int32, count=num_los)      # iaxis
        np.fromfile(f, dtype=np.double, count=num_los)     # xaxis
        np.fromfile(f, dtype=np.double, count=num_los)     # yaxis
        np.fromfile(f, dtype=np.double, count=num_los)     # zaxis

        # Axes
        np.fromfile(f, dtype=np.double, count=nbins)       # pos_axis
        vel_axis = np.fromfile(f, dtype=np.double, count=nbins)  # km/s

        # Read full cubes (we need per-sightline rows; simplest is full reshape)
        density = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
        h1_frac = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
        np.fromfile(f, dtype=np.double, count=nbins * num_los)  # temp (unused for surrogate)
        v_pec   = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))

    return vel_axis, density[SIGHTLINE_IDX], h1_frac[SIGHTLINE_IDX], v_pec[SIGHTLINE_IDX]


def read_tau_two_halves(path: str, nbins: int, num_los: int):
    """Read both halves of tauH1_*.dat. Returns (half1, half2) as
    (num_los, nbins) arrays."""
    n = nbins * num_los
    file_size = os.path.getsize(path)
    expected = 2 * n * 8
    print(f"  tau file size: {file_size}  expected (2 halves): {expected}  "
          f"ratio: {file_size / (n * 8):.3f}")
    assert file_size >= expected, f"file too small for two halves: {file_size} < {expected}"

    with open(path, "rb") as f:
        half1 = np.fromfile(f, dtype=np.double, count=n).reshape((num_los, nbins))
        half2 = np.fromfile(f, dtype=np.double, count=n).reshape((num_los, nbins))
    return half1, half2


def boxcar_redistribute(values: np.ndarray, src_velocity: np.ndarray,
                        obs_velocity: np.ndarray, box_kms: float) -> np.ndarray:
    """Redistribute per-source-bin `values` into observed velocity bins by
    nearest-neighbor binning of (vel_axis + offset) modulo the box velocity.

    This is the no-Voigt, no-thermal-broadening surrogate: every source bin
    deposits its contribution into the obs bin nearest to its observed velocity.
    Periodic wrap matches Sherwood's box convention.
    """
    nbins = obs_velocity.size
    dv = obs_velocity[1] - obs_velocity[0]
    # Map src_velocity (with periodic wrap) to obs-bin index
    shifted = np.mod(src_velocity - obs_velocity[0], box_kms)
    idx = np.floor(shifted / dv + 0.5).astype(np.int64) % nbins
    out = np.zeros(nbins, dtype=np.float64)
    np.add.at(out, idx, values)
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def main():
    print(f"=== File-half confirmation diagnostic [D-24] ===")
    print(f"LOS file: {LOS_FILE}")
    print(f"Tau file: {TAU_FILE}")
    print(f"Sightline index: {SIGHTLINE_IDX}")
    print()

    print("Reading LOS file...")
    vel_axis, density, h1_frac, v_pec = read_los_minimal(LOS_FILE)
    print(f"  vel_axis range: [{vel_axis[0]:.2f}, {vel_axis[-1]:.2f}] km/s, "
          f"dv={vel_axis[1]-vel_axis[0]:.4f} km/s")
    box_kms = (vel_axis[-1] - vel_axis[0]) + (vel_axis[1] - vel_axis[0])
    print(f"  box_kms (periodic): {box_kms:.2f}")
    print(f"  density[0:5]: {density[:5]}")
    print(f"  h1_frac[0:5]: {h1_frac[:5]}")
    print(f"  v_pec range: [{v_pec.min():.2f}, {v_pec.max():.2f}] km/s")
    print()

    print("Reading tau file (both halves)...")
    tau_h1_half1, tau_h1_half2 = read_tau_two_halves(TAU_FILE, NBINS, NSPEC)
    tau1 = tau_h1_half1[SIGHTLINE_IDX]
    tau2 = tau_h1_half2[SIGHTLINE_IDX]
    print(f"  half1 sightline {SIGHTLINE_IDX}: min={tau1.min():.3e}, "
          f"max={tau1.max():.3e}, mean={tau1.mean():.3e}")
    print(f"  half2 sightline {SIGHTLINE_IDX}: min={tau2.min():.3e}, "
          f"max={tau2.max():.3e}, mean={tau2.mean():.3e}")
    print()

    print("Building surrogate tau profiles...")
    # Surrogate intensity in each source bin: rho * f_HI (proportional to n_HI)
    n_hi_proxy = density * h1_frac

    # (1) RSD surrogate: source velocity = vel_axis + v_pec
    src_vel_rsd = vel_axis + v_pec
    surrogate_rsd = boxcar_redistribute(n_hi_proxy, src_vel_rsd, vel_axis, box_kms)

    # (2) Real-space surrogate: source velocity = vel_axis (no v_pec offset)
    surrogate_real = boxcar_redistribute(n_hi_proxy, vel_axis, vel_axis, box_kms)
    print(f"  surrogate_rsd  : min={surrogate_rsd.min():.3e}, max={surrogate_rsd.max():.3e}")
    print(f"  surrogate_real : min={surrogate_real.min():.3e}, max={surrogate_real.max():.3e}")
    print()

    print("Pearson correlations (full-profile, raw tau vs surrogate):")
    c1_rsd  = pearson(tau1, surrogate_rsd)
    c1_real = pearson(tau1, surrogate_real)
    c2_rsd  = pearson(tau2, surrogate_rsd)
    c2_real = pearson(tau2, surrogate_real)
    print(f"  half1 vs surrogate_rsd : {c1_rsd:+.4f}")
    print(f"  half1 vs surrogate_real: {c1_real:+.4f}")
    print(f"  half2 vs surrogate_rsd : {c2_rsd:+.4f}")
    print(f"  half2 vs surrogate_real: {c2_real:+.4f}")
    print()

    # The redshift-space half should correlate MORE with surrogate_rsd than
    # with surrogate_real; the real-space half should correlate MORE with
    # surrogate_real than with surrogate_rsd.
    print("Decision rule:")
    print("  redshift-space half: corr(half, surrogate_rsd) > corr(half, surrogate_real)")
    print("  real-space half    : corr(half, surrogate_real) > corr(half, surrogate_rsd)")
    print()
    half1_is_rsd = c1_rsd > c1_real
    half2_is_rsd = c2_rsd > c2_real
    print(f"  half1 prefers RSD? {half1_is_rsd}  (Δ = {c1_rsd - c1_real:+.4f})")
    print(f"  half2 prefers RSD? {half2_is_rsd}  (Δ = {c2_rsd - c2_real:+.4f})")
    print()

    # Cross-check: which half has overall stronger correlation with the
    # RSD surrogate? That is the redshift-space half.
    if c1_rsd > c2_rsd and half1_is_rsd:
        verdict = "FIRST"
    elif c2_rsd > c1_rsd and half2_is_rsd:
        verdict = "SECOND"
    else:
        verdict = "AMBIGUOUS"

    print("=" * 70)
    print(f"VERDICT: {verdict} half is redshift-space "
          f"(half1_corr_rsd={c1_rsd:+.4f}, half2_corr_rsd={c2_rsd:+.4f})")
    print("=" * 70)

    if verdict == "FIRST":
        print("Confirms PI's tentative ruling: loader keeps reading first block.")
    elif verdict == "SECOND":
        print("HALT: second half is redshift-space. Loader must be re-pointed.")
        print("Stage 2a re-validation, P1 tier-1, and the 16 micro-grid cells")
        print("are invalidated under the wrong target.")
    else:
        print("AMBIGUOUS: surface to PI; do not change loader.")


if __name__ == "__main__":
    main()

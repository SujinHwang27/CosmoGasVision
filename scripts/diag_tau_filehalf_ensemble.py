"""
Diagnostic [D-24] / S2 generalization: file-half ruling on >=10 sightlines x 4 physics.

Background
----------
The original `scripts/diag_tau_filehalf.py` validated PI's tentative ruling
(first half of `tauH1_*.dat` is redshift-space tau, second half is real-space)
on a single sightline (P1, idx 0). Defense-panel attack S2 noted: "N=1 is not
a confirmation; could be a lucky sightline; the boxcar surrogate is itself
approximate." This script generalizes: 10 sightlines per physics x 4 physics
x z=0.300, deterministic seed.

Pass criterion (per (physics, sightline)):
    corr(half1, surrogate_RSD) > corr(half2, surrogate_RSD)
Per physics: PASS if >= 9 of 10 sightlines pass; otherwise FAIL.

A failing physics is surfaced immediately and the loader's "confirmed
numerically" wording in `src/data/loader.py` MUST be revisited (potential
per-physics format drift in `tauH1_*.dat`).

Run:
    PYTHONPATH=. uv run python -u scripts/diag_tau_filehalf_ensemble.py
"""
from __future__ import annotations

import os
import numpy as np

# --------------------------------------------------------------------- config
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Sherwood")
PHYSICS_DIRS = (
    "Physics1_nofeedback",
    "Physics2_stellarwind",
    "Physics3_windAGN",
    "Physics4_windstrongAGN",
)
REDSHIFT = 0.300
NSPEC = 16384
NBINS = 2048
N_SIGHTLINES = 10
SEED = 42
PASS_THRESHOLD = 9  # >= 9 of 10 must pass per physics


def read_los_full(path: str):
    """Stream the full LOS file once and return vel_axis + per-bin
    density/h1_frac/v_pec arrays for ALL sightlines (we only ever index by
    row later, so reading once is cheaper than re-streaming per sightline)."""
    with open(path, "rb") as f:
        np.fromfile(f, dtype=np.double, count=7)  # header doubles
        nbins = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        num_los = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        assert nbins == NBINS, f"nbins mismatch: {nbins} vs {NBINS}"
        assert num_los == NSPEC, f"num_los mismatch: {num_los} vs {NSPEC}"

        np.fromfile(f, dtype=np.int32, count=num_los)      # iaxis
        np.fromfile(f, dtype=np.double, count=num_los)     # xaxis
        np.fromfile(f, dtype=np.double, count=num_los)     # yaxis
        np.fromfile(f, dtype=np.double, count=num_los)     # zaxis

        np.fromfile(f, dtype=np.double, count=nbins)       # pos_axis
        vel_axis = np.fromfile(f, dtype=np.double, count=nbins)  # km/s

        density = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
        h1_frac = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
        np.fromfile(f, dtype=np.double, count=nbins * num_los)  # temp (unused)
        v_pec   = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))

    return vel_axis, density, h1_frac, v_pec


def read_tau_two_halves(path: str, nbins: int, num_los: int):
    """Return (half1, half2) as (num_los, nbins) arrays."""
    n = nbins * num_los
    expected = 2 * n * 8
    file_size = os.path.getsize(path)
    assert file_size >= expected, (
        f"tau file too small for two halves: {file_size} < {expected}"
    )
    with open(path, "rb") as f:
        half1 = np.fromfile(f, dtype=np.double, count=n).reshape((num_los, nbins))
        half2 = np.fromfile(f, dtype=np.double, count=n).reshape((num_los, nbins))
    return half1, half2


def boxcar_redistribute(values: np.ndarray, src_velocity: np.ndarray,
                        obs_velocity: np.ndarray, box_kms: float) -> np.ndarray:
    """No-Voigt no-thermal-broadening surrogate: deposit each source bin's
    n_HI proxy into the obs bin nearest its observed velocity (periodic wrap)."""
    nbins = obs_velocity.size
    dv = obs_velocity[1] - obs_velocity[0]
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


def evaluate_physics(physics_dir: str, sightline_idx: np.ndarray):
    """Return per-sightline records for one physics. Each record is a dict
    with the four correlations + the pass flag."""
    los_file = os.path.join(ROOT, physics_dir, f"los2048_n{NSPEC}_z{REDSHIFT:.3f}.dat")
    tau_file = os.path.join(ROOT, physics_dir, f"tauH1_2048_n{NSPEC}_z{REDSHIFT:.3f}.dat")

    print(f"  reading LOS    {los_file}")
    vel_axis, density_full, h1_frac_full, v_pec_full = read_los_full(los_file)
    box_kms = (vel_axis[-1] - vel_axis[0]) + (vel_axis[1] - vel_axis[0])
    print(f"  reading TAU    {tau_file}")
    tau_half1, tau_half2 = read_tau_two_halves(tau_file, NBINS, NSPEC)
    # NaN sanitization (loader does this; mirror here so corr is defined)
    density_full = np.nan_to_num(density_full, nan=0.0)
    h1_frac_full = np.nan_to_num(h1_frac_full, nan=0.0)
    v_pec_full   = np.nan_to_num(v_pec_full,   nan=0.0)
    tau_half1    = np.nan_to_num(tau_half1,    nan=0.0)
    tau_half2    = np.nan_to_num(tau_half2,    nan=0.0)

    records = []
    for s in sightline_idx:
        density = density_full[s]
        h1_frac = h1_frac_full[s]
        v_pec   = v_pec_full[s]
        n_hi_proxy = density * h1_frac

        surrogate_rsd  = boxcar_redistribute(n_hi_proxy, vel_axis + v_pec,
                                             vel_axis, box_kms)
        surrogate_real = boxcar_redistribute(n_hi_proxy, vel_axis,
                                             vel_axis, box_kms)
        c1_rsd  = pearson(tau_half1[s], surrogate_rsd)
        c1_real = pearson(tau_half1[s], surrogate_real)
        c2_rsd  = pearson(tau_half2[s], surrogate_rsd)
        c2_real = pearson(tau_half2[s], surrogate_real)

        # Per-sightline pass: half1 favors RSD-surrogate over half2 doing the same.
        passes = c1_rsd > c2_rsd
        records.append({
            "sightline": int(s),
            "c1_rsd":  c1_rsd,
            "c1_real": c1_real,
            "c2_rsd":  c2_rsd,
            "c2_real": c2_real,
            "delta_rsd": c1_rsd - c2_rsd,
            "passes": bool(passes),
        })

    # Free large arrays before next physics
    del density_full, h1_frac_full, v_pec_full, tau_half1, tau_half2
    return records


def median_iqr(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def print_physics_table(physics_dir: str, records):
    print()
    print(f"### {physics_dir}")
    print()
    print("| sightline | c1_rsd | c1_real | c2_rsd | c2_real | delta_rsd | pass |")
    print("|-----------|--------|---------|--------|---------|-----------|------|")
    for r in records:
        flag = "PASS" if r["passes"] else "FAIL"
        print(
            f"| {r['sightline']:9d} | "
            f"{r['c1_rsd']:+.4f} | {r['c1_real']:+.4f} | "
            f"{r['c2_rsd']:+.4f} | {r['c2_real']:+.4f} | "
            f"{r['delta_rsd']:+.4f} | {flag} |"
        )

    # Median + IQR summary
    c1_rsd_med, c1_rsd_q1, c1_rsd_q3 = median_iqr([r["c1_rsd"]  for r in records])
    c1_re_med,  c1_re_q1,  c1_re_q3  = median_iqr([r["c1_real"] for r in records])
    c2_rsd_med, c2_rsd_q1, c2_rsd_q3 = median_iqr([r["c2_rsd"]  for r in records])
    c2_re_med,  c2_re_q1,  c2_re_q3  = median_iqr([r["c2_real"] for r in records])

    print()
    print(f"  median (IQR) c1_rsd  = {c1_rsd_med:+.4f} ({c1_rsd_q1:+.4f} .. {c1_rsd_q3:+.4f})")
    print(f"  median (IQR) c1_real = {c1_re_med:+.4f}  ({c1_re_q1:+.4f} .. {c1_re_q3:+.4f})")
    print(f"  median (IQR) c2_rsd  = {c2_rsd_med:+.4f} ({c2_rsd_q1:+.4f} .. {c2_rsd_q3:+.4f})")
    print(f"  median (IQR) c2_real = {c2_re_med:+.4f}  ({c2_re_q1:+.4f} .. {c2_re_q3:+.4f})")

    n_pass = sum(1 for r in records if r["passes"])
    verdict = "PASS" if n_pass >= PASS_THRESHOLD else "FAIL"
    print(f"  pass count: {n_pass}/{len(records)}  -> {verdict}")
    return verdict, n_pass


def main():
    print("=== File-half ensemble diagnostic [D-24] / S2 ===")
    print(f"redshift={REDSHIFT}  nspec={NSPEC}  nbins={NBINS}  "
          f"sightlines/physics={N_SIGHTLINES}  seed={SEED}")
    print(f"pass criterion: corr(half1, RSD-surrogate) > corr(half2, RSD-surrogate)")
    print(f"per-physics gate: >= {PASS_THRESHOLD}/{N_SIGHTLINES} sightlines pass")
    print()

    rng = np.random.RandomState(SEED)
    sightline_idx = np.sort(rng.choice(NSPEC, N_SIGHTLINES, replace=False))
    print(f"sampled sightline indices: {sightline_idx.tolist()}")

    physics_verdicts = {}
    for pdir in PHYSICS_DIRS:
        print()
        print(f"--- {pdir} ---")
        records = evaluate_physics(pdir, sightline_idx)
        verdict, n_pass = print_physics_table(pdir, records)
        physics_verdicts[pdir] = (verdict, n_pass)

    print()
    print("=" * 72)
    print("OVERALL VERDICT")
    print("=" * 72)
    all_pass = True
    for pdir, (verdict, n_pass) in physics_verdicts.items():
        flag = "PASS" if verdict == "PASS" else "FAIL"
        print(f"  {pdir:30s}  {n_pass}/{N_SIGHTLINES}  {flag}")
        if verdict != "PASS":
            all_pass = False
    print("=" * 72)

    if all_pass:
        print("OVERALL: PASS")
        print("PI's tentative ruling is upgraded to confirmed across the ensemble.")
        print("The 'tentative' wording in [D-06]/[D-24] can be struck;")
        print("loader docstring is now backed by 40 datapoints (10/physics x 4).")
    else:
        print("OVERALL: FAIL")
        print("At least one physics failed the 9/10 gate. SURFACE TO PI.")
        print("Possible cause: per-physics format drift in tauH1_*.dat.")
        print("Do NOT amend the loader docstring or LEDGER until PI re-rules.")


if __name__ == "__main__":
    main()

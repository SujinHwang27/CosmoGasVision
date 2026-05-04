"""Find a P2 sightline with a confirmed DLA core (tau > 1e5) for the
[D-24] regression test."""
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Sherwood")
NBINS = 2048
NSPEC = 16384
N = NBINS * NSPEC

for physics_dir in ("Physics2_stellarwind", "Physics3_windAGN", "Physics4_windstrongAGN"):
    p = os.path.join(ROOT, physics_dir, f"tauH1_2048_n{NSPEC}_z0.300.dat")
    print(f"\n=== {physics_dir} ===")
    # Read first half = redshift-space tau
    with open(p, "rb") as f:
        tau = np.fromfile(f, dtype=np.double, count=N).reshape((NSPEC, NBINS))
    tau = np.nan_to_num(tau, nan=0.0)
    # Sightlines whose max exceeds DLA core threshold
    row_max = tau.max(axis=1)
    dla_rows = np.where(row_max > 1.0e5)[0]
    print(f"  rows with tau_max > 1e5: {dla_rows.size}")
    if dla_rows.size:
        for r in dla_rows[:8]:
            arg = int(np.argmax(tau[r]))
            print(f"    row {r:6d}  tau_max={tau[r].max():.3e}  argmax_bin={arg}  "
                  f"row_mean={tau[r].mean():.3e}")

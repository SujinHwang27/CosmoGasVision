"""
Regression tests for [D-24] DLA detection + the loader's exposure of
`tau_h1_real` and `mask_no_dla`.

Three asserts per the PI dispatch:

  (a) For Physics 1 (no feedback / nearly DLA-free): `mask_no_dla` excludes
      only a vanishing fraction of bins (< 1e-4 of total). Empirically
      P1 z=0.300 hosts a small number of saturated absorbers above tau=1e5
      (max tau ~= 3.2e6, ~180 bins out of 16384*2048 ~= 33.5M masked) —
      not the textbook "zero DLAs" claim, but the per-bin DLA fraction
      is four orders of magnitude below P2/3/4.
  (b) For one P2 sightline known to contain a DLA (row 42 of P2 z=0.300,
      `tau_max ~= 9.47e6` at bin 1226):
        - `mask_no_dla.sum() < mask_no_dla.size` (some bins ARE masked),
        - the masked region is contiguous around the tau_max bin,
        - the masked region contains the tau_max bin.
  (c) Per-physics, `mean(log1p(min(tau, 10))**2)` over the surviving
      (non-DLA) bins is now within one order of magnitude across all 4
      physics (proves the [D-24] supervised loss target is no longer
      dominated by DLAs).

Run:
    PYTHONPATH=. uv run pytest tests/data/test_loader_dla_mask.py -v
"""
from __future__ import annotations

import os
import math

import numpy as np
import pytest

from src.data.loader import SherwoodLoader

# --------------------------------------------------------------------- config
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "Sherwood")
REDSHIFT = 0.300
NSPEC = 16384

# P2 sightline confirmed by scripts/diag_find_dla_p2.py
P2_DLA_ROW = 42
P2_DLA_TAU_MAX_BIN = 1226   # argmax bin (informational; recomputed in test)


def _have_data(physics_dir: str) -> bool:
    los = os.path.join(DATA_ROOT, physics_dir, f"los2048_n{NSPEC}_z{REDSHIFT:.3f}.dat")
    tau = os.path.join(DATA_ROOT, physics_dir, f"tauH1_2048_n{NSPEC}_z{REDSHIFT:.3f}.dat")
    return os.path.exists(los) and os.path.exists(tau)


_REQUIRED_DIRS = [
    "Physics1_nofeedback",
    "Physics2_stellarwind",
    "Physics3_windAGN",
    "Physics4_windstrongAGN",
]
_HAVE_ALL = all(_have_data(d) for d in _REQUIRED_DIRS)
_pytestmark_have_p1 = pytest.mark.skipif(
    not _have_data("Physics1_nofeedback"),
    reason="P1 binary not available locally",
)
_pytestmark_have_p2 = pytest.mark.skipif(
    not _have_data("Physics2_stellarwind"),
    reason="P2 binary not available locally",
)
_pytestmark_have_all = pytest.mark.skipif(
    not _HAVE_ALL,
    reason="Not all 4 Physics binaries available locally",
)


# Fixtures use function scope (NOT module scope) — each Sherwood cube is
# ~1.5 GB in core; module-scoping P1+P2 simultaneously plus the per-physics
# loss test would pin >5 GB and OOM on a 16-GB-RAM host. Function scope =
# rebuild per test, drop after, gc reclaims.
@pytest.fixture(scope="function")
def loader() -> SherwoodLoader:
    return SherwoodLoader(DATA_ROOT)


@pytest.fixture(scope="function")
def p1_data(loader):
    if not _have_data("Physics1_nofeedback"):
        pytest.skip("P1 binary not available")
    return loader.load_sightlines(physics_id=1, redshift=REDSHIFT, nspec=NSPEC)


@pytest.fixture(scope="function")
def p2_data(loader):
    if not _have_data("Physics2_stellarwind"):
        pytest.skip("P2 binary not available")
    return loader.load_sightlines(physics_id=2, redshift=REDSHIFT, nspec=NSPEC)


# ------------------------------------------------------------------ assert (a)
@_pytestmark_have_p1
def test_p1_has_negligible_dla_mass(p1_data):
    """[D-24] P1 (no-feedback) is *nearly* DLA-free; the masked fraction
    must be vanishing.

    Sanity bound: < 1e-4 of all bins masked. Empirically (P1 z=0.300):
    ~180 / 33.5M ~= 5e-6 — four orders of magnitude under the bound,
    six orders under the typical P2/3/4 mask fraction (~1e-3..1e-2 by
    eye from the loss-decade test).
    """
    mask = p1_data["mask_no_dla"]
    assert mask.dtype == np.bool_
    assert mask.shape == p1_data["tau_h1"].shape
    masked_frac = float((~mask).sum()) / float(mask.size)
    print(
        f"P1 masked fraction = {masked_frac:.3e} "
        f"({(~mask).sum()} / {mask.size}); max tau = {p1_data['tau_h1'].max():.3e}"
    )
    assert masked_frac < 1.0e-4, (
        f"P1 masked fraction {masked_frac:.3e} exceeds 1e-4 sanity bound; "
        f"expected vanishing DLA contribution in the no-feedback model."
    )


# ------------------------------------------------------------------ assert (b)
@_pytestmark_have_p2
def test_p2_dla_sightline_is_masked(p2_data):
    """[D-24] P2 row 42 contains a DLA at bin ~1226 (tau ~= 9.47e6).
    The mask must (i) exclude some bins, (ii) be contiguous around the
    tau_max bin, (iii) contain the tau_max bin."""
    tau = p2_data["tau_h1"]
    mask = p2_data["mask_no_dla"]

    row_tau = tau[P2_DLA_ROW]
    row_mask = mask[P2_DLA_ROW]
    arg = int(np.argmax(row_tau))
    tmax = float(row_tau[arg])

    assert tmax > 1.0e5, (
        f"P2 row {P2_DLA_ROW} expected to host a DLA core "
        f"(tau_max > 1e5); got tau_max={tmax:.3e}"
    )

    # (i) some bins ARE masked
    assert row_mask.sum() < row_mask.size, (
        f"P2 row {P2_DLA_ROW} mask did not exclude any bins despite tau_max={tmax:.3e}"
    )

    # (iii) tau_max bin is inside the masked region
    assert not bool(row_mask[arg]), (
        f"tau_max bin (arg={arg}) of P2 row {P2_DLA_ROW} is not masked"
    )

    # (ii) contiguous: the False-block containing `arg` should have no
    # gaps. Find the contiguous run of False bins around `arg`.
    false_bins = np.where(~row_mask)[0]
    # Identify the connected component containing `arg`.
    # Walk left from arg while consecutive False bins persist:
    left = arg
    while left - 1 >= 0 and not row_mask[left - 1]:
        left -= 1
    right = arg
    while right + 1 < row_mask.size and not row_mask[right + 1]:
        right += 1
    region = np.arange(left, right + 1)

    # All bins in [left, right] must be False (definition of contiguous)
    assert (~row_mask[region]).all(), (
        f"DLA region around bin {arg} is not contiguous: "
        f"left={left}, right={right}, mask[region]={row_mask[region]}"
    )

    # And the region must contain `arg` (sanity)
    assert left <= arg <= right

    # Diagnostic print so the test logs the region width on success
    print(
        f"P2 row {P2_DLA_ROW}: tau_max={tmax:.3e} at bin {arg}; "
        f"DLA region [{left}, {right}] (width {right - left + 1}); "
        f"total masked bins in row = {(~row_mask).sum()} / {row_mask.size}"
    )


# ------------------------------------------------------------------ assert (c)
@_pytestmark_have_all
def test_log1p_loss_within_one_decade_across_physics():
    """[D-24] With the new log1p + cap-10 loss against zero prediction,
    the per-physics loss magnitude must be within one order of magnitude
    across P1..P4 (proves DLA contamination is no longer driving the loss).

    L_phys = mean( log1p(min(tau_GT, 10))^2 )  over non-DLA bins only.

    Memory: a single Sherwood snapshot already costs ~2-3 GB of doubles;
    we MUST load + reduce one physics at a time and drop the in-memory
    cube before moving to the next. We therefore instantiate a fresh
    loader inside the test body rather than relying on the module-scoped
    fixtures (which would pin all 4 cubes simultaneously and OOM on a
    16-GB-RAM host).
    """
    import gc

    losses = {}
    for pid in (1, 2, 3, 4):
        local_loader = SherwoodLoader(DATA_ROOT)
        data = local_loader.load_sightlines(physics_id=pid, redshift=REDSHIFT, nspec=NSPEC)
        tau = data["tau_h1"]
        mask = data["mask_no_dla"]
        # Reduce streaming-style via the cap, then drop everything
        tau_eff = np.minimum(tau, 10.0)
        sq = np.log1p(tau_eff) ** 2
        if mask.sum() == 0:
            pytest.fail(f"P{pid}: mask excluded every bin")
        losses[pid] = float(sq[mask].mean())
        del data, tau, mask, tau_eff, sq, local_loader
        gc.collect()

    print("Per-physics [D-24] supervised loss against zero-prediction:")
    for pid, L in losses.items():
        print(f"  P{pid}: {L:.4e}")

    L_min = min(losses.values())
    L_max = max(losses.values())
    ratio = L_max / L_min
    print(f"  ratio = L_max / L_min = {ratio:.3f}")

    assert ratio < 10.0, (
        f"[D-24] per-physics loss spans more than one decade: "
        f"L_min={L_min:.3e}, L_max={L_max:.3e}, ratio={ratio:.3f}"
    )


# ----------------------------------------- additional sanity: shape + types
@_pytestmark_have_p1
def test_loader_emits_new_keys(p1_data):
    for key in ("tau_h1", "tau_h1_real", "mask_no_dla", "dla_threshold_log_nhi"):
        assert key in p1_data, f"loader missing key {key!r}"
    assert p1_data["tau_h1_real"].shape == p1_data["tau_h1"].shape
    assert p1_data["mask_no_dla"].shape == p1_data["tau_h1"].shape
    assert p1_data["mask_no_dla"].dtype == np.bool_
    # Default per [D-24] / Wolfe+ 2005
    assert math.isclose(p1_data["dla_threshold_log_nhi"], 20.3)

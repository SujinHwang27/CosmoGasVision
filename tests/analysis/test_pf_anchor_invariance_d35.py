"""Anchor-invariance regression test for P_F(k_||) — [D-35] gate.

Spec
----
Under a uniform flux rescale F -> r * F, the **normalized** delta_F
= F / <F> - 1 is invariant (since both F and <F> scale by r), so
P_F(k_||) is identical to numerical floor.

The OLD mean-subtracted formula delta_F = F - <F> picks up an overall
r-scaling, so P_F -> r^2 * P_F. The empirical anchor-invariance demo
in ``scripts/eval_anchor_invariance_d34.py`` recorded 2.77 %–5.26 %
band drift across P1/P2/P4 under r != 1, well outside the 0.5 % gate.
This test guards against regression of the [D-35] fix in
``src/analysis/p_flux.py:73``.

Two assertions per rescale factor:

(A) FIX: ``compute_p_flux(F)`` and ``compute_p_flux(r*F)`` agree to
    1e-12 in every bin. This is the anchor-invariance gate.

(B) BUGGY DIRECTION: applying the OLD ``F - <F>`` formula in-line
    yields ``P_F_buggy(r*F) ~ r^2 * P_F_buggy(F)`` to 1e-10. This is
    a positive control documenting the bug we just fixed; the
    [D-35] LEDGER entry cites this as evidence.

Determinism: numpy seed 42; runtime <1 s on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.p_flux import compute_p_flux


# Build the test fixture once at import; both tests share it.
def _make_fixture():
    rng = np.random.default_rng(42)
    n_sl, n_bins = 64, 1024
    dv = 1.0  # km/s
    vel = np.arange(n_bins, dtype=np.float64) * dv
    # Choose F in [0.4, 0.5] so r * F stays in (0, 1) for all r in {0.5..2.0}.
    # 2.0 * 0.5 = 1.0 is the worst case but only touches the boundary, never
    # exceeds it; F is strictly < 0.5 thanks to the open interval below.
    F = 0.4 + 0.1 * rng.random((n_sl, n_bins))  # in [0.4, 0.5)
    return F, vel


def _compute_buggy_p_flux(F: np.ndarray, vel: np.ndarray, **kwargs):
    """In-line replication of the OLD mean-subtracted pipeline.

    Mirrors ``src.analysis.p_flux.compute_p_flux`` exactly except for
    the line under test (``delta_F = F - F.mean(...)`` instead of
    ``F / F.mean(...) - 1``). Used only as a positive control in
    assertion (B); never imported into production code.
    """
    k_min = kwargs.get("k_min", 10 ** -3)
    k_max = kwargs.get("k_max", 10 ** -1)
    n_kbins = kwargs.get("n_kbins", 20)

    F = np.asarray(F)
    vel = np.asarray(vel)
    n_sl, n_bins = F.shape
    dv = float(vel[1] - vel[0])

    # OLD formula — mean-subtraction (the bug).
    delta_F = F - F.mean(axis=1, keepdims=True)

    window = np.hanning(n_bins)
    delta_F = delta_F * window[None, :]
    F_k = np.fft.rfft(delta_F, axis=1)
    psd = (np.abs(F_k) ** 2) * (dv / np.sum(window ** 2))
    if n_bins % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0
    psd_mean = psd.mean(axis=0)

    f = np.fft.rfftfreq(n_bins, d=dv)
    k_axis = 2.0 * np.pi * f
    log_edges = np.linspace(np.log10(k_min), np.log10(k_max), n_kbins + 1)
    edges = 10 ** log_edges
    centers = 10 ** (0.5 * (log_edges[:-1] + log_edges[1:]))

    P_binned = np.full(n_kbins, np.nan, dtype=np.float64)
    valid = k_axis > 0
    k_pos = k_axis[valid]
    psd_pos = psd_mean[valid]
    idx = np.digitize(k_pos, edges) - 1
    in_range = (idx >= 0) & (idx < n_kbins)
    idx_ir = idx[in_range]
    psd_ir = psd_pos[in_range]
    if idx_ir.size > 0:
        sums = np.bincount(idx_ir, weights=psd_ir, minlength=n_kbins)
        cnts = np.bincount(idx_ir, minlength=n_kbins)
        nz = cnts > 0
        P_binned[nz] = sums[nz] / cnts[nz]
    return centers, P_binned


@pytest.mark.parametrize("r", [0.5, 0.8, 1.0, 1.2, 2.0])
def test_d35_normalized_p_flux_is_anchor_invariant(r):
    """[D-35] FIX: P_F(F) == P_F(r*F) under normalized delta_F."""
    F, vel = _make_fixture()
    F_rescaled = r * F
    # Sanity: rescaled values stay in (0, 1) for our chosen F range.
    assert F_rescaled.max() < 1.0 + 1e-12
    assert F_rescaled.min() > 0.0

    k_a, P_a = compute_p_flux(F, vel)
    k_b, P_b = compute_p_flux(F_rescaled, vel)

    assert np.array_equal(k_a, k_b), "k bin centers must be identical"
    finite = np.isfinite(P_a) & np.isfinite(P_b)
    assert finite.sum() > 0, "no finite bins to compare"
    np.testing.assert_allclose(
        P_a[finite],
        P_b[finite],
        atol=1e-12,
        rtol=1e-12,
        err_msg=(
            f"normalized P_F not anchor-invariant under r={r}: "
            f"max |dP| = {np.max(np.abs(P_a[finite] - P_b[finite])):.3e}"
        ),
    )


@pytest.mark.parametrize("r", [0.5, 0.8, 1.2, 2.0])
def test_d35_buggy_mean_subtraction_scales_as_r_squared(r):
    """POSITIVE CONTROL: OLD F - <F> formula gives P_F(r*F) ~ r^2 * P_F(F)."""
    F, vel = _make_fixture()
    F_rescaled = r * F

    _, P_buggy = _compute_buggy_p_flux(F, vel)
    _, P_buggy_rescaled = _compute_buggy_p_flux(F_rescaled, vel)

    finite = np.isfinite(P_buggy) & np.isfinite(P_buggy_rescaled)
    assert finite.sum() > 0
    np.testing.assert_allclose(
        P_buggy_rescaled[finite],
        (r ** 2) * P_buggy[finite],
        atol=1e-10,
        rtol=1e-10,
        err_msg=(
            f"buggy mean-subtracted P_F did not scale as r^2 under r={r}; "
            f"this would invalidate the [D-35] regression-direction control."
        ),
    )
    # And confirm the buggy formula is NOT invariant — i.e., r^2 scaling
    # actually moves the answer for r != 1.
    if r != 1.0:
        max_rel_drift = float(
            np.max(np.abs(P_buggy_rescaled[finite] - P_buggy[finite])
                   / np.abs(P_buggy[finite]))
        )
        assert max_rel_drift > 0.01, (
            f"buggy formula drift {max_rel_drift:.3e} too small to act as "
            f"a regression control at r={r}"
        )

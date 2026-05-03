"""Spec tests for ``src.analysis.flux_pdf.{compute_flux_pdf, ks_distance}``.

Per the dispatch spec:

1. Identical samples -> KS = 0 (within ``atol = 1e-10``).
2. Clearly different samples -> KS > 0.1.

Plus a quick smoke test on the histogram normalization of
``compute_flux_pdf``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.flux_pdf import compute_flux_pdf, ks_distance


def test_ks_identical_is_zero():
    rng = np.random.default_rng(0)
    F = rng.uniform(0.05, 0.99, size=10_000)
    ks = ks_distance(F, F)
    assert ks == pytest.approx(0.0, abs=1e-10), f"KS(F, F) = {ks} (expected 0)"


def test_ks_distinct_samples_above_threshold():
    rng = np.random.default_rng(1)
    # Pred concentrated near F=0.3, truth concentrated near F=0.8 ->
    # the empirical CDFs are well separated and KS must exceed 0.1.
    F_pred = rng.normal(loc=0.30, scale=0.05, size=10_000)
    F_truth = rng.normal(loc=0.80, scale=0.05, size=10_000)
    F_pred = np.clip(F_pred, 0.05, 0.99)
    F_truth = np.clip(F_truth, 0.05, 0.99)
    ks = ks_distance(F_pred, F_truth)
    assert ks > 0.1, f"KS for clearly-different samples = {ks} (expected > 0.1)"


def test_compute_flux_pdf_normalization():
    """Density PDF integrates to 1 over the supplied bin range."""
    rng = np.random.default_rng(2)
    F = rng.uniform(0.05, 0.99, size=50_000)
    bins = np.linspace(0.05, 0.99, 50)
    centers, pdf = compute_flux_pdf(F, bins)
    widths = np.diff(bins)
    integral = float(np.sum(pdf * widths))
    assert centers.shape == (49,)
    assert pdf.shape == (49,)
    assert integral == pytest.approx(1.0, rel=1e-6), (
        f"PDF integral = {integral} (expected 1.0)"
    )


def test_compute_flux_pdf_default_bins():
    """Default bins (per spec ``np.linspace(0.05, 0.99, 50)``) work."""
    rng = np.random.default_rng(3)
    F = rng.uniform(0.05, 0.99, size=10_000)
    centers, pdf = compute_flux_pdf(F)
    assert centers.shape == (49,)
    assert pdf.shape == (49,)
    assert np.all(pdf >= 0)

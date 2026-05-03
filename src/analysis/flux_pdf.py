"""Flux PDF p(F) and Kolmogorov-Smirnov distance — spec [D-13] gating module.

The PDF catches calibration drift in the absorber population (e.g.,
the network systematically over- or under-predicting saturated
absorbers). KS distance condenses the comparison to a single scalar
for the [D-13] gating criterion (KS < 0.05).

Two APIs are exposed:

- **Spec-compliant** (preferred):
  ``compute_flux_pdf(F, F_bins)`` and ``ks_distance(F_pred, F_truth, F_range)``.
  These take raw flux samples (F = exp(-tau)).

- **Legacy** (used by ``stage2b_report.py``):
  ``compute_F_PDF(tau_arr, F_bins)`` and ``ks_distance_pdf(pdf_a, pdf_b, F_bins)``.
  These take optical-depth arrays / pre-binned PDFs.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- spec API


def compute_flux_pdf(
    F: np.ndarray,
    F_bins: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-density PDF of the transmitted flux F.

    Parameters
    ----------
    F : ndarray
        Transmitted flux F = exp(-tau), any shape (will be flattened).
    F_bins : (n_bins+1,) ndarray, optional
        PDF bin EDGES on F. Default ``np.linspace(0.05, 0.99, 50)`` per spec.

    Returns
    -------
    F_bin_centers : (n_bins,) ndarray
        Bin centers.
    pdf_density : (n_bins,) ndarray
        Normalized density (integrates to 1 over the supplied bins).
    """
    if F_bins is None:
        F_bins = np.linspace(0.05, 0.99, 50)
    F = np.asarray(F).ravel()
    counts, edges = np.histogram(F, bins=F_bins, density=False)
    widths = np.diff(edges)
    total = counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    if total == 0:
        return centers, np.zeros_like(widths, dtype=float)
    pdf = counts / (total * widths)
    return centers, pdf


def ks_distance(
    F_pred: np.ndarray,
    F_truth: np.ndarray,
    F_range: tuple[float, float] = (0.05, 0.99),
) -> float:
    """Kolmogorov-Smirnov distance between two flux samples.

    Computes ``max |CDF_pred(F) - CDF_truth(F)|`` over the empirical
    distributions restricted to ``F_range``. Pure NumPy implementation
    using sorted-merge of the two empirical CDFs (no SciPy dependency).

    Parameters
    ----------
    F_pred, F_truth : ndarray
        Transmitted-flux samples F = exp(-tau). Any shape, will be flattened.
    F_range : (F_min, F_max)
        Restrict to this interval before computing the KS distance. The
        [D-13] threshold of 0.05 is defined on this restricted CDF to
        sidestep the saturated-absorber pile-up at F~0 and the
        continuum-noise tail at F~1.

    Returns
    -------
    float
        max |CDF_pred(F) - CDF_truth(F)| over ``F_range``. Returns 0 if
        either sample is empty after the range cut.
    """
    F_pred = np.asarray(F_pred).ravel()
    F_truth = np.asarray(F_truth).ravel()
    F_min, F_max = F_range
    a = F_pred[(F_pred >= F_min) & (F_pred <= F_max)]
    b = F_truth[(F_truth >= F_min) & (F_truth <= F_max)]
    if a.size == 0 or b.size == 0:
        return 0.0

    # Standard two-sample KS via sorted-merge of empirical CDFs.
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    all_vals = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


# ------------------------------------------------------------------ legacy API


def compute_F_PDF(
    tau_arr: np.ndarray,
    F_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy: histogram-density PDF on the transmitted flux F = exp(-tau).

    Differs from :func:`compute_flux_pdf` only in that the input is the
    optical-depth array (the conversion ``F = exp(-tau)`` is applied here).
    Retained because ``stage2b_report.py`` and the existing GRF tests call
    this signature.
    """
    F = np.exp(-tau_arr).ravel()
    counts, edges = np.histogram(F, bins=F_bins, density=False)
    widths = np.diff(edges)
    total = counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    if total == 0:
        return centers, np.zeros_like(widths, dtype=float)
    pdf = counts / (total * widths)
    return centers, pdf


def ks_distance_pdf(
    pdf_a: np.ndarray,
    pdf_b: np.ndarray,
    F_bins: np.ndarray,
) -> float:
    """Legacy: KS distance between two pre-binned flux PDFs.

    Used by ``stage2b_report.py`` which pre-bins the PDFs for plotting
    and reuses them for the gating scalar. New callers should prefer
    :func:`ks_distance` on raw samples.
    """
    if pdf_a.shape != pdf_b.shape:
        raise ValueError(
            f"PDF shapes differ: {pdf_a.shape} vs {pdf_b.shape}"
        )
    widths = np.diff(F_bins)
    cdf_a = np.cumsum(pdf_a * widths)
    cdf_b = np.cumsum(pdf_b * widths)
    if cdf_a[-1] > 0:
        cdf_a = cdf_a / cdf_a[-1]
    if cdf_b[-1] > 0:
        cdf_b = cdf_b / cdf_b[-1]
    return float(np.max(np.abs(cdf_a - cdf_b)))

"""Flux PDF p(F) and Kolmogorov-Smirnov distance between two PDFs.

The PDF catches calibration drift in the absorber population (e.g.,
the network systematically over- or under-predicting saturated
absorbers). KS distance condenses the comparison to a single scalar
for the [D-13] gating criterion (KS < 0.05).
"""

from __future__ import annotations

import numpy as np


def compute_F_PDF(
    tau_arr: np.ndarray,
    F_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-density PDF of the transmitted flux F = exp(-tau).

    Parameters
    ----------
    tau_arr : ndarray
        Optical depths; any shape is fine, will be flattened.
    F_bins : (n_bins+1,) ndarray
        PDF bin EDGES on F in [0, 1]. Typical: ``np.linspace(0, 1, 51)``.

    Returns
    -------
    centers : (n_bins,) ndarray
        Bin centers.
    pdf : (n_bins,) ndarray
        Normalized density: integrates to 1 over the supplied bin range.
    """
    F = np.exp(-tau_arr).ravel()
    counts, edges = np.histogram(F, bins=F_bins, density=False)
    widths = np.diff(edges)
    total = counts.sum()
    if total == 0:
        return 0.5 * (edges[:-1] + edges[1:]), np.zeros_like(widths, dtype=float)
    pdf = counts / (total * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, pdf


def ks_distance(
    pdf_a: np.ndarray,
    pdf_b: np.ndarray,
    F_bins: np.ndarray,
) -> float:
    """Kolmogorov-Smirnov distance between two binned PDFs.

    Parameters
    ----------
    pdf_a, pdf_b : (n_bins,) ndarray
        Density values from :func:`compute_F_PDF`.
    F_bins : (n_bins+1,) ndarray
        Same bin edges used to build the two PDFs.

    Returns
    -------
    float
        max |CDF_a(F) - CDF_b(F)| over the right edges.
    """
    if pdf_a.shape != pdf_b.shape:
        raise ValueError(
            f"PDF shapes differ: {pdf_a.shape} vs {pdf_b.shape}"
        )
    widths = np.diff(F_bins)
    cdf_a = np.cumsum(pdf_a * widths)
    cdf_b = np.cumsum(pdf_b * widths)
    # Re-normalize to [0, 1] in case the binned support is incomplete.
    if cdf_a[-1] > 0:
        cdf_a = cdf_a / cdf_a[-1]
    if cdf_b[-1] > 0:
        cdf_b = cdf_b / cdf_b[-1]
    return float(np.max(np.abs(cdf_a - cdf_b)))

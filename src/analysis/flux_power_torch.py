"""Differentiable 1D Lyα flux power spectrum P_F(k_||) — Torch mirror of
:mod:`src.analysis.p_flux`.

The NumPy implementation in ``p_flux.compute_p_flux`` is the evaluation-side
ground truth (Walther+ 2018 / Boera+ 2019 convention, Hann window,
$dv/\\sum w^2$ normalization, angular wavenumber $k=2\\pi f$). This module
reproduces the same pipeline through ``torch.fft.rfft`` so the resulting P_F
tensor stays in the autograd graph, which is the prerequisite for the
[D-39] saturation-aware P_F training loss term.

Conventions reproduced
----------------------
- per-sightline mean-normalized contrast $\\delta_F = F/\\langle F\\rangle - 1$
  (mean computed over the same velocity axis, with autograd) — matches the
  [D-35] anchor-invariant form in ``p_flux.py``.
- Hann apodization, one-sided PSD via $\\,|\\hat F_k|^2 \\cdot dv / \\sum w^2$
  with the positive-frequency $\\times 2$ correction for an even-N grid.
- angular wavenumber $k = 2\\pi f$ in s/km.

What this module deliberately does NOT reproduce
------------------------------------------------
- log-spaced k binning. The training loss only needs a band-integrated
  residual over the [D-13] inertial range $k_\\| \\in [10^{-2.5}, 10^{-1.5}]$
  s/km; binning would just introduce an extra non-differentiable
  ``digitize`` step. We compute a uniform mean over the FFT bins that fall
  inside the band instead, which is the correct differentiable analog of the
  binned mean used in evaluation.

Returned tensors live on the same device / dtype as the input ``F``.
"""

from __future__ import annotations

import torch


_K_MIN_INERTIAL = 10.0 ** -2.5  # s/km, [D-13] lower edge
_K_MAX_INERTIAL = 10.0 ** -1.5  # s/km, [D-13] upper edge


def compute_p_flux_torch(
    F: torch.Tensor,
    dv: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable per-sightline P_F(k_||) PSD.

    Parameters
    ----------
    F : (n_sightlines, n_bins) tensor
        Transmitted flux F = exp(-tau) on a uniform velocity grid.
        Must be a Torch tensor; autograd is preserved through the FFT.
    dv : float
        Uniform velocity spacing in km/s.

    Returns
    -------
    k_axis : (n_freq,) tensor
        Angular wavenumber grid $k_\\| = 2\\pi f$ in s/km, on F's device.
        DC bin (k=0) included for indexing parity; callers should mask it
        out for band integration.
    psd : (n_sightlines, n_freq) tensor
        One-sided PSD in s/km, per sightline. Autograd is live.
    """
    if F.dim() != 2:
        raise ValueError(f"F must be 2D (n_sightlines, n_bins); got {tuple(F.shape)}")
    n_bins = F.shape[1]
    if dv <= 0:
        raise ValueError("dv must be > 0")

    # [D-35] anchor-invariant contrast: delta_F = F/<F> - 1. Mean is taken
    # along the velocity axis, with autograd through F.
    F_mean = F.mean(dim=1, keepdim=True)
    # Guard against the structurally-impossible <F>=0 case (Lyα <F> >= 0.5
    # at z=0.3 per [D-11]). Adding a tiny epsilon keeps autograd well-defined
    # at the synthetic-dummy-data edge case.
    delta_F = F / F_mean.clamp(min=1e-8) - 1.0

    # Hann window (matches numpy.hanning, which uses the N-point symmetric
    # form). torch.hann_window default is the periodic form; set
    # periodic=False to align with numpy.hanning.
    window = torch.hann_window(
        n_bins, periodic=False, dtype=F.dtype, device=F.device
    )
    sum_w2 = (window * window).sum()
    delta_F_w = delta_F * window.unsqueeze(0)

    # rfft along velocity axis; autograd-supported.
    F_k = torch.fft.rfft(delta_F_w, dim=1)
    # |F_k|^2 with autograd via complex abs.
    psd = (F_k.real ** 2 + F_k.imag ** 2) * (dv / sum_w2)
    # One-sided correction: double positive-frequency bins (excluding DC and,
    # for even N, Nyquist). Out-of-place arithmetic to keep autograd clean.
    correction = torch.ones_like(psd)
    if n_bins % 2 == 0:
        correction[:, 1:-1] = 2.0
    else:
        correction[:, 1:] = 2.0
    psd = psd * correction

    # Angular wavenumber axis. rfftfreq returns ordinary frequency f; multiply
    # by 2pi for k = 2pi f, matching Walther+ 2018 / Boera+ 2019.
    freqs = torch.fft.rfftfreq(n_bins, d=dv).to(F.device)
    k_axis = 2.0 * torch.pi * freqs

    return k_axis, psd


def band_mean_inertial(
    psd: torch.Tensor,
    k_axis: torch.Tensor,
    k_min: float = _K_MIN_INERTIAL,
    k_max: float = _K_MAX_INERTIAL,
) -> torch.Tensor:
    """Average a per-sightline PSD over the [D-13] inertial band.

    Parameters
    ----------
    psd : (n_sightlines, n_freq) tensor
        PSD on the wavenumber axis.
    k_axis : (n_freq,) tensor
        Wavenumber grid in s/km.
    k_min, k_max : float
        Band edges in s/km. Defaults: [D-13] inertial range
        $[10^{-2.5}, 10^{-1.5}]$.

    Returns
    -------
    psd_band : (n_sightlines,) tensor
        Sightline-wise mean PSD over the band. Autograd preserved.
        Sightlines with no FFT bin in band raise — this is a sanity guard;
        the [D-13] inertial range is well inside the Sherwood grid's
        resolved range so this never triggers at run time.
    """
    band_mask = (k_axis >= k_min) & (k_axis <= k_max)
    n_in_band = int(band_mask.sum().item())
    if n_in_band == 0:
        raise ValueError(
            f"No FFT bins fall in inertial band [{k_min:.4g}, {k_max:.4g}] s/km. "
            f"k_axis spans [{k_axis.min().item():.4g}, {k_axis.max().item():.4g}]."
        )
    # Pure tensor reduction, autograd-friendly.
    return psd[:, band_mask].mean(dim=1)

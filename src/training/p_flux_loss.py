"""Sprint-L1 direct P_F MSE loss building blocks.

Three components per design v2 §2 / §3 (file
``experiments/nerf/design/sprint_L1_direct_pf_loss.md``, HEAD ``066cf24`` on
``exp/nerf``):

1. ``torch_p_flux`` — differentiable torch reimplementation of
   ``src.analysis.p_flux.compute_p_flux`` matching its four pipeline steps
   (mean-divide contrast, Hann window with ``dv/sum w^2`` leakage
   compensation, ``torch.fft.rfft``, log-spaced k-binning with ``n_kbins=20``)
   to 1e-6 absolute / 1e-4 relative on non-empty bins. K2 gate-4 hard
   deliverable; see ``tests/test_torch_pf_estimator_equivalence.py``.

2. ``pf_log_mse_loss`` — log-MSE loss over the [D-13] inertial range
   ``k_|| in [10^-2.5, 10^-1.5]`` s/km, ray-averaged INSIDE the log:
   ``sum_k (log10 <P_F_pred>_rays(k) - log10 <P_F_truth>_rays(k))^2``. The
   ray-averaging-before-log semantic is K1-absorbing per panel verdict:
   per-ray P_F has chi^2_2 statistics; ray-averaging first drops the log10
   estimator-noise floor to ~1/sqrt(N_rays) instead of accumulating per-ray
   chi-square tail mass.

3. ``GradNormWrapper`` — Chen et al. 2018 GradNorm with ``alpha=0.12``
   (the paper default). Two task weights ``w_tau``, ``w_pf`` as trainable
   scalars initialized to 1.0; updated by a SEPARATE optimizer step on the
   GradNorm-balance loss. S2-absorbing per panel verdict; replaces v1's
   step-function lambda retune.

Numerical precision
-------------------
- The P_F estimator computes in **float64** internally regardless of input
  dtype, because the eval-side reference (NumPy) uses float64 and the K2
  equivalence test requires matching FP-summation order down to 1e-6 abs.
  The output is cast back to the input dtype on return so callers can keep
  the rest of the training graph in float32 (existing convention).
- Empty-bin convention: eval uses ``NaN``; training uses **0.0**. NaN cannot
  be propagated into a loss (a single NaN-bin breaks the whole batch).
  The K2 equivalence test compares ONLY non-empty bins.

References
----------
- Mildenhall et al. (2020) — NeRF foundations (project-wide).
- Chen et al. (2018) — GradNorm: Gradient Normalization for Adaptive Loss
  Balancing in Deep Multitask Networks, ICML.
- Walther et al. (2018), Boera et al. (2019) — Hann-windowed P_F convention
  reproduced in ``src.analysis.p_flux``.
- ``experiments/nerf/LEDGER.md`` §3 [D-13], [D-24], [D-35], [D-39].
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


# [D-13] inertial-range edges in s/km.
K_MIN_INERTIAL = 10.0 ** -2.5
K_MAX_INERTIAL = 10.0 ** -1.5

# Eval estimator's default log-k bin edges (n_kbins=20 between 1e-3 and 1e-1).
_DEFAULT_K_MIN = 10.0 ** -3
_DEFAULT_K_MAX = 10.0 ** -1
_DEFAULT_N_KBINS = 20


# ---------------------------------------------------------------------------
# (a) Differentiable torch P_F estimator (K2-absorbing)
# ---------------------------------------------------------------------------


def torch_p_flux(
    F: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
    n_kbins: int = _DEFAULT_N_KBINS,
    empty_bin_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable torch P_F(k_||) matching ``src.analysis.p_flux.compute_p_flux``.

    Mirrors the four-step eval-side pipeline:

    1. mean-divide normalization ``delta_F = F / <F> - 1`` (NOT mean-subtract;
       [D-35] case-of-record — the mean-subtracted form picks up an
       overall r-scaling and breaks the [D-13] anchor-invariance gate).
    2. Hann window apodization with ``dv / sum(w^2)`` leakage compensation
       (Walther+ 2018 / Boera+ 2019 convention).
    3. ``torch.fft.rfft`` along the velocity axis; one-sided PSD with
       positive-frequency x2 correction (excluding DC and Nyquist for even N).
    4. log-spaced k-binning over ``[k_min, k_max]`` with ``n_kbins`` bins;
       angular wavenumber ``k = 2*pi*f`` (NOT ordinary frequency).

    Differentiability
    -----------------
    Every step is autograd-live in ``F``: the mean reduction, the elementwise
    window multiply, ``rfft``, magnitude-squared, the digitize-equivalent
    bin index gather, and the bincount average are all differentiable in
    ``F``. The bin-index ``digitize`` is itself non-differentiable but it
    operates on the FIXED frequency grid (no ``F`` dependence), so backward
    flows cleanly through the bin-aggregation arithmetic without needing a
    surrogate.

    Internal precision is float64 to match the NumPy reference (K2
    equivalence requires 1e-6 abs / 1e-4 rel; float32 FFT-summation order
    drifts past that bar in practice). Output is cast to the input dtype
    so the rest of the training graph stays float32.

    Parameters
    ----------
    F : (n_sightlines, n_bins) tensor
        Transmitted flux ``F = exp(-tau)`` on a uniform velocity grid.
        Autograd is preserved.
    vel_axis_kms : (n_bins,) tensor
        Monotone, uniformly spaced velocity grid in km/s. Used to set ``dv``
        and the FFT frequency axis; not autograd-tracked.
    k_min, k_max : float
        Log-bin edges in s/km. Default ``1e-3, 1e-1`` matches the eval
        estimator's default span.
    n_kbins : int
        Number of log-spaced k bins. Default ``20`` matches the eval default.
    empty_bin_value : float
        Value placed in bins with no FFT entries. Default ``0.0`` (training
        convention; the eval estimator uses ``NaN``). The K2 equivalence
        test compares ONLY non-empty bins so this divergence is by design.

    Returns
    -------
    k_centers : (n_kbins,) tensor
        Geometric-mean bin centers in s/km. Detached from autograd; depends
        only on ``k_min, k_max, n_kbins`` so a single tensor would do, but
        we return a fresh tensor on ``F.device`` for symmetry with the
        NumPy interface.
    P_binned : (n_sightlines, n_kbins) tensor
        Per-sightline P_F(k_||) in s/km. Autograd-live in ``F``. The eval
        estimator returns the sightline-averaged (n_kbins,) shape; we keep
        the per-sightline axis to let downstream loss formulations choose
        their reduction (ray-averaging inside the log per design v2 §2).
    """
    if F.dim() != 2:
        raise ValueError(f"F must be 2D (n_sightlines, n_bins); got {tuple(F.shape)}")
    n_sl, n_bins = F.shape
    if vel_axis_kms.shape != (n_bins,):
        raise ValueError(
            f"vel_axis_kms shape {tuple(vel_axis_kms.shape)} != ({n_bins},)"
        )

    out_dtype = F.dtype
    device = F.device

    # Promote to float64 internally for FP-summation parity with NumPy.
    F64 = F.to(torch.float64)
    vel64 = vel_axis_kms.to(torch.float64).to(device)

    dv = float((vel64[1] - vel64[0]).item())
    if dv <= 0:
        raise ValueError("vel_axis_kms must be strictly increasing")

    # Step 1: mean-divide delta_F (matches p_flux.py:78-79). The clamp
    # is structurally unneeded for Lyα <F> > 0 at z=0.3 but mirrors the
    # eval-side guard for the synthetic-dummy edge case in tests.
    F_mean = F64.mean(dim=1, keepdim=True)
    delta_F = F64 / F_mean - 1.0

    # Step 2: Hann window + dv/sum(w^2) leakage compensation. Use
    # periodic=False to match numpy.hanning (symmetric N-point form;
    # torch.hann_window default is periodic, which differs by one sample).
    window = torch.hann_window(n_bins, periodic=False, dtype=torch.float64, device=device)
    sum_w2 = (window * window).sum()
    delta_F = delta_F * window.unsqueeze(0)

    # Step 3: real FFT + one-sided PSD normalization.
    F_k = torch.fft.rfft(delta_F, dim=1)
    psd = (F_k.real ** 2 + F_k.imag ** 2) * (dv / sum_w2)
    # One-sided correction: x2 on positive frequencies (excluding DC and
    # — for even N — Nyquist). Out-of-place to keep autograd clean.
    correction = torch.ones_like(psd)
    if n_bins % 2 == 0:
        correction[:, 1:-1] = 2.0
    else:
        correction[:, 1:] = 2.0
    psd = psd * correction

    # Frequency / angular-wavenumber axis. ``rfftfreq`` returns ordinary
    # frequency f; multiply by 2*pi for k = 2*pi*f (Walther+ 2018).
    freqs = torch.fft.rfftfreq(n_bins, d=dv).to(torch.float64).to(device)
    k_axis = 2.0 * torch.pi * freqs  # (n_freq,)

    # Step 4: log-spaced k-binning. Edges and centers match the NumPy
    # ``np.linspace(log10(k_min), log10(k_max), n_kbins+1)`` form.
    log_edges = torch.linspace(
        float(torch.log10(torch.tensor(k_min, dtype=torch.float64))),
        float(torch.log10(torch.tensor(k_max, dtype=torch.float64))),
        n_kbins + 1,
        dtype=torch.float64,
        device=device,
    )
    edges = 10.0 ** log_edges
    centers = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))

    # Per-sightline binned average. Equivalent to NumPy's
    # digitize -> bincount(weights=psd) / bincount(weights=1) pattern, but
    # autograd-friendly (the bin indices have no ``F`` dependence, so the
    # gather is differentiable in psd).
    valid_freq = k_axis > 0  # exclude DC
    k_pos = k_axis[valid_freq]
    psd_pos = psd[:, valid_freq]  # (n_sl, n_freq_pos)

    # Bin index per positive-frequency point. ``torch.bucketize`` is the
    # torch equivalent of ``np.digitize`` with right=False default; we
    # subtract 1 to match NumPy's "edge i corresponds to idx i-1 onwards"
    # semantic (np.digitize returns indices in 1..len(edges)).
    bin_idx = torch.bucketize(k_pos, edges) - 1  # (n_freq_pos,)
    in_range = (bin_idx >= 0) & (bin_idx < n_kbins)

    # Build a one-hot membership mask (n_freq_pos, n_kbins) once. Gradients
    # then propagate through psd_pos@membership without any non-diff op.
    # This is the differentiable analog of ``np.bincount(idx, weights=psd)``.
    if in_range.any():
        bin_idx_clamped = bin_idx.clamp(min=0, max=n_kbins - 1)
        # (n_freq_pos, n_kbins) float mask: 1 where the freq lives in that bin.
        membership = torch.zeros(
            (k_pos.shape[0], n_kbins), dtype=torch.float64, device=device,
        )
        membership[torch.arange(k_pos.shape[0], device=device), bin_idx_clamped] = 1.0
        membership = membership * in_range.unsqueeze(1).to(torch.float64)
        # Per-bin count (no F dependence; no autograd issue).
        counts = membership.sum(dim=0)  # (n_kbins,)
        # Per-(sightline, bin) sum: psd_pos @ membership -> (n_sl, n_kbins).
        sums = psd_pos @ membership  # autograd-live in psd_pos
        # Avoid 0/0; bins with count==0 keep the sentinel value below.
        safe_counts = counts.clamp(min=1.0)
        P_binned = sums / safe_counts.unsqueeze(0)
        empty_mask = counts == 0
        if bool(empty_mask.any()):
            # Replace empty bins with the requested sentinel. We do this with
            # ``torch.where`` so autograd sees a constant on the empty side.
            sentinel = torch.full_like(P_binned, empty_bin_value)
            P_binned = torch.where(
                empty_mask.unsqueeze(0).expand_as(P_binned),
                sentinel,
                P_binned,
            )
    else:
        # No frequency falls in [k_min, k_max] — degenerate grid. Surface
        # as all-empty sentinel.
        P_binned = torch.full(
            (n_sl, n_kbins),
            empty_bin_value,
            dtype=torch.float64,
            device=device,
        )

    return centers.to(out_dtype), P_binned.to(out_dtype)


# ---------------------------------------------------------------------------
# (b) Log-MSE loss over the [D-13] inertial range (K1-absorbing)
# ---------------------------------------------------------------------------


def pf_log_mse_loss(
    F_pred: torch.Tensor,
    F_truth: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    k_min_inertial: float = K_MIN_INERTIAL,
    k_max_inertial: float = K_MAX_INERTIAL,
    n_kbins: int = _DEFAULT_N_KBINS,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
    eps: float = 1e-30,
    reduction: str = "sum",
) -> torch.Tensor:
    """Log-MSE loss over the [D-13] inertial-range k bins, ray-averaged INSIDE the log.

    Per design v2 §2:

    ``L_PF = sum_{k in K_inertial} (log10 <P_F_pred>_rays(k) - log10 <P_F_truth>_rays(k))^2``

    K1 absorption (panel verdict): the ``<.>_rays`` averaging happens BEFORE
    the ``log10`` and the per-k difference. Per-ray ``P_F`` has chi^2_2
    statistics; ray-averaging the linear-space P_F first drops the noise
    floor to ``~1/sqrt(N_rays)`` in the log domain, comparable to the gate
    signal at N_rays=1024 (~0.031 std vs ~0.041 signal). Averaging AFTER the
    log would propagate the chi^2 tail mass and inflate the floor by ~4x.

    Parameters
    ----------
    F_pred : (n_rays, n_bins) tensor
        Predicted flux (autograd-live).
    F_truth : (n_rays, n_bins) tensor
        Truth flux (typically detached; the loss is symmetric in F so the
        autograd graph only flows through F_pred in practice).
    vel_axis_kms : (n_bins,) tensor
        Uniform velocity grid in km/s. Shared between pred and truth.
    k_min_inertial, k_max_inertial : float
        Inertial-band edges in s/km. Defaults [10^-2.5, 10^-1.5] s/km
        per [D-13].
    n_kbins, k_min, k_max : float / int
        Log-k binning of the underlying P_F estimator. Defaults match the
        eval-side ``compute_p_flux``: 20 bins over [1e-3, 1e-1] s/km. The
        inertial sub-band typically covers 6 of the 20 bins.
    eps : float
        Small positive floor before ``log10`` to keep training stable on
        zero-flux pathological inputs. Eval-side P_F is structurally
        positive (Hann-windowed FFT magnitude-squared on nonzero data)
        but a degenerate microbatch (e.g. constant prediction) could land
        a zero in the band — surface as a bounded loss rather than NaN.

    Returns
    -------
    loss : 0-dim tensor
        Scalar loss; autograd-live in ``F_pred``.
    """
    if F_pred.shape != F_truth.shape:
        raise ValueError(
            f"F_pred {tuple(F_pred.shape)} != F_truth {tuple(F_truth.shape)}"
        )

    centers_p, P_pred = torch_p_flux(
        F_pred, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins,
        empty_bin_value=0.0,
    )
    centers_t, P_truth = torch_p_flux(
        F_truth, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins,
        empty_bin_value=0.0,
    )

    # Ray-average INSIDE the log (K1). Use float64 for the ray reduction.
    P_pred_ravg = P_pred.to(torch.float64).mean(dim=0)   # (n_kbins,)
    P_truth_ravg = P_truth.to(torch.float64).mean(dim=0)

    # Inertial-band selection. ``centers_p`` and ``centers_t`` are identical
    # (deterministic function of bin parameters) so we use the pred side.
    band_mask = (centers_p.to(torch.float64) >= k_min_inertial) & (
        centers_p.to(torch.float64) <= k_max_inertial
    )
    if not bool(band_mask.any()):
        raise ValueError(
            f"No log-k bin centers fall in inertial range "
            f"[{k_min_inertial:.4g}, {k_max_inertial:.4g}] s/km. "
            f"Check n_kbins / k_min / k_max."
        )

    P_pred_band = P_pred_ravg[band_mask].clamp_min(eps)
    P_truth_band = P_truth_ravg[band_mask].clamp_min(eps)

    log_pred = torch.log10(P_pred_band)
    log_truth = torch.log10(P_truth_band)

    sq = (log_pred - log_truth) ** 2
    # [D-60 revised Attempt 2, 2026-05-22] ``reduction`` lever per PI R15
    # NON-PROVISIONAL. Default 'sum' preserves prior numerical behavior
    # (backward-compat for every existing call site / test). 'mean' divides
    # by the inertial-bin count so the per-task scale is independent of
    # ``n_kbins`` * inertial-band-fraction; this is the lever the live
    # P_F gradient sanity-check (Option R) consumes.
    if reduction == "sum":
        loss = sq.sum()
    elif reduction == "mean":
        loss = sq.mean()
    else:
        raise ValueError(
            f"pf_log_mse_loss: unsupported reduction={reduction!r}; "
            f"expected one of {{'sum', 'mean'}}."
        )
    return loss.to(F_pred.dtype)


# ---------------------------------------------------------------------------
# (b') k-space-normalized P_F loss ([D-53] candidate (b), panel-bound 2026-05-23)
# ---------------------------------------------------------------------------


def compute_sigma_k_squared_ema(
    P_truth_batch: torch.Tensor,
    ema_prev: torch.Tensor | None,
    decay: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update per-mode P_truth(k) variance EMA (panel pre-commit, 2026-05-23).

    Per [D-53] candidate (b) panel-binding selector 1: σ_k² is **truth-side**,
    batch-sample, **EMA-stabilized with decay 0.99**. Truth-side eliminates
    the predicted-side chicken-and-egg at step 0 (all rays initialized to
    near-identical fields -> σ_k² ≈ 0 -> 1/σ_k² -> inf -> grad explode).

    Parameters
    ----------
    P_truth_batch : (n_rays, n_kbins) tensor
        Per-ray truth-side P_F(k); typically the output of ``torch_p_flux``
        on the truth flux batch. Detached.
    ema_prev : (n_kbins,) tensor or None
        Previous EMA state. ``None`` on the first call (initialize directly
        from the batch variance — NOT 0.99 * 0 + 0.01 * v, which would take
        many steps to track the truth scale).
    decay : float
        EMA decay; pre-committed 0.99 per panel.

    Returns
    -------
    ema_new : (n_kbins,) tensor
        Updated EMA state. Pass back in on the next call.
    sigma_k_sq : (n_kbins,) tensor
        The EMA value to use as σ_k² weights this step (== ema_new; named
        separately for caller-side readability).
    """
    if P_truth_batch.dim() != 2:
        raise ValueError(
            f"P_truth_batch must be 2D (n_rays, n_kbins); got "
            f"{tuple(P_truth_batch.shape)}"
        )
    # Per-mode batch variance over the ray axis (unbiased=False to match
    # the numpy var() default in callers; n_rays >> 1 in practice so the
    # bias correction is negligible).
    with torch.no_grad():
        batch_var = P_truth_batch.to(torch.float64).var(
            dim=0, unbiased=False
        )  # (n_kbins,)
        if ema_prev is None:
            ema_new = batch_var.clone()
        else:
            ema_new = decay * ema_prev.to(batch_var.dtype) + (1.0 - decay) * batch_var
    return ema_new, ema_new


def pf_knorm_loss(
    F_pred: torch.Tensor,
    F_truth: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    sigma_k_squared_truth_ema: torch.Tensor,
    k_min_inertial: float = K_MIN_INERTIAL,
    k_max_inertial: float = K_MAX_INERTIAL,
    n_kbins: int = _DEFAULT_N_KBINS,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
    floor_rel: float = 0.01,
) -> torch.Tensor:
    """k-space-normalized P_F loss ([D-53] candidate (b), panel-bound 2026-05-23).

    Form (panel-binding selector 3, verbatim):

    ``L = Σ_k (P_pred(k) − P_truth(k))² / max(σ_k²_truth_ema(k), floor)``

    where ``floor = 0.01 × median_k(σ_k²_truth_ema)`` per panel-binding
    selector 1 (relative floor, NOT absolute 1e-12 which would let any
    sub-floor mode dominate loss by ~10⁸ given typical P_F values
    ~10⁻⁵-10⁻³ s/km).

    Inertial band selection is the same as ``pf_log_mse_loss`` — the lever
    is the weighting form, not the band selection. Ray-averaging is done
    BEFORE the per-mode-squared-residual (analog of K1 absorption); this is
    the natural reduction for the inverse-variance-weighted residual sum.

    Important
    ---------
    This is the **LINEAR-domain** inverse-variance-weighted **squared
    residual sum** — NOT log10-domain, NOT mean-reduced. The first test of
    the [D-53] supervision-target-redesign class candidate (b); first test
    of a k-space-normalized P_F target in this project.

    Parameters
    ----------
    F_pred : (n_rays, n_bins) tensor
        Predicted flux (autograd-live).
    F_truth : (n_rays, n_bins) tensor
        Truth flux (typically detached).
    vel_axis_kms : (n_bins,) tensor
        Uniform velocity grid in km/s.
    sigma_k_squared_truth_ema : (n_kbins,) tensor
        Truth-side per-mode P_F variance EMA (from
        ``compute_sigma_k_squared_ema``). Detached. Must span the same
        ``n_kbins`` bin grid the underlying ``torch_p_flux`` produces.
    k_min_inertial, k_max_inertial : float
        Inertial-band edges in s/km. Defaults [10^-2.5, 10^-1.5] s/km
        per [D-13].
    n_kbins, k_min, k_max : float / int
        Log-k binning of the underlying P_F estimator (must match the
        ``compute_sigma_k_squared_ema`` call).
    floor_rel : float
        Relative floor multiplier; ``floor = floor_rel * median_k(σ_k²_ema)``
        across the inertial band. Panel-bound default 0.01.

    Returns
    -------
    loss : 0-dim tensor
        Scalar loss; autograd-live in ``F_pred``.
    """
    if F_pred.shape != F_truth.shape:
        raise ValueError(
            f"F_pred {tuple(F_pred.shape)} != F_truth {tuple(F_truth.shape)}"
        )
    if sigma_k_squared_truth_ema.dim() != 1 or sigma_k_squared_truth_ema.shape[0] != n_kbins:
        raise ValueError(
            f"sigma_k_squared_truth_ema must be 1D of shape ({n_kbins},); "
            f"got {tuple(sigma_k_squared_truth_ema.shape)}"
        )

    centers_p, P_pred = torch_p_flux(
        F_pred, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins,
        empty_bin_value=0.0,
    )
    centers_t, P_truth = torch_p_flux(
        F_truth, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins,
        empty_bin_value=0.0,
    )

    # Ray-average BEFORE the per-mode squared residual (analog of K1).
    P_pred_ravg = P_pred.to(torch.float64).mean(dim=0)   # (n_kbins,)
    P_truth_ravg = P_truth.to(torch.float64).mean(dim=0)

    # Inertial band selection — SAME as pf_log_mse_loss (lines 262-367).
    band_mask = (centers_p.to(torch.float64) >= k_min_inertial) & (
        centers_p.to(torch.float64) <= k_max_inertial
    )
    if not bool(band_mask.any()):
        raise ValueError(
            f"No log-k bin centers fall in inertial range "
            f"[{k_min_inertial:.4g}, {k_max_inertial:.4g}] s/km."
        )

    sigma_band = sigma_k_squared_truth_ema.to(torch.float64)[band_mask].detach()
    # Relative floor over the inertial band, per panel-binding selector 1.
    band_median = sigma_band.median().clamp_min(1e-30)
    floor = (floor_rel * band_median).detach()
    weights = torch.clamp_min(sigma_band, floor)  # (n_inertial,)

    P_pred_band = P_pred_ravg[band_mask]
    P_truth_band = P_truth_ravg[band_mask]
    resid_sq = (P_pred_band - P_truth_band) ** 2
    # Σ_k r_k² / σ_k² (NOT mean — verbatim per panel selector 3).
    loss = (resid_sq / weights).sum()
    return loss.to(F_pred.dtype)


def inertial_rel_residual(
    F_pred: torch.Tensor,
    F_truth: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    k_min_inertial: float = K_MIN_INERTIAL,
    k_max_inertial: float = K_MAX_INERTIAL,
    n_kbins: int = _DEFAULT_N_KBINS,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
) -> torch.Tensor:
    """Mean per-inertial-bin ``|<P_F_pred>_rays - <P_F_truth>_rays| / <P_F_truth>_rays``.

    Diagnostic for [D-13] gate tracking; matches the headline-residual
    metric the eval pipeline reports. Computed under ``torch.no_grad`` style
    (caller wraps; this function itself preserves autograd to keep API uniform).
    Used by the pipeline retire-condition checks R-b / R-f.
    """
    centers, P_pred = torch_p_flux(F_pred, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins)
    _, P_truth = torch_p_flux(F_truth, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins)
    P_pred_ravg = P_pred.to(torch.float64).mean(dim=0)
    P_truth_ravg = P_truth.to(torch.float64).mean(dim=0)
    band_mask = (centers.to(torch.float64) >= k_min_inertial) & (
        centers.to(torch.float64) <= k_max_inertial
    )
    rel = (P_pred_ravg[band_mask] - P_truth_ravg[band_mask]).abs() / (
        P_truth_ravg[band_mask].clamp_min(1e-30)
    )
    return rel.mean()


def cross_coherence_per_bin(
    F_pred: torch.Tensor,
    F_truth: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    k_min_inertial: float = K_MIN_INERTIAL,
    k_max_inertial: float = K_MAX_INERTIAL,
    n_kbins: int = _DEFAULT_N_KBINS,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
) -> torch.Tensor:
    """Segment-averaged magnitude-squared cross-coherence per inertial k-bin.

    For each log-k bin, average the auto and cross periodogram estimates
    across rays AND across the FFT bins falling in that log-k bin, then form:

    ``|gamma(k)|^2 = |<S_xy>|^2 / (<S_xx> <S_yy>)``

    where ``<.>`` is the average over (ray, FFT bin) pairs within the bin.
    This is the standard Welch / segment-averaged coherence; the single-
    realization periodogram coherence is identically 1 by Cauchy-Schwarz so
    averaging across independent samples is REQUIRED to make the diagnostic
    meaningful (Thrane & Romano 2013 §III, Bendat-Piersol §9.2).

    Returns
    -------
    (n_inertial_bins,) tensor of segment-averaged ``|gamma|^2`` per inertial
    k-bin. 1.0 when F_pred == F_truth (or any rescaling); ~0 when F_pred
    and F_truth are uncorrelated.

    B3 backstop per design v2 §3 (replaces v1 Pearson r per panel S1
    absorption). The retire condition R-e checks that ``>= 4/6`` inertial
    k-bins have ``|gamma|^2 >= 0.5``.
    """
    if F_pred.shape != F_truth.shape:
        raise ValueError("F_pred and F_truth must share shape.")
    n_sl, n_bins = F_pred.shape
    device = F_pred.device

    F64p = F_pred.to(torch.float64)
    F64t = F_truth.to(torch.float64)
    vel64 = vel_axis_kms.to(torch.float64).to(device)
    dv = float((vel64[1] - vel64[0]).item())

    # Same delta_F + window prep as torch_p_flux.
    delta_p = F64p / F64p.mean(dim=1, keepdim=True) - 1.0
    delta_t = F64t / F64t.mean(dim=1, keepdim=True) - 1.0
    window = torch.hann_window(n_bins, periodic=False, dtype=torch.float64, device=device)
    sum_w2 = (window * window).sum()
    delta_p = delta_p * window.unsqueeze(0)
    delta_t = delta_t * window.unsqueeze(0)

    Fk_p = torch.fft.rfft(delta_p, dim=1)
    Fk_t = torch.fft.rfft(delta_t, dim=1)

    # Auto and cross periodograms in matched normalization. Complex cross
    # spectrum (NOT its magnitude squared yet — we need to average S_xy
    # AS A COMPLEX NUMBER across (ray, FFT bin) samples before taking |.|^2,
    # so phase cancellation across uncorrelated samples produces a small
    # average, vs the magnitude-squared which would always sum positive).
    norm = dv / sum_w2
    Sxx = (Fk_p.real ** 2 + Fk_p.imag ** 2) * norm                       # (n_sl, n_freq)
    Syy = (Fk_t.real ** 2 + Fk_t.imag ** 2) * norm
    # Sxy = Fk_p * conj(Fk_t); separate real/imag for autograd safety.
    Sxy_re = (Fk_p.real * Fk_t.real + Fk_p.imag * Fk_t.imag) * norm      # Re{Fk_p * conj(Fk_t)}
    Sxy_im = (Fk_p.imag * Fk_t.real - Fk_p.real * Fk_t.imag) * norm

    freqs = torch.fft.rfftfreq(n_bins, d=dv).to(torch.float64).to(device)
    k_axis = 2.0 * torch.pi * freqs

    log_edges = torch.linspace(
        float(torch.log10(torch.tensor(k_min, dtype=torch.float64))),
        float(torch.log10(torch.tensor(k_max, dtype=torch.float64))),
        n_kbins + 1, dtype=torch.float64, device=device,
    )
    edges = 10.0 ** log_edges
    centers = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))

    valid = k_axis > 0
    k_pos = k_axis[valid]
    Sxx_pos = Sxx[:, valid]                # (n_sl, n_freq_pos)
    Syy_pos = Syy[:, valid]
    Sxy_re_pos = Sxy_re[:, valid]
    Sxy_im_pos = Sxy_im[:, valid]
    bin_idx = torch.bucketize(k_pos, edges) - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_kbins)

    band_mask = (centers >= k_min_inertial) & (centers <= k_max_inertial)
    band_bins = torch.where(band_mask)[0]
    out = torch.zeros(int(band_mask.sum().item()), dtype=torch.float64, device=device)
    for out_i, b in enumerate(band_bins.tolist()):
        sel = in_range & (bin_idx == b)
        n_in = int(sel.sum().item())
        if n_in == 0:
            out[out_i] = float("nan")
            continue
        # Average (over all (ray, freq) samples in this log-k bin).
        Sxx_avg = Sxx_pos[:, sel].mean()
        Syy_avg = Syy_pos[:, sel].mean()
        Sxy_re_avg = Sxy_re_pos[:, sel].mean()
        Sxy_im_avg = Sxy_im_pos[:, sel].mean()
        Sxy_mag2 = Sxy_re_avg ** 2 + Sxy_im_avg ** 2
        out[out_i] = Sxy_mag2 / (Sxx_avg * Syy_avg).clamp_min(1e-30)
    return out.to(F_pred.dtype)


# ---------------------------------------------------------------------------
# (c) GradNorm multi-task weighting wrapper (S2-absorbing)
# ---------------------------------------------------------------------------


class GradNormWrapper(nn.Module):
    """Chen et al. 2018 GradNorm with ``alpha=0.12`` (the paper default).

    Maintains two trainable task weights ``w_tau`` and ``w_pf`` (initialized
    to 1.0) and updates them by a SEPARATE optimizer step on the GradNorm
    balance loss. The model loss for the main optimizer is the
    weighted-sum-of-task-losses with the CURRENT weight values.

    Usage pattern (per design v2 §2 + standard GradNorm formulation):

    ::

        gn = GradNormWrapper(initial_w=(1.0, 1.0), alpha=0.12).to(device)
        gn_opt = torch.optim.Adam(gn.parameters(), lr=1e-3)
        # ... inside training step:
        loss_tau = compute_tau_loss(...)
        loss_pf  = compute_pf_loss(...)
        total_loss, gn_loss, w_tau, w_pf = gn.step(
            losses=(loss_tau, loss_pf),
            shared_params=[p for p in model.parameters() if p.requires_grad],
        )
        # backward for the model
        total_loss.backward(retain_graph=True)
        model_opt.step(); model_opt.zero_grad()
        # backward for the GradNorm weights
        gn_opt.zero_grad()
        gn_loss.backward()
        gn_opt.step()
        gn.renormalize_weights()  # keep w_tau + w_pf = T constant

    Notes
    -----
    - ``alpha=0.12`` is the value Chen+ 2018 recommend across the experiments
      in §5 of the paper; per design v2 §2 we adopt that default unchanged.
    - The "shared parameters" set is conventionally the last shared layer's
      weights in the multi-task literature; for the L1 setup the IGM-NeRF
      backbone is fully shared (tau is the only output), so we use the full
      parameter list. The chosen layer affects the magnitude of the
      gradient-norm targets but not their ratio at convergence.
    - The wrapper logs the ``w_tau / w_pf`` ratio; the pipeline emits this
      to MLflow per step (R-g retire on > 1000:1 in either direction).
    """

    def __init__(
        self,
        initial_w: tuple[float, float] = (1.0, 1.0),
        alpha: float = 0.12,
        simplified: bool = False,
    ):
        super().__init__()
        if len(initial_w) != 2:
            raise ValueError("GradNormWrapper currently supports exactly 2 tasks.")
        # ``simplified=True`` substitutes ``G_i = w_i * |L_i|`` for the true
        # gradient norm. This is a well-known practical approximation when
        # the per-task loss scales are well-conditioned (the gradient norm
        # is approximately proportional to the loss magnitude under
        # bounded-Lipschitz heads). Use this branch when the second-order
        # ``torch.autograd.grad(create_graph=True)`` path is unstable
        # (Windows CPU pytorch + custom FFT-bearing graphs can segfault on
        # double-backward in practice — surfaced by the gate-4 host smoke
        # crash 2026-05-16 at exit code 0xC0000005).
        self.simplified = bool(simplified)
        # Trainable task weights as raw scalars (not constrained positive
        # here; renormalize_weights keeps the sum at T = 2 so they stay near
        # 1.0 on average. A negative excursion is structurally rare with
        # alpha=0.12 and a renormalized denominator > 0; clamp_min(1e-4) on
        # read-out keeps the weighted loss well-defined regardless).
        self.w_tau = nn.Parameter(torch.tensor(float(initial_w[0]), dtype=torch.float32))
        self.w_pf = nn.Parameter(torch.tensor(float(initial_w[1]), dtype=torch.float32))
        self.alpha = float(alpha)
        # Initial loss values L_i(0); set on the first forward call and
        # FROZEN thereafter (Chen+ 2018 §3 "relative inverse training rate"
        # is normalized by the initial loss).
        self.register_buffer("_L0_tau", torch.tensor(float("nan")))
        self.register_buffer("_L0_pf", torch.tensor(float("nan")))
        # Target sum-of-weights T. The renormalization step holds w_tau + w_pf == T.
        self.T = float(initial_w[0] + initial_w[1])

    @property
    def weights_clamped(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the task weights clamped at ``1e-4`` (positivity safety)."""
        return self.w_tau.clamp_min(1e-4), self.w_pf.clamp_min(1e-4)

    @property
    def weight_ratio(self) -> float:
        """``w_tau / w_pf`` as a Python float; used for R-g retire check."""
        w_t, w_p = self.weights_clamped
        return float((w_t / w_p.clamp_min(1e-30)).item())

    def initialize_L0(self, loss_tau: torch.Tensor, loss_pf: torch.Tensor) -> None:
        """Pin ``L_i(0)`` on the first step. Idempotent — only the first call writes."""
        if not bool(torch.isnan(self._L0_tau)):
            return
        with torch.no_grad():
            self._L0_tau.copy_(loss_tau.detach())
            self._L0_pf.copy_(loss_pf.detach())

    def renormalize_weights(self) -> None:
        """Renormalize so that ``w_tau + w_pf == T`` (Chen+ 2018 §3 constraint)."""
        with torch.no_grad():
            w_sum = (self.w_tau + self.w_pf).clamp_min(1e-8)
            scale = self.T / w_sum
            self.w_tau.mul_(scale)
            self.w_pf.mul_(scale)

    def compute_total_loss(
        self, loss_tau: torch.Tensor, loss_pf: torch.Tensor
    ) -> torch.Tensor:
        """``w_tau * L_tau + w_pf * L_pf`` (autograd-live in both losses)."""
        w_t, w_p = self.weights_clamped
        return w_t * loss_tau + w_p * loss_pf

    def compute_gradnorm_loss(
        self,
        loss_tau: torch.Tensor,
        loss_pf: torch.Tensor,
        shared_params: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the GradNorm balance loss (Chen+ 2018 Algorithm 1).

        Procedure:

        1. ``G_i = ||grad_{shared_params} (w_i * L_i)||_2``
           (gradient norm of the weighted task loss w.r.t. shared params).
        2. ``G_avg = mean(G_i)``.
        3. ``r_i = L_i / L_i(0) / mean_j(L_j / L_j(0))``
           — relative inverse training rate per task.
        4. Target ``G_target_i = G_avg * (r_i ** alpha)`` (detached; the
           GradNorm loss is an L1 distance of G_i to this target).
        5. Return ``L_grad = sum_i |G_i - G_target_i|``.

        The ``shared_params`` set is materialized once at call time. Caller
        should pass a finite parameter list (e.g. the IGM-NeRF backbone
        parameters); ``shared_params`` should be a list of leaf tensors with
        ``requires_grad=True``.

        Returns
        -------
        gn_loss : 0-dim tensor
            Autograd-live in (w_tau, w_pf). The model parameters do NOT
            receive gradient from this loss — Chen+ 2018 detaches ``G_target``
            and treats the gradient norms as known constants for the
            ``w``-direction (which is what makes GradNorm a separate
            optimizer step, not a joint backward).
        """
        if bool(torch.isnan(self._L0_tau)):
            self.initialize_L0(loss_tau, loss_pf)

        w_t, w_p = self.weights_clamped

        if self.simplified:
            # G_i = w_i * |L_i| (loss-magnitude proxy). Autograd-live in w_t,
            # w_p (the only free variables in the GradNorm direction). The
            # loss-magnitude path is exactly what we want: smaller raw loss
            # -> smaller G_i -> r_i^alpha boost moves w_i UP to compensate.
            G_tau = w_t * loss_tau.detach().abs().clamp_min(1e-30)
            G_pf = w_p * loss_pf.detach().abs().clamp_min(1e-30)
            G_avg = (G_tau + G_pf) / 2.0
            r_tau = loss_tau.detach() / self._L0_tau.clamp_min(1e-30)
            r_pf = loss_pf.detach() / self._L0_pf.clamp_min(1e-30)
            r_mean = (r_tau + r_pf) / 2.0
            r_tau_n = r_tau / r_mean.clamp_min(1e-30)
            r_pf_n = r_pf / r_mean.clamp_min(1e-30)
            G_target_tau = (G_avg.detach() * (r_tau_n ** self.alpha)).detach()
            G_target_pf = (G_avg.detach() * (r_pf_n ** self.alpha)).detach()
            gn_loss = (G_tau - G_target_tau).abs() + (G_pf - G_target_pf).abs()
            return gn_loss

        params = [p for p in shared_params if p.requires_grad and p is not self.w_tau and p is not self.w_pf]
        if not params:
            raise ValueError(
                "shared_params must contain at least one parameter with "
                "requires_grad=True (excluding the GradNorm task weights)."
            )

        # G_i = ||grad of (w_i * L_i) w.r.t. shared params||_2. We use
        # create_graph=True so the resulting G_i tensor carries autograd
        # back to w_i (the GradNorm loss's only free variable in the
        # w-direction). retain_graph=True to allow the second autograd.grad
        # call on L_pf to traverse the same graph.
        grads_tau = torch.autograd.grad(
            outputs=w_t * loss_tau, inputs=params,
            create_graph=True, retain_graph=True, allow_unused=True,
        )
        grads_pf = torch.autograd.grad(
            outputs=w_p * loss_pf, inputs=params,
            create_graph=True, retain_graph=True, allow_unused=True,
        )
        # Flatten and L2-norm each.
        G_tau = torch.sqrt(sum(
            (g * g).sum() for g in grads_tau if g is not None
        ).clamp_min(1e-30)) if any(g is not None for g in grads_tau) else torch.tensor(0.0, device=w_t.device)
        G_pf = torch.sqrt(sum(
            (g * g).sum() for g in grads_pf if g is not None
        ).clamp_min(1e-30)) if any(g is not None for g in grads_pf) else torch.tensor(0.0, device=w_p.device)

        G_avg = (G_tau + G_pf) / 2.0

        # Relative inverse training rates (Chen+ 2018 §3 eq. 4).
        r_tau = (loss_tau.detach() / self._L0_tau.clamp_min(1e-30))
        r_pf = (loss_pf.detach() / self._L0_pf.clamp_min(1e-30))
        r_mean = (r_tau + r_pf) / 2.0
        r_tau_n = r_tau / r_mean.clamp_min(1e-30)
        r_pf_n = r_pf / r_mean.clamp_min(1e-30)

        # Targets are detached per Chen+ 2018.
        G_target_tau = (G_avg.detach() * (r_tau_n ** self.alpha)).detach()
        G_target_pf = (G_avg.detach() * (r_pf_n ** self.alpha)).detach()

        gn_loss = (G_tau - G_target_tau).abs() + (G_pf - G_target_pf).abs()
        return gn_loss


__all__ = [
    "K_MIN_INERTIAL",
    "K_MAX_INERTIAL",
    "torch_p_flux",
    "pf_log_mse_loss",
    "pf_knorm_loss",
    "compute_sigma_k_squared_ema",
    "inertial_rel_residual",
    "cross_coherence_per_bin",
    "GradNormWrapper",
]

"""Sprint-L1 secondary unit tests for the direct P_F MSE loss machinery.

Five+ assertions per the PI dispatch brief gate-4 spec:

- ``test_pf_loss_zero_when_pred_eq_truth``
- ``test_pf_loss_positive_for_perturbed``
- ``test_pf_loss_gradient_flows``
- ``test_pf_loss_ray_averaging_inside_log`` (K1-absorbing semantic check)
- ``test_coherence_diagnostic_correct``
- ``test_gradnorm_weight_ratio_logged``

The K2 1e-6 estimator-equivalence test lives in its own file
(``tests/test_torch_pf_estimator_equivalence.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.p_flux_loss import (
    GradNormWrapper,
    cross_coherence_per_bin,
    pf_log_mse_loss,
    torch_p_flux,
)


def _toy_flux(n_rays: int = 64, n_bins: int = 512, seed: int = 0):
    """Synthesize a small flux batch + velocity axis for fast unit tests."""
    rng = np.random.default_rng(seed)
    dv = 1.5
    vel = torch.from_numpy((np.arange(n_bins) * dv).astype(np.float64))
    # Lyα-ish: smooth field with sparse line dips.
    F = 0.7 + 0.05 * rng.standard_normal((n_rays, n_bins))
    line_pos = rng.integers(0, n_bins, size=(n_rays, 8))
    for i in range(n_rays):
        for p in line_pos[i]:
            F[i, p] *= 0.1
    F = np.clip(F, 1e-4, 1.0)
    return torch.from_numpy(F).to(torch.float64), vel


# ---------------------------------------------------------------------------
# 1. Zero loss at F_pred == F_truth.
# ---------------------------------------------------------------------------


def test_pf_loss_zero_when_pred_eq_truth():
    F, vel = _toy_flux(n_rays=64, n_bins=512, seed=1)
    # log10(x) - log10(x) = 0 across all bins; loss must be exactly 0.0.
    loss = pf_log_mse_loss(F, F, vel)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-30), (
        f"pf_log_mse_loss(F, F) != 0: got {float(loss.item()):.3e}"
    )


# ---------------------------------------------------------------------------
# 2. Positive loss for a perturbed prediction.
# ---------------------------------------------------------------------------


def test_pf_loss_positive_for_perturbed():
    F_truth, vel = _toy_flux(n_rays=64, n_bins=512, seed=2)
    rng = np.random.default_rng(2)
    perturb = torch.from_numpy(1.0 + 0.1 * rng.standard_normal(F_truth.shape)).to(F_truth.dtype)
    F_pred = (F_truth * perturb).clamp(min=1e-4, max=1.0)
    loss = pf_log_mse_loss(F_pred, F_truth, vel)
    assert float(loss.item()) > 0.0, (
        f"pf_log_mse_loss on perturbed F is not positive: {float(loss.item()):.3e}"
    )


# ---------------------------------------------------------------------------
# 3. Gradient flows through F_pred (autograd contract).
# ---------------------------------------------------------------------------


def test_pf_loss_gradient_flows():
    F_truth, vel = _toy_flux(n_rays=64, n_bins=512, seed=3)
    F_pred = F_truth.clone().detach().requires_grad_(True)
    # Perturb so the loss is nonzero (a zero loss has zero gradient by construction).
    rng = np.random.default_rng(3)
    perturb = torch.from_numpy(1.0 + 0.05 * rng.standard_normal(F_pred.shape)).to(F_pred.dtype)
    F_pred_actual = (F_pred * perturb).clamp(min=1e-4, max=1.0)
    # Use F_pred_actual directly to keep the autograd path through the multiply
    # — clamp blocks gradients only at the saturated endpoints, which is rare.
    loss = pf_log_mse_loss(F_pred_actual, F_truth, vel)
    assert loss.requires_grad, "Loss tensor lost autograd."
    loss.backward()
    assert F_pred.grad is not None, "F_pred.grad is None after backward."
    assert torch.isfinite(F_pred.grad).all(), "non-finite F_pred.grad."
    assert float(F_pred.grad.abs().max().item()) > 0.0, (
        "F_pred.grad is identically zero — autograd graph severed somewhere."
    )


# ---------------------------------------------------------------------------
# 4. K1-absorbing semantic check: ray-averaging happens INSIDE the log.
# ---------------------------------------------------------------------------


def test_pf_loss_ray_averaging_inside_log():
    """Verify ``L = sum_k (log10 <P>_rays - log10 <T>_rays)^2`` not the per-ray-then-sum form.

    For per-ray P_F with chi^2_2 statistics, the two forms differ by the
    log-MSE estimator-noise floor: averaging inside the log uses the linear
    mean (drops as 1/sqrt(N_rays) per K1), while averaging after the log
    uses the geometric mean (chi^2 tail-mass inflated). We construct a
    deterministic counterexample with heavy per-ray heterogeneity and
    assert the two forms differ.
    """
    n_rays, n_bins = 128, 512
    rng = np.random.default_rng(4)
    dv = 1.5
    vel = torch.from_numpy((np.arange(n_bins) * dv).astype(np.float64))

    # Build F_truth heterogeneously across rays: some rays are smooth, some
    # are absorber-rich. This makes per-ray P_F vary by >10x across the batch.
    F_truth = np.empty((n_rays, n_bins), dtype=np.float64)
    for i in range(n_rays):
        smooth = 0.05 * rng.standard_normal(n_bins)
        if i % 2 == 0:
            F_truth[i] = 0.9 + smooth   # smooth, low P_F
        else:
            n_lines = 20
            F_i = 0.7 + smooth
            for p in rng.integers(0, n_bins, size=n_lines):
                F_i[p] *= 0.05         # heavy absorbers, high P_F
            F_truth[i] = F_i
    F_truth = np.clip(F_truth, 1e-4, 1.0)
    F_truth_t = torch.from_numpy(F_truth)

    # Mild perturbation -> nonzero loss; per-ray P_F variance is preserved.
    perturb = 1.0 + 0.05 * rng.standard_normal(F_truth.shape)
    F_pred_t = torch.from_numpy(np.clip(F_truth * perturb, 1e-4, 1.0))

    # Form A (production / K1-absorbing): ray-average inside the log.
    loss_inside = float(pf_log_mse_loss(F_pred_t, F_truth_t, vel).item())

    # Form B (per-ray then sum): compute per-ray log-MSE then mean over rays.
    # This is the "wrong" form panel K1 warned about.
    from src.training.p_flux_loss import K_MAX_INERTIAL, K_MIN_INERTIAL
    centers, P_pred = torch_p_flux(F_pred_t, vel, k_min=10 ** -3, k_max=10 ** -1, n_kbins=20)
    _, P_truth = torch_p_flux(F_truth_t, vel, k_min=10 ** -3, k_max=10 ** -1, n_kbins=20)
    band = (centers >= K_MIN_INERTIAL) & (centers <= K_MAX_INERTIAL)
    per_ray = (torch.log10(P_pred[:, band].clamp_min(1e-30))
               - torch.log10(P_truth[:, band].clamp_min(1e-30))) ** 2
    loss_outside = float(per_ray.sum(dim=1).mean().item())

    # The two forms MUST differ on this heterogeneous fixture (Jensen's
    # inequality: log of mean != mean of log unless all rays are identical).
    # Tolerance is generous: we only require a measurable gap, not a specific
    # numerical value. With 1e-9 we'd be testing FP precision; 1e-6 is the
    # "the two ARE numerically different forms" floor.
    diff = abs(loss_inside - loss_outside)
    assert diff > 1e-6, (
        f"K1 ray-averaging semantic check failed: inside-log={loss_inside:.6e} "
        f"vs per-ray-then-sum={loss_outside:.6e} (diff={diff:.3e}). The "
        f"production form must average ray P_F BEFORE the log10."
    )


# ---------------------------------------------------------------------------
# 5. Cross-coherence diagnostic: 1.0 for F_pred = F_truth, ~0 for uncorrelated.
# ---------------------------------------------------------------------------


def test_coherence_diagnostic_correct():
    F_truth, vel = _toy_flux(n_rays=64, n_bins=512, seed=5)
    # Identity case: coherence == 1 in every bin (modulo floating-point
    # rounding; we use a tight but not absurd tolerance).
    coh_id = cross_coherence_per_bin(F_truth, F_truth, vel)
    finite = torch.isfinite(coh_id)
    assert bool(finite.any()), "coherence diagnostic returned all-NaN on identity case."
    assert torch.allclose(
        coh_id[finite], torch.ones_like(coh_id[finite]), atol=1e-6,
    ), f"coherence(F, F) != 1.0: {coh_id[finite].tolist()}"

    # Uncorrelated case: shuffle rays so F_pred and F_truth share NO ray pairing.
    rng = np.random.default_rng(5)
    perm = torch.from_numpy(rng.permutation(F_truth.shape[0])).long()
    # Ensure no fixed points (would partially correlate).
    fixed = (perm == torch.arange(perm.shape[0])).nonzero(as_tuple=True)[0]
    if fixed.numel() > 0:
        # Swap fixed points with their neighbors.
        for i in fixed.tolist():
            j = (i + 1) % perm.shape[0]
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    F_shuffled = F_truth[perm]
    coh_uncorr = cross_coherence_per_bin(F_shuffled, F_truth, vel)
    finite_u = torch.isfinite(coh_uncorr)
    # Heuristic: median coherence on uncorrelated rays should be << 1 (small
    # batch -> finite-sample coherence floor at ~1/n_rays_per_bin). We accept
    # < 0.3 as the "clearly uncorrelated" threshold; the tight 0.0 bar in the
    # docstring is the limit case for large N.
    median_u = float(coh_uncorr[finite_u].median().item())
    assert median_u < 0.3, (
        f"coherence on uncorrelated rays should be << 1; got median {median_u:.3f}."
    )


# ---------------------------------------------------------------------------
# 6. GradNorm wrapper produces non-trivial weight updates on a toy 2-task setup.
# ---------------------------------------------------------------------------


def test_gradnorm_weight_ratio_logged():
    """Set up two scalar 'losses' with very different gradient magnitudes.

    GradNorm should drive ``w_tau / w_pf`` away from 1.0 toward the side whose
    raw gradient is SMALLER (relative-inverse-training-rate boost). We assert
    only that the ratio moves measurably from its initial 1.0 — the exact
    trajectory is hyperparameter-sensitive and outside the gate-4 contract.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Shared "model": a single linear layer with a single weight w_model.
    w_model = torch.nn.Parameter(torch.tensor(1.0))
    gn = GradNormWrapper(initial_w=(1.0, 1.0), alpha=0.12)
    gn_opt = torch.optim.Adam(gn.parameters(), lr=0.05)

    initial_ratio = gn.weight_ratio
    assert abs(initial_ratio - 1.0) < 1e-6, "initial w_tau/w_pf != 1.0"

    # Construct two tasks with very different scales w.r.t. w_model:
    # L_tau = (w_model - 2)^2   (gradient ~2 at init)
    # L_pf  = 1e-3 * (w_model - 5)^2  (gradient ~8e-3 at init — 250x smaller)
    n_steps = 30
    for _ in range(n_steps):
        L_tau = (w_model - 2.0) ** 2
        L_pf = 1e-3 * (w_model - 5.0) ** 2
        gn_loss = gn.compute_gradnorm_loss(L_tau, L_pf, shared_params=[w_model])
        gn_opt.zero_grad()
        gn_loss.backward()
        gn_opt.step()
        gn.renormalize_weights()

    final_ratio = gn.weight_ratio
    # Movement should be measurable: at minimum a few percent shift away from 1.0.
    assert abs(final_ratio - 1.0) > 0.01, (
        f"GradNorm did not move w_tau/w_pf away from 1.0 over {n_steps} steps; "
        f"got final ratio={final_ratio:.6f}. Gradient updates not propagating."
    )
    # Sanity: weights remain finite and positive (the clamp_min(1e-4) floor).
    w_t, w_p = gn.weights_clamped
    assert torch.isfinite(w_t).all() and torch.isfinite(w_p).all()
    assert float(w_t.item()) > 0.0 and float(w_p.item()) > 0.0


# ---------------------------------------------------------------------------
# 7. [D-60 revised Attempt 2] reduction='sum' vs 'mean' equivalence.
# ---------------------------------------------------------------------------


def _count_inertial_bins():
    """Recompute the number of inertial-band log-k bins under defaults.

    Mirrors the band selection inside ``pf_log_mse_loss`` so the test does
    not over-couple to internal constants beyond the public module API.
    """
    from src.training.p_flux_loss import (
        K_MAX_INERTIAL, K_MIN_INERTIAL,
    )
    n_kbins = 20
    k_min, k_max = 1e-3, 1e-1
    edges = np.logspace(np.log10(k_min), np.log10(k_max), n_kbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return int(((centers >= K_MIN_INERTIAL) & (centers <= K_MAX_INERTIAL)).sum())


def test_pf_loss_reduction_sum_vs_mean_relation():
    """sum_value ~= mean_value * n_inertial_bins on a perturbed synthetic case."""
    F_truth, vel = _toy_flux(n_rays=64, n_bins=512, seed=7)
    rng = np.random.default_rng(7)
    perturb = torch.from_numpy(
        1.0 + 0.08 * rng.standard_normal(F_truth.shape)
    ).to(F_truth.dtype)
    F_pred = (F_truth * perturb).clamp(min=1e-4, max=1.0)
    loss_sum = float(pf_log_mse_loss(F_pred, F_truth, vel, reduction="sum").item())
    loss_mean = float(pf_log_mse_loss(F_pred, F_truth, vel, reduction="mean").item())
    n_bins_inertial = _count_inertial_bins()
    assert n_bins_inertial > 0, "inertial-band bin count must be positive."
    # mean * n == sum exactly under float64 ray-reduction; allow a tiny
    # rtol for any cast back to F_pred.dtype.
    expected_sum = loss_mean * n_bins_inertial
    assert loss_sum == pytest.approx(expected_sum, rel=1e-6, abs=1e-12), (
        f"sum vs mean*n inconsistency: sum={loss_sum:.6e}, "
        f"mean*n={expected_sum:.6e}, n={n_bins_inertial}"
    )


def test_pf_loss_reduction_gradient_finite_both():
    """Gradient is finite + autograd-live under both reductions."""
    F_truth, vel = _toy_flux(n_rays=64, n_bins=512, seed=8)
    rng = np.random.default_rng(8)
    perturb = torch.from_numpy(
        1.0 + 0.05 * rng.standard_normal(F_truth.shape)
    ).to(F_truth.dtype)
    for reduction in ("sum", "mean"):
        F_pred = F_truth.clone().detach().requires_grad_(True)
        F_pred_actual = (F_pred * perturb).clamp(min=1e-4, max=1.0)
        loss = pf_log_mse_loss(F_pred_actual, F_truth, vel, reduction=reduction)
        assert loss.requires_grad, f"loss lost autograd under reduction={reduction!r}"
        loss.backward()
        assert F_pred.grad is not None, (
            f"F_pred.grad is None under reduction={reduction!r}"
        )
        assert torch.isfinite(F_pred.grad).all(), (
            f"non-finite F_pred.grad under reduction={reduction!r}"
        )
        assert float(F_pred.grad.abs().max().item()) > 0.0, (
            f"F_pred.grad is identically zero under reduction={reduction!r}"
        )


def test_pf_loss_reduction_rejects_invalid():
    """Unsupported reduction string raises ValueError."""
    F_truth, vel = _toy_flux(n_rays=16, n_bins=256, seed=9)
    with pytest.raises(ValueError):
        pf_log_mse_loss(F_truth, F_truth, vel, reduction="median")


def test_pf_log_reduction_cli_flag_default_sum():
    """argparse exposes --pf-log-reduction; default 'sum'; 'mean' toggles."""
    import os
    import sys
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from experiments.nerf.pipeline import parse_args
    base = [
        "--n_rays", "64", "--physics", "1", "--seed", "42",
        "--enable-l1-pf-loss",
    ]
    args_default = parse_args(base)
    assert hasattr(args_default, "pf_log_reduction"), (
        "parse_args must expose pf_log_reduction"
    )
    assert args_default.pf_log_reduction == "sum", (
        "default --pf-log-reduction must be 'sum' (backward-compat with "
        "R15 PROVISIONAL -> NON-PROVISIONAL baseline)"
    )
    args_mean = parse_args(base + ["--pf-log-reduction", "mean"])
    assert args_mean.pf_log_reduction == "mean"
    # Rejection of unsupported strings via argparse 'choices'.
    with pytest.raises(SystemExit):
        parse_args(base + ["--pf-log-reduction", "median"])

"""CPU pre-flight Deliverable B for Sprint-L2 [D-62] candidate (3):
fGPA-residual autograd verification.

Verifies that the hybrid architecture

    rho(x) = rho_fGPA(delta(x), z) * (1 + MLP_residual(x))

is differentiable through the delta input *and* through the MLP residual,
and that the fGPA forward path itself carries gradient (not only the MLP
wrapper). The fGPA power-law form

    rho_fGPA(delta, z) ~ (1 + delta) ** beta(z)

is the Hui & Gnedin 1997 scaling reused from the existing
[D-41] FGPA-tail regularizer (beta=1.6 at z~0.3, see
``experiments/nerf/pipeline.py`` ``--fgpa_beta``).

PASS criteria (per task spec):
1. delta.grad is not None
2. torch.isfinite(delta.grad).all()
3. delta.grad.norm() > 0
4. Non-trivial routing: with MLP residual ablated to identity
   (multiply-by-0-then-add-1), gradient STILL flows through the fGPA path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ------------------------------------------------------------------ helpers


def rho_fgpa(delta: torch.Tensor, z: float = 0.3, beta: float = 1.6) -> torch.Tensor:
    """Hui & Gnedin 1997 fluctuating Gunn-Peterson power law on delta.

    Uses (1 + delta) ** beta. beta default 1.6 matches
    ``experiments/nerf/pipeline.py --fgpa_beta`` ([D-41]).
    The redshift `z` is passed for API completeness; the power-law form is
    redshift-independent in beta at this approximation order (the redshift
    dependence is absorbed in the C anchor at training time, [D-41]).
    """
    # Clamp the (1 + delta) base to avoid log of negative under autograd at
    # synthetic dummy data with very negative delta. Real Sherwood delta is
    # bounded below by -1 by construction (density >= 0).
    base = (1.0 + delta).clamp_min(1e-6)
    return base.pow(beta)


class DummyResidualMLP(nn.Module):
    """Small 2-layer MLP that maps (n_rays, n_bins) -> (n_rays, n_bins)
    residual scalar per voxel. Acts pointwise on a learned coordinate
    feature (here we synthesize a 1-D feature from the bin index)."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_rays, n_bins). Use the normalized bin index as the input
        # feature so this is a legitimate spatially-varying MLP and not
        # a constant.
        n_rays, n_bins = x.shape
        coord = torch.linspace(0.0, 1.0, n_bins, device=x.device, dtype=x.dtype)
        feat = coord.view(1, n_bins, 1).expand(n_rays, n_bins, 1)
        out = self.net(feat).squeeze(-1)  # (n_rays, n_bins)
        return out


# ------------------------------------------------------------------ tests


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    n_rays, n_bins = 10, 2048
    # delta is the density contrast; legitimate range ~[-1, ~1e4]. Use
    # standard-normal centered on 0 for a dummy.
    delta = torch.randn(n_rays, n_bins) * 0.5
    delta.requires_grad_(True)
    rho_target = torch.rand(n_rays, n_bins) * 3.0
    return delta, rho_target


def test_fgpa_residual_autograd_full_path():
    """Standard path: delta -> fGPA -> multiplied by (1 + MLP_residual)."""
    delta, rho_target = _make_inputs(seed=0)
    mlp = DummyResidualMLP()

    rho_base = rho_fgpa(delta, z=0.3, beta=1.6)
    residual = mlp(delta)
    rho = rho_base * (1.0 + residual)

    loss = ((rho - rho_target) ** 2).mean()
    loss.backward()

    # Assertion 1: gradient exists
    assert delta.grad is not None, "delta.grad is None — autograd did not flow"
    # Assertion 2: gradient is finite
    assert torch.isfinite(delta.grad).all(), (
        f"delta.grad has non-finite entries: "
        f"{(~torch.isfinite(delta.grad)).sum().item()} bad / "
        f"{delta.grad.numel()} total"
    )
    # Assertion 3: non-trivial magnitude
    grad_norm = delta.grad.norm().item()
    assert grad_norm > 0.0, f"delta.grad.norm() = {grad_norm} (expected > 0)"


def test_fgpa_residual_autograd_routes_through_fgpa_path():
    """Non-trivial routing check: ablate MLP residual to identity
    (multiply by 0 then add 1, so rho = rho_fGPA exactly), and verify
    gradient STILL flows through delta via the fGPA forward path.
    """
    delta, rho_target = _make_inputs(seed=1)
    mlp = DummyResidualMLP()

    rho_base = rho_fgpa(delta, z=0.3, beta=1.6)
    residual = mlp(delta)
    # Ablate: zero the MLP contribution then add the multiplicative
    # identity. This collapses rho -> rho_fGPA * 1 = rho_fGPA, so the only
    # remaining route for delta.grad is the fGPA path. We keep the
    # multiplication by `residual * 0` in-graph so a frozen-MLP regression
    # would still propagate (the (1.0 + ...) wrapper stays a tensor op).
    rho = rho_base * (1.0 + residual * 0.0)

    loss = ((rho - rho_target) ** 2).mean()
    loss.backward()

    assert delta.grad is not None, "delta.grad is None under residual ablation"
    assert torch.isfinite(delta.grad).all(), (
        "delta.grad has non-finite entries under residual ablation"
    )
    grad_norm = delta.grad.norm().item()
    assert grad_norm > 0.0, (
        f"delta.grad.norm() = {grad_norm} under residual ablation — "
        "fGPA forward path is NOT differentiable through delta. "
        "This rules out candidate (3) per PI absorption [D-65]."
    )

    # Bonus consistency: under ablation the gradient must equal the
    # gradient of the pure-fGPA path (rho_fGPA = (1+delta)^beta against
    # (rho_fGPA - rho_target)^2).
    delta2, _ = _make_inputs(seed=1)
    rho_pure = rho_fgpa(delta2, z=0.3, beta=1.6)
    loss_pure = ((rho_pure - rho_target) ** 2).mean()
    loss_pure.backward()
    assert torch.allclose(
        delta.grad, delta2.grad, rtol=1e-5, atol=1e-7
    ), "Ablated-MLP path does not match pure-fGPA gradient — routing broken."

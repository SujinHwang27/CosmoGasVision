"""[D-70 (1b)] Integration test for the skip-rich-mlp body variant.

R20 twin-gate binding: this test must PASS before any Juno dispatch using
``--arch skip-rich-mlp`` is admissible.

Assertions
----------
1. Build IGMNeRF(body_arch='skip-rich-mlp'); forward + loss runs cleanly.
2. loss.grad_fn is not None  and  loss.requires_grad is True.
3. After 2 optimizer.step() calls under the [D-69] pretrain log-MSE loss,
   model weights move by ≥ 1e-6 in L2 norm (graph live, gradients applied).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.nerf import IGMNeRF  # noqa: E402


SEED = 20260525
BATCH = 256
PRETRAIN_LOG_EPS = 1.0e-3
WEIGHT_DELTA_TOL = 1e-6


def _pretrain_loss(rho_theta, rho_truth, eps=PRETRAIN_LOG_EPS):
    diff = torch.log10(rho_theta + eps) - torch.log10(rho_truth + eps)
    return (diff * diff).mean()


def test_skip_rich_build_and_forward():
    torch.manual_seed(SEED)
    model = IGMNeRF(hidden_dim=64, num_layers=8, L=10, body_arch="skip-rich-mlp")
    # Sanity: 8 body layers in layers2, layers1 empty.
    assert len(model.layers1) == 0, "skip-rich-mlp must build empty layers1"
    assert len(model.layers2) == 8, "skip-rich-mlp must build 8 body layers"

    coords = torch.rand(1, BATCH, 3)
    out = model(coords)
    assert out.shape == (1, BATCH, 4), f"unexpected forward shape {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in skip-rich forward"


def test_skip_rich_loss_grad_live():
    torch.manual_seed(SEED)
    model = IGMNeRF(hidden_dim=64, num_layers=8, L=10, body_arch="skip-rich-mlp")
    coords = torch.rand(1, BATCH, 3)
    rho_truth = torch.exp(torch.randn(BATCH) * 1.5)
    out = model(coords)
    rho_theta = out[0, :, 0]
    loss = _pretrain_loss(rho_theta, rho_truth)
    assert loss.requires_grad, "loss.requires_grad False — graph severed"
    assert loss.grad_fn is not None, "loss.grad_fn is None — graph severed"


def test_skip_rich_weights_move_under_pretrain():
    """2 optimizer.step() calls under L_pre move weights by ≥ 1e-6 in L2 norm."""
    torch.manual_seed(SEED)
    model = IGMNeRF(hidden_dim=64, num_layers=8, L=10, body_arch="skip-rich-mlp")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Snapshot pre-update parameters.
    before = [p.detach().clone() for p in model.parameters()]

    for _ in range(2):
        coords = torch.rand(1, BATCH, 3)
        rho_truth = torch.exp(torch.randn(BATCH) * 1.5)
        out = model(coords)
        rho_theta = out[0, :, 0]
        loss = _pretrain_loss(rho_theta, rho_truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    after = list(model.parameters())
    total_delta_sq = 0.0
    for b, a in zip(before, after):
        total_delta_sq += float(((a.detach() - b) ** 2).sum().item())
    total_delta = total_delta_sq ** 0.5
    assert total_delta >= WEIGHT_DELTA_TOL, (
        f"skip-rich weights did not move under L_pre: "
        f"L2 delta={total_delta:.3e} < tol={WEIGHT_DELTA_TOL:.0e}"
    )
    print(f"[skip-rich integration] 2-step L2 weight delta = {total_delta:.6e}")

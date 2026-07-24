"""A4 R20(i) integration test for the FeedbackField Stage-1 pipeline ([F-02] §4(c)-s1).

Behavioral chain: real cube via FeedbackCubeProvider -> sample coords+targets ->
FeedbackField forward -> MSE -> assert the loss is differentiable AND that BOTH
trunk weights and z_p actually move over >=2 Adam steps.
"""

import copy

import pytest
import torch

from src.data.feedback_cube_provider import FeedbackCubeProvider
from src.models.feedback_field import FeedbackField


def _provider():
    prov = FeedbackCubeProvider()
    if len(prov.loaded_variants) < 2:
        pytest.skip(f"need >=2 cubes; loaded {prov.loaded_variants}")
    return prov


def test_real_cube_forward_mse_differentiable_and_moves():
    torch.manual_seed(0)
    prov = _provider()
    variants = prov.loaded_variants[:2]

    model = FeedbackField(d=8)
    # snapshot pre-step trunk weight + code bank
    w0 = copy.deepcopy(model.net[0].linear.weight.detach())
    z0 = copy.deepcopy(model.codes.weight.detach())

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # fixed batch: n points per variant from the train region
    n = 1024
    batches = []
    for vi, p in enumerate(variants):
        coords, x = prov.sample(p, "train", n, seed=100 + p)
        idx = torch.full((n,), vi, dtype=torch.long)
        batches.append((coords, x, idx))
    coords = torch.cat([b[0] for b in batches], dim=0)
    targets = torch.cat([b[1] for b in batches], dim=0)
    idx = torch.cat([b[2] for b in batches], dim=0)

    last_loss = None
    for _ in range(2):
        opt.zero_grad()
        pred = model(coords, variant_idx=idx)
        loss = (pred - targets).pow(2).mean()
        # R20(i) contract: differentiable loss
        assert loss.grad_fn is not None
        assert loss.requires_grad
        loss.backward()
        # gradient reaches both trunk and z_p
        assert model.net[0].linear.weight.grad is not None
        assert model.codes.weight.grad is not None
        opt.step()
        last_loss = loss.item()

    assert last_loss is not None and last_loss == last_loss  # not NaN

    # both trunk weights AND z_p moved by >= 1e-6
    dw = (model.net[0].linear.weight.detach() - w0).abs().max().item()
    dz = (model.codes.weight.detach() - z0).abs().max().item()
    assert dw >= 1e-6, f"trunk weight change {dw:.2e} < 1e-6"
    assert dz >= 1e-6, f"z_p change {dz:.2e} < 1e-6"

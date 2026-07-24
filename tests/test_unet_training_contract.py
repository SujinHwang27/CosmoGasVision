"""[U-06] Stage-2 A3 — s1 R20(i) behavioral integration test.

Spec of record: ``experiments/unet-inversion/design/u06_stage2_spec.md``
S(b) s1: real ``UNetPairDataset`` -> rasterize -> forward -> MSE; asserts
``loss.grad_fn is not None and loss.requires_grad``; asserts max-abs weight
change >= 1e-6 over >= 2 optimizer steps. No HPC dispatch without this test
existing and green. CPU, small synthetic source (exercises the REAL dataset /
rasterizer / model classes; no sim data dependency in CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.truth_crop_provider import TruthCropProvider
from src.data.unet_crop_sampler import geometry_from_sightlines
from src.data.unet_pair_dataset import PhysicsSource, UNetPairDataset
from src.models.unet3d import UNet3D

BOX = 60000.0


def _make_source(seed=7, n_rays=300, nbins=2048):
    """Random-but-real PhysicsSource on the 192 lattice (mirrors the
    Stage-1 suite's fixture construction, tests/test_unet_stage1_pairs.py)."""
    rng = np.random.default_rng(seed)
    pitch = BOX / 192.0
    ia = rng.integers(1, 4, size=n_rays).astype(np.int32)
    xyz = (rng.integers(0, 192, size=(n_rays, 3)) + 0.5) * pitch
    sl = {
        "header": {"box_kpc_h": BOX},
        "iaxis": ia,
        "xaxis": xyz[:, 0],
        "yaxis": xyz[:, 1],
        "zaxis": xyz[:, 2],
        "pos_axis": (np.arange(nbins) + 0.5) * BOX / nbins,
    }
    geom = geometry_from_sightlines(sl)
    cube = rng.uniform(0.05, 3.0, (192, 192, 192))
    provider = TruthCropProvider(cube, region="train", crop_size=64, seed=seed)
    delta_f = rng.uniform(0.0, 0.2, (n_rays, nbins)).astype(np.float32)
    return PhysicsSource(1, provider, geom, delta_f)


def test_training_contract_grad_flow_and_weight_movement():
    torch.manual_seed(0)
    ds = UNetPairDataset([_make_source()], length=2, seed=42, augment=True)
    x = torch.stack([ds[i][0] for i in range(2)])
    y = torch.stack([ds[i][1] for i in range(2)])
    assert x.shape == (2, 2, 64, 64, 64) and y.shape == (2, 1, 64, 64, 64)
    assert x.dtype == torch.float32 and y.dtype == torch.float32

    model = UNet3D()
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    losses = []
    for _ in range(2):                       # >= 2 optimizer steps
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        # R20(i) gradient-flow contract
        assert loss.grad_fn is not None
        assert loss.requires_grad
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

    max_move = max(
        float((p.detach() - w0[n]).abs().max().item())
        for n, p in model.named_parameters()
    )
    assert max_move >= 1e-6, f"weights did not move: max |dw| = {max_move}"


def test_every_parameter_receives_gradient():
    torch.manual_seed(0)
    ds = UNetPairDataset([_make_source(seed=11)], length=1, seed=1,
                         augment=False)
    x, y = ds[0]
    model = UNet3D()
    loss = torch.nn.functional.mse_loss(model(x[None]), y[None])
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()
               or float(p.grad.abs().sum()) == 0.0]
    assert not missing, f"params without finite nonzero grad: {missing[:5]}"

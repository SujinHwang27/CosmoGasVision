"""Sprint-4 [D-51] unit tests for the 3D ResNet truth-baseline + trivial
baselines in ``src/models/cnn3d.py``. Synthetic-input only, no Sherwood
dependency.

Run:
    PYTHONPATH=. uv run pytest tests/test_cnn3d.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.cnn3d import (
    BasicBlock3D,
    MeanOverdensityBaseline,
    MeanVarianceBaseline,
    ResNet3D,
    resnet18_3d_4class,
)


def test_resnet18_3d_4class_forward_shape_contract():
    """Spec: input (B, 1, 32, 32, 32) -> logits (B, 4)."""
    net = resnet18_3d_4class(in_channels=1, num_classes=4)
    net.eval()
    x = torch.randn(8, 1, 32, 32, 32)
    out = net(x)
    assert out.shape == (8, 4), f"expected (8, 4), got {tuple(out.shape)}"
    assert torch.isfinite(out).all()


def test_resnet18_3d_4class_param_budget_in_range():
    """Design doc \xa73: target ~10-12M params. Confirm we're in that
    ballpark (allow 5M-25M as a generous window — the architecture
    surfaces design changes if the count walks outside this window).
    """
    net = resnet18_3d_4class()
    n_params = sum(p.numel() for p in net.parameters())
    assert 5_000_000 <= n_params <= 25_000_000, (
        f"param count {n_params:,} outside expected 5M-25M envelope; "
        f"design-doc target was ~10-12M"
    )


def test_resnet18_3d_4class_supports_crop_sizes_16_32_64():
    """Sanity: the architecture works at the design-doc-considered crop
    sizes. ``crop=32`` is the locked sprint-4 choice; ``16`` and ``64``
    must work for the deferred ablation."""
    net = resnet18_3d_4class()
    net.eval()
    for crop in (16, 32, 64):
        x = torch.randn(2, 1, crop, crop, crop)
        out = net(x)
        assert out.shape == (2, 4), f"crop={crop}: shape {out.shape}"


def test_resnet18_3d_4class_forward_deterministic_under_seed():
    """Gate (c) split-determinism: same seed -> bit-identical output."""
    torch.manual_seed(0)
    net1 = resnet18_3d_4class()
    torch.manual_seed(0)
    net2 = resnet18_3d_4class()
    net1.eval()
    net2.eval()
    torch.manual_seed(42)
    x = torch.randn(4, 1, 32, 32, 32)
    out1 = net1(x)
    out2 = net2(x)
    assert torch.equal(out1, out2), (
        f"two seed-identical models produced different outputs: "
        f"max abs diff = {(out1 - out2).abs().max().item():.3e}"
    )


def test_basicblock3d_residual_path_when_stride_1_same_channels():
    """Identity shortcut should be used when no projection is needed."""
    blk = BasicBlock3D(in_channels=32, out_channels=32, stride=1)
    assert isinstance(blk.shortcut, torch.nn.Identity)
    x = torch.randn(2, 32, 8, 8, 8)
    out = blk(x)
    assert out.shape == x.shape


def test_basicblock3d_projection_shortcut_when_stride_2():
    """1x1x1 projection shortcut activates when stride>1 or
    in_channels != out_channels."""
    blk = BasicBlock3D(in_channels=32, out_channels=64, stride=2)
    assert not isinstance(blk.shortcut, torch.nn.Identity)
    x = torch.randn(2, 32, 8, 8, 8)
    out = blk(x)
    assert out.shape == (2, 64, 4, 4, 4), f"got {out.shape}"


def test_resnet3d_custom_channels_param_budget():
    """Sanity-check the configurability surface — custom channels."""
    net = ResNet3D(
        block_counts=(2, 2, 2, 2),
        channels=(16, 32, 64, 128),
    )
    n_params = sum(p.numel() for p in net.parameters())
    # Should be well under the halved-32-channel default's ~10M
    assert 1_000_000 <= n_params <= 5_000_000


# -------------------------------------------------------- trivial baselines

def test_mean_overdensity_baseline_one_scalar_feature():
    """Gate (e₁): 1-scalar mean -> FC(4)."""
    net = MeanOverdensityBaseline(num_classes=4)
    x = torch.randn(8, 1, 32, 32, 32) + 1.0  # mean ~ 1
    out = net(x)
    assert out.shape == (8, 4)
    # Verify FC sees exactly 1 input feature
    assert net.fc.in_features == 1


def test_mean_variance_baseline_two_scalar_features():
    """Gate (e₂): 2-scalar mean+var -> FC(4)."""
    net = MeanVarianceBaseline(num_classes=4)
    x = torch.randn(8, 1, 32, 32, 32) + 1.0
    out = net(x)
    assert out.shape == (8, 4)
    assert net.fc.in_features == 2


def test_mean_baseline_invariant_to_voxel_permutation():
    """The 1-scalar mean baseline must give bit-identical output when
    voxels are permuted (since mean is permutation-invariant). This is
    the core anti-degeneracy reason a high accuracy here is a red flag.
    """
    net = MeanOverdensityBaseline()
    net.eval()
    x = torch.randn(4, 1, 32, 32, 32)
    out_orig = net(x)
    # Permute spatial voxels
    flat = x.flatten(2)  # (4, 1, 32**3)
    perm = torch.randperm(flat.size(2))
    x_perm = flat[:, :, perm].reshape_as(x)
    out_perm = net(x_perm)
    torch.testing.assert_close(out_orig, out_perm)

"""Stage-1 unit gates for the SIREN auto-decoder FeedbackField ([F-02] §4(c)-s1).

Covers: forward finite; output shape [B]; gradient reaches trunk AND z_p; SIREN
init statistics match Sitzmann+2020 spec; seeded determinism; explicit-z path;
param count ~ expected.
"""

import math

import torch

from src.models.feedback_field import FeedbackField


def _make(d=8, seed=0):
    torch.manual_seed(seed)
    return FeedbackField(d=d)


# (1) forward finite on random coords ---------------------------------------
def test_forward_finite():
    model = _make()
    B = 64
    coords = torch.rand(B, 3) * 2.0 - 1.0            # in [-1, 1]
    idx = torch.randint(0, 4, (B,))
    x = model(coords, variant_idx=idx)
    assert torch.isfinite(x).all()


# (2) output shape [B] -------------------------------------------------------
def test_output_shape():
    model = _make()
    B = 37
    coords = torch.rand(B, 3) * 2.0 - 1.0
    idx = torch.randint(0, 4, (B,))
    x = model(coords, variant_idx=idx)
    assert x.shape == (B,)


# (3) gradient reaches trunk AND z_p ----------------------------------------
def test_gradient_reaches_trunk_and_code():
    model = _make()
    B = 32
    coords = torch.rand(B, 3) * 2.0 - 1.0
    idx = torch.randint(0, 4, (B,))
    x = model(coords, variant_idx=idx)
    loss = x.pow(2).mean()
    loss.backward()

    # z_p (embedding) received gradient
    assert model.codes.weight.grad is not None
    assert model.codes.weight.grad.abs().sum() > 0

    # trunk weights received gradient
    first_lin = model.net[0].linear
    assert first_lin.weight.grad is not None
    assert first_lin.weight.grad.abs().sum() > 0
    assert model.out_layer.weight.grad is not None
    assert model.out_layer.weight.grad.abs().sum() > 0


# (4) SIREN init statistics match spec (spot-check weight bounds) -------------
def test_siren_init_bounds():
    model = _make(d=8)
    # first layer: U(-1/n_in, 1/n_in), n_in = 3 + d = 11
    first = model.net[0].linear
    first_bound = 1.0 / first.in_features
    assert first.weight.min() >= -first_bound - 1e-9
    assert first.weight.max() <= first_bound + 1e-9
    # a fresh uniform should nearly reach the bounds
    assert first.weight.abs().max() > 0.9 * first_bound

    # hidden layer: U(-sqrt(6/n)/w0, +sqrt(6/n)/w0), n = 256
    hidden = model.net[1].linear
    hidden_bound = math.sqrt(6.0 / hidden.in_features) / model.omega_0
    assert hidden.weight.min() >= -hidden_bound - 1e-9
    assert hidden.weight.max() <= hidden_bound + 1e-9
    assert hidden.weight.abs().max() > 0.9 * hidden_bound

    # code bank ~ N(0, 0.01^2): magnitude sanity (loose, small sample)
    assert model.codes.weight.abs().max() < 0.1


# (5) seeded determinism -----------------------------------------------------
def test_seeded_determinism():
    m1 = _make(seed=123)
    m2 = _make(seed=123)
    coords = torch.rand(16, 3) * 2.0 - 1.0
    idx = torch.randint(0, 4, (16,))
    x1 = m1(coords, variant_idx=idx)
    x2 = m2(coords, variant_idx=idx)
    assert torch.allclose(x1, x2, atol=0.0)


# (6) explicit-z path works and differs from embedding path when z differs ----
def test_explicit_z_path():
    model = _make()
    B = 24
    coords = torch.rand(B, 3) * 2.0 - 1.0

    # explicit-z equals embedding path when z == the looked-up code
    idx = torch.zeros(B, dtype=torch.long)
    z0 = model.codes.weight[0].detach()
    x_emb = model(coords, variant_idx=idx)
    x_z = model(coords, z=z0)
    assert torch.allclose(x_emb, x_z, atol=1e-6)

    # a different z produces a different output
    z_other = z0 + 1.0
    x_other = model(coords, z=z_other)
    assert not torch.allclose(x_z, x_other, atol=1e-4)

    # exactly-one-of guard
    try:
        model(coords, variant_idx=idx, z=z0)
        raised = False
    except ValueError:
        raised = True
    assert raised


# (7) param count ~ expected -------------------------------------------------
def test_param_count():
    d = 8
    model = _make(d=d)
    n = model.num_parameters()
    # first(11->256): 3072; 4 x (256->256): 263168; out(256->1): 257;
    # codes(4x8): 32  => 266529
    expected = 3072 + 4 * (256 * 256 + 256) + 257 + 4 * d
    assert n == expected
    # trunk-only ~ 0.26 M
    trunk = n - 4 * d
    assert abs(trunk - 266497) == 0
    assert 0.25e6 < trunk < 0.27e6

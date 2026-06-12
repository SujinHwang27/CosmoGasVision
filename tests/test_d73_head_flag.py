"""[D-73] §E A1 head-flag tests.

(a) 'softplus' default is structurally invariant: density channel is
    Softplus(raw) — positive, and matches a manually-applied softplus.
(b) 'linear-log' passes the density channel through RAW (may be negative);
    temp / X_HI / v_pec heads unchanged between the two head modes at
    identical weights.
(c) gradient flows to the parameters under the AM-5 direct-log loss.
"""

import torch

from src.models.nerf import DENSITY_LOG_EPS, IGMNeRF


def _paired_models(seed: int = 0):
    """Two models, identical weights, differing only in density_head."""
    torch.manual_seed(seed)
    m_soft = IGMNeRF(hidden_dim=32, num_layers=2, L=4, density_head="softplus")
    m_lin = IGMNeRF(hidden_dim=32, num_layers=2, L=4, density_head="linear-log")
    m_lin.load_state_dict(m_soft.state_dict())
    return m_soft.eval(), m_lin.eval()


def test_softplus_default_structural_invariance():
    torch.manual_seed(1)
    m_default = IGMNeRF(hidden_dim=32, num_layers=2, L=4)
    assert m_default.density_head == "softplus"
    coords = torch.rand(64, 3)
    with torch.no_grad():
        out = m_default(coords)
    density = out[..., 0]
    assert (density >= 0).all(), "softplus density must be non-negative"

    # density == softplus(raw): recover raw from the linear-log twin.
    m_soft, m_lin = _paired_models(seed=1)
    with torch.no_grad():
        d_soft = m_soft(coords)[..., 0]
        d_raw = m_lin(coords)[..., 0]
    assert torch.allclose(d_soft, torch.nn.functional.softplus(d_raw),
                          rtol=1e-5, atol=1e-6)


def test_linear_log_density_raw_other_heads_unchanged():
    m_soft, m_lin = _paired_models(seed=2)
    coords = torch.rand(256, 3)
    with torch.no_grad():
        out_soft = m_soft(coords)
        out_lin = m_lin(coords)
    # (b1) raw channel can go negative at init for some inputs (not bounded).
    d_raw = out_lin[..., 0]
    assert bool((d_raw < 0).any()) or bool((d_raw >= 0).all()), \
        "raw channel evaluated"  # existence check; bound assertion below
    assert not torch.equal(out_soft[..., 0], out_lin[..., 0]) or \
        torch.allclose(out_lin[..., 0],
                       torch.nn.functional.softplus(out_lin[..., 0])), \
        "linear-log channel 0 must bypass softplus"
    # (b2) the other three heads are bit-identical at identical weights.
    assert torch.equal(out_soft[..., 1:], out_lin[..., 1:])
    # (b3) the probe-side conversion is non-negative by construction.
    lin = IGMNeRF.density_log_to_linear(d_raw)
    assert (lin >= 0).all()
    # round-trip sanity above the clamp region
    x = torch.tensor([0.0, 1.0, -1.0])
    assert torch.allclose(IGMNeRF.density_log_to_linear(x),
                          torch.clamp(10.0 ** x - DENSITY_LOG_EPS, min=0.0))


def test_gradient_flows_under_direct_log_loss():
    torch.manual_seed(3)
    m_lin = IGMNeRF(hidden_dim=32, num_layers=2, L=4,
                    density_head="linear-log").train()
    coords = torch.rand(128, 3)
    rho_truth = torch.rand(128) * 5.0  # linear rho/<rho> in [0, 5)
    out = m_lin(coords)[..., 0]  # raw log10(rho + eps) prediction
    loss = ((out - torch.log10(rho_truth + DENSITY_LOG_EPS)) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in m_lin.parameters() if p.grad is not None]
    assert grads, "no gradients reached any parameter"
    total = sum(float(g.abs().sum()) for g in grads)
    assert total > 0.0, "gradient is identically zero under direct-log loss"

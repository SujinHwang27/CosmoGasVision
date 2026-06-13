"""[D-73] (1d') Tests for the explicit four-field voxel-grid producer.

Three obligations from the dispatch brief:
  (a) 4-field output contract matches IGMNeRF's [..., 4] shape + the head-map
      ranges (density >= 0, T in [1e3, ~1e7], X_HI in [0,1], v_pec in
      [-500, 500]).
  (b) gradient flows to all 4 grids under a flux-supervised loss step.
  (c) trilinear interp is autograd-live (no detached numpy in the path).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.nerf import IGMNeRF, volume_render_physics
from src.models.voxel_grid_field import (
    VoxelGridField,
    DENSITY_LOG_EPS,
    MEAN_LOG_RHO_INIT,
)


def _coords(n_rays: int, n_bins: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_rays, n_bins, 3, generator=g)


# --------------------------------------------------------------------------- (a)
def test_output_shape_matches_igmnerf_contract():
    """VoxelGridField returns the same [..., 4] stack shape as IGMNeRF."""
    coords = _coords(8, 16)
    grid = VoxelGridField(grid_size=16)
    mlp = IGMNeRF()

    out_grid = grid(coords)
    out_mlp = mlp(coords)

    assert out_grid.shape == (8, 16, 4)
    assert out_grid.shape == out_mlp.shape


def test_head_map_ranges():
    """density >= 0 ; T in [1e3, ~1e7] ; X_HI in [0,1] ; v_pec in [-500, 500].

    Stress the ranges with large-magnitude raw grid values so the head maps
    (not just the near-mean init) are exercised.
    """
    grid = VoxelGridField(grid_size=8, init_noise_std=0.0)
    # Push raw grids to extreme values to probe the saturating head maps.
    with torch.no_grad():
        grid.log_rho_grid.fill_(3.0)        # 10^3 -> linear ~1000
        grid.temp_grid.uniform_(-5.0, 30.0)
        grid.xhi_grid.uniform_(-30.0, 30.0)
        grid.vpec_grid.uniform_(-30.0, 30.0)

    coords = _coords(32, 8, seed=1)
    out = grid(coords)
    density, temp, h1_frac, vpec = out.unbind(dim=-1)

    assert torch.all(density >= 0.0), "density (rho/<rho>) must be >= 0"
    assert torch.all(temp >= 1.0e3 - 1e-3), "T must be >= 1e3 K floor"
    assert torch.all(temp <= 1.0e7), f"T exceeded ~1e7 K: max={temp.max()}"
    assert torch.all(h1_frac >= 0.0) and torch.all(h1_frac <= 1.0)
    assert torch.all(vpec >= -500.0 - 1e-3) and torch.all(vpec <= 500.0 + 1e-3)


def test_density_head_softplus_vs_linear_log_contract():
    """softplus contract returns LINEAR rho/<rho>; linear-log returns raw log.

    The linear-log raw value passed through density_log_to_linear must equal
    the softplus-contract channel 0 (both read the same stored log grid).
    """
    coords = _coords(4, 4, seed=2)
    g_soft = VoxelGridField(grid_size=8, init_noise_std=0.0, density_head="softplus")
    g_log = VoxelGridField(grid_size=8, init_noise_std=0.0, density_head="linear-log")
    # Force identical grids so the only difference is the channel-0 contract.
    with torch.no_grad():
        for p_l, p_s in zip(g_log.parameters(), g_soft.parameters()):
            p_l.copy_(p_s)
        g_log.log_rho_grid.uniform_(-2.0, 2.0)
        g_soft.log_rho_grid.copy_(g_log.log_rho_grid)

    d_soft = g_soft(coords)[..., 0]
    d_log_raw = g_log(coords)[..., 0]
    d_log_lin = VoxelGridField.density_log_to_linear(d_log_raw)

    assert torch.allclose(d_soft, d_log_lin, atol=1e-5)


def test_constant_mean_init_recovers_unit_density():
    """At constant-mean init (no noise) every voxel is rho/<rho> ~= 1."""
    grid = VoxelGridField(grid_size=8, init_noise_std=0.0)
    coords = _coords(4, 4, seed=3)
    density = grid(coords)[..., 0]
    # log10(1+1e-3) stored -> density_log_to_linear -> ~1.0
    assert MEAN_LOG_RHO_INIT == pytest.approx(math.log10(1.0 + DENSITY_LOG_EPS))
    assert torch.allclose(density, torch.ones_like(density), atol=1e-4)


# --------------------------------------------------------------------------- (b)
def test_gradient_flows_to_all_four_grids_under_flux_loss():
    """A flux-supervised loss step produces non-None, non-zero grads on all 4 grids.

    Uses the production renderer (volume_render_physics) so the gradient path is
    the IDENTICAL [D-24] forward path the MLP uses — the grid drops in unchanged.
    """
    torch.manual_seed(7)
    n_rays, n_bins, n_obs = 6, 24, 24
    coords = _coords(n_rays, n_bins, seed=5)
    vel_axis = torch.linspace(0.0, 2000.0, n_obs)

    grid = VoxelGridField(grid_size=12, init_noise_std=0.01)
    tau_amp = torch.nn.Parameter(torch.tensor(1.0))

    tau_pred = volume_render_physics(grid, coords, vel_axis=vel_axis, tau_amp=tau_amp)
    # Synthetic flux-supervised target: a non-trivial tau profile.
    tau_gt = torch.rand(n_rays, n_obs) * 2.0
    tau_pred_eff = tau_pred.clamp_max(10.0)
    tau_gt_eff = tau_gt.clamp_max(10.0)
    loss = ((torch.log1p(tau_pred_eff) - torch.log1p(tau_gt_eff)) ** 2).mean()
    loss.backward()

    for name in ("log_rho_grid", "temp_grid", "xhi_grid", "vpec_grid"):
        p = getattr(grid, name)
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"
        assert float(p.grad.abs().sum()) > 0.0, f"{name} grad is all-zero"


def test_vpec_grid_feeds_rsd_shift():
    """The free vpec grid changes tau (RSD-parity, SERIOUS-2).

    Zeroing vs perturbing the vpec grid must change the rendered tau, proving
    the vpec grid feeds the v_source = vel_axis + vpec RSD shift (nerf.py:328).
    """
    torch.manual_seed(11)
    coords = _coords(4, 24, seed=6)
    vel_axis = torch.linspace(0.0, 2000.0, 24)

    grid = VoxelGridField(grid_size=10, init_noise_std=0.0)
    # Make density/xhi non-trivial so tau is non-zero and RSD-sensitive.
    with torch.no_grad():
        grid.log_rho_grid.uniform_(0.0, 1.0)
        grid.xhi_grid.fill_(0.0)  # sigmoid(0)=0.5

    with torch.no_grad():
        grid.vpec_grid.fill_(0.0)
        tau_zero = volume_render_physics(grid, coords, vel_axis=vel_axis).clone()
        grid.vpec_grid.uniform_(-5.0, 5.0)  # large raw -> ~+/-500 km/s tanh
        tau_shift = volume_render_physics(grid, coords, vel_axis=vel_axis)

    assert not torch.allclose(tau_zero, tau_shift, atol=1e-4), (
        "vpec grid did not affect tau — RSD shift not wired"
    )


# --------------------------------------------------------------------------- (c)
def test_trilinear_interp_is_autograd_live():
    """Interpolated value carries grad back to the grid (no detached numpy)."""
    grid = VoxelGridField(grid_size=8, init_noise_std=0.0)
    coords = _coords(3, 5, seed=8)
    out = grid(coords)
    assert out.requires_grad, "forward output is detached from the grid params"

    out.sum().backward()
    assert grid.log_rho_grid.grad is not None
    assert grid.temp_grid.grad is not None
    assert grid.xhi_grid.grad is not None
    assert grid.vpec_grid.grad is not None


def test_interp_respects_grid_values_at_voxel_centers():
    """grid_sample with align_corners=True hits voxel centers at box edges.

    A coordinate at the box corner (0,0,0) should read the corner voxel value
    (within trilinear tolerance), confirming the [0,1]->[-1,1] mapping and the
    axis ordering are correct.
    """
    grid = VoxelGridField(grid_size=4, init_noise_std=0.0)
    with torch.no_grad():
        # Set a distinctive value at corner voxel [0,0,0] of the log_rho grid.
        grid.log_rho_grid.fill_(0.0)
        grid.log_rho_grid[0, 0, 0] = 5.0
    corner = torch.tensor([[[0.0, 0.0, 0.0]]])  # (1,1,3)
    raw = grid._sample_grid(grid.log_rho_grid, corner)
    assert float(raw.reshape(-1)[0].detach()) == pytest.approx(5.0, abs=1e-4)


def test_g_and_physics_id_rejected():
    """Passing g or physics_id must raise (one-lever: no conditioning)."""
    grid = VoxelGridField(grid_size=6)
    coords = _coords(2, 4, seed=9)
    with pytest.raises(RuntimeError):
        grid(coords, g=torch.zeros(2, 4, 1))
    with pytest.raises(RuntimeError):
        grid(coords, physics_id=torch.zeros(2, dtype=torch.long))

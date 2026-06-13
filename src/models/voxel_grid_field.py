"""[D-73] (1d') Explicit four-field voxel-grid IGM field producer (option (a)).

Drop-in replacement for ``IGMNeRF`` as the field producer consumed by
``src.models.nerf.volume_render_physics``. Where ``IGMNeRF`` maps a 3D
coordinate through an 8-layer MLP to a 4-field stack, ``VoxelGridField`` maps
the same coordinate through trilinear interpolation of four independent dense
(G, G, G) grids — one grid per production MLP output head.

The four heads are the production MLP's four FREE heads (``nerf.py:204-218``);
there is NO FGPA closure here, exactly as there is none in the production
forward path. The output-head transforms are byte-for-byte the
``nerf.py:206-216`` maps applied to the trilinearly-interpolated raw grid
value:

    density : linear-log head — raw grid value IS log10(rho/<rho> + 1e-3);
              passed through RAW (may be negative), exactly the
              ``nerf.py:208-213`` density_head='linear-log' branch. Conversion
              to linear space is the caller's job via density_log_to_linear()
              (probe-side); the renderer consumes channel 0 as-is, identical to
              how it consumes the MLP's linear-log channel 0. [see note below]
    temp    : softplus(raw) * 1e4 + 1e3              (nerf.py:214)
    h1_frac : sigmoid(raw)                            (nerf.py:215)
    vpec    : tanh(raw) * 500                         (nerf.py:216)

Note on the density head: the production publication path uses the
density_head='softplus' default (channel 0 = Softplus(raw), linear rho/<rho>).
The [D-73] §E linear-log head stores log10(rho/<rho>+1e-3) per cell. Per design
doc §2 the voxel grid stores log10(rho/<rho>+1e-3) directly for parity with the
production density representation and dynamic-range stability, then exposes the
SAME forward contract as ``density_head='softplus'``: channel 0 of the returned
stack is the LINEAR rho/<rho> (>= 0), obtained via density_log_to_linear on the
interpolated log value. This keeps the renderer call site
(``volume_render_physics`` line 304: ``density = fields[..., 0]  # rho/<rho>``)
byte-identical to the production MLP path. The ``density_head`` kwarg below lets
a caller instead pass channel 0 through raw (linear-log forward contract) to
match an MLP built with density_head='linear-log'; default 'softplus' matches
the production publication route the (1d') test compares against.

CLAUDE.md differentiability contract: the interpolation is
``torch.nn.functional.grid_sample`` (autograd-live), no detached numpy in the
ray-sampling path; the four grids are leaf nn.Parameters and the forward
applies no in-place op on them.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mirror nerf.DENSITY_LOG_EPS (kept independent so src/models has no
# cross-import coupling; same 1e-3 value).
DENSITY_LOG_EPS = 1.0e-3

# Constant-mean log-density init: log10(<rho>/<rho> + 1e-3) = log10(1 + 1e-3).
# This is the constant-mean basin A1 collapsed into; initializing here makes the
# trainability gate a clean test of whether flux gradients drive the grid OUT of
# constant-mean (design §2).
MEAN_LOG_RHO_INIT = math.log10(1.0 + DENSITY_LOG_EPS)  # ~= 4.34e-4


def _inv_softplus(y: float) -> float:
    """Raw pre-activation s such that softplus(s) == y (y > 0)."""
    # softplus(s) = log(1 + exp(s)) ; inverse = log(exp(y) - 1)
    return float(math.log(math.expm1(y)))


def _logit(p: float) -> float:
    return float(math.log(p / (1.0 - p)))


class VoxelGridField(nn.Module):
    """Four independent dense (G, G, G) grids -> 4-field stack.

    forward(coords, g=None, physics_id=None) -> (..., 4) stack matching the
    IGMNeRF output contract (density, temp, h1_frac, vpec), so it drops into
    ``volume_render_physics`` unchanged.

    Args:
        grid_size: G. Each of the four grids is (G, G, G).
        init_noise_std: sigma of the symmetry-breaking Gaussian noise added to
            every grid's constant-mean init (design §2: sigma=0.01).
        density_head: 'softplus' (default) returns channel-0 as LINEAR rho/<rho>
            via density_log_to_linear(interp_log_rho); 'linear-log' returns the
            raw interpolated log10 value (matching an MLP built with
            density_head='linear-log'). The grid ALWAYS STORES log10(rho/<rho>+
            1e-3) regardless (design §2 output-transform); the flag only selects
            the forward contract on channel 0.
        align_corners: grid_sample alignment. True maps the [0,1] box edges to
            the outermost voxel CENTERS (so the corner voxels are addressable),
            matching the trilinear convention where coords index voxel centers.
    """

    def __init__(
        self,
        grid_size: int = 128,
        init_noise_std: float = 0.01,
        density_head: str = "softplus",
        align_corners: bool = True,
        # Sensible constant-mean inits for the other three fields. These are the
        # raw pre-activation values whose head-map outputs land at physically
        # central values:
        #   T ~ 1e4 K  -> softplus(s)*1e4+1e3 = 1e4 -> softplus(s)=0.9
        #   X_HI ~ 1e-5 (highly ionized IGM at z=0.3) -> sigmoid(s)=1e-5
        #   v_pec ~ 0  -> tanh(s)*500 = 0 -> s=0
        init_temp_K: float = 1.0e4,
        init_xhi: float = 1.0e-5,
    ):
        super().__init__()
        if density_head not in ("softplus", "linear-log"):
            raise ValueError(
                f"density_head must be 'softplus' or 'linear-log'; "
                f"got {density_head!r}"
            )
        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2; got {grid_size}")
        self.grid_size = int(grid_size)
        self.density_head = density_head
        self.align_corners = bool(align_corners)
        self.init_noise_std = float(init_noise_std)

        G = self.grid_size

        # ---- Raw pre-activation constant-mean inits -------------------------
        # density grid stores log10(rho/<rho>+1e-3) directly.
        rho_mean = MEAN_LOG_RHO_INIT
        # temp raw: softplus(s)*1e4+1e3 = init_temp_K -> softplus(s) = (T-1e3)/1e4
        temp_softplus_target = (init_temp_K - 1.0e3) / 1.0e4
        if temp_softplus_target <= 0:
            raise ValueError(
                f"init_temp_K={init_temp_K} below the 1e3 K floor; "
                "softplus target must be > 0."
            )
        temp_raw = _inv_softplus(temp_softplus_target)
        # x_HI raw: sigmoid(s) = init_xhi
        xhi_raw = _logit(init_xhi)
        # v_pec raw: tanh(s)*500 = 0 -> s = 0
        vpec_raw = 0.0

        # ---- Four leaf parameter grids, constant-mean + symmetry-break noise -
        def _init_grid(mean_val: float) -> torch.Tensor:
            base = torch.full((G, G, G), float(mean_val), dtype=torch.float32)
            if self.init_noise_std > 0:
                base = base + torch.randn_like(base) * self.init_noise_std
            return base

        self.log_rho_grid = nn.Parameter(_init_grid(rho_mean))
        self.temp_grid = nn.Parameter(_init_grid(temp_raw))
        self.xhi_grid = nn.Parameter(_init_grid(xhi_raw))
        self.vpec_grid = nn.Parameter(_init_grid(vpec_raw))

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def density_log_to_linear(log_density: torch.Tensor) -> torch.Tensor:
        """[D-73] §E: rho_theta = clamp(10**log_density - 1e-3, min=0).

        Identical to IGMNeRF.density_log_to_linear. Used to convert the stored
        log10 density to linear rho/<rho> for the renderer (softplus-contract
        forward) and for variance/correlation probes.
        """
        return torch.clamp(
            torch.pow(10.0, log_density) - DENSITY_LOG_EPS, min=0.0
        )

    def _sample_grid(self, grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Autograd-live trilinear sample of a single (G,G,G) grid at coords.

        Args:
            grid: (G, G, G) leaf parameter.
            coords: (..., 3) in the [0, 1] unit cube ([D-08] convention).

        Returns:
            (...,) interpolated raw grid values.
        """
        orig_shape = coords.shape[:-1]
        n = int(torch.tensor(orig_shape).prod().item()) if len(orig_shape) else 1
        flat = coords.reshape(n, 3)

        # grid_sample expects normalized coords in [-1, 1]; our box coords are
        # [0, 1]. Map [0,1] -> [-1,1]. With align_corners=True the endpoints
        # -1/+1 land on the outermost voxel CENTERS.
        gs = flat * 2.0 - 1.0  # (n, 3) in [-1, 1]

        # grid_sample(input, grid):
        #   input: (N, C, D, H, W)
        #   grid : (N, D_out, H_out, W_out, 3) with last-dim order (x, y, z)
        #          where x indexes W, y indexes H, z indexes D.
        # We treat the (G,G,G) grid as input[0,0] over (D=G, H=G, W=G) with
        # axis order (axis0->D, axis1->H, axis2->W). To keep coordinate axis
        # i addressing grid axis i, grid_sample's last-dim (x,y,z) must be
        # (coord2, coord1, coord0) since x->W(axis2), y->H(axis1), z->D(axis0).
        inp = grid[None, None]  # (1, 1, G, G, G)
        # Build sampling grid of shape (1, n, 1, 1, 3).
        samp = torch.stack(
            [gs[:, 2], gs[:, 1], gs[:, 0]], dim=-1
        )  # (n, 3) reordered to (x=axis2, y=axis1, z=axis0)
        samp = samp.view(1, n, 1, 1, 3)

        out = F.grid_sample(
            inp,
            samp,
            mode="bilinear",  # trilinear for 5D input
            padding_mode="border",
            align_corners=self.align_corners,
        )  # (1, 1, n, 1, 1)
        out = out.reshape(n)
        return out.reshape(orig_shape) if len(orig_shape) else out.reshape(())

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        physics_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """coords -> (..., 4) field stack [density, temp, h1_frac, vpec].

        ``g`` and ``physics_id`` are accepted for call-signature parity with
        ``IGMNeRF.forward`` (the renderer passes them) but are UNUSED — the
        voxel grid carries no velocity-gradient conditioning ([D-42]) and no
        physics embedding ([D-46]). If a non-None value is passed we raise, so a
        misconfigured caller fails loudly rather than silently dropping the
        conditioning.
        """
        if g is not None:
            raise RuntimeError(
                "VoxelGridField does not support velocity-gradient conditioning "
                "(g); the (1d') one-lever test runs without [D-42] conditioning."
            )
        if physics_id is not None:
            raise RuntimeError(
                "VoxelGridField does not support physics embedding (physics_id); "
                "the (1d') one-lever test runs single-physics (P1)."
            )

        # Interpolate the four raw grids at the sample coords.
        log_rho = self._sample_grid(self.log_rho_grid, x)
        temp_raw = self._sample_grid(self.temp_grid, x)
        xhi_raw = self._sample_grid(self.xhi_grid, x)
        vpec_raw = self._sample_grid(self.vpec_grid, x)

        # Apply the nerf.py:206-216 head maps byte-for-byte.
        if self.density_head == "softplus":
            # Production publication contract: channel 0 is LINEAR rho/<rho>.
            # The grid stores log10(rho/<rho>+1e-3); convert to linear here so
            # the renderer (which reads channel 0 as rho/<rho>) is unchanged.
            density = self.density_log_to_linear(log_rho)
        else:
            # linear-log forward contract: channel 0 IS the raw log value.
            density = log_rho

        temp = self.softplus(temp_raw) * 10**4 + 10**3  # T ~ 1e3 to ~1e6 K
        h1_frac = self.sigmoid(xhi_raw)                  # 0 to 1
        vpec = torch.tanh(vpec_raw) * 500                # +/- 500 km/s

        return torch.stack([density, temp, h1_frac, vpec], dim=-1)

import math
from typing import Optional

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L

    def forward(self, x):
        """
        x: (..., 3) coordinates
        returns: (..., 3 + 6*L) fourier features
        """
        res = [x]
        for l in range(self.L):
            freq = (2.0 ** l) * torch.pi
            res.append(torch.sin(freq * x))
            res.append(torch.cos(freq * x))
        return torch.cat(res, dim=-1)

# [D-73] §E: numerical stabilizer for the linear-log density head. Matches
# experiments/nerf/pipeline.PRETRAIN_LOG_EPS (kept as a separate constant so
# src/models has no import dependency on the experiment pipeline).
DENSITY_LOG_EPS = 1.0e-3


class IGMNeRF(nn.Module):
    """
    Continuous MLP mapping 3D position -> density, temp, h1_frac, vpec.

    body_arch: 'current' (default) | 'skip-rich-mlp' [D-70 Rev 5.1 (1b)]
        - 'current'      — original 4+4 layers1/layers2 with single mid-skip.
        - 'skip-rich-mlp' — DeepSDF every-layer skip-input variant: 8 body
          layers, each consuming concat([h, skip_in]) (encoded coords [+g]
          re-injected at every layer). ReLU activations unchanged; head
          unchanged. Tests the Lu+2019 dying-ReLU body pathology under the
          same activation regime as 'current' (operative-cause swap test).

    density_head: 'softplus' (default) | 'linear-log' [D-73 §E A1 probe]
        - 'softplus'   — channel 0 of the output is Softplus(out[..., 0]),
          i.e. linear-space rho/<rho> >= 0. Structurally invariant default
          (mirrors the [D-70] body_arch precedent: identical module
          construction, no behavioural change).
        - 'linear-log' — channel 0 passes through RAW: out[..., 0] IS
          log10(rho/<rho> + 1e-3). The Softplus is bypassed for the density
          channel ONLY; temp / X_HI / v_pec heads are unchanged. Training
          loss under this head is computed DIRECTLY on the raw log-space
          output (AM-5: never round-trip through clamp(10**out) — the clamp
          kills gradient below out = -3). Use density_log_to_linear() to
          obtain linear-space rho_theta for variance/correlation probes only.
    """
    def __init__(self, hidden_dim=256, num_layers=8, L=10,
                 use_velocity_gradient_conditioning: bool = False,
                 use_physics_embedding: bool = False,
                 n_physics: int = 4,
                 physics_embedding_dim: int = 16,
                 body_arch: str = "current",
                 density_head: str = "softplus"):
        super().__init__()
        if body_arch not in ("current", "skip-rich-mlp"):
            raise ValueError(
                f"body_arch must be 'current' or 'skip-rich-mlp'; got {body_arch!r}"
            )
        if density_head not in ("softplus", "linear-log"):
            raise ValueError(
                f"density_head must be 'softplus' or 'linear-log'; "
                f"got {density_head!r}"
            )
        self.body_arch = body_arch
        self.density_head = density_head
        # use_velocity_gradient_conditioning toggles the [D-42] sidecar feature
        # (Sherwood-truth dv_pec/dchi, z-scored, detached) concatenated onto the
        # encoded coordinate before layer 1 — see LEDGER §3 [D-42] Math contract.
        #
        # use_physics_embedding toggles the [D-46] joint-physics conditioning:
        # a learned nn.Embedding(n_physics, physics_embedding_dim) indexed by
        # per-ray physics_id is concatenated onto the encoded coordinate at
        # MLP input (layer 1) ONLY — NOT at the skip-connection layer 5.
        # This *may* let a single MLP fit the 4 Sherwood feedback variants
        # jointly (hedged: see LEDGER §3 [D-46]). Default-OFF preserves
        # byte-equivalent baseline; tested in tests/test_physics_embedding.py.
        self.encoding = PositionalEncoding(L)
        self.use_velocity_gradient_conditioning = use_velocity_gradient_conditioning
        self.use_physics_embedding = use_physics_embedding
        self.n_physics = n_physics
        self.physics_embedding_dim = physics_embedding_dim

        encoded_dim = 3 + 2 * 3 * L
        g_dim = 1 if use_velocity_gradient_conditioning else 0
        # [D-46]: physics embedding adds physics_embedding_dim to layer-1
        # in_features only (not to the skip-connection re-injection vector).
        e_dim = physics_embedding_dim if use_physics_embedding else 0
        in_dim = encoded_dim + g_dim + e_dim
        # Skip-connection re-injects the (encoded, g) vector but NOT the
        # physics embedding e_p — per [D-46] Math contract, physics_id is a
        # coord-level conditioning applied at MLP input only.
        skip_dim = encoded_dim + g_dim

        if use_physics_embedding:
            self.physics_embedding = nn.Embedding(n_physics, physics_embedding_dim)

        if body_arch == "current":
            # Estimator-equivalence path: identical construction order/shape to
            # the pre-Rev-5.1 model. Do NOT touch this branch.
            self.layers1 = nn.ModuleList()
            for i in range(4):
                dim = in_dim if i == 0 else hidden_dim
                self.layers1.append(nn.Linear(dim, hidden_dim))

            self.layers2 = nn.ModuleList()
            for i in range(num_layers - 4):
                dim = hidden_dim + skip_dim if i == 0 else hidden_dim
                self.layers2.append(nn.Linear(dim, hidden_dim))

            # Output layer input dimension depends on whether we have layers after the skip connection
            out_in_dim = hidden_dim if (num_layers - 4) > 0 else (hidden_dim + skip_dim)
        else:
            # [D-70 (1b)] Skip-rich MLP: 8 body layers, every layer consumes
            # concat([h, skip_in]) where skip_in = encoded[, g] (matches the
            # current single-mid-skip skip_dim contract — physics embedding
            # excluded from re-injection, same as 'current').
            #
            # First layer input: h_in (= skip_in [+e_p]) has dim `in_dim`, then
            # concatenated with skip_in again -> in_dim + skip_dim.
            # Subsequent layers: hidden_dim + skip_dim.
            #
            # We still expose .layers1 (length 0 ModuleList) + .layers2 (the
            # 8 body layers) so the d70 pre-flight measurement code that
            # iterates list(model.layers1) + list(model.layers2) sees 8 sites
            # under both arches.
            self.layers1 = nn.ModuleList()  # empty under skip-rich
            self.layers2 = nn.ModuleList()
            for i in range(num_layers):
                in_features = (in_dim + skip_dim) if i == 0 else (hidden_dim + skip_dim)
                self.layers2.append(nn.Linear(in_features, hidden_dim))
            out_in_dim = hidden_dim

        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(out_in_dim, 4)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g: Optional[torch.Tensor] = None,
                physics_id: Optional[torch.Tensor] = None):
        encoded = self.encoding(x)
        # Skip-connection input is (encoded[, g]) — without e_p, per [D-46].
        if g is not None:
            # [D-42]: concatenate the z-scored velocity-gradient feature as a
            # raw scalar onto the encoded coordinate (no positional encoding —
            # g is already ~ N(0, 1)). Constructor must have been built with
            # use_velocity_gradient_conditioning=True so layer_1 has the +1
            # in_features.
            skip_in = torch.cat([encoded, g], dim=-1)
        else:
            skip_in = encoded

        # [D-46]: physics embedding is concatenated AFTER coords (order:
        # [fourier_encoded_coords, g?, e_p]). Embedding enters layer 1 only;
        # it is excluded from skip_in so the residual fan-in matches the
        # baseline skip_dim contract.
        if physics_id is not None:
            if not self.use_physics_embedding:
                raise RuntimeError(
                    "physics_id passed to forward() but model was built with "
                    "use_physics_embedding=False."
                )
            # physics_id shape: (n_rays,) long. Broadcast across n_bins to
            # match (n_rays, n_bins, ...) coords/encoded shape.
            e_p = self.physics_embedding(physics_id)            # (n_rays, e_dim)
            # Add a bin axis and broadcast to (n_rays, n_bins, e_dim).
            target_shape = list(encoded.shape[:-1]) + [self.physics_embedding_dim]
            e_p_expanded = e_p.unsqueeze(1).expand(target_shape)
            h_in = torch.cat([skip_in, e_p_expanded], dim=-1)
        else:
            if self.use_physics_embedding:
                raise RuntimeError(
                    "Model was built with use_physics_embedding=True but no "
                    "physics_id was passed to forward()."
                )
            h_in = skip_in

        h = h_in

        if self.body_arch == "current":
            for layer in self.layers1:
                h = self.relu(layer(h))

            # NeRF residual/skip connection re-injects the (encoded[, g]) vector
            # — physics embedding e_p is intentionally excluded per [D-46].
            h = torch.cat([h, skip_in], dim=-1)

            for layer in self.layers2:
                h = self.relu(layer(h))
        else:
            # [D-70 (1b)] Skip-rich: concat skip_in into every body layer input.
            for layer in self.layers2:
                h_in_layer = torch.cat([h, skip_in], dim=-1)
                h = self.relu(layer(h_in_layer))

        out = self.out_layer(h)
        # Bounding fields to physical ranges
        if self.density_head == "softplus":
            density = self.softplus(out[..., 0])  # rho/rho_bar >= 0
        else:
            # [D-73] §E linear-log head: channel 0 IS log10(rho/<rho> + 1e-3),
            # passed through raw (may be negative). Conversion to linear space
            # is the caller's job via density_log_to_linear() — probe-side
            # only, never in the loss (AM-5).
            density = out[..., 0]
        temp = self.softplus(out[..., 1]) * 10**4 + 10**3 # T ~ 10^3 to 10^6 K
        h1_frac = self.sigmoid(out[..., 2])   # 0 to 1
        vpec = torch.tanh(out[..., 3]) * 500  # Peculiar velocity +/- 500 km/s

        return torch.stack([density, temp, h1_frac, vpec], dim=-1)

    @staticmethod
    def density_log_to_linear(log_density: torch.Tensor) -> torch.Tensor:
        """[D-73] §E: convert the linear-log head's raw log-space density
        output to linear-space rho_theta = clamp(10**out - 1e-3, min=0).

        PROBE-SIDE ONLY (variance / correlation diagnostics). Do NOT use in
        the training loss — the clamp kills gradient below out = -3 (AM-5);
        the linear-log loss is computed directly on the raw output.
        """
        return torch.clamp(
            torch.pow(10.0, log_density) - DENSITY_LOG_EPS, min=0.0
        )

def tepper_garcia_voigt(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Tepper-García (2006) analytic approximation to the Voigt-Hjerting function
    with a defensive Taylor branch for |x| << 1.

    Fully gradient-safe and numerically stable.
    """
    x2 = x ** 2
    small = x2 < 1e-4

    # Sanitize x2 for the main branch to avoid NaN/Inf in gradients
    x2_safe = torch.where(small, torch.ones_like(x2), x2)

    # Main branch (expanded LEDGER form)
    exp_x2 = torch.exp(-x2_safe)
    exp_2x2 = torch.exp(-2.0 * x2_safe)
    poly = 4 * x2_safe**2 + 7 * x2_safe + 4 + 1.5 / x2_safe
    bracket = exp_2x2 * poly - 1.5 / x2_safe - 1.0
    H_main = exp_x2 - (a / (math.sqrt(math.pi) * x2_safe)) * bracket

    # Leading-order Taylor expansion for very small |x|
    H_small = torch.exp(-x2) - 2.0 * a / math.sqrt(math.pi)

    # Blend branches
    H = torch.where(small, H_small, H_main)

    return torch.clamp(H, min=0.0)

def volume_render_physics(mlp, ray_points, vel_axis, tau_amp=None, window=64, z=0.3, return_tau_local=False, g=None, physics_id=None):
    """
    Differentiable Lyman-alpha optical depth rendering with windowed RSD convolution.

    Implements the discrete form of
        tau(v_obs) = A * sum_{src in window(obs)} n_HI[src] * phi(v_obs - v_src - vpec[src])
    where phi = H(a, x) / (b * sqrt(pi)) is the normalized Voigt-Hjerting line
    profile. For each source bin, the Voigt weight is evaluated only at the
    +/- `window` nearest observed-velocity bins around the line center
    (v_src + v_pec). The Tepper-Garcia profile decays faster than exp(-x^2/2)
    in the core, so a window of ~64 bins (~10-30 b-widths at typical Lyα-forest
    temperatures with the Sherwood velocity grid) covers the significant region
    while keeping the intermediate tensor at O(n_rays * n_src * (2*window+1))
    rather than O(n_rays * n_src * n_obs). The full-grid version segfaulted at
    n_obs=2048 with autograd retention; this version scales linearly in n_obs
    rather than quadratically.

    The constant prefactor (sigma_0 * cell_path_length * mean_n_H_norm) is
    absorbed into a single multiplicative amplitude `tau_amp`. To avoid the
    tau_amp <-> density rescaling degeneracy [D-10], the caller is responsible
    for adding a regularization term on `log(tau_amp)` to the loss.

    Args:
        mlp: IGMNeRF model.
        ray_points: (n_rays, n_bins, 3) — coordinates in unit cube [0, 1].
        vel_axis: (n_obs,) — simulation velocity grid (km/s). Source and
            observed bins share this grid (same simulation cell structure).
        tau_amp: optional scalar / nn.Parameter; defaults to 1.0.
        window: int — half-width of the convolution kernel in observation
            bins. Total kernel length is 2*window+1.
        z: redshift (informational; reserved for future Hubble-flow scaling).

    Returns:
        tau: (n_rays, n_obs) — full tau(v) profile per ray, suitable for per-bin
            MSE against the simulation ground-truth tauH1.
    """
    # [D-42] When `g` is provided, the MLP forward concatenates it onto the
    # encoded coordinate before layer 1. Default-OFF path (g=None) is
    # bit-identical to the pre-[D-42] forward.
    # [D-46] When `physics_id` is provided, the MLP forward concatenates the
    # learned physics embedding e_p onto the layer-1 input only. Default-OFF
    # path (physics_id=None) is bit-identical to the pre-[D-46] forward.
    fields = mlp(ray_points, g=g, physics_id=physics_id)  # (n_rays, n_bins, 4)
    density = fields[..., 0]              # rho / <rho>
    temp = fields[..., 1]                 # K
    h1_frac = fields[..., 2]              # X_HI in [0, 1]
    vpec = fields[..., 3]                 # km/s

    # Thermal Doppler width and damping parameter (per source bin)
    b = 12.85 * torch.sqrt(temp / 10000.0)   # km/s
    # a = Gamma * lambda / (4 * pi * b); 6.063e-3 km/s = (6.2649e8 s^-1)(1.21567e-5 cm)/(4 pi)
    # Yields a = 4.72e-4 at T=1e4 K (b=12.85 km/s), matching standard tables (Tepper-Garcia 2006).
    # The prior coefficient 4.7e-4 produced a 12.9x under-estimate; audited + fixed [D-57], 2026-05-16.
    a = 6.063e-3 / b                          # dimensionless

    # n_HI proxy: rho/<rho> * X_HI (mean column n_bar_H absorbed into tau_amp)
    n_hi = density * h1_frac                  # (n_rays, n_src)

    n_obs = vel_axis.shape[0]
    n_rays, n_src = density.shape
    device = vel_axis.device
    dtype = density.dtype

    # Uniform velocity-grid spacing
    dv_per_bin = (vel_axis[-1] - vel_axis[0]) / (n_obs - 1)

    # Source-frame line center: simulation v_grid + peculiar velocity
    v_source = vel_axis[None, :] + vpec       # (n_rays, n_src)

    # Center observation-bin index per source line
    center_idx = ((v_source - vel_axis[0]) / dv_per_bin).long()      # (n_rays, n_src)

    # Window indices around each center: shape (1, 1, 2W+1)
    offsets = torch.arange(-window, window + 1, device=device)
    obs_idx = center_idx[..., None] + offsets[None, None, :]         # (n_rays, n_src, K)
    valid_mask = (obs_idx >= 0) & (obs_idx < n_obs)
    obs_idx_safe = obs_idx.clamp(0, n_obs - 1)

    # Gather observed velocities along the window
    v_obs_window = vel_axis[obs_idx_safe]                            # (n_rays, n_src, K)

    # Voigt argument and profile
    dv_window = v_obs_window - v_source[..., None]                   # (n_rays, n_src, K)
    x = dv_window / b[..., None]
    H = tepper_garcia_voigt(a[..., None], x)                          # (n_rays, n_src, K)

    # Mask out-of-range positions so they contribute zero
    H = H * valid_mask.to(dtype)

    # Normalized line profile phi(v) = H / (b * sqrt(pi))
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))
    contrib = (n_hi[..., None] * H) / (b[..., None] * sqrt_pi)        # (n_rays, n_src, K)

    # Scatter-add into the observed-velocity grid
    tau = torch.zeros((n_rays, n_obs), dtype=dtype, device=device)
    tau.scatter_add_(
        1,
        obs_idx_safe.reshape(n_rays, -1),
        contrib.reshape(n_rays, -1),
    )

    if tau_amp is not None:
        tau = tau * tau_amp

    if return_tau_local:
        # [D-41] FGPA-tail regularizer per-source-bin probe. The "local"
        # optical depth is the column-source amplitude that would appear
        # if the line profile were a delta function: integrated under the
        # normalized line profile, this is exactly n_hi * dv / (b * sqrt(pi)),
        # i.e., the per-source-bin contribution stripped of Voigt convolution
        # mixing. Multiplied by tau_amp so the FGPA residual is comparable
        # to truth-side tau_truth_local at the same tau_amp. We also surface
        # density and temp at the source-bin grid so the caller can evaluate
        # log(tau_local) - beta*log(density) - gamma*log(temp) - C without
        # re-running the MLP forward pass.
        amp_scalar = tau_amp if tau_amp is not None else 1.0
        tau_local = amp_scalar * n_hi * dv_per_bin / (b * sqrt_pi)        # (n_rays, n_src)
        fgpa_fields = {
            "tau_local": tau_local,
            "density": density,
            "temp": temp,
        }
        return tau, fgpa_fields

    return tau

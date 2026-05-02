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

class IGMNeRF(nn.Module):
    """
    Continuous MLP mapping 3D position -> density, temp, h1_frac, vpec.
    """
    def __init__(self, hidden_dim=256, num_layers=8, L=10):
        super().__init__()
        self.encoding = PositionalEncoding(L)
        in_dim = 3 + 2 * 3 * L
        
        self.layers1 = nn.ModuleList()
        for i in range(4):
            dim = in_dim if i == 0 else hidden_dim
            self.layers1.append(nn.Linear(dim, hidden_dim))
            
        self.layers2 = nn.ModuleList()
        for i in range(num_layers - 4):
            dim = hidden_dim + in_dim if i == 0 else hidden_dim
            self.layers2.append(nn.Linear(dim, hidden_dim))

        self.relu = nn.ReLU()
        
        # Output layer input dimension depends on whether we have layers after the skip connection
        out_in_dim = hidden_dim if (num_layers - 4) > 0 else (hidden_dim + in_dim)
        self.out_layer = nn.Linear(out_in_dim, 4)
        
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoded = self.encoding(x)
        h = encoded
        
        for layer in self.layers1:
            h = self.relu(layer(h))
            
        # NeRF residual/skip connection
        h = torch.cat([h, encoded], dim=-1)
        
        for layer in self.layers2:
            h = self.relu(layer(h))
            
        out = self.out_layer(h)
        # Bounding fields to physical ranges
        density = self.softplus(out[..., 0])  # rho/rho_bar >= 0
        temp = self.softplus(out[..., 1]) * 10**4 + 10**3 # T ~ 10^3 to 10^6 K
        h1_frac = self.sigmoid(out[..., 2])   # 0 to 1
        vpec = torch.tanh(out[..., 3]) * 500  # Peculiar velocity +/- 500 km/s
        
        return torch.stack([density, temp, h1_frac, vpec], dim=-1)

def tepper_garcia_voigt(a, x):
    """
    Tepper-García (2006) analytic approximation to the Voigt-Hjerting function H(a, x).
    Highly efficient and differentiable for absorption profile modelling.
    """
    x2 = x**2
    # Handling small x to avoid division by zero in (1.5 * x^-2)
    # Using a small epsilon or clamping for the polynomial wings
    x2_safe = torch.clamp(x2, min=1e-5)
    
    # H(a, x) ~ exp(-x^2) - (a / (sqrt(pi) * x^2)) * [exp(-2x^2)*(4x^4 + 7x^2 + 4 + 1.5x^-2) - 1.5x^-2 - 1]
    exp_x2 = torch.exp(-x2)
    exp_2x2 = torch.exp(-2 * x2)
    
    poly = 4 * x2_safe**2 + 7 * x2_safe + 4 + 1.5 / x2_safe
    term1 = exp_2x2 * poly
    term2 = 1.5 / x2_safe + 1
    
    h = exp_x2 - (a / (torch.sqrt(torch.tensor(torch.pi)) * x2_safe)) * (term1 - term2)
    return torch.clamp(h, min=0.0)

def volume_render_physics(mlp, ray_points, vel_axis, tau_amp=None, window=64, z=0.3):
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
    fields = mlp(ray_points)              # (n_rays, n_bins, 4)
    density = fields[..., 0]              # rho / <rho>
    temp = fields[..., 1]                 # K
    h1_frac = fields[..., 2]              # X_HI in [0, 1]
    vpec = fields[..., 3]                 # km/s

    # Thermal Doppler width and damping parameter (per source bin)
    b = 12.85 * torch.sqrt(temp / 10000.0)   # km/s
    a = 4.7e-4 / b                            # dimensionless

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

    return tau

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

def volume_render_physics(mlp, ray_points, z=0.3):
    """
    Physically consistent differentiable volume rendering.
    Maps 3D fields to 1D Lyman-Alpha optical depth (tau).
    """
    fields = mlp(ray_points) # (batch, n_bins, 4)
    density = fields[..., 0]
    temp = fields[..., 1]
    h1_frac = fields[..., 2]
    vpec = fields[..., 3]
    
    # Constants (Astrophysical units)
    # sigma_alpha ~ 4.45e-18 cm^2 * (lambda_alpha / 1215A) ...
    # Simplified cross-section scaling for NeRF learning
    # b-parameter: b = sqrt(2kT/m_H) ~ 12.8 km/s * sqrt(T/10^4 K)
    b = 12.85 * torch.sqrt(temp / 10000.0)
    
    # x = (v - v_center) / b
    # In one-to-one mapping (no RSD for initial validation), x=0 means we are at the bin center
    # For full RSD, we'd convolve bins. For Stage 2a, we evaluate the kernel at the center.
    x = vpec / b 
    
    # a: damping parameter (Small for Lyman-alpha forest)
    a = 4.7e-4 / b 
    
    h_ax = tepper_garcia_voigt(a, x)
    
    # Local optical depth tau_local ~ n_HI * sigma(v)
    # n_HI ~ density * h1_frac
    tau_local = density * h1_frac * h_ax
    
    # Integrate along the ray
    tau_rendered = torch.sum(tau_local, dim=-1)
    return tau_rendered

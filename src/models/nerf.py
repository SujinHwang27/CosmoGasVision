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
        self.out_layer = nn.Linear(hidden_dim, 4)
        
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
        density = self.softplus(out[..., 0])
        temp = self.softplus(out[..., 1])
        h1_frac = self.sigmoid(out[..., 2])
        vpec = out[..., 3]
        
        return torch.stack([density, temp, h1_frac, vpec], dim=-1)

def volume_render_dummy(mlp, ray_points):
    """
    Differentiable step proxy for volume rendering optical depth (tau).
    In true rendering, tau depends non-linearly on temp and vpec via Voigt profiles.
    For Stage 2-a dummy testing, we treat tau purely as density integrated along the ray bin.
    """
    fields = mlp(ray_points)
    density = fields[..., 0] 
    
    # Simple discrete sum along the ray bins proxying ray-integration
    tau_rendered = density.sum(dim=-1)
    return tau_rendered

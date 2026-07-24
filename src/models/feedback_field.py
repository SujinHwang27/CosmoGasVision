"""SIREN auto-decoder conditioned field for the exp/feedback-latent track.

[F-02] §4(a) Stage-1 architecture of record. This is the banked next-candidate
trunk per [F-01]: a SIREN (Sitzmann+2020) periodic coordinate network, EXPLICITLY
NOT the falsified ReLU+PE trunk of exp/nerf ([D-60..71]/[D-73] Mode-B collapse).

Design (concrete, from f02_founding_ratification.md §4(a)):
  - Trunk: SIREN, 5 sine layers of width 256, omega_0 = 30. Sitzmann+2020 init
    EXACTLY: first layer weights ~ U(-1/n_in, 1/n_in); hidden layers ~
    U(-sqrt(6/n_in)/omega_0, +sqrt(6/n_in)/omega_0); sine activation with the
    omega_0 scaling. Input dim = 3 (coords in [-1, 1], SIREN convention) + d
    (physics-vector dim). Single scalar output x = log10-overdensity, NO output
    activation (x spans ~[-3, +3.6]).
  - Auto-decoder physics vectors (DeepSDF, Park+2019): a bank of 4 learnable
    vectors z_p in R^d, init N(0, 0.01^2), optimized JOINTLY with the trunk.
    Concatenation-at-input wiring. Forward takes (coords[B,3], variant_idx[B]),
    looks up z_p, concatenates, returns x[B]. An explicit-z forward path is also
    exposed (for the c2 swap-test and R-C).
  - Code z-prior: an L2 penalty on the ACTIVE z_p vectors, weight lambda_z.
    DERIVATION-AT-SPEC-TIME (binding on this track from birth): lambda_z is NOT
    baked with a magic default here. The driver tunes it; the coordinator/PI pins
    lambda_z in the Stage-1 exit artifact at the value where anti-collapse control
    c1 separates while fit-quality Q does not degrade (§4(a), §4(e)). Callers MUST
    pass lambda_z explicitly to `code_prior`.

Param target: ~0.26 M for the 5x256 SIREN (+ negligible 4*d embedding params).

Fully torch.autograd-compatible: gradients reach BOTH the trunk weights AND z_p.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """A single SIREN layer: Linear -> sin(omega_0 * .).

    Sitzmann+2020 init (exact):
      - is_first: weight ~ U(-1/in_features, 1/in_features)
      - else:     weight ~ U(-sqrt(6/in_features)/omega_0, +sqrt(6/in_features)/omega_0)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class FeedbackField(nn.Module):
    """SIREN auto-decoder conditioned IGM field ([F-02] Stage-1).

    Forward maps (coords in [-1,1]^3, variant index) -> x = log10-overdensity.

    Args:
        d: physics-vector (code) dimensionality. Default 8; Stage-1 also
            instantiates d=1 for the R-A fit-quality ladder.
        hidden_dim: SIREN width (default 256).
        num_hidden_layers: number of sine layers of width `hidden_dim`
            (default 5 = first-input layer + 4 inner 256->256 layers). The
            output is a separate linear layer with NO activation.
        n_variants: number of feedback variants in the auto-decoder bank
            (default 4: P1..P4 of the Sherwood feedback suite).
        omega_0: SIREN frequency (default 30, Sitzmann+2020).
        code_init_std: std of the N(0, .) init on z_p (default 0.01, DeepSDF).
    """

    def __init__(self, d: int = 8, hidden_dim: int = 256,
                 num_hidden_layers: int = 5, n_variants: int = 4,
                 omega_0: float = 30.0, code_init_std: float = 0.01):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_variants = n_variants
        self.omega_0 = omega_0
        self.code_init_std = code_init_std
        self.in_dim = 3 + d

        # --- DeepSDF auto-decoder code bank: 4 learnable vectors z_p in R^d ---
        self.codes = nn.Embedding(n_variants, d)
        with torch.no_grad():
            self.codes.weight.normal_(0.0, code_init_std)

        # --- SIREN trunk: first sine layer + (num_hidden_layers-1) inner ---
        layers = [SineLayer(self.in_dim, hidden_dim, is_first=True, omega_0=omega_0)]
        for _ in range(num_hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False,
                                    omega_0=omega_0))
        self.net = nn.ModuleList(layers)

        # --- Output: linear, NO activation (x = log10-overdensity, unbounded) ---
        # Sitzmann final-layer init: U(-sqrt(6/n)/omega_0, +sqrt(6/n)/omega_0).
        self.out_layer = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.out_layer.weight.uniform_(-bound, bound)

    # ------------------------------------------------------------------ #
    def _trunk(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate code at input and run the SIREN trunk -> x[B]."""
        h = torch.cat([coords, z], dim=-1)          # (B, 3+d)
        for layer in self.net:
            h = layer(h)
        out = self.out_layer(h)                     # (B, 1)
        return out.squeeze(-1)                       # (B,)

    def forward(self, coords: torch.Tensor,
                variant_idx: Optional[torch.Tensor] = None,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(coords[B,3], variant_idx[B]) -> x[B] = log10-overdensity.

        Exactly one of `variant_idx` (embedding lookup) or `z` (explicit code)
        must be supplied. The explicit-z path is for the c2 swap-test and R-C.

        `z` may be (B, d) or (d,); a (d,) vector is broadcast across the batch.
        """
        if (variant_idx is None) == (z is None):
            raise ValueError(
                "Pass exactly one of variant_idx (embedding path) or z "
                "(explicit-code path)."
            )
        B = coords.shape[0]
        if z is None:
            z = self.codes(variant_idx)             # (B, d)
        else:
            if z.dim() == 1:
                z = z.unsqueeze(0).expand(B, -1)     # broadcast (d,) -> (B, d)
        return self._trunk(coords, z)

    # ------------------------------------------------------------------ #
    def code_prior(self, variant_idx: torch.Tensor, lambda_z: float) -> torch.Tensor:
        """DeepSDF-style L2 code prior on the ACTIVE z_p vectors.

        Returns lambda_z * mean_b ||z_{p(b)}||^2 over the batch's active codes.

        DERIVATION-AT-SPEC-TIME: lambda_z is a REQUIRED argument. It is NOT
        baked with a magic default anywhere in this module. The driver tunes it
        and the coordinator/PI pins it in the Stage-1 exit artifact at the value
        where anti-collapse control c1 separates while Q does not degrade.
        """
        z = self.codes(variant_idx)                 # (B, d)
        return lambda_z * (z.pow(2).sum(dim=-1)).mean()

    def code_prior_explicit(self, z: torch.Tensor, lambda_z: float) -> torch.Tensor:
        """L2 code prior on an explicit z tensor (B,d) or (d,)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return lambda_z * (z.pow(2).sum(dim=-1)).mean()

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Total parameter count (trunk + output + code bank)."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

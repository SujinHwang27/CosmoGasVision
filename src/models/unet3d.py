"""[U-06] Stage-2 A1: 3D U-Net for amortized IGM inversion (spec (a)).

Spec of record: ``experiments/unet-inversion/design/u06_stage2_spec.md``
S(a), commit 489a0d3. Rulings honored:

* 4 levels (3 downsamplings), base 32 channels doubling to 256 at the
  bottleneck: encoder channels (32, 64, 128, 256).
* Per level: 2 x [Conv3d 3^3 -> GroupNorm(8) -> SiLU]. The stride-2
  downsampling conv IS the first conv of encoder levels 2-4 ("down =
  stride-2 conv"); level 1 is stride 1 throughout.
* Up path: nearest-upsample x2 + 3^3 conv (= decoder conv #1, halves
  channels), skip concatenation, then 3^3 conv #2 (2*out -> out).
* Head: 1x1x1 conv to 1 channel, NO output activation (x spans ~[-3, 3.6]).
* Input (2, 64^3): ch0 = 12.5*delta_F on ray voxels, ch1 binary ray mask.
* GroupNorm(8) not BatchNorm (batch <= 4 locally).

Parameter budget: spec estimate ~5.5 M within the 5-25 M envelope. This
realization counts 5,839,713 (asserted at construction: within 10% of the
5.5 M estimate AND inside the envelope; exact count recorded by callers).
"""

from __future__ import annotations

import torch
import torch.nn as nn

PARAM_ESTIMATE = 5.5e6          # spec S(a) budget of record
PARAM_ENVELOPE = (5.0e6, 25.0e6)  # pre-registered envelope
GN_GROUPS = 8


def _cgs(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    """Conv3d 3^3 -> GroupNorm(8) -> SiLU (spec unit block)."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1),
        nn.GroupNorm(GN_GROUPS, c_out),
        nn.SiLU(),
    )


class UNet3D(nn.Module):
    """4-level 3D U-Net, base 32, bottleneck 256, 1-channel x-field head."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)
        # encoder: level = [conv#1 (stride 2 for levels 2-4), conv#2]
        self.enc1 = nn.Sequential(_cgs(in_channels, c1), _cgs(c1, c1))
        self.enc2 = nn.Sequential(_cgs(c1, c2, stride=2), _cgs(c2, c2))
        self.enc3 = nn.Sequential(_cgs(c2, c3, stride=2), _cgs(c3, c3))
        self.enc4 = nn.Sequential(_cgs(c3, c4, stride=2), _cgs(c4, c4))
        # decoder: nearest-up + conv#1 (in -> out), concat skip, conv#2
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3a, self.dec3b = _cgs(c4, c3), _cgs(2 * c3, c3)
        self.dec2a, self.dec2b = _cgs(c3, c2), _cgs(2 * c2, c2)
        self.dec1a, self.dec1b = _cgs(c2, c1), _cgs(2 * c1, c1)
        self.head = nn.Conv3d(c1, 1, kernel_size=1)  # no activation

        n = self.n_parameters()
        lo, hi = PARAM_ENVELOPE
        assert lo <= n <= hi, f"param count {n} outside envelope {PARAM_ENVELOPE}"
        if base_channels == 32:
            assert abs(n - PARAM_ESTIMATE) / PARAM_ESTIMATE <= 0.10, (
                f"param count {n} deviates >10% from spec estimate "
                f"{PARAM_ESTIMATE:.1e}"
            )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)                     # (c1, 64^3)
        s2 = self.enc2(s1)                    # (c2, 32^3)
        s3 = self.enc3(s2)                    # (c3, 16^3)
        b = self.enc4(s3)                     # (c4,  8^3)
        d3 = self.dec3b(torch.cat([self.dec3a(self.up(b)), s3], dim=1))
        d2 = self.dec2b(torch.cat([self.dec2a(self.up(d3)), s2], dim=1))
        d1 = self.dec1b(torch.cat([self.dec1a(self.up(d2)), s1], dim=1))
        return self.head(d1)

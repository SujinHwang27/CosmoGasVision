"""3D ResNet-18 classifier for the sprint-4 [D-51] truth-baseline.

A standard ResNet-18 3D architecture (BasicBlock x {2,2,2,2}) operating
on cubic single-channel volumetric inputs of shape ``(B, 1, D, H, W)``.
Channel widths are halved from video-defaults to fit the cubic crop
regime; full configuration documented in
``experiments/nerf/design/sprint4_truth_baseline.md`` \xa73.

Purpose: **measurement instrument** for the [D-47] option-C step-1
empirical ceiling Â_truth(r). NOT a scientific contribution — verbs in
any usage downstream must reflect that (see design doc \xa710 / [D-37]-ext
rule 2).

Standard usage:

    >>> from src.models.cnn3d import resnet18_3d_4class
    >>> net = resnet18_3d_4class(in_channels=1, num_classes=4)
    >>> x = torch.randn(8, 1, 32, 32, 32)
    >>> logits = net(x)                # shape (8, 4)
    >>> # 4-way CE on physics_id labels 0..3
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ----------------------------------------------------------------- block

class BasicBlock3D(nn.Module):
    """ResNet BasicBlock adapted to 3D (Conv3d/BN3d/ReLU).

    Two 3x3x3 conv layers with BN + ReLU, plus an identity (or 1x1x1
    projection) shortcut. The first conv handles the stride; the
    projection shortcut is created when stride > 1 or when in/out
    channels differ.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm3d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out


# --------------------------------------------------------------- resnet

class ResNet3D(nn.Module):
    """ResNet-style 3D classifier with configurable block counts and
    channel widths. The sprint-4 default is ``resnet18_3d_4class``.

    Args:
        block_counts: number of BasicBlock3D modules at each of the 4
            stages. ResNet-18 = (2, 2, 2, 2).
        channels: channel width at each of the 4 stages. The sprint-4
            halved-from-video defaults are (32, 64, 128, 256), giving
            ~10-12M params at crop_size=32.
        stem_channels: width of the conv-7-stride-2 stem (defaults to
            first-stage channel count).
        in_channels: number of input channels. Sprint-4 = 1 (single
            scalar overdensity field).
        num_classes: number of output logits. Sprint-4 = 4 (physics_id
            P1..P4).
        stem_stride: stride of the stem conv. Default 2 = downsample
            by 2 immediately, matching the design doc table.
        stem_kernel_size: kernel size of the stem conv. Default 7,
            standard ResNet stem.
    """

    def __init__(
        self,
        block_counts: tuple[int, int, int, int] = (2, 2, 2, 2),
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        stem_channels: Optional[int] = None,
        in_channels: int = 1,
        num_classes: int = 4,
        stem_stride: int = 2,
        stem_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if stem_channels is None:
            stem_channels = channels[0]

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, stem_channels,
                kernel_size=stem_kernel_size, stride=stem_stride,
                padding=stem_kernel_size // 2, bias=False,
            ),
            nn.BatchNorm3d(stem_channels),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(stem_channels, channels[0], block_counts[0], stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], block_counts[1], stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], block_counts[2], stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], block_counts[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(channels[3] * BasicBlock3D.expansion, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int,
        num_blocks: int, stride: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            BasicBlock3D(in_channels, out_channels, stride=stride)
        ]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock3D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(B, in_channels, D, H, W)`` -> logits ``(B, num_classes)``."""
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.avg_pool(h).flatten(1)
        return self.fc(h)


def resnet18_3d_4class(in_channels: int = 1, num_classes: int = 4) -> ResNet3D:
    """Sprint-4 default: ResNet-18 3D with halved channel widths for the
    4-class physics_id task on cubic single-channel overdensity crops.
    Returns a model with ~10-12M parameters at the design-doc spec.
    """
    return ResNet3D(
        block_counts=(2, 2, 2, 2),
        channels=(32, 64, 128, 256),
        stem_channels=32,
        in_channels=in_channels,
        num_classes=num_classes,
        stem_stride=2,
        stem_kernel_size=7,
    )


# ---------------------------------------------------- trivial baselines


class MeanOverdensityBaseline(nn.Module):
    """Gate (e₁) trivial baseline: 1-scalar ``crop.mean()`` -> FC(num_classes).

    If this baseline achieves Â_overall within 5 pp of the 3D ResNet, the
    classification task is dominated by mean overdensity and the 3D
    ResNet is decorative — see design doc \xa78 gate (e₁) + \xa79 AD-3.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(1, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(B, 1, D, H, W)`` -> logits ``(B, num_classes)``."""
        # Spatial mean over all voxels per batch item (single scalar feature).
        feat = x.flatten(2).mean(dim=2)  # shape (B, 1)
        return self.fc(feat)


class MeanVarianceBaseline(nn.Module):
    """Gate (e₂) trivial baseline: 2-scalar ``[crop.mean(), crop.var()]``
    -> FC(num_classes). Catches "task dominated by low-order moments
    beyond just the mean" — see design doc \xa78 gate (e₂) + \xa79 AD-3.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(2, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(B, 1, D, H, W)`` -> logits ``(B, num_classes)``."""
        flat = x.flatten(2)  # (B, 1, D*H*W)
        mean = flat.mean(dim=2)              # (B, 1)
        var = flat.var(dim=2, unbiased=False)  # (B, 1)
        feat = torch.cat([mean, var], dim=1)  # (B, 2)
        return self.fc(feat)

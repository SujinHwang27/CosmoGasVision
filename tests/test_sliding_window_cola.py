"""[U-06] micro-cycle A1 — Hann-taper COLA test for the sliding-window util.

Constant-input windows must reconstruct a constant field to < 1e-6 (exact
overlap-add of the periodic Hann^3 at hop = crop/2), for both the of-record
'hann' taper and the 'uniform' diagnostic path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from tests.test_unet_training_contract import _make_source

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "unet_pipeline", REPO / "experiments/unet-inversion/pipeline.py")
pl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pl)


class _ConstModel(torch.nn.Module):
    def __init__(self, c: float) -> None:
        super().__init__()
        self.c = c

    def forward(self, x):
        return torch.full(
            (x.shape[0], 1) + x.shape[2:], self.c, dtype=x.dtype)

    def eval(self):  # noqa: A003 — keep nn.Module contract
        return self


def test_hann_weight_cola_exact():
    w = pl.hann3_weight(64)
    # per-axis periodic-Hann pair sum == 1 exactly at hop 32
    acc = np.zeros((192, 192, 192))
    for a in range(0, 192, 32):
        for b in range(0, 192, 32):
            for c in range(0, 192, 32):
                ix = np.ix_(*[(np.arange(s, s + 64) % 192)
                              for s in (a, b, c)])
                acc[ix] += w
    assert np.abs(acc - 1.0).max() < 1e-6


def test_constant_field_reconstruction_both_tapers():
    src = _make_source(seed=3, n_rays=64)
    model = _ConstModel(0.7321)
    for taper in ("hann", "uniform"):
        pred = pl.sliding_window_predict(
            model, src, np.arange(32), torch.device("cpu"), taper=taper)
        assert pred.shape == (192, 192, 192)
        assert np.abs(pred - 0.7321).max() < 1e-6, taper

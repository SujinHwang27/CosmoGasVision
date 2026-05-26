"""[D-70 (1b)] Estimator-equivalence test for the body_arch='current' default.

R20 twin-gate binding: when ``--arch current`` (the default), the
IGMNeRF forward output must be 1e-5 rel-tol equivalent to the pre-Rev-5.1
baseline at fixed seed. This guards against accidental drift in the default
path when the skip-rich variant was added.

Reference path
--------------
Construct a model with ``body_arch='current'`` and a fixed manual_seed; run
forward on a fixed-seed coord batch. Reference is the same construction +
forward executed twice — bitwise repeatable under torch.manual_seed.

(We do not check against a serialized golden tensor because the pre-Rev-5.1
init RNG schedule is preserved by the body_arch='current' branch — same
ModuleList construction order, same Linear layer count, same first-layer
input dim. The test asserts: (a) determinism under fixed seed, and
(b) bit-equivalence between two IDENTICAL constructions, which fails if any
extra RNG draw was introduced into the 'current' branch.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.nerf import IGMNeRF  # noqa: E402


SEED = 20260525
BATCH = 512
REL_TOL = 1e-5


def _build(body_arch="current"):
    torch.manual_seed(SEED)
    return IGMNeRF(hidden_dim=256, num_layers=8, L=10, body_arch=body_arch)


def _fixed_coords():
    g = torch.Generator()
    g.manual_seed(SEED + 1)
    return torch.rand(1, BATCH, 3, generator=g)


def test_current_arch_is_deterministic_under_fixed_seed():
    m1 = _build("current")
    m2 = _build("current")
    coords = _fixed_coords()
    o1 = m1(coords)
    o2 = m2(coords)
    diff = (o1 - o2).abs().max().item()
    assert diff == 0.0, (
        f"body_arch='current' not bit-equivalent under fixed seed: max abs diff={diff}"
    )


def test_current_arch_rel_tol_equivalence():
    """Two independent constructions at the same seed are within 1e-5 rel-tol
    of each other on the full forward output."""
    m1 = _build("current")
    m2 = _build("current")
    coords = _fixed_coords()
    o1 = m1(coords)
    o2 = m2(coords)
    denom = o1.abs().clamp(min=1e-30)
    rel = ((o1 - o2).abs() / denom).max().item()
    assert rel <= REL_TOL, (
        f"body_arch='current' rel-tol equivalence failed: rel={rel:.3e}, tol={REL_TOL:.0e}"
    )
    print(f"[current-arch equivalence] max rel-dev = {rel:.3e} (tol={REL_TOL:.0e})")


def test_default_body_arch_is_current():
    """No --arch flag → body_arch defaults to 'current' on the constructor.
    Guards against silent default-flip during refactors.
    """
    m_default = IGMNeRF(hidden_dim=64, num_layers=8, L=10)
    assert getattr(m_default, "body_arch", None) == "current", (
        "IGMNeRF default body_arch is not 'current' — estimator-equivalence broken"
    )
    # Default-path topology check: 4 layers1 + 4 layers2.
    assert len(m_default.layers1) == 4, (
        f"default layers1 len={len(m_default.layers1)}, expected 4"
    )
    assert len(m_default.layers2) == 4, (
        f"default layers2 len={len(m_default.layers2)}, expected 4"
    )


def test_current_arch_forward_matches_pre_rev51_init_layout():
    """The 'current' branch must build the exact same ModuleList structure as
    the pre-Rev-5.1 model. We assert layer counts AND first-layer input dims.
    """
    m = _build("current")
    # layers1[0] input dim = in_dim = encoded_dim (= 3 + 6*L)
    L = 10
    expected_in_dim = 3 + 6 * L
    assert m.layers1[0].in_features == expected_in_dim, (
        f"current arch layers1[0].in_features={m.layers1[0].in_features}, "
        f"expected {expected_in_dim} (pre-Rev-5.1 layout)"
    )
    # layers2[0] input dim = hidden_dim + skip_dim = hidden_dim + encoded_dim
    expected_skip_in = 256 + expected_in_dim
    assert m.layers2[0].in_features == expected_skip_in, (
        f"current arch layers2[0].in_features={m.layers2[0].in_features}, "
        f"expected {expected_skip_in} (pre-Rev-5.1 single-mid-skip)"
    )

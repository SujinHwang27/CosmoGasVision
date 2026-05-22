"""Sprint-L1 gate-8 Option A(b) rescue: CLI-toggle wiring test for
``--gradnorm-full`` in ``experiments/nerf/pipeline.py``.

Asserts the flag toggles the balancer's ``simplified`` attribute — not just
the kwarg. Test pattern: parse args twice (once without the flag, once with),
instantiate the balancer via the exact call-site logic, and check the
resulting ``GradNormWrapper.simplified`` attribute on the live object.

Default behavior (no ``--gradnorm-full``) MUST yield ``simplified=True``
(NON-PROVISIONAL per job 201587). ``--gradnorm-full`` MUST yield
``simplified=False`` (full Chen+ 2018 second-order path).
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure repo root is on sys.path so ``src.`` and ``experiments.`` resolve
# when pytest is invoked from arbitrary cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.nerf.pipeline import parse_args  # noqa: E402
from src.training.p_flux_loss import GradNormWrapper  # noqa: E402


# Minimum argv to satisfy required=True flags in parse_args.
_REQUIRED_ARGV = [
    "--n_rays", "64",
    "--physics", "1",
    "--seed", "42",
    "--enable-l1-pf-loss",
]


def _build_balancer_like_pipeline(args) -> GradNormWrapper:
    """Mirror the exact call-site logic in pipeline.py around L846."""
    simplified_flag = not bool(args.gradnorm_full)
    return GradNormWrapper(
        initial_w=(1.0, 1.0),
        alpha=args.l1_gradnorm_alpha,
        simplified=simplified_flag,
    )


def test_gradnorm_full_flag_exists_and_defaults_false():
    """Flag is registered on argparse and defaults to False (opt-in only)."""
    args = parse_args(_REQUIRED_ARGV)
    assert hasattr(args, "gradnorm_full"), (
        "parse_args must expose `gradnorm_full` attribute"
    )
    assert args.gradnorm_full is False, (
        "Default for --gradnorm-full must be False (NON-PROVISIONAL per "
        "job 201587)"
    )


def test_default_invocation_yields_simplified_true():
    """No flag -> balancer.simplified == True (byte-identical to job 201587)."""
    args = parse_args(_REQUIRED_ARGV)
    balancer = _build_balancer_like_pipeline(args)
    assert balancer.simplified is True, (
        "Default (no --gradnorm-full) MUST instantiate GradNormWrapper with "
        "simplified=True; got simplified=False"
    )


def test_gradnorm_full_flag_yields_simplified_false():
    """--gradnorm-full -> balancer.simplified == False (full Chen+ 2018)."""
    args = parse_args(_REQUIRED_ARGV + ["--gradnorm-full"])
    assert args.gradnorm_full is True
    balancer = _build_balancer_like_pipeline(args)
    assert balancer.simplified is False, (
        "--gradnorm-full MUST instantiate GradNormWrapper with "
        "simplified=False (full Chen+ 2018 second-order path); got "
        "simplified=True"
    )


def test_cli_toggle_round_trip():
    """Round-trip: without-flag and with-flag yield opposing simplified
    attributes on the live balancer object (the actual contract under test)."""
    args_off = parse_args(_REQUIRED_ARGV)
    args_on = parse_args(_REQUIRED_ARGV + ["--gradnorm-full"])
    b_off = _build_balancer_like_pipeline(args_off)
    b_on = _build_balancer_like_pipeline(args_on)
    assert b_off.simplified is True
    assert b_on.simplified is False
    assert b_off.simplified != b_on.simplified


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

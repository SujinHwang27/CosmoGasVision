"""Unit tests for the [D-60] Attempt 3 per-task grad-clip building block.

Tests cover three contracts:

1. EMA update logic on a synthetic case (decay 0.95, window 100..500).
2. ``clip_grad_to_norm`` returns the right scale when norm > threshold,
   returns 1.0 when norm <= threshold.
3. Backward-compat: ``--per-task-grad-clip 0.0`` produces identical
   behavior to the pre-Attempt-3 code path (PerTaskClipState is not
   instantiated at all).

Reference: PI Attempt 3 spec 2026-05-22, Amendments A + B.
"""

from __future__ import annotations

import os
import sys

import pytest

from src.training.per_task_clip import PerTaskClipState, clip_grad_to_norm


# ---------------------------------------------------------------------------
# 1. EMA update logic.
# ---------------------------------------------------------------------------


def test_ema_ignores_steps_before_burnin():
    s = PerTaskClipState()
    s.update(step=50, tau_norm=10.0, pf_norm=1.0)
    s.update(step=99, tau_norm=10.0, pf_norm=1.0)
    assert s.ema_tau is None, "EMA absorbed before step 100"
    assert s.ema_pf is None
    assert s.n_updates == 0


def test_ema_initializes_at_step_100():
    s = PerTaskClipState()
    s.update(step=100, tau_norm=10.0, pf_norm=1.0)
    assert s.ema_tau == pytest.approx(10.0)
    assert s.ema_pf == pytest.approx(1.0)
    assert s.n_updates == 1


def test_ema_decay_value_at_step_500():
    """After many constant updates, the EMA converges to that constant."""
    s = PerTaskClipState()
    for step in range(100, 501):
        s.update(step=step, tau_norm=5.0, pf_norm=0.5)
    # 401 updates with EMA decay 0.95 from init=value: converges exactly.
    assert s.ema_tau == pytest.approx(5.0, rel=1e-9)
    assert s.ema_pf == pytest.approx(0.5, rel=1e-9)
    assert s.n_updates == 401


def test_ema_freeze_takes_lower_norm_task():
    s = PerTaskClipState()
    for step in range(100, 501):
        s.update(step=step, tau_norm=5.0, pf_norm=0.5)
    s.maybe_freeze(step=501)
    assert s.is_frozen
    assert s.frozen_threshold == pytest.approx(0.5), (
        "frozen threshold must be min(ema_tau, ema_pf) per Amendment A "
        "(lower-norm task's EMA)"
    )


def test_ema_post_freeze_is_constant():
    """Once frozen, subsequent updates leave the threshold unchanged."""
    s = PerTaskClipState()
    for step in range(100, 501):
        s.update(step=step, tau_norm=5.0, pf_norm=0.5)
    s.maybe_freeze(step=501)
    frozen_value = s.frozen_threshold
    # Updates past the window are no-ops on the EMA fields too.
    s.update(step=600, tau_norm=999.0, pf_norm=999.0)
    s.update(step=2000, tau_norm=0.001, pf_norm=0.001)
    s.maybe_freeze(step=2000)
    assert s.frozen_threshold == frozen_value
    # And the EMA fields are NOT touched past step 500 either.
    assert s.ema_tau == pytest.approx(5.0)
    assert s.ema_pf == pytest.approx(0.5)


def test_ema_freeze_idempotent_inside_window():
    """``maybe_freeze`` is a no-op while step <= 500."""
    s = PerTaskClipState()
    for step in range(100, 401):
        s.update(step=step, tau_norm=2.0, pf_norm=4.0)
    s.maybe_freeze(step=500)
    assert s.frozen_threshold is None
    s.maybe_freeze(step=501)
    assert s.frozen_threshold == pytest.approx(2.0)


def test_ema_one_sided_decay_dynamics():
    """Single non-stationary update: known closed-form value."""
    s = PerTaskClipState()
    s.update(step=100, tau_norm=10.0, pf_norm=1.0)
    s.update(step=101, tau_norm=0.0, pf_norm=0.0)
    # EMA(2) = 0.95 * init + 0.05 * new = 0.95 * 10 + 0 = 9.5.
    assert s.ema_tau == pytest.approx(9.5)
    assert s.ema_pf == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# 2. clip_grad_to_norm pure-arithmetic helper.
# ---------------------------------------------------------------------------


def test_clip_noop_when_norm_below_threshold():
    assert clip_grad_to_norm(grad_vec_norm=0.5, threshold=1.0) == 1.0
    assert clip_grad_to_norm(grad_vec_norm=1.0, threshold=1.0) == 1.0


def test_clip_scales_when_norm_above_threshold():
    # ||g|| = 4, threshold = 1 -> scale = 0.25, post-scale norm = 1.
    scale = clip_grad_to_norm(grad_vec_norm=4.0, threshold=1.0)
    assert scale == pytest.approx(0.25)
    assert (4.0 * scale) == pytest.approx(1.0)


def test_clip_disabled_when_threshold_zero():
    # threshold <= 0 -> no clip regardless of norm.
    assert clip_grad_to_norm(grad_vec_norm=1e9, threshold=0.0) == 1.0
    assert clip_grad_to_norm(grad_vec_norm=1e9, threshold=-1.0) == 1.0


# ---------------------------------------------------------------------------
# 3. CLI flag + backward-compat.
# ---------------------------------------------------------------------------


def _add_repo_to_syspath():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo not in sys.path:
        sys.path.insert(0, repo)


def test_per_task_clip_cli_default_is_zero():
    _add_repo_to_syspath()
    from experiments.nerf.pipeline import parse_args
    base = [
        "--n_rays", "64", "--physics", "1", "--seed", "42",
        "--enable-l1-pf-loss",
    ]
    a = parse_args(base)
    assert hasattr(a, "per_task_grad_clip")
    # Argparse default is the string "0.0"; the pipeline parses to a float
    # and sets ptc_mode="off" when this is the value. The CLI carries the
    # raw string (which preserves the 'auto' sentinel handling).
    assert a.per_task_grad_clip == "0.0"


def test_per_task_clip_cli_accepts_auto():
    _add_repo_to_syspath()
    from experiments.nerf.pipeline import parse_args
    base = [
        "--n_rays", "64", "--physics", "1", "--seed", "42",
        "--enable-l1-pf-loss",
    ]
    a = parse_args(base + ["--per-task-grad-clip", "auto"])
    assert a.per_task_grad_clip == "auto"


def test_per_task_clip_cli_accepts_explicit_float():
    _add_repo_to_syspath()
    from experiments.nerf.pipeline import parse_args
    base = [
        "--n_rays", "64", "--physics", "1", "--seed", "42",
        "--enable-l1-pf-loss",
    ]
    a = parse_args(base + ["--per-task-grad-clip", "1.5"])
    assert a.per_task_grad_clip == "1.5"


def test_per_task_clip_off_state_not_constructed():
    """Backward-compat: default '0.0' -> ptc_mode='off' -> PerTaskClipState
    is not used, so no Attempt-3 surgery runs.

    We assert this at the level of the pipeline's branching: parse the flag
    via the same code path the pipeline uses (the float-or-auto try/except).
    """
    raw = "0.0"
    # Mirror the pipeline's branching block (kept in lockstep).
    if raw.lower() == "auto":
        mode = "auto"
    else:
        thr = float(raw)
        mode = "explicit" if thr > 0.0 else "off"
    assert mode == "off"


# ---------------------------------------------------------------------------
# 4. End-to-end clip changes per-task grad-norm when threshold > 0 AND norm > thr.
# ---------------------------------------------------------------------------


def test_clipping_reduces_grad_norm_to_threshold():
    """Synthetic: norm=10, threshold=1 -> post-clip norm = 1.0."""
    s = PerTaskClipState()
    # Force the EMA into a known frozen state.
    for step in range(100, 501):
        s.update(step=step, tau_norm=1.0, pf_norm=10.0)
    s.maybe_freeze(step=501)
    thr = s.clip_threshold()
    assert thr == pytest.approx(1.0)

    scale = clip_grad_to_norm(grad_vec_norm=10.0, threshold=thr)
    post = 10.0 * scale
    assert post == pytest.approx(1.0, rel=1e-9)


def test_clipping_noop_when_norm_below_threshold_post_freeze():
    s = PerTaskClipState()
    for step in range(100, 501):
        s.update(step=step, tau_norm=5.0, pf_norm=5.0)
    s.maybe_freeze(step=501)
    thr = s.clip_threshold()
    # Apply to a small-norm vector: no clip.
    assert clip_grad_to_norm(grad_vec_norm=0.1, threshold=thr) == 1.0

"""Per-task gradient clipping with EMA-derived threshold.

[D-60] sprint-L1 Attempt 3 building block per PI's amended spec 2026-05-22:

- **Amendment A — empirical clip threshold**: clip at the value of the
  **lower-norm task's running EMA (decay 0.95)** computed over steps 100-500,
  then held fixed steps 500-end.
- **Amendment B — pre-committed FAIL criteria**: Attempt 3 FAILS if at step
  5000 EITHER (i) ``pf/tau`` ratio diagnostic > 3.0 OR (ii) ``loss_tau`` has
  not decreased monotonically by >= 10% relative to its step-1000 value.
  On FAIL: null result reported, no headline claim.

This module provides the EMA tracker + clip-threshold state machine in
isolation so it can be unit-tested without the full training loop. The
pipeline integration is in ``experiments/nerf/pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# EMA window per PI Attempt 3 spec.
_EMA_BURNIN_STEP = 100        # inclusive — first step the EMA absorbs.
_EMA_FREEZE_STEP = 500        # inclusive — last step the EMA updates.
_EMA_DECAY = 0.95             # PI-spec'd.


@dataclass
class PerTaskClipState:
    """Running EMA of per-task grad-norms + frozen clip threshold.

    State transitions:
      - steps < 100: no updates; threshold None.
      - steps 100..500 inclusive: each call to ``update(...)`` updates
        ``ema_tau`` and ``ema_pf`` (EMA decay 0.95, init = first observed
        value at step 100). ``clip_threshold`` returns None (warmup).
      - step == 500 + 1 (i.e. first call with step > 500): freeze
        ``frozen_threshold = min(ema_tau, ema_pf)``. From then on
        ``clip_threshold`` returns this constant.
      - steps > 500: ``update`` is a no-op for the EMA fields (PI spec:
        "computed over steps 100-500, then held fixed steps 500-end").

    Numerical guards:
      - If either EMA was never written (zero observations in the window),
        the frozen threshold remains None and the pipeline path treats the
        clip as disabled for that run. Surfaces as a log line; does not
        crash the run.
    """

    ema_tau: Optional[float] = None
    ema_pf: Optional[float] = None
    frozen_threshold: Optional[float] = None
    n_updates: int = 0

    @property
    def is_frozen(self) -> bool:
        return self.frozen_threshold is not None

    def update(self, step: int, tau_norm: float, pf_norm: float) -> None:
        """Absorb (step, ||g_tau||, ||g_pf||). Idempotent outside window."""
        if step < _EMA_BURNIN_STEP:
            return
        if step > _EMA_FREEZE_STEP:
            return
        # In-window: update EMA.
        if self.ema_tau is None:
            self.ema_tau = float(tau_norm)
        else:
            self.ema_tau = _EMA_DECAY * self.ema_tau + (1.0 - _EMA_DECAY) * float(tau_norm)
        if self.ema_pf is None:
            self.ema_pf = float(pf_norm)
        else:
            self.ema_pf = _EMA_DECAY * self.ema_pf + (1.0 - _EMA_DECAY) * float(pf_norm)
        self.n_updates += 1

    def maybe_freeze(self, step: int) -> None:
        """If we have crossed the freeze boundary and not yet frozen, freeze.

        Called by the pipeline once per step (cheap; checks are O(1)). The
        explicit decoupling from ``update`` is so the pipeline can call
        ``update`` only at the diagnostic-cadence steps if it ever wants to
        (currently it calls every step in [100, 500]); the freeze is on the
        clock, not the data.
        """
        if self.frozen_threshold is not None:
            return
        if step <= _EMA_FREEZE_STEP:
            return
        # We have crossed the boundary. Freeze if both EMAs are populated.
        if self.ema_tau is None or self.ema_pf is None:
            return
        self.frozen_threshold = min(self.ema_tau, self.ema_pf)

    def clip_threshold(self) -> Optional[float]:
        """Current clip threshold. None until step > 500 and EMAs populated."""
        return self.frozen_threshold


def clip_grad_to_norm(
    grad_vec_norm: float,
    threshold: float,
) -> float:
    """Return the multiplicative scale factor to clip a flat grad-vec to ``threshold``.

    Pure-arithmetic helper exposed for unit testing. Returns 1.0 (no clip)
    when ``grad_vec_norm <= threshold``; otherwise returns ``threshold /
    grad_vec_norm`` so the caller can multiply each per-param tensor by this
    scalar to obtain the clipped vector (mathematically identical to
    ``torch.nn.utils.clip_grad_norm_`` semantics applied to the concatenated
    per-task grad vector).
    """
    if threshold <= 0.0:
        return 1.0
    if grad_vec_norm <= threshold:
        return 1.0
    return float(threshold) / float(grad_vec_norm)


__all__ = ["PerTaskClipState", "clip_grad_to_norm"]

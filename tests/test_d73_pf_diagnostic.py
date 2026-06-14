"""[D-73] amendment-6 §Q R20 twin-gate tests for ``--pf-diagnostic-only``.

The ``--pf-diagnostic-only`` flag is the one-lever contract enforcer for the
(1d') grid-vs-MLP test: it computes the ``var_pf_band_ratio`` trainability
observable as a DETACHED readout (``torch.no_grad``) WITHOUT adding any P_F
loss term to the backward objective. The per-microbatch loss handed to
``.backward()`` must stay EXACTLY the plain-[D-24] objective
(``loss_data_mb + mean_F_grad_coef * mean_F_mb``).

These tests exercise the REAL assembly path: they call ``train()`` on the
dummy-data fallback (the §M smoke fallback — no Sherwood required) with a tiny
voxel grid / few rays / few steps, and capture the scalar handed to
``torch.Tensor.backward`` per step. The load-bearing assertion is the
**tol-0 byte-for-byte** check: the backward-value sequence with
``--pf-diagnostic-only`` ON is bit-identical to the same run with NEITHER flag
set (plain [D-24]). Any pf contribution leaking into backward would perturb
those values. The diagnostic branch runs entirely under ``no_grad`` and draws
no RNG, so the two runs are deterministically identical.

  (a) integration: real-path assembly — tol-0 backward equality vs plain
      [D-24]; var_pf_band_ratio computed + finite; grid weights MOVE over
      >= 2 steps; the data-path loss is graph-attached (requires_grad +
      grad_fn).
  (b) mutual-exclusion: ``--pf-diagnostic-only`` + ``--enable-l1-pf-loss``
      -> ValueError at parse.

NO R20 step-100 GradNorm assertion: there is no GradNorm in the diagnostic
path (per dispatch brief).
"""

from __future__ import annotations

import copy
import importlib

import pytest
import torch

pipeline = importlib.import_module("experiments.nerf.pipeline")


# ---------------------------------------------------------------------------
# Shared harness: run the REAL train() path on dummy data, capturing every
# scalar handed to .backward() plus the diagnostic var_pf readouts and a
# pre/post snapshot of the voxel grid weights.
# ---------------------------------------------------------------------------

def _base_argv(tmp_path, *, diagnostic: bool):
    """CLI for a tiny CPU voxel-grid run on the dummy-data fallback.

    data_root points at a non-existent path so load_dataset takes the
    deterministic dummy-data branch (the §M smoke fallback). microbatch >=
    n_rays => accum_steps == 1 => the captured backward scalar is exactly
    loss_mb (no 1/accum_steps rescale ambiguity).
    """
    argv = [
        "--n_rays", "64",
        "--physics", "1",
        "--seed", "0",
        "--arch", "voxel-grid",
        "--voxel-grid-size", "8",
        "--microbatch", "64",
        "--max_steps", "3",
        "--warmup_steps", "1",
        "--checkpoint_interval", "0",
        "--checkpoint_dir", str(tmp_path / "ckpts"),
        "--data_root", str(tmp_path / "NO_SUCH_SHERWOOD_ROOT"),
    ]
    if diagnostic:
        argv.append("--pf-diagnostic-only")
    return argv


def _run_capture(tmp_path, monkeypatch, *, diagnostic: bool):
    """Run train() on the real path; return captured backward + diag state.

    Returns dict with:
      backward_vals : list[float] — exact scalar handed to .backward(), per call
      grad_fn_flags : list[bool]  — whether that scalar had a grad_fn (live graph)
      diag_vals     : list[float] — var_pf readouts from the shared helper
      grid_before   : dict[str, Tensor] — voxel grid params before training
      grid_after    : dict[str, Tensor] — voxel grid params after training
    """
    args = pipeline.parse_args(_base_argv(tmp_path, diagnostic=diagnostic))

    # Force the MLflow path OFF so the run uses nullcontext (no tracker
    # dependency, no hang). set_experiment raising flips mlflow_active=False.
    def _boom(*a, **k):
        raise RuntimeError("mlflow disabled for hermetic test")

    monkeypatch.setattr(pipeline.mlflow, "set_experiment", _boom)

    state = {
        "backward_vals": [],
        "grad_fn_flags": [],
        "diag_vals": [],
        "grid_before": None,
        "grid_after": None,
        "model_ref": [],
    }

    from src.models.voxel_grid_field import VoxelGridField

    # NOTE: backward + helper + __init__ are patched with explicit
    # save/restore (try/finally) rather than monkeypatch so two _run_capture
    # calls within a single test do NOT nest (monkeypatch only restores at
    # test teardown, which would make the second call wrap the first call's
    # patch and double-record).
    orig_backward = torch.Tensor.backward
    orig_helper = pipeline._compute_var_pf_band_ratio
    orig_init = VoxelGridField.__init__

    # Capture every scalar handed to backward. (loss_mb / accum_steps).backward()
    # — with accum_steps == 1 the recorded value is loss_mb itself.
    def _patched_backward(self, *a, **k):
        state["backward_vals"].append(float(self.detach().item()))
        state["grad_fn_flags"].append(self.grad_fn is not None)
        return orig_backward(self, *a, **k)

    # Wrap the single-source-of-truth helper to record its return (proves the
    # diagnostic readout is computed + lets us assert finiteness). The real
    # helper is still executed.
    def _patched_helper(*a, **k):
        v = orig_helper(*a, **k)
        state["diag_vals"].append(v)
        return v

    # Snapshot the voxel grid weights immediately after construction. We hook
    # VoxelGridField.__init__ so we grab the actual model the loop trains.
    def _patched_init(self, *a, **k):
        orig_init(self, *a, **k)
        state["model_ref"].append(self)
        state["grid_before"] = {
            n: p.detach().clone()
            for n, p in self.named_parameters()
        }

    torch.Tensor.backward = _patched_backward
    pipeline._compute_var_pf_band_ratio = _patched_helper
    VoxelGridField.__init__ = _patched_init
    try:
        pipeline.train(args)
    finally:
        torch.Tensor.backward = orig_backward
        pipeline._compute_var_pf_band_ratio = orig_helper
        VoxelGridField.__init__ = orig_init

    model = state["model_ref"][0]
    state["grid_after"] = {
        n: p.detach().clone() for n, p in model.named_parameters()
    }
    return state


# --------------------------------------------------------------------------- (a)
def test_diagnostic_real_path_backward_is_plain_d24_tol0(tmp_path, monkeypatch):
    """LOAD-BEARING: the loss handed to backward under --pf-diagnostic-only is
    byte-for-byte the plain-[D-24] objective.

    We run the REAL train() assembly twice with the same seed — once with
    --pf-diagnostic-only, once with neither flag — and assert the per-step
    backward-value sequences are bit-identical (tol 0). The diagnostic branch
    runs under no_grad and draws no RNG, so any deviation would mean a pf
    contribution leaked into the backward objective.
    """
    diag = _run_capture(tmp_path, monkeypatch, diagnostic=True)
    # Second run on a clean child path. _run_capture save/restores backward +
    # helper + __init__ itself (try/finally), so this call does not nest on the
    # first call's patches.
    plain = _run_capture(tmp_path / "plain", monkeypatch, diagnostic=False)

    assert diag["backward_vals"], "diagnostic run recorded no backward calls"
    assert plain["backward_vals"], "plain run recorded no backward calls"
    assert len(diag["backward_vals"]) == len(plain["backward_vals"]), (
        f"backward-call count differs: diag={len(diag['backward_vals'])} "
        f"plain={len(plain['backward_vals'])}"
    )
    # tol-0 byte-for-byte: identical float bit patterns, not approx.
    for i, (d, p) in enumerate(zip(diag["backward_vals"], plain["backward_vals"])):
        assert d == p, (
            f"backward value {i} differs between diagnostic and plain-[D-24]: "
            f"diag={d!r} plain={p!r} (tol-0 byte-for-byte failed — a pf term "
            f"leaked into the backward objective)"
        )


def test_diagnostic_var_pf_band_ratio_computed_and_finite(tmp_path, monkeypatch):
    """var_pf_band_ratio is computed every step and is finite under the
    diagnostic path."""
    diag = _run_capture(tmp_path, monkeypatch, diagnostic=True)
    assert diag["diag_vals"], "the var_pf helper was never called"
    # one readout per step (max_steps=3)
    assert len(diag["diag_vals"]) == 3, (
        f"expected one var_pf readout per step (3); got {len(diag['diag_vals'])}"
    )
    import math
    for i, v in enumerate(diag["diag_vals"]):
        assert math.isfinite(v), f"var_pf readout at step {i} not finite: {v!r}"


def test_diagnostic_data_path_loss_is_graph_attached(tmp_path, monkeypatch):
    """The scalar handed to backward carries a live graph (requires_grad +
    grad_fn is not None) — the [D-24] gradient is live, not detached."""
    diag = _run_capture(tmp_path, monkeypatch, diagnostic=True)
    assert diag["grad_fn_flags"], "no backward calls captured"
    assert all(diag["grad_fn_flags"]), (
        "a backward scalar had grad_fn=None — the [D-24] data path is detached"
    )


def test_diagnostic_grid_weights_move_over_steps(tmp_path, monkeypatch):
    """The voxel grid weights MOVE over the >=2 training steps — the live
    [D-24] gradient drives parameter updates under the diagnostic path."""
    diag = _run_capture(tmp_path, monkeypatch, diagnostic=True)
    before, after = diag["grid_before"], diag["grid_after"]
    assert before is not None and after is not None
    moved_any = False
    for name in before:
        delta = (after[name] - before[name]).abs().sum().item()
        if delta > 0.0:
            moved_any = True
    assert moved_any, (
        "no voxel grid parameter moved over training — the [D-24] gradient "
        "did not drive updates under --pf-diagnostic-only"
    )


# --------------------------------------------------------------------------- (b)
def test_mutual_exclusion_both_flags_raises_at_parse():
    """--pf-diagnostic-only + --enable-l1-pf-loss -> ValueError at parse
    (the one-lever contract guard)."""
    argv = [
        "--n_rays", "64", "--physics", "1", "--seed", "0",
        "--pf-diagnostic-only", "--enable-l1-pf-loss",
    ]
    with pytest.raises(ValueError, match="mutually"):
        pipeline.parse_args(argv)


def test_diagnostic_only_flag_alone_parses_clean():
    """--pf-diagnostic-only alone parses (no false-positive guard trip)."""
    argv = ["--n_rays", "64", "--physics", "1", "--seed", "0",
            "--pf-diagnostic-only"]
    args = pipeline.parse_args(argv)
    assert args.pf_diagnostic_only is True
    assert args.enable_l1_pf_loss is False

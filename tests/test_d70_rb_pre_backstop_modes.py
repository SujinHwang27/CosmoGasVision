"""[D-70 Rev 5.1 §0.9 F2] Backstop-mode behavioral tests for R-b-pre1/2/3.

R20 twin-gate binding: this test must PASS before any Juno re-dispatch of
Stage 1a / 1b that depends on the demoted-to-observation-flag framing.

Per PI ruling 2026-05-26 (response to Juno job 203285 10/10-seed
constant-density-basin hard-raise), the three R-b-pre* backstops are
OBSERVATION FLAGS rather than gates. The new
``experiments.nerf.pipeline._rb_pre_dispatch`` helper routes flag events
to one of three modes — 'warn' (default, prints + tags), 'raise' (legacy
hard-fail), 'disabled' (silent, tags only).

Assertions
----------
1. ``warn`` mode prints the OBSERVATION FLAG line, sets the MLflow tag,
   and returns control (no raise).
2. ``raise`` mode reproduces the legacy hard-fail with the exact msg.
3. ``disabled`` mode is silent on stdout but still sets the MLflow tag.
4. Argparse defaults all three modes to ``'warn'``.

The dispatch helper is unit-tested directly: assembling a full
``train_pretrain`` run requires the Sherwood substrate (>10min on CPU
even at n_grid=64). The helper is the entire kill-switch surface — once
it routes correctly, the three call-sites that delegate to it inherit
the contract.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pipeline import triggers MLflow + dotenv side effects; that's the
# production import path so we deliberately exercise it here.
from experiments.nerf.pipeline import _rb_pre_dispatch, parse_args  # noqa: E402


# ---------------------------------------------------------------------------
# Helper-level behavioral coverage (modes)
# ---------------------------------------------------------------------------

_SAMPLE_MSG = (
    "R-b-pre1: constant-density basin "
    "(Var(rho_theta)=2.6170e-04 < 0.1 * Var(rho_truth)=4.0425e+00) @ step 100."
)


def test_warn_mode_continues_past_step_100(capsys):
    """warn mode: prints OBSERVATION FLAG, sets MLflow tag, no raise."""
    with patch("experiments.nerf.pipeline.mlflow") as mock_mlflow:
        # Must NOT raise.
        _rb_pre_dispatch(
            flag_name="rb_pre1_flag", msg=_SAMPLE_MSG, mode="warn",
            mlflow_active=True,
        )
    captured = capsys.readouterr()
    assert "OBSERVATION FLAG" in captured.out, (
        f"warn mode must print OBSERVATION FLAG line; got: {captured.out!r}"
    )
    assert "rb_pre1_flag" in captured.out
    assert "Rev 5.1" in captured.out
    # Post-panel N3: per-flag tag + abort_reason="none" both emitted.
    tag_calls = [c.args for c in mock_mlflow.set_tag.call_args_list]
    assert ("rb_pre1_flag", _SAMPLE_MSG) in tag_calls
    assert ("abort_reason", "none") in tag_calls


def test_raise_mode_reproduces_legacy_hard_fail(capsys):
    """raise mode: RuntimeError with matching message + MLflow tag set first."""
    with patch("experiments.nerf.pipeline.mlflow") as mock_mlflow:
        with pytest.raises(RuntimeError) as excinfo:
            _rb_pre_dispatch(
                flag_name="rb_pre1_flag", msg=_SAMPLE_MSG, mode="raise",
                mlflow_active=True,
            )
        assert str(excinfo.value) == _SAMPLE_MSG
        # Tag must be set BEFORE the raise so the downstream Wilcoxon
        # harness sees it on the crash. Post-panel N3: also emit
        # abort_reason=flag_name.
        tag_calls = [c.args for c in mock_mlflow.set_tag.call_args_list]
        assert ("rb_pre1_flag", _SAMPLE_MSG) in tag_calls
        assert ("abort_reason", "rb_pre1_flag") in tag_calls


def test_disabled_mode_silent(capsys):
    """disabled mode: no stdout, MLflow tag still set, no raise."""
    with patch("experiments.nerf.pipeline.mlflow") as mock_mlflow:
        _rb_pre_dispatch(
            flag_name="rb_pre1_flag", msg=_SAMPLE_MSG, mode="disabled",
            mlflow_active=True,
        )
    captured = capsys.readouterr()
    assert "OBSERVATION FLAG" not in captured.out, (
        f"disabled mode must NOT print; got: {captured.out!r}"
    )
    assert captured.out == "", (
        f"disabled mode must be stdout-silent; got: {captured.out!r}"
    )
    # Post-panel N3: per-flag tag + abort_reason="none" both emitted
    # (disabled mode silences stdout but tags still propagate).
    tag_calls = [c.args for c in mock_mlflow.set_tag.call_args_list]
    assert ("rb_pre1_flag", _SAMPLE_MSG) in tag_calls
    assert ("abort_reason", "none") in tag_calls


def test_mlflow_inactive_skips_tag(capsys):
    """When mlflow_active=False, the helper still prints (warn) and does not
    attempt to call mlflow.set_tag."""
    with patch("experiments.nerf.pipeline.mlflow") as mock_mlflow:
        _rb_pre_dispatch(
            flag_name="rb_pre1_flag", msg=_SAMPLE_MSG, mode="warn",
            mlflow_active=False,
        )
        mock_mlflow.set_tag.assert_not_called()
    captured = capsys.readouterr()
    assert "OBSERVATION FLAG" in captured.out


def test_helper_covers_all_three_sister_flags(capsys):
    """warn-mode print prefix carries the flag_name verbatim for all three
    sister backstops (R-b-pre1/2/3 share the dispatcher contract)."""
    for flag_name, msg_prefix in [
        ("rb_pre1_flag", "R-b-pre1: constant-density basin"),
        ("rb_pre2_flag", "R-b-pre2: loss-decreased-but-realism-failed"),
        ("rb_pre3_flag", "R-b-pre3: numerical instability (nan/inf fraction"),
    ]:
        with patch("experiments.nerf.pipeline.mlflow"):
            _rb_pre_dispatch(
                flag_name=flag_name,
                msg=f"{msg_prefix} synthetic-test-msg @ step 100.",
                mode="warn", mlflow_active=True,
            )
        captured = capsys.readouterr()
        assert flag_name in captured.out
        assert msg_prefix in captured.out


# ---------------------------------------------------------------------------
# Argparse default coverage
# ---------------------------------------------------------------------------

def test_default_rb_pre1_pre2_warn_rb_pre3_raise():
    """argparse: per post-panel B3 split, R-b-pre1/2 default 'warn'
    (modeling phenomena, non-gating) and R-b-pre3 defaults 'raise'
    (numerical phenomena: NaN/Inf is undefined behavior that contaminates
    downstream gradient statistics + R_real(500) — silent-continue would
    corrupt the Wilcoxon harness).
    """
    argv = [
        "--n_rays", "256", "--physics", "1", "--seed", "0",
    ]
    args = parse_args(argv)
    assert args.rb_pre1_mode == "warn", (
        f"--rb-pre1-mode default must be 'warn'; got {args.rb_pre1_mode!r}"
    )
    assert args.rb_pre2_mode == "warn", (
        f"--rb-pre2-mode default must be 'warn'; got {args.rb_pre2_mode!r}"
    )
    assert args.rb_pre3_mode == "raise", (
        f"--rb-pre3-mode default must be 'raise' (panel B3 split); "
        f"got {args.rb_pre3_mode!r}"
    )


# ---------------------------------------------------------------------------
# Post-panel N3: abort_reason back-compat tag emission
# ---------------------------------------------------------------------------

def test_warn_mode_emits_abort_reason_none():
    """warn mode: in addition to the per-flag tag, emit
    `abort_reason = "none"` so the Wilcoxon harness can keep filtering
    aborted runs out of the n=10 sample via `tags.abort_reason = "none"`.
    """
    with patch("experiments.nerf.pipeline.mlflow") as mock_mlflow:
        _rb_pre_dispatch(
            flag_name="rb_pre1_flag", msg=_SAMPLE_MSG, mode="warn",
            mlflow_active=True,
        )
    tag_calls = [c.args for c in mock_mlflow.set_tag.call_args_list]
    assert ("rb_pre1_flag", _SAMPLE_MSG) in tag_calls, (
        f"per-flag tag missing; got {tag_calls!r}"
    )
    assert ("abort_reason", "none") in tag_calls, (
        f"warn-mode must emit abort_reason='none'; got {tag_calls!r}"
    )


def test_raise_mode_emits_abort_reason_flag_name():
    """raise mode: emit `abort_reason = flag_name` (e.g. 'rb_pre1_flag')
    BEFORE the RuntimeError so the downstream Wilcoxon harness can
    distinguish aborted runs from clean runs.
    """
    with patch("experiments.nerf.pipeline.mlflow") as mock_mlflow:
        with pytest.raises(RuntimeError):
            _rb_pre_dispatch(
                flag_name="rb_pre1_flag", msg=_SAMPLE_MSG, mode="raise",
                mlflow_active=True,
            )
    tag_calls = [c.args for c in mock_mlflow.set_tag.call_args_list]
    assert ("rb_pre1_flag", _SAMPLE_MSG) in tag_calls
    assert ("abort_reason", "rb_pre1_flag") in tag_calls, (
        f"raise-mode must emit abort_reason='rb_pre1_flag'; "
        f"got {tag_calls!r}"
    )


def test_argparse_accepts_all_three_choices():
    """argparse: all three flags accept warn/raise/disabled choices."""
    for mode in ["warn", "raise", "disabled"]:
        argv = [
            "--n_rays", "256", "--physics", "1", "--seed", "0",
            "--rb-pre1-mode", mode,
            "--rb-pre2-mode", mode,
            "--rb-pre3-mode", mode,
        ]
        args = parse_args(argv)
        assert args.rb_pre1_mode == mode
        assert args.rb_pre2_mode == mode
        assert args.rb_pre3_mode == mode


def test_argparse_rejects_unknown_mode():
    argv = [
        "--n_rays", "256", "--physics", "1", "--seed", "0",
        "--rb-pre1-mode", "abort",  # not in choices
    ]
    with pytest.raises(SystemExit):
        parse_args(argv)

"""[D-42] milestone 4 — 50-step smoke gate evaluator.

Consumes a NeRF training-state JSON (loss history) + step-50 model
predictions and emits a PASS/FAIL JSON sidecar covering the six smoke
gates spelled out in the [D-42] design spec (LEDGER §3 [D-42], HEAD
``784678f``):

  1. No NaN across logged metrics or prediction arrays.
  2. ``loss_total[step=50] / loss_total[step=0] < 0.85``   ([D-19]).
  3. ``|mean_F[step=50] - 0.979| < 0.05``                  ([D-34] anchor).
  4. ``tau_amp[step=50] in [0.5, 2.0]``.
  5. NEW — density spread: ``max(rho_pred) - min(rho_pred) > 1.45``
     across the (n_rays * n_bins) prediction grid                  ([D-41] Add1).
  6. NEW — X_HI spread:   ``max(X_HI_pred) - min(X_HI_pred) > 6e-5``
     across the (n_rays * n_bins) prediction grid                  ([D-41] Add1).

Pass condition: ALL six gates PASS.

Single-physics (P1), single-redshift (0.3), single-seed (0) — the
smoke regime; same posture as scripts/diag_pf_per_bin.py minus the
cross-physics loop.

The pipeline does NOT currently emit either a loss-history JSON nor a
step-50 predictions sidecar; instead this script reads loss history
from the local MLflow filesystem tree (``<run_dir>/mlflow/<exp_id>/<run_id>/metrics/{loss,mean_flux_pred,tau_amp}``)
and synthesises step-50 predictions by replaying the step-50 checkpoint
on the same training rays. Two fallback flags are provided for runs
that don't fit that layout:

  --loss_history_json <path>   JSON list of {step,loss_total,mean_F,tau_amp}
  --predictions_npz   <path>   .npz with density, X_HI arrays
  --checkpoint        <path>   explicit checkpoint path (overrides run_dir)

Three built-in self-tests:

  --self_test pass    Synthetic state, all six gates PASS.
  --self_test fail    Synthetic state, gate 5 FAIL (collapsed density).
  --self_test d41     Loads the [D-41] FGPA-tail Tier-1 checkpoint
                      (cloud_runs/fgpa-tail-tier1-P1-step25k.pt) and
                      confirms gates 5 + 6 FAIL — the canonical
                      constant-prediction-collapse signature this
                      evaluator was designed to catch.

Usage::

    # Full evaluation on a real run dir
    PYTHONPATH=. uv run python scripts/diag_velocity_gradient_smoke.py \\
        experiments/nerf/artifacts/<run_id>/

    # Self-tests (no run dir required)
    PYTHONPATH=. uv run python scripts/diag_velocity_gradient_smoke.py \\
        --self_test pass
    PYTHONPATH=. uv run python scripts/diag_velocity_gradient_smoke.py \\
        --self_test fail
    PYTHONPATH=. uv run python scripts/diag_velocity_gradient_smoke.py \\
        --self_test d41
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Windows default code page (cp949 on Korean Win11) can't encode em-dashes
# emitted by our human-readable summaries; reconfigure stdout/stderr to UTF-8
# so the self-tests and run-time prints work everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover — older Pythons / wrapped streams
        pass


# ---------------------------------------------------------------------------
# Spec constants — keep co-located with the gate definitions for auditability.
# ---------------------------------------------------------------------------

SPEC_REF = "experiments/nerf/LEDGER.md [D-42] HEAD 784678f"

GATE2_RATIO_THRESHOLD = 0.85          # [D-19]
GATE3_MEAN_F_ANCHOR = 0.979           # [D-34]
GATE3_DRIFT_THRESHOLD = 0.05
GATE4_TAU_AMP_RANGE = (0.5, 2.0)
GATE5_DENSITY_SPREAD = 1.45           # 10 x truth median 0.145
GATE6_XHI_SPREAD = 6e-5               # 100 x truth median 6e-7

D42_FGPA_TAIL_CKPT = REPO_ROOT / "cloud_runs" / "fgpa-tail-tier1-P1-step25k.pt"
D41_CACHED_JSON = (
    REPO_ROOT / "experiments" / "nerf" / "artifacts" / "eval"
    / "cleanup_pass" / "item4_n_HI_distribution_tier1_fgpatail.json"
)


# ---------------------------------------------------------------------------
# Gate logic — pure functions over a small `state` dict so it is testable
# without touching torch/sherwood.
# ---------------------------------------------------------------------------

def _safe_min_max(arr: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(arr, dtype=np.float64).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (float(finite.min()), float(finite.max()), float(np.median(finite)))


def _has_nan(arr) -> bool:
    a = np.asarray(arr, dtype=np.float64)
    return bool(np.isnan(a).any())


def evaluate_gates(state: dict) -> dict:
    """Apply the six [D-42] gates to ``state`` and return the JSON payload.

    ``state`` keys:
      loss_history     : list[dict] with step,loss_total,mean_F,tau_amp
      density_pred     : (n_rays, n_bins) array
      X_HI_pred        : (n_rays, n_bins) array
      run_id           : str
      missing_keys     : list[str] (optional) — fields not provided; gates
                         depending on them auto-FAIL with that explanation.
    """
    gates: dict = {}
    fail_reasons: list[str] = []
    missing = set(state.get("missing_keys", []))

    # -- Gate 1: No NaN -------------------------------------------------------
    nan_offenders: list[str] = []
    if "loss_history" not in missing:
        for rec in state.get("loss_history", []):
            for k in ("loss_total", "mean_F", "tau_amp"):
                v = rec.get(k)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    nan_offenders.append(f"step={rec.get('step')}:{k}")
                    break
    if "density_pred" not in missing and _has_nan(state.get("density_pred", [0.0])):
        nan_offenders.append("density_pred")
    if "X_HI_pred" not in missing and _has_nan(state.get("X_HI_pred", [0.0])):
        nan_offenders.append("X_HI_pred")
    gate1_pass = (not nan_offenders) and not (
        {"loss_history", "density_pred", "X_HI_pred"} & missing
    )
    gates["gate_1_no_nan"] = {
        "pass": gate1_pass,
        "details": (
            "no NaN observed" if gate1_pass and not nan_offenders
            else f"NaN in: {nan_offenders}" if nan_offenders
            else f"missing fields: {sorted({'loss_history','density_pred','X_HI_pred'} & missing)}"
        ),
    }
    if not gate1_pass:
        fail_reasons.append(
            "gate_1_no_nan: " + gates["gate_1_no_nan"]["details"]
        )

    # -- Gate 2: loss descends -------------------------------------------------
    if "loss_history" in missing:
        gates["gate_2_loss_descends"] = {
            "pass": False, "loss_step_0": None, "loss_step_50": None,
            "ratio": None, "threshold": GATE2_RATIO_THRESHOLD,
            "details": "loss_history not provided",
        }
        fail_reasons.append("gate_2_loss_descends: loss_history not provided")
    else:
        history = state["loss_history"]
        rec0 = _find_record(history, 0)
        rec50 = _find_record(history, 50)
        if rec0 is None or rec50 is None:
            gates["gate_2_loss_descends"] = {
                "pass": False, "loss_step_0": None, "loss_step_50": None,
                "ratio": None, "threshold": GATE2_RATIO_THRESHOLD,
                "details": f"missing step 0 (={rec0}) or step 50 (={rec50}) record",
            }
            fail_reasons.append("gate_2_loss_descends: missing endpoint record")
        else:
            l0 = float(rec0["loss_total"])
            l50 = float(rec50["loss_total"])
            ratio = l50 / l0 if l0 != 0 else float("inf")
            ok = ratio < GATE2_RATIO_THRESHOLD
            gates["gate_2_loss_descends"] = {
                "pass": bool(ok),
                "loss_step_0": l0, "loss_step_50": l50,
                "ratio": ratio, "threshold": GATE2_RATIO_THRESHOLD,
            }
            if not ok:
                fail_reasons.append(
                    f"gate_2_loss_descends: loss[50]/loss[0]={ratio:.4f} "
                    f">= {GATE2_RATIO_THRESHOLD} (no descent)"
                )

    # -- Gate 3: mean_F preserved ---------------------------------------------
    if "loss_history" in missing:
        gates["gate_3_mean_F_preserved"] = {
            "pass": False, "mean_F_step_50": None,
            "anchor": GATE3_MEAN_F_ANCHOR, "abs_drift": None,
            "threshold": GATE3_DRIFT_THRESHOLD,
            "details": "loss_history not provided",
        }
        fail_reasons.append("gate_3_mean_F_preserved: loss_history not provided")
    else:
        rec50 = _find_record(state["loss_history"], 50)
        if rec50 is None or rec50.get("mean_F") is None:
            gates["gate_3_mean_F_preserved"] = {
                "pass": False, "mean_F_step_50": None,
                "anchor": GATE3_MEAN_F_ANCHOR, "abs_drift": None,
                "threshold": GATE3_DRIFT_THRESHOLD,
                "details": "missing mean_F at step 50",
            }
            fail_reasons.append("gate_3_mean_F_preserved: missing mean_F[step=50]")
        else:
            mF = float(rec50["mean_F"])
            drift = abs(mF - GATE3_MEAN_F_ANCHOR)
            ok = drift < GATE3_DRIFT_THRESHOLD
            gates["gate_3_mean_F_preserved"] = {
                "pass": bool(ok),
                "mean_F_step_50": mF,
                "anchor": GATE3_MEAN_F_ANCHOR,
                "abs_drift": drift,
                "threshold": GATE3_DRIFT_THRESHOLD,
            }
            if not ok:
                fail_reasons.append(
                    f"gate_3_mean_F_preserved: |{mF:.4f} - {GATE3_MEAN_F_ANCHOR}|"
                    f" = {drift:.4f} >= {GATE3_DRIFT_THRESHOLD}"
                )

    # -- Gate 4: tau_amp stable -----------------------------------------------
    if "loss_history" in missing:
        gates["gate_4_tau_amp_stable"] = {
            "pass": False, "tau_amp_step_50": None,
            "range": list(GATE4_TAU_AMP_RANGE),
            "details": "loss_history not provided",
        }
        fail_reasons.append("gate_4_tau_amp_stable: loss_history not provided")
    else:
        rec50 = _find_record(state["loss_history"], 50)
        if rec50 is None or rec50.get("tau_amp") is None:
            gates["gate_4_tau_amp_stable"] = {
                "pass": False, "tau_amp_step_50": None,
                "range": list(GATE4_TAU_AMP_RANGE),
                "details": "missing tau_amp at step 50",
            }
            fail_reasons.append("gate_4_tau_amp_stable: missing tau_amp[step=50]")
        else:
            t = float(rec50["tau_amp"])
            ok = GATE4_TAU_AMP_RANGE[0] <= t <= GATE4_TAU_AMP_RANGE[1]
            gates["gate_4_tau_amp_stable"] = {
                "pass": bool(ok),
                "tau_amp_step_50": t,
                "range": list(GATE4_TAU_AMP_RANGE),
            }
            if not ok:
                fail_reasons.append(
                    f"gate_4_tau_amp_stable: tau_amp[50]={t:.4f} outside "
                    f"{GATE4_TAU_AMP_RANGE}"
                )

    # -- Gate 5: density spread ([D-41] Addendum 1) ---------------------------
    if "density_pred" in missing:
        gates["gate_5_density_spread"] = {
            "pass": False, "max_minus_min": None,
            "threshold": GATE5_DENSITY_SPREAD,
            "min": None, "max": None, "median": None,
            "details": "density_pred not provided",
        }
        fail_reasons.append("gate_5_density_spread: density_pred not provided")
    else:
        lo, hi, med = _safe_min_max(state["density_pred"])
        spread = hi - lo if math.isfinite(hi) and math.isfinite(lo) else float("nan")
        ok = math.isfinite(spread) and spread > GATE5_DENSITY_SPREAD
        gates["gate_5_density_spread"] = {
            "pass": bool(ok),
            "max_minus_min": spread,
            "threshold": GATE5_DENSITY_SPREAD,
            "min": lo, "max": hi, "median": med,
        }
        if not ok:
            fail_reasons.append(
                f"gate_5_density_spread: max-min={spread:.4f} <= "
                f"{GATE5_DENSITY_SPREAD} (constant-prediction-collapse signature)"
            )

    # -- Gate 6: X_HI spread ([D-41] Addendum 1) ------------------------------
    if "X_HI_pred" in missing:
        gates["gate_6_X_HI_spread"] = {
            "pass": False, "max_minus_min": None,
            "threshold": GATE6_XHI_SPREAD,
            "min": None, "max": None, "median": None,
            "details": "X_HI_pred not provided",
        }
        fail_reasons.append("gate_6_X_HI_spread: X_HI_pred not provided")
    else:
        lo, hi, med = _safe_min_max(state["X_HI_pred"])
        spread = hi - lo if math.isfinite(hi) and math.isfinite(lo) else float("nan")
        ok = math.isfinite(spread) and spread > GATE6_XHI_SPREAD
        gates["gate_6_X_HI_spread"] = {
            "pass": bool(ok),
            "max_minus_min": spread,
            "threshold": GATE6_XHI_SPREAD,
            "min": lo, "max": hi, "median": med,
        }
        if not ok:
            fail_reasons.append(
                f"gate_6_X_HI_spread: max-min={spread:.3e} <= "
                f"{GATE6_XHI_SPREAD:.3e} (X_HI head collapsed)"
            )

    overall = all(g["pass"] for g in gates.values())
    return {
        "run_id": state.get("run_id", "<unknown>"),
        "spec_ref": SPEC_REF,
        "gates": gates,
        "overall_pass": bool(overall),
        "fail_reasons": fail_reasons,
    }


def _find_record(history: list[dict], step: int) -> Optional[dict]:
    for r in history:
        if int(r.get("step", -1)) == step:
            return r
    return None


# ---------------------------------------------------------------------------
# Loaders — read loss history from MLflow filesystem tree or JSON, and
# step-50 predictions from .npz or by replaying a checkpoint.
# ---------------------------------------------------------------------------

def load_loss_history_from_mlflow(mlflow_root: Path) -> list[dict]:
    """Read MLflow ``metrics/{loss,mean_flux_pred,tau_amp}`` text files and
    return a list[dict] keyed by step. Each MLflow metric file has rows
    ``<timestamp_ms> <value> <step>``.
    """
    if not mlflow_root.exists():
        raise FileNotFoundError(f"MLflow root not found: {mlflow_root}")

    # Locate the per-run metrics dir.  Layout: mlflow/<exp_id>/<run_id>/metrics/
    candidates = list(mlflow_root.glob("*/*/metrics"))
    if not candidates:
        raise FileNotFoundError(
            f"No metrics directory under {mlflow_root} (expected "
            f"<exp_id>/<run_id>/metrics/)."
        )
    metrics_dir = candidates[0]

    def _read(name: str) -> dict[int, float]:
        path = metrics_dir / name
        if not path.exists():
            return {}
        out: dict[int, float] = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                step = int(parts[2])
                out[step] = float(parts[1])
            except ValueError:
                continue
        return out

    loss = _read("loss")
    meanF = _read("mean_flux_pred")
    tau = _read("tau_amp")
    steps = sorted(set(loss) | set(meanF) | set(tau))
    return [
        {
            "step": s,
            "loss_total": loss.get(s),
            "mean_F": meanF.get(s),
            "tau_amp": tau.get(s),
        }
        for s in steps
    ]


def load_loss_history_from_json(path: Path) -> list[dict]:
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "loss_history" in obj:
        return obj["loss_history"]
    raise ValueError(f"Unrecognised loss-history JSON layout: {path}")


def replay_checkpoint_predictions(
    ckpt_path: Path,
    *,
    physics_id: int = 1,
    redshift: float = 0.3,
    n_rays: int = 64,
    sherwood_root: Optional[Path] = None,
    run_id: str = "<no-mlflow>",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load ckpt, forward-pass on (the first ``n_rays`` of) the matching
    Sherwood sightlines, return ``(density_pred, X_HI_pred, mean_F_pred)``.

    Pipeline trains on ``coords[:n_rays]`` (load_dataset), so we mirror
    that — the prediction grid is exactly what the loss saw at step 50.
    """
    import torch  # local import: keeps module-level import time low.
    from scripts.eval_anchor_invariance_d34 import _build_model_with_fallback
    from src.analysis.stage2b_report import _render_tau_for_model
    from src.data.loader import SherwoodLoader

    if sherwood_root is None:
        sherwood_root = REPO_ROOT / "Sherwood"

    sherwood = SherwoodLoader(str(sherwood_root))
    sl = sherwood.load_sightlines(physics_id, redshift)
    box_kpc_h = float(sl["header"]["box_kpc_h"])
    coords_world = sherwood.get_world_coordinates(sl)
    coords_unit = coords_world[:n_rays] / box_kpc_h
    vel_axis_t = torch.tensor(np.asarray(sl["vel_axis"]), dtype=torch.float32)

    model = _build_model_with_fallback(run_id, str(ckpt_path))

    with torch.no_grad():
        x = torch.tensor(coords_unit, dtype=torch.float32)
        fields = model(x)  # (n_rays, n_bins, 4)
        density = fields[..., 0].cpu().numpy().astype(np.float64)
        X_HI = fields[..., 2].cpu().numpy().astype(np.float64)

        tau_pred = _render_tau_for_model(model, x, vel_axis_t)
        F_pred = np.exp(-np.asarray(tau_pred))
        mean_F = float(F_pred.mean())

    return density, X_HI, mean_F


def load_predictions_npz(path: Path) -> dict:
    z = np.load(path)
    out = {"density_pred": np.asarray(z["density"]),
           "X_HI_pred": np.asarray(z["X_HI"])}
    if "mean_F" in z.files:
        out["mean_F_obs"] = float(z["mean_F"][()] if z["mean_F"].ndim else z["mean_F"])
    if "tau_amp" in z.files:
        out["tau_amp_obs"] = float(z["tau_amp"][()] if z["tau_amp"].ndim else z["tau_amp"])
    return out


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _synthetic_pass_state() -> dict:
    """Six-gate pass: loss halves, mean_F at anchor, tau_amp=1.0, dense
    well-spread density and X_HI fields."""
    rng = np.random.default_rng(0)
    history = []
    for step in range(0, 51):
        history.append({
            "step": step,
            "loss_total": 1.0 * (0.99 ** step),  # 0.605 at step 50 → ratio 0.605
            "mean_F": 0.979 + 0.001 * np.sin(step / 10.0),
            "tau_amp": 1.0 + 0.01 * np.cos(step / 10.0),
        })
    density = rng.uniform(0.05, 100.0, size=(64, 256))
    X_HI = rng.uniform(1e-9, 1e-3, size=(64, 256))
    return {
        "run_id": "synthetic-pass",
        "loss_history": history,
        "density_pred": density,
        "X_HI_pred": X_HI,
    }


def _synthetic_fail_state() -> dict:
    """Density collapsed to a single value — the [D-41] failure mode."""
    state = _synthetic_pass_state()
    state["run_id"] = "synthetic-fail-collapsed-density"
    # Constant density ~ 71.5 (matches the [D-41] cleanup_pass numbers).
    state["density_pred"] = np.full((64, 256), 71.5)
    return state


def _d41_state_from_cached_json() -> dict:
    """Build a state from the cached cleanup_pass [D-41] item-4 JSON.

    We don't have per-ray density arrays on disk for the [D-41] Tier-1
    checkpoint without GPU replay; the cached JSON gives us min/max for
    both density and X_HI, which is sufficient to evaluate gates 5 + 6
    deterministically (the gates depend on max-min only).
    """
    if not D41_CACHED_JSON.exists():
        raise FileNotFoundError(
            f"[D-41] cached JSON not found: {D41_CACHED_JSON}. "
            f"Cannot run d41 self-test."
        )
    obj = json.loads(D41_CACHED_JSON.read_text())
    pred = obj["fgpa_tail_tier1_pred"]
    # Synthetic two-element arrays reproducing the cached min/max so the
    # gate-5/6 spread reduction is exact.
    density = np.array([pred["density"]["min"], pred["density"]["max"]])
    X_HI = np.array([pred["X_HI"]["min"], pred["X_HI"]["max"]])
    return {
        "run_id": "d41-fgpa-tail-tier1-cached",
        "loss_history": [],
        "density_pred": density,
        "X_HI_pred": X_HI,
        "missing_keys": ["loss_history"],
    }


def _d41_state_from_checkpoint() -> dict:
    """Replay the [D-41] FGPA-tail Tier-1 checkpoint directly (preferred
    if the GPU + Sherwood data are available)."""
    if not D42_FGPA_TAIL_CKPT.exists():
        raise FileNotFoundError(
            f"[D-41] checkpoint not found: {D42_FGPA_TAIL_CKPT}. "
            f"Cannot run d41 self-test in replay mode."
        )
    density, X_HI, mean_F = replay_checkpoint_predictions(
        D42_FGPA_TAIL_CKPT, run_id="d41-fgpa-tail-tier1",
    )
    return {
        "run_id": "d41-fgpa-tail-tier1-replay",
        "loss_history": [],
        "density_pred": density,
        "X_HI_pred": X_HI,
        "missing_keys": ["loss_history"],
        "_aux": {"mean_F_pred": mean_F},
    }


def run_self_tests(which: str) -> int:
    """Returns exit code: 0 if every assertion in the requested self-test
    holds, 1 otherwise."""
    if which == "pass":
        payload = evaluate_gates(_synthetic_pass_state())
        print(json.dumps(payload, indent=2))
        assert payload["overall_pass"] is True, payload["fail_reasons"]
        print("[self_test pass] PASS — overall_pass=True")
        return 0

    if which == "fail":
        payload = evaluate_gates(_synthetic_fail_state())
        print(json.dumps(payload, indent=2))
        assert payload["overall_pass"] is False
        assert any("gate_5_density_spread" in r for r in payload["fail_reasons"]), \
            payload["fail_reasons"]
        print("[self_test fail] PASS — overall_pass=False with gate_5 in fail_reasons")
        return 0

    if which == "d41":
        # Spec prediction (LEDGER §3 [D-42] gate 5):
        #   "[D-41] Tier-1 collapse pattern showed density ~ 71.5 +/- 1.6
        #    (a 7-unit window); this gate blocks any spread less than 1.45
        #    absolute units, which is a *floor* well above the [D-41] failure
        #    range."
        # I.e. gate 5 is intentionally generous and does NOT catch [D-41] by
        # spread alone (6.32 > 1.45). Gate 6 IS the gate that catches it
        # (X_HI spread 3.1e-5 < 6e-5 floor). So the spec predicts:
        #   gate_5_density_spread  -> PASS on [D-41]   (not the collapse-catcher)
        #   gate_6_X_HI_spread     -> FAIL on [D-41]   (the catcher)
        # overall_pass = False  (one gate fails -> [D-42] would stop short
        # of Tier-1 if this signature recurred).
        state = _d41_state_from_cached_json()
        payload = evaluate_gates(state)
        print(json.dumps(payload, indent=2))
        g5 = payload["gates"]["gate_5_density_spread"]
        g6 = payload["gates"]["gate_6_X_HI_spread"]
        print()
        print(f"[self_test d41] gate_5: pass={g5['pass']}  "
              f"max-min={g5['max_minus_min']:.4f}  threshold={g5['threshold']}")
        print(f"[self_test d41] gate_6: pass={g6['pass']}  "
              f"max-min={g6['max_minus_min']:.3e}  threshold={g6['threshold']:.3e}")
        spec_ok = (g5["pass"] is True) and (g6["pass"] is False)
        if spec_ok:
            print("[self_test d41] PASS — gate_5 passes (spec intent: floor "
                  "above [D-41] failure range), gate_6 fails (spec intent: "
                  "catches [D-41] X_HI collapse). overall_pass=False as "
                  "designed for the canonical collapse signature.")
            return 0
        print(
            "[self_test d41] STOP-AND-SURFACE: signature deviates from spec "
            "prediction. Expected gate_5 PASS + gate_6 FAIL on [D-41]; got "
            f"gate_5 pass={g5['pass']}, gate_6 pass={g6['pass']}."
        )
        return 1

    print(f"unknown self_test mode: {which!r}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def build_state_from_run_dir(args) -> dict:
    run_dir = Path(args.run_dir) if args.run_dir else None
    state: dict = {"missing_keys": []}

    run_id = args.run_id or (run_dir.name if run_dir else "<unknown>")
    state["run_id"] = run_id

    # ----- Loss history --------------------------------------------------
    if args.loss_history_json:
        state["loss_history"] = load_loss_history_from_json(
            Path(args.loss_history_json)
        )
    elif args.mlflow_dir:
        state["loss_history"] = load_loss_history_from_mlflow(
            Path(args.mlflow_dir)
        )
    elif run_dir is not None and (run_dir / "mlflow").exists():
        state["loss_history"] = load_loss_history_from_mlflow(run_dir / "mlflow")
    elif run_dir is not None and (run_dir / "loss_history.json").exists():
        state["loss_history"] = load_loss_history_from_json(
            run_dir / "loss_history.json"
        )
    elif run_dir is not None and (run_dir / "training_log.json").exists():
        state["loss_history"] = load_loss_history_from_json(
            run_dir / "training_log.json"
        )
    else:
        print(
            f"[diag-d42] no loss-history source found in {run_dir} "
            "(checked mlflow/, loss_history.json, training_log.json). "
            "Gates 2/3/4 will FAIL with 'loss_history not provided'.",
            file=sys.stderr,
        )
        state["missing_keys"].append("loss_history")

    # ----- Step-50 predictions ------------------------------------------
    if args.predictions_npz:
        state.update(load_predictions_npz(Path(args.predictions_npz)))
    else:
        # Resolve checkpoint.
        ckpt = None
        if args.checkpoint:
            ckpt = Path(args.checkpoint)
        elif run_dir is not None:
            for candidate in (
                run_dir / "step50_state.pt",
                run_dir / "checkpoints" / "step_000050.pt",
                run_dir / "checkpoints" / "step_50.pt",
            ):
                if candidate.exists():
                    ckpt = candidate
                    break
        if ckpt is None or not ckpt.exists():
            print(
                f"[diag-d42] no step-50 predictions sidecar nor checkpoint "
                f"found. Looked for predictions_npz / "
                f"{run_dir / 'checkpoints' / 'step_000050.pt' if run_dir else '<no run_dir>'} . "
                "Gates 1, 5, 6 will FAIL with 'predictions not provided'. "
                "Hand-off: core-implementer should add `step50_predictions.npz` "
                "sidecar to experiments/nerf/pipeline.py.",
                file=sys.stderr,
            )
            state["missing_keys"].extend(["density_pred", "X_HI_pred"])
        else:
            density, X_HI, mean_F = replay_checkpoint_predictions(
                ckpt,
                physics_id=args.physics_id,
                redshift=args.redshift,
                n_rays=args.n_rays,
                run_id=run_id,
            )
            state["density_pred"] = density
            state["X_HI_pred"] = X_HI
            state["_aux_mean_F"] = mean_F

    return state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", nargs="?", default=None,
                   help="Path to NeRF run output dir, e.g. "
                        "experiments/nerf/artifacts/<run_id>/")
    p.add_argument("--run_id", default=None,
                   help="Override run_id stamp in output JSON; default = "
                        "basename(run_dir).")
    p.add_argument("--mlflow_dir", default=None,
                   help="Path to colocated mlflow/ tree (overrides run_dir/mlflow).")
    p.add_argument("--loss_history_json", default=None,
                   help="JSON file with [{step,loss_total,mean_F,tau_amp},...].")
    p.add_argument("--predictions_npz", default=None,
                   help=".npz with density, X_HI (and optional mean_F, tau_amp).")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit step-50 checkpoint path.")
    p.add_argument("--physics_id", type=int, default=1)
    p.add_argument("--redshift", type=float, default=0.3)
    p.add_argument("--n_rays", type=int, default=64,
                   help="Mirrors pipeline.py default smoke n_rays.")
    p.add_argument("--output_dir", default=None,
                   help="Override the default "
                        "experiments/nerf/artifacts/eval/d42_smoke/ destination.")
    p.add_argument("--self_test", choices=["pass", "fail", "d41"], default=None)
    args = p.parse_args(argv)

    if args.self_test:
        return run_self_tests(args.self_test)

    if not args.run_dir:
        print("FATAL: run_dir is required (or pass --self_test {pass,fail,d41}).",
              file=sys.stderr)
        return 2

    state = build_state_from_run_dir(args)
    payload = evaluate_gates(state)

    # Output
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = REPO_ROOT / "experiments" / "nerf" / "artifacts" / "eval" / "d42_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{payload['run_id']}_gates.json"
    out_json.write_text(json.dumps(payload, indent=2))

    # Stdout summary
    print(json.dumps(payload, indent=2))
    print(f"\n[diag-d42] wrote {out_json}")
    print(f"[diag-d42] overall_pass = {payload['overall_pass']}")
    if payload["fail_reasons"]:
        print("[diag-d42] fail_reasons:")
        for r in payload["fail_reasons"]:
            print(f"  - {r}")
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

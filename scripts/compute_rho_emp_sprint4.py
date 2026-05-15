"""
Compute empirical correlation rho_emp of correctness indicators between the
sprint-4 3D ResNet-18 and the [mean, var] 2-scalar baseline, on the same 8000
test crops.

Design-doc obligation: S2 (sprint5_cprime_48cube.md v3 §4.3).

Formula:
    rho_emp = Corr(1{ResNet correct on crop i}, 1{baseline correct on crop i})
              for i = 1..8000

Acceptance:
    rho_emp in [0.0, 0.5] -> MDE table at §4.2 unchanged, PASS.
    rho_emp outside [0.0, 0.5] -> recompute MDE at empirical rho, surface for
                                  PI re-rule (design-doc-amendment trigger).

Usage:
    python -u scripts/compute_rho_emp_sprint4.py \
        --headline cloud_runs/Sprint4-30ep-a13dce8-20260514-110740-e72fca/\
eval/sprint4_1778774878_headline.json \
        --out experiments/nerf/artifacts/eval/sprint5_cprime/rho_emp_sprint4.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np


def _extract_per_crop_correctness(headline: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Look for per-crop correctness arrays in headline.json under any of the
    expected keys. Returns (resnet_correct, baseline_correct) if both present,
    else None.
    """
    candidate_pairs = [
        ("per_crop_resnet_correct", "per_crop_mean_variance_correct"),
        ("per_crop_resnet_correct", "per_crop_baseline_correct"),
        ("resnet_per_crop_correct", "mean_variance_per_crop_correct"),
    ]
    for r_key, b_key in candidate_pairs:
        if r_key in headline and b_key in headline:
            r = np.asarray(headline[r_key], dtype=np.int8)
            b = np.asarray(headline[b_key], dtype=np.int8)
            if r.shape == b.shape and r.ndim == 1:
                return r, b
    # Or per-crop predictions + per-crop labels for both classifiers
    if (
        "per_crop_labels" in headline
        and "per_crop_resnet_pred" in headline
        and "per_crop_mean_variance_pred" in headline
    ):
        y = np.asarray(headline["per_crop_labels"], dtype=np.int8)
        r_pred = np.asarray(headline["per_crop_resnet_pred"], dtype=np.int8)
        b_pred = np.asarray(headline["per_crop_mean_variance_pred"], dtype=np.int8)
        return (r_pred == y).astype(np.int8), (b_pred == y).astype(np.int8)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    headline = json.loads(args.headline.read_text(encoding="utf-8"))
    extracted = _extract_per_crop_correctness(headline)

    out_payload: dict = {
        "source": args.headline.name,
        "computed_at": _dt.date.today().isoformat(),
        "design_doc_obligation": (
            "S2 (§4.3 sprint5_cprime_48cube.md v3) — "
            "rho_emp pre-flight"
        ),
    }

    if extracted is None:
        out_payload.update(
            {
                "status": "BLOCKER",
                "blocker_reason": (
                    "headline.json contains only aggregated accuracies + 4x4 "
                    "confusion matrix; no per-crop predictions or correctness "
                    "indicators for either the ResNet or the [mean, var] "
                    "baseline. rho_emp cannot be computed from the sprint-4 "
                    "disk artifact alone."
                ),
                "available_keys": sorted(headline.keys()),
                "remediation": (
                    "Either (a) re-run sprint-4 eval with per-crop logging, or "
                    "(b) discharge S2 obligation downstream when (c′) refactor "
                    "produces 48^3 evals with per-crop arrays."
                ),
            }
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        print(f"[BLOCKER] no per-crop correctness in {args.headline.name}")
        print(f"  wrote {args.out}")
        return 0

    r, b = extracted
    n = int(r.shape[0])
    # numpy.corrcoef handles binary inputs as 0/1 floats; result is Pearson.
    # Guard zero-variance edge case.
    if r.std() == 0.0 or b.std() == 0.0:
        rho = float("nan")
        verdict_band = "undefined_zero_variance"
    else:
        rho = float(np.corrcoef(r.astype(np.float64), b.astype(np.float64))[0, 1])
        verdict_band = "pass_unchanged" if 0.0 <= rho <= 0.5 else "outside_band_recompute_mde"

    out_payload.update(
        {
            "status": "OK",
            "rho_emp": rho,
            "n_crops": n,
            "resnet_acc_observed": float(r.mean()),
            "baseline_acc_observed": float(b.mean()),
            "acceptance_band": [0.0, 0.5],
            "acceptance_verdict": verdict_band,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"rho_emp = {rho:.4f} (n={n}); verdict: {verdict_band}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

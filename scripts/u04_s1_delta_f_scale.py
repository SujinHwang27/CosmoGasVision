"""[U-04] Stage-1 R5: one-time global delta_F scale measurement.

Spec SS2(b): "measure the global delta_F std over a sample of sightlines
ONCE, pick a round number, PIN it in code + record the measurement in the
Stage-1 record JSON". Sample of record: first 1024 sightlines x all 4
physics, z=0.3, redshift-space tau via the canonical loader
(``SherwoodLoader.load_sightlines`` -> ``tau_h1``), flux via the pipeline
convention F = exp(-tau) (``src.data.sightline_rasterizer.flux_decrement``).

Writes experiments/unet-inversion/artifacts/stage1/r5_delta_f_scale.json.
Run:  PYTHONPATH=. .venv/bin/python -u scripts/u04_s1_delta_f_scale.py
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from src.data.loader import SherwoodLoader
from src.data.sightline_rasterizer import DELTA_F_SCALE, flux_decrement

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/unet-inversion/artifacts/stage1/r5_delta_f_scale.json"
N_SAMPLE_RAYS = 1024
Z = 0.3


def main() -> None:
    loader = SherwoodLoader(str(REPO / "Sherwood"))
    per_physics = {}
    pooled = []
    for pid in (1, 2, 3, 4):
        sl = loader.load_sightlines(pid, Z)
        tau = np.asarray(sl["tau_h1"][:N_SAMPLE_RAYS], dtype=np.float64)
        del sl
        gc.collect()
        df = flux_decrement(tau)
        per_physics[f"P{pid}"] = {
            "n_rays": int(df.shape[0]),
            "n_bins": int(df.shape[1]),
            "delta_f_std": float(df.std()),
            "delta_f_mean": float(df.mean()),
            "mean_flux": float(np.mean(1.0 - df)),
            "percentiles_50_90_99": [
                float(v) for v in np.percentile(df, [50, 90, 99])
            ],
        }
        pooled.append(df)
        print(f"P{pid}: std={per_physics[f'P{pid}']['delta_f_std']:.6f}")
    allc = np.concatenate(pooled)
    std = float(allc.std())
    payload = {
        "rung": "R5 — global delta_F scale measurement + pin (A5)",
        "spec": "u04_stage1_ratification.md SS2(b), commit 58ac831",
        "flux_convention": "F = exp(-tau_h1) (redshift-space half, [D-24]); "
                           "reused from src/analysis/p_flux.py / "
                           "flux_power.py; delta_F = 1 - F",
        "sample": f"first {N_SAMPLE_RAYS} sightlines x 4 physics, z={Z}",
        "per_physics": per_physics,
        "pooled": {
            "delta_f_std": std,
            "delta_f_mean": float(allc.mean()),
            "one_over_std": 1.0 / std,
        },
        "pinned_scale": {
            "DELTA_F_SCALE": DELTA_F_SCALE,
            "where": "src/data/sightline_rasterizer.py",
            "rule": "round number nearest 1/std(pooled delta_F); "
                    "GLOBAL constant; per-crop/data-dependent normalization "
                    "FORBIDDEN (spec SS2(b))",
            "deviation_note": "spec guessed '~x50 expected'; measured "
                              "1/std = 12.19 (std is spike-dominated: p50 "
                              "delta_F ~ 0.0016, p99 ~ 0.4) -> pin 12.5. "
                              "Measurement supersedes the guess per SS2(b) "
                              "('final value pinned by core-implementer').",
        },
        "consistency_check_pass": bool(abs(1.0 / std - DELTA_F_SCALE) / DELTA_F_SCALE < 0.25),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}; pooled std={std:.6f}, pin={DELTA_F_SCALE}")


if __name__ == "__main__":
    main()

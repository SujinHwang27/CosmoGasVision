"""Pipeline entry for the unet-inversion track (amortized 3D U-Net inversion).

Stage 0 scaffold: MLflow-wired skeleton only. No model, no data loading yet.
Run from repo root with PYTHONPATH=. ; falls back to a no-op context when the
MLflow tracking server is unreachable (repo convention, see CLAUDE.md).
"""

import os
import sys
from contextlib import nullcontext

EXPERIMENT_NAME = "CosmoGasVision/unet-inversion"
RUN_NAME = "Stage1-Bootstrap"
MANDATORY_TAGS = {
    "model_type": "unet3d",
    "stage": "1-bootstrap",
    "physics_id": os.environ.get("COSMOGAS_PHYSICS_ID", "1"),
    "redshift": os.environ.get("COSMOGAS_REDSHIFT", "0.3"),
}


def _mlflow_run():
    """Return an active MLflow run context, or nullcontext if unreachable."""
    try:
        import mlflow

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow.start_run(run_name=RUN_NAME, tags=MANDATORY_TAGS)
    except Exception as exc:  # noqa: BLE001 — any tracker failure degrades to local run
        print(f"[pipeline] MLflow unavailable ({exc!r}); continuing without tracking.", flush=True)
        return nullcontext()


def main() -> int:
    print(f"[pipeline] {EXPERIMENT_NAME} :: {RUN_NAME}", flush=True)
    with _mlflow_run():
        # Stage 1 (pair-manufacture plumbing) lands here:
        #   1. crop x ray-pattern sampler over the [D-49] train region
        #   2. flux rasterizer -> (2, N, N, N) input channels
        #   3. truth log-rho crop target via SherwoodLoader.extract_rho_crops_split
        print("[pipeline] Stage 0 scaffold — nothing to run yet.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

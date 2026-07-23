"""Pipeline entry for the diffusion-posterior track (generative prior + posterior sampling).

Stage 0 scaffold: MLflow-wired skeleton only. Track is DORMANT pending the
[DP-01] activation gate (see LEDGER §3). Run from repo root with PYTHONPATH=. ;
falls back to a no-op context when the MLflow tracking server is unreachable
(repo convention, see CLAUDE.md).
"""

import os
import sys
from contextlib import nullcontext

EXPERIMENT_NAME = "CosmoGasVision/diffusion-posterior"
RUN_NAME = "Stage1-Bootstrap"
MANDATORY_TAGS = {
    "model_type": "diffusion3d",
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
        # Stage 1 (diffusion prior over truth log-rho crops) lands here once
        # the [DP-01] activation gate fires:
        #   1. crop sampler over the [D-49] train region (truth-only, no flux)
        #   2. 3D denoiser + noise schedule
        #   3. sample-fidelity eval vs held-out truth statistics
        print("[pipeline] Stage 0 scaffold — track dormant per LEDGER [DP-01].", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pipeline entry for the feedback-latent track (conditioned neural field).

Stage 0 scaffold: MLflow-wired skeleton only. No model, no data loading yet.
Run from repo root with PYTHONPATH=. ; falls back to a no-op context when the
MLflow tracking server is unreachable (repo convention, see CLAUDE.md).
"""

import os
import sys
from contextlib import nullcontext

EXPERIMENT_NAME = "CosmoGasVision/feedback-latent"
RUN_NAME = "Stage1-Bootstrap"
MANDATORY_TAGS = {
    "model_type": "conditioned-field",
    "stage": "1-bootstrap",
    "physics_id": os.environ.get("COSMOGAS_PHYSICS_ID", "1,2,3,4"),
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
        # Stage 1 (Phase A, post-ratification) lands here:
        #   1. conditioned field f(x, z_p) — SIREN/hash-grid trunk + 4 learnable z_p
        #   2. truth-supervised joint fit over all 4 variants (crop-scale first)
        #   3. anti-collapse gates inherited from the exp/nerf failure taxonomy
        print("[pipeline] Stage 0 scaffold — awaiting prior-art sweep + PI ratification.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

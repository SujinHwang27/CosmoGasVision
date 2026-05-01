---
name: mlflow-run
description: Wraps an execution in the project's canonical MLflow contract — hierarchical experiment name, stage-prefixed run name, mandatory tag set (model_type, stage, physics_id, redshift), dotenv-loaded AWS auth, and nullcontext fallback when the tracking server is unreachable. Trigger when wiring a new mlflow.start_run call, when reviewing an existing one for tag/name compliance, or when porting a flat-named legacy script (e.g. experiments/3dgs_baseline/pipeline.py) to the hierarchical scheme. Do not trigger for small hyperparameter tweaks inside an already-compliant run.
---

# MLflow run contract (CosmoGasVision)

Every MLflow run in this repo follows the same shape so the tracking server stays queryable across tracks and stages. Apply this contract verbatim when wiring or modifying a run.

## Naming

- **Experiment name** (hierarchical): `CosmoGasVision/<Track>` — e.g. `CosmoGasVision/NeRF`, `CosmoGasVision/3DGS`. The `<Track>` segment matches the active `exp/<name>` branch basename.
- **Run name** (stage-prefixed): `Stage<N>-<ShortDescription>` — e.g. `Stage2a-PhysicsIntegratorValidation`, `Stage1-Bootstrap`. Pascal-case, no spaces.

## Mandatory tags

Every run **must** set these four tags at start:

| Tag | Type | Example |
|---|---|---|
| `model_type` | string | `nerf`, `3dgs` |
| `stage` | string | `1`, `2a`, `2b`, `3` |
| `physics_id` | string | `P1`, `P2`, `P3`, `P4` (Sherwood physics variant) |
| `redshift` | string | `0.3`, `2.0` |

Additional tags are fine; these four are required for LEDGER cross-referencing.

## Boilerplate (use as-is)

```python
import os
from contextlib import nullcontext
from dotenv import load_dotenv

load_dotenv()  # Inject AWS_* for S3 artifact upload (fixes boto3 INTERNAL_ERROR; see 786575f)

try:
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(f"CosmoGasVision/{TRACK}")
    run_ctx = mlflow.start_run(run_name=f"Stage{STAGE}-{DESCRIPTION}")
except Exception as e:
    print(f"[mlflow] unreachable, falling back to nullcontext: {e}")
    run_ctx = nullcontext()

with run_ctx as run:
    if hasattr(run, "info"):
        mlflow.set_tags({
            "model_type": MODEL_TYPE,
            "stage": STAGE,
            "physics_id": PHYSICS_ID,
            "redshift": REDSHIFT,
        })
    # ... training / evaluation body
```

## Post-run

After the run completes, capture the `run_id` and append it to the LEDGER's **Section 6 (Visualization & Artifacts)** for the active branch — this is what makes a run discoverable later. Use the `ledger-update` skill for the write.

## Anti-patterns

- Run name without a stage prefix → blocks chronological filtering.
- Skipping any of the four mandatory tags → orphans the run from LEDGER queries.
- Hard-coding the tracking URI → use `MLFLOW_TRACKING_URI` env so dev/prod switch cleanly. The server is launched locally via `scripts/start_mlflow.ps1`; the URI default is `http://127.0.0.1:5000`.
- Letting an unreachable server crash the script → always wrap in the try/except + `nullcontext` fallback above; CLAUDE.md flags this as a silent-exit cause.

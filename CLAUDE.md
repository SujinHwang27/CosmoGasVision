# CLAUDE.md

CosmoGasVision: 3D IGM gas density reconstruction from Lyman-alpha sightlines (Sherwood Simulation Suite, Bolton+ 2017). Two tracks — **NeRF** (primary, continuous MLP IGM field) and **3DGS** (baseline, explicit Gaussians).

**Before starting work, read the active branch's LEDGER first**: `experiments/<branch_basename>/LEDGER.md`. It is the single source of truth for stage status, decisions (D-XX), data lineage, and next steps.

## Tooling
- Package manager: **uv** (`uv add`, `uv run`, `uv sync`). Don't use `pip install` directly — it desyncs `uv.lock`.
- Quote version specifiers: `uv add "fsspec>=2024.1.0"` (otherwise the shell creates junk redirect files).
- `python -u` for unbuffered output; set `PYTHONPATH=.` from the repo root so `src.` imports resolve.
- Sim binaries (`Sherwood/`, `SherwoodIGM_gal/`) are kept locally read-only — don't modify in place.

## Source layout
- `src/data/` — loaders, validation, all sim-data I/O.
- `src/models/` — neural architectures (`nerf.py`; future `gaussian_field.py`).
- `src/rendering/` — differentiable integrators, projection (planned).
- `experiments/<name>/pipeline.py` — execution entry per track.
- `experiments/<name>/LEDGER.md` — command center (Pulse / Logic / Data / History).
- `experiments/<name>/artifacts/` — DVC-tracked outputs.

## Astrophysical conventions
- Coordinates: comoving kpc/h within a 60,000 kpc/h (60 Mpc/h) box; normalize to unit cube `[0, 1]` for MLP input.
- Units: km/s, K, ρ/⟨ρ⟩, X_HI ∈ [0, 1], τ ≥ 0.
- Every loader/preprocessor must include `_validate_data` sanity checks (positivity bounds, X_HI range, NaN sanitization).
- All forward-pass operations must be `torch.autograd`-compatible — no detached NumPy in the training loop.

## Experiment workflow (mandatory)
- Each methodology lives on its own `exp/<name>` branch with `experiments/<name>/{LEDGER.md, pipeline.py, artifacts/}`.
- MLflow experiment names are hierarchical: `CosmoGasVision/<Track>` (e.g. `CosmoGasVision/NeRF`).
- Every run name is stage-prefixed: `Stage2a-PhysicsIntegratorValidation`. Mandatory tags: `model_type`, `stage`, `physics_id`, `redshift`.
- DVC-track any artifact > 10 MB (`.npy`, `.pt`, `.dat`, `.html`, `.mp4`).

## Git conventions
- Branches: `main` (stable), `exp/<name>` (per methodology), `feat/<name>`, `refactor/<name>`. Hyphenated names.
- Conventional commit prefixes: `feat:`, `fix:`, `chore:`, `paper:`, `docs:`, `refactor:`.

## Failure handling
If the same command fails 3 times with no progress, **stop and surface the issue to the user** with: exact commands run, observed output, hypothesized cause, proposed fix. Don't keep retrying. Don't fabricate an "Error Report" file unless asked.

## Subagents and commands
- `.claude/agents/` — specialized agents (data-engineer, core-implementer, infrastructure-manager, project-architect, support-researcher, latex-author). Dispatched by description match — see each file for triggers.
- `.claude/commands/` — slash commands: `/new-experiment <name>`, `/update-ledger`.

## Toolchain pitfalls (cheatsheet)
- `ImportError: fsspec_loop` → `fsspec`/`dvc` version skew. Stay on `fsspec>=2024.3.1` for modern DVC.
- `'Repo' object has no attribute 'stage'` → upgrade `dvc dvc-s3 dvc-data dvc-objects scmrepo` together.
- `AttributeError: _DIR_MARK` → upgrade DVC to 3.x or pin `pathspec<1.0.0`.
- Silent script exit (code 0, no output) → usually `import mlflow` hang. Test imports sequentially with `print` diagnostics; run with `python -u`; verify `MLFLOW_TRACKING_URI` is reachable; ensure script falls back to `nullcontext` when MLflow is unavailable.

## Remote services
- MLflow tracking URI: `http://127.0.0.1:5000` — **served locally by this machine**. Launch with `pwsh scripts/start_mlflow.ps1`. Backend: SQLite (`mlflow.db`, gitignored). Artifacts: `s3://cosmo-gas-vision-storage/mlflow-artifacts`. Override with `MLFLOW_TRACKING_URI` env var; loaded from `.env`.
- DVC remote: `s3://cosmo-gas-vision-storage/dvc-data`.
- The previous EC2 tracker (`44.201.176.18:5000`) is decommissioned. Run IDs from before this migration are still recorded in each track's LEDGER §6 but their UI links no longer resolve.

---
name: juno-hpc
description: Wraps a CosmoGasVision compute submission to UTD's Juno HPC cluster (SLURM, A30/H100 GPU partitions) — login + storage discipline (home/work/scratch with 45-day purge), partition + sbatch directive selection, conda-on-Juno environment bring-up, the canonical Stage 2b job-script template, and the MLflow file-store round-trip back to the local tracker. Trigger when the user says "submit on Juno", "launch on HPC", or "dispatch Batch N on Juno"; when porting a SageMaker `sagemaker_stage2b_launch.py`-style invocation to Juno; or when wiring a new `scripts/submit_juno_*.sh` / sbatch script. Do not trigger for SageMaker, EC2, or anything outside the UTD HPC environment.
---

# Juno HPC submission contract (CosmoGasVision)

Juno is the alternate compute target for Stage 2b production sweeps (Batch 2/3/4) once the SageMaker cost-survey envelope is exhausted. This skill codifies the access, storage, partition, and MLflow contract so a Juno dispatch is a drop-in replacement for the SageMaker launcher with no scientific drift.

Source of truth for cluster state: <https://hpc.utdallas.edu/systems-resources/juno/> and the Spring 2026 v13 orientation deck. Re-verify partition tables before a multi-batch dispatch — H200 expansion (52 new GPUs) lands ~July 2026 and will add new partitions.

## Access (one-time per user/machine)

- **Account request**: via UTD Atlas service catalog. PI sponsorship required. Email `hpc@utdallas.edu` if blocked.
- **Login host**: `juno.utdallas.edu` (resolves to one of `juno-l-01/02/03`).
- **Network requirement**: must be on UTD wired Ethernet, CometNet WiFi, or UTD VPN. Off-campus without VPN will not connect.
- **Credentials in `.env`** (gitignored; same convention as `SAGEMAKER_ROLE_ARN`):

  ```bash
  JUNO_NETID=<utd-netid>
  JUNO_HOST=juno.utdallas.edu
  JUNO_WORK=/home/<utd-netid>/work/CosmoGasVision
  JUNO_SCRATCH=/home/<utd-netid>/scratch
  ```

  Load with `set -a; source .env; set +a` at the top of any host-side script that needs them; `dotenv.load_dotenv()` for Python (matches `mlflow-run` skill pattern).

- **Login command**:

  ```bash
  ssh "${JUNO_NETID}@${JUNO_HOST}"
  ```

  Optional ergonomics — add a `Host juno` entry in `~/.ssh/config` so `ssh juno` Just Works without env vars:

  ```sshconfig
  Host juno
      HostName juno.utdallas.edu
      User <utd-netid>
      ServerAliveInterval 60
      ServerAliveCountMax 3
  ```

- **Open OnDemand** (web UI for file browse / VS Code / Jupyter): <https://juno-ood.hpcre.utdallas.edu/>. Useful for ≤10 GB ad-hoc transfers; not suitable for the full Sherwood mirror.
- **Support**: `circ-assist@utdallas.edu` (general HPC) or `hpc@utdallas.edu` (Juno-specific).

## Storage layout — pick the right filesystem

| Path | Quota | Backed up | Purge | Use for |
|---|---|---|---|---|
| `~` (home) | 50 GB | daily | none | login scripts, configs, small inputs only — **never** for batch I/O |
| `~/work` | 1 TB | daily | none | user-installed software (conda envs), large source tarballs, results to keep |
| `/groups/<pi-name>` | 1 TB+ | daily | none | shared group software/data/results |
| `~/scratch` | 30 TB soft | **never** | **45 days no-access → deleted** | all batch I/O — sightlines, GT τ, checkpoints, MLflow file store |

Quota check: `mfsgetquota -H <directory>` (also reports inode quota).

**Hard rules**:
1. Compute jobs read/write to `~/scratch`, never to `~` or `/groups`. Scratch is up to 10× faster for large I/O.
2. Job script must follow `copy in → compute → copy out → clean up` to avoid a silent purge wiping in-flight checkpoints.
3. Anything that must survive 45 days of no-access goes to `~/work` or `/groups/<pi>` post-job. The Stage 2b checkpoint at `step_*.pt` qualifies; intermediate optimizer state does not.

## Partition selection for Stage 2b

Per [D-23] post-OOM table, peak VRAM at chunk_size=256 is ~11.3 GB across tiers 2/3/4. A single A30 (24 GB) is sufficient and typically faster to schedule than H100. Use H100 only when concurrent dispatch on `a30` is queue-bound or tier 4 needs absolute wallclock minimization.

| Tier | n_rays | est. peak VRAM | recommended partition | rationale |
|---|---|---|---|---|
| T2 | 256 | ~11 GB measured | `a30` | 24 GB per physical A30 = ~2× headroom |
| T3 | 1024 | ~11 GB projected | `a30` | same |
| T4 | 16384 | ~11 GB projected | `a30` or `h100` | wallclock-dominated; H100 helps if quota allows |

Do **not** use the virtual-GPU partitions (`a30-2.12gb`, `a30-4.6gb`, `h100-2.47gb`) for Stage 2b — the Voigt-intermediate transient can spike past the 12 GB / 47 GB virtual cap even when steady-state fits.

Partition limits to remember:

- **Max running jobs per user across GPU partitions: 4.** Sequential dispatch order (P1→P2→P3→P4 per [D-18]) is compatible with this; 4-parallel within-tier dispatch ([D-18] amendment 2026-05-04) saturates the cap exactly.
- **Max wallclock: 2 days (`2-00:00:00`)** on `a30`/`h100`. Tier 4's projected ~98 min/cell × 4 cells fits comfortably; budget for queue wait separately.
- **Default memory if unspecified: 64 GB** — always set `--mem` explicitly. Stage 2b needs ~32 GB for sightline + GT τ tensors.

Job-priority Fairshare resets monthly and heals to 1.0 in two weeks of disuse — bunching all four batches at month-end gives lower priority than spreading across the month.

## Environment bring-up (one-time per project clone)

Juno's default Python comes from the system; use Miniconda for an isolated env. The project uses `uv`, but `uv` should run **inside** a conda env on Juno so the Python interpreter is owned by miniconda.

```bash
# Login node, one-time. ${JUNO_WORK} from .env resolves to /home/<netid>/work/CosmoGasVision
module load miniconda
conda create -p ~/work/envs/cosmogasvision python=3.12 -y
conda init bash && source ~/.bashrc
conda activate ~/work/envs/cosmogasvision

# Project sources in ~/work (NOT ~/scratch — scratch will purge)
mkdir -p "$(dirname "${JUNO_WORK}")"
git clone <repo-url> "${JUNO_WORK}"
cd "${JUNO_WORK}"
pip install uv
uv sync  # honors uv.lock from CLAUDE.md tooling section
```

Sherwood data (sightlines + `tauH1_*.dat`) are ~1.6 GB per physics; mirror to `~/scratch/sherwood/` before each dispatch. They will be purged after 45 days of no-access — set a calendar reminder to `touch` them if a long pause is planned, or re-mirror from S3 (`s3://cosmo-gas-vision-storage/sherwood/`).

## Canonical Stage 2b sbatch script

Save as `scripts/submit_juno_stage2b.sh`. One sbatch script per cell, one cell per `(physics, tier, seed)`.

Inside the script, `${JUNO_WORK}` and `${JUNO_SCRATCH}` are read from the user's Juno-side `.env` (or the login `~/.bashrc` if mirrored there); they resolve to absolute paths so the script works regardless of which login node SLURM dispatched from.

```bash
#!/bin/bash
#SBATCH --job-name=Stage2b-P${PHYSICS_ID}-N${N_RAYS}-S${SEED}
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Load Juno path conventions (JUNO_WORK, JUNO_SCRATCH) from the project .env
# checked into ~/work/CosmoGasVision/.env on the cluster. Quietly no-op if missing.
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a
  source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"
  set +a
fi
: "${JUNO_WORK:=$HOME/work/CosmoGasVision}"
: "${JUNO_SCRATCH:=$HOME/scratch}"

# --- 1. Copy in ---
RUN_TAG="P${PHYSICS_ID}-N${N_RAYS}-S${SEED}-$(date +%s)-$(uuidgen | cut -c1-6)"
RUN_DIR="${JUNO_SCRATCH}/stage2b/${RUN_TAG}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
ln -s "${JUNO_SCRATCH}/sherwood" Sherwood   # sightlines + tauH1_*.dat (mirrored once, kept warm)

# --- 2. Environment ---
module load miniconda cuda
conda activate "${JUNO_WORK%/CosmoGasVision}/envs/cosmogasvision"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

# MLflow runs to a local file store inside scratch; we ship it back at the end.
export MLFLOW_TRACKING_URI="file://${RUN_DIR}/mlflow"
export GIT_PYTHON_REFRESH=quiet              # silences GitPython deprecation in logs

# --- 3. Compute ---
python -u experiments/nerf/pipeline.py \
    --physics "${PHYSICS_ID}" \
    --n_rays "${N_RAYS}" \
    --microbatch "${MICROBATCH}" \
    --accum_steps "${ACCUM_STEPS}" \
    --max_steps "${MAX_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --seed "${SEED}" \
    --stage_tag "2b-juno"

# --- 4. Copy out ---
DEST="${JUNO_WORK%/CosmoGasVision}/stage2b_results/${RUN_TAG}"
mkdir -p "${DEST}"
cp -r mlflow checkpoints "${DEST}/"

# --- 5. Cleanup ---
cd "${HOME}"
rm -rf "${RUN_DIR}"
```

Submit per-cell with environment overrides (run from inside `${JUNO_WORK}` on the login node):

```bash
sbatch \
  --export=ALL,PHYSICS_ID=2,N_RAYS=256,MICROBATCH=1024,ACCUM_STEPS=1,MAX_STEPS=25000,WARMUP_STEPS=1000,SEED=0 \
  scripts/submit_juno_stage2b.sh
```

Note on `.env` propagation: the host-side `.env` (this Windows machine) holds `JUNO_NETID`/`JUNO_HOST` for SSH and `rsync` from here. The cluster-side `.env` (committed to `${JUNO_WORK}/.env` after the one-time clone) holds `JUNO_WORK`/`JUNO_SCRATCH` and any cluster-specific overrides. Keep them disjoint to avoid leaking AWS keys onto the shared cluster filesystem — copy only the `JUNO_*` block, plus `MLFLOW_TRACKING_URI=file:///dev/null` placeholder so the in-script export above isn't shadowed.

[D-23] tier table → sbatch param mapping:

| Tier | `N_RAYS` | `MICROBATCH` | `ACCUM_STEPS` | `MAX_STEPS` | `WARMUP_STEPS` |
|---|---|---|---|---|---|
| T1 | 64    | 1024 | 1  | 50000 | 1000 |
| T2 | 256   | 1024 | 1  | 25000 | 1000 |
| T3 | 1024  | 256  | 4  | 12500 | 500  |
| T4 | 16384 | 256  | 64 | 12500 | 500  |

## MLflow round-trip back to local tracker

The local MLflow tracker (`http://127.0.0.1:5000`, this machine, see CLAUDE.md "Remote services") is unreachable from Juno. The job logs into a `file://` store on scratch, ships the directory back to `~/work/stage2b_results/<RUN_TAG>/mlflow/`, and a host-side replay imports the run preserving tags, params, and per-step metric history.

After the job completes:

1. **Pull from Juno to host** (host-side `.env` provides `JUNO_NETID` and `JUNO_HOST`):

   ```bash
   set -a; source .env; set +a
   rsync -avzP "${JUNO_NETID}@${JUNO_HOST}:${JUNO_WORK%/CosmoGasVision}/stage2b_results/<RUN_TAG>/" \
       cloud_runs/<RUN_TAG>/
   ```

2. **Replay into local tracker** using the existing SageMaker importer (`scripts/sagemaker_stage2b_import_mlflow.py`) — it operates on a local `mlruns/` tree and is platform-agnostic. Pass the pulled `mlflow/` directory as the source. If the importer is SageMaker-tarball-specific in current form, generalize it; do not write a parallel Juno-only importer.

3. **Append `run_id` to LEDGER §6** in a new `Stage 2b Juno cost-survey` subsection (or `Stage 2b Juno production` if post-quota), per the [D-23] two-tier framework. Use the `ledger-update` skill for the write. Tag set must satisfy the `mlflow-run` skill contract: `model_type=nerf`, `stage=2b`, `physics_id=P{1..4}`, `redshift=0.3`, plus `compute=juno`.

## Multi-batch dispatch pattern (Batch 2 / Batch 3 / Batch 4)

Stage 2b's [D-18] within-tier amendment is "4-parallel within tier, sequential across tiers". This maps cleanly onto Juno's 4-job GPU cap:

```
Batch 2 (T2 × {P1,P2,P3,P4}):  dispatch all 4 sbatch jobs at t0 → wait squeue clears
Batch 3 (T3 × {P1,P2,P3,P4}):  dispatch all 4 only after Batch 2 fully complete + LEDGER §6 logged
Batch 4 (T4 × {P1,P2,P3,P4}):  post-quota only — sanity-check VRAM with one cell first
```

Use `squeue --me` to monitor; `scancel <jobid>` to abort. `sshare` to check Fairshare value before dispatching the next batch — if your share dropped below 0.5, expect long wait times and consider deferring.

## Anti-patterns

- **Running training off `~` or `~/work`** → I/O bottleneck against backup-quality storage; orientation deck is explicit ("Do not use for batch jobs IO needs").
- **Forgetting `--gres=gpu:1`** → job lands on CPU and silently runs at ~100× wallclock.
- **No explicit `--mem`** → defaults to 64 GB (probably fine, but document intent).
- **Submitting from login node without `sbatch`** (i.e. running `python pipeline.py` directly on `juno-l-0X`) → the 8 GB login-node RAM cap and shared-CPU policy will OOM or get the process killed by HPC staff. Always submit, or `salloc` for interactive debugging.
- **Leaving training data in `~/scratch` between batches** without a `touch` refresh → 45-day purge wipes it mid-sweep. Either keep an active access pattern or stage from `~/work` per job.
- **Hardcoding the local MLflow URI inside the SLURM script** → unreachable from Juno; use `file://` store + post-job replay as in §"MLflow round-trip" above. Same `nullcontext` fallback as `mlflow-run` skill applies if the file-store init fails.
- **Skipping the copy-out step** → checkpoints die with the scratch purge; re-running costs another batch budget.

## Triage cheatsheet

| Symptom | Probable cause | Fix |
|---|---|---|
| `sbatch: error: Memory specification can not be satisfied` | partition mem cap exceeded | drop `--mem` to ≤512 G on h100, ≤1024 G on a30 |
| Job sits in `PD` with `(Priority)` | low Fairshare or 4-job cap hit | `sshare`; wait for sibling jobs to finish |
| Job sits in `PD` with `(Resources)` | partition full | `sinfo` to see capacity; consider switching `a30 ↔ h100` |
| Silent OOM mid-training (no exception) | Voigt-intermediate transient on `a30-*.gb` virtual GPU | resubmit on physical `a30` or `h100` |
| `~/scratch` data missing | 45-day purge fired | re-mirror from `s3://cosmo-gas-vision-storage/sherwood/` |
| `module: command not found` | login script regression | `source /etc/profile.d/modules.sh` and re-login |
| MLflow file-store empty after job | `MLFLOW_TRACKING_URI` not exported before `python` invocation | check `env` block in sbatch script; rerun with `set -x` |

If a Juno-side error persists across 3 attempts, surface to the user with the trial log per the CLAUDE.md failure-handling rule. Email `hpc@utdallas.edu` only as the human-side fallback when the failure looks like an HPC-infrastructure issue rather than a job-script bug.

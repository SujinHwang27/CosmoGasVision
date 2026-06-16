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

- **SSH key auth (required for the dispatch workflow)** — UTD password login triggers Duo MFA on every connection, which makes the rsync + sbatch loops unworkable. Set up key auth once per workstation:

  ```bash
  # 1. Generate a Juno-specific Ed25519 key (no passphrase = automation-friendly).
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_juno -N "" -C "cosmogasvision-juno-$(date +%Y%m%d)"

  # 2. Install the public half on Juno — prompts for UTD password + Duo ONCE.
  #    -o IdentitiesOnly=no overrides the Host juno block we add in step 3.
  ssh-copy-id -i ~/.ssh/id_ed25519_juno.pub -o IdentitiesOnly=no <netid>@juno.utdallas.edu

  # 3. Add a Host alias so `ssh juno` (and `rsync ... juno:...`) just work.
  cat >> ~/.ssh/config <<EOF
  Host juno
      HostName juno.utdallas.edu
      User <netid>
      IdentityFile ~/.ssh/id_ed25519_juno
      IdentitiesOnly yes
      ServerAliveInterval 60
      ServerAliveCountMax 3
  EOF
  chmod 600 ~/.ssh/config ~/.ssh/id_ed25519_juno
  ```

  Verify: `ssh juno "hostname; whoami"` should succeed silently with no prompt.

- **Login commands** — both forms are supported; use `ssh juno` for daily work and the env form for portable scripts:

  ```bash
  ssh juno                              # via ~/.ssh/config alias (recommended)
  ssh "${JUNO_NETID}@${JUNO_HOST}"      # via .env (portable across machines)
  ```

- **Open OnDemand** (web UI for file browse / VS Code / Jupyter): <https://juno-ood.hpcre.utdallas.edu/>. Useful for ≤10 GB ad-hoc transfers; not suitable for the full Sherwood mirror.
- **VS Code Remote-SSH — use the dedicated node, NOT the login node** (HPC team notice 2026-06-16, Juno-only, experimental). VS Code's Remote-SSH spawns many background/zombie processes that destabilized the regular login nodes, so its access there was restricted. The sanctioned path is a separate host:
  ```bash
  # Remote-SSH target (VS Code ONLY — do not use as a general login/compute node):
  #   Server:   juno-vscode.utdallas.edu
  #   Username: <netid>   (UTD password, or SSH keys per the UG SSH-keys guide)
  ssh <netid>@juno-vscode.utdallas.edu
  ```
  Hard limits / caveats:
  - **VS Code Remote-SSH only.** Do NOT use `juno-vscode` as a regular node, and do NOT run programs/training on it — all compute still goes through `sbatch` to the a30/h100 **compute nodes**. (Our dispatch workflow is unchanged: `ssh juno` → `sbatch`; `juno-vscode` is only for editing in VS Code.)
  - Memory + max-process limits are enforced (`ulimit -a` to inspect); exceeding them kills the session.
  - Testing phase — expect hangs/disconnects; Remote-SSH zombie processes are reaped by routine cleanup scripts, so the session may drop and restart its remote processes.
  - Available on **Juno only** (not Ganymede 2). SSH-key setup: <https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/getting-started/ssh-keys/>.
- **Support**: `circ-assist@utdallas.edu` (general HPC) or `hpc@utdallas.edu` (Juno-specific).

## Keeping the control machine awake (macOS `caffeinate`)

The Juno **job** is server-side — an `sbatch` job keeps running on the compute node whether or not your laptop sleeps. What dies when the Mac goes idle is anything **local and long-lived**: a polling watcher on the step-gate, an open `rsync` pull, an interactive `ssh`/`salloc` session, or the UTD VPN tunnel itself (macOS suspends the network stack on system sleep, and most VPN clients do not auto-reconnect on wake). On this machine the idle sleep timer is aggressive (`pmset -g` showed `sleep 1` — full system sleep after ~1 min idle), so an unguarded monitoring loop will be cut off within a minute of you stepping away.

Wrap any long-lived local Juno interaction in `caffeinate` so the control machine holds the network up for its duration:

```bash
# Wrap a specific transfer/loop — caffeinate exits when the command exits:
caffeinate -is rsync -avzP "juno:${JUNO_WORK%/CosmoGasVision}/stage2b_results/<RUN_TAG>/" cloud_runs/<RUN_TAG>/

# Hold the machine awake while a local watcher/poll loop runs in this shell:
caffeinate -is squeue --me     # or any monitor command; Ctrl-C releases it

# Pin caffeinate to a background PID (e.g. a detached watcher) and let it self-release:
<your-watcher> &              # capture its PID
caffeinate -is -w $!          # stays awake until that watcher exits
```

Flags: `-i` prevents idle **system** sleep (the one that drops the tunnel), `-s` prevents sleep on AC power, `-w <pid>` ties the awake-window to a process so it self-releases. Scope `caffeinate` to the command rather than disabling sleep system-wide (`sudo pmset -c sleep 0`) — the wrapper reverts automatically and won't leave the laptop awake on battery after the dispatch is done.

Rule of thumb: **if a Juno-related command will outlive you sitting at the keyboard, prefix it with `caffeinate -is`.** Fire-and-forget `sbatch` submissions that return immediately do not need it.

## Storage layout — pick the right filesystem

| Path | Real location | Filesystem | Quota | Backed up | Purge | Use for |
|---|---|---|---|---|---|---|
| `~` (home) | `/home/<netid>` | MFS | 50 GB / 300k inodes | daily | none | login scripts, configs, small inputs only — **never** for batch I/O |
| `~/work` → symlink | `/work/<netid>` (note: NOT `/home/<netid>/work`) | MFS | 1 TB / 3M inodes | daily | none | repo clone, conda envs, `.venv/`, kept results |
| `/groups/<pi-name>` | (varies) | MFS | 1 TB+ | daily | none | shared group software/data/results |
| `~/scratch` → symlink | `/scratch/juno/<netid>` (note: NOT `/scratch/<netid>`) | parallel FS | 30 TB soft | **never** | **45 days no-access → deleted** | batch I/O — sightlines, GT τ, checkpoints, MLflow file store |

Quota check: `mfsgetquota -H <directory>` for `~` and `~/work`. **`~/scratch` is not MFS** — `mfsgetquota` returns "not MFS object"; use `df -h ~/scratch` to see usage. The 30 TB cap is enforced at filesystem level.

Inode-quota footnote: a conda env contains millions of small files; install it under `~/work` (3M inodes), never under `~` (300k will blow within minutes).

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

Juno's default Python comes from the system; use Miniconda to donate a Python interpreter, then let `uv` build a `.venv` next to the project. uv installs the project deps into `.venv/`, **not** into the conda env — the conda env's only job is to provide the interpreter `.venv` symlinks against. (Verified 2026-05-06: `.venv/bin/python` → `/work/<netid>/envs/cosmogasvision/bin/python3.12`.)

```bash
# Login node, one-time.
# Use `eval "$(conda shell.bash hook)"` instead of `conda init bash` to avoid
# polluting ~/.bashrc; the hook only affects the current session.
module load miniconda/24.11.1
eval "$(conda shell.bash hook)"
conda create -p /work/<netid>/envs/cosmogasvision python=3.12 -y
conda activate /work/<netid>/envs/cosmogasvision

# Project sources in /work (NOT /scratch — scratch will purge after 45 days)
git clone git@github.com:<gh-user>/CosmoGasVision.git "${JUNO_WORK}"
cd "${JUNO_WORK}"
pip install uv
export UV_LINK_MODE=copy             # avoids cross-FS hardlink warning between cache and .venv
uv sync                              # creates ./.venv with all project deps
```

**Driver / CUDA constraint** ([D-XX] candidate — see LEDGER): Juno's compute-node NVIDIA driver is `550.163.01` (probed 2026-05-06 on `g-01-01`/A30), which supports CUDA 12.4 maximum. The project's `pyproject.toml` pins `torch>=2.8.0`, but PyTorch 2.7+ ships only `cu126`/`cu128` wheels needing driver 560+/570+. **Force-replace torch with the cu124 wheel after `uv sync`**:

```bash
export VIRTUAL_ENV="${JUNO_WORK}/.venv"
uv pip install --reinstall \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.6.0+cu124 True   (cuda.is_available is False on login node — verify in salloc)
```

This downgrade is **Juno-side only**; the host machine's SageMaker venv keeps `torch 2.8+` per the lock. Re-running `uv sync` on Juno would clobber the override — re-run the `uv pip install` above any time the lock changes.

**Awscli** is needed for the Sherwood mirror but is not in `pyproject.toml`. Install it the same way:

```bash
uv pip install awscli   # requires VIRTUAL_ENV exported as above; silent no-op otherwise
```

**Sherwood data** (sightlines + `tauH1_*.dat`) are ~1.6 GB per physics, ~6.4 GB total. Mirror to `${JUNO_SCRATCH}/sherwood/` once before the first dispatch:

```bash
set -a; source "${JUNO_WORK}/.env"; set +a
mkdir -p "${JUNO_SCRATCH}/sherwood"
.venv/bin/aws s3 sync s3://cosmo-gas-vision-storage/sherwood/ "${JUNO_SCRATCH}/sherwood/"
```

This requires the `cgv-juno-reader` IAM keys in `${JUNO_WORK}/.env` (mode 600) — see "AWS credentials on Juno" below. Mirror is purged after 45 days of no-access; either re-mirror or `touch` the files periodically if a long pause is planned.

## AWS credentials on Juno

Use a **scoped IAM user** (`cgv-juno-reader`) with read-only on `s3://cosmo-gas-vision-storage/sherwood/*`. Do NOT copy the host's `cgv-infrastructure-user` keys to the cluster — those have full SageMaker/ECR/S3 power and would multiply blast radius if Juno is compromised.

Provision in AWS console (one-time):
1. IAM → Users → Create user `cgv-juno-reader` (programmatic access only).
2. Attach inline policy `CGV-JunoSherwoodReadOnly`:
   ```json
   {"Version":"2012-10-17","Statement":[
     {"Sid":"ListSherwoodPrefix","Effect":"Allow","Action":"s3:ListBucket",
      "Resource":"arn:aws:s3:::cosmo-gas-vision-storage",
      "Condition":{"StringLike":{"s3:prefix":["sherwood/*","sherwood"]}}},
     {"Sid":"ReadSherwoodObjects","Effect":"Allow","Action":"s3:GetObject",
      "Resource":"arn:aws:s3:::cosmo-gas-vision-storage/sherwood/*"}
   ]}
   ```
3. Create access key, paste the pair into `${JUNO_WORK}/.env` alongside the `JUNO_*` block. The `.env` must be mode 600.

If the dispatch ever needs DVC pulls from Juno, extend the policy with `s3:GetObject` on `cosmo-gas-vision-storage/dvc-data/*` and `ListBucket` with that prefix — narrowest-needed scope still applies.

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
# torch 2.6.0+cu124 in .venv bundles its own CUDA 12.4 runtime libraries
# (nvidia-cuda-runtime-cu12, libcudnn, etc.) — module load cuda is NOT needed
# at job runtime. The bundled libs link against the kernel-side driver
# (550.163.01 → CUDA 12.4 max) which is why we pin to cu124 wheels.
source "${JUNO_WORK}/.venv/bin/activate"
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

# --- 4. Copy out + Producer-Consumer Verification (PCV) ---
# pipeline.py writes step_*.pt under experiments/nerf/artifacts/checkpoints/
# (its --checkpoint_dir default), NOT under ./checkpoints/. Cite the producer's
# source-of-truth path; recall-confidence here cost ~30 GPU-hr on 2026-05-08.
# See infrastructure-manager.md "Producer-Consumer Verification" section.
DEST="${JUNO_WORK%/CosmoGasVision}/stage2b_results/${RUN_TAG}"
mkdir -p "${DEST}"

# 1. MLflow file-store — required for downstream metric replay.
[[ -d mlflow ]] || { echo "FATAL: pipeline produced no mlflow/ store" >&2; exit 2; }
cp -r mlflow "${DEST}/"

# 2. Checkpoints — required for [D-13] evaluators downstream. Hard-fail on absence.
CKPT_SRC="experiments/nerf/artifacts/checkpoints"
[[ -d "${CKPT_SRC}" ]] || { echo "FATAL: no checkpoint dir at ${CKPT_SRC}" >&2; exit 3; }
N_CKPT=$(ls -1 "${CKPT_SRC}"/*.pt 2>/dev/null | wc -l)
[[ "${N_CKPT}" -gt 0 ]] || { echo "FATAL: ${CKPT_SRC} contains zero *.pt files" >&2; exit 4; }
mkdir -p "${DEST}/checkpoints"
cp "${CKPT_SRC}"/*.pt "${DEST}/checkpoints/"

# 3. Verify what was copied — visible in the .out log for downstream debugging.
echo "=== copied artifacts ==="
find "${DEST}/mlflow" -maxdepth 4 -type d | head -10
ls -la "${DEST}/checkpoints/"

# --- 5. Cleanup ---
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== done, results at ${DEST} ==="
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

1. **Pull from Juno to host** — uses the `Host juno` alias from `~/.ssh/config` (no password prompts during the loop):

   ```bash
   set -a; source .env; set +a
   rsync -avzP "juno:${JUNO_WORK%/CosmoGasVision}/stage2b_results/<RUN_TAG>/" \
       cloud_runs/<RUN_TAG>/
   ```

   Portable alternative for machines without the `Host juno` alias set up: `rsync -avzP -e "ssh -i ~/.ssh/id_ed25519_juno" "${JUNO_NETID}@${JUNO_HOST}:..."`.

2. **Replay into local tracker** using the existing SageMaker importer (`scripts/sagemaker_stage2b_import_mlflow.py`) — it operates on a local `mlruns/` tree and is platform-agnostic. Pass the pulled `mlflow/` directory as the source. If the importer is SageMaker-tarball-specific in current form, generalize it; do not write a parallel Juno-only importer.

3. **Append `run_id` to LEDGER §6** in a new `Stage 2b Juno cost-survey` subsection (or `Stage 2b Juno production` if post-quota), per the [D-23] two-tier framework. Use the `ledger-update` skill for the write. Tag set must satisfy the `mlflow-run` skill contract: `model_type=nerf`, `stage=2b`, `physics_id=P{1..4}`, `redshift=0.3`, plus `compute=juno`.

## Multi-batch dispatch pattern (Batch 2 / Batch 3 / Batch 4)

Stage 2b's [D-18] within-tier amendment is "4-parallel within tier, sequential across tiers". This maps cleanly onto Juno's 4-job GPU cap:

```
Batch 2 (T2 × {P1,P2,P3,P4}):  dispatch all 4 sbatch jobs at t0 → wait squeue clears
Batch 3 (T3 × {P1,P2,P3,P4}):  dispatch all 4 only after Batch 2 fully complete + LEDGER §6 logged
Batch 4 (T4 × {P1,P2,P3,P4}):  post-quota only — sanity-check VRAM with one cell first
```

Use `squeue --me` to monitor; `scancel <jobid>` to abort. `sshare` to check Fairshare value before dispatching the next batch — if your share dropped below 0.5, expect long wait times and consider deferring. If you monitor from a macOS laptop that sleeps, run the poll loop under `caffeinate -is` (see "Keeping the control machine awake") so an idle-sleep doesn't sever the VPN mid-watch.

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

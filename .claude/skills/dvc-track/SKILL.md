---
name: dvc-track
description: Adds a binary artifact to DVC at s3://cosmo-gas-vision-storage/dvc-data, pushes it, and records the lineage row in the active branch's LEDGER §4. Trigger when a script just produced (or is about to produce) a binary output meeting the >10 MB rule or matching the tracked extensions (.npy, .pt, .dat, .html, .mp4, .h5, .hdf5) — model checkpoints, sightline tensors, ground-truth snapshots, visualization HTMLs, rendered videos. Do not trigger for source code, configs (.yaml/.toml), markdown, or small CSVs (<1 MB).
---

# DVC artifact tracking (CosmoGasVision)

Project DVC remote: `s3://cosmo-gas-vision-storage/dvc-data`. AWS credentials come from `.env` (loaded via `python-dotenv`). Always run from the repo root so `.dvc/config` is picked up.

## When to track

Track an artifact under DVC if **either** condition holds:

1. Size > 10 MB on disk, OR
2. Extension matches the canonical heavy-output set: `.npy`, `.pt`, `.dat`, `.html`, `.mp4`, `.h5`, `.hdf5`.

Do **not** DVC-track: source code, configs (`.yaml`, `.toml`), markdown, small CSVs (<1 MB), MLflow's own `mlruns/` (already gitignored).

## Procedure

```bash
# 1. Add — produces <path>.dvc pointer file (small, git-tracked)
uv run dvc add <relative/path/to/artifact>

# 2. Stage the pointer + .gitignore update produced by `dvc add`
git add <relative/path/to/artifact>.dvc .gitignore

# 3. Push the blob to the S3 remote
uv run dvc push <relative/path/to/artifact>.dvc

# 4. Verify
uv run dvc status     # should report "Cache and remote are in sync"
```

If `dvc add` rejects the path because it's already gitignored, that's expected for paths under `experiments/<name>/artifacts/` — DVC will still track it. If git refuses to add the `.dvc` pointer because of the same ignore rule, force-add: `git add -f <path>.dvc`.

## Update the LEDGER

Append a row to **Section 4 (The Data — Lineage & Governance)** in `experiments/<branch_basename>/LEDGER.md`:

```markdown
| **<Implementation Area>** | `<filename>` | `<short metadata: shape, redshift, MLflow run_id, DVC hash>` |
```

Use the `ledger-update` skill for the write so the table format stays consistent.

## Recovery / removal

- **Restore** an artifact on a fresh clone: `uv run dvc pull <path>.dvc`.
- **Remove** an artifact from DVC tracking: `uv run dvc remove <path>.dvc`. The blob stays in the S3 remote until `dvc gc -c` is run.
- **Orphaned blob** (e.g., when the `.dvc` pointer was deleted via `git rm` instead of `dvc remove`): blob remains addressable by hash on the remote; recoverable until next `dvc gc -c`.

## Anti-patterns

- Committing a >10 MB binary directly to git → bloats history, breaks remote clones over slow links.
- Adding the artifact to `.gitignore` without DVC-tracking it → silently breaks reproducibility on collaborator machines.
- DVC-tracking a small text file (config, log) → wasted indirection; just commit it to git.
- Forgetting `dvc push` after `dvc add` → pointer is in git but blob lives only on your machine; collaborators get cache-miss errors.

---
description: Scaffold a new experiment track (LEDGER, pipeline.py, artifacts/) and create the exp/<name> branch
argument-hint: <name>
---

Scaffold a new experiment track named `$ARGUMENTS`. Steps:

1. **Confirm branch creation** with the user before switching: propose `git checkout -b exp/$ARGUMENTS` from the current branch. Ask which branch to base it on if not obvious.
2. **Create scaffolding** under `experiments/$ARGUMENTS/`:
   - `LEDGER.md` — copy the section structure from `experiments/nerf/LEDGER.md` (Architecture Diagram, Pulse, Methodology, Logic, Data, Evaluation Plan, Visualization, History) with placeholder content for the new track.
   - `pipeline.py` — minimal MLflow-wired skeleton:
     - Hierarchical experiment name `CosmoGasVision/$ARGUMENTS`.
     - Mandatory tags: `model_type`, `stage`, `physics_id`, `redshift`.
     - Standardized run name `Stage1-Bootstrap`.
     - Falls back to `nullcontext` if MLflow is unreachable.
   - `artifacts/.gitkeep` (the directory itself is gitignored; this just keeps it in tree).
3. **Commit on the new branch** with message `chore($ARGUMENTS): scaffold experiment track`.

Don't run the pipeline. Just scaffold and commit.

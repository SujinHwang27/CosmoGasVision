---
description: Update the active branch's LEDGER.md with this session's progress
---

Update `experiments/<branch_basename>/LEDGER.md` for the active `exp/*` branch.

1. Determine the active branch via `git branch --show-current`. If it isn't `exp/*`, ask which LEDGER to update.
2. Gather context:
   - `git status`, `git diff`, `git log -5 --oneline` — what changed.
   - The latest MLflow run state, if any was triggered this session.
3. Propose updates to the LEDGER's four core sections:
   - **Pulse**: stage status changes (✅ / 🚀 / ⏳).
   - **Logic**: any new D-XX decision with rationale (what + why).
   - **Data**: new snapshots, redshifts, MLflow run IDs, DVC hashes.
   - **History**: dated session snapshot with completions, immediate next steps, blockers.
4. Show the proposed diff to the user before writing. Don't commit unless asked.

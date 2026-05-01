---
name: ledger-update
description: Writes a section update to the active branch's experiments/<name>/LEDGER.md following the canonical 7-section schema (Pulse / Methodology / Logic / Data / Evaluation / Visualization / History) with monotonic D-XX decision numbering. Trigger when (a) an agent finishes a unit of work that changes stage status, decisions, data lineage, or run artifacts; (b) the mlflow-run skill needs to record a fresh run_id in §6; or (c) the dvc-track skill needs to append a row to §4. The /update-ledger slash command is the user-facing equivalent for end-of-session reviews — prefer this skill for in-task writes triggered by other agents.
---

# LEDGER write contract (CosmoGasVision)

The LEDGER at `experiments/<branch_basename>/LEDGER.md` is the single source of truth for an experiment track. Every other agent reads it. Inconsistent writes degrade the whole track. Use this contract.

## Locating the right LEDGER

```bash
git branch --show-current        # → exp/<name>
# LEDGER lives at: experiments/<name>/LEDGER.md
```

If the active branch is not `exp/*`, ask the user which track to update. Don't guess.

## Section schema (canonical)

```
1. The Pulse (Progress & Roadmap)        — stage status table + completed milestones
2. Methodology & Architecture            — what & how (mostly stable; rarely edited)
3. The Logic (Decision Log)              — D-XX entries
4. The Data (Lineage & Governance)       — file table + responsibility matrix
5. Evaluation Plan                       — metrics & datasets
6. Visualization & Artifacts             — MLflow run IDs, primary traces, insights
7. Session History & Next Handoff        — dated snapshots, next steps, blockers
```

Match section names exactly when writing — downstream queries depend on the headings.

## Format conventions per section

### Pulse (§1)

Status table uses three states only:
- ✅ **DONE** — stage is closed, target metric met
- 🚀 **NEXT** — actively in progress
- ⏳ **PENDING** — not yet started

Append completed milestones as `- **YYYY-MM-DD**: <one-line achievement>.`

### Logic (§3)

Decisions are numbered monotonically across the LEDGER lifetime — find the highest existing `[D-NN]` and increment. Format:

```markdown
- **[D-XX] <Short Title>**: <rationale — what was chosen and why, 1-2 sentences>.
```

Never reuse a D-XX number; never reorder existing entries.

### Data (§4)

Append rows to the lineage table:

```markdown
| **<Implementation Area>** | `<filename>` | `<short metadata>` |
```

Metadata should include shape/size/redshift/run_id/DVC hash as relevant.

### Visualization (§6)

```markdown
- **Run ID**: `<mlflow_run_id>` (MLflow)
- **Primary Trace**: `experiments/<name>/artifacts/<file>`
- **Insights**: <1-2 sentences, scientific takeaway>.
```

### History (§7)

Append a session snapshot:

```markdown
### **Session Snapshot: <Month DD, YYYY> (<Phase>)**

- <bullet of completion>
- <bullet of completion>

### **Immediate Next Steps**
- <bullet>
- <bullet>

### **Blockers**
- <bullet, or "None">
```

## Write protocol

1. Read the current LEDGER first — never blind-append.
2. Identify the highest existing `[D-NN]` if writing to §3.
3. Show the proposed diff to the user before writing.
4. Write only the affected section(s); don't reflow the rest.
5. Don't commit unless explicitly asked — the LEDGER is often staged alongside code changes.

## Anti-patterns

- Writing free-form prose into the Pulse table → breaks the status filter.
- Reusing or reordering D-XX numbers → invalidates external references in commits and papers.
- Logging a new MLflow run only in §7 (History) and not §6 (Visualization) → run_id becomes hard to find later.
- Editing the Methodology section (§2) for an in-progress change → §2 captures the *committed* methodology; record the change in §3 (Logic) as a D-XX entry first, then update §2 when the change ships.
- Creating a separate handoff/report file → forbidden by CLAUDE.md; everything goes in the LEDGER.

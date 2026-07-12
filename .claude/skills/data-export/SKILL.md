---
name: data-export
description: The project's canonical outbound data-export contract — how CosmoGasVision data leaves the project for external consumers. Cross-cutting infrastructure, NOT part of any experiment track: it lives on the dedicated service/data-export branch and is not assigned experiment LEDGER D-XX numbers. Codifies ownership (data-engineer primary; support-researcher secondary only for new-statistic derivations), the canonical landing path under results/exports/<consumer>/, the src/export.py + scripts/export/ + MANIFEST.md pattern, the git-stamped provenance-sidecar requirement (src.utils.provenance), the honest-reporting verb-ceiling gate on any claim-bearing export ([D-37] + the [D-73] close-out), and the step-by-step recipe to service a new request. Trigger when the user asks to "export X for the website / for <external consumer>", "ship a CSV / figure / dataset out", "fulfill a <consumer> data request", or when wiring a new src/export.py function or scripts/export/ wrapper. Do NOT trigger for internal pipeline outputs (those are dvc-track), MLflow runs (mlflow-run), or in-repo experiment artifacts under experiments/<track>/artifacts/.
---

# Data-export contract (CosmoGasVision)

> **Transplant note:** ported from the CosmoGasPeruser `data-export` skill via the
> skill-transplant protocol (2026-06-25), localized to CosmoGasVision conventions
> (SherwoodLoader, `src/export.py`, `src.utils.provenance`, the [D-37]/[D-73]
> verb-ceiling). The upstream's `SignalClustering` science did NOT come across — it
> is CosmoGasPeruser-specific. The **`selements-website`** consumer IS shared (it
> curates sources across the author's projects, CosmoGasVision among them) and is
> registered below.

CosmoGasVision data leaves the project through ONE audited boundary:
`src/export.py`. Every export carries a no-bit-lost integrity guarantee —
canonical-loader ingestion (`SherwoodLoader`), shape/finiteness/bounds validation,
full-float64 precision (no silent rounding), and a git-stamped provenance sidecar.

## Branch & governance discipline
This export workflow is **cross-cutting infrastructure, not an experiment**.
- All export work is committed on the dedicated **`service/data-export`** branch
  — NOT on `main`, NOT on any `exp/<name>` experiment branch. (Servicing a
  request: `git checkout service/data-export`, do the work, commit, push;
  rebase onto a newer base only if an export needs a loader that postdates the
  branch point.)
- Exports are **not** assigned experiment LEDGER `D-XX` decision numbers and do
  not touch `experiments/<track>/LEDGER.md`. **This skill is the contract of
  record.** Decisions about the export workflow itself are recorded here.

## Ownership
- **Primary owner: `data-engineer`.** All export functions, landing paths, the
  MANIFEST, and this skill are data-engineer-owned.
- **Secondary: `support-researcher`** — ONLY when an export requires deriving a
  *new statistic* (a quantity not already produced by an existing pipeline
  stage / `src/analysis/` evaluator). support-researcher derives + validates the
  statistic; data-engineer still owns the export function, validation, landing,
  and MANIFEST row.

## Canonical landing path
```
results/exports/<consumer>/<request-slug>/
```
Define the root constant in code when the export module is first created:
`src.export.EXPORT_ROOT` (`results/exports`), with one subdir per consumer. Each
consumer gets its own subdir + its own MANIFEST.

## The pattern (always all three)
1. **Logic** — a named, deterministic, re-runnable function in
   `src/export.py` (docstring stating scientific purpose; type hints with
   explicitly-imported annotations; np.float64; a `_validate_*` bounds/finiteness
   guard; loads via the canonical `SherwoodLoader`
   (`src/data/loader.py`), never raw `np.load`).
2. **Thin wrapper** — `scripts/export/<request-slug>.py`: argparse (with an
   `--out-dir` defaulting to the canonical landing dir) + a single call into the
   `src/export.py` function. NO logic in the script.
3. **Registry row** — append to `results/exports/<consumer>/MANIFEST.md`:
   `request-slug | producing function | source-data path | git SHA at export |
   date | consumer-facing filename | caveats`.

## Provenance-sidecar requirement (mandatory)
Every export writes a sidecar JSON next to the artifact, stamped via
`src.utils.provenance.get_git_info()` (git SHA + branch + dirty + timestamp; or
`provenance_header(params)` to fold in export params), plus: source-data path,
producing-function name, export timestamp, semantic labels (physics_id, redshift,
sightline_idx, k_parallel / r units, etc.), source lineage, and an HONEST
selection/limitation caveat. The sidecar makes the exact source array and code
state recoverable from the shipped file alone. A non-`unknown` git SHA is required.

## Honest-reporting verb-ceiling gate (claim-bearing exports)
If an export carries or implies a *scientific claim* (not just raw spectra/fields),
the caveat string and any consumer-facing copy MUST respect the project's
honest-reporting rule (CLAUDE.md "[D-37] honest-reporting rule") and the
scientific verb-ceiling fixed by the **[D-73] close-out** — cited as the source of
what the project can and cannot claim, not as export-workflow decision coupling.
Sources: `experiments/nerf/LEDGER.md` §3 [D-37] + [D-73] amendment-9/amendment-10;
the close-out spine `experiments/nerf/design/D73_project_diagnosis_v2.md`.

- The headline finding is a **characterization of the optimization/identifiability
  wall at z = 0.3** — NEVER "we reconstruct the 3D IGM", "neural IGM tomography
  works", or "feedback/structure recovered". The z = 0.3 flux inverse problem is
  **under-constrained under this FGPA forward model**.
- The production MLP **fails two of three directly-evaluated [D-13] gates** (P_F,
  flux-PDF KS); the **3D ξ gate was never evaluated on the MLP** in its defined
  form — **"fails all three gates" is BARRED**.
- If exporting ξ: the **0.6 ξ-gate is DEMOTED** (unreachable as implemented —
  truth-vs-truth ≈ 0.03 at r = 2; report ceiling-relative) and is **frame-confounded**
  (real-space cube vs redshift-space flux). If exporting a P_F MLP-vs-grid contrast:
  carry the **n_rays config caveat** (verdicts comparable, magnitudes not a
  controlled same-config contrast).
- Always flag the **single-realization / fixed-cosmology** limitation (Sherwood
  Suite, one box) and the **60 cMpc/h box** provenance, and the **z = 0.3
  scope-lock** (no claim beyond z = 0.3; CLAMATO/TARDIS succeed at z ≈ 2–3 where
  the forest is information-rich).

Lead with the empirical observation; the honest framing wins on first pass
(the project honest-reporting rule). A purely descriptive export (e.g. "first
stored P1 sightline, not a typical spectrum") still carries its honest selection
caveat.

## Internal-identifier scrub gate (ALL external exports — data AND figures)
Every artifact leaving the project — CSV/JSON, provenance sidecar, **and any figure
ART (its rendered labels, titles, axes, captions)** — MUST be free of the project's
*internal* protocol identifiers before it ships. Barred on any consumer-facing surface:
- LEDGER decision-log tags — `[D-XX]` (`[D-24]`, `[D-73]`, …).
- R-rule codes (`R15`, `[D-37]-ext`) and sprint/run codenames (`pub-t1`, `Sprint-4`,
  `T3`, `Rung-4`).
- The literal word `LEDGER` / "Mirrors LEDGER §X".

These are internal — they mean nothing to an external reader and expose private
process. Caveats may cite a decision's **content** ("the mean-flux anchor", not "the
[D-11] anchor"); the D-XX provenance belongs in the sidecar's internal-lineage field,
never in consumer copy or on the figure. **For figures, verify the rendered image, not
just the filename**: grep the generator for `D-[0-9]`/`LEDGER`, and if it hits, ship a
publication variant (`support-researcher` owns the stripped-label render) — never the
internal LEDGER-mirror diagram. *(Added 2026-07-11 after the `method_pipeline` figure
shipped to selements-website carrying `[D-06]/[D-11]/[D-24]` labels.)*

## How to service a new <consumer> request
1. **Enumerate the source data this session** (data-engineer mandatory trigger):
   `ls -la` the source dir; confirm path, shape, dtype, finiteness, physical
   bounds (ρ/⟨ρ⟩ ≥ 0, X_HI ∈ [0,1], τ ≥ 0, etc.). Lead with the empirical
   observation, not a derived conclusion.
2. **Pick a request-slug** (kebab-case, e.g. `primer-synthetic-spectrum`).
3. **Add the logic function** to `src/export.py` — `SherwoodLoader`, `_validate_*`
   guard, float64, full precision, deterministic, type-hinted. (Create `src/export.py`
   + `EXPORT_ROOT` on the first request.)
4. **Add the thin wrapper** `scripts/export/<slug>.py` (argparse + one call).
5. **Run it** from repo root: `PYTHONPATH=. uv run python scripts/export/<slug>.py`.
6. **Verify** the artifact lands at the canonical path, row/shape counts match,
   bounds hold, and the sidecar JSON has a real (non-`unknown`) git SHA.
7. **Append the MANIFEST row** with the git SHA at export + the honest caveat.
8. **Add tests** to `tests/test_export.py` (synthetic fixtures for write/validate
   logic; mark any real-`Sherwood/` test `@pytest.mark.slow`).
9. If the export is claim-bearing, run it past the **honest-reporting
   verb-ceiling gate** above before shipping copy. For **every** export (claim-bearing
   or not, data or figure), also run the **internal-identifier scrub gate** — no
   `[D-XX]` / `LEDGER` / sprint-codename on any consumer-facing surface, figure ART
   included.

## Registered consumers
- **selements-website** — curates project sources across the author's projects;
  CosmoGasVision is one. First (and currently only) registered consumer.
  Landing: `results/exports/selements-website/`. Registry:
  `results/exports/selements-website/MANIFEST.md`. Because it is a public-facing
  curation site, **every claim-bearing export inherits the [D-73] close-out
  verb-ceiling** (see the verb-ceiling gate above) — lead with the honest
  characterization, never a reconstruction/“it works” claim. No requests serviced
  yet; the first export adds the first MANIFEST row.

## Anti-patterns
- Logic in `scripts/export/` instead of `src/export.py`.
- Raw `np.load()` in the export path (bypasses the canonical loader + validation).
- Shipping without a provenance sidecar, or with `"commit": "unknown"`.
- Rounding floats on write (information loss — use full float64 precision).
- Comment lines in a CSV destined for a comment-unaware downstream parser.
- A claim-bearing caveat that exceeds the honest-reporting verb-ceiling
  (e.g. "reconstructs the 3D IGM", "fails all three gates").
- Shipping an artifact **or figure** that renders internal identifiers
  (`[D-XX]`, `LEDGER`, "Mirrors LEDGER §X", sprint/run codenames) to an external
  consumer — verify the figure ART, not just its filename (internal-identifier scrub gate).
- Forgetting the MANIFEST row (orphan export, no lineage).
- Committing export work on `exp/<name>` or `main` instead of `service/data-export`.

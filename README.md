# CosmoGasVision

**Characterizing an under-constrained inverse problem in continuous-neural-field
Lyman-α IGM tomography at z = 0.3.**

A completed, documented research project: can a NeRF-style neural field, trained
through a differentiable physics renderer, recover the 3D structure of
intergalactic gas from Lyman-α absorption spectra? At redshift 0.3, under the
forward model used here, the answer is no — and the project's contribution is
*showing why*, with the evidence to back it.

## The finding

A neural field (MLP + Fourier features) trained against simulated absorption
spectra reproduces the mean flux and the flux distribution, but misses the flux
power spectrum by roughly four times the tolerance, with the error concentrated
in saturated absorption regions. A systematic intervention campaign — loss
design, per-pixel physics regularizers, conditioning inputs, data pooling,
direct spectral supervision, alternative network bodies and output heads —
failed to close the gap; most interventions collapsed to constant-field
solutions that still render acceptable spectra.

The decisive experiment replaced the network entirely with an explicit 192³
voxel grid — every voxel an independent trainable parameter, no inductive bias —
under the byte-identical training objective. The grid escaped the collapse and
passed the flux gates the MLP failed. It also fit the observed spectra **about
four times better than the true gas field itself does** through the same
renderer, while recovering only weak 3D structure. A wrong field out-fits the
right one: the optimization target does not identify the true field.

**Conclusion (scoped):** at z = 0.3, under this forward model, the flux inverse
problem is under-constrained — the failure is in the information content of the
problem as posed, not in the model or its training. No claim is made beyond this
redshift or forward model; at z ≈ 2–3, where the forest is information-rich,
tomographic reconstruction demonstrably works (CLAMATO, TARDIS).

Caveats carried with the headline: the 4× margin includes slack from the
approximate forward model (the truth's residual is our integrator's error, not
nature's); results are from a single simulation realization (Sherwood, 60 cMpc/h
box, fixed cosmology).

## The approach

- **Model:** multi-head MLP (8×256, Fourier positional encoding) mapping 3D
  position → density, temperature, neutral-hydrogen fraction, peculiar velocity.
- **Forward model:** differentiable Voigt-profile absorption integrator with
  redshift-space distortions (Tepper-García 2006 kernel); fully
  autograd-compatible.
- **Supervision:** per-bin log-optical-depth MSE with saturation masking, plus a
  mean-flux anchor (Kirkman+ 2007) that breaks the density–amplitude degeneracy.
- **Data:** Sherwood Simulation Suite (Bolton+ 2017) — four feedback variants
  (no feedback / stellar winds / AGN / strong AGN), 16,384 stored sightlines per
  variant at z = 0.3, aligned with the three box axes.
- **Evaluation:** pre-registered gates on the flux power spectrum, flux PDF, and
  3D correlation, at fixed sightline density (1,024 rays).

A separate track probes how much feedback-physics information survives in small
density crops (3D CNN vs. moment baselines) — see the decision log.

## Repository map

| Path | What it is |
|---|---|
| `src/` | Library code: data loaders and validation, the neural field and voxel grid models, the differentiable integrator, analysis/evaluators |
| `experiments/nerf/` | The research track: `LEDGER.md` (the full decision log — see below), `pipeline.py` (training entry point), `design/` (experiment design docs), `artifacts/` (evaluation outputs) |
| `papers/shared/` | Paper source of truth: section atoms, numerical macros, figures |
| `papers/cvpr2026/` | A compiled 8-page paper draft |
| `scripts/` | Run drivers, diagnostics, figure generators, HPC job scripts |
| `results/exports/` | Audited data exports to external consumers, with provenance sidecars |
| `cloud_runs/` | Training run bundles (large artifacts tracked via DVC) |

## Documentation conventions

The project is documented decision-first. `experiments/nerf/LEDGER.md` records
every methodological decision as a numbered entry (`[D-NN]`) with its rationale,
the empirical result that prompted it, and links to the artifacts that back it —
including the interventions that failed and why. Appendix A of the ledger gives
a plain-language walkthrough of the experiment campaign. The paper draft is
generated from `papers/shared/`, where every quantitative claim resolves to a
macro in `numbers.tex` traceable back to a ledger entry.

Reporting follows an observation-first rule: findings are recorded as observed,
with the honest framing preferred over the strengthening one, and null results
treated as outcomes rather than problems.

## Reproducibility

- **Environment:** Python ≥ 3.9, managed with [uv](https://docs.astral.sh/uv/)
  — `uv sync`, then run from the repo root with `PYTHONPATH=.`.
- **Tracking:** MLflow (hierarchical experiment names, stage-prefixed runs).
- **Artifacts:** DVC (S3 remote) for checkpoints and large outputs; provenance
  sidecars (git SHA, branch, timestamp) on exported data.
- **Simulation data is not included** — Sherwood snapshots and sightlines are
  pulled from the upstream source; loaders validate physical bounds on ingest.
- **Tests:** `uv run pytest tests/`.

## Status and open directions

Research is complete; the paper write-up is in progress. Extensions documented
in the close-out, in rough order of scientific interest:

1. **Port the pipeline to z ≈ 2–3**, where the forest carries far more
   information and the same machinery can be tested where reconstruction is
   known to be possible.
2. **A corrected 3D correlation estimator** (the gate used here was found to be
   unreachable as implemented and frame-confounded; both defects are documented).
3. **Modern field representations** — SIREN, hash grids, triplanes — as the
   untested representation axes.
4. **A sightline-density ablation** (designed, never run): does the wall move
   with more rays?

## Author

Sujin Hwang — independent research project, 2025–2026 (UT Dallas, 2026).
Questions, collaboration, or mentorship inquiries welcome via GitHub issues.

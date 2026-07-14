# the-direct-attack — data exports (episode 04)

Cover note for the selements-website curator. This bundle is **claim-bearing and
public-facing**; every sentence is written to sit inside the project's
honest-reporting verb-ceiling. All four artifacts are **BANKED** — assembled from
the decision record and the on-disk per-bin diagnostics; nothing recomputed from
simulation data. Internal decision tags live only in the sidecars'
`internal_lineage` fields, per the scrub gate: **none of them may appear in
episode copy or on figure art.**

Producer: `data-engineer` on the `service/data-export` branch. Not committed at
authoring time — a PI verb-ceiling gate reviews this note before the landing
commit is minted.

## Artifacts

| file | what it is | status |
|---|---|---|
| `fig1-verdict-table.csv` | training-vs-eval contrast: the added term's training endpoints + the eval verdict vs baseline | BANKED |
| `fig2-pf-per-bin.csv` | per-bin P_F(k∥): intervention vs truth, P1, with in-gate-band flag | BANKED |
| `fig2-shape-amplitude-summary.csv` | the failure signature in one table: shape vs amplitude, intervention vs production baseline (P1–P4) | BANKED |
| `spec-run-config.csv` | the three added loss terms, run config, pre-committed stop discipline | BANKED |

Overlay note: the production-baseline P_F curve is **already couriered** as the
episode-03 `the-neural-field/fig1-pf-miss.csv` — its 17-bin k-axis is
float-identical to `fig2-pf-per-bin.csv` here, so the two files overlay
directly. Reference it; do not request a re-ship.

**One beat has no figure and must be prose-carried:** the per-step training
trajectory of the added loss term did not survive the compute-site round-trip
(endpoints of record only: 0.99 → 6.77e-6 over 12,500 steps). Cite the descent
as endpoints; do not draw it as a curve — no per-step data exists to back one.

## Episode tightenings (BINDING — write the episode to quote these)

1. **The failure signature (licensed one-line form).** *"The intervention failed
   by amplitude-shrink with shape preservation: the run kept the shape of the
   flux power spectrum — log-space correlation 0.83 with the truth, about what
   the baseline already had — but rendered it at roughly 0.43× the true
   amplitude, settling into a scale-distorted optimum of the relative-residual
   term."* NOT a constant-prediction
   collapse (that was the first hypothesis, and the per-bin check refuted it —
   an honest correction beat the episode may narrate), and NOT the
   flux→1.0 transparency collapse: that tell belongs to the later interventions,
   not this one. The contrast that makes the signature vivid: the production
   baseline holds amplitude ratio ~0.74–0.98 with high bin-to-bin scatter
   (0.31–0.46); the intervention holds ~0.43 with LOW scatter (0.12) — it
   traded amplitude for the very term it was told to minimize.

2. **Training loss vs eval (the licensed sentence).** *"The added term did
   exactly what it was asked: its training value fell five orders of magnitude.
   Everything the term was supposed to fix got worse: the flux-power residual
   rose 37% above the baseline it was meant to beat, and the flux-distribution
   gate — which the baseline passed — broke."* The five-orders descent is NOT a
   win and must never be framed as partial progress. One caution: the record
   reads part of the tiny final training value as the term overfitting on the
   small 64-sightline training pool, so the licensed general form is "the term
   can be driven arbitrarily low in training while the statistic it names gets
   worse in evaluation" — not "the network beat the statistic by five orders."

3. **Scope of the lesson.** Demonstrated: *this integrated-statistic loss
   family is gameable by scale distortion* — on the fiducial physics variant,
   single seed, 12,500 steps, at minimal-intervention weights. The episode-era
   redirect ("a per-pixel physics constraint is structurally immune to this
   degeneracy") is the hook to the next episode and must be narrated as **the
   belief the next experiment killed** — never as a standing conclusion. And
   "retuning the weights would not have fixed it" is **argued from the loss
   shape, not tested** — no retry was run; do not imply one was.

4. **Small test, pre-committed stop — the discipline is the story.** One
   physics variant, a quarter of the production schedule, a FAIL criterion
   fixed before dispatch; on falsification the all-four-physics full-schedule
   follow-up was cancelled (~$4–6 saved). Not "a big experiment failed" — a
   small pre-registered test retired the intervention family. Schedule caveat:
   the P_F/KS reference values come from the 50k-step production baseline while
   the intervention ran 12,500 steps — the FAIL verdict is the pre-committed
   criterion and is verdict-level; magnitudes carry the schedule difference
   (only `loss_data` is compared at matched step count).

5. **The group caption (first of four).** This is the first of four
   interventions on a falsification queue, each retired with its **own distinct
   failure signature** (this one: amplitude-shrink with shape preservation).
   The licensed arc-level form is *"four specific interventions produced four
   distinct degeneracy signatures"* — never "the cascade proves nothing can
   work," and no completeness verbs ("the axes are exhausted/foreclosed" is
   barred).

6. **Carry-over from episode 03.** Any restated episode-03 result keeps its
   own tightenings (seed-qualified mean flux; "scored directly on two of three
   checks"; train-at-64/eval-at-1024). The eval geometry here is the same
   n_rays=1024 / seed=42 convention.

## Honest takeaway (verb-ceiling-compliant)

The first intervention against the saturation-band deficit put the missed
statistic itself into the objective. The term descended five orders of
magnitude in training while the evaluated statistic worsened by 37% and a
previously passing gate broke: the network satisfied the letter of the new
lesson — partly by scale distortion (shape preserved, amplitude suppressed),
partly by overfitting the term on its small training ray pool — rather
than by learning the structure the statistic was meant to enforce. A
pre-committed stop rule retired the intervention family at one physics variant,
a quarter of the production schedule, and minimal-intervention weights. This is
a characterization of how an
integrated-statistic loss can be gamed under this forward model at z = 0.3, on
a single Sherwood realization — not a claim that loss-based fixes are
exhausted, and not yet an answer to whether the model is too weak or the
problem too hard.

# removing-the-clamp — data exports (episode 09)

Cover note for the selements-website curator. This bundle is **claim-bearing and
public-facing**; every sentence is written to sit inside the project's
honest-reporting verb-ceiling. Internal decision tags and mode labels live only
in the sidecars' provenance-lineage fields. Light by design, as requested — one
experiment, one pre-registered verdict.

Producer: `data-engineer` on the `service/data-export` branch. Not committed at
authoring time — a PI verb-ceiling gate reviews this note before the landing
commit is minted.

## Two notes on the request's premises (one good surprise, one hazard)

1. **A real trajectory survived — better than endpoints.** Each cell logged its
   variance ratio AND correlation at six recorded steps (0–1,000), and the
   summary banks median-across-seeds trajectories per learning-rate cell.
   `fig2` ships them; draw the trace — the endpoints-only rule does not apply
   to this episode.
2. **HAZARD — the per-cell files carry a trap.** The probe reused an older
   harness, and every cell JSON contains a legacy `verdict` field from that
   harness's old verdict matrix — reading "PASS" on one cell and other legacy
   labels ("FAIL_SINKING", "UNKNOWN_NON_MONOTONIC") on the rest — which the
   pre-registration explicitly **voided** for this gate. The probe verdict is computed solely
   from the trajectories. The shipped CSVs exclude that field; if you ever
   read the raw artifacts, do not surface it.

## Artifacts

| file | what it is | status |
|---|---|---|
| `fig1-per-cell-verdicts.csv` | all 15 cells (9 unclamped + 6 clamped controls): final ratio + correlation vs the bar | RE-READ |
| `fig2-median-trajectories.csv` | median trajectories per learning-rate cell, six steps, both gate quantities | RE-READ |
| `fig3-gate-spec.csv` | the pre-registered gate as quotable rows (disjunct, guard, null anchor, voided legacy verdict) | BANKED |
| `spec-probe-config.csv` | what removing the clamp means; config; cost; the scope-lock | BANKED |

## Episode tightenings (BINDING — write the episode to quote these)

1. **The acquittal, licensed form.** *"The clamp was removed and nothing
   changed: all nine unclamped cells collapsed to the same floor as the
   clamped controls — ten-thousand to a hundred-thousand times below the
   escape bar, correlations never leaving the noise floor, the two
   populations statistically indistinguishable. The head is acquitted."* How
   far "acquitted" may be leaned on: it acquits the clamp **as the cause of
   this collapse, at this probe's scope** — never "the head plays no role
   anywhere," and never as evidence about any other regime.

2. **The scope-lock (the hardest-working line of the episode).** *"This probe
   supervises the density directly, at a reduced grid, for a thousand steps.
   It does not touch the flux-supervised regime — the production lesson —
   and no sentence about that regime may cite it."* The root-cause sentence
   the acquittal unlocks, at its exact scope: *"the collapse is upstream of
   the output clamp, in the optimization landscape itself — under direct
   density supervision, at this grid, for this step count."*

3. **The pre-registration framing (both outcomes informative).** *"The gate
   was written before the run, and both of its outcomes decided something:
   an escape would convict the clamp; a collapse acquits it and points
   upstream. The probe could not produce a result that needed spin."* The
   spurious-fire guard is the checks-that-learn motif continued, and that
   continuity is licensed: *"the gate named, in advance, two specific ways a
   naive threshold could fool itself — an unclamped head at initialization
   can clear the variance bar with zero structure, and a few extreme voxels
   can dominate the variance — so an escape required variance AND structure
   at the same recorded step, with structure judged against what a random
   field could do."*

4. **The two-clusters figure reading.** The intended figure (two
   indistinguishable clusters against the escape bar) is licensed exactly as
   intended: the interesting feature IS the absence of a difference. Caption
   register: *"the clamp's removal moved nothing"* — never "proved the head
   irrelevant."

5. **The ep10 hook (hedged — no outcome spoiler).** *"One suspect acquitted;
   the deeper one still standing: the representation itself — the network,
   as a way of holding a field. But there is a trap in testing it: a free
   grid, with a parameter for every voxel, would ace this direct test
   trivially — it can simply memorize the answer. The only informative test
   of the representation is under the flux lesson, the one the whole arc has
   been failing. That is the arc's final experiment."* BARRED: any hint of
   the grid run's outcome.

6. **Cost/scale framing (as-spent form — this one differs from prior
   episodes).** *"About four minutes of compute on the host machine, nothing
   paid — and nothing skipped: this probe was host-scale by design."* There
   was no cancelled paid stage here; do not import the "saved ~$X" pattern.
   (14 of 15 cells ran ~5–6 seconds; one first-run outlier took ~2.5
   minutes.)

## Honest takeaway (verb-ceiling-compliant)

The suspect list from the wall had two names on it, and this episode crossed
off the first. Remove the clamp — replace the bounded density head with a bare
linear readout of the log density, nothing for the optimizer to hide behind —
and the collapse does not care: nine unclamped cells, three learning rates,
three seeds, all landing ten-thousand to a hundred-thousand times below the
pre-registered escape bar, indistinguishable from the six clamped controls
run beside them. The gate was written before the run, both outcomes were
informative by construction, and the verdict is the quiet kind: an acquittal.
The collapse is upstream of the head, in the optimization landscape itself —
under direct density supervision, at this reduced grid, for this step count,
on one simulated universe at z = 0.3. What remains is the second name on the
list: the representation. And because a free grid would trivially memorize
this direct test, the only honest way to ask that question is under the flux
lesson itself — which is where the arc goes last.

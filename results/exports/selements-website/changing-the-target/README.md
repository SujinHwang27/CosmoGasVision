# changing-the-target — data exports (episode 08)

Cover note for the selements-website curator. This bundle is **claim-bearing and
public-facing**; every sentence is written to sit inside the project's
honest-reporting verb-ceiling. Internal decision tags, job numbers, and method
names live only in the sidecars' provenance-lineage fields. You asked for too
many licensed sentences rather than too few — this note leans that way.

Producer: `data-engineer` on the `service/data-export` branch. Not committed at
authoring time — a PI verb-ceiling gate reviews this note before the landing
commit is minted.

## Four corrections to the request's premises (all binding)

1. **The per-seed deltas are NOT the skip-rich headline.** The record *amended
   itself* here after an adversarial review: "all 10 deltas negative" was ruled
   init-confounded (the starting values span ~13× from initialization alone,
   and the deltas track the starting values at r ≈ −0.99). The **primary
   observable of record is the ~3-decade variance deficit**: predicted-to-true
   variance ≈ 6e-4 at every seed at the end of training. The deltas are
   demoted to a direction-of-motion indicator and always carry the
   init-confound caveat.
2. **The p = 1.0 is not a test of regression.** The test was pre-registered to
   detect *improvement*; p = 1.0 means improvement is maximally unsupported.
   No reversed test was run (deliberately). Licensed order: descriptive
   observation first (all ten seeds moved down), then "the improvement test
   returned p = 1.0," never "statistics confirmed the regression."
3. **The healthy reference comes from the production-run control artifact**,
   not the frozen-init calibration file the resource lists might suggest — and
   it never appears without its two-part caveat (below).
4. **No campaign-total dollar figure is banked for this group — do not quote
   one.** The licensed cost framing is qualitative: a sequence of small capped
   runs, each stopped by a pre-committed gate, most retired minutes into
   budgets of hours. (Also: the sharpest-datum numbers of record are
   **20,807 → 1.008** — the step-200 same-run pair from the authoritative
   seven-attempt table; the cut's "~22,000 → 0.98" is a different, banked
   step-100 pair from the first, instrumentation-blocked dispatch, welded
   there to a variance outcome that belongs to the re-dispatch — flagged for
   amendment.)

## Artifacts

| file | what it is | status |
|---|---|---|
| `fig1-direct-target-arc.csv` | the opening movement: instrument falsified → wiring saga → the basin's first sighting | BANKED |
| `fig2-collapsed-basin-cluster.csv` | the decisive figure: seven levers, one 0.89-decade band | BANKED (healthy ref + two-part caveat in sidecar) |
| `fig3-closing-probes.csv` | the three closing probes, one row each, amended readings baked in | RE-READ (verdict artifacts) |
| `spec-campaign-config.csv` | what changing-the-target meant; discipline; honest cost framing | BANKED |

## Episode tightenings (BINDING — write the episode to quote these)

1. **The collapsed-basin inference of record.** *"Balancing the two tasks was
   necessary but not sufficient: the one lever that fixed the gradient
   imbalance completely — four decades of correction, to within one percent of
   perfect balance — left the collapse exactly where it was. The pathology
   lives upstream of how the loss is built, in the structure of the
   supervision target itself."* Licensed lay naming of the basin: *"the
   network finds a resting place where it predicts something close to the
   average everywhere — a flat answer whose statistics barely vary — and
   every lever the campaign pulled left it in that same resting place."*

2. **The two-part caveat, one quotable line for the figure margin.** *"Healthy
   production runs hold this ratio near 1.0 from step five thousand onward;
   the retired runs were read at step 200, and no healthy run has a banked
   step-200 reading — so this is a collapsed floor against a healthy plateau,
   not a matched-step comparison."* Mandatory beside every rendering of the
   cluster figure.

3. **The "falsified" bar, per beat.** First dispatch: *"what failed was the
   balancing instrument — a simplified variant of a published method — not
   the direct-target idea; the hypothesis narrowed."* Clean pilot: *"real
   evidence at step 200: the first sighting of the basin."* Arc level, after
   the cluster: what the campaign established is the basin's robustness to
   every lever pulled and the upstream inference — NOT that the
   direct-target idea is impossible, and NOT that the loss-construction class
   was exhausted: *one named alternative was de-prioritized on this evidence,
   never tested.* That distinction is binding.

4. **The wiring-saga honesty beat (narratable, compact form).** *"The first
   run of the new lesson returned nothing at all — five thousand steps of
   silence — and the project treated the silence as a suspect, not a result:
   it broke its own code on purpose to prove the test could see anything,
   fixed the first bug it found — the new term had never been wired into the
   gradients — then watched the same tripwire catch a second, subtler break,
   fixed that too, and only then believed the clean run."* This is licensed;
   the beat is the discipline of distrusting your own null. (One deliberate
   break proving the tripwire; the tripwire then caught the second bug — not
   two deliberate breaks.)

5. **The 1-of-3 caveat, quotable.** *"One learning rate of the three showed
   the active shrinkage; the other two were non-monotonic and unreadable —
   this probe is cited with that count, always."* And its honesty beat is
   licensed and worth telling: *the project's own first summary said "all
   cells failing," and its adversarial review convicted that framing and
   corrected the record.*

6. **The skip-rich scope-lock, quotable.** *"What failed is exactly this:
   one skip-rich network body, under direct density pretraining, on the
   fiducial gas at this resolution, for five hundred steps, at a learning
   rate inherited from a probe of a different body. Ten seeds, one verdict.
   It does not say the architecture axis fails — one architecture is not an
   axis."*

7. **The ep09 hook (hedged — no outcome spoiler).** *"Two suspects were left
   standing. The output head — a mathematical clamp that keeps density
   positive and might also be flattening it — and the deeper possibility
   that a network of this kind cannot hold the field at all. The next test
   was the surgical one: remove the clamp entirely, supervise the raw
   quantity, and see whether the collapse cares."* BARRED: any hint of the
   outcome; the licensed register is the suspect list plus the surgical test.

8. **External citation ruling (pending PI confirm).** The balancing method is
   a published one (gradient normalization for multi-task learning, Chen et
   al. 2018) and the simplified-variant distinction is load-bearing for the
   honest framing of beat 1 — recommendation: cite it reader-facing, in the
   references section, tied to the "simplified variant of a published
   method" clause. The PI gate will confirm or overrule.

## Honest takeaway (verb-ceiling-compliant)

The fifth campaign changed the one thing the first four had held fixed: what
the network is scored against. It ended at a wall, and the wall's shape is the
finding. Seven runs, each pulling a different lever — learning rate twice,
gradient clipping, the reduction operator, a re-normalized target, batch size,
even the physics variant — all landed in the same collapsed basin, their
flux-power variance pinned hundreds of thousands to millions of times below
truth in a band less than one decade wide. The lever that perfectly balanced the two tasks did not move
the collapse, so the balance hypothesis is necessary but not sufficient, and
the pathology sits upstream, in the target's structure. Three closing probes
sharpened the wall without breaching it: a candidate physics-residual target
failed a feasibility check computed directly from the fields, before any
training; direct density pretraining
produced one actively-shrinking cell in three; a different network body under
that pretraining left a three-decade variance deficit at every seed. All of it
is scope-locked — this supervision family, this machine class, z = 0.3, one
Sherwood realization — and none of it yet answers the deep fork; what it
narrows is where the pathology sits: upstream of how the loss is built,
untouched by any lever the campaign pulled — with one named loss-construction
alternative de-prioritized on this evidence, never tested.

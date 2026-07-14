# the-planted-clue — data exports (episode 06)

Cover note for the selements-website curator. This bundle is **claim-bearing and
public-facing**; every sentence is written to sit inside the project's
honest-reporting verb-ceiling. Internal decision tags and run identifiers live
only in the sidecars' provenance-lineage fields — none may appear in episode
copy or on figure art.

Producer: `data-engineer` on the `service/data-export` branch. Not committed at
authoring time — a PI verb-ceiling gate reviews this note before the landing
commit is minted.

**Per your ep05 offer:** the gate and head-asymmetry CSVs ship a reader-facing
`label` column, and no verdict cell carries a snake_case machine key — the ep05
Table-1 finding should not recur if you render these `label` columns directly.

## Two notes on the request's premises

1. **The gate-table JSON gap is real — resolved by exporting the record, not
   the file.** The on-disk gates JSON is header-only (every gate reads
   `pass:false` / "loss_history not provided" — a tracker wiring gap, not the
   result). `fig1-gate-table.csv` is exported from the **authoritative six-gate
   table in the decision record**; the broken file's path is named only in the
   sidecar lineage.
2. **The smoke trace survived only in part.** The *scalar* trace is real and
   plottable (`fig3-smoke-trace.csv`: the mean-flux climb to 1.0000). But the
   *per-head spread* was never logged per step — the density-spread and
   X_HI-spread numbers are **step-50 endpoints only** (`fig2`). So there is a
   mean-flux curve to draw, but the head-collapse is a two-bar endpoint
   comparison, not a trajectory.

## Artifacts

| file | what it is | status |
|---|---|---|
| `fig1-gate-table.csv` | the six pre-committed smoke gates (label / measured / floor / verdict) — 5 pass, density-spread fails | BANKED (record; JSON header-only) |
| `fig2-head-asymmetry.csv` | the D3 signature: density head dead near zero vs X_HI head still structured | BANKED (step-50 endpoints) |
| `fig3-smoke-trace.csv` | the scalar 50-step trace: the mean-flux climb to 1.0000 | RE-READ (local tracker) |
| `spec-run-config.csv` | conditioning input, unchanged loss, gates + floor provenance, cost | BANKED |

## Episode tightenings (BINDING — write the episode to quote these)

1. **The failure signature (licensed one-line form of D3).** *"The intervention
   failed by asymmetric-head collapse: in a machine with separate heads for
   density and neutral fraction, the density head collapsed to a sliver near
   zero while the neutral-fraction head kept its structure — a partial collapse
   hiding in one head of the model."* Distinctness from the two prior failures:
   the first **shrank a structured answer** (amplitude wrong, shape kept); the
   second **abandoned structure everywhere** (all fields constant); this one is
   **partial** — one head died, the other kept working. And note the shape is a
   *third* kind: near-**zero** density, not the previous episode's
   constant-at-71.5.

2. **The tell's second appearance, with its twist (quotable).** *"The mean flux
   landed on 1.0000 again — but this time it slipped through. It sat inside a
   pre-registered check that only asked the mean flux to stay within a wide band
   of the anchor, and 1.0000 is inside that band, so the number that looks like
   passing was logged as a pass. The tell had learned to hide in its own gate."*
   The banked lesson, in the form to quote: *"a mean flux is only a safe check
   if it is tested away from BOTH the anchor and the value a collapsed,
   transparent universe returns — 1.000; a gate that watches only the anchor
   waves the collapse through."* (This is the lesson that hardens the gate set
   in later work.)

3. **The gate-discipline debut, with its honest edge.** This was the first spec
   written under the recalibrated discipline: hedged design verbs (no
   "structurally immune," no "should fix"), six pre-committed smoke gates, and
   an anti-degeneracy audit written *before* the run. The honest edge is the
   whole point of the episode and must be told: the audit anticipated collapse
   re-emerging in the previous episode's shape — a constant, non-zero density —
   and it even named a different partial risk it would accept (temperature
   going flat while the other fields kept structure); what it did not picture
   was the density head itself, the very head being conditioned, dying to
   near-zero while its neighbor kept structure. What caught this new shape was the density-spread
   **floor**, a backstop inherited from that previous failure: because a floor
   asks only "did the field keep *any* spread," it is blind to *which value* the collapse
   sits near — near zero or near a constant far from truth — and so it caught a
   collapse the audit's imagination had missed.
   Licensed framing: *the discipline caught the failure it was built to catch,
   by a gate general enough to catch a failure it had not pictured.*

4. **The belief under test, and why the clue was not enough (licensed scope).**
   The clue was real: a genuine, truth-anchored signal (the line-of-sight
   velocity gradient, taken from the simulation's own gas and frozen so the
   network could not reshape it) — exactly the "ground-truth-anchored" fix the
   previous episode's failure called for, and the network *could not* game it by
   shrinking its outputs the way it gamed the last regularizer. And it still
   found an exit. The licensed reading: *"a truth-anchored input cannot be
   gamed — the network cannot reshape the clue — but it can be ignored: nothing
   forces every head to use it. The clue was consumed somewhere in the machine —
   the neutral-fraction head kept its structure — while the head the clue was
   meant to rescue collapsed anyway."* Scope-lock: this conditioning design, on the
   fiducial variant, at smoke scale (P1, 50 steps). Do **not** generalize to
   "input conditioning cannot work" — the claim is that *this* input, wired to
   *this* head, did not prevent *this* head from collapsing when the data loss
   constrained it weakly.

5. **Scale and discipline framing.** A 50-step host smoke, minutes on an
   ordinary machine, no paid dispatch. Cost stated the right way around: the
   floor breach was unambiguous, so **~$1.50 of paid GPU was saved** by not
   dispatching the full stage, and **nothing paid was spent** — no confirmation
   run this time (unlike the previous episode, where a review-mandated
   confirmation was warranted). Never "a big experiment failed."

6. **The group caption (third of four).** Three of the four attempts are now
   down, each with a distinct signature: shrank the answer / abandoned it
   everywhere / collapsed one head. Licensed arc-level form: *three specific
   interventions, three distinct failure signatures* — never "the cascade proves
   nothing can work," and no completeness verbs. The deep fork is unmoved:
   nothing here says whether the network is too weak or the problem too hard.

7. **The hook to the next episode (hedged — no outcome spoiler).** *"Three of
   the four axes are down: the lesson as a summary statistic, the lesson as a
   per-pixel rule, and the machine's input. The one axis left untried is the
   data itself. The deficit looked the same in all four
   feedback variants of the gas — so read that sameness as a clue: pool all four
   into one training set, and every gradient step sees four times as many of the
   strong-absorption sightlines where the miss lives. That was the next thing to
   try."* BARRED: any promise the pooling resolves the deficit; the licensed
   register is *the next axis to try*, outcome unspoiled.

## Rulings still owed — status (these were RATIFIED; recording it here so it
travels with the drop you actually read)

All three were ratified by the PI on 2026-07-14 during the ep04 gate and are
recorded in **`sources/experiment-epic-cut/experiment-epic-cut.md` §4
Amendment 1** (plus the MANIFEST). If your copy of the epic-cut note has no §4,
re-pull that folder — the amendments landed after it was first couriered.

- **(a) Fork 4 — RATIFIED as proposed.** Lesson-building (the loss contract +
  the three-gate definitions) cedes from ep02 to ep03; rebuilt ep02 narrows to
  the machine + the density–amplitude degeneracy + the mean-flux anchor
  mechanism + the Two-Pass Surrogate + the pipeline figure.
- **(b) The ablation-narrowing decision — RATIFIED, with one modification.**
  Assigned to **ep03 as a supporting (not headline) beat**, with a binding
  provenance caveat: it ran at the *old, broken* mean-flux anchor on a reduced
  preview schedule, so it must **never** be narrated as an ablation *of* the
  corrected-anchor production run — only as the config-scoped finding that the
  extra guards (cap and mask) were not load-bearing on the one un-stressed
  configuration tested — the novelty claim narrowed to the log compression —
  with the guards retained as defense-in-depth, explicitly untested on
  stressed configurations.
- **(c) Old-numbering addendum — RATIFIED as proposed, sharpened.** The
  narrative-arc review's "ep02" = new **ep03**; its "ep03" and "ep04" (the grid
  victory and the wall) = new **ep10 jointly** (one experiment; the reversal is
  intra-episode). Old-ep03 guidance must never be applied to new-ep03.

## Honest takeaway (verb-ceiling-compliant)

The third intervention changed the machine's input, not its lesson: it handed
the network a real, truth-anchored clue — the velocity gradient, frozen beyond
the network's reach — precisely the fix the previous failure asked for. The
network consumed the clue in one head and let the other, the very head the clue
was fed to, collapse to near-zero density; the resulting transparent gas drove
the mean flux to 1.0000, which slipped through a wide-thresholded gate
as a pass. A pre-registered spread floor, inherited from the previous episode's
collapse, caught the failure the audit had not pictured, and a 50-step smoke
retired the intervention for no paid compute. This is a characterization of how
a truth-anchored input, on this design at z = 0.3, failed to prevent a partial
collapse on a single Sherwood realization — not a claim that input conditioning
is exhausted, and not yet an answer to whether the model is too weak or the
problem too hard.

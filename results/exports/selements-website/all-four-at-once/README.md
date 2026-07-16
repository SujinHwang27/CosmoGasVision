# all-four-at-once — data exports (episode 07)

Cover note for the selements-website curator. This bundle is **claim-bearing and
public-facing**; every sentence is written to sit inside the project's
honest-reporting verb-ceiling. Internal decision tags and run identifiers live
only in the sidecars' provenance-lineage fields — none may appear in episode
copy or on figure art.

Producer: `data-engineer` on the `service/data-export` branch. Not committed at
authoring time — a PI verb-ceiling gate reviews this note before the landing
commit is minted.

## Three corrections to the request's premises

1. **The smoke pooled 1,024 sightlines TOTAL, not 4×1024.** The *design* pooled
   1,024 per variant (4,096 total, interleaved per microbatch) — but that scale
   was never dispatched; the smoke that retired the intervention ran 1,024
   pooled across all four. Keep the two numbers apart (the spec CSV states
   both). The episode must not describe the smoke at design scale.
2. **The X_HI spreads are [3.5e-3, 1.0e-3, 6.8e-3, 2.9e-4]** — 4.9× to 113×
   above the 6e-5 floor — not "3e-3–7e-3" (two variants sit below that range).
   "Hundreds of times above" is licensed for no variant except P3.
3. **The gates JSON is healthy this episode** (unlike the last one): full loss
   history, per-gate readouts, and the complete six-pair embedding-distance
   matrix. Everything here is re-read from it and consistency-checked against
   the decision record. The per-step **mean-flux trace was not logged** —
   the tell is endpoints-only (four 1.000s at step 50); the one drawable
   trajectory is the flat loss curve (`fig3`).

## Artifacts

| file | what it is | status |
|---|---|---|
| `fig1-gate-table.csv` | the seven pre-committed gates (3 FAIL / 4 PASS), reader-facing labels | RE-READ (healthy JSON) |
| `fig2a-d4-signature.csv` | per-variant: dead density heads vs alive X_HI heads vs mean flux 1.000 | RE-READ |
| `fig2b-embedding-distances.csv` | all six pairwise distances between the four learned codes (4.6–7.0) | RE-READ |
| `fig3-loss-trace.csv` | the 50-step flat-loss history (ratio 1.0036 story) | RE-READ |
| `spec-run-config.csv` | pooling design, gate-hardening lineage, stop discipline, cost | BANKED |

## Episode tightenings (BINDING — write the episode to quote these)

1. **The failure signature (licensed one-line form of D4).** *"The fourth
   intervention failed by combined trivial collapse with an active embedding:
   the network learned four genuinely distinct codes for the four kinds of gas
   — and routed all four into the same trivial solution, transparent gas with
   a dead density head, in every variant at once."* The quartet, each distinct:
   the first **shrank** a structured answer; the second **abandoned structure
   everywhere**; the third **collapsed one head**; the fourth **collapsed the
   same way four times in parallel, with the labels still alive**.

2. **How far "read the labels, same exit" may be leaned on.** Licensed: the
   network did **not** ignore the labels — gate 7 shows four distinct codes
   (every pair 46×–70× above the distinctness floor), so the one failure mode
   the audit had named ("the model ignores the labels") is ruled out, and the
   honest phrasing is *"the labels were read; the exit was the same."* NOT
   licensed: any claim the codes were doing physics-relevant work — the
   record's own lesson is that label-distinctness alone cannot certify that;
   the field-level spread gates are the real check, and they are what fired.
   So: "read the labels" = "kept them distinct," never "understood the
   physics."

3. **The tell's third beat (licensed sentence).** *"The mean flux landed on
   1.000 for the third time — this time in four-part unison, once per kind of
   gas — and this time the gate caught it: after two collapses had taught the
   project that a perfect-looking mean flux is a fingerprint, the check had
   been rewritten to reject any mean flux within a hair of 1.000, and it
   fired exactly as designed."* This is the arc's payoff beat and it is real:
   the hardened gate's threshold (away from both the anchor band edge and
   1.000) is in the pre-committed spec, and gate 3 FAILED on the tell.

4. **The cascade-close group caption (binding form).** *"Four interventions,
   pre-committed and retired in sequence, produced four distinct failure
   signatures — a shrunk copy, a constant universe, a dead head, and a
   four-fold collapse wearing live labels. That is a characterization of four
   specific designs against one deficit, under one supervision scheme — not
   proof that nothing can work, and not a claim that the possibilities are
   exhausted."* Completeness verbs are BARRED (no "axes exhausted," "options
   foreclosed," "proved impossible"). The one load-bearing qualifier that must
   ride every cascade-close sentence: the four failures happened **within the
   same supervision scheme** — how the network is scored against the light
   never changed — which is precisely what the next episode picks up. The
   cumulative-cost line is licensed in this form: *"all four retirements
   together cost well under ten dollars of compute"* (the record's own
   figure; scope it to the four interventions, never the whole project).

5. **The governance beat (one licensed sentence, if the episode wants it).**
   *"Pooling the four kinds of gas revived an idea the project had once
   rejected for a different reason — a network that sees the physics label
   risks leaking the answer to a later classification test — so before any
   full-scale run, the plan was amended to hold out a region of the volume
   entirely, and an adversarial review was put in front of the dispatch; the
   smoke verdict made the question moot."* Nothing further is licensed (the
   details are internal governance); the episode may also drop this entirely.

6. **Scale and discipline framing.** 50 steps, 196.6 seconds on the host
   machine's GPU, ~$0 paid; the paid full-scale ladder was cancelled unrun.
   Never "a big experiment failed." The gate set itself is the story's
   through-line: gate 3 is two episodes of collapse written into procedure,
   and gate 7 is the audit's one named risk made into a check — the check
   passed, and the failure came from a surface the audit had not named. (That
   last clause is licensed and worth keeping: it is the same honest edge as
   the last episode, one ring further out.)

7. **The hook to the next episode (hedged — no outcome spoiler).** *"An
   adversarial review of the close pointed at the one thing all four attempts
   had held fixed. Each changed the lesson's terms, the machine, or the data —
   but every one of them was still scored against the same target, the flux,
   compared the same way. The review argued that target sits upstream of
   everything tried so far — and it had never been questioned. Changing what
   the network is asked to match — that was the next move."* BARRED: any promise the
   supervision-target change works, and any suggestion the four axes *implied*
   the target was the problem (the review *surfaced* it as untested, nothing
   more).

## Honest takeaway (verb-ceiling-compliant)

The fourth intervention changed the data instead of the lesson or the machine:
all four kinds of gas in one network, each with a learned label, so every
gradient step saw more of the strong-absorption sightlines where the deficit
lives. Fifty steps later the loss was flat, the mean flux sat on 1.000 in all
four variants at once, and every density head had collapsed — while the four
labels remained genuinely distinct. The hardened gate set caught all of it, on
the host machine, for nothing. That closes the four-counterfactual sequence
against the saturation-band deficit: four designs, four distinct collapses,
all
retired by small pre-committed tests — two at fifty-step smokes, one at a
quarter-schedule run, one ratified at full scale after its smoke — for well
under ten dollars of paid compute in total, all under one unchanged
supervision scheme at z = 0.3 on a single Sherwood realization. It is a
characterization of those four designs, not a proof about the problem — and
the question the sequence leaves is the one it never varied: what the network
is asked to match.

# the-physics-constraint — data exports (episode 05)

Cover note for the selements-website curator. This bundle is **claim-bearing and
public-facing**; every sentence is written to sit inside the project's
honest-reporting verb-ceiling. Internal decision tags live only in the
sidecars' `internal_lineage` fields — none may appear in episode copy or on
figure art.

Producer: `data-engineer` on the `service/data-export` branch. Not committed at
authoring time — a PI verb-ceiling gate reviews this note before the landing
commit is minted.

## Two corrections to the request's premises (both binding)

1. **This was not a smoke-only retirement.** The 50-step smoke *triggered* the
   FAIL and the full-scale stage was skipped under the standing stop rule —
   but the retrospective review then challenged whether a smoke-scale verdict
   could stand, and one full-scale confirmation run was authorized (25,000
   steps, ~56 minutes, ~$1.50). It **ratified** the verdict: the flux-power
   residual pinned at its ~1.0 ceiling (essentially no flux structure), and
   the collapse-signature numbers you asked for (density flat ≈ 71.5, X_HI
   flat ≈ 3.3e-5) were measured on **that run's checkpoint, not the smoke**.
   Narrate the episode with both halves: the cheap test that called it, and
   the review that refused to let a cheap verdict stand unchallenged.
2. **There was no pre-committed per-field flatness floor at this episode.**
   The formal pre-registered gate tables (including the density-spread floor)
   debut in the *next* episode — a discipline this failure helped create. Do
   not narrate a "gate table" for this run; narrate the standing smoke checks,
   the tell, and the two-stage verdict.

## Artifacts

| file | what it is | status |
|---|---|---|
| `fig1-smoke-trace.csv` | the full 50-step smoke trace: regularizer descent vs the mean-flux tell (this one IS plottable — the trajectory survived, unlike episode 04's) | RE-READ (local tracker) |
| `fig2-collapse-signature.csv` | per-field truth-vs-prediction distribution table from the confirmation run: every field a near-constant | BANKED |
| `fig3-verdict-table.csv` | the two-stage verdict: smoke trigger + full-scale ratification | BANKED |
| `spec-run-config.csv` | regularizer form, weights, config, two-stage stop discipline, the frozen-exponents confession | BANKED |

## Episode tightenings (BINDING — write the episode to quote these)

1. **The failure signature (licensed one-line form of the collapse).** *"The
   intervention failed by constant-prediction collapse: the network stopped
   rendering structure at all — density flat near 71.5 everywhere, neutral
   fraction flat near 3.3e-5 — because constants trivially satisfy the physics
   relation the new term enforces, and the mean-flux anchor then pulls the
   resulting constant absorption toward transparency."* Distinctness from the
   previous episode: that failure **shrank a structured answer**; this one
   **abandoned structure** — and the previous one never fired the transparency
   tell (its flux-distribution test came back populated — a real failing
   score, not an empty sample — while the amplitude shrank). This episode
   is the debut of the transparent-gas family.

2. **The mean-flux tell (debut — this motif recurs in the next two
   episodes).** *"The predicted mean flux landed on 1.0000 exactly — a number
   that looks like passing and is the fingerprint of collapse: perfectly
   transparent gas matches the average of almost-transparent gas to within the
   anchor's tolerance, so the score the anchor reports is quietly satisfied by
   a universe with nothing in it."* Here the tell was read correctly at sight
   and the FAIL was called on it; keep that plain.

3. **The corrected mechanism (the honest-correction beat — narrate both
   reads).** The smoke-time reading was *"the network drove absorption to
   zero, evading the anchor."* The confirmation run's field diagnostic proved
   that reading empirically wrong in the strongest way: predicted n_HI is
   ~28,000× **larger** than the truth median — nothing was driven to zero;
   everything was driven to a **constant** (density 68.3–74.7 around 71.5,
   against a truth that spans six decades). The licensed form of record is
   constant-prediction collapse; the zero-absorption reading may appear only
   as the corrected first guess. And the cross-episode beat is licensed and
   worth telling: **the constant-collapse guess that the previous episode's
   per-bin check refuted turned out to be exactly right — one experiment
   late.**

4. **The load-bearing architectural finding (licensed scope).** Of record:
   *"any regularizer evaluated only at the network's own predicted state is
   gameable when the data loss is weakly informative on the diffuse-bin
   majority"* — demonstrated for this regularizer family on the fiducial
   variant, at smoke scale and ratified at one full-scale config, single
   seed. The refined form the record added: such regularizers are gameable
   *by collapsing all input fields to constants, which trivially satisfy any
   algebraic relation among them.* The refutation line is licensed plainly:
   **the belief the previous episode ended on — a per-pixel constraint leaves
   no room for the trick — lasted exactly one experiment.**

5. **Scale and discipline framing.** The smoke cost minutes of host CPU and
   no paid dispatch; skipping the full stage saved ~$1.50 of paid GPU; the
   review-mandated confirmation spent ~$1.50 to ratify at scale. Never "a big
   experiment failed" — and also never "the smoke alone settled it." Two
   small numbers to keep exactly right: the descent is ~20× (6.37 → 0.31),
   NOT five orders of magnitude (that was the previous episode); and the
   confirmation's flux-power residual is stated as **pinned at its ~1.0
   ceiling — no flux structure at all** — never as a "N× worse" multiple.

6. **The KS zero (quotable).** *"The flux-distribution test returned 0.0 — a
   zero that means empty, not perfect: gas this transparent leaves no pixels
   inside the analysis window at all."*

7. **The frozen-exponents confession (optional beat, licensed).** The
   physics relation's exponents were frozen from the textbook scaling
   (Hui & Gnedin 1997); a later fit to the simulation's own truth found the
   temperature exponent materially off (−0.41 vs the frozen −0.7). The
   verdict is unaffected — the collapse is a separate mechanism — but the
   banked lesson is quotable: *measure the relation from your own truth
   before freezing a textbook value.*

8. **The hook to the next episode (hedged — this wording is load-bearing).**
   *"Condition the network on a quantity anchored in the simulation's own
   ground truth — one the network cannot reach by shrinking its outputs. That
   was the new design rule; the next experiment was its first test."* BARRED: "structurally
   immune," "cannot be gamed," or any assertion-form of immunity — the record
   explicitly recalibrated this exact language after two falsified
   "immune/highest-leverage" design claims, and the licensed register is
   *first test of the new discipline*, outcome unspoiled.

## Honest takeaway (verb-ceiling-compliant)

The second intervention replaced the summary statistic with a per-pixel
physics prior — and the network defeated it more completely than the first:
it collapsed every field to a constant, which satisfies the enforced relation
trivially, and let the mean-flux anchor pull the resulting uniform absorption
to perfect transparency. A 50-step smoke caught it in minutes on the tell of
an exactly-1.0000 mean flux; a review-mandated full-scale run confirmed the
verdict rather than let a cheap test stand alone. The finding this banked —
any regularizer scored only on the network's own predictions can be satisfied
by structureless output when the data itself constrains the diffuse majority
weakly — is a statement about this loss family under this forward model at
z = 0.3, on a single Sherwood realization; it is not a claim that physics
priors are exhausted, and not yet an answer to whether the model is too weak
or the problem too hard.

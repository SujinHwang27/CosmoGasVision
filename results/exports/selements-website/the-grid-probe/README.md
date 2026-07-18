# the-grid-probe — data exports (episode 10, THE FINALE)

Cover note for the selements-website curator. This bundle is **claim-bearing and
public-facing**; every sentence is written to sit inside the project's
honest-reporting verb-ceiling. Internal decision tags, job numbers, and method
codenames live only in the sidecars' provenance-lineage fields. You asked for
too many licensed sentences rather than too few; this is the arc's heaviest
ruleset and the note leans that way hard.

Producer: `data-engineer` on the `service/data-export` branch. Not committed at
authoring time — a PI verb-ceiling gate reviews this note before the landing
commit is minted.

## Six notes on the request's premises (two good surprises, four corrections)

1. **GOOD SURPRISE — the full trainability trajectory survived.** The run's
   preserved metric store holds all 50,000 per-step readings of the variance
   ratio. `fig2` ships a downsampled trace (plus the healthy production
   reference and the prior campaign's collapsed floor): draw the trace, don't
   cite endpoints. Two mandatory caveats ride it — the cadence caveat (floor
   read at step 200; healthy reference exists only from step 5,000) and the
   initialization caveat (the near-zero start is the pre-committed uniform
   initialization, NOT a visit to the prior campaign's basin — a basin
   reached under a different training objective —; do not narrate
   "it started in the basin and climbed out").
2. **GOOD SURPRISE — the grid's full 3D correlation profile is banked.** Not
   just the gate-scale point: twenty distance bins from 0.25 to 9.75 Mpc/h,
   measured in the run's own evaluation. `fig4` ships the whole curve with the
   ceiling, noise references, classical lower bound, and the demoted bar.
3. **CORRECTION — "production sightline geometry" is not quite right.** The
   grid trained on 1,024 sightlines — the density the study's gates are
   *defined* at — while the production network *trained* on 64 (its published
   numbers are evaluated at 1,024). Same simulated survey, 16× denser
   training sampling for the grid. This is precisely why the verdicts-only
   rule exists on every grid-vs-network surface.
4. **CORRECTION — "a real paid run — as-spent" has no banked dollar figure.**
   The banked cost shape is: one GPU job, about 7.2 recorded wall-clock
   hours on one datacenter GPU (the tracked run's own start-to-end span,
   in-job evaluation included), inside a pre-committed 30-GPU-hour budget
   (about a quarter used). Print the budget shape; do not print a price.
5. **CORRECTION — "the mirror of episode 03's decisive figure" cannot be a
   spectrum overlay.** No per-wavenumber flux power spectrum is banked for
   the grid run; the banked quantity is the band-mean residual and its
   verdict (0.0352 vs the 0.10 gate, PASS). The mirror is a verdict figure —
   the same gate, now passing — not a curve-vs-curve redraw of episode 03.
6. **CORRECTION — the amplitude sweep is banked at three points, flat.** The
   truth-through-the-integrator loss is banked at the sweep's best amplitude
   (which lands at the swept edge, 4.0), at amplitude 1, and at the grid's own
   learned amplitude — all ≈ 0.0101; the three banked points agree to ~5e-6,
   and the recorded sweep verdict is flat across [0.2, 4]. "Flat" is
   licensed; a dense sweep curve is not banked, so a drawn flat line should
   be styled as the three measured points plus the licensed flatness clause.

## Artifacts

| file | what it is | status |
|---|---|---|
| `fig1-flux-gates.csv` | the gate finally opens: grid PASS 0.0352 vs the production FAIL 0.4155 (verdicts only) | RE-READ |
| `fig2-trainability-trace.csv` | the grid's 50k-step variance-ratio trace + healthy reference + collapsed floor | RE-READ |
| `fig3-truth-vs-grid.csv` | the decisive datum: truth through the same instrument, ~4× worse than the grid | RE-READ |
| `fig4-xi-ceiling-relative.csv` | the structure it actually holds: full profile, ceiling-relative, demoted bar | RE-READ |
| `spec-probe-config.csv` | what the probe is; pre-commitments; scale; cost shape; scope | BANKED |

## Episode tightenings (BINDING — write the episode to quote these)

1. **The headline, scope-locked (the finale's one sentence).** *"At z = 0.3,
   the most expressive field this study allows — a value free at every one of
   192³ cells — closes both flux gates the study defines and fits the observed
   flux better than the true field does through the same instrument, yet
   recovers only weak 3D structure. The z = 0.3 flux inverse problem, under
   this forward model, is under-constrained: escaping the collapse was
   necessary but not sufficient."* Every element load-bearing; none optional.

2. **The triple-hedged verdict, verbatim.** Under this forward model; at this
   redshift (z = 0.3); on one simulated universe. The licensed claim boundary:
   *"minimizing this flux loss does not identify the true field"* — NEVER
   *"the flux cannot determine the field"* (a claim about nature the record
   refuses), and the split between problem-intrinsic and forward-model-induced
   under-determination is explicitly **not separated** at 98% mean
   transmission. No "impossible" anywhere in the episode.

3. **The decisive-datum quotables (all three ride together).**
   - The datum: *"the true gas, forward-modeled through the same instrument
     with a free amplitude, scores 0.0101 — flat across the swept amplitude
     range — while the grid reached 0.0026: the field that fits the flux
     better than the truth does is carrying the wrong gas."*
   - THE caveat (mandatory beside every 4×): *"the truth's residual is our
     own forward model's error floor, not nature's — the margin includes
     slack the grid can exploit and the truth cannot, so 'four times better
     than truth' is never 'four times closer to reality.'"*
   - The non-triviality note: the content is the **margin plus the
     amplitude-flatness** (which rules out the amplitude-calibration escape),
     not bare "grid beats truth" — the grid is the argmin of this loss, so
     beating any single field is near-guaranteed by construction.

4. **The pre-registration-defect honesty beat (compact licensed form).** *"The
   run landed in an outcome cell the pre-registered decision table had not
   enumerated — it closed the flux-power gate and still failed the structure —
   so the table was declared void, and the verdict was decided on the
   evidence: the truth-vs-grid comparison and the weak structure, on their
   merits, with the defect disclosed as a limitation."* Register: "our own
   prediction scheme was wrong and we said so" — NEVER "we predicted this."
   The implicit assumption the run falsified (closing the flux-power gate ⟺
   recovering structure) is narratable as exactly that: an assumption the
   experiment broke on both sides.

5. **The structure verdict, one pass, neutral (state-once discipline).** The
   demotion is disclosed once: *"the estimator returns a correlation that
   decays with distance, not the coefficient its old 0.6 bar assumed — even
   truth against itself scores 0.0298, so the bar was unreachable as
   implemented and is retired."* Then the honest re-read: *"against the
   achievable ceiling the grid recovers about 25%, below the
   truth-plus-noise reference — genuinely weak."* And the frame confound,
   named once for the whole figure family: *"a real-space-versus-
   redshift-space mismatch of 1.3–2.5 Mpc/h — comparable to the 2 Mpc/h
   scale being scored — depresses every number on this figure, the classical
   reference included, independent of reconstruction error."* This whole
   axis is SUPPORTING; the flux-loss datum carries the verdict.

6. **The classical reference (lower bound, never a floor).** *"An idealized
   classical linear reconstruction reaches 0.079 — and was still rising when
   our compute ran out, so it is a lower bound on the classical best case,
   not a ceiling, not a floor, and not our bar."* BARRED: any "classical
   beats neural by N×" reading — the two numbers are not
   configuration-matched (different reconstruction support and smoothing).

7. **Verdicts-not-magnitudes, restated for every grid-vs-network surface.**
   "The grid passed the gate the production network failed" — licensed.
   "The grid's flux error was ~12× smaller" — BARRED (training density 1,024
   vs 64; not a controlled contrast). **"Closes both flux gates" is an
   enumerated claim**: the trainability check and the flux-power gate, and
   only those — no verdict is banked for this run on the mean-flux or
   pixel-distribution gates, and the episode must not imply a clean sweep of
   "all gates but structure."

8. **The honest landing (the arc's last page).** The positive anchors are
   mandatory: at z ≈ 2–3, where the forest is dense, classical tomography
   demonstrably works — the wall is a property of the information-sparse
   low-redshift forest (~2% absorption), under our forward model, not of
   neural methods or of tomography. **The requested close register "a wall of
   information, not of method" is BARRED in that form** — it asserts the
   information-vs-forward-model split the record explicitly refuses. The
   licensed close register: *"not a wall of architecture — a better network
   will not climb it. How much of it is the universe's information and how
   much is our own approximate forward model, we cannot yet say — only that
   the wall is there."* Future work is a one-liner license, never a promise:
   *"the honest forward note is a cell at z ≈ 2, where the signal is
   strong."* Denser-sightline tiers were **not run**; the study rests on the
   1,024-sightline probe — never "more sightlines wouldn't help."

9. **The arc-close register (how the finale may echo episode 01).** The
   opening question may be answered in its narrowed form only: asked "can we
   reconstruct the gas from the light?", the arc's licensed answer is *"under
   this forward model, at this redshift: the light does not pin down the gas
   — and we proved it by letting the truth compete and watching a wrong
   answer win."* The falsified-hypothesis arc is reported as **result and
   lesson, not virtue**: no self-congratulation on rigor, no "our discipline
   was the real discovery" framing — the discipline is shown by the beats
   themselves (pre-registration, voided tables, demoted estimators,
   truth-in-the-race), never claimed as a trophy.

10. **Numeral bars (the standing precision class).** "About four times" —
    never a sharper multiple in prose (the exact quotient 3.88 lives in the
    data). "~25% of the achievable ceiling" — always ceiling-relative, never
    against 0.6. The grid is **192 cells per side** — no other mesh number
    exists in this arc. The flat band is 1.0959–1.0970 from step 5,000 —
    "flat" is licensed, "constant" is not. The gate-scale correlation reading
    is the banked shell nearest 2 Mpc/h (centered at 1.75) — if a caption
    prints the radius, print the shell convention from the sidecar, not a
    rounded "at exactly 2".

## The prior arc review's corrected lines — RE-CONFIRMED as current

The narrative-arc review's two corrected takeaway lines (its old "ep03"/"ep04",
which map to this single episode's victory and reversal beats) are re-issued
**verbatim, unchanged**, as current post-amendments — no later amendment
touches either line:

- **Victory beat:** *"So we replaced the neural field with the most expressive
  field this study could build — a value free at every one of 192³ voxels —
  and it closed the flux-power gate nothing else had. Escaping the collapse
  was necessary. The episode's question: was it sufficient?"* (Intra-episode,
  the "was it sufficient?" turn is the hinge between the two beats.)
- **Reversal beat:** *"Yet that grid fits the observed flux about four times
  better than the true gas field does — while recovering only a fraction of
  the 3D structure it should. The flux can be matched, and even over-matched,
  by the wrong gas. At z = 0.3, the flux simply does not pin down the field.
  We cannot yet say how much of that wall is the universe and how much is our
  own approximate forward model — only that the wall is there, and a better
  network will not climb it."*
- **Indivisibility rider (binding):** the reversal beat is quoted whole or
  not at all. The sentence *"At z = 0.3, the flux simply does not pin down
  the field"* may not be excerpted without the *"We cannot yet say how much
  of that wall…"* sentence attached — standing alone it collapses into the
  barred "the flux cannot determine the field" register. The freestanding
  licensed paraphrase remains: *"minimizing this flux loss does not identify
  the true field."*

That review's eight binding non-claims also remain in force unchanged; the two
it added late are this episode's core: the integrator-slack rider on the 4×,
and the voided-not-satisfied reading of the decision table.

## Honest takeaway (verb-ceiling-compliant)

The arc's last experiment gave the problem to a field that cannot plead
weakness: four free voxel grids, one per physical field, a value free at every
one of 192³ cells, trained under the byte-identical production lesson from a
pre-committed start. It trained — the variance ratio climbed to the healthy
band and held flat for forty-five thousand steps, where every run of the
previous campaign had collapsed — and it closed the flux-power gate that
nothing else in the arc could touch. Then the decisive comparison: the true
gas itself, forward-modeled through the same instrument with a free amplitude,
fits the observed flux about four times *worse* than the grid does, flat
across the swept amplitude range — a margin padded by our own instrument's
error floor, slack the grid can exploit and the truth cannot. The field that
fits the flux better than
the truth is carrying the wrong gas — about a quarter of the achievable 3D
correlation, below the truth-plus-noise reference. The verdict is
triple-hedged and exact: under this forward model, at this redshift, on one
simulated universe, minimizing this flux loss does not identify the true
field. Escaping the collapse was necessary but not sufficient. How much of
the wall is the universe's information and how much is our own approximate
forward model, the record declines to say — and where the forest is dense,
at z ≈ 2–3, tomography demonstrably works. That is where the honest forward note
points, and where the story ends.

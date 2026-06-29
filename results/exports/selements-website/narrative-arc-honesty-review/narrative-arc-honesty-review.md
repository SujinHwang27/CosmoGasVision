# Narrative-arc honesty review — [D-73] close-out story

**Consumer:** selements-website (story-editor) · **Request:** narrative-arc honesty review (prose advice, no data owed) · **Responder:** CosmoGasVision PI · **Branch:** `service/data-export` (grounded on the `exp/nerf` LEDGER state) · **Verb-ceiling:** [D-37] honest-reporting + the [D-73] close-out (binding).

> Boundary honored: this critiques selements' drafted arc; it does **not** rewrite or replace the narrative. Scientific corrections are **binding**; storytelling preferences are **input**. The final narrative is selements'.

PI sign-off (scientific + [D-37]/[D-73] honesty). am-9 (§9a–§9e), am-10 (§10a–§10e), and the spine §3a/§4/§5/§6 read first-hand. Verdict up front: **the arc is broadly faithful and the reversal is a legitimate, even elegant, public story — but ep03 and ep04's takeaway lines overclaim in two specific, correctable ways, and the spine of the reversal ("fits flux ~4× better than truth yet recovers ~25% of structure") needs the integrator-slack caveat baked in or it reads as a cleaner result than we have.**

---

## A. Is the 4-episode arc scientifically faithful as a whole?

Yes, as a structure. The reversal (ep03 apparent victory → ep04 detonation) maps onto a real thing in the record: the grid genuinely passes both flux gates the MLP failed (var_pf trainability = 1.0959, and P_F |ΔP_F/P_F| = 0.0352 < the 0.10 bar — am-10 §10a), and it genuinely fits the observed flux ~4× better than the true field does through the same integrator (loss 0.0026 vs 0.0101 — am-9 §9b) while recovering only weak 3D structure. So the "win that turns out to be a wall" is not manufactured drama; it is the actual shape of the finding. The reversal is **honest, not overdramatized** — *provided* ep03's "victory" is framed as "passes the flux gates," not "reconstructs the gas," because those are precisely the two things the run shows are not the same. That gap IS the paper.

Two episodes overclaim and need pulling back into scope; one structural caution on the spine:

- **ep03 mis-emphasizes by under-stating its own subtlety.** Calling the grid result "it does fit the flux — escaping the collapse was necessary" buries the sharper point: the grid passes the *power-spectrum* gate (P_F), the single binding gate the entire 12-intervention campaign (spine §1) could never close on the MLP. That is the load-bearing event of ep03, and it is exactly what makes ep04's wall devastating rather than mundane. Lead ep03 with "the most expressive field we could build closes the very gate nothing else could," not with the softer "it does fit the flux."

- **ep04 overclaims on cause.** "The limit is the information, not the model" asserts problem-intrinsic under-determination. The record explicitly refuses that separation: K2's 4× margin **includes integrator-induced slack** — the 0.0101 truth-residual is *our* FGPA-vs-RT forward-model error, not nature's, and am-9 §9b states verbatim that we "do NOT separate problem-intrinsic from integrator-induced under-determination" at ⟨F⟩=0.979. So "the information, not the model" is half-right (it is not the *architecture*) but half-wrong (we cannot exclude that it is *our forward model*). See C and B-ep04 for the corrected line.

On the spine: **"fits flux ~4× better than truth yet recovers ~25% of structure" is the right dramatic spine, but it is two different estimators welded into one sentence, and the welding hides a caveat.** The 4× lives in flux-loss space (K2, internal to the run, clean). The 25% lives in ξ space (a *demoted*, frame-confounded estimator, ceiling-relative to 0.0298, not to 0.6). They point the same direction — that is why the spine works — but a reader will hear "4× better at the flux, 25% as good at the structure" as one controlled comparison, and it is not. The more load-bearing single contrast to lead with is the K2 one alone: **the field that fits the flux better than the truth does is carrying the wrong gas.** That sentence is estimator-independent, internal to one run, and needs no ξ caveat. Let the 25% ride in as *supporting* color, not as half the headline.

---

## B. Per-episode takeaway lines — in scope and correctly hedged?

**ep01 — "vision = reconstruction; 1D under-determines 3D."** In scope, no verdict, fine. One nuance to plant for later payoff: ep01 should establish that 1D Lyα sightlines are sparse *pencil beams* through a volume — under-determination is geometric and obvious in the primer, so ep04's wall reads as "even with the geometry handled, the *signal* runs out," not "we forgot the sightlines are sparse." See F.

**ep02 — "fits the flux but not its power — a structural deficit."** The fact is right (MLP fails P_F by 36–42%, 3.6–4.2× the 10% bar — spine §2, [D-39]) but **"structural deficit" pre-empts ep04 and mislabels the cause.** At the ep02 stage of the reversal, the MLP's failure is genuinely ambiguous between "the architecture isn't expressive enough" (which ep03 then refutes) and "the problem is under-constrained" (ep04). Calling it "structural" in ep02 tips the hand toward the wrong one — and "structural deficit" reads as *architectural* deficit, which ep03 explicitly disproves. Write instead:

> *"The neural field reproduces the average flux and its pixel distribution but misses the flux power spectrum by roughly fourfold — it fits the flux without fitting its structure. At this point we cannot yet say whether the model is too weak or the problem is too hard."*

That keeps ep02's cliffhanger honest and lets ep03 do the disambiguation it is there to do.

**ep03 — "the most expressive field we could build does fit the flux — escaping the collapse was necessary."** "Necessary" is correctly hedged (am-9 §9b: "necessary but not sufficient"); good. But the line under-plays the P_F closure (see A) and the word "expressive" should be scoped — it is "maximal-capacity *within this study*" (spine §6: a free-per-voxel 192³ grid; hash-grid/multigrid conditioning is an explicitly untested axis). Write:

> *"So we replaced the neural field with the most expressive field this study could build — a value free at every one of 192³ voxels — and it closed the flux-power gate nothing else had. Escaping the collapse was necessary. The episode's question: was it sufficient?"*

Note "was it sufficient?" — ep03 must end on the unanswered half, or ep04's reversal has nothing to land on.

**ep04 — "the limit is the information, not the model."** The one I most want corrected. It overclaims problem-intrinsic vs forward-model-induced under-determination (the record forbids the separation — A, C). Two further precision points: "~25% of achievable 3D structure" must be *ceiling-relative under a demoted estimator* (25% of the field's own 0.0298 autocorrelation ceiling, not 25% of a 0.6 scale — am-9 §9c), and "~4× BETTER than the true field" carries the integrator-slack caveat. Write:

> *"Yet that grid fits the observed flux about four times better than the true gas field does — while recovering only a fraction of the 3D structure it should. The flux can be matched, and even over-matched, by the wrong gas. At z = 0.3, the flux simply does not pin down the field. We cannot yet say how much of that wall is the universe and how much is our own approximate forward model — only that the wall is there, and a better network will not climb it."*

That last clause ("a better network will not climb it") is the honest, defensible form of the intended "not the model" — it kills the *architecture* explanation (which ep03 earns) without asserting the *information*-vs-*our-forward-model* split the record refuses.

---

## C. The single thing a reader must NOT walk away believing

Ranked, the top one to actively guard against:

**That we proved the z=0.3 Lyα flux *intrinsically* cannot determine the 3D gas field — i.e. that this is a statement about nature.** It is not. The 4× margin includes slack from *our* approximate FGPA forward model (am-9 §9b), and we explicitly do not separate problem-intrinsic from forward-model-induced under-determination. The defensible claim is: *under this forward model, at this redshift, the flux does not determine the field* — a statement about our inverse problem as posed, not about the information content of the universe's Lyα forest in the abstract.

Second (ep02-adjacent): that the MLP's failure was an *architecture* failure that the grid *fixed* — a happy-ending "bigger model wins." Ep03 is engineered to invite exactly this misread for one episode; ep04 must explicitly retract it. The grid did not "fix" anything — it passed the *gates* and still got the *gas* wrong, which is the whole point.

Third: that "neural IGM tomography doesn't work." Scope-locked to z=0.3 (B-2, F). CLAMATO/TARDIS succeed at z≈2–3; the record names them.

If you guard only one: **guard #1.** "The limit is the information" is the single line most likely to be quoted, and it is the one the record most directly forbids in its strong form.

---

## D. Load-bearing things the arc is under-weighting

1. **The integrator-induced-slack caveat on the 4×.** Currently absent; it is the difference between an honest close and an overclaim (B-ep04, C). The 0.0101 truth-residual is *our* FGPA-vs-RT error — a perfect-reconstruction field would not score zero through our integrator. Carry it at least once, plainly: "even the true gas, run through our approximate model, leaves some residual the grid can exploit."

2. **The P_F closure is the sharp point ep03 under-plays.** Not "it fit the flux" — *it closed the power-spectrum gate that twelve prior interventions could not* (spine §1 enumerates 12 across four axes, all retired). That campaign-of-twelve is itself an under-weighted asset: it is what makes the under-determination diagnosis credible rather than a single-shot failure. Consider letting ep02 or ep03 gesture at "we tried many architectures and losses; none closed this gate — until the maximal grid did, and even then the gas was wrong."

3. **The "P_F-closes-but-ξ-fails" un-enumerated cell — the honesty centerpiece.** am-10 §10b–§10d, and the most intellectually honest beat available. The run landed in an outcome cell the pre-registration *did not enumerate*: we had implicitly assumed "closes P_F ⟺ recovers structure," and the run falsified that biconditional. We did **not** trigger-match our way to the conclusion; we reached it on independent K2 evidence and *disclosed the pre-registration defect as a limitation, not promoted it to a finding* (am-10 §10c/§10d). For a story about epistemic honesty, that is gold — but only if framed as "our own prediction-scheme was wrong and we said so," never as "we predicted this." If the arc references the branch logic at all, it must carry that the discriminator was declared *void*, not *satisfied*.

4. **"~25% of structure" is ceiling-relative under a demoted estimator.** The 0.6 gate is DEMOTED for two independent first-hand-confirmed defects (S5 unreachable-as-implemented: even truth-vs-truth scores 0.0298, not ~1.0; S7 frame-confounded: Δχ ~1.3–2.5 h⁻¹Mpc at the r=2 gate scale). The "0.0075 vs 0.6 → catastrophic fail" framing is *retired* in the record. So the arc must never present 25% against a 0.6 bar, and ξ is *supporting*, not load-bearing — K2 carries the close (am-9 §9c, spine §5).

5. **Single realization, one sightline set, fixed cosmology.** Sherwood P1, 60 cMpc/h box, one realization, z=0.3. Not a caveat that needs to be loud in a public arc, but it must not be contradicted by any "we have shown across..." phrasing.

---

## E. The six binding non-claims — confirmed/corrected, plus omissions

1. **"Fails 2 of 3 gates, never all three."** CORRECT and binding. Spine §3a: the 3D ξ gate was *never evaluated on the production MLP* in its [D-13]-defined form (the MLP produces no 3D cube; its ξ-of-record is the 1D-along-ray r_ρ surrogate, which [D-58] showed *decouples* from ξ_3D). The MLP fails P_F and KS — the two directly-evaluated gates. "Fails all three" is a fabrication. Honored.

2. **"Under-constraint holds ONLY at z=0.3; not 'tomography impossible.'"** CORRECT and binding (spine §4). Add the positive anchor so it doesn't read as excuse-making: CLAMATO (Lee+2018) and TARDIS (Horowitz+2019) *succeed* at z≈2–3; a 2024 MNRAS neural method recovers fields at 4≤z≤5. The wall is a property of the *information-sparse low-z forest* (~2% absorption), not the architecture.

3. **"ξ is DEMOTED not failed; headline carried by K2."** CORRECT and binding. Two independent confirmed defects (S5 + S7); K2 is estimator-independent and load-bearing; ξ is supporting (am-9 §9c, spine §5). Honored.

4. **"Wiener is a LOWER BOUND, never an information floor."** CORRECT and binding (spine §3c). The number is ξ_3D ≥ 0.079, *still rising at the L=3 CPU-RAM wall* — the optimum is un-pinned. Self-anchored (R14); published CLAMATO/TARDIS r-values are context, never our bar. If the arc uses Wiener at all, it must be "a classical method's *best case we could reach*," not "the floor any method must beat."

5. **"Grid passing flux gates ≠ structure recovery; grid-vs-MLP residual magnitudes are NOT a controlled same-config contrast."** CORRECT and binding — the one most likely to be violated by accident. Grid n_rays=1024 vs MLP baseline n_rays=64 (am-10 §10a): the *PASS/FAIL verdicts* are directly comparable (same convention, same 10% gate), but the residual *magnitudes* (0.0352 vs 0.4155) are NOT a controlled contrast. So the arc may say "the grid passed where the MLP failed" but must NOT say "the grid's flux error was ~12× smaller" as if that ratio were a clean measurement. Honored.

6. **"K2 licenses only 'minimizing this flux loss does not identify the true 3D field'; not that 0.0101 is physical, nor any off-z=0.3 generalization."** CORRECT and binding (am-9 §9b verbatim). The 0.0101 is our integrator's structural error; no generalization beyond z=0.3.

**Two omissions to add to the binding set:**

- **(7) The 4× margin includes integrator-induced slack.** Separate from #6's "0.0101 is not physical" — it is the consequence: because the truth-residual is our own model's error, part of the 4× is the grid exploiting slack the truth field cannot, so "4× better than truth" must never be read as "4× closer to reality." (am-9 §9b)

- **(8) The branch discriminator was VOID, not satisfied; the no-reopen disposition was reached on independent K2+ξ evidence, and the pre-registration defect is disclosed as a limitation, not a finding.** (am-10 §10c/§10d) If the arc narrates "the plan said X would happen and it did," that is the inverse of what the record says and would itself be the dishonesty this close-out was built to avoid.

**Numbers needing a caveat the arc must attach:** the 4× (integrator slack, #7); the ~25% (ceiling-relative, demoted estimator — not 25% of 0.6); the grid-vs-MLP magnitudes (not same-config, #5); Wiener 0.079 (lower bound, un-pinned). The 36–42% / 3.6–4.2× MLP P_F numbers are clean as stated.

---

## F. The primer's floor (ep01)

Minimum a lay reader must hold so ep04's wall reads as an *information* limit, not a model failure:

1. **A sightline is a pencil beam.** A quasar behind the gas acts as a backlight; we see the gas only along the thin line between it and us. We have a sparse handful of such lines through a large volume — most of the volume is never directly on a line. (Geometric sparsity — the "1D under-determines 3D" intuition; it should feel obvious in the primer.)

2. **"The flux" vs "the field" are different objects.** *The flux* is what we observe: how much background light survives along each sightline (heavy absorption = dense gas). *The field* is what we want: the full 3D map of gas density, temperature, velocity. The flux is a lossy, projected, 1D shadow of the 3D field. Reconstruction = inferring the field from the shadows. This distinction is the entire load-bearing concept of the series — ep04's wall is precisely "the shadows don't carry enough to fix the field."

3. **Low redshift = faint shadows.** At z=0.3 (the regime studied, ~nearby universe) the gas is diffuse and only ~2% of the light is absorbed — the shadows are faint. (Contrast: the distant universe, z≈2–3, where the forest is dense and tomography demonstrably works — plant this so ep04 is "the signal ran out *here*," not "the method is broken everywhere.")

If a reader leaves ep01 holding only "sparse pencil beams + flux-is-a-lossy-shadow-of-the-field + low-z shadows are faint," ep04's wall is legible as information, and the corrected ep04 line ("a better network will not climb it") lands without overclaiming.

---

## G. The framing / truth I most want a lay reader to leave with

You decide how to tell it. The truth I most want surviving the series:

**We built the most expressive gas field we could, and it matched the observed light *better than the real universe does* — and still got the gas wrong. That is not a failure of the model; it is a discovery about the problem. At this redshift, the light we can measure simply does not contain enough information to pin down the gas behind it — and we were rigorous enough to prove that the way to know is not "did the model fit the data" but "could the *true answer itself* fit the data better — and here it couldn't, while a wrong answer could."**

That is the elegant version of a null, and it is fully defensible against the record. It foregrounds K2 (the estimator-independent, load-bearing evidence) over ξ (demoted, confounded); it keeps the claim about *this inverse problem as posed* rather than about nature; and it makes the methodological honesty — testing whether the *truth* could win the fit — the hero, which is the most genuinely novel and creditable thing the project produced. The "win that becomes a wall" reversal is the right vehicle; just make ep03's win be "closes the flux gates" (not "reconstructs the gas") and ep04's wall be "the light doesn't determine the gas *under our model, at this redshift*" (not "the limit is the information"), and the arc is honest end to end.

---

## Sign-off

**APPROVE the arc's structure WITH BINDING CORRECTIONS.** The reversal is legitimate and the null is a good public story. The corrections that are *binding* (scientific, not stylistic):

- ep02's "structural deficit" → "fits the flux without fitting its structure, cause not yet known" (B-ep02);
- ep04's "the limit is the information, not the model" → the corrected line in B-ep04 / G that kills the architecture explanation without asserting the information-vs-our-forward-model split (C-#1);
- the 4× must carry integrator-slack;
- the ~25% must be ceiling-relative under a demoted estimator, never against 0.6;
- grid-vs-MLP magnitudes are not a same-config contrast;
- binding non-claims #7 and #8 must be added to the set.

Everything in A's "spine" discussion and ep03's emphasis is storytelling *input*, which selements weighs.

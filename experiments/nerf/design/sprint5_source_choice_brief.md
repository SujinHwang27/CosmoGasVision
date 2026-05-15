# Sprint-5 source-choice brief — candidate options for dual-panel pre-review

**Status**: DRAFT — candidate-options package for gate-4 parallel dual-panel pre-review (3-examiner `defense-panel` + 4-persona `support-researcher`). **NOT a decision-of-record.** PI re-orientation post sprint-4 branch-iv PROCESS-FAILURE outcome ([D-51] DONE / [D-52] amendment 7 routing). Decision-of-record lands at gate-5 PI re-review on dual-panel verdicts as a new [D-XX] entry.

**Predecessor**: [D-52] (sprint-5 scope-locked to option (c), 2026-05-13a/b; full design spec PENDING under the assumption that Â_truth(r) would PASS sprint-4 gate-(a)). [D-51] (sprint-4 30-epoch Juno H100 run 2026-05-14, branch-iv PROCESS-FAILURE: Â_overall = 0.379 [CI 0.362, 0.396]; mean+var baseline = 0.368; AD-5 margin 1.15 pp ≪ 10 pp).

**Governance posture** (R15 binding, 2026-05-13c): this brief is a PI-authored candidate-options package, NOT a PI-authorized stage spec. PI preliminary lean in §5 is **explicitly PROVISIONAL** and is lifted only by the dual-panel pre-review verdict + PI re-review at gate-5.

---

## §1 Context & rationale

[D-52]-as-written scope-locked sprint-5 to **option (c)**: ship Â_truth(r) alone as a probe-classifier discriminability lower bound (Alain & Bengio 2017 framing), with Δ̂(r) demoted to follow-on. The lock was conditional on the unstated assumption that the sprint-4 truth-baseline 3D ResNet-18 would PASS gate-(a) sanity floor (Â_overall CI lower > 0.50) AND clear AD-5 trivial-baseline margin ≥ 10 pp ([D-52] amendments 6 + 7).

The 2026-05-14 Juno H100 run ([D-51] DONE block, run_id `sprint4_1778774878`) returned:

- Â_overall = **0.379** [ord-bootstrap CI 0.369, 0.390; block-bootstrap CI 0.362, 0.396] — gate-(a) CI lower 0.369 < 0.50 → **FAIL**.
- mean+variance 2-scalar baseline = **0.368**; margin = **1.15 pp ≪ 10 pp** → AD-5 **FAIL**.
- Pre-committed [D-52] amendment 7 routing fires **branch-iv PROCESS-FAILURE**: NO Â_truth(r) value publishable as headline-claim in this submission cycle per R14 self-anchored-bar protection.

Two structural consequences:

1. **Option (c) as written is structurally inapplicable.** The probe-classifier-lower-bound deliverable was anchored on the ResNet-18 at 32³ ρ-crop substrate clearing AD-5; it did not. The instrument fails to discriminate above 2-scalar moments at this substrate — there is no Â_truth(r) ceiling to ship.

2. **[D-37]-Extension R10 retired-model-reuse contract is engaged.** The branch-iv ResNet-18 checkpoint (`resnet18_3d_4class_best.pt`, 32 MB, banked at `cloud_runs/Sprint4-30ep-a13dce8-20260514-110740-e72fca/checkpoints/`) is retired-for-reason-X = "fails AD-5 + gate-(a) on its own substrate". Any sprint-5 candidate that proposes to reuse it MUST supply an explicit orthogonality argument: reason-for-retirement ⊥ purpose-for-reuse. Default presumption per R10: **NOT admissible**.

The dual-panel pre-review is therefore being convened to stress-test the candidate options enumerated below before any PI sign-off + LEDGER decision-of-record entry, per R15-banking discipline (PI option-(a) overturn 2026-05-13a + PI option-(c)-as-written overturn 2026-05-13b + PI inherited-claim overturn 2026-05-13c).

---

## §2 Candidate options enumeration

Five options enumerated. Hedging verbs throughout per R9 invariance-verb discipline; falsifiable predicates per R8 cascade-close formality; R10/R13/R14 audits in §3.

### Option (c′) — Re-source Â_truth(r) at a different substrate scale

**Description**: Build a new substrate-discriminability instrument with **enlarged crops** (candidates: 48³ at the same n_grid=768 → ~3.75 h⁻¹ Mpc per crop, or 64³ → ~5 h⁻¹ Mpc) and/or **richer per-voxel features** (candidates: stack [ρ, T, X_HI] channels instead of ρ-only; or [log ρ, ∇·v] for FGPA-aware features). Re-run the [D-51] 5-gate + [D-52] AD-5 protocol; the headline claim, if it clears, is the same probe-classifier-lower-bound but at the new substrate scale.

**Falsifiable hypothesis** (R9-compliant hedging): *"We test whether the 4-way Sherwood-recipe signature, demonstrated by Bolton+2017 at the 1D flux-statistic scale and shown absent beyond 2 summary statistics at 3D ρ crop=32³ scale ([D-51] branch-iv), becomes discriminable above the AD-5 ≥ 10 pp bar at a larger 3D substrate (48³ or 64³) and/or under richer features."*

**Pros**: Most direct rescue of the [D-52] deliverable surface; preserves the cite-precedent narrative (Bolton+2017 / Iršič+2017 1D-scale prior-art context, [D-52] amendment 12). The substrate-scoping finding from branch-iv ([D-51] DONE block) becomes a *partial result* that motivates the larger-substrate sprint rather than the headline finding.

**Cons**:
- The substrate-scoping question may answer "the 32³ scale is fundamentally underresolved, so 48³/64³ will succeed" OR "the 4-class signature is genuinely low-dimensional at all 3D substrate scales below the box scale; expanding crops won't change the answer". Branch-iv did not falsify either; the experiment is unfinished.
- A larger crop reduces n_crops per held-out region, weakening the [D-52] amendment 5 MDE on Â_truth(r) at 80% power (which was already 0.05 at 8k crops). At 64³ on the [D-49] 70/15/15 axis-0 split, test-set n_crops drops to ~200, pushing MDE above 0.10 — unusable.
- 48³ is the more defensible scale (test-set n_crops ≈ 700, MDE ≈ 0.06–0.07 at 80% power).
- Richer-feature path engages a [D-53]-adjacent risk: stacking [T, X_HI] channels may smuggle in supervision-regime signal absent at the ρ-only substrate, complicating the orthogonality argument vs the eventual Δ̂(r) follow-on under [D-47] option (b).

**Closes** ([D-52] amendment 7 4-branch routing under the new substrate): re-engages all four branches (i/ii/iii/iv) freshly; branch-iv at 32³ is informative but does not pre-foreclose 48³.

### Option (c″) — Pivot to the supervision-target axis per [D-53] upstream-untested obligation

**Description**: Change WHAT is supervised on the truth side, not WHERE the substrate-discrimination instrument runs. Two sub-flavors:

- **(c″-A)** Supervise on a **different physical observable**: replace the ρ field as the classifier input substrate with one of T (temperature), X_HI (neutral hydrogen fraction), or v_pec (peculiar velocity). The same 3D ResNet-18 at 32³ crop is re-trained on the alternative observable, AD-5 + 5-gate protocol unchanged.
- **(c″-B)** Supervise on a **different reduction of the same sightline data**: e.g., 1D spectrum-derived τ profiles directly as classifier input (closer to Bolton+2017's 1D-scale prior-art), bypassing the ρ-via-CIC reduction altogether.

**Falsifiable hypothesis** (R9-compliant): *"We test whether the [D-53]-flagged supervision-target axis upstream of D1–D4 — operationalized as: which physical observable carries the cross-physics discriminability signature at NeRF-substrate scale — is the upstream cause of the branch-iv null at ρ-32³, or whether the null is substrate-scale-bound (in which case (c′) is the better-targeted experiment)."*

**Pros**:
- Discharges the [D-53] obligation that the supervision-target axis is **structurally upstream** of D1–D4 (panels' KILLER finding 2026-05-13a). Branch-iv at ρ-substrate is exactly the kind of null that motivates the upstream-untested re-test.
- (c″-A) on T or X_HI directly probes whether the Bolton+2017 flux-statistic-scale signal is carried by the thermal state rather than the density field — a substantively novel scientific question with cite-precedent leverage (Iršič+2017 cross-physics P_F differential is dominantly thermal-state driven via T₀/γ, not purely density).
- (c″-B) is the closest 3D-to-1D translation: it brings the substrate back to where Bolton+2017 published positive 4-class discriminability, with the addition of a 3D classifier architecture as a methodological contribution.

**Cons**:
- (c″-A) on X_HI tests something that may be circular: X_HI is downstream of T + ρ + UV background, so the discriminability question is "do the four physics recipes leave a fingerprint on X_HI that a 3D CNN can read at 32³" — possibly yes, possibly because of the same low-order moments that branch-iv flagged in ρ. Audit risk: does this collapse to the same finding under a relabeled axis?
- (c″-B) is methodologically closer to Bolton+2017 — risks becoming a re-derivation of published prior art rather than a novel contribution; CVPR-register fit weakened.
- [D-53] orthogonality argument is OBLIGATORY before dispatch: 4-item checklist (alternative supervision-target formulations enumerated + failure-mode-of-D-24 stated + falsification rule pre-committed + orthogonality vs D1–D4 declared).
- Open question whether (c″) is genuinely [D-53] upstream-untested OR a sneaky [D-46] physics_id-conditioning re-litigation under a relabeled axis (see §7 panel question).

**Closes** ([D-52] amendment 7 routing): re-engages all 4 branches on the new supervision target; does not foreclose (c′) future-work follow-on.

### Option (d) — Substrate-scoping pivot: accept branch-iv as the headline empirical finding

**Description**: Formally retire the Â_truth(r) gate-(a) pursuit. Reframe Stage 3 as a **substrate-scoping study**: the [D-51] branch-iv outcome IS the deliverable — "at the 3D ρ crop=32³ NeRF-substrate scale, the four Sherwood feedback variants are discriminable only via low-order moments; a 12M-param 3D ResNet adds 1.15 pp over a 2-scalar `[mean, var]` baseline; the higher-order spatial signal that 1D flux statistics resolve (Bolton+2017, Iršič+2017) does not survive the ρ-substrate reduction at this scale." This finding is already partially absorbed into CVPR §3 (B9 substrate-discriminability-probe subsection, 2026-05-14 close of [D-43] Step 5).

**Falsifiable hypothesis** (R9-compliant): *"The substrate-scoping finding (32³ ρ-crop ≈ 2-scalar problem for Sherwood 4-class discriminability) is reported as the empirical result of Stage 3; we do not claim it generalizes beyond 32³, beyond ρ-only features, beyond Sherwood, beyond z=0.3, or beyond a 3D ResNet-18 architecture. Falsifiable by any follow-on at a different substrate scale producing > 10 pp margin over `[mean, var]`."*

**Pros**:
- Honors the [D-37] honest-reporting rule cleanly: branch-iv was the empirical observation, and that IS the publication-worthy result if appropriately scope-statemented.
- Minimal new compute; reuses sprint-4 banked numbers; B9 absorption already underway.
- Cleanly closes the [D-52] sprint-5 source-choice — answer is "sprint-5 ships nothing new; the empirical reference work happened at sprint-4 in the form of a substantive null".
- CVPR submission-ready under [D-43] Step 5 closure (already DONE 2026-05-14).

**Cons**:
- R14 re-trigger check is the load-bearing audit. The [D-15] 85% bar is a self-anchored project-internal bar (per [D-36]); branch-iv at Â_overall=0.379 is below it. Reporting "32³ is below the 85% bar" is rule-7-fragile (self-anchored bar + author-defined measurement + symmetric-disclosure publication route) — same R14 trigger as the [D-52] option-(c)-as-written critique. Rescue conditions required: external observational anchor (NONE exists per [D-36]) OR pre-committed process-failure path producing NO-publication-under-failure (already partially satisfied by [D-52] amendment 7 branch-iv routing) OR deliverable-demoted to NO-headline-claim (satisfied — §3 B9 frames as a substrate-scoping finding, not as a "Sherwood 4-class is undecidable" universal claim).
- Symmetric-disclosure surface (R14 + [D-37]-ext rule 5): paper text must NOT spin branch-iv as a methodology-contribution-only result while sneaking in the substantive substrate-scoping claim as a positive deliverable. The B9 absorption already has the "substrate-scoping finding" verb-level; the brief must declare whether this is honest or rule-7-fragile.
- Stage 3 closes empirically null at the ρ-32³ substrate; the [D-15] 85% bar is **not met** AND **not refuted** — only locally falsified at one substrate. R8 cascade-close audit: does (d) "close" the [D-15] question or only "scope-statement" it? Panels must rule.

**Closes** ([D-52] amendment 7 routing): retires the gate-(a) pursuit entirely. Branch-iv outcome IS the result; no further branches engaged.

### Option (e) — Defer / null-out sprint-5

**Description**: Archive sprint-5; declare Stage 3 closed at sprint-4 branch-iv outcome; route compute that would have gone to sprint-5 elsewhere. Equivalent to option (d) but without the "substrate-scoping is the headline" framing — Stage 3 simply closes and the paper's Stage 3 contribution is methodological (R8–R15 discipline + 4-branch routing caught a substrate-level signal deficit) without a substantive empirical headline.

**Falsifiable hypothesis** (R9-compliant): *"The methodology contribution (pre-committed 4-branch outcome routing with R14 self-anchored-bar protection caught branch-iv cleanly) is the publishable result; the substantive empirical finding at branch-iv is reported as evidence but not as a headline."*

**Pros**: 
- Cleanest [D-37] honest-reporting verb-level; lowest rule-7-fragility risk.
- Zero compute cost on sprint-5 line.
- Frees compute headroom for Stage 2b follow-ons that the [D-43] CVPR cut deferred (T4 sightline-density × physics ablation matrix, etc.).

**Cons**:
- §3 CVPR text loses the substantive empirical headline; methodology-only register risks Lipton & Steinhardt 2019 §4 "negative result without positive deliverable" troubling-trend pattern (cited under R11 venue-register distinction).
- Stage 3's investment (sprints 1–4 infrastructure: [D-48] cache + [D-49] split + [D-50] CIC refactor + [D-51] truth-baseline 5-gate + AD-5 protocol) becomes a methodology paper for which the empirical instance is "and then we ran it once and got branch-iv". Audit risk: is this a CVPR-worthy paper or a journal-track methodology paper?
- The [D-43] journal-track split ([D-45] master-source architecture, separate venue manifest for journal venue) is the natural home for option-(e)-style long-form methodology reporting; CVPR submission may need to lean on option (d)'s substrate-scoping finding to carry a substantive empirical headline.

**Closes**: same as (d), but with the framing pivoted from "substrate-scoping IS the result" to "methodology-discipline IS the result; substrate-scoping is supporting evidence".

### Further options considered but not enumerated as primary

- **Option (f)** — Re-architecture the classifier (3D Vision Transformer, 3D PointNet, etc.): rejected at brief-level because it confounds substrate-scale + architecture; the (c′) substrate-scoping test should run before architecture is varied. Available as panel push-back if either panel raises.
- **Option (g)** — Re-source the truth field itself (e.g., upgrade from CIC n_grid=768 to TSC or n_grid=1024 deposition): rejected because [D-50] gate (b) numerical equivalence pinned CIC chunked-scatter to the per-corner reference; the substrate is faithful. Sub-pixel-resolution gain unlikely to flip discriminability by 10 pp.

---

## §3 R-rule audit per option

### Option (c′) re-source at different substrate scale

- **R8 cascade-close formality**: (c′) does NOT close [D-52] amendment 7 branch-iv routing; it re-engages all four branches at a new substrate. The branch-iv outcome at 32³ is informative but not foreclosing. Verb: "we extend the substrate-discriminability test to a larger scale" — NOT "we resolve the substrate-scoping question".
- **R9 invariance-verb discipline**: scope sentence obligatory — "the Sherwood 4-class signature, established at 1D flux-stat scale (Bolton+2017) and shown absent beyond 2 summary statistics at 3D ρ-crop=32³ (this work, [D-51] branch-iv), is tested at 3D ρ-crop=48³ (or 64³) under the same 3D ResNet-18 instrument". NO invariance claim; "discriminability at a substrate" is the verb, not "physics-invariant discriminability".
- **R10 retired-model-reuse contract**: branch-iv ResNet-18 checkpoint is RETIRED for "fails AD-5 + gate-(a) at 32³ ρ-substrate". (c′) trains a FRESH ResNet-18 at 48³ from scratch — same architecture, different input shape, different optimizer state. Orthogonality: reason-for-retirement (AD-5 fail at substrate X) ⊥ purpose-for-reuse (the architecture as instrument at substrate Y). Default-NOT-admissible per R10; explicit orthogonality argument is "the substrate is the retired axis, not the architecture; a fresh init at the new substrate is structurally a new experiment". Panels must verify this is sufficient.
- **R13 scope-lock re-verbing**: (c′)'s deliverable surface IS the probe-classifier-lower-bound headline-claim originally [D-52]-as-written intended. Re-verbing required if (c′) succeeds: the framing-verb shifts from "[D-52] sprint-5 ships ceiling-claim" to "[D-52] sprint-5 reopened at substrate-scale 48³ ships ceiling-claim at that substrate". Pre-commit scope sentence + R9 in advance.
- **R14 self-anchored bar + symmetric disclosure**: the [D-15] 0.85 bar is still the project-internal self-anchored bar (per [D-36]); (c′) replays the same R14-fragility unless rescue conditions hold. Rescue option (i) external anchor: NONE. Rescue option (ii) pre-committed process-failure path: [D-52] amendment 7 4-branch routing is INHERITED — if (c′) at 48³ also hits branch-iv, NO Â_truth(r) value publishes. Rescue (ii) verified: ✓.

### Option (c″) supervision-target pivot

- **R8 cascade-close formality**: (c″) discharges the [D-53] upstream-untested obligation directly. (c″) DOES close the supervision-target axis question — partially, depending on which sub-flavor; (c″-A) on T closes the "is the signal carried by thermal state vs density" sub-question; (c″-B) on 1D τ closes the "is the substrate the issue or the 3D reduction itself" sub-question. Falsifiable predicate cleanly framed; not a cascade-close completeness claim.
- **R9 invariance-verb discipline**: scope sentence — "we test which physical observable (ρ, T, X_HI, v_pec, or 1D τ) carries the Sherwood 4-class signature at 3D-CNN substrate-scale 32³ (or 1D for τ); we do NOT claim invariance across observables". Panels must verify the supervision-target framing isn't smuggled-in physics_id-conditioning ([D-46] D4) under a relabeled axis.
- **R10 retired-model-reuse contract**: the branch-iv ResNet-18 checkpoint is NOT reused; (c″) trains FRESH for each new input observable. Architecture (3D ResNet-18) is reused as the instrument; reason-for-retirement was substrate-scale-bound, not architecture-bound; orthogonality argument: instrument-architecture ⊥ input-observable. R10-admissible.
- **R13 scope-lock re-verbing**: (c″)'s deliverable surface is NOT the same as [D-52]-as-written. Re-verbing OBLIGATORY: from "probe-classifier discriminability lower bound on ρ at 32³" → "probe-classifier discriminability lower bound on `<observable>` at 32³, scoping which physical observable carries the discriminative signature at NeRF substrate scale". This is a substantive scope change, NOT a parameter sweep. Panels must verify the new deliverable surface is CVPR-publishable as a positive contribution OR routes to journal-length register per R11.
- **R14 self-anchored bar + symmetric disclosure**: the [D-15] 0.85 bar transfers if (c″) is framed as a ceiling-claim on the new observable. Rescue path (ii) [D-52] amendment 7 routing applies — but the AD-5 ≥ 10 pp margin bar needs re-calibration: what IS the trivial baseline for T crops? for X_HI crops? for 1D τ profiles? Each sub-flavor needs its own pre-committed `[mean, var]` analog. Panels must rule on whether this is a clean transfer or a new R14 surface.

### Option (d) substrate-scoping pivot

- **R8 cascade-close formality**: (d) closes the [D-15] 85% bar pursuit at ρ-32³ substrate. Does NOT close the bar question at other substrates — explicit scope statement obligatory.
- **R9 invariance-verb discipline**: NO invariance language. The verb is "at 3D ρ-crop=32³ NeRF-substrate scale, we report 4-class discriminability ≈ 2-scalar baseline performance; we do not generalize this finding beyond the substrate, beyond the architecture, beyond Sherwood, beyond z=0.3". Already in §3 B9 absorption.
- **R10 retired-model-reuse contract**: the branch-iv ResNet-18 checkpoint is REPORTED on, not reused. R10 is not triggered (no follow-on inference run). The checkpoint serves as evidence-of-record, not as an instrument-for-further-use.
- **R13 scope-lock re-verbing**: (d) flips the deliverable surface from "ceiling-claim instrument" → "substrate-scoping headline finding". Re-verbing OBLIGATORY and audit-heavy. The CVPR §3 B9 absorption already started this — but the brief flags that the framing-verb ("substrate-scoping finding") and outcome-verb ("the headline IS this null") need a coordinated audit per R13 KILLER pattern (framing assertive + outcome hedged is the trigger; here both must align).
- **R14 self-anchored bar + symmetric disclosure**: BIGGEST RISK. The construction "we ran a self-anchored bar (85%), used a self-defined instrument (32³ ρ-crop 3D ResNet), failed it (37.9% vs 85%), and now publish the failure as a substantive substrate-scoping finding" is the rule-7-fragile core pattern. Rescue conditions checklist:
  - (i) external observational anchor: NONE for the 85% bar per [D-36]. ✗
  - (ii) pre-committed process-failure path producing NO publication: [D-52] amendment 7 branch-iv routes to "NO Â_truth(r) headline-claim" — but option (d) routes to "branch-iv IS the headline-claim, restated as substrate-scoping". Panels must rule on whether this restatement is a valid R14 rescue or a rule-7 fragility re-introduction.
  - (iii) deliverable demoted to NO-publication-as-headline-claim, deferred to follow-on paper: option (e) is this option. Option (d) does NOT take this path.
- **R14 rescue verdict (PI preliminary, panels must confirm)**: option (d) rescues by R14-(ii) IFF the §3 paper text frames the substrate-scoping finding via the 1.15-pp-margin numerical fact (which IS observational-honest) rather than via the implicit "we failed the 85% bar" framing. The B9 absorption appears to do this — but the panels should verify the precise paper-text wording resists rule-7 re-trigger.

### Option (e) defer / null-out

- **R8**: closes Stage 3 without engaging further [D-52] branches; cascade-close formality is "Stage 3 closes at sprint-4 outcome; the methodology contribution stands; no further empirical claims at the substrate-discrimination axis in this paper".
- **R9**: invariance verbs absent. Methodology-only framing.
- **R10**: branch-iv checkpoint preserved as evidence; not reused. Not triggered.
- **R13**: scope-lock re-verbing trivial — deliverable surface IS "Stage 3 methodology contribution", which is what the [D-51] design spec [D-37]-ext rule 2 hedged framing always was at the instrument-level. No re-verbing needed.
- **R14**: LOWEST risk. The self-anchored bar is named, the failure-to-meet is named, the substantive empirical finding is reported as evidence-not-headline, and the headline is the methodology discipline that caught it. Closest to rule-7-immune of the five options.

---

## §4 Pre-committed process-failure paths per [D-52] amendment 7 per option

[D-52] amendment 7 4-branch routing (paraphrased):
- (i) AD-1 / gate-(c) / gate-1 / training-divergence → PROCESS-FAILURE; no value publishable.
- (ii) gate-(b) sparsity / gate-(d) wild-oscillation → rerun with adjusted parameters.
- (iii) all 5 gates PASS + AD-5 PASS → report per above-bar / indistinguishable / below-bar.
- (iv) gate-(a) sanity-floor FAIL OR AD-5 FAIL → ceiling-disqualified; substantive null result, publishable as §4 follow-on caveat, NOT headline.

| Option | (i) routing | (ii) routing | (iii) routing | (iv) routing |
|---|---|---|---|---|
| (c′) substrate-rescale | Inherited unchanged; PROCESS-FAILURE → no publication | Inherited; rerun at adjusted crop size or batch | Above-bar / indistinguishable / below-bar at NEW substrate; report scope-statement explicit | Re-fires branch-iv at NEW substrate → substrate-scoping finding extended to two scales; ceiling-disqualified at both ⇒ stronger substrate-scoping headline; falls back to option (d)-flavored framing |
| (c″) supervision-pivot | Inherited; PROCESS-FAILURE on the new observable's pipeline | Inherited; rerun with adjusted reduction | Above-bar / indistinguishable / below-bar on the new observable; new scope sentence | Branch-iv on new observable → [D-53] upstream-axis empirically falsified at this observable; informative null; rerun on next [D-53] alternative or fall back to option (d) |
| (d) substrate-scoping headline | N/A — no new sprint-5 run; sprint-4 branch-iv IS the result | N/A | N/A | branch-iv IS the (iv) outcome; option (d) IS the (iv) → substrate-scoping reframe |
| (e) defer / null-out | N/A — no sprint-5 run | N/A | N/A | branch-iv (iv) outcome stands; methodology-only framing; no substantive substrate-scoping headline |

Branch-iv handling is the load-bearing differentiator: options (c′) and (c″) ROUTE through it (potentially with informative null outcomes that compound the branch-iv evidence at 32³); options (d) and (e) ACCEPT it as the closing outcome.

---

## §5 PI preliminary lean (R15 PROVISIONAL)

**Preliminary lean**: **(c″) supervision-target pivot** — preferred over (c′) substrate-rescale, (d) substrate-scoping headline, (e) defer.

**Rationale (hedged, PROVISIONAL)**:
- (c″) discharges the [D-53] upstream-untested obligation that 2026-05-13a dual-panel KILLER findings escalated to a binding rule. (c′) does NOT discharge [D-53]; (d)/(e) close Stage 3 without discharging it. Discharging [D-53] before further sprint-N dispatches is structurally upstream of any further saturation-band-deficit or substrate-discriminability work.
- (c″-A on T) is the most leverage-rich sub-flavor: thermal state is the dominant driver of Iršič+2017 cross-physics P_F differential (T₀/γ), so a positive result is the strongest substantive scientific finding available in the remaining options; a negative result is the strongest informative null available.
- branch-iv at ρ-32³ is itself evidence FOR the [D-53] upstream-axis hypothesis (the panels' interpretation 2026-05-13a was prescient); (c″) is the direct test of that hypothesis.
- (c′) is a defensible second choice if the panels rule (c″) is non-CVPR-publishable on R11 venue-register grounds; (d) is a defensible substrate-scoping fallback; (e) is the cleanest R14-immune choice but loses the substantive empirical headline.

**EXPLICITLY NOT a PI authorization.** Per R15 (BANKED 2026-05-13c per the three-failure operational test: option-(a) overturn + option-(c)-as-written overturn + inherited-claim overturn), this PI sign-off is PROVISIONAL. Provisional status is lifted only by gate-4 dual-panel pre-review APPROVE + gate-5 PI re-review. No downstream dispatch is authorized off this lean.

---

## §6 Compute budget envelope (rough)

All estimates assume Juno H100 (current cleared dispatch path; quota granted 2026-05-12; sprint-4 30-epoch ran in 4 min 26 s on the cleared path).

| Option | Compute | Data-prep prerequisite | Approx. $ via Juno (informational) |
|---|---|---|---|
| (c′) 48³ rescale | ~10–15 min/seed H100; 3 seeds ≈ 45 min. Plus AD-5 baseline retrain at 48³ (~5 min). | None new — same [D-50] CIC product. | ≪ $5 effective (Juno quota; no per-hour billing) |
| (c′) 64³ rescale | Same wallclock band; test-set n_crops drops to ~200 making AD-5 unusable — exclude. | None new. | n/a (excluded) |
| (c″-A) T-observable at 32³ | Same wallclock as sprint-4 (~5 min H100); fresh training. | Requires T field extraction from snap_012 — likely a CIC analog of [D-50] for T. Possibly significant data-engineering cost (1–2 days). | < $5 effective + ~1–2 days data-engineering |
| (c″-A) X_HI at 32³ | Same wallclock. | X_HI extraction analog. | < $5 + data-engineering |
| (c″-A) v_pec at 32³ | Same wallclock + vector-field input adaptation (3 input channels). | v_pec extraction analog. | < $5 + data-engineering |
| (c″-B) 1D τ direct | Significantly different architecture (1D-CNN, not 3D-ResNet). Implementation cost. | None — uses existing τ files. | < $5 + ~3–5 days re-implementation |
| (d) substrate-scoping | $0; reuses sprint-4 branch-iv numbers. | None. | $0 |
| (e) defer | $0. | None. | $0 |

No commitment in this brief — informational for panel cost-benefit weighing.

---

## §7 Open questions for the panels

The candidate options have R8/R9/R10/R13/R14 audits in §3, but several substantive questions remain where the panels are explicitly invited to push:

1. **(c″) [D-53] discharge vs [D-46] re-litigation**: is supervising on T at 32³ ρ-substrate a genuine [D-53] upstream-axis test, or is it a sneaky [D-46] physics_id-conditioning re-litigation under a relabeled axis (D4 was data-axis/conditioning; T is a different physical observable but at the same crop substrate)? Panels rule on whether the [D-53] orthogonality argument (different supervision target ⊥ same conditioning regime) holds.

2. **(d) R14 rescue verdict on substrate-scoping framing**: does framing branch-iv as a substantive substrate-scoping finding via the 1.15-pp-margin numerical fact rescue R14, or does the §3 B9 absorption text contain implicit "we failed the 85% bar" framing that re-triggers rule 7? Panels read the CVPR §3 B9 paragraph and rule on rule-7 fragility.

3. **(c′) substrate-rescale R10 admissibility**: is "architecture reused at new substrate" a sufficient R10 orthogonality argument (reason-for-retirement ⊥ purpose-for-reuse), or does the panels' R10 default-NOT-admissible presumption require something stronger (e.g., a fresh-architecture instrument at 48³)?

4. **R11 venue-register fit for each option**: CVPR-publishable headline ≠ journal-length headline. (c″-A) probe-on-T is potentially a stronger CVPR contribution than (d); (e) is potentially a stronger journal-track contribution than (d). Panels rule on the venue-register suitability of each option's headline-claim.

5. **(c″-B) prior-art collision with Bolton+2017**: is "3D ResNet on 1D τ profiles" sufficiently novel to clear CVPR R2's "this is a re-derivation of Bolton+2017's published result with a new model" attack, given the methodology cite-precedent landed in [D-52] amendment 12?

6. **AD-5 transferability under (c″)**: what IS the AD-5 trivial-baseline analog for T, X_HI, v_pec, or 1D τ substrates? Is `[mean, var]` still the right 2-scalar baseline, or does T need `[mean, var, T_lower_bound, T_upper_bound]` (4-scalar) because of the [D-39] saturation-band finding? Panels suggest the right per-substrate AD-5 spec.

7. **R8 cascade-close formality on (d)**: does (d) "close" the [D-15] 85% bar pursuit (at ρ-32³ only) per R8, or only "scope-statement" it pending a future substrate rescale? Both wordings have different paper-text implications.

8. **Option ordering under multiple panels' aggregate preferences**: if the 3-examiner panel and the 4-persona panel converge on different orderings (e.g., 3-examiner prefers (c′), 4-persona prefers (c″)), what is the gate-5 PI re-review's rule for reconciliation? Per [D-37]-ext rule 6 + R15: PI re-review absorbs both verdicts and re-issues a non-provisional [D-XX] decision-of-record; no panel has tie-breaking authority a priori. Brief flags this in advance so panels do not assume convergence.

9. **What is the right scope of "Stage 3" as a CVPR §3 contribution under each option?** Option (c′)/(c″) preserve "Stage 3 = empirical ceiling work, branched twice"; option (d) makes Stage 3 = "substrate-scoping headline finding"; option (e) makes Stage 3 = "methodology discipline contribution". Each implies a different §3 / §4 / §0 register. Panels rule on what's CVPR-publishable.

10. **Stage 3 closure timing under each option** — for [D-43] downstream implications: option (d)/(e) close Stage 3 now (submission-ready); (c′)/(c″) extend Stage 3 to a sprint-5 run before submission (~1–2 days of compute + 1–2 days of data-engineering depending on sub-flavor) OR after (post-CVPR follow-on paper). Panels' verdict implies a [D-43] cut-sequence amendment.

---

## §8 What this brief is NOT

- NOT a decision-of-record. The decision-of-record lands at gate-5 PI re-review as a new [D-XX] entry in LEDGER §3 after both panel verdicts arrive.
- NOT an authorization to dispatch. No sprint-5 work begins on the basis of §5's preliminary lean alone.
- NOT a foreclosure on options not enumerated. Panels are invited to surface option (f)/(g) or others on the merits; the brief named them but rejected at brief-level rather than at decision-of-record level.
- NOT a paper-text edit. The CVPR §3 B9 absorption from 2026-05-14 stands; any paper-text consequence of the gate-5 decision-of-record lands as a separate latex-author dispatch under [D-43] Step 5 carry-forward.

---

## References

[D-12] anti-leakage rule. [D-13] gate scale. [D-15] Stage 3 85% bar (project-internal per [D-36]). [D-24] supervision regime ([D-53] upstream-untested target). [D-36] [D-15] external-anchor retraction. [D-37] honest-reporting rule + Extension 1 rules 1–7 + Extension 2 rules R8–R11 binding + R12 deferred + R13/R14 binding + R15 binding-with-clause-(c). [D-39] saturation-band positive ID. [D-43] CVPR submission cut-sequence. [D-44] sightline-unit bootstrap convention. [D-46] 4-cascade-close (D1/D2/D3/D4 retired). [D-47] option-C hybrid framing. [D-48]/[D-49]/[D-50] sprint-1/2/3 infrastructure. [D-51] sprint-4 30-epoch Juno H100 run (branch-iv PROCESS-FAILURE; the empirical anchor of this brief). [D-52] sprint-5 scope-lock (option (c) as-written, NOW REOPENED by this brief). [D-53] supervision-target upstream-untested obligation. Bolton+2017 MNRAS 464:897 (Sherwood-suite cite-precedent). Iršič+2017 MNRAS 466:4332 (cross-physics P_F differential). Alain & Bengio 2017 arXiv:1610.01644 (probe-classifier framing). Politis & Romano 1994 (block-bootstrap). Lipton & Steinhardt 2019 (R11 venue-register cite). D'Amour+ 2020 arXiv:2011.03395 (underspecification). Manheim & Garrabrant 2018 (Goodhart taxonomy, [D-53] methodology cite).

**Review trail**: PI authored as gate-3 deliverable 2026-05-14 (this session) per user authorization. Status: PROVISIONAL (R15). Next gate: gate-4 parallel dual-panel pre-review (3-examiner `defense-panel` + 4-persona `support-researcher`). Gate-5: PI re-review absorbing both panel verdicts → new [D-XX] decision-of-record in LEDGER §3.

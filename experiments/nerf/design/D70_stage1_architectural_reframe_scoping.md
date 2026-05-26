# [D-70]-stage-1a (1)-reframed architectural scoping — PI design doc

**Status**: **Revision 5 PROVISIONAL** pending defense-panel re-review (2026-05-25). R15-PROVISIONAL + R28-PROVISIONAL until panel APPROVE on this revision.
**Revision: 5.**
**Authored**: 2026-05-25, PI self-dispatch per [D-69] amendment-7 dispatch clause. **Rev 2**: 2026-05-25, K1–K4 defense-panel KILLER absorption (see §0.6); K5/K6 PI-discharged at design-doc layer. **Rev 3**: 2026-05-25, PI absorption of Rev-2 M0 baseline findings — M0 re-spec'd as direction-of-motion gate (option δ); §1.5 dead-basin cite-attribution + skip-count rationale added (Finding 2a+b). **Rev 4**: 2026-05-25, defense-panel REJECT absorption — 3 BLOCKING (K-A ε_tol category-mix; K-B cite-attribution + missing mechanism evidence; K-C checkpoint cherry-fit) + 4 NEEDS-WORK (S-A/B/C/D) + 4 PROBE + OOS, plus 2 CPU pre-flights (A: dying-ReLU mechanism survives; B: σ_within and Bin-D dominance confirmed). §0.7 is merged into §0.8 per PI amendment 3; M0 gate construction is re-built around Spearman ρ + permutation p + integral-metric; M2 gets Bin-D mandatory sub-clauses; (1b) renamed "skip-rich MLP" repo-wide in this doc. **Rev 5 (this revision)**: 2026-05-25, defense-panel REJECT absorption — 3 BLOCKING (B1 (2b) integral PASSes peak-then-sink trajectories; B2 Bin-D 2× threshold clears frozen-init doing nothing; B3 §1.5 amendment 1 "operative cause" overreaches the symmetric-preact null) + 5 SERIOUS (S1 Wilcoxon n=10 marginal-band re-bake; S2 trapezoid grid bias — moots under B1; S3 PDF dump void-pile-up flag; S4 §7 K6 auto-dispatch path retired; S5 Bin-D log-MSE min-samples gate) + 4 PROBE + R30 candidate banked. M0 basket reduced from {1a,1b,1c,2a,2b,2c} to {1a,1b,1c,2a,2c} per B1; Bin-D (i) threshold ratcheted 2× → 10×; §1.5 amendment 1 weakened to "necessary-condition consistent with" + pre-flight C training-time dead-fraction trajectory measurement embedded (executed before Rev 5 finalized; verdict embedded in §1.5).
**Parent decisions**: [D-62] §Architectural Candidates lines 90–100 (candidate (1) BLOCKED standalone — supervision-target-coupled reframe required); [D-69] amendment-7 (γ FAIL_SINKING-narrative-corrected-at-Rev-2 → "uniform R-b-pre1 fire with mixed sub-classification across a decade of lr"; pattern claim ESCALATED Ext-2 R9 provisional-pending-third-regime-test); [D-53] supervision-target axis (NOT discharged; upstream-untested obligation); [D-40] / [D-41] / [D-46] / [D-42] constant-collapse attractor history.
**[D-37]-Ext R2 double-cascade verb-hedge applied throughout**: this is the *first scoping of (1)-reframed under post-(3)-fGPA-AND-(γ)-direct-ρ-MSE double-falsification framing*. Two convergent same-track same-confidence falsifications in <48h: (3) fGPA (R_feas=8.33e-3) and (γ) direct ρ-MSE (mixed-outcome falsification — see §0.6 K1 absorption). Verb register: "candidate", "first scoping", "may address", "hypothesized to break", "untested architectural axis", "first-test architectural intervention against constant-collapse attractor". Forbidden: "structurally correct", "by-design", "highest-leverage", "principled escalation", "addresses the diagnosed pathology directly", "all-cell FAIL_SINKING", "active anti-convergence pathology across a full decade of lr".

---

## §0.5 Parent-envelope CAVEATS discharge — [D-62] §Architectural Candidates → this doc

R30 re-grep performed this session against `experiments/nerf/design/D62_architectural_pivot_scoping.md` (lines 90, 92, 94, 96, 98, 100, 141). Verbatim line content matches inheritance.

| [D-62] line | Requirement | Rev-1 coverage |
|---|---|---|
| L90 / L92 | (1) BLOCKED standalone — backbone-class swap inherits same supervision target → inherits upstream pathology | **§1 RESOLVES via §1.5 collision-resolution**: amendment-7 binds "no supervisory-regime swap"; the only re-eligible (1)-reframed under BOTH constraints is **architecture-couples-to-the-supervision-it-already-has, NOT swap-of-supervision**. Pick is a parameterization that *changes how the existing direct-ρ-MSE supervision lands on weight-space*, not a new supervision target. See §1.5. |
| L94 | Reframe re-eligibility = supervision-target-coupled (e.g., transformer + density-pretraining = 1∩2) | **§1.5 collision-resolution**: literal 1∩2 reading (transformer + new pretraining target) is FORBIDDEN by amendment-7 (the new-target was just falsified as (γ)). Reading adopted: "supervision-target-coupled" = the architecture must be picked *with explicit reference to* the constant-collapse attractor under the *current* direct-ρ-MSE supervision, i.e., must specify the gradient-flow / loss-surface mechanism by which the new architecture changes the attractor structure. This is a stricter reading than 1∩2, not a looser one. |
| L96 | Honesty-audit downgrade: heaviest implementation, weakest prior evidence; no IGM-3D-reconstruction architecture-class precedent | **§1 inherits unchanged**: every (1)-reframed candidate scored in §1 carries the "no IGM-NeRF precedent" tag; supporting literature is from general-purpose NeRF / SIREN / hybrid-grid tradition, not IGM. |
| L98 | Convolutional discriminator / GAN-class: mode collapse is a different failure regime; do NOT re-introduce without explicit anti-mode-collapse spec | **§1 EXCLUDES** GAN/adversarial candidates from the slate. |
| L100 | Within-class close routes to [D-65 stub] further-class-pivot (out of scope) | **§2 routing**: M0 / M1 / M2 / M3 FAIL routes to [D-65 stub] per [D-62] L100. No within-(1) iteration past first-candidate FAIL without panel re-review (Ext-2 R3 anti-degeneracy). |
| L141 | (1) BLOCKED standalone per panel BINDING | §1.5 collision-resolution does NOT lift the BINDING; it states the constructive reading that satisfies BOTH the [D-62] BINDING and the amendment-7 supervision-swap prohibition. Panel re-review on this Rev 1 is the gate. |

R30 discharge note: Rev 3 claimed "R30 BANKED" on the strength of this single same-session re-grep. **Rev 4 retracted that claim per panel P-B 2026-05-25 and Rev 4 success-criterion 19**: same-cycle re-grep does NOT constitute n=2 genuinely-distinct events. **Rev 5 status: DEFERRED-BANKED** — PI banked R30 candidate this turn (2026-05-25, Rev 4 → Rev 5 cycle) after a second occurrence of PI-brief grep miscount (Rev 3 K-B :81 line-cite miscount + Rev 4 `torch.cat` hit-count miscount). The new PI procedural rule banked as a candidate: briefs that load-bear on grep/glob/line-count must include the exact command + raw count, NOT paraphrased summaries. Pending one more operational test before BANKED promotion (third sighting OR next panel-cycle pre-review APPROVE of a brief that follows the rule). The Rev 5 §1.5 amendment 1, §0.9 absorption block, and §1 (1b) bullet all cite explicit grep commands + raw counts in-session per the new discipline.

---

## §0.6 Rev 1 → Rev 2 absorption (defense-panel K1–K6 ruling)

PI re-verified the d69 lr-sensitivity per-cell JSONs first-hand (`experiments/nerf/artifacts/d69_lr_probe/`) and ruled on the K1–K6 KILLERs from the closing defense-panel session. Rev 2 absorbs K1–K4 into the body; K5 and K6 are PI-discharged at design-doc layer (see rows below). **Rev 4 retracts K6 PI-discharge per success-criterion 18**: K6 pattern-claim pre-commit is DE-ACTIVATED — (1b) M0 PASS under Rev 4 does NOT discharge the Ext-2 R9 pattern claim, because the Rev 3 M0 spec on which K6 was load-bearing is now retired. Discharge requires a fresh gate-2 panel cycle on the Rev 4 spec.

| KILLER | Caught | Rev 2 disposition |
|---|---|---|
| **K1** | Rev 1 over-extended the d69 probe as "all-cell FAIL_SINKING" / "active anti-convergence pathology across a full decade of lr". Actual per-cell JSONs: lr=1e-4 is FAIL_SINKING (monotone decay 1.63e-5 → 2.25e-6); lr=5e-4 is UNKNOWN_NON_MONOTONIC (peak 1.48e-5 @step 250, then sink); lr=1e-3 is UNKNOWN_NON_MONOTONIC (peak 1.97e-5 @step 250, then sink). All 3 cells fire R-b-pre1 constant-density backstop at step=200 with var_ratio_spread/mean ∈ {2.07, 1.59, 1.99}. | **ABSORBED into §0 framing, §1.5 mechanism rationale ¶1, §6 Pattern-claim status.** Narrative narrowed to "uniform R-b-pre1 fire with mixed sub-classification across a decade of lr." Mechanical (γ) FREEZE per pre-committed verdict-matrix `fallback_mixed_outcome` row still holds — verdict-matrix routing is unchanged. The constant-collapse-attractor mechanism story still holds (R-b-pre1 = constant-density backstop fires in all 3 cells). |
| **K2** | Rev 1 (1b) mechanism story ("identity-bias breaks constant-collapse"; "Softplus head still allows constant-prediction collapse") was post-hoc rationalization without literature anchor. Panel required either a citation or a CPU toy demonstrating the mechanism on a 1D analogue. | **ABSORBED via literature anchors only** per PI ruling. Anchors added to §1 (1b) bullet + §1.5 mechanism rationale ¶1: Sitzmann et al. 2020 (SIREN) §3 and Lu et al. 2019 ("Dying ReLU and Initialization"). **Rev 4 strengthens this via pre-flight A measurement evidence (§1.5 amendment paragraph)** — the mechanism story is no longer literature-anchor-only at Rev 4. |
| **K3** | Rev 1 M0 PASS bar of `> 0.1` was hand-picked ("10× above the worst (γ) floor"), not calibrated against any baseline. | **ABSORBED into §2 M0 row (Rev 2)**, but the Rev 2 noise-floor-bar formulation was itself superseded at Rev 3 by the direction-of-motion gate, and at **Rev 4 by the Spearman ρ + permutation + integral-metric construction** (see §2). The K-A panel REJECT on Rev 3's ε_tol category-mix is the immediate ancestor of the Rev 4 re-spec. |
| **K4** | Rev 1 M3 band [0.980, 1.021] was calibrated on (γ) target-data (K3-bootstrap, [D-69] Rev 5) and may not transfer to (1b)-reframed. Panel required either recalibration or demotion to diagnostic. | **ABSORBED: M3 demoted to diagnostic-only at Rev 2, NOT gate-binding.** Recalibration deferred to gate-2 (post-M0 PASS). §2 M3 row annotated. |
| K5 (R9 admissibility) | Panel asked whether "architectural variant" is an admissible third regime distinct from "supervisory regime swap" for the Ext-2 R9 pattern-claim semantics. | **PI-discharged at design-doc layer.** Confound-minimization grounds in §1.5 — same input encoding, same output head, same loss, same data; ONLY MLP body topology changes. |
| K6 (third-regime semantics) | Panel asked for explicit pre-commit on whether a (1b) M0 PASS *discharges* or only *narrows* the Ext-2 R9 pattern claim. | **Rev 4: DE-ACTIVATED per success-criterion 18.** Withdrawal note: *(1b) M0 PASS does NOT discharge pattern-claim under Rev 4 — discharge requires gate-2 panel cycle on the revised spec; this pre-commit was load-bearing on the broken Rev 3 M0 and is retired.* |

**K1 narrative discipline**: throughout Rev 2–Rev 4 the (γ) outcome is referred to as "mixed-outcome falsification" or "uniform R-b-pre1 fire with mixed sub-classification" — never "all-cell FAIL_SINKING" or "active anti-convergence across a full decade of lr". Anti-pattern of record for this design doc.

---

## §0.8 Rev 3 → Rev 4 absorption (defense-panel REJECT + pre-flights A/B + PI amendments)

§0.7 from Rev 3 (the first M0-baseline-findings absorption) is **merged into this §0.8** per PI amendment 3. The lead is the σ_within evidence: pre-flight B (`experiments/nerf/artifacts/d70_m0_baseline/baseline.json` augmentation) measured **σ_within_median = 5.825e-06**, ~8× σ_frozen = 7.278e-07. This confirms the panel K-A magnitude diagnosis: the Rev 3 ε_tol construction (`0.1 × σ_frozen / √N_crops = 7.28e-09`) was using a **cross-seed-of-means** SE in place of the **within-seed crop-to-crop** scale that the per-seed monotonicity test actually probes — a category mix-up, not a unit error. The Spearman ρ + permutation-p path adopted at Rev 4 (§2 M0 re-spec) is robust to this scale precisely because it sidesteps SE construction altogether — it tests monotonicity from rank statistics. This is a **post-hoc validation of an ex-ante robustness pick, not a lucky escape**: the SE-free path was on the table at the Rev 3→Rev 4 re-design specifically because the ε_tol construction looked fragile.

**Panel REJECT items absorbed** (BLOCKING in bold, NEEDS-WORK plain, PROBE/OOS noted):

| Item | Panel finding | Rev 4 PI ruling |
|---|---|---|
| **K-A** | Rev 3 ε_tol = `0.1 × σ_frozen / √N_crops` mixed two estimator categories: σ_frozen is cross-seed SD of crop-mean ratios; the per-seed monotonicity test needs within-seed crop-to-crop SD. The √N_crops shrinkage compounded the category mix-up. | ABSORBED: ε_tol construction RETIRED. Replaced by Spearman ρ + permutation p-value (no σ-knob, no √N, no shrink factor) plus an integral-metric co-spec. See §2 M0 row + success-criteria 1, 2, 6. |
| **K-B** | Rev 3 cited "(`src/models/nerf.py:81, 126, 133`)" but :81 is the Softplus head only; the ReLU body sites were not located. Mechanism evidence was literature-anchor-only with no in-track measurement. | ABSORBED: §1.5 corrected K-B line cites with grep evidence + pre-flight A measurement paragraph inserted verbatim (§1.5 amendment 1). See success-criteria 8, 9, 10. |
| **K-C** | Rev 3 checkpoint grid {0, 100, 250, 500} matched the (γ) lr=1e-3 peak at 250 exactly — read by the panel as cherry-fit. | ABSORBED: checkpoint grid expanded to **6 checkpoints {0, 50, 100, 175, 250, 350, 500}** for the integral metric. See success-criterion 2. |
| S-A | Cosmology cite mismatch in §1.5 hedge (the original hedge cited Tepper-García). | ABSORBED: §1.5 honest-hedge now cites Lukić+ 2015 §3 + Bolton+ 2017 Sherwood density-PDF panels for the 5-decade log-density cosmological context. See success-criterion 10. |
| S-B | 8/10 binomial gives weak power against modest effect sizes. | ABSORBED: per-seed Wilcoxon signed-rank on `(Var_ratio(500) − Var_ratio(0))` replaces the 8/10 binomial as the primary aggregate test; α=0.05 one-sided pre-committed. The 8/10 binary still appears as a coarse PASS/MARGINAL/FAIL band routing layer (≥8/10 PASS, 6–7/10 MARGINAL, <6 FAIL — success-criterion 5) but is no longer load-bearing for the M0 verdict. See success-criteria 5, 6. |
| S-C | "Every-layer skip" was ambiguous (DeepSDF-style? per-layer adders? what's the in_features count per layer?); the "ResMLP" name was misleading. | ABSORBED: §1 per-layer spec table added (success-criterion 12) clarifying the DeepSDF skip-input variant (concatenate encoded coords [+g] into the input of every body layer; body ReLU unchanged; head unchanged). Repo-wide rename **"ResMLP" → "skip-rich MLP"** applied throughout this doc; future code flag is `--arch skip-rich-mlp` not `--arch resnet-skip-rich`. LEDGER §3 owes the same rename — flagged for PI write at panel APPROVE. See success-criteria 12, 13. |
| **S-D** | Bin-D void-floor saturation could let aggregate Var_ratio PASS while Bin-D collapses. | ABSORBED: pre-flight B confirmed at frozen init that Bin-D aggregate ratio = **7.91e-09** vs aggregate mean = **2.41e-06** (~300× smaller). Per-bin gating is **mandatory**, not routing-lighter. **M2 Bin-D promoted to mandatory sub-clause (both (i) and (ii) AND-gated)**; M1 inherits the same structure routing-lighter per prior PI memo. See §2 M2 row + amendment 2. |
| P-A/C/D | Panel PROBE items (not BLOCKING): probe checkpoint density between 250 and 500; probe whether the void-floor PDF tail is the right discriminator; probe whether the 10-seed × 100-crop protocol scales. | Checkpoint density addressed by the 6-checkpoint integral grid; void-floor discriminator addressed by the (γ) output log-density PDF histogram dump at M0 eval (success-criterion 11); 10×100 protocol carried forward as the M0 protocol unchanged. |
| OOS | Out-of-scope follow-ups: pre-flight C (training-time dying-ReLU recovery dynamics); pre-flight D (skip-rich-MLP dead-fraction baseline); LEDGER rename writeback. | PARKED at design-doc layer; will be PI-dispatched after panel APPROVE. |

**Pre-flight A — dying-ReLU mechanism evidence** (`cloud_runs/d70_preflight_A_dying_relu/hist.json` + `hist.png`):
3 seeds (0, 1, 2) × 4096 coordinates × 8 ReLU application sites in the current IGMNeRF body. Pre-committed PASS threshold: any site with dead_fraction > 5% across any seed → mechanism_supported = True (threshold set 2026-05-25 *prior* to running the probe).
- **`max_dead_fraction_across_seeds_per_site` = [0.503, 0.503, 0.500, 0.518, 0.514, 0.537, 0.524, 0.503]** (all 8 body sites)
- **`site_with_max_dead_fraction`**: site_index 5 (`layers2[1]`), max_dead_frac = **0.5371** at seed 1
- `any_site_above_5_percent = true`; mechanism_supported = True with order-of-magnitude margin (observed ~54% dead-fraction vs 5% threshold)

(Note: the dispatch brief rounded 0.5371 → 0.54 and quoted the rounded value as 0.54; the exact JSON value is 0.5371. This is a rounding-only deviation; the order-of-magnitude conclusion is unchanged.)

**Pre-flight B — σ_within + Bin-D dominance** (`experiments/nerf/artifacts/d70_m0_baseline/baseline.json` Rev 4 augmentation):
- `sigma_within_median = 5.825e-06` (~8× σ_frozen = 7.278e-07); `sigma_within_worst = 1.032e-05`. K-A magnitude diagnosis confirmed.
- Per-bin aggregate frozen-init Var_ratio: Bin-A = 2.310e-03 (void; largest contribution by 6 orders of magnitude over Bin-D), Bin-B = 1.123e-05, Bin-C = 7.428e-07, **Bin-D = 7.913e-09**. Bin-D / aggregate-mean = 7.913e-09 / 2.412e-06 ≈ 1/305 — the 300× dominance ratio cited in the dispatch is confirmed (within rounding).
- (Note: the dispatch describes Bin-A as the dominant bin numerically; the per-bin bin_A/bin_D ratio is ~3e5. The Rev 4 S-D pre-commit binds Bin-D specifically because Bin-D is the void-floor regime where constant-collapse hides — Bin-A dominates the *variance budget* under direct-ρ-MSE but is not the bin where the constant-collapse pathology is masked.)

**σ_within disclosure mandatory throughout §2** (amendment 4): every M0/M1/M2 numerical report (current + future) must disclose σ_within per-bin AND aggregate. No bare point estimates. See §2 preamble standing disclosure clause.

**Carry-forwards (unchanged from Rev 3)**: K5 PI-discharged at design-doc layer; (1b) skip-rich-MLP wiring not dispatched (gated on Rev 4 panel APPROVE); LEDGER write deferred to PI write at panel APPROVE.

---

## §0.9 Rev 4 → Rev 5 absorption (defense-panel REJECT — 3 BLOCKING + 5 SERIOUS + 4 PROBE + R30 candidate banking)

Defense-panel REJECTed Rev 4 with 3 BLOCKING (B1/B2/B3) + 5 SERIOUS (S1–S5) + 4 PROBE. PI ruled all defense paths and authored this Rev 5 absorption. Lead item B1 is load-bearing: it removes a test from the M0 gate basket.

| Item | Panel finding | Rev 5 PI ruling |
|---|---|---|
| **B1** | Rev 4 §2.1 (2b) integral metric `∫ Var_ratio dt over [0, 500] ≥ Var_ratio(0) × 500` PASSes peak-then-sink (γ) trajectories: a trajectory that peaks at t=250 and sinks back below baseline by t=500 still has positive integral area dominated by the transient peak. The integral metric was meant to *catch* exactly this failure class; as written, it does the opposite. Re-specs (time-above-baseline fraction, ground-truth-anchored integral) are post-hoc threshold engineering on a quantity without physical anchor. | **ABSORBED: (2b) integral RETIRED entirely from the M0 gate basket.** M0 gate basket reduced from {1a, 1b, 1c, 2a, 2b, 2c} to {1a, 1b, 1c, 2a, 2c}. The three retained tests (1b Spearman monotone trend, 1c Wilcoxon paired sign, 2c per-bin spatial structure) discharge the M0-dynamics question without a redundant aggregate-area test. S2 (trapezoid grid bias on the integral) automatically moots under this retirement. See §2.1 verbatim absorption note. |
| **B2** | Rev 4 §2.2 Bin-D (i) threshold `2 × 7.913e-09 = 1.583e-08` is below the frozen-init worst-seed Bin-D Var_ratio (seed 1 = 2.19e-8 per pre-flight B per-seed disclosure), i.e., the frozen-init model PASSes sub-clause (i) doing nothing. | **ABSORBED: ratchet 2× → 10×, threshold = `10 × 7.913e-09 = 7.913e-08`.** This sits ~3.6× above the seed-1 outlier (2.19e-8). Honest framing per PI: the 10× multiplier itself is K3-disease (smallest multiplier that clears the worst frozen-init seed; no physical anchor). Sub-clause (i) is therefore a sanity check; sub-clause (ii) (Bin-D log-MSE strictly-improving Wilcoxon) carries the mechanism load. See §2.2 amendment. |
| **B3** | Rev 4 §1.5 amendment 1 prose claimed pre-flight A's 50–54% dead-fraction provided "direct empirical support for the Lu+ 2019 dying-ReLU body-pathology mechanism as the operative cause." This overreaches: 50% dead-fraction at frozen init is the **symmetric-preact null** (E[ReLU dead frac] = 0.5 under any zero-mean preact through a Linear). The order-of-magnitude margin (~54% vs 5% threshold) is the null bar, not pathological-asymmetry evidence. | **ABSORBED via two-part fix:** (a) weaken §1.5 amendment 1 prose to "necessary-condition consistent with" (NOT "operative cause"); (b) commit pre-flight C (training-time dead-fraction trajectory under (γ) lr=1e-4) as Rev 5 precondition — measures whether dead-frac CHANGES under training, distinguishing the null from a true pathology signal. Pre-flight C executed before Rev 5 §1.5 finalized; verdict embedded inline in §1.5 amendment 1. PASS = ≥ 5pp asymmetric drift in ≥ 2/3 seeds; FAIL → amendment 1 retracts to "necessary-condition only; mechanism unconstrained" per [D-37]-ext rule 5 symmetric disclosure. |
| S1 | §2.1 (1c) Wilcoxon n=10 has W_crit=10 for α=0.05 one-sided; observations in [10, 15] would be MARGINAL but Rev 4 had no pre-committed re-bake rule. | ABSORBED: §2.1 (1c) Wilcoxon entry adds re-bake-to-n=20 stopping rule (seeds 11–20 from the same RNG schedule); final ruling on combined n=20. Re-bake decision pre-committed, NOT post-hoc. |
| S2 | Trapezoid integration on the 7-checkpoint grid {0, 50, 100, 175, 250, 350, 500} (non-uniform spacing) introduces a grid-bias to the integral metric. | **MOOTS under B1** retirement of (2b). Recorded for ledger discipline; no §2 edit required beyond removing the (2b) reference. |
| S3 | Rev 4 §2.1 success-criterion 11 PDF dump did not call out the void-vs-collapse pile-up ambiguity at the `1e-3` floor: a model that has collapsed to the void floor would also have low Bin-A density that hits the floor, indistinguishable from a model that has correctly fit a deep void. | ABSORBED: §2.1 success-criterion 11 adds mass-fraction report at `log10 ρ ∈ [-3.05, -2.95]`; if > 5% of Bin-A voxels pile at floor, Bin-A PDF interpretation flagged AMBIGUOUS. Does not change PASS/FAIL. |
| S4 | §7 dispatch chain step 3 (infra-manager sbatch) implied auto-dispatch from Stage 1a M0 PASS without an intervening panel-cycle gate. K6 retirement (Rev 4 success-criterion 18) implicitly removes the auto-dispatch path but §7 was not edited accordingly. | ABSORBED: §7 explicit edit — "Stage 1a PASS → infra dispatches Stage 1b ONLY after panel-cycle gate on Stage 1a artifacts; K6 retired removes auto-dispatch path." |
| S5 | §2.2 Bin-D (ii) Wilcoxon on per-seed log-MSE deltas had no min-samples-per-crop gate; Bin-D log-MSE is undefined for crops with zero Bin-D voxels and unstable with very few. | ABSORBED: §2.2 (ii) adds `min_samples_per_crop ≥ 5` gate; crops with < 5 Bin-D samples excluded from σ_within calculation. If > 20% of crops excluded, Bin-D test flagged INSUFFICIENT-COVERAGE. |
| PROBE 1–4 | Panel PROBEs on (a) the 5pp drift threshold sufficiency for pre-flight C, (b) whether n=20 re-bake suffices for marginal-band Wilcoxon power, (c) pre-flight D (healthy-MLP calibration) commit timing, (d) §1.5 (1c) SIREN as a backup architectural rung. | NOT BLOCKING. (a) 5pp threshold is operational definition of "emergent asymmetric drift" set 2026-05-25 PRIOR to running pre-flight C; (b) addressed by pre-committed re-bake rule per S1; (c) PI ruling: pre-flight D deferred to Stage-1-gate-2 if operative-cause framing is later re-asserted; (d) (1c) SIREN routing PROVISIONAL-pending-Stage-1a-gate-2 — if (1b) FAILs cleanly the (1d) hybrid voxel-grid rung is the panel-suggested next; (1c) carried as parallel option subject to fourth-REJECT-pivot trigger pre-committed: a fourth defense-panel REJECT on a (1b)-track design-doc revision triggers PI-mandatory pivot evaluation between (1c) and (1d). |
| **R30 candidate banked** | PI banked R30 candidate this turn (Rev 4 → Rev 5 cycle) after second occurrence of PI-brief grep miscount: Rev 3 K-B :81 line-cite miscount + Rev 4 `torch.cat` hit-count miscount. New procedural rule: briefs that load-bear on grep/glob/line-count must include the exact command + raw count, NOT paraphrased summaries. | DEFERRED-BANKED per §0.5 R30 row update. Promote to BANKED on third sighting OR on next panel-cycle pre-review APPROVE of a brief that follows the rule. This Rev 5 absorption demonstrates the discipline: every grep claim in Rev 5 prose cites the in-session command + raw count. |

**Pre-flight C — training-time dead-fraction trajectory under (γ) lr=1e-4** (`cloud_runs/d70_preflight_C_relu_trajectory/result.json` + `traj.png`):
3 seeds (0, 1, 2) × 500 training steps × 4 checkpoints {0, 100, 250, 500} × 8 ReLU sites; AdamW lr=1e-4 (matches d69 FAIL_SINKING cell); log10-MSE on P1 Sherwood overdensity with `+1e-3` floor; microbatch=1024, crops_per_step=4, crop_size=48 (canonical pipeline defaults). Pre-committed PASS threshold (set 2026-05-25 PRIOR to running the probe): asymmetric drift = `max_site |dead_frac(500) − 0.5|` ≥ 5 percentage points in ≥ 2/3 seeds.

- **`asymmetric_drift_per_seed` = [0.2078, 0.2004, 0.1722]** (seeds 0/1/2; all three exceed the 5pp threshold by ~3–4×)
- **`n_seeds_above_5pp` = 3/3**
- **`pre_flight_C_verdict` = "PASS"**
- Per-checkpoint losses: seed 0 = {0: 2.34, 100: 1.70, 250: 1.46, 500: 1.57}; seed 1 = {0: 2.56, 100: 1.64, 250: 1.56, 500: 1.80}; seed 2 = {0: 2.82, 100: 1.67, 250: 1.59, 500: 1.60}.
- **Honest disclosure (per-seed loss order-of-magnitude)**: step-0 losses (~2.3–2.8) are ~10× larger than the d69 `L_pre_step0 = 0.277` reference. Cause: this pre-flight uses a separate sampling generator (`torch.Generator(seed + 10_000)`) so the crop draws are distinct from d69's RNG schedule; log10-MSE is sensitive to which crops are drawn at step 0 (a single draw of a high-density crop can dominate the loss). The per-step trajectory is sane (losses fall to ~1.5–1.8 by step 500, same order of magnitude as d69 `L_pre_at_M1 = 0.21`). The verdict bar (asymmetric drift in dead-fraction) is measured on a SEPARATE forward pass with uniform [0,1]³ coords on a fresh torch RNG, so it is not coupled to the sampling-RNG offset. This is a sampling-RNG-driven loss-scale offset, not a forward-path bug; not a blocker.

**Verdict interpretation**: pre-flight C PASS confirms that under (γ) lr=1e-4 training the dead-fraction at body ReLU sites DRIFTS asymmetrically away from the 50% symmetric-preact null by ~17–21 percentage points within 500 steps. This is a dynamic signal distinct from the frozen-init null — pre-flight A's frozen-init 54% dead-fraction is consistent with the null, but pre-flight C's training-time drift away from the null is not. The combined pre-flight A + pre-flight C evidence supports §1.5 amendment 1's weakened "necessary-condition consistent with" framing (per B3 path a). It does NOT promote to "operative cause" — pre-flight D (healthy-MLP calibration: does a known-healthy MLP under the same supervision also exhibit ≥ 5pp dead-frac drift?) remains the discriminator and is deferred per PI ruling.

**Pre-committed FAIL framing (for the record, since the probe PASSed)**: had pre-flight C FAILed (n_seeds_above_5pp < 2), Rev 5 §1.5 amendment 1 would have retracted further to "necessary-condition only; mechanism unconstrained," and pre-flight A would have lost its operative-cause framing entirely. This was pre-committed 2026-05-25 PRIOR to running the probe per [D-37]-ext rule 5 symmetric-disclosure.

**(1b) skip-rich MLP fourth-REJECT-pivot trigger pre-committed** (PROBE 4 closure): a fourth defense-panel REJECT on a (1b)-track design-doc revision triggers PI-mandatory pivot evaluation between (1c) SIREN and (1d) hybrid voxel-grid as the next architectural rung, with [D-65 stub] further-class-pivot routing as the fallback. PI commits to NOT iterating to Rev 6 on (1b) absent panel APPROVE of a Rev 5-class document; this is anti-degeneracy discipline per Ext-2 R3.

**R30 candidate banking — in-session grep evidence audit** (per the new procedural rule):
- `grep -n 'torch.cat' src/models/nerf.py` → 4 hits at lines 22, 94, 114, 130 (raw `grep -c 'torch.cat' src/models/nerf.py` = 4). The single body skip-cat between layers1 (4 layers) and layers2 (4 layers) is line 130; lines 94 and 114 are pre-body input-stacking (encoded+g and skip_in+e_p respectively); line 22 is the PositionalEncoding helper.
- `nerf.py:75` is `self.relu = nn.ReLU()` — constructor, NOT an application site.
- ReLU application sites: `nerf.py:126` (`h = self.relu(layer(h))` inside `for layer in self.layers1:` loop, 4 sites by loop iteration) + `nerf.py:133` (same pattern inside `for layer in self.layers2:`, 4 sites) = 8 total application sites. Pre-flight A and pre-flight C both hook these 8 Linear modules to capture the pre-ReLU (= Linear output) values.

---

## §1 (1)-reframed architectural candidate slate

Four candidates scored. All four carry the [D-62] L96 "no IGM-NeRF precedent" tag. The Rev 3 placeholder name "ResMLP" is replaced repo-wide in this doc by **"skip-rich MLP"** per S-C absorption (success-criterion 13).

**(1a) Deeper/wider MLP** (e.g., 16-layer × 512-hidden vs current 8×256).
- *Anchor*: vanilla scaling (Tancik+ 2020); positional-encoding fields tolerate large depth.
- *Mechanism for breaking constant-collapse*: increased capacity may sample richer initializations; if attractor basin is capacity-bounded the basin shrinks relative to representable functions. **Hedge**: the (γ) FAIL_SINKING is an *active* shrink trajectory not a flat basin — more capacity does not obviously change the gradient sign that drives shrinkage.
- *Parameter cost*: ~4× params. Modest VRAM increase; well under [D-23] T2 11 GB ceiling.
- *R13 risk*: extrapolation from depth/width fixing constant-collapse is unsupported by §6 history; this is the weakest mechanism story of the four.

**(1b) Skip-rich MLP** (DeepSDF-style every-layer skip-input variant vs current single mid-skip).
- *Anchor*: DeepSDF (Park+ 2019), ResNet (He+ 2016). Skip-input paths bias toward identity-flow at init. **Mechanism-side anchors per Rev 2 K2 absorption**: Sitzmann et al. 2020 (SIREN) §3; Lu et al. 2019 ("Dying ReLU and Initialization"). **Rev 4 measurement anchor**: pre-flight A (`cloud_runs/d70_preflight_A_dying_relu/hist.json`) directly measured 50–54% dead-fraction at every body ReLU site under frozen init.
- *Mechanism for breaking constant-collapse*: per-layer skip-input means at init the network passes the positional-encoded coordinate signal directly into every body layer rather than relying on it surviving through a chain of half-dead ReLUs. The constant-output basin requires actively zeroing the skip-input contributions at every layer — gradient against doing so is non-trivial under direct ρ-MSE.
- *Dead-basin pathway disambiguation* (Rev 3 K2 Finding-2a closure, Rev 4 K-B line-cite correction):
  - (i) **Body activations are ReLU** at every body site. Constructor: `self.relu = nn.ReLU()` at `src/models/nerf.py:75`. Applications: `h = self.relu(layer(h))` inside the `for layer in self.layers1:` loop at `src/models/nerf.py:125-126` (4 sites) and inside the `for layer in self.layers2:` loop at `src/models/nerf.py:132-133` (4 sites). The constructor line `:75` alone does not prove the mechanism — only the 8 application sites in the forward path do; pre-flight A measured dead-fraction at all 8. **Dying-ReLU dead-zones (Lu et al. 2019)** are the body pathology.
  - (ii) Density-head output is Softplus (`src/models/nerf.py:137`, `density = self.softplus(out[..., 0])`) → the c≈0 **stable-basin (Sitzmann et al. 2020 SIREN §3)** is the head pathology: Softplus(0) = ln(2) ≈ 0.693 with vanishing gradient as pre-activation → −∞, giving a constant-mean basin under log-domain MSE with the `+1e-3` floor.
  - (iii) The (1b) per-layer skip-input hypothesis addresses (i) by re-injecting positionally-encoded input gradient into every body layer, bypassing the dead-ReLU region.
  - (iv) (1b) does NOT address (ii) directly — the head-basin escape is a downstream consequence of body-gradient revival, not a parallel intervention.
- *Skip-cat location in current architecture* (for grep contrast with the (1b) variant): the single skip-cat in the current `forward()` is at `src/models/nerf.py:130` (`h = torch.cat([h, skip_in], dim=-1)`), between `self.layers1` and `self.layers2`. `grep -n 'torch.cat' src/models/nerf.py` returns 4 hits total in the file; only line 130 is the body skip-cat (lines 94 and 114 are pre-body `g`-concat and `e_p`-concat respectively; line 22 is the positional encoding helper). The (1b) variant adds a skip-cat at the input of every body layer.
- *Parameter cost*: hidden_dim=256, encoded_dim=63, every-layer skip-input increases per-layer in_features by `skip_dim` (= encoded_dim + g_dim ≈ 64) at every body layer instead of just one. With 8 body layers and 256 hidden_dim, this is ≈ 25% growth in body-layer Linear parameter count over the current single-mid-skip architecture (per-layer in_features growth of ~64/256 = 25%, applied to 7 additional layers). See §1.6 per-layer spec table for the full breakdown.
- *R13 risk*: identity-bias-rescues-constant-collapse is a *hypothesis*; the head Softplus stable-basin remains an independent failure mode. Literature + pre-flight A support the *failure-mode existence* claim at the body; they do not prove the (1b)-rescue claim.

**(1c) SIREN (sinusoidal-activation) MLP** (Sitzmann+ 2020).
- *Anchor*: SIREN demonstrated high-frequency 3D-field fitting where ReLU MLPs fail; positional-encoding-substitute.
- *Mechanism for breaking constant-collapse*: sinusoidal activations have non-zero gradient everywhere (vs ReLU dead-zones) and bias toward oscillatory outputs at init. The constant-mean basin is *not* a stable attractor under Sine activations the way it is under ReLU + positional encoding — the SIREN init scheme deliberately escapes such basins.
- *Parameter cost*: same params as current; replaces ReLU with sin, replaces positional encoding with raw coords + careful init.
- *R13 risk*: SIREN's failure modes (training instability, hyperparameter sensitivity to omega_0) are distinct from but not obviously milder than constant-collapse.

**(1d) Hybrid implicit+explicit voxel-grid** (Instant-NGP-style hash-encoded grid + small MLP; Müller+ 2022).
- *Anchor*: Instant-NGP; multi-resolution hash grids are state-of-art on 3D field reconstruction.
- *Mechanism for breaking constant-collapse*: explicit voxel features cannot collapse to a single learned mean — each voxel has its own trainable feature vector. The constant-output basin requires all hash entries collapse to the same value, which has zero gradient sink under direct ρ-MSE if the supervision targets vary spatially.
- *Parameter cost*: ~10–100× params (hash table dominates); VRAM 4–8 GB for IGM cube; well under [D-23] T2.
- *R13 risk*: heaviest implementation commitment; pipeline integration is non-trivial; estimator-equivalence test must be re-derived. No IGM precedent.

### §1.5 [D-62]-collision resolution + PI PICK

**Collision**: [D-62] L94 binds "(1)-reframed must be supervision-target-coupled" (interpreted literally = 1∩2). Amendment-7 binds "constant-collapse must be addressed architecturally, NOT by another supervisory-regime swap." These BIND in opposite directions on the literal 1∩2 reading.

**Resolution**: I adopt the stricter reading of L94 — "supervision-target-coupled" = the architectural mechanism must be picked *with explicit reference to* how it changes the constant-collapse attractor structure *under the existing direct-ρ-MSE supervision*. This honors [D-62] L92's mechanism-disjointness concern without conflicting with amendment-7. Defense-panel re-review on this collision-resolution is the load-bearing next gate; if panel reads L94 as binding literal 1∩2, [D-70] de-activates and the project routes directly to [D-65 stub] further-class-pivot.

**PI PICK: (1b) Skip-rich MLP.** Single decision per `feedback-pi-decides`.

*Rationale* (lead with mechanism per [D-37] rule (a)):

1. The (γ) lr-sensitivity probe (3 cells × 1000 steps, lr ∈ {1e-4, 5e-4, 1e-3}, seed=0) shows 1/3 FAIL_SINKING and 2/3 UNKNOWN_NON_MONOTONIC peak-then-sink. All 3 cells fire R-b-pre1 at step=200. The (γ) regime claim is "uniform R-b-pre1 fire with mixed sub-classification across a full decade of lr." The constant-collapse-attractor mechanism story holds (R-b-pre1 = constant-density backstop). **(1b) is hypothesized to break this attractor** because per-layer skip-input paths at init pass the positional-encoded coordinate signal directly into every body layer.

2. **Pre-flight A + Pre-flight C inline-evidence paragraph (Rev 5 amendment 1, weakened from Rev 4 "operative cause" → "necessary-condition consistent with" per panel B3)**:

   > Amendment 1 (Rev 4 → Rev 5 panel B3 absorption): "operative cause" framing weakened to "necessary-condition consistent with." Pre-flight A (`cloud_runs/d70_preflight_A_dying_relu/hist.json`, 3 seeds × 4096 coords × 8 ReLU application sites, frozen-init IGMNeRF on P1) measured max dead-fraction = 0.5371 at body site 5 (`layers2[1]`), seed 1, against a 5% pre-committed threshold. 50% is the **symmetric-preact null expectation** for any zero-mean preact through a Linear (E[ReLU dead frac] = 0.5), so the order-of-magnitude margin (54% vs 5%) is the null bar, not pathological-asymmetry evidence. Pre-flight A is **necessary** (a healthy ReLU MLP should also exceed this threshold) but **not sufficient** (the threshold does not discriminate healthy from pathological).
   >
   > **Pre-flight C** (`cloud_runs/d70_preflight_C_relu_trajectory/result.json` + `traj.png`, 3 seeds × 500 training steps × 4 checkpoints × 8 sites, (γ) lr=1e-4 log10-MSE on P1 with `+1e-3` floor, canonical pipeline sampling) measures whether dead-frac CHANGES under training. Pre-committed PASS: asymmetric drift `max_site |dead_frac(500) − 0.5| ≥ 5pp` in ≥ 2/3 seeds (operational definition: emergent pathological asymmetry distinct from frozen-init null), threshold set 2026-05-25 PRIOR to running the probe.
   >
   > **Pre-flight C VERDICT: PASS.** `asymmetric_drift_per_seed = [0.2078, 0.2004, 0.1722]`, `n_seeds_above_5pp = 3/3` (all three seeds drift ~17–21 percentage points away from the 50% null within 500 steps, ~3–4× over threshold). Combined evidence: pre-flight A establishes the frozen-init null is exceeded as a necessary condition; pre-flight C establishes dynamic asymmetric drift under training — the dead-fraction does not stay at the 50% null under (γ) supervision but actively diverges. This supports the "necessary-condition consistent with" framing for the Lu+ 2019 dying-ReLU body-pathology mechanism. Lu+ 2019 cite retained.
   >
   > Pre-flight D (healthy-MLP calibration — does a known-healthy MLP under the same supervision also exhibit ≥ 5pp asymmetric drift?) deferred to Stage-1-gate-2 per PI ruling. If pre-flight D shows a healthy MLP also drifts ≥ 5pp, the framing retracts to "necessary-condition only; mechanism unconstrained" and pre-flight A + C lose their consistent-with-mechanism framing. This deferral is the discriminator gap acknowledged at Rev 5.

   Hedging discipline: "necessary-condition consistent with" not "operative cause"; "supports" not "proves"; mechanism scoped to frozen-init + (γ)-training IGMNeRF on P1, not generalized; pre-flight D deferral flagged as the open discriminator.

3. (1a) deeper/wider does not change the gradient *sign* of the attractor — mechanism story is weakest. (1c) SIREN may also break the attractor but introduces a confound (activation + init both change); attribution at a FAIL would be ambiguous. (1d) hybrid voxel-grid has the strongest mechanism story but is the heaviest implementation commitment with no IGM precedent and the largest R13 surface — if (1b) FAILs cleanly, (1d) is the natural next-rung.

4. (1b) minimizes confound: same input encoding, same output head, same loss, same data; ONLY the MLP body topology changes. A Stage 1a M0 PASS would clean-attribute the rescue to architecture-axis; a FAIL would clean-attribute to "skip-richness insufficient" (and route forward to (1d) or to [D-65 stub]).

5. **Honest hedge per [D-37] rule (a)** (Rev 4 S-A cosmology-cite correction): the (γ) falsification demonstrates direct ρ-MSE *alone* on the current architecture does NOT escape constant-collapse. (1b)-reframed must change the architecture such that direct ρ-MSE on the new architecture DOES escape. There is no IGM precedent that this works. The Lu+ 2019 / Sitzmann+ 2020 anchors are from photometric (SIREN: image regression, audio, video, SDF) and signed-distance-function domains, NOT 5-decade-log-density IGM cosmology — domain transfer is a *separate, unproven step*. The cosmological context for the 5-decade log-density dynamic range is in **Lukić et al. 2015 §3** (Lyman-α forest hydro simulation density-PDF analysis) and **Bolton et al. 2017** (Sherwood Simulation Suite density-PDF panels). The mechanism story is *plausible*; it is *not proven*.

6. **`+1e-3` log-domain floor origin** (success-criterion 15): the floor was introduced in `experiments/nerf/pipeline.py` at inception of the log-domain direct-ρ-MSE supervision; PI grep did not locate a formal ablation in the LEDGER or design-doc decision list discharging the value as 1e-3 specifically (search performed across `experiments/nerf/design/D*.md` for "1e-3" + "floor"). Treated as **operative since pipeline.py inception, no formal ablation**. **HELD CONSTANT across (γ) / (1b) / (1d)** for this Stage 1a gate; an ablation of the floor value is OOS for [D-70] and would be a separate decision.

### §1.6 Per-layer spec table (success-criterion 12, S-C closure)

Current IGMNeRF architecture (8-layer × 256-hidden, single mid-skip, ReLU body, Softplus density head, Sigmoid X_HI head, Tanh v_pec head):

| Layer | Variant: current (single mid-skip) | Variant: (1b) skip-rich MLP | Activation |
|---|---|---|---|
| input encoding | Fourier positional, L=10 → encoded_dim = 63; optional `g` (+1); optional `e_p` (+e_dim) | same | — |
| layers1[0] | in: encoded_dim [+1 g] [+e_dim], out: 256 | in: encoded_dim [+1 g] [+e_dim], out: 256 | ReLU |
| layers1[1] | in: 256, out: 256 | in: 256 + skip_dim, out: 256 | ReLU |
| layers1[2] | in: 256, out: 256 | in: 256 + skip_dim, out: 256 | ReLU |
| layers1[3] | in: 256, out: 256 | in: 256 + skip_dim, out: 256 | ReLU |
| skip-cat | `torch.cat([h, skip_in])` at nerf.py:130 → 256 + skip_dim | (no separate skip-cat; per-layer skip-input absorbed into every layer) | — |
| layers2[0] | in: 256 + skip_dim, out: 256 | in: 256 + skip_dim, out: 256 | ReLU |
| layers2[1] | in: 256, out: 256 | in: 256 + skip_dim, out: 256 | ReLU |
| layers2[2] | in: 256, out: 256 | in: 256 + skip_dim, out: 256 | ReLU |
| layers2[3] | in: 256, out: 256 | in: 256 + skip_dim, out: 256 | ReLU |
| out_layer | in: 256, out: 4 (ρ, T, X_HI, v_pec) | in: 256, out: 4 | — |
| density head | Softplus(out[...,0]) at nerf.py:137 | same | Softplus |

skip_dim = encoded_dim (+ 1 if g) ≈ 63–64. (1b) total param count: ~25% larger Linear-weight bytes vs current (per-layer in_features growth of skip_dim/256 ≈ 25% applied to the 7 layers that gain a skip-input). Hidden width and depth unchanged. Output head unchanged.

Skip-rich-MLP variant identity: **DeepSDF skip-input variant** — concatenate the encoded coords [+g] into the input of every body layer; body ReLU unchanged; head unchanged. Source flag binding: `--arch skip-rich-mlp` (not `--arch resnet-skip-rich`, not `--arch resmlp`).

---

## §2 Loss form + gate ladder

**Loss form** (UNCHANGED from (γ) per amendment-7 binding): direct ρ-MSE in log-domain,
$\mathcal{L}_{\rm pre}(\theta) = (1/N_v) \sum_v (\log_{10}(\rho_\theta(v) + 10^{-3}) - \log_{10}(\rho_{\rm truth}(v) + 10^{-3}))^2$.
The (1b)-reframed test is exactly: *"does the new architecture escape the constant-collapse attractor under the SAME supervision that just falsified (γ)?"* Supervisory-regime swap is forbidden per amendment-7.

**Standing disclosure clause (Rev 4 amendment 4)**: every M0/M1/M2 numerical report (current and future) MUST disclose σ_within per-bin AND aggregate alongside any point estimate of Var_ratio or log-MSE. No bare point estimates. This is non-negotiable per panel S-B/K-A absorption.

**Gate ladder** (M0 Rev 4 re-spec — Spearman ρ + permutation p + integral metric; M1/M2/M3 calibrated bands UNCHANGED from [D-69] Rev 5 with Rev 4 per-bin Bin-D amendments):

| Milestone | Step grid | Metric | PASS | MARGINAL | FAIL |
|---|---|---|---|---|---|
| **M0 — direction-of-motion gate** (Rev 4 re-spec) | {0, 50, 100, 175, 250, 350, 500} (6+1 checkpoints) | per-seed Spearman ρ between `Var_ratio(t)` and `t`; permutation p; integral `∫ Var_ratio dt over [0,500]`; aggregate via per-seed Wilcoxon signed-rank on `(Var_ratio(500) − Var_ratio(0))`; 10 seeds × 100 crops × 48³ | per-seed (1b)+(2b)+(3); aggregate Wilcoxon p ≤ 0.05 one-sided; ≥ 8/10 seeds per-seed PASS; `Corr(ρ_θ, ρ_truth) ≥ 0` at M2 endpoint | aggregate PASS but `max_t Var_ratio(t) > 1.2 × Var_ratio(500)` (peak-then-partial-recovery); or 6–7/10 seeds per-seed PASS | aggregate Wilcoxon p > 0.05; or < 6/10 seeds per-seed PASS |
| M1 — pretrain convergence | 1000 | $R_{\rm pre} = \mathcal{L}_{\rm pre}({\rm step\,1000}) / \mathcal{L}_{\rm pre}({\rm step\,0})$; **per-bin disclosure required, routing-lighter** (Rev 4 S-D Carlton-rule) | $\le 0.1$ | $0.1 < R \le 0.5$ | $> 0.5$ |
| M2 — pretrain saturation | 5000 | $R_{\rm sat} = \mathcal{L}_{\rm pre}({\rm step\,5000}) / \mathcal{L}_{\rm pre}({\rm step\,1000})$; **Bin-D mandatory sub-clauses (i)+(ii) AND-gated** (Rev 4 amendment 2) | $\le 0.5$ AND Bin-D (i) AND Bin-D (ii) | $0.5 < R \le 0.9$ AND both Bin-D pass | $> 0.9$; OR aggregate PASS but either Bin-D sub-clause FAILs → **FAIL on void-floor-saturation** |
| M3 — density-realism handoff (**DIAGNOSTIC ONLY, NOT gate-binding**) | 5000 | $R_{\rm real} = {\rm Var}[\log_{10}(\rho_\theta + 10^{-3})] / {\rm Var}[\log_{10}(\rho_{\rm truth} + 10^{-3})]$, 100 crops at 48³ | $\in [0.980, 1.021]$ — diagnostic | — | — |

### §2.1 M0 critical clause (Rev 4 re-spec — Spearman ρ + permutation + integral)

The Rev 3 ε_tol construction (`0.1 × σ_frozen / √N_crops = 7.28e-09`) is RETIRED per K-A panel REJECT. Pre-flight B confirmed σ_within_median = 5.83e-06 (~8× σ_frozen), confirming the panel diagnosis that the ε_tol formula mixed cross-seed-of-means SE with within-seed crop-to-crop scale. The Rev 4 re-spec uses non-parametric tests that sidestep SE construction entirely.

*Concrete spec*:

- **Per-seed measurement**: compute `Var_ratio(t) = Var(ρ_θ) / Var(ρ_truth)` at **t ∈ {0, 50, 100, 175, 250, 350, 500}** (6+1 = 7 checkpoints; success-criterion 2 + K-C absorption) on the same crop set used for the frozen-init baseline (100 random 48³ crops from `Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy`), repeated across 10 seeds.

- **Per-seed PASS condition** (M0 gate basket reduced from {1a, 1b, 1c, 2a, 2b, 2c} to {1a, 1b, 1c, 2a, 2c} per Rev 5 panel B1 — see §0.9 absorption block + Rev 5 verbatim retirement note below):

  > **M0 gate basket reduced from {1a, 1b, 1c, 2a, 2b, 2c} to {1a, 1b, 1c, 2a, 2c}. (2b) integral-threshold dropped per Rev 4 → Rev 5 panel B1 finding: as written, the integral PASSes on peak-then-sink (γ) trajectories (positive area dominated by transient peak), which is the failure class the gate was meant to catch. Re-specs (time-above-baseline fraction, ground-truth-anchored) are post-hoc threshold engineering on a quantity without physical anchor; the three retained tests (1b Spearman monotone trend, 1c Wilcoxon paired sign, 2c per-bin spatial structure) discharge the M0-dynamics question without a redundant aggregate-area test.** (S2 trapezoid grid bias on the integral automatically moots under this retirement.)

  - **(1b) Spearman ρ + permutation** (RETAINED, Rev 5 unchanged): H0 = monotonic-decline. PASS = `ρ(Var_ratio(t), t) ≥ 0` AND permutation p ≥ 0.05 (one-sided, n_permutations = 1000 minimum, against the H0 of monotone decline). NO σ-knob, NO √N, NO shrink factor.
  - **(1c) Wilcoxon signed-rank, per-seed paired sign** (RETAINED with Rev 5 S1 re-bake rule): per-seed `(Var_ratio(500) − Var_ratio(0))` Wilcoxon at α=0.05 one-sided; aggregated across seeds at the band layer below. **S1 re-bake stopping rule (Rev 5 pre-committed)**: Wilcoxon at n=10 has W_crit=10 for α=0.05 one-sided. If observed `W ∈ [10, 15]` (MARGINAL band), re-bake to n=20 with seeds 11–20 from the same RNG schedule; final ruling on combined n=20. Re-bake decision pre-committed, NOT post-hoc.
  - **(2c) Per-bin spatial structure** (RETAINED, Rev 5 unchanged): the M2 Bin-D mandatory sub-clauses (§2.2) bind at M2 endpoint; at M0 they provide the diagnostic-only per-bin Var_ratio readout to flag void-floor-collapse early. See §2.2 for the AND-gate construction (i+ii) and Rev 5 amendments per B2 + S5.
  - **(3) Co-diagnostic** (RETAINED, Rev 5 unchanged): `Corr(ρ_θ, ρ_truth) ≥ 0` at the M2 endpoint as a sign-check that the model has not learned an inverted field.

- **MARGINAL recategorization** (success-criterion 3): per-seed PASS but `max_t Var_ratio(t) > 1.2 × Var_ratio(500)` → seed-level verdict is MARGINAL (peak-then-partial-recovery profile; rebakes route to extended-n re-bake per band 6).

- **Aggregate PASS/MARGINAL/FAIL bands** (success-criterion 5 + S-B Wilcoxon, success-criterion 6):
  - **Primary aggregate test**: per-seed Wilcoxon signed-rank on `(Var_ratio(500) − Var_ratio(0))`, one-sided alternative `Var_ratio(500) > Var_ratio(0)`, α = 0.05 pre-committed. Aggregate Wilcoxon p ≤ 0.05 is a NECESSARY condition for PASS.
  - **Coarse band layer**: ≥ 8/10 seeds per-seed PASS → PASS band; 6–7/10 seeds per-seed PASS → MARGINAL band (routes to extended-n re-bake at 20 seeds × 100 crops); < 6/10 → FAIL band.

*Discriminating power* (regimes from the (γ) lr-sensitivity probe map cleanly onto FAIL under the Rev 5 reduced basket):
- FAIL_SINKING (e.g., (γ) lr=1e-4 monotone decay 1.63e-5 → 2.25e-6) → FAIL on (1b) (Spearman ρ < 0, permutation p < 0.05 against H0 of decline) AND FAIL on aggregate (1c) Wilcoxon (paired sign Var_ratio(500) − Var_ratio(0) < 0 with high consistency).
- UNKNOWN_NON_MONOTONIC peak-then-sink (e.g., (γ) lr=5e-4 peak 1.48e-5 @ step 250 then sink): per (1b) Spearman ρ ≈ 0 with permutation p ambiguous; the MARGINAL recategorization clause (`max_t Var_ratio(t) > 1.2 × Var_ratio(500)`) routes peak-then-partial-recovery profiles to the MARGINAL band; aggregate (1c) Wilcoxon on `(Var_ratio(500) − Var_ratio(0))` is the binding sign-test (peak-then-sink trajectories that return to baseline produce small-magnitude paired deltas straddling zero — Wilcoxon p > 0.05, FAILing the necessary aggregate condition). This is the case the retired (2b) integral metric was meant to catch but failed to (per B1).
- (1b) hypothesized PASS profile: monotone-increasing Var_ratio with positive Spearman ρ, aggregate Wilcoxon p ≤ 0.05 one-sided, and `Corr(ρ_θ, ρ_truth) ≥ 0` at M2.

*Output log-density PDF histogram dump* (success-criterion 11, P-C absorption + Rev 5 S3 floor-pile-up flag): at M0 eval, dump the histogram of `log10(ρ_θ + 1e-3)` over all sampled crops, on the same x-axis as `log10(ρ_truth + 1e-3)`. This is a void-floor-collapse discriminator: if the (1b) variant has collapsed to the void-floor (Bin-A dominance with empty Bin-D / Bin-C tails), the PDF will sit at the low-log-density peak with no high-density tail. Spec: 50 log-spaced bins from `log10(1e-3) = -3` to `log10(max(ρ_truth))`, dump as PNG + JSON to `experiments/nerf/artifacts/d70_stage1a_skip_rich_mlp/m0_eval_pdf.{png,json}`. **Rev 5 S3 addition**: report mass-fraction at `log10 ρ ∈ [-3.05, -2.95]` separately; if > 5% of Bin-A voxels pile at the `1e-3` floor, Bin-A PDF interpretation flagged AMBIGUOUS in the result. (A model that has collapsed to the void floor is indistinguishable in Bin-A from a model that has correctly fit a deep void without this disclosure.) Does not change M0 PASS/FAIL routing but flags void-vs-collapse ambiguity for panel inspection.

*Baseline JSON role*: `experiments/nerf/artifacts/d70_m0_baseline/baseline.json` is **diagnostic-only**, NOT gate-binding (re-labeling from Rev 3 stands). Its Rev 4 augmentation supplies the σ_within and per-bin Bin-D values that bind §0.8 / §2.1 / §2.2 sub-clauses.

### §2.2 M2 Bin-D mandatory sub-clauses (Rev 4 amendment 2 — S-D PROMOTED)

Replacing the Rev 3 "Bin-D as routing-lighter" wording, both sub-clauses are binding and AND-gated:

> **M2 Bin-D mandatory sub-clauses (both must PASS for M2 verdict = PASS):**
> (i) **Bin-D Var_ratio at M2 endpoint ≥ 10× frozen-init Bin-D baseline** (`10 × 7.913e-09 = 7.913e-08`) — Rev 5 panel B2 ratchet from Rev 4 2× = 1.583e-08, which the frozen-init worst seed (seed 1 = 2.19e-8) PASSed doing nothing. 10× = 7.913e-08 sits ~3.6× above the seed-1 outlier. **Honest framing per PI (B2 absorption)**: the 10× multiplier itself is K3-disease (smallest multiplier that clears the worst frozen-init seed; no physical anchor). Sub-clause (i) is a sanity check; sub-clause (ii) carries the mechanism load. Measured on the same 10-seed × 100-crop × 48³ protocol as pre-flight B, with σ_within disclosure per amendment 4.
> (ii) **Bin-D log-MSE at M2 endpoint strictly lower than at M2 step 0** by ≥ 1 σ_within of the Bin-D log-MSE across the 10-seed panel (per-seed log-MSE deltas, Wilcoxon signed-rank test at α=0.05 one-sided). **Rev 5 S5 min-samples gate**: `min_samples_per_crop ≥ 5` required for Bin-D log-MSE to be defined; crops with < 5 Bin-D samples excluded from σ_within calculation. If > 20% of crops excluded, Bin-D test flagged INSUFFICIENT-COVERAGE (does not auto-FAIL but routes M2 verdict to MARGINAL and surfaces the coverage issue to panel inspection).
>
> If aggregate Var_ratio PASSES but either Bin-D sub-clause FAILS, M2 verdict = **FAIL on void-floor-saturation**, not MARGINAL. The ~300× Bin-A-vs-Bin-D aggregate-mean dominance documented in pre-flight B (`experiments/nerf/artifacts/d70_m0_baseline/baseline.json` augmentation, per-bin aggregate: Bin-A = 2.31e-03, Bin-D = 7.91e-09) is the binding evidence that aggregate Var_ratio cannot stand alone.

M1 per-bin sub-clauses inherit the same structure but routing-lighter per prior PI memo S-D handling: M1 per-bin disclosure required, MARGINAL band on any per-bin failure but does not auto-FAIL aggregate.

*Pre-committed FAIL paths* (per [D-37]-ext rule 5 symmetric disclosure):
- (1b) failing M0 is reported as **falsification of the (1b) escape hypothesis** — the dead-ReLU-bottleneck mechanism story (§1.5 ¶i+iii) and skip-density rationale do not hold; no spin.
- (1b) PASSing M0+M1 aggregate but FAILing M2 Bin-D is reported as **void-floor-saturation failure** — the architecture has escaped constant-collapse globally but the void-floor remains a constant-mean basin; this is a partial-success that does NOT discharge the pattern claim (per success-criterion 18 K6 retirement).
- M0 FAIL triggers immediate FREEZE → escalate to next architectural candidate ((1d) hybrid voxel-grid is the panel-suggested rung; defense-panel re-review required per Ext-2 R3 anti-degeneracy before any (1c)/(1d) dispatch).

**M3 demotion clause (Rev 2, unchanged)**: M3 is **diagnostic-only, NOT gate-binding** for Stage 1a. The K3-bootstrap band [0.980, 1.021] was calibrated on (γ) target-data; recalibration is **deferred to gate-2** (post-M0 PASS).

**R-b-pre1/2/3 backstops**: UNCHANGED from [D-69] Rev 5 §2. At Rev 4 M0 is a Spearman/permutation/integral gate on the trajectory shape; R-b-pre1 (step-1000 absolute-threshold backstop) remains the downstream R-rule safety net.

### §2.5 Per-bin diagnostic + pre-committed MARGINAL thresholds

UNCHANGED from [D-69] Rev 5 §2.5 for the aggregate flatness check; **superseded for Bin-D specifically by §2.2 Bin-D mandatory sub-clauses (Rev 4 amendment 2)**. Pre-committed thresholds: (a) `max_i(log-MSE_bin_i) / median_i(log-MSE_bin_i) > 5.0` at M2; (b) `log-MSE(Bin D) > 5 × log-MSE(Bin B)` at M2. Either triggers Stage 1 MARGINAL (and stacks AND with §2.2 Bin-D AND-gate for the M2 verdict).

### §2.6 Optimizer config + estimator-equivalence re-cert spec

Inherit AdamW + warmup-cosine matching `experiments/nerf/pipeline.py:1022 / :71 / :287-300` (R30 re-verified this session per §0.5 audit; defense-panel re-cert required at gate-2 if [D-69] amendment-6 code-landing shifted line numbers, expected harmless drift only). lr_max=5e-4, lr_min=5e-6, warmup 1000 steps, cosine decay over max_steps=5000.

**Estimator-equivalence re-cert spec** (success-criterion 16, per [D-60] / [D-69] precedent): re-cert MANDATORY after the (1b) skip-rich-MLP wiring lands. The loss path is unchanged but the forward computation graph is new (per-layer skip-input concatenation changes the autograd DAG even though the supervision is identical). Test path placeholder: `tests/test_skip_rich_mlp_estimator_equivalence.py` — verifies forward + backward parity between (a) the skip-rich-MLP `forward()` and (b) an explicit reference implementation that materializes the per-layer skip-cats by hand, on a fixed seed, fixed coords batch. **Tolerance: 1e-5 relative** per [D-69] precedent. The re-cert is a precondition for the Juno dispatch, not for the design-doc panel APPROVE.

---

## §3 Data-locality audit

- **P1 z=0.3 n_grid=768 ρ-field cache**: VERIFIED LOCAL **this session (R26 in-session re-verification block per success-criterion 17)**:
  - Glob: `ls <repo-root>/Sherwood/.rho_field_cache/` returned `rho_field_p1_z0.300_n768.json` + `rho_field_p1_z0.300_n768.npy` (plus n=64 cache + 2 tmp shards).
  - `ls -la rho_field_p1_z0.300_n768.npy` → 1,811,939,456 bytes (≈ 1.81 GB), mtime 2026-05-13 21:48. Matches LEDGER size assertion within rounding.
  - R26 obligation discharged for this Rev 4 panel cycle; re-verify before any Juno dispatch.
- **P1 z=0.3 n_grid=64 ρ-field cache**: also LOCAL per same glob (32 MB tier), inherited.
- **P2/P3/P4 HDF5 trees ALREADY EXTRACTED** per [D-51c] amendment-6 correction (`SherwoodIGM_gal/extracted/planck1_60_768_ps13_z0.300/snapdir_012/`); P2 CIC deposit UNBLOCKED by amendment-7 OOM fix landing.
- **Stage 1a scope-lock**: **P1 only**. The architectural pivot is the load-bearing question; cross-physics testing inherits [D-69] Rev 5 routing as Stage 1b unconditional re-cal AFTER Stage 1a M0/M1/M2/M3 returns.

---

## §4 Compute-shape estimate

- **Stage 1a target**: Juno A30, P1 z=0.3, n_grid=768.
- **Step budget**: 5000 steps (M2 saturation gate); M0 at step 500 triggers FREEZE if FAIL.
- **(1b) architectural delta**: skip-rich MLP topology is ~25% larger Linear-weight bytes than current 8×256 (per §1.6 spec table); forward pass adds 7 elementwise concat ops per body forward. Expected per-step overhead: 5–15% vs (γ) baseline.
- **VRAM**: same order ~3–5 GB (MLP-only forward, no Voigt).
- **Wall-clock estimate**: **~30–35 min Juno A30** if (γ) Stage 1 budget held; +5–15% architectural overhead is in noise. CPU pre-flight estimator-equivalence ~5 min on n_grid=64 P1.

---

## §5 R-rule audit

- **R8**: HOLDS. (1)-reframed is one slot in [D-62] ladder; no completeness claim.
- **R13**: HOLDS. (1b) scope-locked to "skip-rich MLP architectural variant under direct-ρ-MSE supervision on P1 z=0.3 n_grid=768"; FAIL does NOT extrapolate to "all (1)-class architectures infeasible" — only to "this rung exhausted, escalate to (1d) per [D-65 stub] gate".
- **R15**: PROVISIONAL. Rev 4 itself marked **R15-PROVISIONAL** per success-criterion 20 until panel APPROVE.
- **R26**: SATISFIED this session for the P1 ρ-field cache (§3 R26 in-session re-verification block, glob + size check); FORWARD obligation for full re-verification immediately before infra-manager dispatch.
- **R27**: HOLDS. 5-stage ladder unchanged.
- **R28**: PROVISIONAL. Rev 4 itself marked **R28-PROVISIONAL** per success-criterion 20 until panel APPROVE.
- **R29**: HOLDS. Direct ρ-MSE loss is frame-matched to (γ) baseline; this is exactly the point (architectural test under identical supervision).
- **R30**: **DEFERRED-BANKED per Rev 5 PI banking 2026-05-25** (PI-brief grep-discipline rule banked this turn; pending one more operational test before BANKED promotion). Banking provenance: Rev 3 K-B :81 line-cite miscount + Rev 4 `torch.cat` hit-count miscount. The new procedural rule: briefs that load-bear on grep/glob/line-count must include the exact command + raw count, NOT paraphrased summaries. Promote to BANKED on third sighting OR on next panel-cycle pre-review APPROVE of a brief that follows the rule. Rev 5 §0.9 absorption block + §1.5 amendment 1 + §1 (1b) bullet all cite explicit grep commands + raw counts per the new discipline.

---

## §6 Defense-panel handoff — KILLER attack surface (Rev 4 update)

Panel should attack:

1. **§1.5 collision-resolution legitimacy.** Is the "stricter reading" of [D-62] L94 a genuine constructive reading or PI re-verbing to evade panel BINDING? Panel must independently verify the reading does NOT lower the BINDING bar.

2. **(1b) mechanism story strength + pre-flight A evidence transferability.** Pre-flight A (§1.5 amendment 1) directly measured 50–54% dead-fraction at every body ReLU site under frozen init. Panel: is "frozen-init dead-fraction" the right proxy for the *training-time* dying-ReLU mechanism? Lu+ 2019's argument is that dead units cannot recover under standard SGD; this is a *training-dynamics* claim, not a frozen-init claim. Pre-flight A is necessary but may not be sufficient. (Pre-flight C training-time dying-ReLU recovery measurement is parked OOS for post-APPROVE.)

3. **(1d) skip-over.** Hybrid voxel-grid has the strongest mechanism story per §1; PI picked (1b) on confound-minimization grounds. Is this the right tradeoff or is "pick the strongest mechanism first" the correct discipline?

4. **M0 Rev 4 re-spec (Spearman ρ + permutation + integral metric + Wilcoxon aggregate).** Panel should attack: (a) whether 7 checkpoints {0, 50, 100, 175, 250, 350, 500} is dense enough to catch sharper peaks than (γ) lr=1e-3's peak at t=250; (b) whether 1000 permutations is enough for the per-seed permutation-p estimator at the α=0.05 boundary; (c) whether the integral co-spec's threshold `Var_ratio(0) × 500` is the right reference (Var_ratio(0) is the t=0 measurement on the seeded init, not a population baseline); (d) whether the Wilcoxon n=10 aggregate has sufficient power against the smallest effect size of practical interest.

5. **M2 Bin-D mandatory sub-clauses (Rev 4 amendment 2).** Is `2 × 7.913e-09 = 1.583e-08` the right Bin-D Var_ratio target — i.e., is "2× frozen-init Bin-D baseline" a meaningfully-larger threshold than the variance of the Bin-D estimator itself (which the per-bin σ_within disclosure will quantify but only at M2 eval time)? Should the threshold be re-stated as `≥ N × σ_within(Bin-D)` instead?

6. **Pattern-claim DE-ACTIVATION (Rev 4 success-criterion 18).** K6 retirement: panel pre-commit binding from Rev 2/3 ("PASS → DISCHARGED & re-verbed; FAIL → PROMOTES → BANKED") is RETIRED. Panel: is the Rev 4 reasoning ("the Rev 3 M0 spec on which K6 was load-bearing is broken; discharge requires a fresh gate-2 panel cycle on the Rev 4 spec") sufficient to retire the pre-commit, or does the pre-commit carry forward to the Rev 4 spec automatically?

7. **R30 retraction (Rev 4 success-criterion 19).** Panel P-B 2026-05-25 ruled same-cycle re-grep does not constitute n=2. Is the Rev 4 retraction the right ledger-discipline outcome, or does the substantive correctness of the re-grep (verbatim line content matching inheritance) merit a softer disposition?

---

## §7 Next dispatch (single, per `feedback-pi-decides`)

**DISPATCH: `defense-panel` adversarial pre-review on this Revision 5 doc.** Same gating discipline as the Rev 1 → Rev 4 cycle. Returns APPROVE / NEEDS-WORK / REJECT; on REJECT the B1 (2b) retirement + B2 Bin-D 10× ratchet + B3 §1.5 weakening + pre-flight C result are the load-bearing items. Fourth-REJECT-pivot trigger pre-committed per §0.9 PROBE 4 closure.

**Next-Dispatch chain** (CONDITIONAL on panel APPROVE on Rev 5):
0. **Pre-flight C executes before Rev 5 §1.5 amendment 1 finalizes** — DONE this turn (PASS, see §0.9 + §1.5 inline).
1. defense-panel re-review on Rev 5 ← CURRENT
2. core-implementer: (1b) skip-rich-MLP wiring in `src/models/nerf.py` behind `--arch skip-rich-mlp` flag; estimator-equivalence test at `tests/test_skip_rich_mlp_estimator_equivalence.py` (1e-5 rel tol); CPU pre-flight P1 n_grid=64
3. infrastructure-manager: sbatch authoring with binding 4-item pre-validation checklist per `feedback-infra-sbatch-pre-validation`
4. Juno Stage 1a run on P1 → M0 (Spearman ρ + per-seed Wilcoxon + per-bin spatial-structure + output-PDF dump with floor-pile-up flag) / M1 (per-bin disclosed) / M2 (Bin-D AND-gated at 10× threshold with S5 min-samples gate) / M3 (diagnostic)
5. **Stage 1a PASS → infra dispatches Stage 1b ONLY after panel-cycle gate on Stage 1a artifacts** (Rev 5 S4 absorption: K6 retired removes auto-dispatch path; the panel-cycle gate is mandatory, not implicit).
6. (Conditional, post panel-cycle gate on Stage 1a) Stage 1b P2 re-cal per [D-69] amendment-5 routing
7. OOS follow-ups (pre-flight D healthy-MLP calibration baseline if §1.5 operative-cause framing is later re-asserted; LEDGER §3 rename writeback to "skip-rich MLP")

---

## §Pattern-claim status (Ext-2 R9 provisional-pending-third-regime-test) — Rev 4 update

The amendment-7 pattern claim — "vanilla MLP IGM-NeRF fails constant-collapse-escape under flux-domain (3-fGPA) AND direct ρ-MSE (γ) supervision" — remains provisional-pending-third-regime-test. **Rev 4 K6 DE-ACTIVATION (success-criterion 18)**: the prior PI pre-commit ("(1b) PASS → DISCHARGED & re-verbed; (1b) FAIL → PROMOTES → BANKED") is **RETIRED**. Rationale: that pre-commit was load-bearing on the Rev 3 M0 spec, which the panel REJECTed for ε_tol category-mix (K-A) and other blockers; the same pre-commit cannot be transplanted onto a fundamentally re-built Rev 4 M0 spec without a fresh panel cycle. **(1b) M0 PASS under Rev 4 does NOT discharge the pattern claim** at the design-doc layer — discharge requires gate-2 panel cycle on the revised spec.

The (1b)-reframed Stage 1a third-regime-test framing carries forward (K5 admissibility unchanged), but its discharge semantics are now panel-cycle-gated, not pre-committed.

---

## §[D-53] supervision-target axis interaction

[D-53] is NOT discharged by amendment-7. (γ) was a direct-ρ-MSE supervision test, which is one axis of the [D-53] supervision-target candidate space; [D-53] also contemplates flux-domain alternative supervision targets (e.g., normalized log-flux, k-binned residuals). (1b)-reframed Stage 1a does NOT test these. The [D-53] upstream-untested obligation carries forward to whichever ladder rung activates after [D-70] returns.

---

## Rev 5.1 amendment block (panel S1-S5 + P1/P3 absorption + R31 candidate)
Status: Revision 5.1 PROVISIONAL pending R26 in-session re-verification by PI; Juno-dispatch authorized on amendment-block landing (panel pre-committed no fifth cycle).
Date: 2026-05-25

### S1 amendment — §1.5 amendment 1 "necessary∧necessary not sufficient" tightening
Amends §1.5 amendment 1 (the "necessary-condition consistent with" prose on the dead-ReLU body-pathology hypothesis). PI verbatim:
> Pre-flight A (init-scale gradient flow) and pre-flight C (ReLU-death trajectory) are each necessary-condition tests for the dead-ReLU body-pathology hypothesis; their conjunction remains necessary, not sufficient. Sufficiency requires pre-flight D (operative-cause swap-back: (1b)→ReLU at matched seed), which is hereby committed as a Stage 1a M0-PASS-conditional precondition (see §7 step 4b) — not deferred to gate-2.

### S2 amendment — §2.2 Bin-D gate restructure (drop (i), let (ii) carry)
Amends §2.2 (Bin-D constant-floor sanity + Wilcoxon sign-rank composite gate). PI replacement:
> Gate-1 PASS criterion: per-seed Wilcoxon signed-rank on Bin-D Var_ratio(500)−Var_ratio(0), n=10 seeds, one-sided α=0.05 in the IMPROVE direction. The prior (i) constant-floor 10× Var_ratio sanity check is retired (K3-disease residue per panel S2); (ii) carries discriminating load. Void-floor-collapse diagnostic logged as a §0.9 observation flag, not a gate.

### S3 amendment — §2.1 (1c) Wilcoxon re-bake Bonferroni adjustment
Amends §2.1 (1c) Wilcoxon re-bake clause. PI replacement:
> If n=10 stage-1 Wilcoxon returns MARGINAL (0.025 < p ≤ 0.05), re-bake to n=20 with final-stage α=0.025 (Bonferroni adjustment for two-stage adaptive design; family-wise α ≈ 0.05 under Lehmacher & Wassmer 1999, Brannath et al. 2002). Family-wise α disclosed in §0.9 row.

### S4 amendment — §0.9 σ_redraw control debt footnote
Appended to §0.9 (assumption-flag log) as new footnote. PI verbatim:
> Pre-flight C measurement-coord RNG: σ_redraw control owed at Stage 1a artifact write-up — 100 fresh-draw measurements on fixed pre-flight-C-seed-0 checkpoint, report std vs observed 17–21pp inter-checkpoint drift. If σ_redraw / drift > 0.2, re-run pre-flight C with fixed measurement-coord set. Origin: `scripts/d70_preflight_C_relu_trajectory.py:139` uses global torch RNG for measurement coords (not fresh independent generator); confound likely small by LLN but unmeasured.

### S5 amendment — §0.9 R30 promotion-rule clarification
Appended to §0.9. PI verbatim:
> R30 promotion rule clarification: 'next panel-cycle pre-review APPROVE' includes NEEDS-WORK verdicts where grep-discipline / pre-flight-evidence-discipline is verified operationally intact in the panel transcript. Panel ruling 2026-05-25 Rev 5 transcript is the qualifying instance.

### P1 acknowledgement — Spearman ρ n=7 low-power note
> Spearman ρ at n=7 checkpoints has critical ρ_0.05 one-sided ≈ 0.64; per-seed Wilcoxon (gate-1 criterion) is the actual sign discriminator and is not affected.

### P3 acknowledgement — §7 step 5 post-Stage-1a panel-cycle-gate sub-spec
§7 step 5 sub-spec verbatim:
> APPROVE → Stage 1b dispatch; NEEDS-WORK → Stage 1a artifact augmentation loop; REJECT → routing per [D-65 stub] (out-of-scope this Rev).

### R31 candidate — cross-section noun-consistency audit
> **R31 candidate (DEFERRED-BANKED, this turn first sighting)**: Cross-section architectural-noun consistency audit before PI memo issuance. Banking provenance: PI absorption memo this turn (2026-05-25) §8a/§8b named "SIREN" while §6 ruling deferred SIREN and §1.6 spec table named "skip-rich MLP"; brief-reviewer caught at single dissenting locus vs five-locus consensus. Mechanically checkable (set-equality on cross-section architectural-noun sets). Promotion to BANKED requires one more cross-track sighting per R12 precedent. Logged for next §3 LEDGER entry.

### R15 lift confirmation
> R15 PROVISIONAL sign-off (PI ruling §6 + dispatch corrections) lifted by brief-reviewer (R26 in-session re-verification): §1.6 line 208 confirms canonical flag `--arch skip-rich-mlp` (NOT `--body_arch skip-rich-mlp` as PI's correction memo paraphrased — minor R31-class sub-discrepancy logged; corrected flag spelling propagates to dispatch chain §8a/§8b).

### Conditional contingency — (1c) SIREN re-rank for M0 PASS / M2 FAIL path
> Conditional contingency: if Stage 1a returns M0 PASS / M2 FAIL, (1c) SIREN re-ranks above (1d) hybrid-grid for void-floor-collapse remediation candidates; rule deferred to result.

### Juno-dispatch greenlight gate
> Per panel pre-commitment + PI ruling §7: greenlight requires (a) this amendment block committed verbatim; (b) LEDGER §3 R30 entry status flipped to BANKED with panel-transcript citation; (c) LEDGER §3 [D-70] block updated with "Rev 5.1 absorbed, Juno-dispatch authorized" sub-bullet; (d) PI in-session R26 re-read of this amendment block. No fifth panel cycle owed.

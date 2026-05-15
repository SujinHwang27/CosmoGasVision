# Sprint-5 (c′) at 48³ — substrate-scale extension probe — design doc

**Status**: DRAFT v3 — PI-authored design doc; gate-2 RE-pre-review APPROVED-WITH-AMENDMENT; v3 absorbs 7 amendments inline; sign-off NON-PROVISIONAL per [D-37]-Ext 2 R15 clause (b) — the gate-2 APPROVE on rescue path discharges PROVISIONAL.

**Predecessor decisions**: [D-43] (CVPR cycle CLOSED 2026-05-14), [D-49] (held-out region), [D-51] (sprint-4 truth-baseline branch-iv outcome), [D-52] (option-(c) scope-lock + amendments 6/7), [D-53] (supervision-target upstream-untested), [D-54] (sprint-5 source-choice resolution; (c′) ranked #3 post-CVPR follow-on; R16/R17/R18 BANKED), [D-55] (v1 → v2 → v3 verdict-absorption arc; R19/R20/R21/R22 BANK).

**Governance posture**: This document authors the design only. Execution authorization at v3 is gated on (i) MDE re-derivation with between-seed variance term below 0.10 (LOAD-BEARING per panel K1; satisfied at k=5, MDE 0.055), (ii) defense-panel gate-2 RE-pre-review APPROVE-WITH-AMENDMENT on v2 (received 2026-05-15), (iii) PI gate-3 decision-of-record absorbing gate-2 verdict and applying the 7 amendments inline (this v3). The [D-43] CVPR submission cycle remains CLOSED; (c′) is post-CVPR work and does not amend the submitted CVPR PDF.

**v1 → v3 absorption changelog** (combined):

*v1 → v2 (gate-2 v1 NEEDS-WORK absorption, 4 KILLERs + 6 SERIOUS + 2 R-rule candidates)*:
- §3 R10 item 1 rewritten with correct stride-2-stage count (4) and corrected token-count math (2³ at 32³ vs 3³ at 48³; ratio 3.375×); v1's "5 stages" claim retracted.
- §3 R10 item 2 Tan & Le primary-cite replaced with Touvron+2019 FixRes; Tan & Le retained ONLY as "null test under compound-scaling suboptimality" — honest cite-direction.
- §3 R17 expanded with R19 (BANK) pitch-preserving vs resolution-preserving distinction.
- §4.2 MDE re-derived with R20 (BANK) between-seed variance decomposition; Var_total = Var_within + Var_between/k.
- §2 hypothesis verb narrowed to single-point per K4 option-(b).
- §5.4 multi-seed protocol pre-commits to k=5; variance-derived dispersion trigger.
- §6 gate-(c) cuDNN-determinism pre-flight banked as gate-4 prerequisite.
- §8.5 AD-5 expanded to [mean, var, skew, kurtosis] at 48³ per-voxel-information-ratio preservation.
- §4.3 ρ-sensitivity empirical anchor obligation banked.

*v2 → v3 (gate-2 RE-pre-review APPROVE-WITH-AMENDMENT, 3 new KILLERs + 4 SERIOUS + R21/R22)*:
- **A1**: §4.2 σ_seed=0.05 anchor re-verbed with explicit domain-transfer acknowledgment (NeRF flux-statistic → discriminative CNN classifier accuracy is an analog-domain transfer, not a same-domain estimate); BANK obligation for empirical σ_seed recovery from first 2 (c′) seeds; pre-commit to 7-seed escalation if σ_seed_emp > 0.07; footnote disclosure obligatory in any branch-(iii) paper-text.
- **A2**: §4.2 ρ_seed = 0.3 re-verbed from "conservative estimate" to "fiducial; conservative-toward-larger-MDE direction is ρ_seed → 0 (MDE = 0.065 at ρ_seed = 0, still clears 0.10)." BANK ρ_seed empirical recovery alongside ρ_emp in §4.3.
- **A3 (load-bearing)**: §3 R10 item 1 — "expressivity-on-substrate argument is *stronger* under corrected arithmetic" REMOVED. `AdaptiveAvgPool3d(1)` at `cnn3d.py:135` collapses both 8-token (32³) and 27-token (48³) spatial grids to identical 256-dim readout vectors; token-count ratio is NULLIFIED at readout. R10 admissibility now rests load-bearingly on items 3 (orthogonal substrate axis) and 4 (fresh init), NOT on item 1's token-count ratio. Item 1 retained only as "the corrected token ratio is a modest, not decisive, expressivity factor at 48³; the global-pool architecture forecloses token-count-leveraging at the readout." Honest absorption per [D-37] rule (a); this is the second R10 item-1 failure on consecutive rounds (v1 arithmetic; v2 readout-topology).
- **A4**: §6 gate-(e) — pre-flight bullet added: compute `[mean, var, skew, kurtosis]` 4-scalar baseline accuracy on sprint-4 32³ test set BEFORE (c′) dispatch. If MVSK-at-32³ ≥ 0.42, the 10 pp threshold has been silently tightened (the [mean, var] sprint-4 baseline was 0.368; adding skew/kurt may raise the 32³ bar already); disclose in §6 footnote and branch-iv interpretation.
- **A5**: §2 branch-iv row re-verbed — "LEDGER §7 entry MANDATORY; paper-text propagation ELIGIBLE under single-point register-compliant null framing in non-CVPR venues, subject to R16 cross-atom audit; multi-point scoping curve REQUIRED only for 'extends' / 'characterizes-the-dependence' verbs." §8.6 question 7 reworded to match. v2's "paper-text propagation deferred to multi-point scoping curve" framing was asymmetric-disclosure / publication-bias-by-design.
- **A6**: §3 R19 binding text expanded — "AND state which axis is load-bearing for the design's hypothesis verb; if both axes could test the hypothesis, declare why one was chosen."
- **A7**: §2 scope clause (vi) — "z=0.3 is the weakest redshift for inter-physics discriminability" softened to "any single-redshift test inherits the redshift-conditional generalization unknown" (v1 panel-statement provenance not localizable to a single [COSMO-S2] line at the gate-2 RE-pre-review record).
- §3 R21 + R22 BANK entries added with case-of-record + binding text.
- Status header advanced to NON-PROVISIONAL per R15 clause (b).

---

## §1 Context & rationale

The 2026-05-14 sprint-4 30-epoch Juno H100 run ([D-51], run_id `sprint4_1778774878`) returned `Â_overall = 0.379` [block CI 0.362, 0.396] against a `[mean, var]` 2-scalar baseline at 0.368 — margin 1.15 pp, ≪ 10 pp AD-5 floor → gate-(e) FAIL. Gate-(a) sanity floor also FAILed. Branch-iv routing per [D-52] amendment 7 fired; the substantive substrate-scoping finding landed as observation-not-headline in `papers/shared/sec/3_experiments_main.tex` per [D-54] R16.

(c′) tests *at 48³ ρ-crop substrate, whether the 4-way Sherwood-recipe signature is or is not discriminable beyond [mean, var, skew, kurtosis] low-order moments by an architecturally-identical 3D ResNet-18 probe.* Single-point (K4 option-(b)); pitch-preserving (R19); architecture-controlled (R10 admissibility per items 3 + 4 — see §3 v3 amendment).

**Explicit non-discharge of [D-53]**: (c′) does NOT discharge the supervision-target upstream-untested obligation.

**Pillar-aligned authoring**: design doc under `experiments/nerf/design/`; paper-text propagation under any outcome routes through R16 cross-atom audit; branch-iv at 48³ paper-text disposition per §2 v3 A5 amendment below.

---

## §2 Falsifiable hypothesis + pre-committed outcome routing

**Hypothesis** (R9; R13; K4 option-(b) single-point):

*We test whether, at the single substrate-edge value 48³, an architecturally-identical 3D ResNet-18 probe-classifier discriminates the 4-way Sherwood-recipe signature beyond a `[mean, var, skew, kurtosis]` 4-scalar baseline by the AD-5 ≥ 10 pp margin, on ρ crops extracted from a single Sherwood snapshot (z=0.3, n_grid=768) on the fixed [D-49] 70/15/15 axis-0 held-out region.*

*Scope statement (R9 obligatory): (i) NO claim about substrate scales other than 48³; (ii) NO substrate-scoping "extends" claim; (iii) NO claim about non-ρ observables; (iv) NO claim about non-ResNet-18 architectures; (v) NO claim that 48³ "recovers" 1D flux-statistic-scale signal of Bolton+2017; **(vi) NO claim about redshifts other than z=0.3 — any single-redshift test inherits the redshift-conditional generalization unknown** (A7 v3 amendment; this is a known limitation banked as a multi-redshift follow-on).*

**4-branch outcome routing** (inherited from [D-52] amendment 7 + [D-54]; v3 A5 amendment to branch-iv disposition):

| Branch | Trigger | Disposition |
|:---|:---|:---|
| (i) | AD-1 / gate-(c) determinism / gate-1 sanity / training-divergence FAIL | **PROCESS-FAILURE**. No publication. §7 LEDGER entry; debug; re-dispatch if root cause identified. |
| (ii) | gate-(b) sparsity OR gate-(d) wild-oscillation FAIL without leakage signature | Rerun with adjusted parameters. NOT branch-iv. |
| (iii) | 5 gates PASS + AD-5 PASS (Â_overall − `[mean, var, skew, kurtosis]` ≥ 10 pp, per-seed AND seed-averaged) | Per [D-52] amendment 7 sub-table: (iii-a) above-bar, (iii-b) indistinguishable-from-bar, (iii-c) below-bar with AD-5 PASS. Paper-text eligible; R11 register check; only single-point "at 48³ the margin is X pp" verb admitted (no "extends"). |
| (iv) | gate-(a) sanity floor FAIL OR AD-5 FAIL (margin < 10 pp) | **Ceiling-disqualified; substantive null result at 48³.** **LEDGER §7 entry MANDATORY; paper-text propagation ELIGIBLE under single-point register-compliant null framing in non-CVPR venues, subject to R16 cross-atom audit; multi-point scoping curve REQUIRED only for "extends" / "characterizes-the-dependence" verbs** (A5 v3 amendment). R14-(ii) symmetric-disclosure performed as designed; under the v3 disposition, branch-iv is publishable as a single-point null in a non-CVPR venue, not asymmetrically suppressed pending a curve. |

**Pre-commit attestation**: all 4 branches are publication-routed BEFORE the experiment runs. Branch-iv is a valid scientific outcome; this design doc is NOT optimized to wring a positive number out of any branch.

---

## §3 R-rule audit per [D-37]-Extension 2 (R8–R22)

**R8 cascade-close formality**: (c′) does NOT close any cascade. The 4-counterfactual saturation-band cascade (D1/D2/D3/D4 per [D-46] Addendum 2) is closed *within the [D-24] supervision regime*, with the supervision regime itself untested per [D-53]. (c′) operates on the probe-classifier substrate-scale axis; no cascade open/close.

**R9 invariance-verb discipline**: hypothesis text uses "we test whether" / "the 4-way signature is or is not discriminable beyond" — no "invariance" / "physics-invariant" / "structurally immune" / "extends". Scope statement obligatory (6 clauses, vi softened per A7).

**R10 retired-model-reuse contract — orthogonality argument** (LOAD-BEARING per panel K2; v3 A3 amendment to item 1):

The sprint-4 3D ResNet-18 (12M nominal; 8.3M canonical per `headline.json` `model_params=8298916`) was retired-for-reason-X = "fails gate-(a) sanity floor + gate-(e) AD-5 at 32³ ρ-crop substrate." (c′) reuses the same architecture at 48³.

Default presumption: NOT admissible. Argument for admissibility:

1. **The retired axis is the substrate, not the architecture (token-count framing nullified at readout; A3 v3 amendment).** Reason-X is substrate-bound. The 3D ResNet-18 has 4 stride-2 stages (verified directly against `src/models/cnn3d.py` lines 120–134; total stride 16). At 32³ → 2³ = 8 spatial tokens entering `AdaptiveAvgPool3d(1)`; at 48³ → 3³ = 27 spatial tokens. The token ratio is 3.375×.

   **HOWEVER**, the global-pool readout (`cnn3d.py:135`) collapses both 8-token and 27-token spatial grids to identical 256-dim feature vectors. The token-count ratio is **nullified at the readout**. The corrected token ratio is a **modest, NOT decisive, expressivity factor** at 48³; the global-pool architecture forecloses token-count-leveraging at the readout. v2's "the R10 expressivity-on-substrate argument is *stronger* under the corrected arithmetic" claim is RETRACTED — that was a second consecutive R10 item-1 fabrication (v1 arithmetic wrong; v2 readout-topology argument unsupported). Honest absorption per [D-37] rule (a). R22 (NEW, BANK) is the case-of-record (see below).

   **R10 admissibility rests load-bearingly on items 3 (orthogonal substrate axis) and 4 (fresh init), NOT on item 1's token-count ratio.** Item 1 contributes only a modest spatial-information budget delta inside the per-token feature pipeline (pre-global-pool), which does not flow to the readout.

2. **Precedent — substrate-scale coupling (Touvron+2019 FixRes primary; Tan & Le retained with corrected cite-direction).** Touvron et al. 2019 (FixRes; "Fixing the train-test resolution discrepancy", NeurIPS) establishes the fixed-architecture / varying-input-resolution regime as a well-defined experimental design space: the same trained architecture evaluated at different input resolutions exhibits measurably different task-readout behavior. Liu et al. 2022 (ConvNeXt, §4) establishes that input resolution at fixed architecture changes the receptive-field-to-content coupling materially.

   Tan & Le 2019 (EfficientNet, §3 compound scaling) is retained with corrected cite-direction: their thesis is that depth, width, and resolution should be **co-varied**; fixed-architecture resolution variation is provably sub-optimal under their formulation. (c′) is therefore explicitly a **null test under that suboptimality**: if Tan & Le's compound-scaling principle is the dominant effect, we should observe NO improvement at 48³ versus 32³ at fixed (depth, width). v1's "Tan & Le as positive precedent" framing was a [D-37] rule (a) cite-direction violation and is retracted.

3. **Empirical-level precedent**: [D-54] gate-4a panel + gate-5 PI ruling both accepted that sprint-4 retirement is substrate-bound. (c′) at 48³ left OPEN as a defensible measurement.

4. **Fresh initialization**: weights re-initialized; checkpoint `resnet18_3d_4class_best.pt` NOT loaded. Architecture *specification* reused, not trained model.

**Verdict (PI sign-off, NON-PROVISIONAL per R15 clause (b))**: R10 orthogonality argument is sufficient under v3 corrections to items 1–4, with item 1 explicitly demoted from load-bearing to modest-contributor at the readout.

**R13 scope-lock — deliverable surface declaration** (K4 + S5 absorption): (c′) deliverable is *not* a headline-claim accuracy number, *not* a substrate-scoping curve, *not* an "extends" claim. It is one of: (iii) "first single-point measurement at 48³ exceeds AD-5 floor over [mean, var, skew, kurtosis] by X pp" (paper-text eligible BUT no "extends" verb), OR (iv) "single-point measurement at 48³ fails AD-5 floor" (LEDGER mandatory + paper-text eligible in non-CVPR venues per A5).

**R14 self-anchored-bar + R18 nesting** (clarification per [D-54]):
- The [D-15] 85% bar is project-internal per [D-36] (no external observational anchor).
- The AD-5 ≥ 10 pp margin is a nested project-internal bar (no external anchor per [D-52] amendment 6).
- Pre-committed outcome routing per [D-52] amendment 7 satisfies R14-(ii).
- R18 nesting disclosure obligation: any (iii) paper-text propagation must footnote that AD-5 is project-internal at the 48³ nesting site AND that the baseline was expanded from [mean, var] (sprint-4) to [mean, var, skew, kurtosis] (sprint-5 c′) per S3 absorption, so the AD-5 measure is not directly comparable across sprints.

**R15 PROVISIONAL by default → NON-PROVISIONAL at v3**: PI sign-off on v3 is **NON-PROVISIONAL** per clause (b) — the gate-2 RE-pre-review APPROVE-WITH-AMENDMENT verdict on the rescue path (the 7 v3 amendments) discharges PROVISIONAL. The panel explicitly stated no third panel round is required IF gate-3 absorbs the 7 amendments inline; gate-3 has done so in this v3.

**R16 cross-atom propagation pre-commit**: IF (c′) outcome ships as paper-text under branch-iii OR under branch-iv in a non-CVPR venue, propagation to §0 abstract + §1 intro + §3 experiments + §4 next-steps atoms MUST follow [D-54] R16 BANKED protocol (PI re-review of latex-author cross-atom audit before paper-text close).

**R17 axis-relabeling guard + R19 pitch-vs-resolution distinction** (K3 + R19-BANK absorption; A6 v3 amendment):

- (c′) operates on the **probe-classifier substrate-scale axis** (crop edge 32³ → 48³ at fixed voxel pitch).
- **R19 pitch-preserving vs resolution-preserving declaration**: (c′) varies crop edge at FIXED n_grid=768, FIXED comoving box 60000 kpc/h, therefore FIXED voxel pitch 78.125 kpc/h. Spatial extent grows 2500 → 3750 kpc/h. Pitch-preserving / extent-varying, NOT resolution-preserving.
- **R19 binding (A6 v3 amendment)**: future substrate-axis design docs must declare pitch-preserving vs resolution-preserving AND state which axis is load-bearing for the design's hypothesis verb; if both axes could test the hypothesis, declare why one was chosen. For (c′): the pitch-preserving axis is load-bearing because the architecture's expressivity-vs-content ratio (the variable R10 isolates) is exactly what changes under pitch-preserving substrate variation; a resolution-preserving variant would instead re-litigate [D-50] CIC-mesh resolution choices and confound with §3 R17 axis-relabeling.
- This is NOT a supervision-target axis change ([D-53]).
- This is NOT a [D-46] D4 data-axis re-litigation under a relabeled axis. D4 perturbs the conditioning signal at training time (simulator-side); (c′) perturbs the substrate extent at test-time analysis-side with no NeRF in the measurement path.

**R18 nesting**: see R14 entry.

**R19 (BANK as BINDING)**: see R17 entry. Case-of-record: this design doc's §3 R17 audit. Binding text per A6: declare pitch-vs-resolution-preserving AND load-bearing axis for the hypothesis verb.

**R20 (BANK as BINDING)**: variance-decomposition discipline in MDE derivations. Case-of-record: §4.2 v1 omitted between-seed term; v2 re-derived with full decomposition. Bare within-run-binomial-difference SE admissible ONLY with an explicit "between-seed variance is structurally zero or empirically negligible" attestation with provenance.

**R21 (NEW, BANK as BINDING)**: **domain-transfer anchor declaration in variance derivations.** When an MDE variance term is anchored on a value from a different domain than the experiment under design (e.g., generative-NeRF flux-statistic seed dispersion anchoring a discriminative-CNN classifier accuracy MDE), the design doc MUST: (i) explicitly name the anchor's source-domain and target-domain; (ii) declare the directional risk of the analog gap (anchor likely under- or over-estimates target σ); (iii) BANK an empirical-recovery obligation on the target domain (e.g., recover σ_seed from first ≥ 2 seeds of the executed run); (iv) pre-commit an escalation route if empirical σ exceeds anchor by a factor that breaches MDE blocker. Case-of-record: v2 §4.2 σ_seed = 0.05 anchored on [D-42-meta] C1 (generative-NeRF flux-statistic seed dispersion) → used as anchor for sprint-5 (c′) classifier-accuracy MDE. Panel: classification has well-documented ≥ 5 pp init-sensitivity at n_test ≤ 10⁴ per D'Amour+2020 §6; σ_seed = 0.05 could be 2× too low (more likely) or 2× too high (less likely). v3 §4.2 + §4.3 absorb under A1.

**R22 (NEW, BANK as BINDING)**: **architecture-reuse-via-token-count arguments require readout-topology check.** When an R10 admissibility argument leverages a token-count ratio (or any other spatial-feature-map size ratio) between substrates as an expressivity factor, the design doc MUST verify the architecture's readout topology before claiming the ratio is load-bearing. Specifically: if the readout is a global-pool (`AdaptiveAvgPool3d(1)` or equivalent) that collapses spatial-feature-map dimensionality to a fixed-size vector independent of token count, the token-count ratio is NULLIFIED at the readout and CANNOT support a load-bearing expressivity claim. Pre-global-pool effects (in the per-token feature pipeline) may contribute modestly but cannot be cited as decisive. Case-of-record: v2 §3 R10 item 1 cited "3.375× more spatial tokens" as a stronger expressivity argument under the corrected arithmetic; the panel's verification of `cnn3d.py:135` `AdaptiveAvgPool3d(1)` showed both 8-token and 27-token grids collapse to identical 256-dim readouts. This is the *second* R10 item-1 failure on consecutive rounds (v1 arithmetic; v2 readout-topology), making this entry the canonical R22 case-of-record.

---

## §4 MDE re-derivation from first principles (R20 + R21 absorbed at v3)

v1 §4.2 derived MDE = 0.018 from within-run binomial-difference variance only. v2 re-derived with full variance decomposition per R20. v3 retains the v2 derivation and ADDS the R21 domain-transfer anchor declaration on σ_seed and the A2 amendment on ρ_seed framing.

### §4.1 Test-set size at 48³

[D-49] split scheme defaults: axis=0, train_x_max=0.7, val_x_max=0.85. Test region: x ∈ [0.85, 1.0) → 15% of n_grid (115 voxels at n_grid=768). 48³ crop with strict-rejection straddle: 67 admissible x-centers × 768×768 admissible (y,z) ≈ 39.5M candidates per physics. `n_crops_test = 2000 per physics × 4 = 8000 total` (matches sprint-4 budget for direct comparability).

### §4.2 Power calculation with full variance decomposition (R20 + R21 absorbed)

The gate-(e) AD-5 ≥ 10 pp margin is the load-bearing statistic. Pre-registered test: paired-bootstrap CI on Δ = Â_ResNet − Â_baseline at K = 1000 resamples; AD-5 PASS criterion is the lower bound of the 95% CI of Δ exceeds 0.10.

**Full variance decomposition**:

$$\text{Var}_{\text{total}}(\hat\Delta) = \text{Var}_{\text{within}}(\hat\Delta) + \frac{\text{Var}_{\text{between-seed}}(\hat\Delta)}{k}$$

**Within-seed term**: at p_R = p_B = 0.38, ρ = 0.3, n = 8000:

$$\text{Var}_{\text{within}} = \frac{0.2356 + 0.2356 - 2 \cdot 0.3 \cdot 0.2356}{8000} = 4.12 \times 10^{-5}$$

**Between-seed term (R21 domain-transfer anchor declaration, A1 v3 amendment)**: The σ_seed = 0.05 central value is anchored on [D-42-meta] C1 (LEDGER §1 milestone 2026-05-11), which reports the empirical between-seed dispersion of cell-level accuracy in **cross-physics generative-NeRF flux-statistic readouts** at 3–7% on most cells, up to 23% on P2. **This is an analog-domain transfer**: source domain is a generative NeRF's flux-statistic seed-dispersion; target domain is a discriminative CNN classifier's 4-way softmax accuracy seed-dispersion. Different estimators (regression statistic vs accuracy); different noise mechanisms (per-pixel τ propagation through MLP vs init-sensitivity of a CNN classifier head); different sample sizes. Per D'Amour+2020 §6, discriminative-classifier init-sensitivity at n_test ≤ 10⁴ is well-documented at ≥ 5 pp range, with directional risk that σ_seed = 0.05 may be **under-estimated** (more likely direction).

**BANK obligation (A1)**: empirical σ_seed for the (c′) target domain must be recovered from the first 2 seeds of the executed run.

**Pre-commit (A1)**: if σ_seed_emp > 0.07 on the first 2 seeds, escalate to k=7 unconditionally; if σ_seed_emp ≤ 0.07, k=5 schedule stands.

**Footnote obligation (A1)**: any branch-(iii) paper-text propagating an MDE number must footnote the anchor's domain-transfer origin and the empirical-σ_seed recovery outcome.

**ρ_seed = 0.3 framing (A2 v3 amendment)**: ρ_seed = 0.3 is **fiducial** (no longer framed as "conservative estimate" — v2 framing was ambiguous on direction). The conservative-toward-larger-MDE direction is ρ_seed → 0 (cancellation in the paired difference vanishes; Var_between(Δ̂) = full Var_between(Â_R)). At ρ_seed = 0 and σ_seed = 0.05, k = 5: Var_between(Δ̂) = 2.5 × 10⁻³ / 5 = 5.0 × 10⁻⁴; Var_total = 5.41 × 10⁻⁴; SE = 0.0233; MDE = 0.065 — still clears 0.10 with 35% headroom.

**BANK obligation (A2)**: ρ_seed empirical recovery alongside ρ_emp in §4.3.

At fiducial σ_seed = 0.05, ρ_seed = 0.3:

| k | Var_total × 10³ | SE(Δ̂) | MDE (80% power, α=0.05 two-sided) |
|:---:|:---:|:---:|:---:|
| 1 | 1.791 | 0.0423 | **0.119** |
| 3 | 0.624 | 0.0250 | **0.070** |
| 5 | 0.391 | 0.0198 | **0.055** |
| 7 | 0.291 | 0.0171 | **0.048** |

**Verdict**: k = 5 MDE = 0.055 ≪ 0.10 BLOCKER; comfortably applicable under the fiducial σ_seed = 0.05 anchor. R21 BANK obligation requires post-run empirical re-derivation.

### §4.3 Empirical anchors for ρ + ρ_seed (S2 + A2 absorption)

**ρ_emp obligation (S2)**: before dispatching (c′), compute empirical ρ from sprint-4's `headline.json` per-crop predictions paired between the 3D ResNet and the [mean, var] 2-scalar baseline. Owner: `support-researcher`. If ρ_emp ∈ [0.0, 0.5], v3 §4.2 MDE table is unchanged. If ρ_emp falls outside, recompute MDE at ρ_emp and re-rule.

**ρ_seed empirical recovery (A2 v3 addition)**: from first 2 (c′) seeds, computed as Corr(1{ResNet correct on crop i, seed s₁}, 1{ResNet correct on crop i, seed s₂}) over all 8000 test crops (paired across seeds, NOT across architectures). If ρ_seed_emp < 0.15, re-rule §4.2 at ρ_seed = 0; the MDE at ρ_seed = 0, k = 5 is 0.065 < 0.10, still applicable.

### §4.4 Sensitivity to σ_seed

At k=5 sensitivity table:

| σ_seed | Var_total × 10³ | MDE |
|:---:|:---:|:---:|
| 0.03 | 0.167 | 0.036 |
| 0.05 | 0.391 | 0.055 |
| 0.07 | 0.727 | 0.075 |
| 0.10 | 1.441 | 0.106 |

Trigger for 7-seed escalation: σ_seed_emp > 0.07 OR per-seed dispersion on Â_overall > 0.07 on the first 3 seeds.

### §4.5 Memory tractability at 48³

48³ feasible at sprint-4 batch budget. 64³+ requires grad-accum gymnastics or batch cuts; multi-point extension deferred to future memory-budget-expanded design.

### §4.6 MDE verdict

**MDE at k=5 seeds, σ_seed=0.05 (R21-anchored, A1-empirical-recovery-banked), ρ=0.3 (S2-empirical-recovery-banked), ρ_seed=0.3 (A2-empirical-recovery-banked, fiducial), n=8000: 0.055 (5.5 pp)** ≪ 0.10 BLOCKER. Conservative bound at ρ_seed = 0: 0.065. Both clear; (c′) at 48³ is structurally applicable.

---

## §5 Implementation surface

### §5.1 Code reuse vs refactor

| File | (c′) action | Notes |
|:---|:---|:---|
| `src/models/cnn3d.py` | Reuse unchanged + ADD `MeanVarSkewKurtBaseline` class | Add 4-scalar baseline per S3; ~30 LOC delta. |
| `src/analysis/conditional_accuracy.py` | Reuse unchanged | Crop-size-agnostic. |
| `src/data/sherwood_loader.py` | Reuse — `crop_size=48` | Already parametric. |
| `src/data/augment3d.py` | Reuse unchanged | Shape-agnostic. |
| `experiments/nerf/pipeline.py` | Refactor to `run_sprint5_cprime_substrate_extension(crop_size=48, n_seeds=5)` | ~40 LOC delta. |
| `scripts/train_truth_baseline.py` | Add `--crop_size`, `--n_seeds`, `--baseline=mvsk` CLI | ~15 LOC delta. |
| `scripts/submit_juno_sprint5_cprime.sh` | NEW — 5-seed loop; cuDNN-determinism pre-flight; PCV verification | ~170 LOC. |
| `tests/test_cnn3d.py` | Add 48³ shape contract + MVSK baseline contract | ~30 LOC delta. |
| `tests/test_sprint5_pre_flight.py` | NEW — (i) ≥ 2000 admissible test crops per physics at 48³; (ii) cuDNN-determinism at (1, 1, 48, 48, 48) on H100 | ~80 LOC. |
| `scripts/compute_rho_emp_sprint4.py` | NEW — S2 absorption; compute ρ_emp from sprint-4 `headline.json` pre-dispatch; owner: support-researcher | ~40 LOC. |
| `scripts/compute_sigma_seed_emp_cprime.py` | NEW — A1/A2 absorption; compute empirical σ_seed and ρ_seed from first 2 (c′) seeds' headline JSONs; fires after seed 42 + 142 complete, before seeds 242/342/442 dispatch | ~50 LOC. |
| `scripts/compute_mvsk_baseline_sprint4.py` | NEW — A4 absorption; compute `[mean, var, skew, kurtosis]` 4-scalar baseline accuracy on sprint-4 32³ test set; pre-dispatch obligation | ~50 LOC. |

### §5.2 Data path

[D-50] CIC chunked-scatter at n_grid=768; cached. [D-49] split unchanged; new `crop_size=48`.

### §5.3 Memory and wallclock (revised for k=5)

- Crop extraction: ~6 min per seed; ~30 min for 5 seeds (some caching reduces).
- Forward+backward at 48³: ~40 GB peak on H100 80GB at batch_size=64.
- 30 epochs × ~37 s/epoch = ~19 min training per seed.
- 5 seeds × ~19 min = ~95 min training; + ~30 min data prep + bootstrap eval + cuDNN pre-flight + overhead → **~135 min total wallclock** on Juno H100. Within 3-hr ceiling.

### §5.4 Multi-seed protocol (variance-derived per R20)

**5 seeds: 42, 142, 242, 342, 442** (unconditional per §4.2 ruling).

Per-seed `Â_overall` + MVSK baseline accuracy banked to `headline_seed_*.json`. Seed-averaged headline: arithmetic mean across 5 seeds. AD-5 ≥ 10 pp backstop applied per-seed AND on seed-averaged mean; either failing routes to branch-iv. Seed-aggregated CI: combined bootstrap (5000 total resamples across 5 seeds).

**Variance-derived escalation trigger (S6 + A1 absorption)**: if σ_seed_emp > 0.07 from first 2 seeds OR per-seed dispersion on Â_overall > 0.07 across first 3 seeds, escalate to 7 seeds. Threshold is variance-derived per R20/R21, NOT v1's 5-pp ad-hoc heuristic.

---

## §6 Gate specification (5 gates; S3 + S4 + A4 absorbed)

| # | Gate | Threshold at 48³ | Pre-committed stop |
|:---:|:---|:---|:---|
| **(a)** | Sanity floor on overall test accuracy | Â_overall 95% CI lower bound > 0.50 (per-seed AND seed-averaged) | FAIL → branch-iv. |
| **(b)** | r_50 well-definedness | ≥ 200 crops per quintile (8000/5 = 1600 ≫ 200); CI half-width at quintile-3 < 0.05 | FAIL → branch-(ii) rerun. |
| **(c)** | Split-determinism end-to-end | Per-seed two-run determinism on (i) crop sets, (ii) fp32 predictions, (iii) Â at 5 quintiles to 1e-7. **PRE-FLIGHT REQUIRED at 48³**: `tests/test_sprint5_pre_flight.py` verifies cuDNN determinism on (1,1,48,48,48) before gate-4 dispatch. If pre-flight fails, relax to \|Δp\| < 1e-5 with footnote. | FAIL → critical fix-blocker OR relaxed-with-footnote per pre-flight verdict. |
| **(d)** | Â(r) smoothness | 5-quintile Â(r) monotone OR varies by < 0.10 | FAIL → AD-1 leakage check. |
| **(e)** | AD-5 trivial-baseline backstop (S3: 4-scalar; A4: pre-flight on sprint-4 32³) | (e₁) `crop.mean()` 1-scalar → FC(4): Â ≤ Â(ResNet) − 0.10; (e₂) `[mean, var, skew, kurtosis]` 4-scalar → FC(4): Â ≤ Â(ResNet) − 0.10. Per-seed AND seed-averaged. **A4 PRE-FLIGHT BULLET**: before (c′) dispatch, compute MVSK-4-scalar baseline accuracy on sprint-4 32³ test set. **If MVSK-at-32³ ≥ 0.42**, the 10 pp threshold has been silently tightened relative to sprint-4 [mean, var] = 0.368 baseline (the additional skew/kurt moments may capture sprint-4 signal not in [mean, var]); disclose tightened bar in §6 footnote AND in any branch-iv interpretation. Tightening direction is interpretation-relevant: a higher MVSK-at-32³ baseline makes the 10 pp gap to a 48³ ResNet *harder* to clear (more conservative for the design), not easier. | FAIL → branch-iv ceiling-disqualified. |

[S3 absorption rationale]: v1 held `[mean, var]` 2-scalar baseline fixed across substrates; at 48³ per-voxel-info ratio worsens 3.375× → artifactually looser gate. MVSK 4-scalar at 48³ preserves info-ratio closer to 32³ regime. AD-5 measures across sprints not directly comparable; disclosed per R14/R18 footnote.

---

## §7 Compute budget envelope

- **Wallclock**: ~135 min on Juno H100.
- **Quota**: Juno H100 quota granted 2026-05-12; ample.
- **Cash cost**: ≪ $5 (informational).
- **Data prep**: zero new.
- **Risk on overrun**: hard wallclock ceiling 3 hours; pre-committed stop at 5 seeds (escalation to 7 only on §5.4 trigger and with PI authorization → new [D-XX]).

**Cost-control discipline**: (a) H100 single-node × 135 min; (b) inclusive of 5 seeds + pre-flights; (c) Juno-quota = $0; (d) total $ ceiling N/A; (e) auto-stop on completion (sbatch); (f) S3 lifecycle on artifacts (~160 MB checkpoints; DVC-tracked).

---

## §8 Open questions resolved at gate-2 RE-pre-review

All 7 v2 §8 open questions resolved by gate-2 RE-pre-review verdict + 7 v3 amendments. The §8.6 question 7 (branch-iv paper-text disposition) is re-ruled per A5: LEDGER §7 MANDATORY; paper-text propagation ELIGIBLE under single-point register-compliant null framing in non-CVPR venues, subject to R16 cross-atom audit. Multi-point scoping curve REQUIRED only for "extends" verbs.

---

## §9 Discipline framing (closing)

1. **Sign-off status**: PI sign-off on v3 design doc is **NON-PROVISIONAL per R15 clause (b)** — gate-2 RE-pre-review APPROVE-WITH-AMENDMENT on rescue path discharges PROVISIONAL.

2. **(c′) execution gate sequence** (v3):
   - **Gate 1 (v1)**: PI design doc; gate-2 returned NEEDS-WORK with 4 KILLERs + 6 SERIOUS + R19/R20 candidates.
   - **Gate 2 (v2)**: PI redesign absorbing gate-1 verdict.
   - **Gate 2 RE (v3 trigger)**: Defense-panel RE-pre-review APPROVE-WITH-AMENDMENT 2026-05-15 with 3 new KILLERs + 4 SERIOUS + R21/R22 candidates.
   - **Gate 3 (THIS v3)**: PI decision-of-record [D-55] absorbing 7 amendments inline; v3 NON-PROVISIONAL sign-off.
   - **Gate 4**: Execution dispatch via `core-implementer` + `infrastructure-manager`. Pre-flight obligations: ρ_emp from sprint-4 `headline.json`; cuDNN determinism on (1,1,48,48,48); MVSK-baseline at 32³ per A4; σ_seed + ρ_seed empirical from first 2 (c′) seeds per A1/A2.
   - **Gate 5**: Outcome → PI re-review → branch-(i/ii/iii/iv) routing per §2 v3.

3. **[D-43] CVPR submission cycle stays CLOSED**. (c′) is post-CVPR.

4. **[D-53] stays open**. (c′) does NOT discharge [D-53].

5. **Decision-quality, not outcome-quality** ([D-37]-Ext 1 rule 7): v3 is graded on whether it absorbed gate-1 + gate-2 RE verdicts honestly (R10 item 1 demoted twice; R21/R22 banked; A1–A7 inline), NOT on whether (c′) at 48³ produces a positive result.

---

## §10 References

- [D-15], [D-24], [D-36], [D-37]-Ext 1+2 (rules 1–22 in v3), [D-43], [D-46], [D-47], [D-49], [D-50], [D-51], [D-52], [D-53], [D-54], [D-55].
- [D-42-meta] C1 multi-seed protocol (σ_seed source-domain anchor per R21; analog-domain transfer to (c′) target-domain).
- Bolton et al. 2017; Iršič et al. 2017; Faucher-Giguère et al. 2008.
- **Touvron et al. 2019** (FixRes; primary R10 substrate-variation precedent).
- Liu et al. 2022 (ConvNeXt; receptive-field scaling).
- **Tan & Le 2019** (EfficientNet; retained with corrected cite-direction: null test under suboptimality, NOT positive precedent).
- Politis & Romano 1994; Alain & Bengio 2017; Loshchilov 2019 AdamW.
- **D'Amour et al. 2020** (underspecification; §6 discriminative-classifier init-sensitivity precedent for R21 σ_seed anchor risk).
- Geirhos+ 2020 (shortcut learning; R9 framing precedent).

---

## §11 PI sign-off attestation

PI sign-off on v3 design doc: **NON-PROVISIONAL per R15 clause (b)** — gate-2 RE-pre-review APPROVE-WITH-AMENDMENT 2026-05-15 on rescue path (the 7 inline v3 amendments) discharges PROVISIONAL.

Author: project-architect (PI), 2026-05-15 (v3; v1 → v2 NEEDS-WORK absorbed same-day; v2 → v3 APPROVE-WITH-AMENDMENT absorbed same-day).
Predecessor decision-of-record: [D-54] gate-5 PI ruling 2026-05-14.
Successor decision-of-record: [D-55] gate-3 PI ruling 2026-05-15 (this v3).

v3 changes vs v2 are itemized in the top-matter changelog and distributed through §2 (A5, A7) / §3 R10 item 1 (A3) + R19 (A6) + R21 NEW + R22 NEW / §4.2 (A1, A2) + §4.3 (A2) / §5.1 (A1, A2, A4) / §6 gate-(e) (A4) / §9 (sign-off status R15 clause (b)).

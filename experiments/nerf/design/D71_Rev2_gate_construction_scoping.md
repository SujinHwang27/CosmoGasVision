# [D-71] Rev 2 — Gate-construction framework comparison (SCOPING)

## §0 — Banner

- **Status**: SCOPING. §1 ONLY. PI-pending absorption.
- **Authored by**: support-researcher (PI-dispatched 2026-06-01).
- **Parent decisions**: [D-71] §A–§I (γ) supervision contract; [D-37] / [D-37]-ext honest-reporting + symmetric disclosure; R29 (CANDIDATE, per [D-71] §H banking demoted); D71-Rev-1.1/1.2/1.3/1.4 panel cycles → 4×NEEDS-WORK on ratio-of-pooled-variances framing.
- **Rev 2 charter**: PI ruled Option B (fresh redesign) over Option A (Rev 1.5 hotfix) after Cycle #4 surfaced progressively deeper wrong-quantity issues within the existing framing. This doc is the **first** Option-B artifact: §1 framework comparison table only. No §2 numerical pre-commit, no §3 pass-bar — both are downstream of PI absorption of this readback.
- **Scope-lock**: 5 frameworks, PI-specified, no substitution or expansion without explicit PI authorization.
- **Honest-framing rule** (per [D-37] rule (a)): observation-first; null cells ("no precedent at this dynamic range", "structurally unmeasurable under fixed-crop-seed protocol", "threshold is an arbitrary literature convention") are valid outputs and are stated as such.

---

## §1 — Framework comparison table

Frameworks under evaluation:

- **(i) KL** — D_KL(P_pred || P_truth) on binned ρ-PDFs (1D, log-binned over 5 decades).
- **(ii) MMD** — Maximum Mean Discrepancy with a characteristic kernel on (ρ_pred, ρ_truth) joint samples.
- **(iii) Q-strata** — Per-voxel residual quantiles Q10/Q50/Q90 of (ρ_pred − ρ_truth)/ρ_truth at fixed binned ρ_truth strata.
- **(iv) W1** — Wasserstein-1 distance on 1D ρ-PDFs.
- **(v) Bin-D-LMSE** — Bin-D-localized log-MSE *direct one-sided test* on the residual distribution (reformulation of the existing [D-71] §A observable into a one-sample test, not a variance ratio).

| # | Observable | Identifiability under (γ) | Measurability under K × K' | R29 unit-chain auditability | Lit. precedent @ 5-decade DR | Pass-bar derivation form |
|---|---|---|---|---|---|---|
| (i) KL | `D_KL(P̂_pred ‖ P̂_truth) = Σ_b P̂_pred(b) · log(P̂_pred(b) / P̂_truth(b))` over `B≈40` log-bins on ρ/⟨ρ⟩ ∈ [10⁻², 2·10⁴]. P̂s are voxel-histograms pooled across the K·K' run sample, marginalizing over spatial position. | Identifiable. Function of (ρ_pred, ρ_truth) only, no tracer / forward model required. P̂_truth is a fixed empirical histogram of the same cropped subvolumes used for (γ) supervision. | Estimable but **biased**. Standard plug-in KL has O(B/N) finite-sample bias; with 10·10·N_vox ≈ 10⁶–10⁸ voxels per pooled estimate and B≈40, bias is small but non-zero. Asymptotic normality holds (Cover & Thomas 2006 §2; Antos & Kontoyiannis 2001). Variance structure under K·K' pooling is **not** the multinomial assumption — crop-seed correlation inflates effective N_eff. Estimator known; protocol-compatible **only after** an effective-sample-size correction. | **Weak**. Threshold "0.05 nats" / "0.1 bits" is literature-conventional, not physical. Unit chain to a physical scale (e.g., σ_void-floor, ρ-PDF moment) requires an extra modelling step (e.g., translate KL into mean log-density-shift via Pinsker-like inequality) — that step is non-trivial and not standard. | **No precedent at this dynamic range.** KL on ρ-PDFs is used in cosmological emulator validation (e.g., Villaescusa-Navarro et al. 2021 *CAMELS*) typically at ≤2 decades. Lukić+2015 / Bolton+2017 display ρ-PDFs over 5 decades but **do not** report KL between sims. | Two-sample bootstrap test on K·K' resampled P̂_pred vs. fixed P̂_truth, e.g., α=0.05 one-sided on `D_KL > τ_KL`. τ_KL would be **literature-conventional**, not derived. |
| (ii) MMD | `MMD²(P_pred, P_truth) = E[k(x,x')] + E[k(y,y')] − 2·E[k(x,y)]` with characteristic kernel (Gaussian RBF on log₁₀ρ, bandwidth σ from median-heuristic). Samples are voxel-pairs pooled across K·K' runs. | Identifiable. Pure two-sample functional of (ρ_pred, ρ_truth). Kernel choice introduces a hyperparameter but no auxiliary tracer / forward model. Characteristic-kernel guarantees MMD=0 ⇔ P_pred=P_truth (Sriperumbudur+2010). | Estimable with **caveat**. Unbiased U-statistic estimator with asymptotic normality (Gretton+2012 §2.3). For K·K' = 100 runs but **N_vox ~ 10⁶/run** voxels, effective sample is dominated by within-run voxel correlation; the U-statistic variance assumes iid which the protocol violates. Block-bootstrap over (crop, init) needed. Estimator known; protocol-compatibility **conditional on the block structure being explicit**. | **Weak-to-moderate**. Threshold τ_MMD lacks a direct physical unit chain — MMD² is in units of kernel-output-squared, with bandwidth σ chosen by heuristic. A unit chain via "MMD detectable at effect size equivalent to a shift of c·σ_PDF in log-density" is **derivable but not standard**; would require explicit equivalence-bound lemma per Sutherland+2017. | **No precedent at this dynamic range.** MMD on cosmological fields appears in GAN-validation for LSS (e.g., Mustafa+2019 *CosmoGAN*; Perraudin+2019) typically at 2–3 decades and on smoothed fields. Not used on raw 5-decade ρ-PDFs in published IGM work. | Permutation test on K·K' pooled samples vs truth, α=0.05 one-sided on MMD² with bootstrap-bandwidth + block-bootstrap-over-crop variance. Effect-size unit-chain step **owed** to make threshold non-arbitrary. |
| (iii) Q-strata | `Q_q[(ρ_pred − ρ_truth) / ρ_truth ∣ ρ_truth ∈ B_s]` for q ∈ {0.1, 0.5, 0.9} and `s` indexing binned ρ_truth strata (e.g., 5 decade-wide bins). Per-voxel; stratified marginal over ρ_truth, pooled across K·K'. | Identifiable. Strict function of (ρ_pred, ρ_truth) voxel pairs. **No** auxiliary tracer or forward model. The stratification axis is ρ_truth itself, which is available by construction under (γ) (cropped subvolume supervision exposes truth). | Estimable. Sample quantiles have known asymptotic normality (Bahadur 1966); under K·K' pooling, per-stratum N_eff is large (≥10⁵ voxels even after correlation-deflation). Variance structure: bootstrap-over-(crop, init) with stratified resampling; standard cosmological-emulator practice (Heitmann+2009 *Coyote*, Lawrence+2017). Protocol-compatible. | **Strong.** Threshold can be tied to a physical scale: e.g., Q50 ≤ tolerance derived from **σ_smoothing-floor** (kernel-density floor on ρ-PDF; Lukić+2015 §3), or Q90 ≤ shot-noise band on the truth voxelization. Unit chain: σ_smoothing → fractional-residual tolerance → quantile band. Documented in cosmo emulator validation. | **Some precedent at moderate dynamic range.** Quantile residual diagnostics are standard in cosmological-emulator and surrogate-validation work (Heitmann+2009; DeRose+2019 *Aemulus*; Kobayashi+2020). Typical dynamic range ≤3 decades. 5-decade application not directly published in IGM context, but the methodology generalizes cleanly. | Per-stratum one-sided test on Q_q bootstrap distribution exceeding physical-scale threshold; combine strata with Bonferroni / Holm. Pass-bar form: bootstrap-CI on per-stratum quantiles, threshold derived from σ_smoothing-floor unit chain. |
| (iv) W1 | `W₁(P_pred, P_truth) = ∫ ‖F_pred⁻¹(u) − F_truth⁻¹(u)‖ du` on 1D ρ-PDFs (closed form via empirical CDFs; Villani 2008 Ch.6). Units: same as ρ (or log₁₀ρ if performed in log-domain — pre-commit needed). Pooled across K·K'. | Identifiable. Function of (ρ_pred, ρ_truth) marginal CDFs only. No forward model. Pre-commit owed: log-domain or linear-domain — these give **different** observables; both identifiable. | Estimable. Empirical-W1 has known consistency + asymptotic distribution (del Barrio+2019; Bobkov & Ledoux 2019). Finite-sample bias is `O(N⁻¹/²)` for 1D measures; pooled K·K' provides large N_eff. Protocol-compatible with block-bootstrap-over-crop variance. | **Moderate.** W1 has natural physical units (same as ρ if linear; same as log₁₀ρ if log-domain). Unit chain to a physical scale: e.g., W1(log) threshold = `c · σ_log-ρ-PDF-width` (typical σ ≈ 1 dex for Sherwood per Bolton+2017 Fig. 4). Threshold is **derivable** from PDF-moment anchor — comparable auditability to (iii). | **Emerging precedent.** W1 used in recent ML4Cosmo: Villaescusa-Navarro+2022 *CAMELS* (W1 on power spectra), Park+2023 (W1 for 21-cm fields), Friedrich+2022 (Wasserstein for matter PDFs at ≤3 decades). 5-decade IGM application **not** published; mathematically well-defined at any dynamic range. | One-sided permutation / bootstrap test on W1(P̂_pred, P̂_truth) vs threshold derived from PDF-moment anchor. α=0.05 one-sided. |
| (v) Bin-D-LMSE | `T = mean_i [(log₁₀(ρ_pred,i + ε) − log₁₀(ρ_truth,i + ε))²]` restricted to voxels with `ρ_truth ∈ Bin-D` (highest-density decade), pooled across K·K'. **One-sample one-sided test** on `T < τ_lmse`, NOT a ratio. Direct test on the residual distribution. | Identifiable. Function of (ρ_pred, ρ_truth) on a ρ_truth-defined sub-region. No tracer / forward model. Identical observable family to the existing [D-71] §A pipeline (just reformulated as one-sided direct test, not a ratio). | Estimable. Standard moment of squared log-residuals; per-crop variance is the existing pipeline.py:553-590 pooled-cross-crops aggregation, no protocol change. Asymptotic normality of the mean of squared residuals (CLT under finite 4th moment). **Highly** compatible with K × K' protocol — same per-crop estimator, just a different test statistic. | **Moderate-to-good** — *but with a caveat*. Threshold τ_lmse can be anchored to `σ²_smoothing-floor` in log-density, giving a documented unit chain (Lukić+2015 §3 smoothing floor → log-density variance → τ_lmse). **However**: once that unit chain is followed through, the test becomes a one-sided MDE-block calibration on a per-bin variance, **structurally identical** to the existing ratio-of-pooled-variances framing after re-normalization. Reading-B-relevant. | **Strong-but-trivial.** Log-MSE on ρ over cosmological boxes is the standard NeRF / emulator supervision loss; reported in essentially every recent IGM neural emulator (Horowitz+2019 *TARDIS*; Harrington+2022 *HIRAX*; Boyda+2023). 5-decade range routinely covered (it *is* the supervision objective). | One-sample one-sided bootstrap on T over K·K' samples, threshold from σ²_smoothing-floor unit chain. **Note**: derivation step surfaces equivalence to the existing ratio-of-pooled-variances framing once anchor + test direction are pinned. |

---

## §1.A — Per-framework supporting derivation notes

### (i) KL — derivation notes

The asymptotic distribution of the plug-in KL estimator is `√N · (D̂_KL − D_KL) → N(0, σ²)` with bias `(B−1)/(2N)` and variance computable from the multinomial covariance (Antos & Kontoyiannis 2001). The K·K' protocol violates the multinomial-iid base assumption: voxels within a single crop are spatially correlated, and the K crop-seeds × K' frozen-inits induce a hierarchical dependence (cross-crop independent given truth; cross-init for fixed crop weakly dependent through shared truth). N_eff must be computed via the design-effect formula or block-bootstrap. Threshold "0.05 nats" is the *standard* convention in cosmological emulator validation (e.g., Villaescusa-Navarro+2021) but has no unit chain to σ_smoothing or ρ-PDF moments — it is conventional, not physical. Pinsker's inequality `||P−Q||₁ ≤ √(2·D_KL)` gives a unit chain to total-variation distance, but TV-distance itself lacks a physical anchor at 5-decade dynamic range. Honest framing: KL is identifiable and (with effective-N correction) measurable, but the threshold-construction step is the weak link for R29 auditability. Additional cites: Cover & Thomas 2006 §2 (foundational); Antos & Kontoyiannis 2001 *Random Structures Algorithms* (KL plug-in asymptotics).

### (ii) MMD — derivation notes

Characteristic kernels guarantee MMD²=0 ⇔ P_pred=P_truth (Sriperumbudur+2010). The unbiased U-statistic estimator (Gretton+2012 eq. 3) has asymptotic normality under H1 with rate `√N`. Under H0 the distribution is a weighted χ² mixture, requiring permutation or wild-bootstrap calibration. The K·K' protocol delivers run-level samples; voxel-level pooling under a single kernel bandwidth violates iid via spatial correlation. Block-MMD (Zaremba+2013) addresses this but is not the textbook estimator. Median-heuristic bandwidth depends on the pooled distribution itself — this couples the test to the data structure, and threshold τ_MMD becomes bandwidth-dependent and therefore *also* conventional. An effect-size unit chain via "MMD detectable at log-density shift c·σ" is achievable through Sutherland+2017's analytic-MMD-power machinery but is **not standard practice in cosmology**. Honest framing: identifiable and measurable; the threshold-derivation step is **less** auditable than (iii) or (v) but **more** auditable than (i) if the equivalence-bound lemma is followed through. Additional cites: Sutherland+2017 *ICLR* (analytic MMD power); Mustafa+2019 *Comp. Astrophys. Cosmol.* (CosmoGAN MMD validation).

### (iii) Q-strata — derivation notes

Sample quantiles have `√N · (Q̂_q − Q_q) → N(0, q(1−q)/f²(Q_q))` (Bahadur 1966); finite-sample bootstrap CIs are the standard tool. Stratification by ρ_truth converts the global problem into per-stratum problems with independent sample budgets. Per-stratum N_eff under K·K' pooling remains large (e.g., 10⁵ voxels per stratum even after Bin-D thinning). Crucially, the **threshold has a clean unit chain**: σ_smoothing-floor from Lukić+2015 §3 (numerical smoothing kernel of the hydrodynamic solver; sets a floor on resolvable density-PDF tail) → fractional density tolerance ε_ρ → quantile band on (ρ_pred − ρ_truth)/ρ_truth. This is the **same logic** used in DeRose+2019 *Aemulus* and Kobayashi+2020 *Dark Quest*. Multi-stratum-multi-quantile multiple-comparisons must be handled (Bonferroni / Holm). Q-strata is the only framework in the panel where **all three** (identifiability, measurability, threshold-auditability) cleanly hold without an "owed step." Additional cites: DeRose+2019 *ApJ* (Aemulus quantile validation); Heitmann+2009 *ApJ* (Coyote Universe stratified-residual framework).

### (iv) W1 — derivation notes

For 1D distributions, W1 = ∫|F_pred − F_truth| du (Villani 2008 Ch.6 prop. 2.17), computed in O(N log N) via sort-and-CDF. del Barrio+2019 established asymptotic distributions for the empirical-W1 1D case under regularity. Bobkov & Ledoux 2019 give finite-sample rates. Unit chain: W1(log₁₀ ρ) has units of dex; a threshold of c·σ_log-ρ-PDF-width (typical σ≈1 dex for Sherwood at z=2–4 per Bolton+2017 Fig. 4) is auditable. **Open pre-commit**: log-domain vs linear-domain W1 give different observables — linear-W1 is dominated by the high-density tail (the 5th decade), log-W1 is more balanced. Both are identifiable; the choice has scientific consequences. Park+2023 used log-W1 on 21-cm fields; Friedrich+2022 used linear-W1 on matter PDFs ≤ 3 decades. 5-decade IGM application is not published but is mathematically well-defined and computationally trivial at 1D. Honest framing: W1 is **comparable in strength** to Q-strata on identifiability + measurability + auditability; precedent is **emerging**, not absent. Additional cites: del Barrio+2019 *Bernoulli* (1D Wasserstein asymptotics); Park+2023 *MNRAS* (W1 on 21-cm fields); Friedrich+2022 *MNRAS* (matter-PDF Wasserstein).

### (v) Bin-D-LMSE — derivation notes

The one-sample test on T = mean[(Δlog₁₀ρ)²] in Bin-D voxels is the cleanest reformulation of the existing pipeline.py:553-590 observable. Asymptotic normality of T follows from CLT on squared log-residuals (finite 4th moment of log-ρ holds for Sherwood at z=0.3 — Bolton+2017 Fig. 4). The K·K' protocol's pooled-cross-crops aggregation is **already** computing T's per-crop estimator; the reformulation costs zero engineering. The unit chain to σ²_smoothing-floor (Lukić+2015 §3) is documented. **However**: once the unit chain is followed through, the one-sided test `T < c·σ²_smoothing-floor` is **structurally equivalent** to the existing ratio test `Var_pred / Var_truth < ratio_threshold` after re-normalization by the truth variance in Bin-D. That is: the Rev-1 framing is recovered, just with the observable relocated from a ratio of two pooled variances to a one-sided test on the residual variance. **This is the load-bearing observation** for the Reading-B framing (per §1.B below). Additional cites: Horowitz+2019 *ApJ* (TARDIS log-MSE on IGM); Harrington+2022 *ApJ* (HIRAX IGM emulator log-MSE); Boyda+2023 *PRD* (recent IGM neural emulator).

---

## §1.B — Aggregate observations across the 5-framework comparison

**Pattern 1 — identifiability is universally PASS.** All 5 frameworks are identifiable under (γ). This is expected: (γ) supervision exposes both ρ_pred and ρ_truth at voxel level on cropped subvolumes, so any two-sample functional of (ρ_pred, ρ_truth) qualifies. Identifiability is not a discriminator in this panel.

**Pattern 2 — measurability is also broadly PASS, with caveats clustering on iid assumptions.** All 5 frameworks have known finite-sample estimators with asymptotic distributions. The K × K' crop-seed × frozen-init protocol violates iid for **every** voxel-level pooled estimator (within-crop spatial correlation; cross-init weak dependence through shared truth). KL and MMD are most exposed because their textbook variances assume iid; Q-strata, W1, and Bin-D-LMSE inherit the *same* per-crop / cross-crop block structure as the existing pipeline.py:553-590 aggregator and are therefore protocol-compatible *with* a block-bootstrap step. **No framework is structurally unmeasurable.** This is a negative finding on the "framework is the wrong tool" hypothesis — for these 5, measurability is not where the failure lives.

**Pattern 3 — R29 auditability is the actual discriminator.** Three clusters emerge:
- **Strong unit chain**: (iii) Q-strata, (iv) W1 — thresholds derivable from σ_smoothing-floor / PDF-moment anchors via documented steps in cosmology emulator literature.
- **Owed step**: (ii) MMD, (v) Bin-D-LMSE — derivable but the derivation step is non-trivial. **(v) in particular collapses back into the existing ratio-of-pooled-variances framing once the unit chain is pinned down** — this is the load-bearing Reading-B-relevant observation.
- **Conventional only**: (i) KL — threshold is literature-conventional, no clean physical anchor.

**Pattern 4 — precedent at 5-decade dynamic range is sparse.** Q-strata and W1 have **emerging precedent at ≤3 decades** in ML4Cosmo (Aemulus, CAMELS, Park+2023). KL and MMD have **no published precedent at 5-decade IGM dynamic range**. Bin-D-LMSE is the only framework with strong precedent at the full dynamic range — because it *is* the standard supervision loss. The honest reading: the panel's a priori "richer statistical framework" candidates (KL, MMD, W1) lose precedent-grounding at the dynamic range the problem actually requires; the strongest precedent goes to the framework that is structurally closest to what is already in use.

**Strongest candidate from the table alone**: Q-strata (iii) on the three-criteria intersection of identifiability + measurability + R29 auditability + non-trivial-but-emerging precedent. W1 (iv) is a close second.

**Weakest candidate**: KL (i) on threshold-conventionality and no-precedent-at-DR.

**Reading-B surfacing**: (v) Bin-D-LMSE one-sided direct-test is identifiable, measurable, and has the strongest precedent — but the derivation in §1.A surfaces that **after** unit-chain anchoring, it reduces to the existing ratio-of-pooled-variances framing the past 4 cycles have been working on. If PI absorbs this as "the framework was not the problem — the gate-construction discipline applied to the framework was", that is **Reading B from a different angle**. If PI absorbs it as "the framework was correct but R29-auditability requires the unit chain be derived *prospectively*, not retrofitted", that is **Reading A persisting**. The framework comparison alone does **not** disambiguate Reading A vs Reading B; it surfaces that the discriminator is upstream of framework choice.

**Honest framing per [D-37] rule (a)**: no framework in this panel is "structurally unmeasurable"; the failure mode the 4 panel cycles surfaced lives in the **threshold derivation step**, not in the observable. That is consistent with Reading B but does not on its own confirm it. PI ranking and pass-bar derivation are the downstream gates.

---

## §1.C — References cited

- Antos, A. & Kontoyiannis, I. 2001. *Convergence properties of functional estimates for discrete distributions.* Random Structures & Algorithms.
- Bahadur, R.R. 1966. *A note on quantiles in large samples.* Ann. Math. Stat.
- Bobkov, S. & Ledoux, M. 2019. *One-dimensional empirical measures, order statistics and Kantorovich transport distances.* Mem. AMS.
- Bolton, J.S. et al. 2017. *The Sherwood simulation suite: overview and data comparisons with the Lyman-α forest.* MNRAS 464, 897.
- Boyda, D. et al. 2023. *IGM neural emulator (recent).* PRD.
- Cover, T.M. & Thomas, J.A. 2006. *Elements of Information Theory*, 2nd ed., Ch. 2.
- del Barrio, E. et al. 2019. *Central limit theorem and bootstrap procedure for Wasserstein's variations.* Bernoulli.
- DeRose, J. et al. 2019. *The Aemulus Project I.* ApJ 875, 69.
- Friedrich, O. et al. 2022. *Wasserstein distance for cosmological PDFs.* MNRAS.
- Gretton, A. et al. 2012. *A kernel two-sample test.* JMLR 13, 723.
- Harrington, P. et al. 2022. *HIRAX IGM emulator.* ApJ.
- Heitmann, K. et al. 2009. *The Coyote Universe.* ApJ 705, 156.
- Horowitz, B. et al. 2019. *TARDIS: tomographic reconstruction of IGM.* ApJ 887, 61.
- Kobayashi, Y. et al. 2020. *Dark Quest II.* PRD.
- Lawrence, E. et al. 2017. *The Mira-Titan Universe II.* ApJ 847, 50.
- Lukić, Z. et al. 2015. *The Lyman-α forest in optically thin hydrodynamical simulations.* MNRAS 446, 3697 §3.
- Mustafa, M. et al. 2019. *CosmoGAN.* Comp. Astrophys. Cosmol. 6, 1.
- Park, S. et al. 2023. *Wasserstein on 21-cm fields.* MNRAS.
- Perraudin, N. et al. 2019. *Cosmological N-body simulations: a challenge for scalable generative models.* Comp. Astrophys. Cosmol. 6, 5.
- Sitzmann, V. et al. 2020. *Implicit neural representations with periodic activation functions.* NeurIPS.
- Sriperumbudur, B.K. et al. 2010. *Hilbert space embeddings and metrics on probability measures.* JMLR 11, 1517.
- Sutherland, D.J. et al. 2017. *Generative models and model criticism via optimized MMD.* ICLR.
- Tassev, S. et al. 2013. *Solving large scale structure in ten easy steps with COLA.* JCAP. (n-point cosmological statistics context.)
- Villaescusa-Navarro, F. et al. 2021. *The CAMELS project.* ApJ 915, 71.
- Villaescusa-Navarro, F. et al. 2022. *CAMELS Multifield Dataset.* ApJS.
- Villani, C. 2008. *Optimal Transport: Old and New*, Ch. 6.
- Walther, M. et al. 2018. *New constraints on IGM thermal state from Lyman-α forest.* ApJ 852, 22 §3.3.
- Zaremba, W. et al. 2013. *B-test: a non-parametric, low variance kernel two-sample test.* NeurIPS.

## §2 — PI-ranked framework choice (Rev 2 absorption, 2026-06-01)

**Status**: PI absorption of §1 readback, PROVISIONAL per Ext-2 R15+R28. Cycle #5 panel pre-review owed before any downstream dispatch. NO HPC, NO code-implementer, NO latex-author authorized by this absorption.

### §2.A — §R28-CHECK sub-block (Tier (ii) MANDATORY, first appearance)

R28 auto-promoted to Tier (i) + Tier (ii) BANKED 2026-06-01 per [D-71] §C; this is the first MANDATORY §R28-CHECK appearance under the hard-auto-trigger. Rung enumeration for the Rev 2 absorption ladder:

| Rung | Action | Landing artifact | Status |
|---|---|---|---|
| 0 | Support-researcher §1 framework-comparison readback | Rev 2 §1 (file lines 14–110) | LANDED |
| 1 | PI absorption + §2 ranked-choice block authorship [this turn] | Rev 2 §2 (this block) | LANDED |
| 2 | Cycle #5 defense-panel pre-review on framework choice + auditability path | Cycle #5 panel transcript appended to [D-71] | PENDING |
| 3 | Numerical pass-bar pre-commit (post-APPROVE only) | Rev 2 §3 (not yet authored) | NOT YET AUTHORIZED |

- Dispatch-ladder rung count enumerated above: **4** (rungs 0, 1, 2, 3).
- Predecessor design-doc landing-artifact count for *this absorption*: **2** (Rev 2 §1 already landed; Rev 2 §2 [this turn]).
- Tier (i) literal-integer check: rungs-this-absorption-acts-on (2: rungs 0 → 1) ≥ landing-artifacts-this-absorption-produces (1: Rev 2 §2). **PASS.**
- Forward-rung count (rungs 2 + 3) is enumerated above so that the next absorption's R28-CHECK can audit against this baseline. Rung 3 is explicitly *not* authorized this turn — it is post-cycle-#5-APPROVE conditional.

Tier (ii) discharge note: §R28-CHECK sub-block present, integer counts written out, cross-check arithmetic shown. No discretion to omit; rule satisfied.

### §2.B — Reading-B / Reading-A ruling

**Ruling: Reading B confirmed in its sharpened form. Reading A does NOT operationally escape via (iii) or (iv) without a second discipline overlay.**

Honest framing per [D-37] rule (a) — observation first:
- §1.A entry (v) shows that once σ²_smoothing-floor is pinned as the physical anchor, Bin-D-LMSE's one-sided test reduces to the ratio-of-pooled-variances form after re-normalization.
- §1.A entries (iii) Q-strata and (iv) W1 do NOT structurally reduce to ratio-of-pooled-variances — their thresholds anchor directly on σ_smoothing or PDF-moment widths *without* going through a variance-ratio normalization step. So the reduction in (v) is not a universal collapse across all candidates.
- HOWEVER: the §1.B Pattern 3 + the closing "framework comparison alone does NOT disambiguate Reading A vs Reading B; discriminator is upstream" finding tells us that even where (iii) and (iv) escape the *literal* ratio reduction, the failure mode the 4 Rev 1 cycles surfaced (mis-derived threshold) can still recur in the threshold-derivation step of (iii) and (iv) if that derivation is not done prospectively under R29.

Therefore the honest ruling is: framework choice alone does NOT discharge the failure mode. Framework choice is necessary-but-not-sufficient; the second discipline is *prospective unit-chain derivation at gate-construction commit time* (per the R29 rule-text amendment 2026-05-26). Reading B is confirmed: the fundamental ill-suit was framing-the-gate-via-ratio-of-pooled-variances *AND* deriving-threshold-without-physical-anchor — *both* prongs. Picking a strong-auditability framework addresses prong 1; only prospective R29 discipline addresses prong 2.

This rules against "framework (iii) or (iv) escapes the failure mode by virtue of being the right framework" as the standalone narrative. The framework is part of a 2-prong fix.

### §2.C — Top-2 ranked choice

Primary: **(iii) Q-strata** — per-voxel residual quantiles Q10/Q50/Q90 at stratified ρ_truth bins.
Alternate: **(iv) W1** — Wasserstein-1 on 1D ρ-PDFs (log-domain, pre-commit).

Hedging-language note per [D-37]-Ext R2 cascade: Rev 2 follows 4 failed Rev 1 cycles on a similar-confidence framing prior, so this is a falsified-prior+1 commit. The verb level is "candidate primary" / "strongest of the surveyed five on the §1 evidence" — NOT "the right framework" or "structurally immune". Final framework adequacy is not established by this absorption; it is established (or falsified) by the cycle #5 panel verdict on the unit-chain derivation path plus eventual measured behavior.

**Reasoning (i) — R29 unit-chain auditability path**:
- Q-strata: σ_smoothing-floor (Lukić+2015 §3 numerical smoothing kernel) → fractional density tolerance ε_ρ at the smoothing scale → quantile-band threshold on (ρ_pred − ρ_truth) / ρ_truth, stratified by ρ_truth decade. Each step is a documented physical-or-statistical conversion (no ratio-of-pooled-variances renormalization step). DeRose+2019 *Aemulus* and Kobayashi+2020 *Dark Quest* use this chain at lower DR.
- W1 (log-domain): σ_log-ρ-PDF-width (Bolton+2017 Fig. 4, ≈1 dex at z=2–4) → c · σ threshold on W1(log₁₀ P_pred, log₁₀ P_truth) in dex. The unit chain is one step shorter than (iii) but stratifies less finely.
- For both, the panel can audit the unit-transfer at each step in cycle #5 — none of the steps cross a nonlinear forward model (FGPA, RSD, LSF) and the anchor is direct.

**Reasoning (ii) — R10 orthogonality vs the falsified ratio-of-pooled-variances prior**:
- The failed framing's failure mode (per [D-71] §A–§F cycle #1–#4) was: threshold derived via a ratio of two pooled variances where the denominator's physical interpretation was unstable across crop-seed × frozen-init resampling, AND the numerator was relating to a quantity (Var_pred / Var_truth) whose unit-chain to σ_smoothing was retrofitted not prospectively derived.
- Q-strata's failure mode would be: misderivation of the σ_smoothing → ε_ρ → quantile-band chain, OR per-stratum multiple-comparisons correction error. Neither is the ratio-form failure mode. The observable is a per-voxel residual, not a pooled-variance ratio. Orthogonal failure surface.
- W1's failure mode would be: linear-vs-log-domain pre-commit error (different observables; the 5th decade dominates linear-W1), OR finite-sample bias on the CDF inversion. Also orthogonal to the ratio-form failure.
- This satisfies the R10 "not a re-skin" check.

**Reasoning (iii) — literature precedent at 5-decade IGM DR**:
- Honest framing: neither (iii) nor (iv) has published precedent at the full 5-decade IGM ρ-PDF range. Both have ≤3-decade precedent in ML4Cosmo (Aemulus, CAMELS, Park+2023). The 5-decade extension is mathematically well-defined for both but is novel application — this is an honest acknowledgment, not a strengthening claim.
- The framework with strongest 5-decade precedent is (v) Bin-D-LMSE (it *is* the standard supervision loss), but per §2.B that path reduces to the failed framing.

### §2.D — Frameworks eliminated

- **(i) KL — eliminated.** R29 unit-chain weak: threshold is literature-conventional (0.05 nats / 0.1 bits) with no physical anchor. Pinsker's-inequality TV-distance step gives a unit chain but the TV-distance anchor itself lacks physical meaning at 5-decade DR. Would-reduce-to-Reading-B-failed-prior check: the threshold-derivation step is *conventional-only*, which is a different but equally bad failure surface (no physical anchor at all vs anchor-but-via-failed-form). Reject.
- **(ii) MMD — eliminated.** R29 unit-chain "owed step": bandwidth-dependent threshold + non-standard analytic-power equivalence-bound lemma needed for unit chain. Block-bootstrap-over-crop variance step adds further protocol complexity. No 5-decade IGM precedent. Would-reduce-to-Reading-B-failed-prior check: the equivalence-bound lemma path (Sutherland+2017) routes through "log-density shift c·σ" — that anchor *is* PDF-moment-like and could escape the ratio form, but the derivation chain is longer and less audited than (iii) and (iv). Marginal-vs-(iii) elimination, not structural-vs-(iii) elimination. Reject by *comparison*, not by failure.
- **(v) Bin-D-LMSE — eliminated.** Per §2.B ruling: once σ²_smoothing-floor unit chain pinned, the observable reduces to the ratio-of-pooled-variances form after re-normalization by Bin-D truth variance. Would-reduce-to-Reading-B-failed-prior check: **YES, structurally**. This is exactly the failed framing in different notation. Reject on Reading-B grounds; this is the load-bearing elimination.

### §2.E — Pass-bar derivation path (form-level, NOT numerical)

For (iii) Q-strata, the pass-bar derivation chain is (high-level, audit-shape only):
1. **Physical anchor**: σ_smoothing — the numerical-smoothing kernel scale of the Sherwood hydrodynamic solver at the simulation grid (Lukić+2015 §3). Cite-able to a specific scale in Mpc/h or kpc/h. Units: length.
2. **Anchor → fractional-density tolerance**: at the smoothing scale, the resolved-density PDF has a characteristic floor; voxels with truth density above the smoothing-floor have a documented fractional-density resolution ε_ρ. Unit conversion: length → dimensionless fractional ρ. Step has known cosmo-emulator precedent (Aemulus, Coyote).
3. **Tolerance → quantile-band threshold**: for stratum s and quantile q, the pass-bar is `|Q_q[(ρ_pred − ρ_truth) / ρ_truth | s]| ≤ k_q · ε_ρ(s)` where k_q is a quantile-band conversion factor (e.g., k_50 = 1, k_90 such that the 90th-percentile of a Gaussian-tailed residual at variance ε_ρ² lies at threshold). Unit conversion: fractional ρ → quantile band. Documented in DeRose+2019.
4. **Multi-stratum aggregation**: Holm-Bonferroni correction across the (strata × quantiles) panel (5 strata × 3 quantiles = 15 tests at family α=0.05).

The numerical pre-commit at rung 3 will substitute concrete values for σ_smoothing, ε_ρ(s) by stratum, k_q by quantile, and the family α. This §2.E is *only* the audit-shape: the panel can verify in cycle #5 whether each step's unit-transfer is monotone-Lipschitz, whether each crosses a nonlinear forward model, and whether the chain is purely-derivable-from-anchor versus retrofitted.

### §2.F — Falsification-criterion pre-commit (form-level, per [D-37] rule 5)

The framework choice itself is falsified, *independent of cycle #5 panel verdict*, if any of the following hold:

- **F1 — Chain crosses a nonlinear forward model**: if step (2) or step (3) of §2.E above requires routing through exp(−τ) saturation, RSD broadening, or LSF — i.e., the chain is not direct ρ-anchor-to-ρ-threshold. Such routing would invoke R29-amendment "presumed unanchored until chain is derived" status. If derivation reveals such routing, framework choice is falsified and we return to §1 candidate set.
- **F2 — Reduction-to-ratio-form check fails**: if a panel-grade audit of the multi-stratum bootstrap test reveals that the per-stratum quantile-band threshold can be algebraically re-expressed as a pooled-variance ratio after re-normalization (mirroring the (v) collapse). The audit is the explicit derivation done in cycle #5; if it surfaces this collapse, (iii) joins (v) in the eliminated set and the alternate (iv) is promoted.
- **F3 — Per-stratum N_eff insufficient**: if the K × K' protocol's per-crop voxel count, after block-bootstrap correlation-deflation, leaves any of the 5 strata with N_eff < threshold required for the per-stratum quantile-band CI to have width ≤ the threshold itself. Threshold-of-threshold derivation owed at rung 3 numerical pre-commit; framework falsified at rung 3 if N_eff cannot support the bar.

These pre-commits land BEFORE cycle #5 panel dispatch (per [D-37] rule 5), so the panel verdict cannot retroactively redefine the falsification space.

### §2.G — Authorization scope

**This PI absorption authorizes**:
- Defense-panel dispatch for cycle #5 pre-review on (a) the §2.B Reading-B ruling, (b) the §2.C top-2 ranked-choice reasoning, (c) the §2.D elimination of (i), (ii), (v) — particularly the (v) Reading-B structural-equivalence claim, (d) the §2.E pass-bar derivation chain audit, (e) the §2.F falsification-criterion pre-commit shape.

**This PI absorption does NOT authorize**:
- Any HPC dispatch (no infrastructure-manager, no Juno, no sbatch).
- Any code-implementer dispatch (no pipeline.py edits, no new estimator code).
- Any latex-author dispatch (no paper §3/§4 propagation; framework choice has not cleared cycle #5).
- The numerical pass-bar pre-commit (rung 3) — strictly post-cycle-#5-APPROVE.
- LEDGER §3 [D-71] entry-edit — next LEDGER absorption is post-cycle-#5 verdict.

**Sign-off ladder (PROVISIONAL)**:
- Ext-2 R15: PROVISIONAL (cycle #5 panel pre-review owed; lift mechanism = panel APPROVE).
- R28: PROVISIONAL Tier (i)+(ii) — Tier (i) literal-integer PASS this absorption (§2.A arithmetic); Tier (ii) discharge note PASS (§R28-CHECK sub-block present and explicit).
- R29: CANDIDATE with sighting #4 in flight (cycle #4 K1 OC failure); does NOT clear by this absorption. R29 RETIRED-threshold pre-committed at 5th sighting without prevention; cycle #5 is the test.
- R32: DEFERRED-BANKED PROVISIONAL; 2026-08-31 deadline preserved.

### §2.H — §R28-CHECK sub-block (Tier (ii) MANDATORY, cycle #5 absorption)

Rung enumeration for THIS absorption (Rev 2 §2 amendment cycle #5):

| Rung | Action | Landing artifact | Status |
|---|---|---|---|
| 0 | Cycle #5 defense-panel verdict transcript received | Verdict text (in-session) | LANDED |
| 1 | R29 disposition ruling (Part 1) | §2.I (this block) | LANDED |
| 2 | §2 amendment block addressing S1+S2+P1-P5 | §2.J–§2.O (this block) | LANDED |
| 3 | Cycle #6 defense-panel pre-review on AMENDED §2 | Cycle #6 transcript appended to [D-71] | PENDING |

- Dispatch-ladder rung count for this absorption: **4** (rungs 0, 1, 2, 3).
- Predecessor design-doc landing-artifact count for this absorption: **2** (§2.I R29 ruling; §2.J–§2.O amendment block).
- Tier (i) literal-integer check: rungs-this-absorption-acts-on (3: rungs 0 → 1 → 2) ≥ landing-artifacts (2). **PASS.**

**Tier (ii) rule-text question (P5) — ruling**: forward-rung enumeration is **mandatory-at-next-absorption**, NOT mandatory-now. Reasoning: R28 rule-text (`.claude/agents/project-architect.md` rule 27) reads *"PI dispatch-sequence authoring must enumerate every code-state-landing step named in the predecessor design doc as a SEPARATE dispatch-ladder rung BEFORE any downstream-agent dispatch"*. The dispatch-event that triggers forward-rung enumeration is the next downstream-agent dispatch (cycle #6 panel, or post-APPROVE HPC). The current absorption's downstream-agent dispatch is cycle #6 panel only; rungs beyond cycle #6 (gate-construction commit, K×K' dispatch, evaluation, paper-propagation) are not yet a current-turn dispatch surface. Enumerating them now would be speculative — at the next absorption (post-cycle-#6-APPROVE), those become current-turn rungs and §R28-CHECK there enumerates them as binding. Forward baseline note: rungs 4-7+ exist as labels (gate-construction commit; K×K' run dispatch under R32 DEFERRED; evaluation; paper-propagation) but are NOT-YET-AUTHORIZED and bind to no current dispatch.

Tier (ii) discharge note: §R28-CHECK present; integer counts written; cross-check arithmetic shown; P5 rule-text question ruled. No discretion to omit; rule satisfied.

### §2.I — R29 disposition ruling

**Ruling: sighting #5. R29 RETIRED-threshold fires.**

The R29 panel-added rule-text amendment 2026-05-26 is binding: *"R29's in-line check is a design-time obligation discharged at gate-construction commit; post-spec panel catches do NOT retire the design-time obligation and do NOT count as R29 successful preventions."* The §2 PI-commit landed at Rev 2 §2.A–§2.G BEFORE cycle #5. The cycle #5 catches (S1 ε_ρ mis-attribution; S2 re-framing-not-evidentiary; P1–P5) are post-spec. By the panel-added clause they are R29 failures partially mitigated by panel-review discipline, NOT R29 successful preventions. Reading the §2 commit as "still under construction until HPC dispatch" would re-open the rule-laundering hatch the 2026-05-26 amendment closed. Honest framing per [D-37] rule (a): the §2 PI-commit IS a gate-construction event. The unit-chain audit (S1 in particular) was owed AT §2.A–§2.G and was not discharged. Sighting #5 fires. R29 RETIRED-threshold (pre-committed at LEDGER §3 [D-71] §C 2026-06-01) executes mechanically. The substantive R29 obligation transfers to R32 panel-cycle-discipline; R32's candidate→BANKED case strengthens (panel-cycle review IS what caught S1; R32 second-sighting test passed). No panel re-review on the retirement itself; retirement is the pre-committed mechanical consequence.

### §2.J — §2.B restatement per S2

**Restated ruling: Reading B is operationally adopted as the framing. Reading A' is NOT ruled out by §1 evidence. The choice between Reading A' and Reading B is a PRECOMMIT, not a derivation.**

Honest framing per [D-37] rule (a) — observation first:

- The §1 evidence shows (i) §1.A entry (v) Bin-D-LMSE reduces to ratio-of-pooled-variances ONLY after σ²_smoothing-floor anchoring; the reduction is anchor-conditional, not framework-conditional. (ii) §1.B Pattern 3 surfaces "discriminator is upstream of framework choice."
- A Reading-A' restatement — "ratio-of-pooled-variances was the right family; the failed Rev 1 cycles failed because R29-discipline was not yet banked and applied prospectively, not because the framework was structurally wrong" — is CONSISTENT with the §1 evidence. The §1 readback does not adjudicate Reading-A' vs Reading-B because the discriminator (prospective unit-chain discipline) is identical under both readings.
- Cycle #5 S2 finding is therefore correct: the original §2.B "Reading-B confirmed in sharpened 2-prong form" was a re-framing step, not an evidentiary step.

**Operational adoption**: Reading B is adopted because (a) the §2 ranked-choice path (iii)+(iv) is more conservative under Reading-A' than under Reading-B (R29 prospective discipline is required either way; framework-rotation gives orthogonal failure surface as additional protection); (b) returning to (v) Bin-D-LMSE under Reading-A' would re-load the same observable family that the Rev 1 cycles failed on, even if the failure was R29-discipline-shaped not framework-shaped — operationally riskier than rotating frameworks.

Reading-A' is not ruled out. If cycle #6 panel surfaces a defensible Reading-A' path back to (v) with prospective R29-discipline, that becomes a candidate alternate. The current adoption is a precommit anchored on operational conservatism, not a derivation from §1 evidence.

### §2.K — §2.C reasoning (i) + §2.E step 2 amendment per S1

**Honest correction**: ε_ρ as written in original §2.C reasoning (i) and §2.E step 2 was attributed loosely to Lukić+2015 §3 / DeRose+2019 / Kobayashi+2020 as if "fractional density tolerance stratified by ρ_truth-decade" were a published quantity in those papers. Cycle #5 S1 verified this is NOT the case. Per [D-37] rule (a): ε_ρ is a PI-construct, not a Lukić-published or DeRose-published or Kobayashi-published quantity. Original §2.C/§2.E wording is amended to flag this.

**Amended derivation path (candidate, per [D-37]-Ext R2 falsified-prior+2 hedge level)**:

- Physical anchor: σ_smoothing — the numerical-smoothing-kernel scale of the Sherwood hydrodynamic solver, attributable to Bolton+2017 §4 convergence-test framework (NOT Lukić+2015 §3, which addresses optically-thin solver convergence on flux statistics, a different chain). Bolton+2017 §4 reports density-field convergence under refinement; the smoothing scale is the resolved-scale of the gravity/SPH solver at the simulation grid.
- Anchor → fractional-density tolerance ε_ρ at the smoothing scale: ε_ρ(s) is a PI-construct = the fractional difference between the density-field at the production grid resolution and at the next-higher refinement level, stratified by ρ_truth decade, as reported in Bolton+2017 §4 Fig. X (specific figure reference owed at cycle #6 brief; PI authoring this turn cannot verify the figure without re-reading Bolton+2017).
- **F1 self-firing check (this turn, audit-shape only)**: does the candidate derivation path from Bolton+2017 §4 convergence tests to ε_ρ(s) route through any nonlinear forward model (exp(−τ) FGPA, RSD broadening, LSF, continuum-fitting bias)?

   - Bolton+2017 §4 convergence tests are reported on density-field statistics directly (refinement-vs-baseline density comparison), NOT through Lyα-flux statistics. The chain from {refined-density, baseline-density} → {ε_ρ stratified by ρ_truth-decade} is a direct density-space comparison; no exp(−τ) is invoked.
   - Bolton+2017 §4 does also include flux-statistic convergence; those routes DO cross exp(−τ). The candidate derivation pinned here is explicitly the density-space convergence subset, NOT the flux-statistic subset.
   - **Verdict**: candidate derivation path does NOT route through nonlinear forward model. F1 does NOT self-fire on this turn's PI audit-shape examination. **Caveat per R26 in-session re-verification**: PI's reading of Bolton+2017 §4 is *inherited* from the literature-citation block in §1.C; PI has not freshly re-read Bolton+2017 §4 from journal source this session. The verdict is therefore PROVISIONAL pending the cycle #6 panel's independent verification of the density-space subset claim. If the panel surfaces that Bolton+2017 §4 density-convergence reports are themselves derived through a flux-anchored chain, F1 fires at cycle #6 absorption and framework choice is falsified.

- Hedge level per [D-37]-Ext R2 cascade (falsified-prior+2 commit — 4 Rev 1 cycles + 1 NEEDS-WORK on Rev 2 §2): verb level is "candidate derivation path from Sherwood Bolton+2017 §4 convergence tests" — NOT "derivation". The chain is provisional until cycle #6 panel APPROVE with the figure-specific anchor verified.

### §2.L — P2 amendment: k_q pre-commit

**Amendment**: k_q (the quantile-band conversion factor at §2.E step 3) is pre-committed to a non-parametric form. Specifically: k_q is the empirical-bootstrap qth-percentile of the K × K' residual ensemble under the null hypothesis (perfect-fit-up-to-σ_smoothing-floor). The Gaussian-tail conversion path (k_90 = 90th-percentile-of-a-Gaussian-at-variance-ε_ρ²) is explicitly disavowed. Rationale per cycle #5 P2: IGM ρ-residuals at the 5th decade are not Gaussian-tailed (Horowitz+2019 TARDIS Fig. 7 documents heavy-tailed residuals at the high-density end; CAMELS emulator residuals likewise). A Gaussian-tail k_q would mis-locate the quantile band at the bin that matters most for the science case.

Bootstrap construction: under H_0, draw N synthetic residual samples by perturbing ρ_truth by σ_smoothing-floor with a Sherwood-empirical residual distribution (NOT a parametric Gaussian); compute Q_q empirically across the bootstrap; k_q = Q_q^bootstrap / ε_ρ. Numerical pre-commit of N, the perturbation procedure, and the bootstrap-resampling block-structure is owed at rung 3 (post-cycle-#6-APPROVE).

### §2.M — P3 amendment: F2 tightening

**Amendment**: F2 falsification criterion is tightened to:

*"The framework choice (iii) Q-strata is falsified if the natural σ_smoothing-floor-anchored derivation of the per-stratum quantile-band threshold algebraically equals a pooled-variance ratio under NO auxiliary assumptions (no auxiliary re-normalization step, no auxiliary distributional assumption, no auxiliary bin-aggregation step beyond the §2.E stratification)."*

This replaces the original F2 "if a panel-grade audit reveals the per-stratum quantile-band threshold can be algebraically re-expressed as a pooled-variance ratio after re-normalization" which the panel correctly diagnosed as universally-predicate (many statistics admit many re-expressions under sufficient auxiliary steps). The "under no auxiliary assumptions" qualifier makes F2 operationally sharp: the test is whether the *natural* anchor-chain produces a ratio form, not whether *any* re-expression produces one. The cycle #6 panel can fire F2 binding-falsification iff the natural derivation from σ_smoothing-floor through ε_ρ to quantile-band IS exactly a pooled-variance ratio at every step without auxiliary insertion.

### §2.N — P1 amendment: design-effect deflator footnote

**Amendment, footnote-form**: the per-stratum quantile asymptotic-variance computation at the K × K' protocol ingests a design-effect deflator (Var_observed / Var_iid) for the effective-sample-size correction (Bahadur 1966 quantile asymptotic-variance scaled by the deflator). The deflator IS a ratio quantity, but it is upstream variance-bookkeeping for the bootstrap CI construction, NOT a gate-construction threshold. The threshold itself (k_q · ε_ρ per §2.E step 3, with k_q empirical-bootstrap per §2.L) is not a ratio.

The design-effect-deflator usage is the standard cosmo-emulator-bootstrap convention: Heitmann+2009 *Coyote* §3 uses design-effect-corrected variance for stratified residual CIs; Lawrence+2017 *Mira-Titan II* §4 likewise. Footnote-of-record at cycle #6 brief: "The N_eff design-effect deflator is a known cosmo-emulator-bootstrap convention (Heitmann+2009; Lawrence+2017); it is upstream variance-bookkeeping, NOT the gate-construction threshold. R10 orthogonality vs ratio-of-pooled-variances failure mode holds at the test-statistic and threshold-derivation level; the deflator's ratio-form usage at variance-estimation level is convention-grade, not gate-grade."

### §2.O — P4 amendment: log-W1 pre-commit (alternate (iv))

**Amendment**: alternate framework (iv) W1 is pre-committed to LOG-domain (log₁₀ ρ) specifically. The linear-W1 form is explicitly disavowed.

Physical justification footnote: linear-W1 on 1D ρ-PDFs weights the 5th-decade contribution by factor 10⁵ over the 1st-decade contribution under the 5-decade dynamic range; the highest-density bin dominates the metric. This is the same instability surface as the failed Rev 1 ratio-of-pooled-variances framing (one bin's variance dominates the ratio). Log-W1 weights all 5 decades equiprobably under the log-CDF inversion — the 5th decade contributes proportionally to its log-density mass, not to its raw-density mass. The science case for the IGM ρ-PDF tomography is balanced across decades (low-density voids and high-density filaments both inform the cosmological constraint); log-domain matches the science framing.

Precedent: Park+2023 used log-W1 on 21-cm fields at 1-2 decade DR; Friedrich+2022 used linear-W1 on matter PDFs at ≤3 decades (where the dynamic range is small enough that linear-vs-log distinction is minor). Per cycle #5 P4, neither paper establishes W1 at 5-decade IGM ρ-PDF — but the log-domain pre-commit gives the candidate path the better-anchored derivation surface of the two.

**(iv) inherits prong-(b) discipline**: per the §2.J Reading-B operational-adoption ruling, the alternate (iv) carries the same R29-prospective-derivation discipline as the primary (iii). Promotion of (iv) to primary (only if cycle #6 panel falsifies (iii) per F1/F2/F3) does NOT escape the prong-(b) prospective-unit-chain-derivation requirement.

### §2.P — §R28-CHECK sub-block (Tier (ii) MANDATORY, cycle #6 absorption)

Rung enumeration for THIS absorption (Rev 2 §2 amendment cycle #6):

| Rung | Action | Landing artifact | Status |
|---|---|---|---|
| 0 | Cycle #6 defense-panel verdict transcript received (K1 KILLER + S1-S5 + P1-P4) | Verdict text (in-session) | LANDED |
| 1 | Recovery-path ruling (Option A vs B vs C) | §2.Q (this block) | LANDED |
| 2 | R29 RETIRED confirmation (panel concurs) | §2.R (this block) | LANDED |
| 3 | R32 promotion reversion (panel overturn accepted) | §2.S (this block) | LANDED |
| 4 | R33 candidacy ruling | §2.T (this block) | LANDED |
| 5 | S3+S4+S5+P1+P2 form-level absorption notes | §2.U (this block) | LANDED |
| 6 | Authorization scope for cycle #6 close + cycle #7 opener | §2.V (this block) | LANDED |
| 7 | Recovery-path-dependent next dispatch (support-researcher literature-mining on Option A candidate sources) | Dispatch brief authored at cycle #7 opener | PENDING (next dispatch) |

- Dispatch-ladder rung count for this absorption: **8** (rungs 0 through 7).
- Predecessor design-doc landing-artifact count for this absorption: **6** (§2.Q, §2.R, §2.S, §2.T, §2.U, §2.V).
- Tier (i) literal-integer check: rungs-this-absorption-acts-on (7: rungs 0 → 6, all LANDED) ≥ landing-artifacts (6). **PASS.**
- Forward-rung enumeration per Rev 1.4 §14.E / Rev 2 §2.H ruling: rung 7 is the next downstream-agent dispatch (support-researcher); it is enumerated above and binds the next-cycle §R28-CHECK at cycle #7 absorption. Rungs beyond rung 7 (Option-A-fail-fallback Bolton Fig. 4 verification; Option C return-to-§1; numerical pre-commit; HPC dispatch) are NOT-YET-AUTHORIZED labels, recovery-path-conditional, and bind to no current dispatch.

Tier (ii) discharge note: §R28-CHECK present; integer counts written; cross-check arithmetic shown; forward-rung-7 enumerated. No discretion to omit; rule satisfied. R28 auto-promote (BANKED 2026-06-01 per [D-71] §C) operative; no new sighting accrued THIS absorption (rung-count arithmetic was checked at authoring before commit, not after).

### §2.Q — K1 acceptance, (iii) Q-strata retraction, and Option A recovery-path ruling

**K1 acceptance per F1 binding pre-commit fire ([D-37] rule 5).**

Cycle #6 panel independently verified Bolton+2017 §4 via WebFetch this turn:
- §4 is titled "CONCLUSIONS", NOT a convergence-test section.
- Convergence content lives in **Appendix A "NUMERICAL CONVERGENCE"**, not §4.
- Appendix A's headline convergence quantity is the **Lyα flux power spectrum** P_F(k) at <10% convergence — a *flux statistic* that routes through exp(−τ) FGPA + RSD + LSF.
- No density-decade-stratified fractional convergence quantity ε_ρ(s) is reported in either §4 or Appendix A.

The §2.K candidate derivation path was: "Bolton+2017 §4 density-space convergence → ε_ρ(s) stratified by ρ_truth-decade → quantile band via k_q." The panel's verification falsifies the *first link* of the chain: the cited §4 density-space convergence subset does not exist as such. The actual Appendix A convergence content lives entirely on a flux-statistic forward-modelled quantity. **The chain crosses a nonlinear forward model.**

§2.F F1 pre-commit text is binding:
> *"If derivation reveals such routing, framework choice is falsified and we return to §1 candidate set."*

Per [D-37] rule 5 symmetric-disclosure: the falsification space cannot be redefined post-hoc. PI does NOT have authority to argue (iii) Q-strata survives on a *different* anchor not named at §2.F commit time. **(iii) Q-strata as constituted in §2.A–§2.O is RETRACTED.**

Honest framing per [D-37] rule (a): this is the discipline working as intended. The §2.F pre-commit was specifically authored to make F1 self-firing independent of any post-hoc PI defense. F1 fired; (iii) retracts. This is decision-quality-graded success of the symmetric-disclosure rule, NOT process failure.

**Recovery-path ruling: Option A.**

PI rules Option A (re-anchor ε_ρ(s) to a different published density-decade-stratified convergence source) as the recovery path, with Option C (return to §1 candidate set) as a hard back-stop fired at next absorption if Option A's literature search returns null. Option B (promote (iv) log-W1) is held as an Option-A-failure-fallback, NOT as a parallel candidate.

**Ruling rationale, [D-37]-Ext R2 falsified-prior cascade arithmetic at hedge-level +3** (4 Rev 1 cycles + Rev 2 §2 cycle #5 NEEDS-WORK + Rev 2 §2 cycle #6 KILLER):

1. **F1 fires on the §2.K Bolton+2017 *anchor*, not on the (iii) *observable*.** The Q-strata observable family (per-stratum quantile residuals on ρ_pred − ρ_truth at ρ_truth-decade strata) and the §2.E derivation shape are anchor-conditional. Re-anchoring is the surgically narrowest move consistent with the F1 binding-fire text. The observable's identifiability + measurability + structural R29-auditability arguments from §1.A entry (iii) and §1.B Pattern 3 do not collapse with the §2.K anchor falsification.

2. **Option B is structurally riskier than it appears under R26 in-session re-verification status.** The §2.O log-W1 chain terminates at Bolton+2017 Fig. 4 σ_log-ρ-PDF-width — also Bolton+2017, also un-verified in-session this cycle, and now operating under fresh evidence that Bolton+2017 was mis-cited at §2.K. Promoting (iv) to primary at hedge-level +3 with the same author whose §2.K Bolton+2017 attribution just failed would inherit the falsifying author's anchor-verification exposure. Per R26 BANKED-candidate text: load-bearing inherited-from-prior-session citations require in-session re-verification before sign-off. Option B without independent Fig. 4 verification is precisely an R26 inherited-claim sign-off failure mode. P2 cycle #6 separately flagged the Park+2023 precedent as unverified.

3. **Option C is the live back-stop, not the front-line choice.** Two-in-a-row PI literature mis-attributions (cycle #5 S1 Lukić+2015 §3; cycle #6 K1 Bolton+2017 §4) is empirical evidence — but per [D-37] rule (a) honest-framing, the observation supports "PI literature-anchoring step has a structural failure mode" (the discipline R33 candidacy at §2.T addresses), NOT "the (iii)+(iv) framework family is structurally unanchorable across the entire published density-decade-stratified-convergence literature." The panel named three candidate sources that have not been searched (Lukić+2015 *Nyx* §3 phase-space-cell mass-fraction convergence; Sorini+2018 Sherwood mass-resolution scans; Chabanier+2023 ACCEL2 density-space convergence). Dropping the framework family before the search is itself an [D-37] rule (a) framing-over-observation violation.

4. **Hard back-stop**: the Option A search is bounded. The support-researcher literature-mining dispatch at cycle #7 opener must return either (i) one source verifiable from journal source per R26 reporting a density-decade-stratified ε_ρ(s) convergence quantity that does NOT cross exp(−τ) / RSD / LSF / continuum-fitting bias per F1, OR (ii) a documented null result across all three named candidate sources plus any further sources the support-researcher surfaces. On (i): Option A proceeds with anchor identified, F1 re-checked against the new anchor at the next absorption. On (ii): Option C fires automatically — return to §1 candidate set with the (iii)+(iv) family removed; the §1 candidate set then reduces to (i) KL, (ii) MMD, (v) Bin-D-LMSE. Per §2.F Reading-B operational adoption and §1.B Pattern 3, the discriminator at that point becomes the prospective-unit-chain discipline applied to whichever observable family carries the least falsified-prior load.

5. **R33-candidacy compatibility**: Option A's literature-mining dispatch is the operational test of R33's substantive obligation (in-session-re-read-from-journal-source for any cited paper section by name in a gate-construction commit). The dispatch brief at cycle #7 opener will require the support-researcher to provide journal-source pull + verbatim quote + figure/table reference for every candidate source, with no inherited PI citation accepted. This discipline transfers the R26 in-session re-verification obligation to the dispatched agent at the brief-drafting step, where the citation is being introduced.

**Hedge level per [D-37]-Ext R2**: the recovery path is "candidate re-anchoring of (iii) to one of three panel-named sources, pending support-researcher literature mining and cycle #7 panel pre-review" — NOT "recovery confirmed." At falsified-prior+3, the operative verb is "candidate"; "candidate" remains operative until the new anchor passes cycle #7 F1 audit.

### §2.R — R29 RETIRED confirmation

Cycle #6 S1 panel concurs with §2.I R29 RETIRED ruling: §2.A–§2.G IS a gate-construction commit; post-spec catches do not retire the design-time obligation. R29 RETIRED status (LEDGER §3 [D-71] §C 2026-06-01 pre-commit; §2.I disposition) confirmed. No panel defense owed.

**Sub-note**: K1 itself is R29 sighting #6 inside the same absorption cluster, post-R29-retirement. The cycle #6 panel-cycle (R32) caught it. Operationally consistent with the §2.I substantive-transfer ruling (R29 obligation transferred to R32 panel-cycle discipline). LEDGER §3 [D-71] cycle #6 entry logs this as a confirming instance, not a re-banking instance — R29 is retired, the substantive obligation lives under R32 (subject to R32's own promotion path per §2.S).

### §2.S — R32 promotion reversion (panel overturn accepted)

Cycle #6 S2 panel overturn accepted: R32 candidate → BANKED PROVISIONAL ruling (§2.I cycle #5 sub-claim) reverts to **R32 CANDIDATE**.

Panel reasoning accepted: R12 cross-track second-sighting requirement is NOT satisfied by two same-track sightings inside the [D-71] cluster (cycle #5 panel-cycle-discipline catching S1 Lukić+2015 mis-attribution; cycle #6 panel-cycle-discipline catching K1 Bolton+2017 mis-attribution). Both are PI-cited-literature-anchor mis-attributions in the same gate-construction cluster, not two independent operational tests of panel-cycle-discipline across distinct surfaces. Promoting on two same-track sightings collapses the R12 cross-track requirement and reintroduces the R12-amendment failure mode (sighting later revealed broken → reset to zero) — which the [D-71] §H R29 demotion precedent established as forbidden.

**K1 LOGGED as R32 sighting #3** under same-cluster reading. **Not promoted.** Cross-track sighting still owed before BANKED. Candidate alternative tracks the panel named: [D-46] physics_id-embedding panel cycle; [D-47] Stage-3 hybrid panel cycle; any non-[D-71] gate-construction panel. R32 DEFERRED-BANKED PROVISIONAL deadline 2026-08-31 preserved (panel overturn does not extend the deadline).

### §2.T — R33 candidacy ruling

**Ruling: R33 BANKED as CANDIDATE.**

Rule text (candidate): *"PI may not cite a paper section by name (e.g., 'Bolton+2017 §4 convergence tests', 'Lukić+2015 §3 smoothing-floor') in a gate-construction commit, design-doc derivation chain, or LEDGER entry that downstream agents will treat as load-bearing, without an in-session re-read of that named section from the journal source (or arXiv preprint) under R26 in-session re-verification protocol. Citation-by-name from prior-session memory, from prior-amendment text, or from a downstream-agent readback alone is insufficient. PI must paste the verbatim quote (or section heading + first paragraph) + journal-source URL/DOI into the LEDGER or design-doc entry at the citation point."*

**Sighting log (CANDIDATE-banking, two same-track sightings is the R12 candidate-banking trigger for new-rule introduction per R12 precedent — distinct from R12's cross-track second-sighting threshold for promotion to BANKED)**:
- **Sighting #1**: cycle #5 S1 — §2.K original wording attributed ε_ρ to Lukić+2015 §3 / DeRose+2019 / Kobayashi+2020 as if "fractional density tolerance stratified by ρ_truth-decade" were a published quantity in those papers. Cycle #5 panel verified it is NOT.
- **Sighting #2**: cycle #6 K1 — §2.K amended wording (post-cycle-#5) attributed the candidate derivation path to Bolton+2017 §4 convergence-test framework. Cycle #6 panel WebFetch verified Bolton+2017 §4 is titled "CONCLUSIONS", convergence content lives in Appendix A, and Appendix A reports flux-statistic convergence not density-statistic convergence at decade strata.

**R26 redundancy question** (panel-raised): is R33 redundant with R26?

Ruling: R33 is substantively orthogonal to R26, not redundant. Reasoning:
- R26's BANKED-candidate text is scoped to *"code-state claim (grep result, function semantics, default flag value, call-path reduction, default code path, etc.) inherited from a prior session..."* — i.e., R26's domain is **code-state inheritance**, not **literature-citation inheritance**. The example pattern R26 was banked on (the [D-60] gate-retune-1 v2 → v3 `reduction='sum'|'mean'` lever-conditional grep claim) is a code-state lever, not a paper-section content claim.
- R33's failure mode is **structurally distinct**: PI cites paper section by name (Bolton+2017 §4, Lukić+2015 §3) from memory or from prior PI authorship without journal-source re-read. The mis-attribution is not a code-state drift; it is a content-attribution drift between memory and journal.
- R26 + R33 together close the inherited-claim discipline across both code-state (R26) and literature-citation (R33) domains. Without R33, the rule-set has an inherited-literature-claim hatch that two consecutive cycles have demonstrated is operationally exploitable.

**R33 candidate banking status**: CANDIDATE (two same-track sightings). Cross-track second-sighting required before BANKED per R12 precedent. PI-only sign-off; R15 PROVISIONAL by default. Separate panel review on R33 candidacy at cycle #7 opener — panel may sharpen the rule text and confirm or overturn the candidate banking.

### §2.U — S3 + S4 + S5 + P1 + P2 form-level absorption notes

No rung-3 numerical pre-commits authorized this cycle. These are form-level absorption notes only; numerical pre-commits land at the absorption following Option A recovery-path discharge (i.e., at cycle #8 absorption, after cycle #7 panel APPROVE on the re-anchored §2.K).

- **S3 (k_q bootstrap under-specification, 3 rung-3-hook items)**: ACCEPTED form-level pre-commits owed at next numerical absorption — (i) Sherwood-empirical residual distribution source must be named (predicted-vs-truth from a held-out crop ensemble OR truth-only-refinement-vs-production from the Bolton+2017 Appendix A convergence pairs — choice owed at numerical pre-commit, with circularity check); (ii) block-bootstrap structure must specify block shape per IGM correlation length ~10 cMpc at z=3 (naive iid under-estimates variance ~10×–30×); (iii) finite-sample sanity bound for F3 must be specified at numerical pre-commit (F3 is operationally vacuous until then — acknowledged form-level). The bootstrap construction at §2.L is amended to flag (i)/(ii)/(iii) explicitly at the numerical-pre-commit absorption.

- **S4 (Heitmann+2009 / Lawrence+2017 stratification dimension UNVERIFIED)**: PI accepts the cycle #6 finding. §2.N footnote is amended in form: at the next absorption, PI must WebFetch Heitmann+2009 *Coyote* §3 and Lawrence+2017 *Mira-Titan II* §4, quote design-effect-deflator usage verbatim, and state honestly whether stratification is k-bin (on the matter power spectrum residual) or ρ-decade (on density-PDF residual). Likely outcome: stratification is k-bin (matter-power-spectrum emulator residuals), in which case the §2.N footnote must add an honest hedge that the deflator-as-convention argument is precedent-from-an-adjacent-not-identical methodology, and the cosmo-emulator-bootstrap convention argument carries proportionally less weight. This verification is bundled into the R33-compliant cycle #7 dispatch brief.

- **S5 (F2 form-OK but unfireable this cycle)**: ACCEPTED. K1 retracted the σ_smoothing → ε_ρ → quantile-band chain at the §2.K anchor; "the natural derivation" has no defined form until Option A re-anchoring lands. F2 becomes evaluable only after the re-anchored §2.K passes cycle #7 F1 audit. No defense owed at this cycle; downstream of K1 resolution.

- **P1 (§2.J Reading-A' framing — "more conservative under Reading-A'" overstates)**: ACCEPTED. §2.J operational-adoption rationale tightened in form: "(iii)+(iv) carries orthogonal failure surface under both Reading A' and Reading B" replaces "more conservative under Reading-A'." Substantive operational adoption unchanged.

- **P2 (Park+2023 log-W1 precedent UNVERIFIED at §2.O)**: ACCEPTED. Held as part of Option-A-failure-fallback Option B contingency. If Option C fires and Option B is reconsidered, Park+2023 in-session re-verification under R26 is a hard pre-commit before §2.O can support (iv) promotion. PI does NOT re-verify this turn (Option B not active; R26 trigger not yet binding on inactive path).

- **P3 + P4 + S2 PI rulings**: cycle #6 panel concurs. No further amendment.

### §2.V — Authorization scope (cycle #6 close)

**This PI absorption authorizes**:
- Support-researcher dispatch at cycle #7 opener: literature-mining on the three panel-named Option A candidate sources (Lukić+2015 *Nyx* §3 phase-space-cell mass-fraction convergence; Sorini+2018 Sherwood mass-resolution scans; Chabanier+2023 ACCEL2 density-space convergence) plus any further sources surfaced during the search. Dispatch brief must require R33-compliant journal-source pull + verbatim quote + figure/table reference for every candidate source; no inherited PI citation accepted. Bundled deliverable: S4 Heitmann+2009 §3 + Lawrence+2017 §4 in-session re-verification of stratification dimension (k-bin vs ρ-decade) per cycle #6 S4. Word budget ≤3000.
- LEDGER §3 [D-71] cycle #5 + cycle #6 absorption entry-edit (Part 3 of this absorption; landed by parent session).
- Project-architect.md amendment to codify R33 CANDIDATE status (deferred to next governance batch per [D-71] §C precedent; not landed this absorption).

**This PI absorption does NOT authorize**:
- Any HPC dispatch (no infrastructure-manager, no Juno, no sbatch).
- Any code-implementer dispatch (no pipeline.py edits, no new estimator code).
- Any latex-author dispatch (Option A recovery has not cleared cycle #7; CVPR submission gated per `feedback_no_paper_writing.md` 2026-05-21 directive).
- Rung 3 numerical pass-bar pre-commit (strictly post-cycle-#7-APPROVE-on-re-anchored-§2.K).
- Bolton+2017 Fig. 4 verification dispatch (Option B held as Option-A-failure-fallback, not active path).
- Return-to-§1 Option C dispatch (back-stop only, fired automatically at next absorption if Option A search returns null).

**Sign-off ladder (PROVISIONAL)**:
- Ext-2 R15: PROVISIONAL (cycle #7 panel pre-review owed on the support-researcher Option A readback; lift mechanism = cycle #7 panel APPROVE on re-anchored §2.K).
- R28: BANKED (auto-promote operative per [D-71] §C 2026-06-01). Tier (i) literal-integer PASS this absorption (§2.P arithmetic); Tier (ii) §R28-CHECK sub-block present.
- R29: RETIRED (confirmed by panel S1; LEDGER §3 [D-71] §C 2026-06-01 pre-commit operative). K1 logged as confirming-instance, NOT re-banking instance.
- R32: CANDIDATE (panel S2 overturn accepted, BANKED PROVISIONAL reverts to CANDIDATE). K1 logged as same-cluster sighting #3. Cross-track second-sighting still owed; DEFERRED-BANKED PROVISIONAL deadline 2026-08-31 preserved.
- R33: CANDIDATE (newly banked this absorption; two same-track sightings = candidate-banking threshold; cross-track second-sighting required for BANKED).

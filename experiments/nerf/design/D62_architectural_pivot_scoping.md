# [D-62] Architectural Pivot — Scoping Memo

## Status

- **PROVISIONAL** throughout. This is a research-mode scoping survey, NOT a design doc.
- **Activated 2026-05-23** per defense-panel out-of-scope flag on [D-53] (commit `3074596`): the [D-62] stop-gate destination was undefined in [D-53]; scoping this BEFORE (b) dispatch ensures the R-b outcome of [D-53] has a concrete next step rather than a panic-design dispatch on a fresh empirical R-b.
- Author: support-researcher per parent-PI scoping commission.
- **No candidate pre-ranked.** Output is research-mode survey for PI's later design-stage commission if [D-53] candidate (b) fails first-binding step-200 R-b (or if all 3 [D-53] candidates exhaust without rescue).
- Intended audience: defense-panel scoping review at [D-62] activation gate.

## Scope definition

[D-62] activates **IF** [D-53] also fails — specifically: (i) the first [D-53] candidate dispatched (panel-bound = (b)) R-b at step 200, OR (ii) all 3 [D-53] candidates exhaust without rescue. [D-62] scope = **architectural intervention**: target the **model-side capacity / inductive-bias / reconstruction-target-class**, NOT loss-side. Loss-side is the closed class of [D-60] L1 (in-loss-function) ∪ [D-53] (supervision-target-redesign).

## Candidate landscape (5 candidates, research-mode survey)

---

### Candidate 1 — Different model architecture (transformer / attention / convolutional)

**Mechanism prediction (1 paragraph):** the current Voigt-integrator + MLP backbone treats each sightline ray as an independent forward pass through a coordinate-conditioned MLP. The variance collapse pathology may be **inductive-bias-coupled**: a coordinate MLP with sinusoidal positional encoding has limited capacity to represent the spatial-correlation structure of the IGM density field along the line-of-sight (the relevant scale is the Jeans length, ~100 kpc, ~1.5e-3 of unit-box; many MLP encodings under-resolve this). A transformer with self-attention along the ray (or across rays in a sightline-batch) could capture sightline-coherent structure that the MLP smooths over; this may break the variance-collapsed basin if the basin is partly an MLP-smoothing artifact. Convolutional discriminator (GAN-style) or VAE adversarial / reconstruction-prior architectures could enforce distributional realism on the density field as an architectural prior, not as a loss-side regularizer.

**Literature support:** Cabayol-García+ 2023 (`arXiv:2305.19064`) uses **transformer for IGM emulation** (P_1D regression from cosmology). Maitra/LyαNNA 2023 (`arXiv:2311.02167`) uses CNN+attention hybrid for inference. **No transformer-or-CNN IGM reconstruction (3-D field output) precedent** this researcher could locate — both cited works are summary-statistic regression, not 3-D field reconstruction. Honest framing per [D-37] rule (a): the architectural-trend support is for emulation/inference, not reconstruction; the inheritance claim is **weaker than apparent**.

**Implementation complexity:** **architectural-rewrite**. Requires replacing `src/models/nerf.py` MLP backbone with transformer/attention module; differentiable Voigt integrator wrapper preserved. Estimate 1-2 sprint cycles (refactor + estimator equivalence re-certification).

**Pre-committed falsification criterion:** same step-200 R-b pattern (`var_pf_band_ratio < 1e-3` at step 200 on P1-T1 smoke) closes this candidate, routing to candidate 2 or 3 per panel re-review.

---

### Candidate 2 — Different reconstruction target class (move outside flux-domain supervision)

**Mechanism prediction (1 paragraph):** the entire [D-53] candidate trio (a/b/c) supervises in flux-domain (or flux-derived) space because the **observable IS the flux**. If variance collapse is fundamentally a flux-domain-supervision pathology (e.g., information-theoretic bottleneck: P_F + PDF do not contain enough information to constrain the 3-D density field uniquely, so the model converges on the maximum-likelihood-under-the-loss constant solution), then no loss-side or flux-supervision-target-side intervention can rescue it. **Honest framing per [D-37] rule (a) (critical caveat):** direct density-field supervision **may not be observationally admissible** — the density field is not directly observed in Lyα; only the flux-after-Voigt-integration is. Surfacing density-field supervision as a candidate is honest-bounded: it would only be admissible in a **simulation-pretraining + flux-finetuning** regime where simulation density-fields are used for pretraining only. The mechanism prediction is then: pretraining on density-field MSE provides a strong field-realism prior that the flux-domain finetuning cannot collapse to constant. Inference-machine reformulation à la Maitra/LyαNNA 2023 is **inference-not-reconstruction** and would change the project's scientific framing — surface this as a scoping option but flag the framing shift.

**Literature support:** simulation-pretraining is standard in computer-vision (ImageNet-pretrain → task-finetune), but **no IGM-NeRF density-field pretraining precedent** this researcher could locate. Maitra/LyαNNA 2023 inference reformulation is a different scientific task (parameter inference, not field reconstruction). Honest framing: **literature support is thin** — this candidate is a class-of-first-tests with weak external anchor.

**Implementation complexity:** **pipeline-refactor**. Requires (i) new pretraining stage with density-field MSE loss on simulation data; (ii) finetune stage with current flux supervision; (iii) handoff-state management. Estimate 1 sprint cycle.

**Pre-committed falsification criterion:** same step-200 R-b pattern on the **finetune stage** (pretraining stage is supervised on direct density-field MSE and is expected to converge trivially; the test is whether finetuning resists collapse). If finetuning collapses by step 200, candidate fails.

---

### Candidate 3 — Hybrid forward-model + ML (residual learning on physics prior)

**Mechanism prediction (1 paragraph):** keep the differentiable Voigt integrator and the current flux-domain supervision, but **constrain the MLP to learn only residuals from a physics-based prior** (e.g., Hui-Gnedin 1997 analytic fluctuating Gunn-Peterson approximation, fGPA, which predicts density ↔ τ via a closed-form power-law). The MLP output is then `ρ = ρ_fGPA(δ, z) × (1 + MLP_residual(coords))` where MLP_residual is initialized small. This **architecturally enforces a non-constant prior** on the density field: the variance collapse basin is excluded by construction because ρ_fGPA already varies spatially per the input δ-field. Alternative: flow-based generative model (Normalizing Flow) for the density field with the Voigt integrator as a fixed forward operator — flow has a Jacobian-determinant constraint that prevents collapse to a delta-distribution (zero-variance).

**Literature support:** fGPA-based density-field initialization is **standard in IGM forward-modeling** (Hui-Gnedin 1997, used in TARDIS baseline Horowitz+ 2019); the residual-MLP-on-fGPA-prior pattern is novel for IGM-NeRF but **inherits the residual-learning success in ResNet / NeRF-W (residual NeRFs)**. Normalizing-flow density-field generative models exist in cosmology (Rouhiainen+ 2021 for matter density), not specifically for IGM. Honest framing per [D-37] rule (a): the inheritance from ResNet / NeRF-W is structural (residual learning generally works); the application to IGM is novel.

**Implementation complexity:** **pipeline-refactor + small architectural change**. Requires (i) fGPA prior module (`src/models/fgpa_prior.py`, ~100 lines, analytic); (ii) residual-MLP wiring in `nerf.py`; (iii) initialization scheme. Estimate 1 sprint cycle.

**Pre-committed falsification criterion:** same step-200 R-b. Additional candidate-3-specific criterion: if the MLP_residual converges to ≈ 0 (the prior dominates and the MLP contributes nothing), this is a **silent-MLP** failure — close as R-b regardless of variance-collapse number, since the architecture has degenerated to fGPA alone.

---

### Candidate 4 — Microbatch / batch-scale ablation (NOT strictly architectural)

**Mechanism prediction (1 paragraph):** panel-flagged as out-of-L1-scope: is the variance-collapse pathology **microbatch-coupled**? The L1 sequence ran at microbatch=256; if the per-mode P_F variance estimator is biased at small batch sizes (small-N sample variance underestimates true variance, biasing the model toward constant-flux solutions which match the underestimated low-variance target), then microbatch=1024 or full-batch training could shift the loss-surface geometry. This is **load-bearing diagnostic before committing to architectural pivots 1-3**: if microbatch=1024 rescues, [D-62] resolves without architectural rewrite.

**Literature support:** small-batch P_F estimator bias is a known cosmology-statistics issue (the ensemble-average P_F is variance-biased at N < ~10³ sightlines; Iršič+2017 §4 systematic budget). **No direct ML-training precedent** for this being the IGM-NeRF-class pathology, but the cosmology-statistics support is solid.

**Implementation complexity:** **loss-module-only / pipeline-config-only**. Requires only changing microbatch flag in submission script. Estimate <1 day; could run as smoke before candidate 1-3 dispatch.

**Pre-committed falsification criterion:** same step-200 R-b at microbatch=1024 closes this candidate (architectural pivot 1-3 still needed). Candidate-4 PASS at step 200 (`var_pf_band_ratio > 1e-3`) is **PROVISIONAL** and gated on step-1000 BINDING per [D-53] §"Stop-gate criteria" convention.

---

### Candidate 5 — Physics-variant ablation (P2/P3/P4, NOT strictly architectural)

**Mechanism prediction (1 paragraph):** panel-flagged: P1 physics-id may itself contribute to the diffuse-bin imbalance. P1 corresponds to a specific (γ, T₀) thermal-state combination in the Sherwood suite; P2/P3/P4 vary these parameters. If the variance collapse is P1-physics-specific (e.g., the diffuse-bin distribution shape is more pathological at P1's thermal state than at P2's), then P2/P3/P4 smoke could rescue without architectural change. This is the **same caveat-class as candidate 4**: load-bearing diagnostic before architectural commit.

**Literature support:** physics-variation-driven loss-surface differences are well-documented in cosmology emulation literature (Cabayol-García+ 2023 Table 2 cross-physics emulator performance); not specifically for IGM-NeRF reconstruction failure modes. Honest framing: the support is for emulator-output-variation, not failure-mode-variation; the inheritance is **weak**.

**Implementation complexity:** **pipeline-config-only**. Requires only changing physics-id flag in submission script. <1 day.

**Pre-committed falsification criterion:** same step-200 R-b at P2/P3/P4 closes this candidate (P1 not the cause; architectural pivot 1-3 still needed). Candidate-5 PASS at step 200 is PROVISIONAL gated on step-1000 BINDING per [D-53] convention.

---

## Honest-framing notes per [D-37] rule (a)

- **All 5 candidates have thin published-precedent support for IGM-NeRF-class reconstruction.** Candidates 1-3 lean on architectural-trend support from adjacent fields (emulation, inference, general residual learning); none has a direct IGM-3-D-field-reconstruction precedent. Candidates 4-5 lean on cosmology-statistics-bias and emulator-output-variation literature respectively; neither directly addresses reconstruction failure modes. **No candidate is "the published fix" for an [D-53]-also-exhausted scenario** — this is genuinely a class of next-tests.
- **Candidates 4 and 5 are load-bearing diagnostics, NOT architectural pivots strictly.** They should likely be dispatched FIRST in any [D-62] activation, as PASS on either closes [D-62] without architectural rewrite. The defense-panel scoping review at [D-62] activation should rank dispatch order with this in mind.
- **Candidate 2 (direct density-field supervision) has a fundamental observational-admissibility constraint:** the observable IS the flux. Pretraining-then-finetuning is the only honest framing; pure density-field supervision is **inadmissible as the production reconstruction target** because it would not generalize to real observations. This is a fundamental constraint, not a [D-62] blocker — surface in panel review.
- **Candidate 1 (transformer/attention/convolutional) is the heaviest implementation commitment** and should require the strongest panel-pre-commitment before dispatch.
- **What this scoping memo does NOT do**: it does NOT pre-rank, does NOT pre-commit any candidate, does NOT specify K2-equivalent test designs (those are design-stage work, commissioned per-candidate if PI activates [D-62] and panel-selects a candidate). It is intentionally a research-mode survey.

---

## Carry-forward

If [D-53] candidate (b) R-b at step 200, this memo seeds the [D-62] activation defense-panel scoping review. Panel scope: (i) pre-commit dispatch order (suggested: candidates 4 + 5 first as diagnostics, then 1-3 by panel-ranking); (ii) commission per-candidate design doc for the panel-selected first dispatch; (iii) re-affirm [D-37]-ext rule 1 anti-degeneracy (no within-[D-62] iteration past first-candidate-R-b without panel re-review).

If all 3 [D-53] candidates exhaust without rescue, same activation pathway with expanded urgency.

If [D-53] candidate (b) PASS at step 1000 BINDING verdict, this memo is **archived non-binding** and not activated.

*Total word count estimate: ~1450 words excluding headers.*

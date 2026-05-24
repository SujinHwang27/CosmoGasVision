# [D-62] Architectural Pivot — Scoping Memo

## Status

- **PROVISIONAL pending defense-panel re-review on this re-authored D62 doc** (panel-bound R15-clause-(c) re-verification not available since panel diagnosis IS the operational test).
- **Re-authored 2026-05-24** per defense-panel design review NEEDS WORK absorption + PI v7 verdict (LEDGER commit pending — see [D-62] addendum + [D-64] stub).
- **Diagnostics (4)/(5) load-bearing on whether the [D-62] activation premise (supervision-target-structure inference) is mechanism-aligned or microbatch-coupled / P1-specific measurement artifact. PASS on either invalidates the [D-60]/[D-61]/[D-63] absorption stack and de-activates [D-62].**
- Originally activated 2026-05-23 per defense-panel out-of-scope flag on [D-53] (commit `3074596`).
- Author: support-researcher per parent-PI scoping commission; re-authored per PI v7 absorption.
- Intended audience: defense-panel re-review on this re-authored D62 doc.

## Scope definition

[D-62] activates **IF** [D-53] also fails AND diagnostics (4) and (5) both R-b — specifically: (i) the first [D-53] candidate dispatched (panel-bound = (b)) R-b at step 200, OR (ii) all 3 [D-53] candidates exhaust without rescue, AND (iii) both diagnostics (4) microbatch=1024 and (5) P2 physics-variant return R-b at step 200. [D-62] scope = **architectural intervention**: target the **model-side capacity / inductive-bias / reconstruction-target-class**, NOT loss-side. Loss-side is the closed class of [D-60] L1 (in-loss-function) ∪ [D-53] (supervision-target-redesign).

---

## §Absorption-Stack Diagnostics (dispatch FIRST; gate the candidate ladder)

> **Framing (binding):** these are **DIAGNOSTICS, NOT architectural pivots.** PASS on either routes to **[D-64] L1-scope-revision** (NOT [D-62] success). FAIL strengthens the supervision-target-structure inference and gates [D-62] candidate ladder activation. The candidate-set in §Architectural Candidates is GATED on (4)+(5) BOTH returning R-b.

### Diagnostic (4) — microbatch=1024 — FIRST DISPATCH

**Mechanism prediction (1 paragraph):** the L1 sequence ran at microbatch=256; if the per-mode P_F variance estimator is biased at small batch sizes (small-N sample variance underestimates true variance, biasing the model toward constant-flux solutions which match the underestimated low-variance target), then microbatch=1024 could shift the loss-surface geometry. This is **load-bearing diagnostic before any architectural commit**: PASS at microbatch=1024 means the variance-collapse pathology is microbatch-coupled at P1, the [D-60]/[D-61]/[D-63] supervision-target-structure inference does NOT hold, and the rescue is a scope-revision (NOT an architectural intervention).

**Literature support:** small-batch P_F estimator bias is a known cosmology-statistics issue (the ensemble-average P_F is variance-biased at N < ~10³ sightlines; Iršič+ 2017 §4 systematic budget). No direct ML-training precedent for this being the IGM-NeRF-class pathology; cosmology-statistics support is solid.

**Implementation complexity:** pipeline-config-only. Change microbatch flag in submission script. ~15 min Juno wall-clock.

**Pre-committed routing:**
- **PASS** (`var_pf_band_ratio > 1e-3` at step 200) → **[D-64] L1-scope-revision ACTIVATES** (variance-collapse microbatch-coupled at P1); **[D-62] candidate ladder DE-ACTIVATES**; no further [D-62] dispatches.
- **FAIL** (R-b at step 200) → L1 in-loss-function exhaustion holds at N=1024; continue to (5).

### Diagnostic (5) — P2 physics-variant — SECOND DISPATCH (parallel-eligible with (4))

**Mechanism prediction (1 paragraph):** P1 physics-id may itself contribute to the diffuse-bin imbalance. P1 corresponds to a specific (γ, T₀) thermal-state combination in the Sherwood suite; P2 varies these parameters. If the variance collapse is P1-thermal-state-specific, then P2 smoke could rescue without architectural change. PASS routes to a scope-revision (CVPR claim narrows to multi-physics generalization gap), NOT to [D-62] success.

**Literature support:** physics-variation-driven loss-surface differences are well-documented in cosmology emulation literature (Cabayol-García+ 2023 Table 2 cross-physics emulator performance); not specifically for IGM-NeRF reconstruction failure modes. Honest framing: support is for emulator-output-variation, not failure-mode-variation; inheritance is weak.

**Implementation complexity:** pipeline-config-only. Change physics-id flag. ~15 min Juno wall-clock.

**Pre-committed routing:**
- **PASS** (P2 stable, `var_pf_band_ratio > 1e-3` sustained at step 200) → **[D-64] L1-scope-revision ACTIVATES** (pathology P1-thermal-state-specific); CVPR claim narrows to multi-physics generalization gap; **[D-62] candidate ladder DE-ACTIVATES on this branch**.
- **FAIL** (R-b at step 200) → pathology NOT P1-specific; inference strengthened. On (4) ALSO FAIL → [D-62] candidate ladder ACTIVATES at (3).

---

## §Architectural Candidates (GATED on (4)+(5) both R-b)

> **Activation precondition (binding):** this section's candidates may only be dispatched after BOTH diagnostics (4) and (5) above return R-b at step 200. Candidate ladder order is FIXED: (3) → (2) → (1). Within-ladder iteration past first-candidate-R-b requires panel re-review per [D-37]-ext rule 1 anti-degeneracy.

### Candidate (3) — fGPA-residual — FIRST in candidate-ladder

**Mechanism prediction (1 paragraph):** keep the differentiable Voigt integrator and current flux-domain supervision, but **constrain the MLP to learn only residuals from a physics-based prior** — Hui-Gnedin 1997 fluctuating Gunn-Peterson approximation (fGPA), which predicts density ↔ τ via a closed-form power-law. The MLP output is parameterized as `ρ = ρ_fGPA(δ, z) × (1 + MLP_residual(coords))` with MLP_residual initialized small. This **may rescue** by architecturally enforcing a non-constant prior: the variance-collapsed basin is excluded by construction because ρ_fGPA already varies spatially per the input δ-field. This is the **first test of the fGPA-residual architectural class** for IGM-NeRF; hedged-verb framing per [D-37].

**Literature support:** fGPA-based density-field initialization is standard in IGM forward-modeling (Hui-Gnedin 1997, used in TARDIS baseline Horowitz+ 2019); residual-MLP-on-fGPA-prior is novel for IGM-NeRF but inherits structural support from residual-learning (ResNet, NeRF-W). Application to IGM is novel.

**Implementation complexity:** pipeline-refactor + small architectural change. Requires (i) fGPA prior module (`src/models/fgpa_prior.py`, ~100 lines, analytic); (ii) residual-MLP wiring in `nerf.py`; (iii) initialization scheme. ~1 sprint cycle.

**CPU pre-flight (MANDATORY before Juno dispatch):**
1. Compute fGPA variance spectrum on P1-T1 truth field; compare to L1's collapsed-basin spectrum (confirms the prior carries the variance structure L1 fails to recover).
2. Verify autograd-through-δ: the `ρ = ρ_fGPA(δ, z) × (1 + MLP_residual)` parameterization must preserve autograd through the δ-field input pathway (no detached NumPy in the prior module; addresses honesty-audit finding #3).

**Estimator-equivalence test:** re-certify after architectural change (estimator must match within tolerance on a fixed seed before Juno commitment).

**Pre-committed falsification criteria (BOTH active):**
- Standard close: same step-200 R-b pattern (`var_pf_band_ratio < 1e-3` at step 200 on P1-T1 smoke).
- **Inverse-failure-mode close (new, per panel KILLER absorption — catches MLP_residual converging to non-trivial-but-wrong correction that fits τ-MSE locally while preserving variance collapse in P_F):**
  - `||MLP_residual||_2 / ||ρ_fGPA||_2 > 0.1` AND `var_pf_band_ratio < 1e-3` at step 200 → close as **"residual-rescued-tau-not-Pf"**.
- Silent-MLP close: if MLP_residual converges to ≈ 0 (the prior dominates and the MLP contributes nothing), close as R-b regardless of variance-collapse number — the architecture has degenerated to fGPA alone.

**Scope note on normalizing-flow sub-variant:** the flow-based generative model with Jacobian-determinant anti-collapse guarantee was previously bundled here. **Dropped from candidate-ladder pending separate scoping** — the Jacobian-determinant constraint operates upstream of post-Voigt post-binning `var_pf_band_ratio`, and propagation to the binned-band statistic is not verified. Re-eligible only after explicit constraint-propagation analysis.

### Candidate (2) — density-pretraining sub-variant only — SECOND

**Observational-admissibility constraint (surfaced explicitly):** **density is NOT directly observed in Lyα.** Only flux-after-Voigt-integration is observable. This candidate is a bet with **no published precedent at this regime**.

**Mechanism prediction (1 paragraph, scope-restricted to pretraining sub-variant):** if variance collapse is fundamentally a flux-domain-supervision pathology (information-theoretic bottleneck: P_F + PDF do not contain enough information to constrain the 3-D density field uniquely), then pretraining on density-field MSE in simulation may provide a strong field-realism prior that the flux-domain finetuning cannot collapse to constant. **Inference sub-variant (Maitra/LyαNNA 2023) REJECTED**: that is parameter inference, NOT 3-D field reconstruction; adopting it would abandon project scope per [D-43] CVPR plan-of-record.

**Literature support:** simulation-pretraining is standard in computer-vision; no IGM-NeRF density-field pretraining precedent located. Literature support is thin.

**Implementation complexity:** pipeline-refactor. Requires (i) pretraining stage with density-field MSE on simulation data; (ii) finetune stage with current flux supervision; (iii) handoff-state management; (iv) **EWC-class anti-forgetting mechanism** (see below). ~1 sprint cycle.

**EWC-class anti-forgetting (REQUIRED IN SCOPE):** Kirkpatrick+ 2017 (Elastic Weight Consolidation) — catastrophic forgetting is the well-studied failure mode under pretrain-then-finetune. The current sub-variant has **no anti-forgetting mechanism spec'd**; adding EWC (Fisher-information-weighted L2 penalty on drift from pretrained weights) is in-scope mandatory before dispatch.

**Pre-committed falsification criteria (BOTH active):**
- Standard close: same step-200 R-b pattern on the finetune stage (`var_pf_band_ratio < 1e-3`).
- **Quantitative prior-decay close (new):** measure density-MSE at pretrain end vs density-MSE after flux-finetune step 200. If `density_MSE(finetune step 200) / density_MSE(pretrain end) > 2` → close as **"pretraining-prior-decayed-beyond-budget"** (the EWC mechanism failed to hold the density-realism prior against the flux-finetune gradient).

### Candidate (1) — architecture-swap — DEFERRED, supervision-target-coupled reframe required

**BLOCKED at first-slot per panel BINDING.** Standalone transformer / CNN / attention architecture inherits the same supervision target → inherits upstream pathology (per [D-53] mechanism evidence + [D-63] inference). Substituting backbone class without changing what the loss looks at is mechanism-disjoint from the diagnosed failure mode.

**Reframe required for re-eligibility:** any future (1) candidate must be **supervision-target-coupled** (e.g., transformer + density-pretraining = candidate 1∩2) before re-eligibility. Standalone architecture-swap is **not re-eligible** as a [D-62] candidate.

**Honesty-audit downgrade per panel:** "heaviest implementation commitment has weakest prior evidence" — no published IGM-3D-reconstruction transformer precedent exists. Cabayol-García+ 2023 and Maitra/LyαNNA 2023 are summary-statistic regression / parameter inference, NOT 3-D field reconstruction; the inheritance claim is materially weaker than the architectural-trend framing previously suggested.

**Convolutional discriminator + adversarial loss specifically:** GAN-class is a different optimization regime entirely; mode collapse is a well-studied failure mode (Salimans+ 2016, Arjovsky+ 2017). Substituting one collapse-prone optimization for another is **not a rescue** and should not be re-introduced without an explicit anti-mode-collapse mechanism spec'd ex ante.

**Pre-committed falsification criterion (if a supervision-target-coupled reframe is eventually dispatched):** same step-200 R-b pattern. Within-class close routes to **[D-65 stub] further-class-pivot (out of current scope)**.

---

## §Stop-gate routing tree

```
1. Dispatch (4) microbatch=1024 diagnostic FIRST [~15 min Juno]
   (4) PASS (var_pf_band_ratio > 1e-3 at step 200)
     → [D-64] L1-scope-revision ACTIVATES (variance-collapse microbatch-coupled at P1)
     → [D-62] candidate ladder DE-ACTIVATES
     → No further [D-62] dispatches
   (4) FAIL (R-b at step 200)
     → L1 in-loss-function exhaustion holds at N=1024
     → Continue to (5)

2. Dispatch (5) P2 physics-variant diagnostic SECOND (parallel-eligible with (4)) [~15 min Juno]
   (5) PASS (P2 stable, var_pf_band_ratio > 1e-3 sustained)
     → [D-64] L1-scope-revision ACTIVATES (pathology P1-thermal-state-specific)
     → CVPR claim narrows to multi-physics generalization gap
     → [D-62] candidate ladder DE-ACTIVATES on this branch
   (5) FAIL (R-b at step 200)
     → Pathology NOT P1-specific; inference strengthened
     → On (4) ALSO FAIL: [D-62] candidate ladder ACTIVATES at (3)

3. [D-62] candidate ladder (activates only on (4)+(5) both R-b):
   (3) fGPA-residual — CPU pre-flight first, then Juno
     (3) R-b at step 200 → escalate to (2-pretraining)
   (2) density-pretraining + EWC — Juno
     (2) FAIL → escalate to (1-reframed)
   (1) architecture-swap-reframed — design re-review required
     (1) R-b at step 200 → [D-65 stub] further-class-pivot (out of current scope)
```

---

## Honest-framing notes per [D-37] rule (a)

- **All candidates have thin published-precedent support for IGM-NeRF-class reconstruction.** (3) inherits structural residual-learning support; (2) has no IGM density-pretraining precedent; (1) has no IGM-3D-reconstruction architecture-class precedent. No candidate is "the published fix."
- **Diagnostics (4)/(5) are load-bearing.** PASS on either invalidates the [D-60]/[D-61]/[D-63] supervision-target-structure absorption stack and routes to [D-64] scope-revision, NOT to [D-62] success. This is surfaced in §Status above.
- **Candidate (2) observational-admissibility constraint is fundamental.** Density is not directly observed in Lyα; the pretrain-then-finetune framing is the only honest framing, and the inference sub-variant is REJECTED as out-of-project-scope.
- **Candidate (1) is BLOCKED standalone** per panel BINDING; supervision-target-coupled reframe required before re-eligibility.
- **No independent verification of [D-63]'s 5-attempt sweep coverage claim** — PI signed off on PI's own coverage assertion. Deferred to a future defense-panel review of [D-63] specifically (out-of-scope for this design doc; logged as carry-forward).
- **What this scoping memo does NOT do:** does NOT pre-commit any architectural candidate without diagnostic-first routing; does NOT specify K2-equivalent test designs beyond the pre-commitments here (those are design-stage work, commissioned per-candidate if PI activates [D-62] post-diagnostic-FAIL and panel re-affirms).

---

## Carry-forward

- **Cross-reference to [D-64] L1-scope-revision stub** (BINDING): diagnostic (4) or (5) PASS routes execution to [D-64]. [D-64] stub authored per PI v7 absorption; LEDGER commit pending.
- **Cross-reference to [D-65 stub]** for further-class-pivot if all of (3), (2-EWC), (1-reframed) exhaust.
- **Cross-reference to a future defense-panel review of [D-63]** (5-attempt sweep coverage claim) — out-of-scope here, surfaced as carry-forward.
- If [D-53] candidate (b) PASS at step 1000 BINDING verdict, this memo is **archived non-binding** and not activated.

*Total word count estimate: ~1850 words excluding headers.*

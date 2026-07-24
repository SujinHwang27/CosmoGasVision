# [F-05] Amended R-B gate — pre-registration of record (2026-07-24)

**Status: PRE-REGISTERED before R-B numbers exist.** Encodes the [F-04] B6-panel-ratified amendments to the R-B "spine" readout. Commit precedes the run (derivation-at-spec-time; commit-before-numbers). Authority: the B6 defense panel recommended exactly this construction; this doc is the coordinator's faithful encoding, owed a PI countersign into the LEDGER decision log after the run.

## Object (unchanged)
For i ∈ {2,3,4}: decoded difference `D_i^dec = f(·; z_Pi) − f(·; z_P1)` vs true difference `D_i^true = x(P_i) − x(P_1)`, on the [D-49] **TEST** region, **real** frame, x = log₁₀(max(ρ/⟨ρ⟩,1e-3)). Estimator = `src/analysis/nccf` r_s(σ) (G0-certified on differences, `b3_g0_acceptance.json`).

## What changed and why (from [F-04])
1. **N1 (mismatched-difference null) is DROPPED as a licenser.** K1 proved the true difference maps are pairwise 0.87–0.92 correlated (σ=2) → N1 cannot discriminate pair-specificity. N1 is retained only as a **reported diagnostic**, never as a PASS condition.
2. **PASS becomes a CONJUNCTION, not scalar-r_s-above-N2.** A single-spatial-direction F-B decoder can clear N2 on all pairs; the conjunction is designed so that only genuine per-pair feedback recovery passes.
3. **F-F (UNDEFINED) is re-defined on decoded-D SNR**, not std-vs-1e-12.

## The amended R-B PASS (per pair, pre-registered)
A pair i PASSES iff ALL THREE hold at the primary σ (see σ-selection):
- **(a) N2-clearance:** r_s(D_i^dec, D_i^true) > phase-randomized-null 97.5th percentile (freshly banked per σ, N≥200; σ=2 ref already 0.118, `b3`).
- **(b) Amplitude-ladder recovery (the K1 discriminator):** the decoded amplitude ratios reproduce the TRUE ladder within 2×seed-SD. True σ=2 ladder std(D) = P2 0.0112 : P3 0.0293 : P4 0.1263 (≈ 1 : 2.6 : 11, `k1_collinearity.json`). Statistic: |std(D_i^dec)/std(D_1ref^dec) − true ratio| ≤ 2×seed-SD. This is the invariant a single-direction (F-B) decoder gets WRONG.
- **(c) Per-region sign agreement:** fraction of TEST sub-regions where sign(D_i^dec) = sign(D_i^true) exceeds a coin-flip null 97.5 (binomial over independent σ-cells). Feedback suppresses/enhances ρ in specific environments; a global-offset decoder fails this.

## σ-selection (pre-registered, not defaulted)
Run the σ-scan {1,2,4} at **192 native** (avoid the 64-grid block-mean lowpass, K3/P3). Report all σ. **Primary σ = the smallest σ at which true-D SNR is resolvable AND decoded-D clears (a).** If no such σ exists → **F-D** (large-scale-only) or dead. Bank a fresh phase-randomized null at each σ (σ=1 cannot reuse the σ=2 0.118).

## F-F (UNDEFINED) pre-registration
Predicted decoded-D SNR per pair = std(D_i^true) / (residual_std · √(2(1−corr_cm))), where residual_std and the common-mode residual correlation corr_cm are MEASURED on the trained model's TEST residuals (K2). Any pair with predicted decoded-D SNR < 0.3 (derived floor: below this the decoded difference is dominated by differential reconstruction error, not signal) is declared **F-F UNDEFINED before scoring** — reported, not spun. Expectation from K1/K2: **P2 → F-F** likely.

## Outcome cells (pre-committed, symmetric [D-37])
- **F-A (latent decodes):** ≥ 2 pairs PASS the conjunction. Given rank-2 structure (K1 SVD 69%+31%), ≥2 independently-validated directions is the minimum that licenses a *latent* (2-parameter) claim, not a single-direction result. Headline: "a ~2-parameter feedback response, truth-validated."
- **F-B (method-not-science):** codes separate (c4) but < 2 pairs pass the conjunction. "Conditioned emulator built; per-pair feedback recovery not validated." The Nasir-small-effect landing; disclosed symmetrically.
- **F-C-analog (single-direction):** exactly 1 pair passes (expected P4). NOT a latent — "field-level feedback recovery demonstrated for the strong-AGN direction only." Verb-capped; no *latent* claim.
- **F-D (large-scale-only):** passes at σ=4 only → re-confirms feedback moves large scales negligibly (no delta vs Nasir); verb-capped.
- **F-F (below-resolution):** per-pair, as above.

## Training budget for the R-B model (pre-registered)
The conditioned field must be trained to a real budget (not the 500-step smoke) so residual_std is minimized and decoded-D SNR is maximized — $0/local (CPU/MPS), NO Juno (B7 infra not cleared). Target: train to VAL-r_s plateau or a fixed step budget (≤ a few CPU-hr), 3 seeds, checkpoint saved. λ_z RE-PINNED on the difference-field c1'/c2' (whole-field pin void, [F-03] §4); derivation banked in the R-B exit artifact.

## Provenance pins (P4)
Pin the exact `nccf.py` commit/hash (currently the fc2ba38 import from exp/unet-inversion 39804b8) into the R-B exit artifact; all nulls reproducible from banked seeds.

# [F-04] B6 defense-panel review + K1 measurement (2026-07-24)

**Scope:** adversarial pre-review (Ext-2 rule 6) of the difference-field R-B readout — the corrected probe from [F-03] — BEFORE any re-run. **Verdict: SOUND-WITH-AMENDMENTS. Do NOT run R-B as scoped in [F-02]/[F-03] until 5 pre-run measurements are banked and the panel rules.**

## The panel's core finding
The subtraction D_i^dec = f(z_Pi) − f(z_P1) is clean (no shared-field free credit), and the plumbing is healthy, but **the dual-null (N1 mismatched + N2 phase-randomized) cannot distinguish F-A (latent decodes true feedback) from F-B (codes separated on a nuisance direction) on a suite whose difference maps are an amplitude ladder**, and the readout is **SNR-scissored** to ≤1 recoverable pair.

## Ranked attacks (hardest first)
- **K1 (killer):** the true difference maps are near-collinear across pairs → N1 dies (D_j^true ∝ D_i^true), and N2 is trivially passable by a decoder that learns ONE spatial "difference direction" and rescales it by code. r_s+dual-null cannot tell "decoding 3 physics responses" from "decoding 1 response 3×." **Fix:** amplitude-ladder recovery test (does std(D_i^dec)/std(D_j^dec) reproduce the true ratio?) + per-region sign agreement.
- **K2 (killer):** SNR scissors. Smoke residual std ≈0.45 (from r_s≈0.55); decoded-D needs error cancellation of 88% (P2) / 71% (P3) / ~18% (P4). Only P4 plausibly recoverable, and only if reconstruction error is common-mode across codes (unmeasured). **Fix:** re-define F-F (UNDEFINED) on decoded-D SNR, not std-vs-1e-12; pre-register P2→F-F.
- **K3 (killer):** σ scissors. Feedback lives at small σ where the decoder is weakest (lowpass, [D-75] k_c≲0.25); the 192→64 block-mean lowpasses again before r_s. **Fix:** σ-scan of decoded-D–true-D r_s AND true-D SNR on the same axes at 192 native + a banked σ=1 phase-randomized null; demonstrate a non-empty σ window or accept F-D (large-scale-only = no delta vs Nasir).
- **S1:** scalar r_s cannot license F-A; PASS must become a conjunction (N2 + amplitude-ratio + per-region sign + per-scale cross-spectrum).
- **S2:** Nasir+2017 collision — frame R-B strictly as ρ-field spatial localization; the observable-side ("feedback Nasir missed") rebuttal is UNLICENSED until R-C (spectra) runs. Pull the Nasir PDF for exact CDDF/line-width deltas.
- **S3:** the held-out TEST slab (~10 voxels ≈ 5 independent modes at σ=2) has too few DOF for a robust difference-map null; compute effective DOF, enlarge region or use a mode-counted k-space estimator; null must be on the same TEST geometry.
- **P1–P4 (probes):** re-derive λ_z on the difference field (whole-field pin void); confirm no per-physics code overfit; disclose the 64-grid lowpass; pin the exact nccf commit/hash (untracked extract) into the exit artifact.

## Out-of-scope flags
- **R-A (d=1 sufficiency) threatened by K1:** if diffs are collinear, "4 points embed in 1D" is a trivial amplitude-scaling artifact, not a discovered one-parameter family. Re-review R-A m_A under the collinearity lens.
- **R-C renderer differentiability is signature-level only** — but ∂spectrum/∂z_p is where the genuine novelty sits; do NOT let R-B's fate frame the whole paper.
- **c4's 5.48× is at provisional λ_z=0 on 64-grid codes;** re-confirm under re-pinned λ_z before citing "F-E falsified" in paper text.

## The 5 pre-run measurements (all $0/local, cubes on disk)
1. Cross-pair true-difference correlation matrix (K1). 2. Predicted decoded-D SNR per pair from residual common-mode (K2). 3. σ-scan + σ=1 null at 192 native (K3). 4. R-B PASS re-defined as conjunction (S1). 5. Nasir+2017 PDF pulled (S2).

---

## K1 measurement — RUN 2026-07-24 (`artifacts/k1_collinearity.json`)
Cross-pair r_s(σ) of TRUE difference maps D_i=x(P_i)−x(P_1):

| σ | P2·P3 | P2·P4 | P3·P4 | off-diag range |
|---|---|---|---|---|
| 1 | 0.823 | 0.703 | 0.731 | 0.70–0.82 |
| 2 | 0.874 | 0.873 | 0.918 | 0.87–0.92 |
| 4 | 0.911 | 0.935 | 0.963 | 0.91–0.96 |

**Off-diagonal ≥0.7 at every σ → K1 FIRES: N1 is non-discriminating; the difference maps share a dominant spatial pattern.** But the SVD of the 3 unit-normed σ=2 difference vectors = **[1.439, 0.960, 0.088]**, top-1 energy fraction **0.690** → the structure is **effectively RANK-2, not rank-1**: two real feedback directions (~69% + ~31% energy), third negligible (~0.3%). So the honest reading is neither "3 independent physics" nor "1 scaled P4" — it is a **~2-dimensional feedback response**.
True-difference std (σ=2): P2 **0.011** / P3 **0.029** / P4 **0.126** — amplitude ladder ≈ 1 : 2.6 : 11; only P4 approaches recoverable amplitude (K2/K3 confirmed).

**Implication for the science claim:** the clean "one-parameter feedback family" (d*=1) headline is likely dead (structure is ~2D); a "2-parameter feedback response" is the honest candidate. R-B, if run, must use the amplitude-ladder + per-region-sign conjunction (K1/S1 fix), and only P4 (± P3) carries recoverable difference SNR at the field level. The observable-side novelty (R-C, ∂spectrum/∂z_p) is untouched by this and remains the strongest unclaimed card.

# LEDGER — feedback-latent (conditioned neural field: the feedback latent of the Sherwood suite)

## **Architecture Diagram (Mermaid)**

```mermaid
flowchart LR
    C[Coordinates x,y,z] --> F[Shared conditioned field network\nSIREN or hash-grid trunk\n(NOT the falsified ReLU+PE MLP)]
    Z[Physics vector z_p\n4 learnable embeddings, d<=8\none per Sherwood variant P1..P4] --> F
    F --> O[Local fields rho, T, X_HI, v_pec\nof variant p at x]
    O --> R[Differentiable Voigt/RSD renderer\n(exp/nerf Stage-2a asset, D-57-fixed)]
    R --> S[Spectra of variant p + d spectrum / d z_p\nfeedback signature in the observable]
    O --> D[Latent-decoded delta maps\nvalidated vs TRUE P_i - P_1 difference maps\n(shared ICs cancel cosmic variance)]
```

---

## 1. The Pulse (Progress & Roadmap)

| Stage | Focus Area | Status | Target Metric | Notes |
|:--- |:--- |:--- |:--- |:--- |
| **Stage 0** | Track founding + proposal of record | ✅ **RATIFIED WITH SCOPING 2026-07-24 — [F-02]** (`design/f02_founding_ratification.md`): founding stands; prior-art sweep + cube enumeration both landed; **Stage-1 local smoke AUTHORIZED after B2 (renderer re-verify) + B3 (G0 re-run) green**; no Juno/R-C until B1(✅)/B6 panel/B7 infra clear | Proposal + ratified gates | Third track of record; user-primary direction ([FW-2] on exp/unet-inversion, banked 2026-07-24). Paper stays HALTED (user directive 2026-07-24) |
| **Stage 1** | Plumbing + smoke (SIREN auto-decoder, ρ-only forward) | 🟨 **PLUMBING PASS / LATENT-SEPARATION DEFERRED — [F-03] disposition 2026-07-24** (`design/f03_stage1_disposition.md`). Built + run: A2 model / A3 provider / B2 renderer / B3 estimator all green. Smoke v1 RED = lr bug (F0 audit: lr=1e-3 too high for SIREN(ω₀=30); `stage1_lr_audit.json`); v2 lr=1e-4 **TRAINING GREEN** (s2 overfit loss200=0.0011; conditioned r_s(σ=2)≈0.55). Anti-collapse c1/c2 FAIL — but **NOT F-E** (c4: codes 5.48× above null) — the whole-field gates are **PROVEN mis-specified** (`true_field_headroom.json`: TRUE variants 0.97–0.9995 similar → no headroom). Correct probe = swap-test on the DIFFERENCE field; needs **B6 panel micro-cycle before re-run**. | [F-02] §4(c) s1–s3 PASS; c1/c2 deferred to D_i probe | Small-effect (Nasir+2017) surfacing; **whole ballgame now = R-B on the difference maps**. Ordered signal confirmed: std(D)/std = P2 0.08 / P3 0.24 / P4 0.68 |
| **Stage 1** | Conditioned-field fit on all 4 variants (Phase A) | ⏳ | Fit-quality gates + anti-collapse gates (to be pinned with derivations at ratification) | Crop-scale, local/MPS first; small compute |
| **Stage 2** | Latent readouts: geometry, dimensionality (d=1 vs d=8), delta-map validation vs true differences | ⏳ | Pre-registered invariant-geometry tests | The true P_i−P_1 maps are the answer key |
| **Stage 3** | Spectra sensitivity ∂(spectrum)/∂(z_p) through the Voigt renderer | ⏳ | Computed sensitivity curves; target-feature list for surveys | Reuses exp/nerf differentiable integrator |

### Completed Milestones
- **2026-07-24**: Track founded on user directive ("start a new branch and setup experiment track"); scaffold committed. No code, no runs.
- **2026-07-24 (same session, user "go" for Stage-1 build)**: Full Stage-1 stack built + verified — A2 SIREN auto-decoder (266,529 params, 7/7 tests), A3 cube provider (18/18 tests, ρ×4, [D-49] split), B2 renderer differentiable (autograd-vs-FD 1.45e-8), B3 estimator certified incl. difference-field path. Smoke run: v1 RED (lr bug, F0 audit) → v2 lr=1e-4 TRAINING GREEN. [F-03] disposition: plumbing PASS (signed off), latent-separation DEFERRED (c1/c2 mis-specified — PROVEN via true-field headroom; NOT F-E). Next = B6 panel micro-cycle on the difference-field probe; the science verdict rests on R-B. Still $0/local; NO Juno; paper HALTED.

### [F-03] Stage-1 disposition (PI, 2026-07-24) — see `design/f03_stage1_disposition.md`
Plumbing PASS; latent-separation BLOCKED-pending-B6-panel. Whole-field c1/c2 gates proven mis-specified for the shared-IC small-effect suite (variants 0.97–0.9995 similar → zero swap headroom); correct probe is on D_i=x(P_i)−x(P_1). NOT F-E (c4 = codes 5.48× above null). λ_z=0 pin provisional (evaporates under fair control). B3-on-differences + true-field-headroom DISCHARGED into the panel packet; §4(c) s2 lr amended 1e-3→1e-4. Honest early warning ([D-37]): the Nasir+2017 small-effect signature is already visible; the entire science claim now rests on R-B clearing its dual-null on the difference maps. Difference amplitudes are real and ordered (P4−P1 std 0.369 ≫ P2−P1 0.045).

---

## 2. Methodology & Architecture (proposal of record, v0 — pre-ratification)

**The one-paragraph mental model.** The four Sherwood variants are the same universe under four gas-feedback settings. Everything that differs between them IS the feedback effect, with cosmic variance cancelled by the shared initial conditions. This track trains ONE field network shared across all four, plus four small learnable physics vectors — the network must use the vector to know which universe it is reproducing. Training therefore forces all common structure into the shared weights and all distinguishing structure into the vectors: the feedback latent is not searched for afterward; the objective squeezes it into a designated container. Forward, truth-supervised — the K2 degeneracy (exp/nerf [D-73]) poisons only inverse-from-flux and does not apply.

**Model.** Conditioned coordinate field f(x, z_p) → (ρ, T, X_HI, v_pec) of variant p at position x. Trunk: **SIREN or hash-grid encoding — explicitly NOT the falsified ReLU+positional-encoding MLP** (Mode-B collapse lineage, exp/nerf [D-60..71]/[D-73] A1; SIREN was the banked next candidate, hash-grid untested). z_p ∈ R^d, d ≤ 8, one per variant, learned jointly (auto-decoder style). **Pre-registered fallback arm:** conditioned convolutional decoder on crops (proven healthy on truth-supervised fields — exp/unet-inversion Stage 2) if the coordinate arm trips the inherited anti-collapse gates; all readouts survive the swap, continuity/renderer-nativeness degrade.

**Supervision.** Truth fields of all 4 variants (192³ scoring-pitch cubes exist for ρ; T/X_HI/v_pec production paths exist in the loader lineage — data-engineer enumeration owed). Loss: MSE on the [D-75] scoring variable x = log₁₀(max(ρ/⟨ρ⟩,1e-3)) (+ per-field transforms for T/X_HI/v_pec to be pinned WITH DERIVATIONS at ratification — derivation-at-spec-time rule is binding on this track from birth).

**The three pre-registered readouts (Phase A deliverable = "the feedback latent of the Sherwood suite"):**
1. **Latent geometry (invariants only):** ordering along the feedback ladder P1→P4; collinearity; distance ratios. Headline pre-registered question: **does d=1 suffice?** (train d=1 vs d=8, compare fit quality) — if yes, feedback acts on the gas field as a one-parameter family.
2. **Delta-map validation against ground truth:** decode field changes along the latent path P1→Pi; compare against the TRUE difference maps (P_i − P_1, same ICs = pure feedback effect) with the corrected-metric machinery ([D-75] suite, imported not re-implemented).
3. **Spectral signature:** ∂(rendered spectrum)/∂(z_p) through the exp/nerf differentiable Voigt/RSD renderer — which absorption features carry feedback; doubles as a survey target list.

**Honest scope pins (from birth):** shared-IC suite ⇒ findings are "feedback response at fixed cosmic structure" (the same fact that scoped exp/unet-inversion G3 down empowers this track — stated symmetrically); one box, one redshift (z=0.3) unless extended; latent claims attach to invariant geometry only (latent coordinates are gauge); interpolated latent values decode HYPOTHETICAL fields (descriptive only — no intermediate-physics sims exist to validate them).

---

## 3. The Logic (Decision Log)

- **[F-01] Track founding (2026-07-24).**
  Provenance: user directive sequence 2026-07-24 (session: Fable) — user-primary direction, banked first as [FW-2] on exp/unet-inversion LEDGER; "start a new branch and setup experiment track" = the go. Design conversation of record: conditioned/shared-trunk resolution of the one-scene-vs-shared-network question; physics vectors as the latent container; NeRF-format primary with modernized trunk; U-Net fallback.
  **Inherited discipline (binding):** [D-37] honest reporting + verb ceilings; every pinned threshold states its derivation at spec time (the U-track rule-2 cascade, 3 ledgered instances); all outcome cells enumerated pre-dispatch; anti-collapse gates inherited from the exp/nerf failure taxonomy (D1–D4 signatures + var-ratio floors) for the coordinate arm; corrected-metric suite ([D-75]/nccf) is the only scoring instrument; panel review per Ext-2 rule 6 before any >$5 compute.
  **Evidence base:** exp/nerf [D-46] (physics embedding separability under joint training, max pairwise L2 7.045); sprint-5 48³ discriminability (+24.3pp over moments, [D-56]); byte-identical LOS geometry + shared ICs across variants (exp/unet-inversion Stage-1 finding); truth-supervised conv training proven healthy (unet Stage 2); the falsified ReLU+PE lineage (what NOT to reuse).
  **Owed before any dispatch:** (a) ✅ **prior-art sweep LANDED 2026-07-24** (`design/prior_art_sweep_20260724.md`): conjunction is OPEN, components are NOT; **"first feedback latent" is dead as a headline** — Lin+2026 (ApJL, 2D CAMELS feedback latent) and Liu&Cuesta 2025 (field-level) own it; **Nasir+2017 (MNRAS 471, 1056) is the same suite + same question with a *small-effect* result** and must be answered; novelty verbs attach ONLY to the conjunction + the truth-paired-delta-map validation standard (the spine) + ∂spectrum/∂z_p through the Voigt renderer (unclaimed). Architecture is DeepSDF auto-decoder (Park+2019) + SIREN/hash trunk — derivative by design; (b) ✅ **PI founding ratification LANDED as [F-02]** (`design/f02_founding_ratification.md`); (c) ✅ **data-engineer cube enumeration LANDED** (`artifacts/f01c_cube_enumeration.json`): ρ×4 exist, T/X_HI/v_pec missing-but-producible, no hard block.

- **[F-02] Founding ratification — RATIFY WITH SCOPING (PI, 2026-07-24).** Full spec: `design/f02_founding_ratification.md`. Verdict: found­ing stands; **Stage-1 plumbing+smoke ($0, local, CPU/MPS, ρ-only, no paper claims) AUTHORIZED after B2+B3 green**; NO Juno/compute/R-C-spectra until B1 (landed), B6 panel pre-review, B7 infra cost-block clear. Key rulings: (1) novelty verbs attach ONLY to the conjunction + truth-paired-delta-map validation (the spine, R-B) + ∂spectrum/∂z_p (R-C, unclaimed) — never "first feedback latent"; 5 dangerous citations named (Lin+2026, Nasir+2017, Liu&Cuesta 2025, Sharma+2025, THALAS). (2) Three readouts as GATES with derived thresholds — R-A d=1-vs-d=8 fit-quality (m_A = 2×seed-SD); R-B decoded-vs-TRUE-difference-map r_s clearing dual measured nulls (mismatched + phase-randomized), all 3 pairs; R-C autograd ∂S/∂z with FD cross-check + random-direction null. (3) 8 enumerated outcome cells F0/F-A…F-G with symmetric disclosure; F-B = the Nasir-small-effect "method-not-science" landing. (4) Stage-1 = SIREN (5×256, ω₀=30) + DeepSDF auto-decoder z_p, ρ-only MSE on x, s1–s3 smoke ladder (mirrors [U-06] amended), anti-collapse controls c1 shared-z / c2 swap-test (the non-vacuous-latent gate) / c3 / c4. (5) R28: 7 rungs ≥ 7 landing artifacts A1–A7; next dispatches core-implementer (A2/A4/A5/A6/A7) + support-researcher (A1 renderer re-verify + B3 G0-on-differences). **R15 PROVISIONAL** — 6 unverified inherited claims flagged (renderer differentiability [B2], G0 acceptance re-run [B3], DVC-tracking [B4], [D-46] separability, SIREN candidacy). **Drift correction of record:** the differentiable renderer lives in `src/models/nerf.py` (`volume_render_physics` L261), NOT `src/rendering/` (empty) as CLAUDE.md/[F-01] diagram state.

---

## 4. The Data (Lineage & Governance)

Same dataset (user constraint of record): Sherwood 60 Mpc/h, z=0.3, 4 physics variants, shared ICs. **Cube enumeration of record: `artifacts/f01c_cube_enumeration.json` (data-engineer, 2026-07-24, current-session verified per R15).** ρ/⟨ρ⟩ truth cubes at 192³ ×4 EXIST + readable (P1: `experiments/nerf/artifacts/d75_rescore/cubes/truth_real_192.npy`, md5 efd9374…, NOT DVC-tracked; P2–P4: `experiments/unet-inversion/artifacts/stage1/cubes/truth_real_192_p{2,3,4}.npy`, DVC-tracked). **T, X_HI, full-vector v_pec: MISSING but PRODUCIBLE** — raw `SherwoodIGM_gal/extracted/` ×4 EXTRACTED-COMPLETE (16 HDF5 each, z=0.300, box 60000 kpc/h; PartType0 InternalEnergy/ElectronAbundance/NeutralHydrogenAbundance/Velocities/Density/Masses all present). No hard block; soft block only: `load_3d_field(field='T'|'xHI'|'vlos')` raises NotImplementedError (`src/data/igm_gal_loader.py:177`). `_cic_deposit_inplace` already does mass-weighted deposit (d75 velocity path) → one mass-weighted intensive-CIC producer covers all three fields in a single streaming pass over the ~40 GB sets (I/O-dominated). DVC rule: >10 MB artifacts → s3 remote. NOTE: prune `._*` AppleDouble litter before any dvc op (breaks `dvc status`).

---

## 5. Evaluation Plan

Adopts the [D-75] corrected-metric suite verbatim for all field/delta-map scoring (import, never re-implement). Track-specific additions to be pinned at ratification: latent-geometry invariant tests; d=1-sufficiency comparison protocol; delta-map vs true-difference scoring (same r_s machinery on difference fields); spectra-sensitivity computation convention through the renderer ([D-57]-fixed kernel).

---

## 6. Visualization & Artifacts

(placeholder — expected first entries: latent-geometry plot; decoded-vs-true delta-map slabs per variant; ∂spectrum/∂z sensitivity curves)

---

## 7. Session History & Next Handoff

### Session Snapshot: 2026-07-24 (track founding — Fable session)
- Founded on explicit user go after the design dialogue (one-scene property, physics vectors, shared trunk). Scaffold only; no dispatches.
- Concurrent context: exp/unet-inversion G2 fired U-A (seed 42: test r_s(σ=2)=0.934 vs grid bar 0.5758); S6 median-seed verdict (seeds 142/242) computing at time of founding.
- **Immediate next steps:** all three founding-owed items DONE (prior-art sweep, [F-02] ratification, cube enumeration). Track is RATIFIED WITH SCOPING; paper stays HALTED. Next, to reach Stage-1 smoke: clear **B2** (support-researcher: read `volume_render_physics` in full, [D-57]-fixed confirm + autograd/FD gradient check with a conditioned field) and **B3** (support-researcher: re-run G0/[D-75] acceptance suite incl. the NEW difference-field path). Then core-implementer builds A2 (SIREN auto-decoder `src/models/`) → A4 (unit+integration tests) → A5 (smoke driver s1–s3 + MLflow nullcontext) → A6 (anti-collapse c1–c4) → A7 (exit artifact, λ_z derivation). Stage-1 is $0/local/ρ-only — no Juno. Juno/R-C gated behind B6 defense-panel pre-review + B7 infra cost-block.
- **Concurrent:** exp/unet-inversion G2 CLOSED — U-A CONFIRMED on median seed (0.9369 vs grid 0.5758); paper-relicensing trigger fired but NOT exercised (user held the paper, elected this track instead).

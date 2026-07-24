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
| **Stage 0** | Track founding + proposal of record | 🟧 **SCAFFOLDED 2026-07-24 (user go)** — proposal in §2/§3 [F-01]; prior-art sweep + PI founding ratification owed before any dispatch | Proposal + ratified gates | Third track of record; user-primary direction ([FW-2] on exp/unet-inversion, banked 2026-07-24) |
| **Stage 1** | Conditioned-field fit on all 4 variants (Phase A) | ⏳ | Fit-quality gates + anti-collapse gates (to be pinned with derivations at ratification) | Crop-scale, local/MPS first; small compute |
| **Stage 2** | Latent readouts: geometry, dimensionality (d=1 vs d=8), delta-map validation vs true differences | ⏳ | Pre-registered invariant-geometry tests | The true P_i−P_1 maps are the answer key |
| **Stage 3** | Spectra sensitivity ∂(spectrum)/∂(z_p) through the Voigt renderer | ⏳ | Computed sensitivity curves; target-feature list for surveys | Reuses exp/nerf differentiable integrator |

### Completed Milestones
- **2026-07-24**: Track founded on user directive ("start a new branch and setup experiment track"); scaffold committed. No code, no runs.

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
  **Owed before any dispatch:** (a) prior-art sweep — CAMELS-family latent/emulator work, conditioned neural fields in cosmology, auto-decoder scene-code literature, feedback emulators; novelty verbs barred until it returns; (b) PI founding ratification (gates with derivations, outcome cells, Stage-1 design spec, agent ladder + R28 count); (c) data-engineer enumeration of T/X_HI/v_pec truth-cube production paths at 192³ ×4 variants.

---

## 4. The Data (Lineage & Governance)

Same dataset (user constraint of record): Sherwood 60 Mpc/h, z=0.3, 4 physics variants, shared ICs. ρ truth cubes at 192³ ×4 exist (P1: exp/nerf d75_rescore; P2–P4: exp/unet-inversion stage1, DVC-tracked). T/X_HI/v_pec cubes: NOT yet produced (loader NotImplementedError lineage for some fields — enumeration owed, [F-01](c)). Snapshot sources: `SherwoodIGM_gal/extracted/` ×4 (verified local, unet R1 audit). DVC rule: >10 MB artifacts → s3 remote.

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
- **Immediate next steps:** (1) prior-art sweep dispatch; (2) PI founding ratification; (3) data-engineer T/X_HI/v_pec cube enumeration. No compute before (a)+(b).

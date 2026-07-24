# [F-02] Founding stage-gate ratification + Stage-1 design spec of record (PI, 2026-07-24)

> Sign-off status: **R15 PROVISIONAL by default — PI-only, deferred panel review.** A narrow defense-panel pre-review (Ext-2 rule 6) is owed before the first Juno/>$5 dispatch; Stage-1 smoke ($0, local, CPU/MPS, no data-product claims, no paper claims) is dispatchable under this PROVISIONAL.
>
> **Verdict: RATIFY [F-01] founding WITH SCOPING. Stage-1 plumbing+smoke AUTHORIZED after the two Stage-1 blockers in §5 are green. No compute, no Juno, no R-C (spectra) until the T/X_HI/v_pec enumeration resolves the data dependency.**

**Prior-failure ledger (Ext-2 rule 4).** Inherited from exp/nerf: [D-40] integrated-stat loss → amplitude shrink; [D-41] self-anchored regularizer → constant collapse; [D-42] input-hint → density-head collapse; [D-60..71]/[D-73] A1 ReLU+PE Mode-B collapse (the trunk this track explicitly does NOT reuse); [D-73] K2 flux-likelihood does not identify the field (does NOT apply here — this track is truth-supervised FORWARD). From exp/unet-inversion: three gate-construction-without-derivation slips (0.20 floor below chance; s2 10×/50 underived; z4-in-trigger) — the reason **derivation-at-spec-time is binding on this track from birth.** Verb discipline: this track is the **first test of a DeepSDF-style auto-decoder physics code inside a coordinate field on a shared-IC feedback suite** — a candidate. No outcome verb attaches to any readout until its gate fires.

---

## 1. Novelty scoping ruling

**What the paper may NOT claim.** Not "the first feedback latent" — that phrase-space is owned (Lin+2026 ApJL 2D CAMELS feedback latent; Liu & Cuesta 2025 field-level). Not "feedback's latent space" in general. Not that interpolated latent values decode physically-validated intermediate fields (no intermediate-physics sims exist — interpolants are HYPOTHETICAL, descriptive only). Not a "discovered" latent geometry from n=4 (four points are a 4-point chart, not a manifold).

**What the paper MAY claim — novelty verbs attach ONLY to the conjunction + the validation standard:** a conditioned coordinate field with auto-decoder physics vectors, trained truth-supervised on the shared-IC Sherwood feedback suite, whose **latent-decoded delta maps are validated against the TRUE same-IC paired difference maps** (the spine — no CAMELS-latent work can do this on LH sets and did not on 1P), plus **∂spectrum/∂z_p through a differentiable Voigt renderer** (the genuinely unclaimed, survey-facing readout). Genitive of record: **"the feedback latent of the Sherwood suite"** — one suite, one box, z=0.3, feedback response at fixed cosmic structure.

**The three honest pins (binding on all paper text):**
1. **n=4 → 4-point chart, not a discovered latent.** Ordering/collinearity of 4 points is weak evidence alone; latent coordinates are gauge, only invariants are reportable.
2. **d=1-vs-d=8 is a strict FIT-QUALITY comparison, not a dimensionality discovery.** Four points always *embed* in 1D; the only admissible question is whether d=1 *decodes as well* as d=8 on held-out space (protocol in R-A). d=8 > n_instances — pre-empt the reviewer with the fit-quality curve.
3. **Truth-paired delta-map validation is the spine.** ∂spectrum/∂z_p is the unclaimed readout. Everything else (architecture = DeepSDF + SIREN/hash; shared-IC differencing = van Daalen+2011 / CAMELS 1P) is derivative by design and must be cited as such.

**The 5 dangerous citations that MUST be answered in the paper:**
1. **Lin+2026, ApJL 996 L41 (arXiv:2509.01881)** — owns "feedback latent." Frame as complementary (field+spectra-native, truth-paired, renderer-differentiable) or be read as a 4-sim rediscovery.
2. **Nasir+2017, MNRAS 471, 1056 (arXiv:1706.04790)** — SAME suite, SAME question, **small-effect** result on CDDF / line-width. The direct reviewer question — *"what does the latent recover that Nasir's statistics missed?"* — is answered by R-B or the contribution collapses to method-not-science (cell F-B). This citation IS the R-B stakes.
3. **Liu & Cuesta 2025, ML4PS** — field-level continuous feedback representation; closest methodological relative; timestamps the idea (verify author spelling before citing).
4. **Sharma+2025, MNRAS 538, 1415 (GPemu, arXiv:2401.15891)** — field-level conditioned decoding of baryonic effects; the readout-2 template minus latent-discovery and truth-paired validation.
5. **Ding, Horowitz & Lukić 2024 (THALAS, arXiv:2407.16009)** — public differentiable Lyα renderer; our renderer is not unique as an artifact, only the ∂spectrum/∂z_p composition is. (Science-first alt slot-5: Tillman+2023 ApJL 945 L17.)

---

## 2. The three readouts as GATES (with derivations)

**G0 (estimator acceptance — hard prerequisite to ALL field/delta scoring, inherited from [U-04] G1).** The [D-75] corrected-metric suite via `src/analysis/nccf.py` must pass, unchanged, AND re-pass on the NEW difference-field scoring path (difference-field scoring is new code — inherits G0, NOT grandfathered): (a) truth-vs-truth r_s(σ) = 1.0 ± 1e-6 at σ∈{1,2,4} on both fields and on difference fields D_i = x(P_i) − x(P_1); (b) degenerate-variance policy verbatim — std(field) < 1e-12 → UNDEFINED, never 0; (c) real frame primary, Pearson-of-smoothed-fields primary (`gaussian_smooth_periodic`+`pearson`), Spearman column. **G0 gates all of R-A/R-B/R-C.**

Scoring convention for all three: x = log₁₀(max(ρ/⟨ρ⟩, 1e-3)); r_s(σ) at σ∈{1,2,4} h⁻¹Mpc; **[D-49] TEST region held-out spatially** (controls the "d=8 has more dims than instances" memorization worry directly); real frame primary.

### R-A — latent geometry / d=1 sufficiency
- **Success criterion:** d=1 is sufficient iff reducing d from 8 to 1 does not degrade held-out reconstruction fidelity beyond seed noise; latent-geometry invariants (ordering along the feedback ladder P1→P4) are coherent.
- **Exact statistic:** fit-quality Q(d) = mean over the 4 variants of r_s(σ=2, real, TEST region) between decoded field f(·; z_{Pp}) and truth x(P_p). Ladder d ∈ {1, 2, 4, 8}, each at ≥3 seeds. Δ_A = Q(8) − Q(1).
- **Threshold WITH derivation (derivation-at-spec-time):** m_A = 2 × pooled seed-SD of Q(8) measured from the ≥3-seed spread. **d=1 SUFFICIENT iff Δ_A ≤ m_A** — two configs whose fit-quality differ by less than 2σ of the measured seed noise are statistically indistinguishable; the bar IS the measured noise, not a chosen number. Minimal d* = smallest d with Q(8) − Q(d) ≤ m_A. The "one-parameter family" headline attaches ONLY if d* = 1. Invariant geometry (reportable, NOT gated because 4 points trivially embed in 1D): does the ordering of the 4 z-vectors along their 1st principal axis match the feedback-strength ordering? Report as corroborating, never as primary evidence.
- **Pre-committed FAIL branch:** Δ_A > m_A → **cell F-C**: report d* > 1; "feedback response on the gas field is not a one-parameter family at this suite/redshift"; symmetric disclosure — this NARROWS the headline, it is a valid scientific outcome (Ext-2 rule 7), not a failure.

### R-B — delta maps vs TRUE P_i − P_1 difference maps (the spine)
- **Success criterion:** decoded feedback difference maps reproduce the TRUE same-IC feedback difference maps above null, for all three pairs.
- **Exact statistic:** true difference D_i^true = x(P_i) − x(P_1); decoded D_i^dec = f(·; z_{Pi}) − f(·; z_{P1}); score r_s(σ) between them, σ∈{1,2,4}, TEST region, real frame, i∈{2,3,4}. **Reuse the [D-75] corrected metric verbatim — NO new estimator** (`gaussian_smooth_periodic`+`pearson`; the difference field is just another cube fed to the same r_s(σ) path, gated by G0-on-differences).
- **Threshold WITH derivation:** two measured nulls — (N1) mismatched-difference: D_i^dec scored vs D_j^true, j≠i; (N2) phase-randomized D_i^true (`phase_randomized` in nccf.py, N≥200, banked seeds). Null bar = 97.5th percentile of each ensemble. **PASS = r_s(σ=2, real) on D_i^dec vs D_i^true clears BOTH null 97.5 bands for ALL three pairs.** The bar is the *measured* null distribution, not an absolute r_s — because there is NO external anchor for "good" small-effect difference recovery (Nasir+2017 small-effect), gating on null-clearance and reporting the r_s value as descriptive effect-size is the honest construction (same discipline as [U-06] K2: "GREEN is null-clearance, not an absolute recovery bar"). Amplitude disclosure mandatory: report var(D_i^true) and its SNR; **if std(D_i^true) is near the estimator floor per G0(b), the readout is UNDEFINED, not FAIL** (do not manufacture a verdict on a signal below resolution).
- **Pre-committed FAIL branch:** difference maps do NOT clear both nulls → **cell F-B** (method-not-science): "a conditioned emulator was built, but its decoded feedback response cannot be validated against the true difference maps at this scale" — this is exactly the Nasir-small-effect landing and MUST be disclosed as such, symmetrically. No "feedback latent" science claim survives F-B; the paper (if any) is a methods/negative-result contribution.

### R-C — ∂spectrum/∂z_p through the Voigt renderer (unclaimed readout; DATA-CONTINGENT)
- **Dependency gate (binding):** R-C renders spectra, which requires **T (b-parameter), X_HI, and v_pec** truth fields, not ρ alone. Until the data-engineer enumeration (§5) confirms T/X_HI/v_pec production paths at 192³×4, R-C is **DEFERRED to Phase-B**. FGPA fallback (τ from ρ via a ρ–T relation) reintroduces forward-model approximations and is a *disclosed, scoped* alternative only, not the primary R-C path.
- **Success criterion:** ∂(rendered spectrum)/∂z_p is (i) computable and finite through the [D-57]-fixed autograd renderer, (ii) non-trivial, and (iii) *predictive* — a first-order Taylor step predicts the real inter-variant spectral change.
- **Exact statistic:** via autograd through `volume_render_physics` (nerf.py), the Jacobian ∂S/∂z_{Pj}. Validation: first-order predicted change ΔŜ = (∂S/∂z)·(z_{Pi} − z_{P1}) vs the ACTUAL rendered difference S(P_i) − S(P_1); statistic = Pearson correlation over spectral bins, per pair.
- **Threshold WITH derivation:** (i) plumbing — gradients non-NaN AND finite-difference cross-check |autograd − FD| / |FD| < 1e-2 (standard autograd-correctness tolerance); (ii) non-triviality — ‖∂S/∂z‖ > the FD numerical-noise floor measured at the FD step size; (iii) predictive — corr(ΔŜ, ΔS) exceeds a **random-direction null** (same-norm random Δz, 97.5th pct, N≥200). Each threshold anchored to a measured quantity (FD floor, random-direction null), not chosen.
- **Pre-committed FAIL branch:** (iii) fails → the sensitivity curves are reported **descriptive-only**; the "survey target list" claim is DOWNGRADED to "candidate features, predictive validation not established." (i)/(ii) fail → renderer-integration audit, R-C blocked (routes to F0 if plumbing).

---

## 3. Enumerated outcome cells (all pre-committed, symmetric disclosure)

The dimensionality axis (R-A: d* = 1 vs d* > 1) is **orthogonal** to the spine axis (R-B: cleared vs not) — cells below cross them explicitly.

| Cell | Condition | Disposition (symmetric) |
|---|---|---|
| **F0** | Stage-1 smoke RED (collapse / no anti-collapse separation), OR renderer not differentiable-usable, OR G0 fails on any path, OR cube-identity mismatch | **Fail-closed; audit first.** No science, no paper claim. Valid end state (Ext-2 rule 7). |
| **F-A** | R-B clears both nulls (all 3 pairs) AND R-A geometry coherent AND (R-C validated OR R-C deferred-Phase-B disclosed) | **Deliverable stands** — "the feedback latent of the Sherwood suite" with the three pins. Headline sub-claim "one-parameter family" ONLY if d* = 1 (else state d*). |
| **F-B** | R-A fit fires (field reconstruction healthy) but R-B does NOT clear null | **Method-not-science, scoped.** "Conditioned emulator built; feedback-response recovery not validated / below resolution." The Nasir-small-effect landing; disclose the reviewer question was answered in the negative. No feedback-latent science verb. |
| **F-C** | R-B clears but d=1 insufficient (Δ_A > m_A) | **Latent is not a single factor.** Report d*; headline narrows from "one-parameter" to "d*-parameter feedback response." Compatible with F-A on the spine. |
| **F-D** | R-B clears at σ=4 only (large-scale), fails σ∈{1,2} | **Scoped large-scale-only** feedback-recovery claim; verb ceiling. |
| **F-E** | Anti-collapse swap-test (§4 c2) fails: z_p do not carry variant information (fit no worse under z-swap) | **Vacuous-latent flag** — the "latent" is decorative; blocks ALL latent claims. Routes to F0-class audit. |
| **F-F** | R-B UNDEFINED (std(D_i^true) below estimator floor for one/more pairs) | **Below-resolution disclosure**, NOT a fabricated verdict; report the SNR; the affected pair carries no claim. |
| **F-G** | R-C (iii) fails only | Sensitivity descriptive-only; survey-target claim downgraded (see R-C FAIL). Does not block F-A on R-A/R-B. |

---

## 4. Stage-1 design spec (plumbing + smoke)

**Scope:** in — auto-decoder wiring, ρ-only truth-supervised forward, smoke ladder, anti-collapse controls. Out — T/X_HI/v_pec heads (owed on cube enumeration), spectra/R-C, any Juno, any latent-geometry science read. Hard cap: $0 cloud, zero Juno, CPU/MPS only, ≤ 1 CPU-hr aggregate, ≤ 30 min unit-test wallclock.

**(a) Architecture (concrete).** Trunk = **SIREN** (`src/models/` new module; the banked next-candidate per [F-01], explicitly NOT ReLU+PE) — 5 hidden layers × width 256, ω₀ = 30 (Sitzmann+2020 init: first-layer U(−1/n, 1/n) scaled by ω₀, hidden U(−√(6/n)/ω₀, √(6/n)/ω₀)). Hash-grid = pre-registered ablation, not Stage-1. **Auto-decoder physics vectors (DeepSDF, Park+2019):** 4 learnable embeddings z_p ∈ R^d, init N(0, 0.01²), optimized JOINTLY with trunk weights. Wiring = **concatenation at input**: input dim = 3 (normalized coords, SIREN convention [−1,1]) + d. Stage-1 head: single ρ output, no output activation (x spans ≈[−3, +3.6]). **Param target ≈ 0.26 M** (5×256 SIREN) + 4·d embedding params (negligible) — deliberately small; capacity is a Stage-2 knob, not a Stage-1 default. Code z-prior: DeepSDF-style L2 on z_p, weight λ_z — **derivation-at-spec-time: λ_z pinned in the Stage-1 exit artifact as the value at which anti-collapse control c1 separates while Q does not degrade; NOT a silent default.**

**(b) Loss.** MSE on x = log₁₀(max(ρ/⟨ρ⟩, 1e-3)) over all sampled coordinates, all 4 variants, shared trunk + per-variant z_p. (T/X_HI/v_pec per-field transforms deferred WITH DERIVATIONS to the cube-enumeration follow-up; the derivation-at-spec-time rule holds them until then.)

**(c) Stage-1 smoke GREEN (numeric gates + derivations).** Ladder mirrors [U-06]:
- **s1 — unit tests + R20(i) integration test.** Coord normalization round-trip; SIREN forward finite; auto-decoder gradient reaches BOTH trunk and z_p (`z_p.grad is not None`); target transform matches [D-75] `x_transform` ≤ 1e-12; seeded determinism; behavioral integration test on real cube → sample → forward → MSE: `loss.grad_fn is not None and loss.requires_grad`, max-abs weight AND z_p change ≥ 1e-6 over ≥ 2 optimizer steps.
- **s2 — overfit-one-batch (2 variants, fixed batch, lr 1e-3, 50 steps; 200-step diagnostic horizon).** Derived exactly as the [U-06] amended s2 (anchored to measured data quantities, not the trajectory): **(i) non-triviality** — loss(50) < 0.9 × MSE(predict-per-example-mean) [recorded mean-floor]; **(ii) anti-collapse** — pred_std(50) > 0.1 × target_std [banked collapse signatures sit ≥2 orders below target variance]; **(iii) memorization-depth** — loss(200) ≤ 0.5 × MSE(mean-floor). All three anchored to recorded floors in the exit artifact.
- **s3 — mini-run (2 variants, ≥ 500 steps) + R20(ii) contract assertion at step 100, outside try/except, raising loud:** pred_std on fixed val batch > 0.01 x-units AND loss(100) < loss(1).

**(d) Controls (analogues of unet z1–z5) — the anti-collapse spine of this track.** The central Stage-1 hazard is the **physics-averaging basin**: the auto-decoder can drive all 4 z_p to one vector and fit a variant-average, making the "latent" vacuous (cell F-E). Controls, each with a DERIVED threshold:
- **c1 — shared-z control** (analogue of zero-input): retrain with a SINGLE shared z for all variants (d effectively 0). Per-variant fit MUST be worse than the conditioned model by > 2× seed-SD of Q. If not, the vectors carry no information → F-E.
- **c2 — swap test** (analogue of z4 cross-physics): decode variant i with z_j (i≠j); reconstruction r_s(σ=2) MUST degrade by > 2× seed-SD. This is THE gate that the latent is non-vacuous. (Cheap: shared-IC byte-identical geometry makes the swap free.)
- **c3 — shuffled-assignment**: permute z_p → variant mapping at eval; fit degrades. Diagnostic.
- **c4 — label-shuffle null**: train with variant labels shuffled across examples; z_p must NOT separate (pairwise ‖z_i − z_j‖ ≈ its own null). Diagnostic that separation is signal, not artifact.

**(e) Stage-1 exit criteria.** S1 unit/integration tests PASS; S2 overfit three-condition PASS; S3 contract PASS; anti-collapse c1+c2 separation demonstrated on the 2-variant mini-run (derived thresholds recorded); λ_z pinned with its derivation in the exit artifact; one rendered figure per variant (central slab of decoded vs truth x) eyeballed at the gate. A RED at any is a valid end state (Ext-2 rule 7), routes to F0/F-E, NOT gate-bending.

---

## 5. Owed-before-compute checklist + R28 agent ladder

**Owed before ANY Juno/>$5 compute (all BLOCK; none block Stage-1 local smoke):**
- **B1** — data-engineer T/X_HI/v_pec enumeration at 192³×4 (**LANDED 2026-07-24**, `artifacts/f01c_cube_enumeration.json`): ρ×4 exist; T/X_HI/v_pec MISSING-but-PRODUCIBLE (raw HDF5 fully local + complete; one mass-weighted intensive-CIC producer covers all three in a single streaming pass; `_cic_deposit_inplace` already does mass-weighted deposit). No hard block → R-C is Phase-B-buildable, Stage-1 stays ρ-only.
- **B2** — R15 re-verification of the renderer: read `volume_render_physics` in full, confirm [D-57]-fixed status, run a gradient/finite-difference correctness check with a conditioned field. (Signature-level existence verified this session; full differentiability NOT — see §6.)
- **B3** — G0 estimator acceptance re-run on the difference-field path (NEW code, not grandfathered).
- **B4** — P2–P4 cubes DVC-tracked (exp/unet Caveat 1 — verify discharged).
- **B5** — R20 twin-gate (integration test + step-100 contract) green.
- **B6** — narrow defense-panel pre-review of THIS spec's gate constructions (Ext-2 rule 6; the R15 lift owed since founding), scope: R-A m_A derivation, R-B dual-null construction + unit-chain on the difference-field r_s, R-C null construction, anti-collapse c1/c2 thresholds.
- **B7** — infrastructure-manager Juno reachability re-verification + full cost block (partition, hrs/run, $/run, ceiling, auto-stop, artifact lifecycle); reject if any element missing.

**§R28-CHECK.** Stage-1 landing artifacts (7): A1 = renderer re-verification + FD-gradient report (B2) · A2 = SIREN auto-decoder module `src/models/<name>.py` · A3 = truth-cube provider (ρ, 4 variants, [D-49] split) in `src/data/` · A4 = Stage-1 unit + R20(i) integration tests · A5 = smoke driver (s2/s3 + step-100 contract) + MLflow contract (`nullcontext` fallback mandatory) · A6 = anti-collapse controls c1–c4 harness · A7 = exit artifact (λ_z derivation, per-variant figures, recorded floors). Rungs (7): R1=A1 → R2=A3 → R3=A2 → R4=A4 → R5=A5 → R6=A6 → R7=A7. **7 rungs ≥ 7 landing artifacts ✓.** Agent dispatches (this ratification): **data-engineer** (A3 + the B1 cube enumeration, done) → **core-implementer** (A2, A4, A5, A6, A7) → **support-researcher** (A1 renderer re-verification + B3 G0-on-differences). 3 dispatches. B6 panel + B7 infra are Stage-2 dispatches, enumerated but NOT issued now.

---

## 6. R15 provisional flags (inherited claims NOT independently re-verified this session)

**Re-verified this session (lifted):**
- ρ 192³ ×4 cubes exist — globbed: P1 `experiments/nerf/artifacts/d75_rescore/cubes/truth_real_192.npy`; P2–P4 `experiments/unet-inversion/artifacts/stage1/cubes/truth_real_192_p{2,3,4}.npy`. ✅
- [D-75] corrected metric exists and is the smoothed-log r_s — `src/analysis/nccf.py`: `gaussian_smooth_periodic` + `pearson` = r_s(σ); Spearman column present; degenerate-variance and low-pass ladder machinery present. ✅ (EXISTENCE only; acceptance suite NOT re-run — see below.)
- Voigt renderer EXISTS and carries conditioning hooks — `src/models/nerf.py`: `tepper_garcia_voigt` (line 233) + `volume_render_physics` (line 261, window=64, `physics_id`/`g` conditioning args, autograd tensors). ✅ signature-level only.

**PROVISIONAL — relied upon but NOT re-verified this session:**
1. **Renderer is [D-57]-fixed, end-to-end differentiable, and usable with a conditioned field** — only the signature was read, not the full function body, and NO gradient/FD test was run. → B2.
2. **T/X_HI/v_pec production paths exist at 192³×4** — RESOLVED by B1 (data-engineer, landed): missing-but-producible, no hard block. R-C and multi-field heads depend on one producer script.
3. **G0/[D-75] acceptance suite passes** (truth-vs-truth = 1.0 ± 1e-6 at σ∈{1,2,4}) — the code exists but I did NOT re-run the acceptance this session, and the difference-field path is new. → B3.
4. **P2–P4 cubes are DVC-tracked** (exp/unet Caveat 1) — not verified. → B4.
5. **[D-46] embedding separability (max pairwise L2 7.045) and sprint-5 +24.3pp** — inherited from exp/nerf LEDGER §3, NOT re-read this session; these are the evidence base for expecting separable physics vectors.
6. **SIREN is the "banked next candidate"** and hash-grid is untested — inherited from [F-01], not re-verified against the exp/nerf failure taxonomy.

**Drift flag (minor):** CLAUDE.md and the [F-01] diagram place the differentiable renderer in `src/rendering/` — that directory is **empty**; the renderer actually lives in `src/models/nerf.py`. Not a loss, a path-of-record correction the paper and any owner brief must use.

# [D-69]-stage-1 (2)-pretraining scoping — PI design doc

**Status**: PROVISIONAL per R15 (PI-only design-layer sign-off; defense-panel re-review queued on revised doc).
**Revision: 5 (amendment-5: R15 LIFTED NON-PROVISIONAL on Rev-3 panel APPROVE clause-a; panel SERIOUS #1 absorbed as zero-shot sub-step 1 + fine-tune sub-step 2; 2026-05-24)**
**Authored**: 2026-05-24, PI self-dispatch per [D-69] dispatch clause.
**Revised**: 2026-05-24, amendment-1 absorption (K1/K2/K3/K4 from defense-panel + S1/S2/S3 SHOULDs + P2/P3 ADDRESS-NOW).
**Parent decisions**: [D-62] candidate ladder; [D-68] amendment R29 BANKED; [D-69] (3) fGPA-residual FALSIFIED at R_feas=8.33e-3.
**[D-37]-Ext R2 cascade verb-hedge applied throughout**: this is the *first scoping of a (2)-class candidate under post-(3)-falsification framing*, NOT "principled escalation" / "structurally correct pivot" / "addresses the diagnosed pathology directly."

---

## R26 in-session re-verification (Revision 1 + Revision 2 confirmatory re-grep)

- `grep -i "pretrain|warm-start|two-stage|EWC|fine-tune"` over LEDGER → only matches are in [D-62]/[D-65]/[D-69] design-language and the [D-62] candidate spec itself. **No prior (2)-shaped attempt exists in §6/§7 run history.** The (3)→(2) cascade is a first-test on (2); R2 verb-hedging inherits from the (3) failure (this morning) only.
- `Sherwood/.rho_field_cache/` filesystem audit → `rho_field_p1_z0.300_n64.npy` + `rho_field_p1_z0.300_n768.npy` exist with manifests. **P1 z=0.3 density-field data is locally present at production resolution.** P2/P3/P4 caches not present (consistent with [D-51c] tarball-extraction state).
- `loader.py:897 extract_rho_crops` is the production API for 3D ρ-field access; produces `rho / <rho>` overdensity float32 on the n_grid native grid. **This is the substrate for any density-side pretraining target.**
- **Revision 2 R30 inaugural re-verification**: re-read of `experiments/nerf/design/D62_architectural_pivot_scoping.md` lines 82, 84, 88 confirms the three [D-62] §Candidate(2) requirements (pipeline-refactor / EWC-class anti-forgetting REQUIRED IN SCOPE / quantitative prior-decay close >2 with provenance-deferral clause); §0.5 absorption table below maps each to this doc's coverage.

---

## §0.5 Parent-envelope CAVEATS discharge (NEW, K2 absorption)

Explicit mapping of `experiments/nerf/design/D62_architectural_pivot_scoping.md` §Candidate(2) requirements → this doc's coverage:

| [D-62] line | Requirement | Revision-2 coverage |
|---|---|---|
| L82 | Pipeline-refactor: pretrain + finetune + handoff-state mgmt | §1 declares Stage-1-only (pretrain-only feasibility); finetune + handoff = Stage 2 (deferred, out of scope) |
| L84 | **EWC-class anti-forgetting (REQUIRED IN SCOPE per [D-62])** | §1 "Stage-2 mandatory; out-of-scope for Stage 1" (deferred per K1 — no fine-tune in Stage 1, so EWC is not yet binding) |
| L88 | **Quantitative prior-decay close `density_MSE(finetune step 200) / density_MSE(pretrain end) > 2`** | §1 "Stage-2 binding gate; out-of-scope for Stage 1 (Stage 1 closes at pretrain saturation, no fine-tune)" |
| L76, L80 | **Observational-admissibility constraint** (density not directly observed); **thin literature precedent** | §1 ¶0 declared at design time; (γ) trains against Sherwood truth-ρ which is sim-output, not observational data. Precedent thinness cited Erhan+ 2010, Bengio+ 2007 per S1 paragraph in §1. |

**Honest framing**: the Stage 2 mandatory items (EWC, prior-decay gate) are *deferred not waived*. Stage 1 closing PASS does NOT bind Stage 2 PASS; Stage 2 is the gating decision for the (2)-candidate as a whole. Stage 1 is a scoped feasibility check on the pretrain leg only.

---

## §1 Pretraining target choice (one pick per `feedback-pi-decides`)

**¶0 Scope declaration (NEW, K1 absorption; D1 SHOULD absorbed at Rev 3).** Stage 1 = **pretrain-only feasibility study** for the (γ) target. The pretrain-then-finetune full pipeline is Stage 2; EWC + prior-decay gate are Stage-2 mandatory and out-of-scope here. **Observational-admissibility constraint** (density not directly observed in Lyα — per [D-62] L76) is declared at design time: (γ) trains against Sherwood *truth-ρ*, which is sim-output not observed; this is a sim-to-sim consistency exercise, not a sim-to-observation alignment. **Literature-precedent thinness** (per [D-62] L80) inherits to Stage 1 unchanged — see S1 paragraph below. **Stage 1 PASS demonstrates MLP capacity for diffuse pretraining — necessary but not sufficient for the L1 saturation-band deficit fix, which is the downstream Stage 2 EWC question.**

**Three candidates scored:**

**(α) P_δ band-power-spectrum target.** 1D matter power spectrum of overdensity δ in inertial-band $k_\parallel \in [10^{-2.5}, 10^{-1.5}]$ s/km, mirroring the L1 production gate but on δ instead of F.
- *Anchor*: theoretical CDM P(k) (Bolton+2017 §3 sim suite emission); literature precedent strong (every cosmology emulator benchmark, e.g., Cabayol-García+ 2023).
- *Loss*: $\mathcal{L}_{P_\delta} = (1/N_{\rm rays}) \sum_{\rm rays} \sum_{k \in K_{\rm inertial}} (\log_{10} P_\delta^{\rm pred}(k) - \log_{10} P_\delta^{\rm truth}(k))^2$.
- *Addresses production deficit?* Indirectly. Constrains spectral structure on δ; downstream L1 fine-tune still has to learn the δ→F mapping (fGPA-class power-law) without re-collapsing. R29 frame-audit: SAME-frame as the L1 production gate (dimensionless log-MSE over inertial band) — strong frame-match.
- *Failure modes / R13*: PASS on $P_\delta$ does NOT bind PASS on $P_F$ (the δ→F nonlinear power-law re-introduces tail-amplification; this is the exact mechanism that just falsified (3)-fGPA in [D-69] Stage 1). R13 scope-locked: $P_\delta$ PASS is "δ-side variance preserved", NOT "$P_F$-side variance will preserve under fine-tune."
- *Data*: Sherwood ρ-field at P1 z=0.3 LOCAL; 1D P_δ extraction along sightline rays uses the same ray-sampler as L1.

**(β) 3D density auto-correlation ξ(r).** Match ξ_3D(r) on δ to N-body matter correlation function at $r \in [0.5, 5]$ h⁻¹Mpc.
- *Anchor*: published N-body ξ(r) (Lukić+ 2015 §3); cosmology-standard.
- *Loss*: $\mathcal{L}_\xi = \sum_{r_i} (\xi^{\rm pred}(r_i) - \xi^{\rm truth}(r_i))^2$ at fixed radii.
- *Addresses production deficit?* Targets *spatial correlation* not *spectral variance amplitude*. The diagnosed saturation-band variance collapse is a P_F-amplitude pathology (per [D-39]/[D-65]); ξ(r) is an indirect cousin via FT. R29 frame-audit: DIFFERENT framing (real-space correlation, not k-space power). Weak frame-match.
- *Failure modes / R13*: PASS-on-ξ binds even less to PASS-on-$P_F$ than (α); ξ is a 1-point projection of the 2-point statistic, so the variance-amplitude information is partially marginalized away.
- *Data*: requires full 3D δ field — local for P1 only; production ξ(r) computation at n_grid=768 has known peak-RSS sensitivity per [D-50].

**(γ) Density slice MSE.** Point-wise δ supervision $\mathcal{L}_{\rho} = (1/N) \sum_v (\delta^{\rm pred}(v) - \delta^{\rm truth}(v))^2$ over voxels.
- *Anchor*: Sherwood simulation output directly (Bolton+2017); no theoretical-anchor uncertainty.
- *Loss*: voxel-MSE on log(δ + ε), ε=1e-3 (see §2 footnote on stabilizer semantics).
- *Addresses production deficit?* Trains *point estimates*; teaches the MLP the field topology directly. R29 frame-audit: completely different frame (real-space point MSE, not k-space variance). Strongest *teach-the-field-realism* signal, weakest *frame-match-to-L1*.
- *Failure modes / R13*: PASS-on-slice-MSE does NOT bind PASS on any spectral statistic; classic "learn the mean, miss the variance" failure is the dominant risk. R13 scope-locked rigidly: slice-MSE PASS ≠ variance-preservation claim.
- *Data*: same as (β) — P1 LOCAL at n_grid=768.

**PI PICK: (γ) density slice MSE, log-domain.**

*Rationale (single decision, no menu; REVISED PER K1):*

> **(γ) is selected for Stage-1 feasibility because frame-mismatch from L1 production prevents inheritance of the (3)-falsified δ→F pathology during the pretrain stage; whether the pretrain-anchored weights survive an EWC-equipped L1 fine-tune is the Stage 2 binding question.**

Supporting points (narrowed):
1. **The diagnosed pathology is "MLP cannot learn the δ field topology under flux-only supervision"** — pretraining must address the upstream cause directly. (α)/(β) regularize statistics-of-the-field; a model that has never seen the field topology has no anchor for those statistics to regularize.
2. **(α)/(β) inherit the [D-69]-falsified pathology by construction**: power-law nonlinearity from δ→F (fGPA β=1.6) tail-amplifies any spectral mismatch; (3)-fGPA proved this morning that statistical-prior-on-δ does NOT carry the inertial-band variance signal through the forward map. Pretraining on δ-spectral statistics inherits exactly that failure.
3. ~~**(γ) gives the MLP the unambiguous voxel-level density field as initialization; the L1 fine-tune then learns the residual δ→F mapping with weights anchored to a realistic field.**~~ **STRUCK (K1)**: this rationale conflates Stage 1 success with Stage 2 success. The fine-tune-survival claim is **demoted to "Stage 2 working hypothesis, untested at Stage 1 close."**
4. **R13 risk** is sharpest on (γ) (PASS-on-slice-MSE binds nothing on $P_F$) but is *honest*: the binding question is whether anchored weights survive the fine-tune (the [D-62] EWC mechanism). The unbinding is *between pretrain and fine-tune*, **not addressed at Stage 1**. (α) hides this risk in spurious frame-match; (γ) surfaces it cleanly and defers it cleanly.

[D-37]-Ext R2 verb-hedging applied: this is the **first scoping of a (2)-class candidate under post-(3)-falsification framing**. NOT "structurally correct pivot" / "the right pretraining target" / "principled escalation."

**S1 forward-justification (NEW).** The pretrain-then-finetune paradigm has strong empirical precedent in the deep-MLP tradition (Erhan et al. 2010, "Why does unsupervised pretraining help deep learning?" JMLR 11; Bengio et al. 2007, "Greedy layer-wise training of deep networks" NeurIPS) — both establish that an unsupervised or auxiliary-target pretraining phase can move the parameter initialization into a basin from which supervised fine-tuning escapes pathological local minima it could not escape from random init. Whether this transfers to neural-radiance-field-style implicit 3D fields with a non-linear physics-forward-map (Voigt + fGPA-class) is an open empirical question — **NeRF-specific neural-field pretraining literature is thin; cited precedent is from deep-MLP pretraining tradition (Erhan+ 2010, Bengio+ 2007). Per [D-62] §Candidate (2) "thin literature precedent" CAVEAT, this is declared as a known weakness.**

---

## §2 Loss form + gate ladder

**Pretraining loss (single equation):**

$$\mathcal{L}_{\rm pre}(\theta) = \frac{1}{N_v} \sum_{v \in \mathcal{V}_{\rm batch}} \big(\log_{10}(\rho_\theta(v) + \epsilon) - \log_{10}(\rho_{\rm truth}(v) + \epsilon)\big)^2 \quad \epsilon = 10^{-3}$$

where $\rho_\theta$ is the NeRF MLP density head output (post-Softplus, [D-06]), $\rho_{\rm truth}$ is the Sherwood overdensity at voxel $v$ from `extract_rho_crops` (P1, z=0.3, n_grid=768), and the log-domain reduction handles the filament tail (peaks at $\delta \sim 10^2$).

**Footnote on ε=1e-3 (NEW, P3 absorption).** Per K3 calibration probe (`scripts/d69_m3_band_calibration.py`), `_RHO_CROP_LO=1e-3` in `loader.py` is **documentation-only** (not enforced per-voxel by `_validate_rho_crops` at `loader.py:1254`, which asserts only non-negativity and max < 1e6). The 25.5% legitimately-zero voxels at n_grid=768 pass through unchanged; the `+1e-3` epsilon in this loss formula is the actual stabilizer for `log10(0)`. The numerical coincidence between the loader's `_RHO_CROP_LO` constant and the loss-formula's ε is by convention only, not by enforcement — both are 1e-3 because both are the chosen sub-resolution density floor.

**R29 in-line frame-audit (mandatory per banked R29):**
- *Production metric framing*: L1 production $P_F$ MSE is dimensionless log-MSE in k-space over inertial band (`pipeline.py:2080-2088`).
- *This loss framing*: log-MSE in voxel-real-space.
- *Substitution justification*: pretraining loss is **deliberately frame-distinct** from production. The point of pretraining is to seed weights with field-realism that the production loss cannot teach. SAME-frame pretraining (α) was REJECTED above on the grounds that frame-match invites the (3)-falsified pathology to inherit through the forward map. **Symmetric-disclosure clause per [D-37] rule 5**: this is a deliberate frame-mismatch, declared at design time; the gate ladder (below) carries the cross-frame measurement burden, NOT the pretrain loss itself.

**Gate ladder (three milestones, pre-committed; M3 REVISED per K3):**

| Milestone | Step | Metric | PASS | MARGINAL | FAIL |
|---|---|---|---|---|---|
| M1 — pretrain convergence | 1000 | $R_{\rm pre} = \mathcal{L}_{\rm pre}({\rm step\,1000}) / \mathcal{L}_{\rm pre}({\rm step\,0})$ | $\le 0.1$ | $0.1 < R \le 0.5$ | $> 0.5$ |
| M2 — pretrain saturation | 5000 | $R_{\rm sat} = \mathcal{L}_{\rm pre}({\rm step\,5000}) / \mathcal{L}_{\rm pre}({\rm step\,1000})$ | $\le 0.5$ | $0.5 < R \le 0.9$ | $> 0.9$ (non-convergent) |
| M3 — density-realism handoff (CALIBRATED) | 5000 | $R_{\rm real} = {\rm Var}[\log_{10}(\rho_\theta + 10^{-3})] / {\rm Var}[\log_{10}(\rho_{\rm truth} + 10^{-3})]$ over 100 crops at crop_size=48³ | $\in [0.980, 1.021]$ | $\in [0.960, 0.980) \cup (1.021, 1.041]$ | outside $[0.960, 1.041]$ |

**M3 provenance footnote (NEW, K3 absorption, verbatim):**
> The PASS band on R_real = Var[log10(ρ_θ + 10⁻³)] / Var[log10(ρ_truth + 10⁻³)] is calibrated to the sampling-noise floor of the 100-crop two-pass estimator used to score the gate. Using the P1 z=0.3 ρ/⟨ρ⟩ field (n_grid=768, CIC-deposited; cache npy MD5 0d9ca217d039b8da8008feb1ea5ff45d), we draw 1000 paired (numerator, denominator) bootstrap resamples — each a fresh 100-crop sample at crop_size=48³ — from the same underlying ρ-field and compute the null empirical distribution of the ratio. We find μ=1.0004, σ=0.0203 (5th–95th percentiles [0.962, 1.042]). PASS = R_real ∈ [μ−σ, μ+σ] = [0.980, 1.021]; MARGINAL = R_real ∈ ([μ−2σ, μ−σ] ∪ [μ+σ, μ+2σ]); FAIL = outside [μ−2σ, μ+2σ] = [0.960, 1.041]. Script: `scripts/d69_m3_band_calibration.py`; artifact: `experiments/nerf/artifacts/d69_m3_band_calibration.json`; seeds: SEED_BASE_SINGLE=100000, SEED_BASE_RATIO_NUM=200000, SEED_BASE_RATIO_DEN=300000.

**M3 crop-size footnote (NEW, K3 absorption sub-choice):**
> Crop size = 48³ matches the sprint-5 (c′)-at-48³ substrate per [D-56]; calibration was performed at the same scale. The L=32³ alternative is calibrated in the same artifact (PASS [0.972, 1.028]) and remains usable if a future Stage 1b/c needs a faster pre-flight, but the load-bearing Stage-1a M3 gate is at 48³ for substrate consistency.

**M3 framing-rejection note (NEW, K3 absorption).** The linear-variance framing R_real_lin = Var[ρ_θ] / Var[ρ_truth] was REJECTED for gate use per K3 calibration: filament-tail voxels skew the null distribution to μ=1.45 ± 1.44 at L=48 (highly non-symmetric); symmetric bands are inappropriate. The log10 framing above is the gate framing.

**μ±σ vs IQR PASS-band footnote (Rev-3 B2 SHOULD absorbed).** μ±σ chosen over IQR p5–p95 to favor type-I (false-PASS) avoidance over type-II (false-MARGINAL ~32% by bootstrap). MARGINAL triggers diagnostic review per §6 routing, not stage failure; type-II cost bounded.

**Routing (REVISED per K4 + Rev-3 B1 unconditional):**
- **Stage 1a (P1) all PASS (M1 + M2 + M3)** + **Stage 1b (P2 M3 re-check, UNCONDITIONAL) PASS** → handoff to Stage 2 (L1-fine-tune + EWC pipeline; separate next-stage spec).
- **Stage 1a ANY FAIL** OR **Stage 1b FAIL** → close (2)-pretraining at Stage 1; escalate to (1)-reframed per [D-62] candidate ladder.
- M1 MARGINAL → continue to M2 (slow convergence is acceptable if monotone).
- M2 MARGINAL → continue to M3 with caveat (plateau, not full saturation).
- M3 MARGINAL → STOP-and-defense-panel-re-review (the variance-realism gate is the load-bearing pre-fine-tune signal; ambiguity here cannot be auto-routed).

**Stage 1b spec (Rev-5 absorbed panel SERIOUS #1 as two-sub-step ordering).** Runs unconditionally on Stage 1a outcome (per Rev-3 B1; data-engineer P2 ρ-field extraction unconditional parallel to Stage 1a launch). Decomposes into two sub-steps:

- **Sub-step 1 — Zero-shot M3 evaluation**: Load Stage 1a P1 checkpoint, run M3 realism evaluation against P2 ground-truth with **NO weight updates**. PASS band `[0.980, 1.021]` confirms *transfer* of the pretrained density basin across physics conditions. FAIL or BOUND-only triggers the interpretive narrowing in sub-step 2's framing.
- **Sub-step 2 — Fine-tune M3 evaluation (secondary)**: 5k AdamW steps on P2, re-eval M3. Interpretive claim is **narrowed conditional on sub-step 1 verdict**: PASS in sub-step 1 → "P1-pretrain generalizes cross-physics, fine-tune sharpens"; FAIL in sub-step 1 → "P1-init basin is fine-tunable to P2 in 5k steps" (panel's option-b fallback wording, used only if zero-shot fails).

Sub-step ordering converts panel SERIOUS #1's either/or (zero-shot vs fine-tune-confounded-with-refit) into a sequential test that yields *both* claims' evidence in one Stage 1b run. Additive cost ~15 min Juno A30 (sub-step 1 = one eval pass; sub-step 2 = 5k AdamW steps). Total Stage 1 budget ~30 min.

**R-b backstop equivalent (anti-degeneracy, mirroring [D-60] B-stack):**
- **R-b-pre1**: if at step 1000 $\rho_\theta$ converges to a constant (Var$(\rho_\theta) < 0.1 \cdot$ Var$(\rho_{\rm truth})$) — close as "constant-density basin" (mirror of [D-40]/[D-41] failure class on the pretrain side).
- **R-b-pre2**: if pretrain loss decreases but $R_{\rm real}$ at M3 is outside the calibrated [0.960, 1.041] FAIL band — close as "loss-decreased-but-realism-failed" (the slice-MSE-only pathology R13 flagged in §1).
- **R-b-pre3**: if NaN/Inf in $\rho_\theta$ output for $> 0.1\%$ of voxels at any checkpoint — close as "numerical instability" (Softplus + log-domain edge case).

**[D-37]-Ext R2 cascade verb-downgrade applied:** (3) fGPA-residual was the immediate-prior same-track same-confidence falsification (this morning). This (2)-scoping is the **first scoping of (2) under post-(3)-falsification framing**. Verb level downgraded one rung: NOT "principled escalation" / "structurally correct pivot" / "addresses the diagnosed pathology directly." The phrasing throughout is "candidate", "first test of", "may seed weights with field-realism, untested for IGM-NeRF" — explicit hedged-verb register.

### §2.5 Diagnostic: per-bin loss decomposition (NEW, S3 absorption)

At M2 evaluation (step 5000), report log-MSE binned by truth-log-δ in 4 bins:
- Bin A: $\delta < 0.1$ (void)
- Bin B: $0.1 \le \delta < 1$ (mean-density)
- Bin C: $1 \le \delta < 10$ (overdense)
- Bin D: $\delta \ge 10$ (filament/halo tail)

Per-bin breakdown surfaces whether filament-tail voxels (Bin D) dominate the loss or are under-represented. **Pre-committed MARGINAL triggers per Rev-3 B3 absorption** (panel SERIOUS amendment-3):
- **(a) Generic dispersion**: `max_i(log-MSE_bin_i) / median_i(log-MSE_bin_i) > 5.0` at M2 evaluation → Stage 1 close downgrades to MARGINAL regardless of M3 aggregate PASS.
- **(b) Panel-specific Bin D vs Bin B**: `log-MSE(Bin D) > 5 × log-MSE(Bin B)` at M2 → Stage 1 MARGINAL.

Either trigger fires MARGINAL. Pre-commitment per [D-37] honest-framing; post-hoc judgment forbidden ("diagnostic-only" without pre-committed thresholds was the Rev-2 escape hatch). 5× anchor rationale: log-MSE is log-scale so 5× = ~0.7 dex spread, well outside expected per-bin σ from sampling noise at the production n_rays. Informs Stage 2 reweighting design if filament-bin loss dominates Bin B (the diffuse-bin majority of voxels). The [D-41] FGPA-tail anti-pattern explicitly warned that diffuse-bin-dominant losses leave the tail unconstrained; this diagnostic + pre-committed threshold checks for the symmetric pre-fine-tune failure mode.

### §2.6 Pretrain optimizer config (Rev-3 C4 absorption — FACTUAL CORRECTION)

- Optimizer: **AdamW** (betas=0.9/0.999, weight_decay=1e-6), lr_max=5e-4, lr_min=5e-6, linear warmup 1000 steps then cosine decay to lr_min over `max_steps`=5000.
- Microbatch: 1024 voxel samples per step from random crops (matches §4 below).
- Rationale: **matches `experiments/nerf/pipeline.py:1022` + `:71` + `:287-300` production config (R30-verified 2026-05-24 amendment-3 cycle).** Rev-2 had factually wrong "Adam, no warmup" claim; corrected at Rev-3 per panel C4 KILLER amendment.
- **Stage 2 carry-flag**: AdamW weight_decay × Softplus pre-activation interaction is a known degeneracy class (own audit at Stage 2 design); flagged here so the Stage-2 design picks up the audit obligation.

---

## §3 Data-locality audit

- **P1 z=0.3 n_grid=768 ρ-field cache**: VERIFIED LOCAL in this session at `Sherwood\.rho_field_cache\rho_field_p1_z0.300_n768.npy` + manifest sidecar (R15 clause (c) in-session re-verification satisfied).
- **P1 z=0.3 n_grid=64 ρ-field cache**: VERIFIED LOCAL (faster-iteration substrate for CPU pre-flight).
- **P2/P3/P4 ρ-field caches**: NOT present locally. Per [D-51] 2026-05-13c, tarballs are local at `SherwoodIGM_gal/*.tar.gz` but P2/P3/P4 are un-extracted; extraction + CIC deposition pipeline per [D-50] required.
- **Stage 1a scope-lock**: P1 z=0.3 SUFFICIENT. The L1 production failure is cross-physics-invariant per [D-39]/[D-65]; demonstrating pretraining viability on P1 is the Stage 1a question.
- **Stage 1b scope (REVISED per Rev-3 B1 UNCONDITIONAL)**: P2 z=0.3 ρ-field extraction + cache build REQUIRED. **Data-engineer dispatch UNCONDITIONAL parallel to Stage 1a launch** (Rev-3 amendment-4 consistency-fix; previous "conditional on Stage 1a PASS" wording at Rev 2 contradicts the §2 B1 amendment-3 routing and is corrected). Stage 1b runs regardless of Stage 1a outcome to provide cross-physics evidence.
- **Stage 2 scope (out of scope here)**: full 4-physics generalization + EWC + fine-tune pipeline.

---

## §4 Compute-shape estimate (REVISED per S2)

- **Stage 1a compute target**: Juno A30, P1 z=0.3, n_grid=768 ρ-crop substrate.
- **Pretrain step budget**: 5000 steps (cap from M2 saturation gate). Microbatch = 1024 voxel samples per step from random crops.
- **Wall-clock estimate (REVISED)**: **~8 min Juno A30** realistic wall-clock. Data-loading bound, not GPU bound: MLP forward+loss+backward = 28.5 ms/step CPU per dry-run; GPU compute extrapolation 1.4–4.8 s GPU-only across the full 5000 steps; crop-draw assembly dominates at ~100 ms/step × 5000 steps = ~8 min. Source: `scripts/d69_m3_band_calibration.py` 10-step CPU dry-run (per K3 calibration session). ~~STRIKE the original 15-min estimate~~ (was based on a Voigt-loaded per-step extrapolation that does not apply here — pretrain forward is MLP-only).
- **Implication**: compute is essentially trivial. The infra-manager dispatch shape is "1 cell, 1 GPU, <15 min wall, plus margin"; cost-control discipline still applies (auto-stop, S3 lifecycle), but the budget worry from panel S2 (30–60 min) is discharged.
- **VRAM**: MLP-only forward, no Voigt intermediate tensors → ~3–5 GB VRAM headroom (well under [D-23] ceiling).
- **Dispatch shape**: **1-cell smoke (P1 only)**, NOT 4-physics dispatch. Stage 1b is conditional (P2-only re-check); Stage 2 (4-physics generalization) gates on Stage 1a+1b PASS first.
- **Estimator-equivalence test (per [D-60] precedent)**: re-certify after pretraining-stage code lands; the MLP density head is unchanged but the loss path is new, so the equivalence test must verify autograd through the log-MSE path on a fixed seed before any Juno commitment.

---

## §5 R-rule audit at [D-69]-stage-1 (2)-scoping (REVISED at amendment-1)

- **R8 (cascade-close formality)**: HOLDS, unchanged from Revision 1. (2)-candidate-class is one slot in the explicit [D-62] candidate ladder; no completeness claim made here.
- **R13 (scope-policing)**: HOLDS, **RE-AUDITED at Rev 2**. Stage-1-narrowing (K1) reduces R13 surface area: the doc no longer makes any fine-tune-survival binding claim, only a pretrain-saturation+realism feasibility claim. R13 risk is consequently *reduced*, not eliminated — the residual risk is M3 PASS being misread as "pretrain (2) works" when it only binds "pretrain leg works in isolation". §1 PI-pick text explicitly narrows the claim to Stage-1-only.
- **R15 (PI sign-off PROVISIONAL by default)**: **STILL PROVISIONAL**. Lifted on panel APPROVE-on-revised. R15 clause (c) inherits from Revision 1 (data-locality re-verification); the substantive design choice (γ over α/β, K3 calibrated band, K4 Stage-1a/1b split) is panel-bound.
- **R26 (in-session re-verification of inherited claims)**: SATISFIED at Rev 1 and **inaugurally tested for R30 at Rev 2** — the K2 absorption table required a fresh re-grep of `D62_architectural_pivot_scoping.md` lines 82/84/88 before the §0.5 table was authored; that re-grep was performed (see §R26 block above) and matched the cited content.
- **R27 (5-stage ladder)**: HOLDS, unchanged.
- **R28 (PI dispatch sequence PROVISIONAL by default)**: HOLDS, unchanged.
- **R29 (gate-construction-vs-production-framing audit IN-LINE)**: HOLDS, applied at §2 frame-audit block. Unchanged at Rev 2 (M3 framing-rejection note inherits the same discipline).

### §5.5 R30 BANKED at amendment-1 (NEW)

**R30 — verbatim rule form:** *"PI absorption of defense-panel verdict requires one file-grep re-verification of cited parent-doc lines before issuing absorption table."*

**Banking trigger**: amendment-1 K2 step required asserting `D62_architectural_pivot_scoping.md` lines 82 / 84 / 88 in an absorption mapping table. Inherited-claim discipline (R15 clause c, post-[D-37]-Ext-R15 banking precedent) requires the asserting PI to independently re-establish the cited line content before publishing the absorption — exactly the failure mode that motivated R15 expansion to include re-verification of inherited claims.

**First operational test (this absorption cycle)**: R26 re-verification of K1/K2 against `D62_architectural_pivot_scoping.md` lines 82, 84, 88 was performed (§R26 block above documents the re-read) before §0.5 absorption table was authored. R30 SATISFIED at inaugural test.

**Future scope**: applies to every PI absorption-table issuance under any defense-panel verdict from this point forward. The discipline cost is trivial (one Read or Grep per cited line); the failure-mode cost it prevents (absorbing a verdict line that mis-cites the parent doc, propagating the mis-citation downstream) is high.

---

## §6 Defense-panel handoff — KILLER-target discharge status (REVISED per amendment-1)

Original panel KILLER-targets, with discharge status:

1. **PI-pick of (γ) over (α)/(β)** — frame-match-inheritance mechanism claim. **DISCHARGED** at amendment-1 K1: PI-pick rationale narrowed to "frame-mismatch prevents inheritance of (3)-falsified pathology at the pretrain stage; fine-tune-survival is Stage 2 binding question, untested here." Cross-ref: §1 revised PI-pick block.

2. **M3 variance-realism gate range `[0.7, 1.3]` PASS** — informal expert-prior provenance. **DISCHARGED** at amendment-1 K3: band replaced with calibrated `[0.980, 1.021]` PASS / `[0.960, 1.041]` FAIL-bound at L=48³, verbatim provenance footnote in §2 M3 row. Calibration artifact: `experiments/nerf/artifacts/d69_m3_band_calibration.json`.

3. **Frame-mismatch design choice (§2)** — is the (3)→(2) cascade actually frame-match-motivated or just "anything other than (3)"? **DISCHARGED** at amendment-1 K1 + S1: revised PI-pick rationale is explicit and narrow (frame-mismatch prevents Stage-1 inheritance; says nothing about Stage 2); S1 forward-justification cites Erhan+ 2010 / Bengio+ 2007 with honest "thin-precedent for IGM-NeRF specifically" hedge.

4. **R30-candidate rule status (§5)** — bank or defer? **DISCHARGED** at amendment-1 R30 BANK: §5.5 banks R30 with verbatim rule form and inaugural operational test.

**New panel attack surface introduced by amendment-1** (panel re-review should focus here):
- **K4 Stage 1a/1b split** — is the "Stage 1a FAIL → no P2 dispatch" routing actually a budget-saver, or is it a sneaky way to avoid disconfirming evidence from a second physics? Panel should adversarially probe whether single-physics close is genuinely sufficient to falsify.
- **K3 calibrated band semantics** — μ ± 1σ as PASS may be too tight (5th-95th percentile is [0.962, 1.042]; the σ-band is narrower than the empirical inter-quantile range). Panel should probe whether the bootstrap-σ is the right summary statistic or whether percentile bands are more honest.
- **S3 per-bin diagnostic** — is "diagnostic-only, not a gate" a [D-37] symmetric-disclosure escape hatch? If the Bin D filament-tail loss is 10× Bin B's, is that *really* not a gate signal?
- **R30 banking status — honest n=1 (Rev-3 amendment-4 correction).** Rev-3 amendment-3 claimed n=2 discharge by counting the pre-absorption re-grep of panel C4 inside the same absorption cycle that challenged R30. Panel Rev-3 re-review (M2 SERIOUS) correctly flagged this as self-referential: the re-grep was a *consequence of* the panel B4 challenge, not an independent operational test in a distinct design-doc absorption event. R30 status corrected to **BANKED at honest n=1**; genuine n=2 awaits the next absorption-table issuance in a future, unrelated design-doc cycle. (The R30 rule itself remains BANKED — the discipline is procedurally trivial and high-value; only the n-count is corrected.)

**R15 PROVISIONAL until panel re-reviews this revised doc**.

---

## §7 Next dispatch (single recommendation per `feedback-pi-decides`) — REVISED per amendment-1

**DISPATCH: `defense-panel` adversarial re-review on this Revision-2 doc.**

*Brief*: 4-persona adversarial review (per [D-42-meta] precedent), focused on (i) discharge-status verification for the four original KILLERs (§6 table), (ii) KILLER attacks against the four new attack surfaces introduced by amendment-1 (§6 last block), (iii) verdict on whether Revision 2 is *dispatch-ready* for core-implementer + infra-manager. Returns APPROVE / NEEDS-WORK / BLOCK per [D-60] gate-2 precedent.

Sequencing rationale: per `feedback-pi-decides` and per the [D-69] amendment-1 dispatch clause, the panel re-review on revised doc is the gating step before any other agent is commissioned. **NOT yet infra-manager, NOT yet data-engineer, NOT yet core-implementer.**

**Next-Dispatch chain** (CONDITIONAL on panel APPROVE):
1. **defense-panel re-review on Revision 2** ← CURRENT
2. core-implementer: pretrain code in `pipeline.py` behind `--pretrain-density` flag, gate-ladder harness mirroring [D-60] B-stack, estimator-equivalence test, CPU pre-flight on P1 n_grid=64, S3 per-bin diagnostic instrumented in M2 eval
3. infrastructure-manager: sbatch authoring with binding 4-item pre-validation checklist per `feedback-infra-sbatch-pre-validation`
4. Juno Stage 1a run on P1 → M1/M2/M3/S3-diagnostic results
5. **data-engineer dispatch UNCONDITIONAL parallel to Stage 1a launch** (Rev-3 amendment-4 consistency-fix per panel B1 amendment-3 — P2 ρ-field cache build for Stage 1b runs regardless of Stage 1a outcome)
6. Juno Stage 1b run on P2 → M3 re-check
7. Stage 1b PASS → Stage 2 design-doc (EWC + fine-tune + prior-decay gate per [D-62] L84/L88)
8. Stage 1a or Stage 1b FAIL → close (2)-candidate at appropriate granularity; escalate to (1)-reframed per [D-62] ladder.

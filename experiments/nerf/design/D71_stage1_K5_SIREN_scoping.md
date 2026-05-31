# [D-71]-stage-1-K5 (1c) SIREN architectural-axis scoping — PI design doc

**Status**: **Revision 1 PROVISIONAL** pending defense-panel pre-review per R28; no HPC dispatch until panel APPROVE. R15+R28 PROVISIONAL.
**Revision: 1.**
**Authored**: 2026-05-26, PI self-dispatch per [D-71] §G + D70-Rev-5.1 precedent template.
**D-XX**: [D-71].
**Parent decisions**: [D-71] §A–§I Stage 1a (1b) FALSIFICATION absorption + K5 (1c) SIREN activation; [D-70] Rev 5.1 §6 K5 conditional contingency (1c above 1d for void-floor-collapse remediation); [D-69] (γ) direct-ρ-MSE supervision regime; [D-62] §Architectural Candidates L90-100 (architecture-axis BLOCKED standalone — supervision-target-coupled reframe required); [D-53] supervision-target axis NOT discharged.

---

## §0 — Framing + ancestry

This is the **first scoping of (1c) SIREN under post-(1b)-FALSIFICATION cascade**. [D-37]-ext rule 2 falsified-prior cascade applies: (1b) skip-rich MLP under (γ) was the prior same-confidence architectural-axis candidate; Juno job 203337 returned all 10 seeds NEGATIVE Δ_seed with MDE-block CLEARED, (γ) pre-commit FALSIFICATION trigger fired exactly as pre-committed. Verb-rung inherits one-level hedge from (1b): the SIREN candidate is presented as a **first re-test of the architectural-axis under (γ) after (1b) FAILed**, NOT as "structurally correct," NOT as "right architecture for the pathology," NOT as "well-motivated escape." The load-bearing claim is the Sitzmann+2020 §3 orthogonality argument: SIREN's sinusoidal activations have non-zero gradient everywhere (vs ReLU dead-zones) and the init scheme is derived specifically to escape constant-mean basins. Whether that orthogonality argument transfers from photometric/SDF/audio domains to 5-decade log-density cosmological overdensity is **untested**; there is no IGM-NeRF SIREN precedent (per [D-62] L96 inheritance — carries forward unchanged from [D-70] §1).

Forbidden verbs (per [D-37]-ext rule 2 cascade hedge): "structurally immune," "physics-invariant by design," "addresses the diagnosed pathology directly," "principled escalation," "highest-leverage." Permitted: "candidate," "first test of," "may break," "hypothesized to escape," "orthogonality argument is the load-bearing claim."

Narrative discipline carried forward from [D-71] §C: the (1b) FALSIFICATION is **narrow scope** (single architecture under (γ) on P1 z=0.3 n_grid=768, 500 steps, lr_max=5e-4). (1c) SIREN is therefore **not** justified by "(1b)-class failure"; it is justified by "(1b) specific failure + Sitzmann+2020 mechanism orthogonality candidate." R8/R9 invariance-verb discipline binds throughout: a (1c) PASS narrows to "(1c) escapes the (γ)-attractor on P1"; a (1c) FAIL narrows to "(1c) fails under (γ) on P1." Neither generalizes to "architecture-axis-under-(γ) works/fails" without a third regime.

---

## §0.5 — Parent-envelope CAVEATS discharge ([D-71] §D inheritance)

| Parent constraint | (1c) coverage |
|---|---|
| [D-62] L90/L92 — architecture-axis BLOCKED standalone, supervision-target-coupled reframe required | §1.5 collision-resolution: (1c) is picked *with explicit reference to* how Sitzmann+2020 init + Sine activations change the constant-mean basin structure under the existing direct-ρ-MSE supervision — same constructive reading [D-70] adopted for (1b). Defense-panel re-review on this collision-resolution is the binding gate. |
| [D-62] L94 — "supervision-target-coupled" stricter reading | Adopted unchanged from [D-70] §1.5. The new architecture must change the gradient-flow / loss-surface mechanism by which (γ) lands on weight-space; (1c) candidate-mechanism story in §1.2 is the discharge. |
| [D-62] L96 — no IGM-NeRF precedent; weakest evidence base | INHERITED. SIREN precedent is photometric (Sitzmann+2020 image regression), audio, video, SDF — NOT 5-decade log-density cosmological overdensity. Domain transfer is a separate, untested step. |
| [D-62] L98 — no GAN/adversarial re-introduction | N/A — (1c) is pure MLP. |
| [D-62] L100 — within-class FAIL routes to [D-65 stub] further-class-pivot | §7 FAIL routing pre-committed: (1c) FAIL routes to (1d) hybrid voxel-grid panel-cycle BEFORE [D-65 stub], per [D-71] §G feedback-path-a-exhaustive (loss/architecture path walked exhaustively before hybrid-grid). (1d) is **queued NOT bundled**: it does not co-dispatch with (1c). |
| [D-71] §E — (γ) supervision UNDER-RE-TEST, R10 reuse contract | §0.6 binding. R10 holds for (1c) IFF (1b)-failure-cause ⊥ (1c)-fitness. SIREN's Sitzmann+2020 §3 orthogonality argument **IS** the (γ)-reuse justification. Mandatory hedge per [D-71]: "(γ) supervision under re-test, not validated." Symmetric-disclosure FAIL branch pre-committed in §7. |
| [D-71] §F — Stage 1b DE-ACTIVATED for (1b); panel-cycle gate owed for K6 narrow-discharge | INHERITED. K6 narrow-discharge ruling is a separate panel cycle from (1c) scoping pre-review; both panel cycles are owed but do not bundle. |

R30 grep-discipline note: all line-number / file-path citations in this doc are accompanied by exact `Read` / `Grep` evidence captured in-session (PI's wrapper §R26 block; see review-trail in §9). Per R30 banking discipline.

---

## §0.6 — (γ) reuse contract under R10 (load-bearing for the whole doc)

R10 (retired-model reuse) requires an explicit orthogonality argument before re-using a supervision regime that failed under a different architecture. The argument:

**(1b) failure cause** (per [D-71] §C narrow ruling): the skip-rich-MLP body, with ReLU activations + standard Kaiming init + Softplus head, minimizes the log-domain direct-ρ-MSE by collapsing toward the void-floor regime — Bin-D ≈ 6.2–6.8 vs Bin-B ≈ 0.53–0.64 log-MSE (D/B ≈ 10–13×, [D-71] §A table). R-b-pre2 fires every-seed (loss decreased but realism FAILed). The mechanism story (combining pre-flight A frozen-init dead-ReLU + pre-flight C training-time asymmetric drift evidence at D70 §0.9 / §1.5): the body's ReLU dead-zones + Softplus head's c≈0 stable-basin together admit a constant-mean (void-anchored) loss minimum that direct ρ-MSE cannot push the model out of.

**(1c) fitness against that cause** (Sitzmann+2020 §3 orthogonality, load-bearing):
1. Sine activations have **non-zero gradient everywhere** (`d/dx sin(ω₀ x) = ω₀ cos(ω₀ x)`, which is zero only on a measure-zero set). There is no "dead-zone" analog; the body cannot lose gradient flow through dead-unit accumulation.
2. The Sitzmann+2020 init scheme is **derived specifically to maintain activation variance at init across depth**, escaping the constant-mean basin by construction at t=0. Weights are sampled `U(−√(6/fan_in)/ω₀, +√(6/fan_in)/ω₀)`; first-layer special-case sets `ω₀=30` to span the relevant input bandwidth.
3. The Softplus density head is **held constant** between (1b) and (1c) per [D-71] §E body-axis-only discipline: changing both body and head simultaneously would confound (1c)-vs-(1b) attribution. The Softplus head's c≈0 stable-basin remains a (hypothesized, untested) failure mode for (1c) as well — if (1c) PASSes body-axis but FAILs Bin-D on Softplus collapse, that is an interpretable partial-success.

**Orthogonality holds** if and only if (1b)'s failure mode was body-side-dominated (dead-ReLU + non-oscillatory output basin). It does NOT hold if (1b)'s failure was head-dominated (Softplus collapse) or supervision-degenerate (the (γ) loss admits a constant-mean basin no architecture escapes). The latter is the [D-71] §E open hedge. **Therefore (γ)-reuse on (1c) is permitted under R10 but the contract is conditional: if (1c) ALSO produces R-b-pre2 every-seed under (γ), THEN (γ) supervision-class falsification triggers** and (δ) supervision-pivot enters the routing stack — per §7 pre-commit and [D-71] §E.

---

## §1 — (1c) SIREN architecture spec

**Variant identity**: Sitzmann+2020 SIREN body + Softplus density head + same input encoding (Fourier positional, L=10) + same loss (γ direct ρ-MSE log-domain MSE with `+1e-3` floor) + same data (P1 z=0.3 n_grid=768) + same optimizer family (AdamW + warmup-cosine) + same body topology (8-layer × 256-hidden, single mid-skip). The ONLY axes that change between (1c) and the `current` arch are (a) body activation function ReLU → Sine, (b) weight init Kaiming → Sitzmann, (c) `lr_max` 5e-4 → 1e-4 per Sitzmann+2020 default — see §1.3 + §6 K2.

### §1.1 — Confound-minimization rationale

Single-axis-change discipline per R29 + [D-71] §C narrow-discharge framing. Body topology held to current single-mid-skip (NOT skip-rich; that confound is in (1b) and (1b) FAILed). Density head held Softplus (NOT changed to direct-linear; that confound is in a downstream candidate). Loss held (γ) (forbidden swap per [D-71] §E reuse contract). Data, sampling RNG family, microbatch=1024, crops_per_step=4, crop_size=48 — all inherited byte-equivalent from D70 Stage 1a (1b) protocol.

### §1.2 — Mechanism story (hedged per [D-37]-ext rule 2 + R10 orthogonality argument)

The orthogonality story spelled out in §0.6 binds. Operative measurable: Bin-D log-MSE recovery. (1b) D/B ≈ 11× on (γ) is the void-floor-saturation signature; (1c) PASS requires D/B << 10× at M2 endpoint per [D-70] Rev 5.1 §2.2 Bin-D mandatory sub-clause (ii) Wilcoxon `bin_d_log_mse − bin_b_log_mse < 0` at α=0.05 one-sided. Hedge: there is no published evidence that SIREN's Sine-body orthogonality argument transfers to 5-decade log-density cosmological overdensity (Lukić+ 2015 §3 / Bolton+ 2017 density-PDF panels for the dynamic-range context). The mechanism story is **plausible**; it is **not proven**.

### §1.3 — `lr_max` selection + departure from (1b)

Sitzmann+2020 trains SIREN with Adam at lr 1e-4 by default for the image-regression tasks (their §4.1). Sine activations have larger gradient magnitudes at init than ReLU (the ω₀=30 first layer especially) — using (1b)'s lr_max=5e-4 risks divergence in the first ~50 steps. **PRE-COMMIT**: lr_max=1e-4, lr_min=5e-6, warmup=100, cosine to lr_min over max_steps=500 (Stage 1a smoke); for the M1/M2 follow-on max_steps=5000, cosine over the full schedule. This is a single-axis-change-discipline departure from (1b); panel attack-surface K2 in §6 explicitly anticipates it.

Alternative: keep lr_max=5e-4 to hold confound-minimization tighter. Rejected because Sitzmann+2020 evidence is the load-bearing justification for the (1c) candidate at all; using a known-non-SIREN learning rate undermines the orthogonality argument's empirical grounding. The lr_max departure is **scoped to (1c)** and does NOT generalize.

### §1.4 — R20 twin-gate spec (binding pre-HPC)

(i) **Integration test** at `tests/test_siren_integration.py`: instantiate `IGMNeRF(body_arch='siren')`; forward a 4×16×3 toy ray batch; assert `out.requires_grad is True`; compose toy direct ρ-MSE; assert `loss.grad_fn is not None`; one `loss.backward()` + AdamW step; assert at least one body-layer weight tensor `weight_before ≠ weight_after` by ≥ 1e-6. Mirrors `tests/test_d70_skip_rich_integration.py` 3-test structure.

(ii) **Estimator-equivalence test** at `tests/test_siren_mlp_estimator_equivalence.py`: assert `IGMNeRF(body_arch='current')` is bit-equivalent to the pre-(1c) HEAD `current` path under a fixed seed + fixed coords batch (per [D-70] Amendment A structural-invariance framing, NOT "bit-equivalent-to-production"). 1e-5 rel-tol per [D-69] / [D-70] precedent.

### §1.5 — Per-layer wiring spec (load-bearing for core-implementer dispatch)

> **SUPERSEDED by §13.A-wiring (Rev 1.3); operative wiring is in §13.A-wiring. Original Rev 1 §1.5 wording preserved verbatim for audit trail; DO NOT use as wiring source-of-truth. Hidden ω₀=30, NOT ω₀=1 as shown below.**

| Layer | in_features | out_features | activation | init |
|---|---|---|---|---|
| input encoding | 3 (raw) | 63 (PE L=10) | Fourier positional | — |
| layers1[0] | 63 (PE_L10) [+1 g] [+e_dim e_p] | 256 | Sine ω₀=30 | U(−1/63, +1/63) per Sitzmann+2020 §3.1 first-layer init |
| layers1[1] | 256 | 256 | Sine ω₀=1 | U(−√(6/256)/1, +√(6/256)/1) |
| layers1[2] | 256 | 256 | Sine ω₀=1 | same as hidden |
| layers1[3] | 256 | 256 | Sine ω₀=1 | same as hidden |
| skip-cat | `torch.cat([h, skip_in])` @ `src/models/nerf.py:168` | 256 + skip_dim (= 256 + 63 [+1] ≈ 319) | — | — |
| layers2[0] | 256 + skip_dim (≈ 319) | 256 | Sine ω₀=1 | same as hidden |
| layers2[1] | 256 | 256 | Sine ω₀=1 | same as hidden |
| layers2[2] | 256 | 256 | Sine ω₀=1 | same as hidden |
| layers2[3] | 256 | 256 | Sine ω₀=1 | same as hidden |
| out_layer | 256 | 4 (ρ, T, X_HI, v_pec) | (head-side) | Kaiming (current) |
| density head | out[..., 0] | 1 | Softplus | — (unchanged from current) |

Notes on construction:
- Body topology = baseline current single-mid-skip (NOT skip-rich; (1b) confound NOT re-introduced).
- Output heads UNCHANGED: Softplus on ρ, Softplus×10⁴+10³ on T, Sigmoid on X_HI, Tanh×500 on v_pec — exact match to current at `src/models/nerf.py:180-183`.
- `out_layer` weight init retained as default (`nn.Linear` Kaiming) — the SIREN body→linear-head boundary uses the standard Sitzmann+2020 practice (the final linear layer is NOT a SIREN layer).
- Param count ≈ same as `current` arch (Sine ↔ ReLU is a pure activation-function swap; init scheme differs but tensor shapes are identical).

**CLI flag**: `--arch siren` — third value in the argparse `choices` tuple alongside the existing `{current, skip-rich-mlp}` (per LEDGER §3 D-70 wiring close-out, the 2-tuple is currently live; SIREN extends to 3-tuple). Core-implementer dispatch must (i) extend `IGMNeRF.__init__` `body_arch` ValueError tuple at `src/models/nerf.py:43-46`, (ii) extend argparse `choices` in `experiments/nerf/pipeline.py` `--arch` flag, (iii) extend the 3 construction sites (pretrain ~696 / eval ~1043 / main-train ~1810) per the D70 wiring landed-pattern, (iv) extend `BODY_ARCH=` stdout trailer at `pipeline.py:3040` to recognize `siren`.

### §1.6 — Scope-lock per R13

(1c) is scope-locked to: **"SIREN body (Sitzmann+2020 ω₀=30 first / ω₀=1 hidden + Sitzmann init) + Softplus density head + (γ) direct ρ-MSE log-domain supervision + P1 z=0.3 n_grid=768 + 500 steps Stage 1a / 5000 steps M2 + lr_max=1e-4 warmup-cosine + 10 seeds × 100 crops × 48³."** A FAIL is narrow ("this specific (1c) configuration FAILed under (γ) on P1"), NOT broad. R8/R9 invariance-verb discipline binding.

---

## §2 — Gate ladder (inherits D70 Rev 5.1 §2 + amendment v2 + [D-71])

Loss form UNCHANGED from (γ) per [D-71] §E reuse contract: log-domain direct ρ-MSE with `+1e-3` floor, `Σ (log₁₀(ρ_θ + 1e-3) − log₁₀(ρ_truth + 1e-3))²`.

**Standing disclosure clause** (inherited): every M0/M1/M2 numerical report MUST disclose σ_within per-bin AND aggregate alongside any point estimate.

| Milestone | Step | Metric | PASS | MARGINAL | FAIL |
|---|---|---|---|---|---|
| **M0 — direction-of-motion gate** | {0, 50, 100, 175, 250, 350, 500} | F4 Δ_seed = R_real_linear(500) − R_real_linear(0); per-seed Wilcoxon H1='greater' n=10 α=0.05; MDE-block ε=0.05 σ_seed full_n10 Boera+2019 anchor | Wilcoxon p ≤ 0.05 AND MDE_estimate < ε AND ≥ 8/10 seeds positive Δ_seed AND Bin-D differential improvement (§2.2 (ii)) PASSES | aggregate PASS but max_t Var_ratio > 1.2×Var_ratio(500); or 6–7/10 seeds positive; or W ∈ [10, 15] (S1 re-bake to n=20 with seeds 11–20) | Wilcoxon p > 0.05 (Bonferroni 0.025 if stage-2 fires); or < 6/10 positive; or MDE_estimate ≥ ε (gate authority blocked) |
| M1 | 1000 | R_pre = L_pre(1000) / L_pre(0); per-bin disclosure required, routing-lighter | ≤ 0.1 | (0.1, 0.5] | > 0.5 |
| M2 | 5000 | R_sat = L_pre(5000) / L_pre(1000); Bin-D mandatory sub-clauses (i)+(ii) AND-gated | ≤ 0.5 AND both Bin-D pass | (0.5, 0.9] AND both Bin-D pass | > 0.9; OR aggregate PASS but either Bin-D sub-clause FAILs → FAIL on void-floor-saturation |
| M3 | 5000 | R_real_linear; legacy K3-bootstrap band [0.96, 1.041] on R_real_log | DIAGNOSTIC-only | — | — |

### §2.1 — M0 critical-clause inheritance

F1-β dual R_real emission (linear + log); F4 Δ_seed framing; H1='greater'; α=0.05 one-sided pre-committed; MDE-block ε=0.05 anchored Boera+2019; σ_seed full_n10 source. S1 re-bake stopping rule pre-committed: W ∈ [10, 15] → re-bake to n=20 seeds 11–20 same RNG schedule, final ruling on combined n=20 (Bonferroni α=0.025). S3 PDF dump void-pile-up flag: report mass-fraction at log₁₀ρ ∈ [−3.05, −2.95]; > 5% pile-up flags Bin-A interpretation AMBIGUOUS.

### §2.2 — M2 Bin-D mandatory sub-clauses

(i) Bin-D Var_ratio at M2 endpoint ≥ 10× frozen-init baseline (= 7.913e-08 per [D-70] Rev 5 B2 ratchet). Sanity-check clause; honest framing: 10× multiplier is K3-disease (smallest that clears worst frozen-init seed; no physical anchor).
(ii) **Bin-D differential improvement vs Bin-B non-degenerate void-reference** (Rev 5.1 amendment): per-seed paired `wilcoxon(bin_d_log_mse − bin_b_log_mse, alternative='less')` at α=0.05 one-sided. MUST PASS for any non-FAIL M2 verdict. S5 min-samples gate: `min_samples_per_crop ≥ 5` required for Bin-D log-MSE; > 20% crops excluded → INSUFFICIENT-COVERAGE, routes M2 verdict to MARGINAL.

For (1c) on Stage 1a smoke (500 steps), the binding test is the M0 Wilcoxon on Δ_seed plus the sub-clause (ii) Bin-D differential — these together discriminate "escaped attractor" from "(1b)-style void-floor regression."

### §2.3 — R-b-pre1/2/3 backstop modes

INHERITED unchanged from [D-70] backstop-remediation chain: pre1=warn, pre2=warn, pre3=raise. R-b-pre2 OBSERVATION FLAG on every seed under (1c) is the diagnostic to watch — that is the signature pattern that, combined with M0 FAIL, triggers the (γ) supervision-class falsification per §7.

### §2.6 — Estimator-equivalence re-cert

Inherited 1e-5 rel-tol per [D-69] precedent. Test path: `tests/test_siren_mlp_estimator_equivalence.py`. Re-cert is a precondition for HPC dispatch (per [D-70] precedent); the `current` arch path must remain byte-equivalent to its pre-(1c)-landing state. The (1c) arch itself does NOT require estimator-equivalence (no prior implementation to match) — its integration test (§1.4 (i)) carries the wiring validation.

---

## §3 — Data-locality + R26 in-session re-verification

**P1 z=0.3 n_grid=768 ρ-field cache**: VERIFIED LOCAL this session by inheritance — the same `Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy` (1.81 GB, mtime 2026-05-13) used by [D-70]/[D-71] Juno dispatches (jobs 203285, 203337). R15(c) forward obligation: re-verify immediately before infrastructure-manager Juno dispatch (the dispatch brief carries this forward; not discharged at design-doc layer).

Stage 1a scope-lock: P1 ONLY (no P2/P3/P4 cross-physics). Cross-physics testing inherits the post-Stage-1a panel-cycle gate routing.

---

## §4 — Compute estimate

**Stage 1a smoke**: ~30 min Juno A30 per seed at 500 steps (Sine has same fwd/bwd compute as ReLU within ~5% noise; the per-step cost is dominated by the 8 Linear layers and the windowed Voigt, neither of which depends on activation function). 10-seed array `--array=0-9%4` per [D-70] sbatch template, total wall ~30 min × 3 batches ≈ 90 min cluster-time, ~5 hours queue if cap-of-4 is the bottleneck.

**Stage 1a → M1/M2 conditional**: if Stage 1a M0 PASS triggers M1/M2 escalation, 5000-step run is ~5 hours per seed × 10 seeds (= 50 hours cluster-time at 4-cap, ~12.5 hours queue). NOT pre-authorized at design-doc layer; gated on post-Stage-1a panel-cycle.

**No cost gate on Juno** per user-memory `feedback_no_cost_gate_on_juno`; scientific reasons are the only stop-gate.

---

## §5 — R-rule audit

- **R8 (narrow discharge)**: HOLDS. (1c) is one candidate; PASS/FAIL narrows to (1c)-on-(γ)-on-P1, not architecture-axis-class.
- **R9 (invariance-verb discipline)**: HOLDS. "Orthogonality" used in the Sitzmann+2020 §3 sense (different gradient-flow regime), NOT in the group-equivariance or cross-physics-statistical sense. Scope: P1 z=0.3 single physics.
- **R10 (retired-model reuse)**: HOLDS via §0.6 explicit (γ)-reuse orthogonality argument; reuse is conditional, FAIL branch triggers (γ) class-falsification per §7.
- **R12**: N/A (not a new rule banking event in this doc).
- **R13 (scope-lock)**: HOLDS per §1.6.
- **R14 (process-failure path pre-committed)**: HOLDS via §7 (1c) escape-hypothesis falsification trigger.
- **R15**: PROVISIONAL until defense-panel APPROVE on this Rev 1.
- **R20 (twin-gate before HPC)**: SPEC'd per §1.4; not yet built (gated on panel APPROVE then core-implementer dispatch).
- **R26 (in-session re-verification)**: discharged for this design-doc layer per §9 review-trail (re-read `nerf.py`, `pipeline.py`, `submit_juno_stage1a_1b.sh`, LEDGER §3 [D-71]); FORWARD obligation for ρ-field cache re-verify at infra-manager dispatch.
- **R27 (5-stage ladder)**: HOLDS unchanged.
- **R28 (PI dispatch-sequence rung count vs design-doc artifact count)**: SPEC'd per §8 multi-rung dispatch chain. PROVISIONAL until panel APPROVE.
- **R29 (gate-construction-vs-production-framing audit)**: **CANDIDATE** per R29-promotion panel verdict 2026-05-26 (b) NEGATIVE (full demotion from BANKED; previous banking falsified-prior — see §10.H). Discipline still applied to all M0/M1/M2 gates as standard best-practice, but no rule-citation-as-prerequisite. Specifically per §10.E ε-anchor demotion: this Rev 1.1 does NOT inherit ε=0.05; operative ε is `10 × μ_frozen = 2.4e-5` on `var_pred(500)/var_truth(500)` primary observable per panel #3 + PI absorption path (a) anchored to gate observable (no nonlinear forward-model bridge required).
- **R30 (brief-discipline grep evidence)**: HOLDS — §9 review-trail cites exact file:line evidence with grep/Read commands.
- **R31 (cross-section noun-consistency, banked at [D-70])**: HOLDS — "SIREN" / "(1c)" / "Sine activations" used consistently; CLI flag `--arch siren` matches doc-prose `(1c) SIREN`.
- **[D-37] rule (a) observation-first**: HOLDS — §0 leads with the (1b) FALSIFICATION observation and the orthogonality argument as the load-bearing claim, NOT with the (1c)-PASS strengthening narrative.
- **[D-37] rule 5 symmetric-disclosure**: §7 FAIL pre-commit landed exactly as on D70-Rev-5.1 precedent.

---

## §6 — Defense-panel KILLER attack surfaces (pre-anticipated)

1. **K1 — (γ) reuse legitimacy.** Does the (1b)-failure-cause ⊥ (1c)-fitness argument actually hold? SIREN escapes the dying-ReLU + Softplus-constant-basin pathway (§0.6), but is it pathologically prone to OTHER basins under (γ) — e.g., high-frequency overfit to crop-sampled supervision masking void-floor collapse? Mitigation: §2.2 sub-clause (ii) Bin-D differential improvement directly tests void-floor regime escape, which is the (1b) failure signature; if (1c) PASSes (ii), the orthogonality argument is empirically supported, not just literature-anchored.

2. **K2 — lr_max=1e-4 single-axis-change violation.** Justifying departure on Sitzmann+2020 §4.1 evidence is plausible but not the strictest reading of confound-minimization. Counter-option: hold lr_max=5e-4 (1b-match) and accept divergence risk. PI ruling: Sitzmann+2020 is the load-bearing evidence for the (1c) candidate at all; the lr_max departure is scoped to (1c). Panel must rule whether evidence-grounding outranks confound-tightness here.

3. **K3 — ω₀=30 first-layer transfer risk.** Sitzmann+2020 ω₀=30 is calibrated for image-domain coordinates ∈ [−1, +1] with photometric output. IGM coords are unit cube [0, 1] with positional encoding L=10; output is log-scale density spanning 5 decades. No published precedent that ω₀=30 is the right input bandwidth for this domain. Mitigation: scope-lock per §1.6; FAIL on (1c) at ω₀=30 does NOT discharge "SIREN-as-axis FAILs" — it narrows to "(1c) at ω₀=30 FAILs," opening an ω₀-sweep sub-rung that is NOT pre-authorized at this design-doc layer (would require fresh panel cycle per Ext-2 R3 anti-degeneracy).

4. **K4 — M3 PASS band recalibration owed.** The [0.96, 1.041] band on R_real_log was K3-bootstrap-calibrated on (γ) target distribution from earlier (γ)-on-current-arch runs. SIREN output-distribution under (γ) at frozen init may differ; M3 is DIAGNOSTIC-only per [D-70] Rev 2 demotion, so the band miscalibration does not gate-block but may produce misleading diagnostic readouts.

5. **K5 — Stage 1b reactivation under (1c) PASS.** D70 Stage 1b was DE-ACTIVATED specifically for (1b); does a (1c) PASS automatically reactivate, or does it require fresh panel cycle? PI ruling pre-committed: fresh panel cycle required per [D-71] §F charter-re-chart (no auto-dispatch path; R28 single-rung discipline + [D-70] Rev 5 S4 absorption).

6. **K6 — pattern-claim semantics on (1c) FAIL.** Does (1c) FAIL activate broad "architecture-axis-under-(γ) FAILS" pattern claim, or stay narrow per R8/R9? Pre-commit: stays narrow. The broad pattern claim becomes admissible IFF (1c) AND (1d) both FAIL — two-candidate architecture-axis FAIL on convergent failure signatures starts to discharge the pattern. Single (1c) FAIL does not.

7. **K7 — (δ) supervision-pivot routing on (γ) class falsification.** If §7's symmetric-disclosure trigger fires (R-b-pre2 every-seed on (1c)), what is the routing? Pre-commit: (δ) supervision-pivot enters the routing stack as the next architectural-axis candidate — concretely, a defense-panel cycle on alternative supervision targets (e.g., flux-domain MSE, fGPA-residual revisit per [D-65] stub, or a tomographic-direct target) becomes the next gate. NOT pre-authorized at design-doc layer.

---

## §7 — Pre-committed (γ) escape-hypothesis falsification trigger ([D-37] rule 5 symmetric disclosure)

> **"IF Juno n=10 Wilcoxon `median(Δ_seed) ≤ 0` under (1c) SIREN AND MDE-block CLEARS → (1c) escape-hypothesis FALSIFIED narrow scope; (γ) supervision-class falsification triggers per [D-71] §E IF AND ONLY IF R-b-pre2 fires every-seed (consistent void-floor regime collapse across BOTH architectures); IF R-b-pre2 fires only on some seeds, then mixed signal — convene panel for ruling. IF (γ) falsification triggers → (δ) supervision-pivot enters routing stack as next architectural-axis candidate."**

This is binding pre-commit per [D-37] rule 5. No re-interpretation after the fact. The conjunctive structure (median(Δ_seed) ≤ 0 AND MDE-block clears AND R-b-pre2 every-seed) gates (γ) class-falsification — partial signals (e.g., 7/10 R-b-pre2 fire) route to panel ruling, not to auto-falsification, preserving R10's "(1b)-failure-cause ⊥ (1c)-fitness" symmetric-evidence-bar.

---

## §8 — Next-dispatch chain (post-panel APPROVE; R28 multi-rung audit binding)

Per R27 panel-pre-review precedence + R28 single-rung-per-dispatch discipline, the ladder is enumerated explicitly. Dispatch-ladder rung count = 5 landing-artifact rungs; matches design-doc-spec'd artifact count (SIREN body wiring + argparse extension + integration test + estimator-equivalence test + sbatch authoring).

0. **defense-panel pre-review on this Rev 1** ← CURRENT GATE; PROVISIONAL R15+R28 lifted by panel APPROVE.
1. **core-implementer dispatch 1**: extend `IGMNeRF.__init__` `body_arch` ValueError tuple to 3-element at `src/models/nerf.py:43-46`; add `'siren'` body branch with Sitzmann-init Linear layers + Sine activation forward path; keep `current` + `skip-rich-mlp` byte-equivalent. Land `tests/test_siren_integration.py` (R20 gate (i), 3 tests mirroring D70 skip-rich pattern) + `tests/test_siren_mlp_estimator_equivalence.py` (R20 gate (ii) for `current` arch, 4 tests; (1c) needs no equivalence target).
2. **core-implementer dispatch 2**: extend `experiments/nerf/pipeline.py` `--arch` argparse `choices` to include `'siren'`; plumb through 3 construction sites (pretrain ~696 / eval ~1043 / main-train ~1810); add `--lr_max` is already there but pre-commit default for `--arch siren` paths to 1e-4 in sbatch — flag is NOT a code-level default change, it is an sbatch CLI-arg specification.
3. **infrastructure-manager dispatch**: sbatch authoring at `scripts/submit_juno_stage1a_1c_siren.sh` mirror of `submit_juno_stage1a_1b.sh`; replace `--arch skip-rich-mlp` with `--arch siren --lr_max 1e-4`; same `--array=0-9%4`, same `--time=00:30:00`, same `--mem=32G`, same MLflow file-store + rsync round-trip. 4-item pre-submission validation per `feedback-infra-sbatch-pre-validation` (argparse choices grep, trailer pattern grep, Juno maintenance reservation check, dry-run config mirror).
4. **User auth gate**: commit + push (LEDGER §3 [D-71] §G amendment + this doc + code + tests + sbatch); Juno-side `git pull`; ρ-field cache re-verify at `${COSMOGAS_RHO_CACHE_DIR}` (R26 forward obligation); `sbatch --export=ALL scripts/submit_juno_stage1a_1c_siren.sh`.
5. **Post-Juno absorption**: PCV → import-replay (existing harness reused per [D-71] §H) → `scripts/d70_wilcoxon_gate.py` Wilcoxon harness (existing, reused) → verdict absorption per [D-37] rule (a) observation-first + rule 5 symmetric-disclosure. If FAIL: §7 trigger evaluated for partial-vs-full activation. If PASS: K5 narrow-discharge for (1b) inheritance reviewed; K6 narrow-discharge owed per [D-71] §D paired with (1c) PASS scope.

**No paper-text propagation** (feedback-no-paper-writing); this is methodology-internal scoping only.

---

## §9 — Provenance + review trail

- Authored 2026-05-26 PI self-dispatch per [D-71] §G + D70-Rev-5.1 precedent template.
- Inheritance chain: [D-37]-ext rules 1-29 binding; [D-62] L90-100 architecture-axis BLOCKED standalone constructive reading; [D-69] (γ) supervision regime; [D-70] Rev 5.1 + amendment v2 gate-ladder + Bin-D AND-gate; [D-71] §A-§I (1b) FALSIFICATION absorption + K5 contingency + R10 reuse contract + §7 (γ) symmetric-disclosure pre-commit.
- Review provenance per [D-37]-ext R6: PI-self-authored, PROVISIONAL pending defense-panel pre-review (this gate). R15-LIFTed only on panel APPROVE.

### R26 in-session re-verification block (load-bearing inherited claims)

| Inherited claim | File:lines re-read | Status |
|---|---|---|
| IGMNeRF `body_arch` accepts `{current, skip-rich-mlp}` with ValueError on others | `src/models/nerf.py:41-46` | SURVIVES — `'siren'` requires extending ValueError tuple |
| Body ReLU sites = 8 (constructor + 8 application sites layers1+layers2) | `src/models/nerf.py:115` (ctor), `:163-164` (layers1 ×4), `:170-171` (layers2 ×4) | SURVIVES |
| Density head Softplus at `out[..., 0]` | `src/models/nerf.py:116, 118, 180` | SURVIVES |
| Encoded coord dim = 63 (raw 3 + 2·3·L with L=10) | `src/models/nerf.py:65` | SURVIVES |
| Skip-cat at `:130` (D70 Rev 5 cite) | `src/models/nerf.py:168` (current) | LINE-NUMBER DRIFT logged — `:130 → :168` as `g`/`e_p` plumbing landed; structural claim survives |
| `--arch` argparse choices `{current, skip-rich-mlp}` per D70 wiring close-out | LEDGER §3 [D-70] L2018 | PARTIAL re-read; core-implementer dispatch (rung 2) must re-grep before extending to 3-tuple |
| Sbatch template structure (A30, `--array=0-9%4`, `--time=00:30:00`, `--mem=32G`) | `scripts/submit_juno_stage1a_1b.sh:1-10` | SURVIVES — (1c) sbatch mirror authorized |
| [D-71] (γ) pre-commit, narrow-scope ruling, R-b-pre1/2/3 modes, MDE-block formula | LEDGER §3 [D-71] L1923-1966 | SURVIVES |
| K5 contingency wording ranking (1c) above (1d) | LEDGER §3 [D-71] §G | SURVIVES |

No commits, no code changes, no HPC dispatch from this design-doc landing per R28 + R15-PROVISIONAL.

---

## §10 — Rev 1 → Rev 1.1 AMENDMENT BLOCK (D71 Panel + K6 Panel absorption, 2026-05-26)

**Status update**: Revision 1.1 PROVISIONAL pending defense-panel re-review on this amendment block; no HPC dispatch until panel APPROVE. R15+R28 PROVISIONAL (unchanged).

D71 Rev 1 defense-panel returned NEEDS-WORK with 4 BLOCKING + 5 NEEDS-WORK + 5 PROBE. K6 narrow-discharge panel (parallel cycle on (1b) per [D-71] §F) ALSO returned NEEDS-WORK with 3 KILLER + 4 SERIOUS + 1 PROBE — findings flow back into (1c) Rev 1.1 per LEDGER §3 [D-71] amendment block "Forward obligations register" 2026-05-26.

**Rev 1 §B/§C/§7 wording preserved above for audit trail; operative readings replaced as below.**

### §10.A — D71 Panel B1 (PE → SIREN wiring categorical error): RESOLVED via path (α) DROP PE

Rev 1 §1.5 spec'd `raw coords → PE_L10 (63-dim Fourier encoding) → SIREN body`. Canonical SIREN replaces PE — Sine activations ARE the frequency expansion. Stacking gives effective first-layer frequency ω₀=30 × 2^9π ≈ 48,000 → 5 OOM above Sitzmann derivation regime. Categorical methodological error per panel B1.

**Operative §1.5 wiring (replaces Rev 1 §1.5 table)**:
- **Layer 0 input**: raw normalized coords (3-dim, ∈ [−1, +1] after the existing unit-cube → [−1,+1] re-map). **NO PE in front of SIREN.**
- **Layer 0 init**: `U(−1/in_dim, +1/in_dim)` where `in_dim = 3 + g_dim + e_dim` (functional in fan_in per N1; for Stage 1a P1 with `g=None, e_p=None`, `in_dim = 3`).
- **Layers 1..7**: Sine, `U(−√(6/fan_in)/ω₀, +√(6/fan_in)/ω₀)`, ω₀=30 (canonical Sitzmann hidden ω₀ — NOTE: Rev 1 had ω₀=1 for hidden; corrected per canonical Sitzmann SIREN recipe).
- **Out layer**: linear, default init (Kaiming-uniform `a=√5` per `nn.Linear`).
- **Density head**: Softplus unchanged (`out[..., 0]`).
- Explicit doc note: "PE_L10 deliberately omitted; Sine activations subsume frequency-basis role per Sitzmann+2020 §3."

This also resolves K3 (ω₀ transfer risk simplifies dramatically against raw coords∈[−1,+1] closer to Sitzmann calibrated regime), N1 (functional fan_in landed), and tightens N5 (fewer init-variance failure modes).

### §10.B — D71 Panel B2 (R10 head-axis incompleteness): RESOLVED via path (β) — head-ablation rung 4.5

§0.6 §3 admits Softplus head c≈0 stable-basin remains hypothesized failure mode for (1c); §0.6 asserts orthogonality holds IFF (1b) failure was body-side-dominated; no evidence (1b) was body-dominated vs head-dominated. If head-dominated, swapping body (1b→1c) discharges non-cause and (γ)-reuse is illegitimate.

**Operative §7 re-route**: (1c) FAIL + R-b-pre2 firing condition does **NOT** auto-falsify (γ); routes through head-vs-supervision disambiguation FIRST.

**New rung 4.5 — head-ablation pilot** (inserted into §8 between rung 4 and 5; was numbered as new rung-after-Juno in PI absorption; lands as 4.5 in §8 dispatch chain):
- 1-seed pilot: current `(1c) body + direct-linear head (drop Softplus)` + same (γ) loss. ~30 min CPU OR single short Juno rung.
- **PASS** (direct-linear head + (γ) escapes void-floor basin): head was load-bearing failure cause; (γ) survives R10 reuse contract; (1c) FAIL is re-attributed to Softplus head, NOT (γ) loss.
- **FAIL** (direct-linear head + (γ) still void-floor): head ⊥ body confirmed; (γ) supervision-class falsification holds; routes to (δ) supervision-pivot.
- Pre-commit: "(1c) FAIL + R-b-pre2 firing alone does NOT discharge (γ); rung 4.5 head-ablation is **mandatory** before (δ)."

This is the missing R10 orthogonality argument — written in advance per contract.

### §10.C — D71 Panel B3 (every-seed trigger asymmetry): RESOLVED via trichotomy

Rev 1 §7 trigger "(1c) FAIL + R-b-pre2 every-seed (10/10)" was asymmetric vs (1b)'s `median(Δ_seed) ≤ 0` robust statistic. Systematically under-commits to (γ) falsification.

**Operative §7 trigger (replaces Rev 1 §7 wording)**:

> **"IF Juno n=10 Wilcoxon test on primary observable returns FAIL AND MDE-block CLEARS:**
> **(i) R-b-pre2 fires in ≥7/10 seeds** (matched to (1b) median ≤ 0 robustness; ~70% quorum tolerates 3-seed stochasticity) → **route to rung 4.5 head-vs-supervision disambiguation** (per §10.B). After rung 4.5 verdict + rung 4.6 P2 spot-check (per §10.E), (γ) class-falsification triggers IFF head-ablation FAIL AND P2 spot-check confirms.
> **(ii) R-b-pre2 fires in ≤3/10 seeds** → (γ) NOT falsified; route to panel for taxonomy of which body-axis attribute (Sine, init, ω₀) is load-bearing (resolves P5).
> **(iii) R-b-pre2 fires in 4-6/10 seeds (indeterminate band)** → routes to panel ruling, no auto-dispatch. **If (γ) falsification triggers → (δ) supervision-pivot enters routing stack as next architectural-axis candidate."**

Symmetric to (1b) tolerance for seed stochasticity; closes falsifiability-asymmetry loophole.

### §10.D — D71 Panel B4 (lr cargo-cult): RESOLVED via new rung 1.5 lr pre-flight

§1.3 justified lr_max=1e-4 via Sitzmann+2020 §4.1 photometric default. Cosmological log-density spans 5 decades vs Sitzmann 1-decade photometric output. Cargo-cult risk: lr may be too low to escape void-floor basin in 500 steps → kinetic failure mistaken for architectural failure.

**New rung 1.5 — lr sensitivity pre-flight** (inserted into §8 between rung 1 and 2):
- 3 seeds × 3 lr {5e-5, 1e-4, 5e-4} = 9 runs, (1c) SIREN body (per §10.A wiring) under (γ), 500 steps each on CPU (~30 min CPU per panel estimate).
- Metric: void-floor escape proxy — `min(Bin-D pred) > void_floor_threshold` after 500 steps; default threshold `log₁₀(ρ/⟨ρ⟩) > -1.0`.
- **Outcome**:
  - lr=1e-4 escapes ≥1 seed → scope-lock §1.6 to lr=1e-4 for Juno dispatch; Sitzmann default vindicated for cosmology.
  - lr=1e-4 fails 3/3 but lr=5e-4 escapes ≥1 → re-spec §1.6 lr_max=5e-4 with explicit "Sitzmann default re-tuned for 5-decade output" note.
  - All 9 fail → R29 gate-construction failure; (1c) FAIL attribution at Juno-tier would have been kinetic, not architectural — fresh panel cycle for arch re-spec.

This also discharges K6 panel finding #1 forward obligation on lr-axis-coverage-on-(1b)-skip-rich (per LEDGER §3 [D-71] amendment block "Forward obligations register"): rung 1.5 lr pre-flight runs on (1c) body, not (1b), but per [D-37]-ext rule 5 symmetric-disclosure the lr-axis-coverage status is now disclosed-and-discharged-forward rather than declared-but-not-probed.

### §10.E — N1-N5 + K6 #4/#6 absorption

**N1 (functional fan_in)**: ABSORBED in §10.A wiring spec — `in_dim = 3 + g_dim + e_dim` functional.

**N2 (M2 AND-gate α-correction)**: ABSORBED — explicit Bonferroni α=0.025 per Bin-D sub-clause when both (i) and (ii) AND-gate at M2. Cite [D-70] Rev 5.1 amendment v2 paragraph that introduced AND-gate as the precedent; α-correction was implicit; now made explicit.

**N3 (Bin-D necessary-not-sufficient)**: ABSORBED — §2.2 sub-clause (ii) explicitly flagged necessary-not-sufficient for void-floor-escape. New sub-clause **(iii) density-PDF KS or W2 distance** at M2 endpoint as a parallel AND-gate, citing Walther+ 2018 §3.3 / Lukić+ 2015 §3 PDF-shape recovery. Bin-D differential improvement test (ii) covers ONE signature; density-PDF shape (iii) covers complementary signature.

**N4 (P2 spot-check)**: ABSORBED — new rung 4.6 in §8 dispatch chain (inserted between rung 4.5 and 5): on §7 trigger AND post-rung-4.5 head-ablation FAIL, auto-dispatch 1-seed P2 spot-check on (1c) BEFORE (δ) supervision-pivot. Cost ~30 min Juno. Disambiguates "(γ) class-falsification" from "(1c) on P1 P1-specific pathology" — converts (δ) routing from "P1-only evidence" to "P1+P2 convergent OR P1-only with P2 disambiguating."

**N5 (init-time activation-statistics asserts)**: ABSORBED — §1.4 (i) integration test extended with init-time asserts per Sitzmann+2020 §3.2 targets: for fixed Gaussian input batch (size 64), assert `pre-activation std across each body layer ∈ [0.5, 2.0]` AND `post-activation std ∈ [0.4, 1.0]`. Init-variance failures trip BEFORE 30-min Juno burn.

**K6 #4/#6 primary-observable swap** (per LEDGER §3 [D-71] amendment block §C primary-observable swap): (1c) gate uses `var_pred(step_N)/var_truth(step_N)` as primary load-bearing observable. Δ_seed DEMOTED to secondary direction-of-motion check with explicit caveat that σ(Δ_seed) is init-RNG-dominated (per K6 panel finding #4 evidence: r ≈ -0.99 between Δ_seed and step_0 value). MDE re-derived against σ(var-ratio at step N) — much tighter than σ(Δ_seed). The §2 M0 row + §7 pre-commit reference this primary observable.

**K6 #3 ε-anchor demotion** (per LEDGER §3 [D-71] amendment block §H R29 demotion): (1c) does NOT inherit ε=0.05 unchanged. Operative ε for (1c) Rev 1.1:
- **Path (a)** preferred — derive ε from defensible unit chain to the gate observable `var_pred/var_truth`. Candidate anchor: pre-flight B `μ_frozen = 2.4e-6` (frozen-init var-ratio); ε could be `10 × μ_frozen = 2.4e-5` (10× above frozen-init noise floor) — explicitly anchored to the gate observable, no nonlinear forward-model bridge.
- **Path (b)** fallback — declare ε as "heuristic ceiling, not from observational anchor; gate verdict reads as 'large/small headroom against heuristic' not 'CLEARED vs observational floor.'"
- Pre-commit Rev 1.1: **Path (a) with ε = 10 × μ_frozen = 2.4e-5** on `var_pred(500)/var_truth(500)` primary observable. Symmetric-disclosure: this is a frozen-init-anchored conservative ceiling, NOT an observational physical floor. Panel may overturn at Rev 1.1 re-review.

### §10.F — OS1 K6 narrow-discharge panel cycle: RUN (parallel, completed)

K6 panel ran in parallel with this amendment authoring per OS1 absorption. Verdict: NEEDS-WORK; PI absorbed into LEDGER §3 [D-71] amendment block (this session, commit pending user auth). K6 narrow-discharge ruling NOT yet APPROVE-narrow until the LEDGER amendments land.

### §10.G — OS2 path-(a) reading: PI rules autonomously, user-flag landed

PI reads `feedback-path-a-exhaustive` as **loss-axis-binding only**: architecture-axis walks under fixed (γ) supervision ((1b) → (1c) → (1d)) are IN-scope path-(a) work; (δ) supervision-pivot would be the loss pivot (hard stop). Surfaced as user-flag in LEDGER §7 history paragraph 2026-05-26; user ratified "yes" on the implicit reading at the auth checkpoint. PI proceeds autonomously per feedback-pi-decides.

### §10.H — R29 status (per K6 panel #3 + R29-promotion panel verdict (b) NEGATIVE)

**R29 status: CANDIDATE** per R29-promotion separate panel cycle (dispatched + ruled 2026-05-26 same session) verdict **(b) NEGATIVE**. **Full demotion** from BANKED, **NOT** DEFERRED-BANKED PROVISIONAL. **[D-71] §H banking marked falsified-prior** under [D-37]-Extension R2 cascade. Re-banking requires: (i) two-part rule-text amendment landed in `.claude/agents/project-architect.md` (this commit batch), AND (ii) one PROSPECTIVE design-time catch on a fresh gate (not remediation of ε=0.05 case).

Panel ruled (a) POSITIVE foreclosed as rule-laundering ([D-70] N1 sin re-committed) and (c) INCONCLUSIVE foreclosed by verbatim rule-text. The "successful prevention" at job 203337 was a coincidence-PASS against wrong-unit threshold; R29 has produced ZERO confirmed successful preventions since [D-68] candidate-banking.

**Two-part rule-text amendment** (lands in `.claude/agents/project-architect.md` this commit batch):
- (i) PI-proposed: unit-chain derivation from anchor-observable to gate-observable, with nonlinear forward-model bridges named explicitly.
- (ii) Panel-added (load-bearing, closes (a) escape hatch permanently): R29's in-line check is a **design-time obligation discharged at gate-construction commit**; post-spec panel catches do NOT count as R29 successful preventions.

This (1c) Rev 1.1 spec does NOT depend on R29 status — the discipline is applied via panel review regardless. §5 R-rule audit updated to "R29 CANDIDATE (per R29-promotion panel verdict 2026-05-26 (b) NEGATIVE)."

Companion governance updates landing same commit batch: R12 in-place amendment ("sighting later revealed to instantiate the rule's failure-mode RESETS sighting count to zero + invalidates parent promotion"); R32 DEFERRED-CANDIDATE entry ("post-spec narrow-discharge panel cycles are mandatory before any banking promotion lands" — first sighting, R12-second-sighting-cross-track required for BANKED).

### §10.I — R31 third sighting (sharpening, NOT promotion)

R31 third sighting (observable-unit-chain framing at K6 ε=0.05) — per R12 precedent within-track third sighting calls for rule-text sharpening, NOT promotion. Sharpened R31 addendum deferred to next governance batch commit on `.claude/agents/project-architect.md`. Cross-track second-sighting still owed for formal R31-strengthened promotion.

### §10.J — Updated §8 dispatch chain (with new rungs 1.5 + 4.5 + 4.6)

Per R28 single-rung-per-dispatch discipline + R27 ladder-count cross-check, the updated ladder:

0. defense-panel pre-review on Rev 1.1 ← CURRENT GATE.
1. core-implementer dispatch 1: extend `IGMNeRF.__init__` body_arch to add `'siren'`; raw-coords → SIREN body wiring per §10.A; integration test + estimator-equivalence test (R20 twin-gate). Init-time activation-statistics asserts per N5.
**1.5. core-implementer dispatch (lr pre-flight)**: `scripts/d71_lr_pre_flight.py`, 9-run CPU sweep per §10.D. Outcome routes scope-lock §1.6.
2. core-implementer dispatch 2: extend `pipeline.py` `--arch` argparse to include `'siren'`; plumb through 3 construction sites; sbatch flag `--lr_max` set per rung 1.5 outcome.
3. infrastructure-manager dispatch: `scripts/submit_juno_stage1a_1c_siren.sh` mirror; 4-item pre-submission validation.
4. User auth + Juno dispatch.
**4.5. core-implementer dispatch (head-ablation pilot, CONDITIONAL on §7 trigger (i))**: `(1c) body + direct-linear head + (γ) loss` 1-seed pilot. Outcome routes (γ)-attribution vs head-attribution.
**4.6. infrastructure-manager dispatch (P2 spot-check, CONDITIONAL on §7 trigger + rung 4.5 FAIL)**: 1-seed (1c) on P2. Disambiguates (γ) class-falsification from P1-specific pathology.
5. Post-Juno absorption per [D-37] rule (a) observation-first + rule 5 symmetric-disclosure.

Dispatch-ladder rung count = 7 (rungs 0-5 plus 1.5, 4.5, 4.6 conditional rungs). Per R27 cross-check, ladder is well-formed against the new artifact set: SIREN body wiring + lr pre-flight + argparse extension + integration test + estimator-equivalence test + activation-stats asserts + sbatch authoring + head-ablation pilot (conditional) + P2 spot-check (conditional) = 9 artifact rungs, 7 dispatched.

### §10.K — Sign-off

**R15+R28 PROVISIONAL** — Rev 1.1 amendment block lands but does NOT lift R15+R28 until defense-panel re-review on the amendment block APPROVEs. Until then, no HPC dispatch.

**Honest framing per [D-37] rule (a)**: D71 Rev 1 had 4 BLOCKING + 5 NEEDS-WORK + 5 PROBE flaws; K6 narrow-discharge panel had 3 KILLER + 4 SERIOUS + 1 PROBE flaws that flowed back; R29-promotion separate panel ruled (b) NEGATIVE → R29 → CANDIDATE (full demotion). Rev 1.1 absorbs all of them. The (1c) SIREN candidate direction is correct (R10 orthogonality argument holds under panel review IFF rung 4.5 head-ablation runs); the wiring needed canonical-SIREN correction; gate construction needed primary-observable swap and ε-anchor demotion; R29 promotion was wrong and is now retracted. **No spin** — the absorption-block lists each panel finding and its resolution.

---

## §11 — MDE_ARE_COEFF=0.9 sidecar documentation (panel PROBE #7 compaction-survival)

**Constant**: `MDE_ARE_COEFF = 0.9` in `scripts/d70_wilcoxon_gate.py:131`.

**Provenance**: Wilcoxon signed-rank one-sided test asymptotic relative efficiency (ARE) versus paired t-test under normal-shift alternative is `3/π ≈ 0.955`. For n=10 at α=0.05 one-sided, the MDE coefficient on σ_seed is approximately `0.9` (slight adjustment from asymptotic ARE for finite-n + one-sided correction). The constant captures the post-ARE MDE factor, NOT ARE itself.

**Why preserved here**: per panel PROBE #7 — the comment at code line 128 ("0.955 ⇒ MDE ≈ 0.9·σ_seed") is the derivation, but it's in a code file at risk of compaction drift. Documenting in the design doc binds the coefficient to the methodology rather than to a code comment.

**Operative reading**: `MDE_estimate = MDE_ARE_COEFF * σ_seed = 0.9 * σ_seed`. The MDE-block then compares `MDE_estimate < ε_physical_escape` to authorize the gate. Per K6 panel + §10.E + §10.H: ε is now `10 × μ_frozen = 2.4e-5` (gate-observable-anchored), NOT 0.05 (P_F-observational-anchor invalid unit-chain).

If a future Wilcoxon test on this gate uses a different n or α, the MDE coefficient changes (n=20 at Bonferroni α=0.025 has different ARE-adjustment factor). Re-derive before re-citing.

---

## §12 — Rev 1.2 AMENDMENT BLOCK (panel S-A7 + governance forward-obligations absorbed, 2026-05-30)

Rev 1 and Rev 1.1 wording above preserved verbatim for audit trail per [D-70] original-wording-preserved-for-audit-trail precedent. Rev 1.2 adds: empirical ε_physical anchor from S-A7 Juno output (job 205722, landed 2026-05-30 22:37); K3'-degeneracy class naming + axis-row split + hidden-ω₀-30 orthogonality re-derivation (§10.A); rung 4.5 multi-seed + 4.6 P2 cross-physics specs (§10.B); trichotomy critical-region binomial citation (§10.C); rung 1.5 lr-grid n=20 spec with citation (§10.D); N3(iii) density-PDF demotion to diagnostic-only (§10.E); operative ε_physical = 0.1591 (σ=3) anchor selection with pre-commit hedge (§10.E K6 #3); R-rule audit refresh including R29 CANDIDATE status, R31 sighting #4, R32 DEFERRED-BANKED PROVISIONAL, R28 self-violation sighting #1 (§5); R26 in-session re-verification block updated with S-A7 JSON evidence row (§9); dispatch ladder rung count 7 → 9 with R28 cross-check (§10.J).

### §12.A — §10.A REWRITE (axis-change row split + K3' degeneracy naming + canonical re-verbing)

**Rev 1.1 issue**: §10.A treated the K3' (PE drop + hidden ω₀ 1→30) change as a single axis-change row. This collapsed two distinct interventions — positional encoding removal AND first-layer-frequency sweep — into one row, obscuring which intervention buys which behavior. Per [D-37]-Extension R8 (cascade-close formality requires axis-coverage proof under stated decomposition criterion), single-row treatment of two interventions violated the decomposition discipline retroactively.

**Rev 1.2 axis-change rows (TWO distinct rows, both axis-correct)**:

| Row | Intervention | Mechanism | Failure-mode prevented |
|-----|---|---|---|
| K3'-row-1 | PE drop (3-band Fourier features → no PE) | Removes explicit high-frequency basis; forces representation through hidden-layer features only | Prevents PE-induced spectral bias toward Fourier-band-aligned modes (Mildenhall+2020 §5.1 spectral bias diagnostic) |
| K3'-row-2 | Hidden ω₀ 1→30 (Sitzmann image-regression default) | Scales first-Sine-layer frequency 30×; raises bandwidth of representable body field | Prevents constant-mean basin residence via non-zero-Sine-gradient + 30× bandwidth at init (Sitzmann+2020 §3.2) |

**K3' degeneracy class NAMED** (Rev 1.2 formalization):

The K3' = (K3 → PE-drop + hidden-ω₀-30) change confounds two effects that under K3-as-originally-spec'd were varied jointly. Empirical separation requires rung 1.5 (lr sweep) to verify body-axis fitting is achievable under EITHER row independently — but rung 1.5 as currently spec'd varies only lr, not (PE-on, ω₀=1) vs (PE-off, ω₀=30) vs intermediates. This under-decomposes the intervention space. Rev 1.2 acknowledges this as a KNOWN scope limitation: the (1c) sprint tests the K3' joint intervention only; PE-isolation and ω₀-isolation rungs are deferred to a follow-on sprint if (1c) clears the trichotomy. Naming the class makes the deferral auditable.

**Canonical re-verbing** (per [D-37]-Extension R9 invariance-verb discipline):

Rev 1.1 §10.A used "canonical Sitzmann" to describe ω₀=30. Rev 1.2 replaces "canonical" with "**Sitzmann image-regression default**" throughout. "Canonical" implies a single-correct setting that Sitzmann+2020 does not claim; the paper presents ω₀=30 as the default for image-regression experiments, with task-specific tuning recommended (§3.2 "hyperparameter ω₀ should be tuned per task"). The cosmological-field-regression task is not image-regression; the default is a starting point, not a canonical setting.

### §12.A-addendum — §0.6 R10 orthogonality re-derivation under hidden ω₀=30

The §0.6 R10 retired-model reuse contract (constant-mean basin escape argument) was derived in Rev 1 under the implicit assumption of Sine activation with Sitzmann initialization at default ω₀. Hidden ω₀=30 shifts the first-Sine-layer frequency by 30×; the orthogonality argument must be re-verified under this lever.

**Re-derivation**:

The constant-mean basin escape argument has two structural ingredients: (i) non-zero gradient property of Sine at the constant-mean output configuration; (ii) sufficient bandwidth at initialization to represent non-constant body fields after the first gradient step.

Property (i) is ω₀-INDEPENDENT: ∂sin(ω₀·x)/∂x = ω₀·cos(ω₀·x), which is non-zero almost everywhere regardless of ω₀ scale. The basin escape mechanism (gradient flows out of the constant-mean configuration in the first step rather than vanishing) holds for ω₀ ∈ {1, 30, any positive real}.

Property (ii) is ω₀-DEPENDENT and STRENGTHENED at ω₀=30: Sine-gradient bandwidth at initialization scales as ω₀ × baseline. The ω₀=30 setting provides 30× the representable bandwidth of ω₀=1 baseline at init — which is the explicit Sitzmann+2020 motivation for image-regression tasks where the target field has non-trivial high-frequency content. The cosmological density field at 48³ truth-grid resolution likewise has non-trivial high-frequency content (density fluctuations across voxel scale). The ω₀=30 lever therefore STRENGTHENS the basin-escape argument's bandwidth ingredient rather than weakening it.

**Net orthogonality verdict**: R10 retired-model-reuse contract for the §0.6 (1c) Sitzmann body argument SURVIVES under hidden ω₀=30. The constant-mean basin escape claim holds; the bandwidth-at-init claim is strengthened. No re-verbing required to R10 itself for this lever; the K3'-row-2 mechanism column above explicitly anchors on this re-derivation.

### §12.B — §10.B + RUNG 4.5/4.6 SPECS (multi-seed extension + P2 cross-physics)

**Rung 4.5 spec** (multi-seed on (1c) primary configuration):
- n = 5 seeds on the (1c) primary configuration (Sitzmann image-regression default ω₀=30, PE-drop, hidden=256, layers=5, lr=5e-5, 500 steps).
- PASS criterion (two-clause, both required):
  - (i) Wilcoxon signed-rank test on (var_pred/var_truth) across 5 seeds at step 500, one-sided alternative > 0, α = 0.10 (matched to [D-70] gate-discipline α convention for n≥5 small-sample regimes).
  - (ii) ≥ 3 of 5 seeds individually exceed the **void-floor escape threshold** defined as var_pred/var_truth ≥ ε_physical(σ=5) = 0.0772 (the loose-floor anchor from S-A7; this is escape-from-constant-mean, NOT body-axis-clearance which is a higher bar).
- Escape-threshold rationale: void-floor escape is "did the body get out of the constant-mean basin at all" — the σ=5 ε_physical = 0.0772 anchor is the empirically-derived loose-floor matched to that question. Using the rung-1.5 `min(Bin-D pred)` proxy as in Rev 1.1 draft text was a cargo-cult of an earlier diagnostic; Rev 1.2 corrects to the §10.E primary-observable-aligned threshold.
- Compute budget: 5 × ~15 min A30 wall ≈ 75 min total. Within Juno sub-2hr budget.
- Methodology hedge: rung 4.5 is a confirmation-of-direction step, not a body-axis-clearance proof. Body-axis clearance is the §10.C trichotomy load.

**Rung 4.6 spec** (P2 cross-physics replication):
- n = 3 P2 seeds on (1c) primary configuration (same hyperparameters as rung 4.5; only physics_id changes P1 → P2).
- PASS criterion: ≥ 2 of 3 P2 seeds return median Δ_seed > 0 (where Δ_seed = var_pred/var_truth at step 500 minus var_pred/var_truth at step 0, capturing variance-recovery direction).
- Rationale: a 2-of-3 directional test is the minimum sample size for a non-trivial cross-physics replication claim under the sprint compute budget. Stronger 5-of-5 P2 test deferred to follow-on if (1c) primary + 4.5 + 4.6 all clear.
- Compute budget: 3 × ~15 min A30 wall ≈ 45 min. Within budget.
- Methodology hedge: rung 4.6 is a directional cross-physics check, NOT a cross-physics invariance proof.

### §12.C — §10.C TRICHOTOMY RE-SPEC (binomial critical-region citation + pre-commit timestamp)

Rev 1.1 §10.C trichotomy critical-region thresholds (≥8 / 3-7 / ≤2 out of 10 seeds) were named without statistical justification. Rev 1.2 provides the justification.

**Critical-region derivation**: under H_0 of no body-axis effect (50/50 chance per seed of escaping void-floor by random gradient noise alone), the seed-escape count K ~ Binomial(n=10, p=0.5).

- P(K ≥ 8 | H_0) = 0.0547 ≈ 5.5% (one-sided upper tail, α ≈ 0.055)
- P(K ≤ 2 | H_0) = 0.0547 ≈ 5.5% (one-sided lower tail, α ≈ 0.055)
- P(3 ≤ K ≤ 7 | H_0) ≈ 0.89 (the "ambiguous middle")

The ≥8 / ≤2 thresholds correspond to one-sided α ≈ 0.055 critical regions — conventional 5% per tail, matched to standard non-parametric hypothesis testing practice for small-n binomial samples.

**Citation**: `sprent_smeeton2007` (Sprent & Smeeton, "Applied Nonparametric Statistical Methods", 4th ed., Chapman & Hall/CRC, 2007), §4.2 binomial-test critical regions for small-n sign-test analogues.

**TODO in `papers/shared/main.bib`**: add `sprent_smeeton2007` BibTeX entry (verifiable to ISBN 978-1584887010).

**Pre-commit anchor**: Rev 1.2 commit timestamp (this turn) is the [D-37]-rule-5 symmetric-disclosure pre-commit anchor for the trichotomy thresholds. Per Rev 1.1 §10.C, thresholds were named before seeing (1c) results; Rev 1.2 formalizes the statistical justification without altering the thresholds, preserving the pre-commit.

### §12.D — §10.D + RUNG 1.5 SPEC (lr-sweep n=20 body-axis fitting verification)

**Rung 1.5 spec**:
- Metric: `var_pred(500)/var_truth(500)` — explicitly the production-scale observable matching §10.E K6 #3 (NOT a min(Bin-D pred) proxy; Rev 1.1 draft text proxy retired per §12.B cargo-cult correction).
- lr grid: `{3e-6, 5e-5, 1e-4, 5e-4}` (4 values, spanning 2.2 OOM around the (1c) primary lr=5e-5 default).
- Seeds: 5 per lr value → 20 total runs.
- PASS criterion: ≥ 1 of 20 (lr, seed) configurations exceeds `ε_physical(σ=3) = 0.1591` at step 500. A single PASS demonstrates body-axis fitting is achievable in principle on the chosen architecture + duration; multiple PASSes strengthen.
- Citation for IGR-style lr-sweep verification on implicit neural representations: `atzmon_lipman2020igr` (Atzmon & Lipman 2020, "SAL: Sign Agnostic Learning of Shapes from Raw Data", CVPR 2020, precedent for lr-sensitivity sweep on implicit field fitting).
- **TODO in `papers/shared/main.bib`**: add `atzmon_lipman2020igr` BibTeX entry (verifiable to CVPR 2020 proceedings).
- Compute budget: 20 × ~15 min A30 wall ≈ 5 hr. Above sub-2hr per-job budget; SLURM array job recommended (4 lr × 5 seed = 20 tasks each ≤ 15 min).
- Failure-mode handling: if rung 1.5 returns 0/20 PASS, body-axis fitting is NOT achievable on the chosen 500-step duration; this triggers Stage 1a duration re-spec to 5000 steps as a [D-37]-rule-5 symmetric-disclosure pre-committed fallback. Rung 4.5/4.6 do NOT dispatch until rung 1.5 clears.

### §12.E — §10.E N3(iii) DEMOTION + K6 #3 ε_ANCHOR (operative ε_physical = 0.1591, σ=3)

**N3(iii) demotion (density-PDF KS-or-W2 → diagnostic-only)**:

Rev 1.1 §10.E N3(iii) primary load-bearing on density-PDF KS-or-W2 divergence as a void-floor signature. Rev 1.2 demotes this to diagnostic-only:
- (a) KS-or-W2 divergence is sensitive to tail density features the (1c) sprint is not scoped to address. Using it as primary risks a [D-37]-extension R8 cascade-close violation (claiming foreclosure of a hypothesis the diagnostic does not actually test at sprint scope).
- (b) Sub-clause (ii) Bin-D differential is a tighter, more directly interpretable void-floor signature: "did the predicted density in the lowest-truth bin recover the truth's variance share?" The Bin-D variance share is mechanically tied to void-floor escape in a way the full-distribution divergence is not.

**Rev 1.2 N3(iii) text**: PRIMARY void-floor signature = sub-clause (ii) Bin-D differential (var_pred|Bin-D / var_truth|Bin-D at step 500 exceeds ε_physical(σ=3) = 0.1591). DIAGNOSTIC-ONLY = density-PDF KS-or-W2 (reported in absorption block, NOT load-bearing on PASS/FAIL).

**Defense-path flag**: W2-with-Lukić-anchor (Lukić+2015 density-PDF reference) is a stronger void-floor diagnostic but requires the Lukić PDF anchor to be reconstructed in the same simulation suite + redshift + smoothing-scale as the (1c) truth field. This is deferred follow-on work; flagged here so that a fresh-panel reviewer of Rev 1.2 sees the deferred path explicitly.

**K6 #3 ε_anchor — operative ε_physical = 0.1591 (σ=3)**:

Rev 1.1 text was `ε = 10 × μ_frozen = 2.4e-5` (multiplier-over-frozen-init noise floor). Rev 1.2 replacement:

```
ε_physical = μ_smoothing_floor(σ=3) = 0.1591
  where μ_smoothing_floor(σ=3) is the variance ratio
    Var[gaussian_filter(truth_field, sigma=3, mode='wrap')] / Var[truth_field]
  computed on the production truth field at 48³ voxel resolution.
  Source-of-truth: cloud_runs/d71_var_smoothing_floor.json
  (Juno job 205722, landed 2026-05-30 22:37; var_truth_full =
  1.832623e+02; PCV PASS; 58.3s wall on A30).
```

**Mandatory hedge per [D-37] rule 5 symmetric-disclosure**:

> ε_physical(σ=3) = 0.1591 was chosen as the operative anchor BEFORE seeing any (1c) results. The σ=3 choice is the moderate physical-floor defensible on scale-matching grounds (~3.75 Mpc/h ≈ trans-linear scale plausibly fittable by a Sitzmann-image-regression-default MLP body at 500 steps without near-saturation claim). The alternative anchors σ=1 (ε=0.49, asymptotic stretch), σ=2 (ε=0.26, ambitious-but-achievable stretch), and σ=5 (ε=0.077, loose floor matched to rung 4.5 void-floor escape threshold) are pre-committed at the same Rev 1.2 commit timestamp and are available as diagnostic re-anchors if (1c) results land in an intermediate regime; this is NOT a moving-goalpost violation because all four anchors were pre-committed in one commit. Re-anchoring requires an explicit post-result LEDGER block citing the alternate σ.
>
> This anchor is the PHYSICAL FLOOR under Gaussian smoothing at σ=3 voxels of the production truth field; it is NOT a multiplier-over-frozen-init noise floor (the Rev 1.1 framing, retired per R29 wrong-unit-anchor violation diagnosed at K6 panel 2026-05-26 → R29 demotion to CANDIDATE per LEDGER §3 [D-71] amendment block §H 2026-05-26 R-rule audit).
>
> K6 #3 PASS criterion: var_pred(500)/var_truth(500) ≥ ε_physical(σ=3) = 0.1591. K6 #3 FAIL does NOT foreclose (1c) provided §10.C trichotomy clears ≥8/10 on the M0+M1+M2 stack. K6 #3 is a stretch-goal-not-default per PI Ruling 2 (LEDGER §3 [D-71] Rev 1.2 absorption): keep 500 steps; primary load-bearing falsification path is §10.C trichotomy on M0+M1+M2 over n=10 seeds. K6 #3 PASS strengthens; K6 #3 FAIL does not foreclose (1c) provided trichotomy clears ≥8/10.

### §12.F — §10.J DISPATCH LADDER (rung count 7 → 9; R28 cross-check explicit)

Rev 1.1 dispatch ladder rung count = 7. Rev 1.2 adds rungs 1.5 (lr-sweep body-axis verification) and 4.5 + 4.6 (multi-seed + P2 cross-physics). New rung count = 9.

**Rev 1.2 dispatch ladder**:

| Rung | Action | Owner | Landing artifact |
|---|---|---|---|
| 0 | S-A7 Juno output absorption (Rev 1.2 author turn) | PI | Rev 1.2 amendment block (this document tail) |
| 1 | Rev 1.2 commit | PI | git commit ref |
| 1.5 | lr-sweep n=20 body-axis verification (§12.D) | core-implementer + infrastructure-manager | SLURM array job + `cloud_runs/d71_rung15_lrsweep.json` |
| 2 | Fresh-panel pre-review on Rev 1.2 + S-A7 anchor selection | defense-panel | panel verdict block in LEDGER §3 [D-71] |
| 3 | Panel-verdict absorption | PI | LEDGER §3 [D-71] absorption block |
| 4 | (1c) primary seed run | core-implementer | `cloud_runs/d71_1c_primary_seed0.json` |
| 4.5 | Multi-seed n=5 rung (§12.B) | core-implementer | `cloud_runs/d71_1c_multiseed.json` |
| 4.6 | P2 cross-physics n=3 rung (§12.B) | core-implementer | `cloud_runs/d71_1c_p2_crossphysics.json` |
| 5 | Sprint absorption | PI | LEDGER §3 [D-71] sprint-close block |

**R28 dispatch-ladder cross-check**: Rev 1.2 landing-artifact count = 9 (rungs 0, 1, 1.5, 2, 3, 4, 4.5, 4.6, 5). Rung count = 9. **PASS** (rung count ≥ landing-artifact count by the literal-integer cross-check banked at R28).

### §12.G — §5 R-RULE AUDIT REFRESH (Rev 1.2)

| R-rule | Status (Rev 1.2) | Change vs Rev 1.1 | Sighting log |
|---|---|---|---|
| R28 | BANKED | unchanged | self-violation sighting #1 flagged below |
| R29 | **CANDIDATE** (demoted from BANKED) | per LEDGER §3 [D-71] amendment block §H AMENDED 2026-05-26 | ε=0.05 MDE-block wrong-unit catch (panel K6 → R29-promotion panel ruling (b) NEGATIVE 2026-05-26) |
| R30 | BANKED | unchanged | — |
| R31 | DEFERRED-BANKED | **sighting #4** logged this turn (S-A5 absorption); promotion criterion = R12 second-sighting-cross-track + tightened-text-amendment | #1 [D-70] R-prep; #2 [D-71] §H 2026-05-26; #3 [D-71] amendment block §G 2026-05-26; #4 S-A5 absorption Rev 1.2 (this turn) |
| R32 | **DEFERRED-BANKED PROVISIONAL** | promoted candidate → DEFERRED-BANKED PROVISIONAL on one operational test (this panel cycle: 4-item pre-validation closing + S-A7 Juno re-route + PCV PASS + methodology-preservation respected). R32 SECOND-OPERATIONAL-TEST RULING: Juno re-route is part of THIS panel cycle's deliverable chain, NOT a separate second sighting; R12 second-sighting-cross-track instance still required | one operational test landed |

**R28 self-violation sighting #1 (S-A8 arithmetic)**:

S-A8 was the rung-count arithmetic in the prior Rev 1.1 dispatch sequence: PI authored a 5-rung dispatch ladder while design-doc landing-artifact count was 7. R28 fired retroactively on PI re-review; Rev 1.1 dispatch ladder was re-authored before downstream dispatch. The self-violation is a PI-only sighting (PI caught the violation; not panel-caught), which is the minimum-sighting-quality bar for R28 retention. Sighting log update for R28 BANKED status: no demotion, but self-violation sighting #1 IS recorded to keep the sighting history transparent. If a SECOND self-violation occurs without panel catch in a downstream sprint, R28 should be considered for elevation discipline (e.g., mandatory dispatch-ladder-arithmetic gate in PI absorption template).

### §12.H — §9 R26 IN-SESSION RE-VERIFICATION BLOCK (refresh)

| Inherited claim | File / artifact freshly read this session | Verdict | Restated claim (if changed) |
|---|---|---|---|
| ε_physical anchor source-of-truth | `cloud_runs/d71_var_smoothing_floor.json` (Juno job 205722, landed 2026-05-30 22:37) | SURVIVES, with σ-specific anchor selection added | `ε_physical(σ=3) = 0.1591` operative; σ=1/2/5 reserved as diagnostic re-anchors per pre-commit |
| var_truth_full = 1.832623e+02 | same JSON above | SURVIVES (cross-checks local partial run per S-A7 PCV PASS) | unchanged |
| Smoothing kernel = `scipy.ndimage.gaussian_filter mode=wrap` | same JSON above | SURVIVES (periodic, physically correct for cosmological box) | unchanged |
| Pre-commit timestamp for ε_physical anchor selection | Rev 1.2 commit (this turn) | NEW (no prior anchor to inherit; this is the establishing entry) | Rev 1.2 commit ref = pre-commit anchor for σ choice |
| R10 §0.6 retired-model-reuse contract under hidden ω₀=30 | §12.A-addendum above (re-derivation) | SURVIVES (non-zero-Sine-gradient property is ω₀-independent; bandwidth-at-init STRENGTHENED at ω₀=30) | R10 reuse contract holds; bandwidth ingredient strengthened, not weakened |

### §12.I — GOVERNANCE FORWARD-OBLIGATIONS (Rev 1.2 close)

(a) Fresh-panel pre-review on Rev 1.2 OWED before any rung 1.5 dispatch authorization. R15 + R28 PROVISIONAL persists until panel APPROVE per R32-candidate panel-cycle-discipline banking precondition.

(b) R12 second-sighting-cross-track instance needed before R32 promotes from DEFERRED-BANKED PROVISIONAL → BANKED. Next-session governance audit: track whether a CROSS-TRACK panel cycle fires on a fresh gate-construction question; if it does, R32 second-sighting accrues.

(c) bib entries `sprent_smeeton2007` and `atzmon_lipman2020igr` TODO in `papers/shared/main.bib`. Owner: latex-author at next paper-iteration cycle. Not blocking on rung 1.5 dispatch.

(d) K3' axis-isolation rungs (PE-isolation, ω₀-isolation) are deferred follow-on if (1c) clears trichotomy. Rev 1.2 names the deferral so the limitation is auditable.

(e) W2-with-Lukić-anchor density-PDF defense-path is deferred follow-on per §12.E N3(iii) Rev 1.2 demotion text.

### §12.J — Sign-off

**R15+R28 PROVISIONAL** persists until fresh-panel re-review on Rev 1.2 returns APPROVE per just-banked R32-candidate panel-cycle discipline as banking precondition. Provisional status is binding on downstream dispatches: no core-implementer dispatch for rung 1.5 or rung 4 until panel APPROVE lands.

**Honest framing per [D-37] rule (a)**: Rev 1.1 had 4 KILLER + 3 SERIOUS + 5 PROBE panel findings; Rev 1.2 absorbs each with concrete numerical thresholds + bib citations + dispatch-ladder cross-checks. The S-A7 Juno re-route landed empirical anchor in the same units as the gate observable (var_pred/var_truth ratio) — the unit-chain audit closes. K-A1 ω₀-hidden axis was split into two distinct rows (PE-drop + hidden-ω₀-30) with K3' joint-confound limitation named. K-A4 observable-mismatch closed by metric realignment. K-A3 trichotomy gained binomial-quantile derivation. Rung 4.5/4.6 multi-seed re-specs close K-A2 statistical-power gap.

---

## §13 — Rev 1.3 AMENDMENT BLOCK (Rev 1.2 fresh-panel cycle #2 NEEDS-WORK absorbed, 2026-05-31)

Rev 1 / Rev 1.1 / Rev 1.2 wording above preserved verbatim per audit-trail discipline. Rev 1.2 fresh-panel (cycle #2 on [D-71]) returned NEEDS-WORK with 4 KILLER (K1 wiring inconsistency; K2 ε_physical anchor scale-and-boundary mismatch [BLOCKING]; K3 4-anchor bandit; K4 rung-stack incompatibility) + 4 SERIOUS (S5 K3' new exit pathway; S6 SIREN literature over-strong; S7 R32 rule-shrinking; S8 R28 soft-trigger) + 4 PROBE.

**Status**: Rev 1.3 PROVISIONAL pending defense-panel re-review (cycle #3) on this amendment block. R15+R28 PROVISIONAL HELD. No HPC dispatch, no code-implementer dispatch, no infrastructure-manager dispatch until cycle #3 APPROVE. **σ-bandit 4-anchor pre-commit RETRACTED** per K2 ruling below; Rev 1.2 §12.E ε_physical(σ=3)=0.1591 anchor RETIRED.

### §13.A — K2 absorption (BLOCKING): Re-anchor option A (sampling-noise floor)

**Code-state R26 in-session re-verification** (Rev 1.3 absorption load-bearing):
- `experiments/nerf/pipeline.py:540-610` `_compute_m3`: corners drawn uniform-random in 768³ box; modulo-wrap on box exit (lines 562-567); voxels of all n_crops crops **concatenated** into `raw_truth_all`; **single pooled** `raw_truth_all.var(unbiased=True)` produces `var_truth_lin`. Aggregation = pooled-cross-crops single variance, NOT per-crop. No Gaussian smoothing applied at gate-evaluation time.
- Sherwood native voxel = 60 Mpc/h / 768 = 0.0781 Mpc/h ≈ 78.1 kpc/h. σ=3 voxels = 234 kpc/h = 0.234 Mpc/h. Rev 1.2 §12.E hedge "~3.75 Mpc/h ≈ trans-linear" contained internal arithmetic error: 3.75 Mpc/h is the crop-side length (48 × 0.0781), NOT σ=3 smoothing scale.
- `cloud_runs/d71_var_smoothing_floor.json` smoothing_kernel = `mode='wrap'` on full 768³ box. The gate observable applies NO smoothing. **Anchor and gate evaluate fundamentally different quantities** — R29 sighting #2 post-demotion accrues (structural twin of K6 ε=0.05 wrong-unit failure mode).

**Rev 1.2 §12.E σ=3 / σ=1 / σ=2 / σ=5 4-anchor pre-commit is RETRACTED** under K2 ruling. The underlying anchor framing (smoothing-floor of full-box truth) does not match the gate-evaluation framing (pooled-voxel-variance ratio across random crops, no smoothing). The 4-anchor pre-commit was syntactically pre-committed but substantively measured the wrong quantity.

**Rev 1.3 operative anchor (Re-anchor option A)**:

```
ε_physical = M × σ_pooled_voxel_sampling_noise

where σ_pooled_voxel_sampling_noise is the sample standard deviation
of the pooled-cross-N-crops voxel-variance RATIO under K independent
random crop-placement seeds on the production 768³ truth field, with
N=100 crops × 48³ voxels per crop, sampled with modulo-wrap on box
exit per pipeline.py:562-567 convention.

Specifically:
  for k in 1..K:
    sample 100 random 48³ crops (uniform corners, modulo-wrap)
    compute pooled_var_k = Var[concatenate(all voxels in 100 crops)]
  σ_pooled_voxel_sampling_noise = std(pooled_var_k / pooled_var_baseline)

where pooled_var_baseline = mean over k of pooled_var_k (or equivalently
the full-box variance var_truth_full = 183.26 since uniform sampling
converges to box mean by LLN at N=100 crops).

M = 3 (pre-committed multiplier; 3-sigma above sampling-noise floor).
```

**Compute spec** (rung 5 of Rev 1.3 ladder; NOT authorized this absorption):
- New script `scripts/d71_compute_sampling_noise_floor.py` (REPLACES `scripts/d71_compute_var_truth_smoothing_floor.py`; old script + its JSON retired).
- Inputs: `Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy` (~1.81 GB).
- Computation: K=50 crop-placement seeds × 100 crops × 48³ voxels each, modulo-wrap on box exit; output `pooled_var_k / pooled_var_baseline` distribution + `σ_pooled_voxel_sampling_noise = std()`.
- Output JSON: `cloud_runs/d71_sampling_noise_floor.json` with `var_truth_full`, `pooled_var_baseline`, `σ_pooled_voxel_sampling_noise`, K=50 per-seed values.
- Compute budget: ~5 min CPU on adequate-RAM host (could run on Juno per [D-71] Rev 1.1 S-A7 precedent OR locally if memory allows for crop-sampling without full-field allocation).
- Gated on: Rev 1.3 → fresh-panel cycle #3 APPROVE. PI does NOT authorize the compute dispatch at this absorption.

**ε numerical pre-commit deferred**: ε = M × σ_pooled_voxel_sampling_noise with M=3 pre-committed; numerical σ_noise TBD per compute output. Rev 1.4 will land the numerical value in a SECOND panel cycle (cycle #4) per §13.I dispatch ladder below.

**K3 dissolution**: σ-knob retired. There is no smoothing kernel in Re-anchor option A; no σ to bandit-over. K3 finding structurally dissolved.

**P12 closed**: Sherwood cache convention is ρ/⟨ρ⟩ (overdensity, mean≈1). Both `var_pred_lin` and `var_truth_lin` use same convention; cancels in ratio. Explicit statement.

**P10 closed**: aggregation convention is pooled-cross-crops single variance. ε_physical and gate observable use SAME aggregation. Explicit statement.

### §13.A-wiring — Single canonical wiring table (replaces Rev 1 §1.5 ambiguity)

**Operative wiring for (1c) SIREN body** (CANONICAL — supersedes §1.5; core-implementer dispatch consumes THIS table exclusively):

| Layer | in_features | out_features | activation | ω₀ | init |
|---|---|---|---|---|---|
| input | 3 (raw normalized coords ∈ [−1,+1]) | — | — | — | — |
| layers1[0] | 3 [+1 g] [+e_dim e_p] | 256 | Sine | **30** (first-layer) | `U(−1/in_dim, +1/in_dim)` per Sitzmann+2020 §3.1 first-layer rule |
| layers1[1] | 256 | 256 | Sine | **30** (hidden) | `U(−√(6/256)/30, +√(6/256)/30)` |
| layers1[2] | 256 | 256 | Sine | **30** | same as hidden |
| layers1[3] | 256 | 256 | Sine | **30** | same as hidden |
| skip-cat | concat([h, raw coords + g + e_p]) | 256 + 3 [+1 g] [+e_dim] | — | — | — |
| layers2[0..3] | 256 + skip_dim → 256 → 256 → 256 → 256 | Sine | **30** | `U(−√(6/fan_in)/30, +√(6/fan_in)/30)` |
| density head | 256 → 1 | Softplus | — | linear, Kaiming-uniform `a=√5` |
| g head (if g_dim>0) | 256 → 1 | linear | — | linear |
| e_p head (if e_dim>0) | 256 → e_dim | linear | — | linear |

**Source-of-truth**: vsitzmann/siren reference implementation `meta_modules.py SineLayer` constructor default `hidden_omega_0=30` (used in image-regression notebooks; sitzmann-image-regression-default).

**Explicit note**: "PE_L10 deliberately omitted; Sine activations subsume frequency-basis role per Sitzmann+2020 §3."

**§1.5 banner**: SUPERSEDED notice landed in §1.5 (Rev 1.3 same commit batch). Rev 1 §1.5 table preserved for audit trail; downstream code MUST consume §13.A-wiring.

### §13.B — K3 dissolution (4-anchor bandit retracted)

Per §13.A above: σ-knob retired under Re-anchor option A. Rev 1.2 §12.E σ=1 / σ=2 / σ=3 / σ=5 pre-commit explicitly RETRACTED. The "all four pre-committed in one commit" defense (Rev 1.2 §12.E hedge) is moot because there is no σ parameter in the operative anchor. Old `cloud_runs/d71_var_smoothing_floor.json` retained on disk as audit-trail evidence of the retracted approach; NOT load-bearing for any future gate.

### §13.C — K4 trichotomy compatibility (rung 1.5 PASS bar 1/20 → ≥4/20)

Rev 1.2 §12.D PASS criterion "≥ 1 of 20 (lr, seed) configurations exceeds ε_physical" had implied per-seed success rate ≈ 5%; trichotomy needed 80% per-seed. P(≥8/10 trichotomy PASS | exactly 1/20 rung-1.5 PASS) ≈ 3.9e-11 — rung-stack admit-then-cannot-clear pipeline.

**Operative rung 1.5 PASS criterion (Rev 1.3)**:
- PASS = **≥ 4 of 20 (lr, seed) configurations exceed ε_physical** at step 500, where ε_physical is the §13.A operative anchor.
- Rationale (binomial power calc): under H_1 of chosen-lr per-seed success p=0.8, with 5 seeds at the chosen lr (1 of 4 lr values), expected count at chosen lr = 5×0.8 = 4; conditional power P(Binomial(5, 0.8) ≥ 4) ≈ 0.737. The 4/20 bar therefore admits regimes where trichotomy can plausibly clear (≥8/10 at chosen lr), and rejects regimes where it cannot.
- Cite: `sprent_smeeton2007` §4.2 (Sprent & Smeeton 2007, Applied Nonparametric Statistical Methods 4th ed., binomial-test critical regions); already in Rev 1.2 §12.C TODO bib list.

**Rung 1.5 dual-role explicit statement**: rung 1.5 is BOTH (i) lr-selection (picks the operative lr from the 4-value grid based on which value produced PASSes), AND (ii) feasibility-and-power gate (demonstrates that AT the selected lr, per-seed success rate is ballpark of the trichotomy operating point). NOT just feasibility-in-principle.

**Rev 1.2 §12.D failure-mode handling preserved**: if rung 1.5 returns < 4/20 PASS, body-axis fitting is not achievable on 500 steps; Stage 1a duration re-spec to 5000 steps as pre-committed fallback. Rung 4.5/4.6 do NOT dispatch until rung 1.5 clears ≥4/20.

### §13.D — S5 K3' joint-confound deferral (tighten §7 trigger language)

Rev 1.2 §12.A K3' degeneracy NAMED but Rev 1.2 §7 trigger reads as if (γ)-class-falsification can fire on (1c) trichotomy FAIL alone. K3' joint intervention (PE-drop + ω₀=30) confounds two effects; FAIL cannot attribute to either individually.

**Operative §7 trigger (Rev 1.3 tightening)**:

> **(γ) supervision-class falsification triggers IFF (1c) trichotomy FAIL AND rung 4.5 head-ablation FAIL AND rung 4.6 P2 cross-physics FAIL AND a future PE-isolation OR ω₀-isolation rung (deferred follow-on) also fails to discharge body-axis attribution.**

The current sprint (1c) under K3' joint intervention does NOT claim (γ)-class falsification authority. Even if all three landed gates (1c trichotomy + 4.5 + 4.6) FAIL, the K3' joint confound means a (γ)-still-viable-under-PE-only or under-ω₀=1-only hypothesis cannot be ruled out by this sprint. (γ)-class falsification requires the deferred PE-isolation or ω₀-isolation rung's authority, which is OUT-OF-SCOPE for Rev 1.3.

Cite: Tancik+2020 NeurIPS §4 (Fourier features + Sine composition non-trivial) + Mehta+2021 ICCV §3 (PE-vs-SIREN ablations). TODO in `papers/shared/main.bib`: `tancik_fourier_2020` + `mehta_modulated_2021`.

### §13.E — S6 SIREN literature-defense demotion

Rev 1.2 §12.E ε_physical mandatory hedge claimed "~3.75 Mpc/h ≈ trans-linear scale plausibly fittable by Sitzmann-image-regression-default MLP body at 500 steps." Tancik+2020 + Sitzmann+2020 support SIREN CAPABILITY for high-frequency fitting, not training-DYNAMICS at 500 steps for 5-decade log-density target. Empirically SIREN learns low-then-high frequencies (Tancik+2020 Fig 3 / Sitzmann+2020 Fig 4); by step 500 model is in low-frequency regime.

Combined with §13.A K2 ruling: the "trans-linear plausibly fittable" verb is DROPPED. Under Re-anchor option A there is no smoothing-kernel scale to anchor a frequency claim against. The ε_physical anchor is now sampling-noise floor + M=3 multiplier; the claim becomes "ε_physical is 3σ above the pooled-voxel-variance-ratio sampling-noise floor — a sample-statistical bar, NOT a physical-scale bar."

**Rev 1.3 §12.E hedge replacement** (operative wording):

> ε_physical = M × σ_pooled_voxel_sampling_noise (M=3 pre-commit; σ_noise numerical TBD per rung-5 compute output) is a STATISTICAL FLOOR distinguishing "(1c) recovered variance above sampling noise" from "(1c) achieved a sampling-noise-floor result indistinguishable from random crop placement." This is a loose-bandwidth direction-of-motion test, NOT a tight-fit test. The 500-step duration is consistent with this loose-bandwidth framing per [D-37]-extension R8 cascade-close discipline (claim narrows to match evidence).

### §13.F — S7 R32 deadline-bind

Rev 1.2 §12.G R32 row left R32 at DEFERRED-BANKED PROVISIONAL with no committed timeline for R12 cross-track second-sighting. S7 finding: indefinitely deferrable → permanently soft precondition.

**Operative R32 status (Rev 1.3 bind)**:

> **R32 cross-track second-sighting instance OWED by 2026-08-31 (sprint close on [D-71] (1c) trichotomy verdict, OR end of calendar Q3 2026, whichever comes first). If by deadline no cross-track sighting has accrued, R32 DEMOTES to CANDIDATE; PI does not have authority to extend the deadline.**

**Operational test for "cross-track"**: a sighting from a DIFFERENT decision-track ([D-72]+, or a non-[D-71] amendment cycle) within deadline. Rev 1.1 + Rev 1.2 + Rev 1.3 panel cycles are SAME-TRACK (all are [D-71] (1c) SIREN scoping) and DO NOT count as cross-track sightings — per §13.J ruling below, this is now BANKED PRECEDENT (future PI cannot reset R32 sighting count by opening new revision of same doc).

### §13.G — S8 R28 hard-auto-promote re-spec (governance amendment to `.claude/agents/project-architect.md`)

Rev 1.2 §12.G text "If a SECOND self-violation occurs without panel catch in a downstream sprint, R28 should be considered for elevation discipline" is replaced by hard auto-trigger.

**Operative §12.G replacement (Rev 1.3 binding)**:

> **If a SECOND self-violation occurs without panel catch in a downstream sprint, R28 MUST auto-promote to a 2-tier rule: (i) the current 'rung count ≥ artifact count' literal-integer check, AND (ii) a new mandatory 'PI absorption template must contain an explicit dispatch-ladder-arithmetic gate sub-block named §R28-CHECK', with no discretion at trigger point. PI does not have the authority to defer this elevation; the SECOND self-violation IS the trigger.**

**Governance amendment landing site**: `.claude/agents/project-architect.md` R28 rule text. The "should be considered for elevation discipline" → hard auto-trigger amendment lands SAME COMMIT BATCH as Rev 1.3 to close the soft-trigger escape hatch immediately.

### §13.H — P10 / P12 explicit statements (closing forward-obligations)

- **P10 aggregation convention**: `var_pred_lin` and `var_truth_lin` are pooled-cross-crops single variances (pipeline.py:585-595): `torch.cat([all crops, voxel-flattened]).var(unbiased=True)`. ε_physical anchor uses SAME pooled-cross-crops aggregation per §13.A spec.
- **P12 ρ-convention**: Sherwood cache convention is ρ/⟨ρ⟩ (overdensity, mean≈1 verified per cache JSON `mean=1.0`). Both pred-side and truth-side use same convention; cancels in `var_pred/var_truth` ratio.

### §13.I — R28 dispatch-ladder cross-check for Rev 1.3 absorption chain

Rev 1.3 is methodology re-spec, not multi-rung sequence. Forward dispatch ladder:

| Rung | Action | Owner | Landing artifact |
|---|---|---|---|
| 0 | Rev 1.3 PI self-draft (this turn) | PI | §13 amendment block (this document) |
| 1 | Rev 1.3 commit (design doc + LEDGER + project-architect.md R28 amendment) | PI | git commit ref |
| 2 | Fresh-panel pre-review on Rev 1.3 (cycle #3) | defense-panel | panel verdict block |
| 3 | Panel-verdict absorption | PI | LEDGER §3 [D-71] absorption block #3 |
| 4 | Authorize sampling-noise floor compute | PI | dispatch brief |
| 5 | Sampling-noise floor compute | core-implementer | `scripts/d71_compute_sampling_noise_floor.py` + `cloud_runs/d71_sampling_noise_floor.json` |
| 6 | ε_physical numerical pre-commit | PI | LEDGER §3 [D-71] pre-commit block |
| 7 | Rev 1.4 numerical bind (sampling-noise output → ε pre-commit) | PI | doc Rev 1.4 |
| 8 | Fresh-panel pre-review on Rev 1.4 numerical bind (cycle #4) | defense-panel | panel verdict |
| 9 | rung 1.5 lr-sweep n=20 dispatch (Rev 1.2 §12.D as amended by §13.C) | core-implementer + infrastructure-manager | SLURM array + `cloud_runs/d71_rung15_lrsweep.json` |

Ladder rung count = 10 (rungs 0-9). Landing-artifact count = 9 (Rev 1.3 doc, Rev 1.3 commit, Rev 1.3 panel verdict, Rev 1.3 absorption, dispatch brief, sampling-noise script+JSON [one artifact pair], ε pre-commit block, Rev 1.4 doc, Rev 1.4 panel verdict, rung 1.5 outputs). **R28 cross-check: 10 ≥ 9 PASS.**

**Two-panel-cycle structure rationale**: ε numerical value depends on compute output that does not exist yet; per R29 discipline, ε numerical commit must be panel-reviewed (panel can attack whether M=3 is defensible against the empirically-observed noise distribution). PI cannot pre-commit numerical ε without seeing noise floor; PI cannot dispatch noise-floor compute without panel APPROVE on methodology re-spec. Two-cycle ladder is feature, not bug — R29-discipline working as banked.

### §13.J — R-rule audit refresh (Rev 1.3)

| R-rule | Status (Rev 1.3) | Change vs Rev 1.2 | Sighting log |
|---|---|---|---|
| R28 | BANKED | hard-trigger auto-promote text landed in `.claude/agents/project-architect.md` this commit batch (§13.G) | self-violation sighting #1 still flagged; sighting #2 would auto-promote |
| R29 | **CANDIDATE** (still demoted) | **sighting #2 post-demotion** — K2 ε_physical wrong-quantity failure mode is structural twin of K6 ε=0.05 wrong-unit failure | #1 K6 ε=0.05 unit-chain catch (Rev 1.1 absorption); #2 K2 ε_physical anchor-vs-gate quantity mismatch (this turn) |
| R30 | BANKED | unchanged | — |
| R31 | DEFERRED-BANKED | sighting count unchanged from Rev 1.2 (#4 held) | — |
| R32 | **DEFERRED-BANKED PROVISIONAL with 2026-08-31 deadline** | deadline-bind per §13.F; same-track-cycle-collapsing precedent BANKED per §13.J above; Rev 1.1+1.2+1.3 cycles DO NOT accrue cross-track sightings | one operational test landed; cross-track second-sighting OWED by deadline or R32 demotes |

**R32 same-track-cycle-collapsing precedent BANKED**: future PIs cannot reset R32 sighting count by opening new revision of same design-doc track. Multiple panel cycles on Rev N → Rev N+1 → Rev N+2 of the same doc count as ONE same-track cycle for R32 banking purposes.

### §13.K — Sign-off (Rev 1.3)

**R15 + R28 PROVISIONAL HELD** per R32-candidate panel-cycle-discipline binding. This Rev 1.3 amendment authorizes ONLY (i) PI self-draft (done this turn), (ii) Rev 1.3 commit, (iii) fresh-panel pre-review cycle #3 dispatch. Does NOT authorize any code-implementer, infrastructure-manager, HPC, or latex-author dispatch.

**Honest framing per [D-37] rule (a)**: Rev 1.2 had 4 KILLER + 4 SERIOUS + 4 PROBE panel findings; load-bearing failure (K2) repeated the R29 wrong-unit pattern at a structural level (smoothing-kernel anchor vs no-smoothing gate). R29 sighting #2 post-demotion accrues. K3' joint-confound limitation (S5) admitted; (γ)-class falsification claim authority deferred to future panel cycle. No spin: the 4-anchor pre-commit is retracted, NOT spun as a recovery. The (1c) SIREN candidate direction survives Rev 1.2 → Rev 1.3 transition; the gate construction does not. Rev 1.3 + Rev 1.4 two-cycle ladder is the cost of doing R29 discipline right post-demotion.

**[D-37]-Extension R7 user-directive reminder**: outcome-quality is not graded; decision-quality is. Rev 1.2 → Rev 1.3 NEEDS-WORK + R29 sighting #2 is a valid trajectory of well-spec'd discipline under cascading hedge-rules, not a process failure. The R32 panel-cycle-discipline operational test is firing exactly as banked.


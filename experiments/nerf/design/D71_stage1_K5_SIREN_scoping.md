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
- **R29 (gate-construction-vs-production-framing audit)**: HOLDS — all M0/M1/M2 gates inherit [D-70] Rev 5.1 + amendment v2 + [D-71] frame-matched specifications (F1-β dual R_real, F4 Δ_seed framing, Wilcoxon H1='greater', MDE-block ε=0.05 anchored Boera+2019, σ_seed full_n10 source). No new gate construction in this doc.
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

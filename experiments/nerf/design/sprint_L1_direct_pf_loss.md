# Sprint L1 — Direct $P_F$ MSE loss test (v2)

**Status**: PI v2 design — NON-PROVISIONAL per [D-37]-Ext R15 clause (b) (gate-2 defense-panel returned NEEDS-WORK on v1 with 3 KILLER + 5 SERIOUS attacks; v2 absorbs all KILLERs and tightens all SERIOUS framings per the panel verdict)
**Predecessor**: v1 design + defense-panel gate-2 NEEDS-WORK verdict (both 2026-05-16)
**Goal**: Test whether adding a direct $P_F$ MSE loss term over the [D-13] inertial range $k_\parallel \in [10^{-2.5}, 10^{-1.5}]$ s/km closes the $P_F$ binding gate that fails uniformly cross-physics at $\sim 4\times$ the bar under the [D-24] supervision regime.

## v1 → v2 amendment summary

The panel surfaced three KILLER attacks that v1 did not anticipate:

- **K1** (estimator noise floor): per-ray $P_F$ at $n_{\rm rays}=64$ has $\chi^2_2$ statistics → noise floor ~0.13 std in $\log_{10} P_F$ per k-bin post-ray-averaging, vs gate signal of ~0.04 → the v1 single-cell P1-T1 test would have descended estimator noise.
- **K2** (training-vs-eval estimator equivalence): v1 said "MUST match exactly" with no verification protocol; the eval estimator (`compute_p_flux`) has 4 non-trivial steps (mean-divide, Hann window, FFT, log-k scatter-bin) all of which must be re-implemented differentiably and tested for equivalence to 1e-6.
- **K3** (R-f at 50% gaming the bar): v1's intermediate retire condition allowed |ΔP_F/P_F| at $\sim 2\times$ over the gate at step 25k to pass, then fail §5 success at step 50k with no early-retire signal.

Plus five SERIOUS framings to tighten: Pearson r → cross-coherence; step-function λ-retune → GradNorm; R10 X⊥Y partial-orthogonality; cross-physics escalation tally pre-committed; cite-precedent gap.

v2 absorbs by: (a) switch the L1 test from P1-T1 (64 rays) to P1-T3 (1024 rays) to drop the noise floor to ~0.03 std in $\log_{10} P_F$, comparable to gate signal; (b) gate-4 explicit equivalence unit test as a hard deliverable; (c) R-f tightened to "$|\Delta P_F/P_F| \leq 0.20$ AND monotone-decreasing over prior 5k steps"; (d) R-e replaced with median per-ray cross-coherence $|\gamma(k)|^2 \geq 0.5$ for $\geq 4/6$ inertial k-bins; (e) GradNorm (Chen+ 2018, α=0.12) replaces fixed-λ; (f) §1 amended to acknowledge L1 inherits D1's supervision-conflict risk on the τ-MSE-degradation axis (R-c is the backstop); (g) §5 pre-committed cross-physics tally; (h) §8 acknowledges no published precedent for Pk-MSE-as-loss in Lyα-reconstruction (closest: Harrington+ 2021 Deep Forest; Cabayol-Garcia+ 2023 emulator).

## 1. Diagnostic context

The [D-24] training loss `log1p(cap(τ_pred, τ_max)) - log1p(cap(τ_truth, τ_max))` permits multiple reconstructions that satisfy single-statistic flux moments (KS, mean-flux) but fail the spectroscopic statistic ($P_F$). The 4-axis counterfactual cascade closed all four candidate fixes at smoke:

- **D1** loss-INTEGRATED ([D-40] sat-aware $P_F$ band loss) → constant-prediction collapse, +37.4% worse
- **D2** loss-PER-PIXEL ([D-41] FGPA-tail per-pixel regularizer) → constant-prediction collapse on self-consistent state
- **D3** architecture-INPUT ([D-42] velocity-gradient conditioning) → density-head collapse
- **D4** data-axis ([D-46] joint-physics conditional MLP) → combined-trivial-collapse-with-active-embedding

The structural lesson from D1/D2: **adding regularization terms while keeping the same loss target invites degenerate collapse**.

**L1 differs from D1 on the COLLAPSE-MODE axis** (constant-flux precluded by k-space weighting: a constant flux has zero power at every $k$ → maximally bad against nonzero truth $P_F$ → D1's flux-band-cheating collapse mode is structurally impossible under L1). **L1 INHERITS D1's SUPERVISION-CONFLICT risk** — two loss terms can pull the network into a compromise that satisfies neither (D1 also retired with τ-MSE degradation at +37.4%). R-c (val τ-MSE > 2.0× [D-24] baseline retire) is the explicit backstop for that shared risk; the v2 framing names both the structural difference and the shared risk.

## 2. Loss formulation

$$\mathcal{L}_{\rm total} = w_\tau(t) \cdot \mathcal{L}_{[D-24]} + w_{P_F}(t) \cdot \mathcal{L}_{P_F}$$

where $w_\tau(t)$ and $w_{P_F}(t)$ are GradNorm-balanced task weights (Chen et al. 2018, $\alpha=0.12$ — their default; references the standard formulation $w_i \cdot \mathcal{L}_i$ where $w_i$ are themselves trainable to keep $|\nabla_\theta w_i \mathcal{L}_i|$ ratios near 1.0 with respect to $|\nabla_\theta w_j \mathcal{L}_j|$). Initialized $w_\tau = w_{P_F} = 1.0$; updated every step alongside the model parameters.

**Loss term** (K1-absorbing — ray-averaged inside the log-MSE):

$$\mathcal{L}_{P_F} = \sum_{k \in K_{\rm inertial}} \left( \log_{10} \langle P_F^{\rm pred}\rangle_{\rm rays}(k) - \log_{10} \langle P_F^{\rm truth}\rangle_{\rm rays}(k) \right)^2$$

where the $\langle \cdot \rangle_{\rm rays}$ averaging is over **all $n_{\rm rays}=1024$ rays in the batch** (the T3 sightline set), matching the eval-time `compute_p_flux` semantic exactly. $K_{\rm inertial} = \{k_\parallel : 10^{-2.5} \leq k_\parallel \leq 10^{-1.5} \text{ s/km}\}$ (~6 k-bins on the Sherwood velocity grid at $n_{\rm obs}=2048$).

**Noise-floor calculation** (K1-absorbing): per-ray periodogram $P_F$ has $\chi^2_2$ statistics → linear var($P_F$)/$\langle P_F\rangle^2 = 1$ → log-space std = $1/\sqrt{N_{\rm rays}} \approx 1/\sqrt{1024} \approx 0.031$ per k-bin post-ray-averaging. Gate signal $|\Delta P_F/P_F| < 0.10$ corresponds to $\log_{10}(1.1) \approx 0.041$. Noise floor ~0.031 vs target ~0.041 → noise-to-signal ratio ~0.76, **comparable but below**. The L1 test at T3 can distinguish "gate passes" from "noise" at marginal statistical resolution; at T1 ($n_{\rm rays}=64$) the noise floor would be 4× the signal and the test would be uninterpretable.

**Implementation requirements** (K2-absorbing — gate-4 hard deliverable):

The training-time $P_F$ estimator must be a differentiable torch reimplementation of `src/analysis/p_flux.compute_p_flux` matching its four steps:

1. mean-divide normalization $\delta_F = F/\langle F\rangle - 1$ (NOT mean-subtract — [D-35] case-of-record)
2. Hann window with $dv/\sum w^2$ leakage compensation (matching `p_flux.py:79-84`)
3. `torch.fft.rfft` (matching `np.fft.rfft` at float64; numerically equivalent up to FP-summation order)
4. log-spaced k-binning with `n_kbins=20` and empty-bin → 0.0 in training (eval uses NaN; equivalence test verifies non-empty bins match)

**Unit test `tests/test_torch_pf_estimator_equivalence.py` is a hard gate-4 deliverable**: assert `torch_p_flux(F_torch).cpu().numpy()` equals `compute_p_flux(F_torch.numpy())` to within 1e-6 absolute and 1e-4 relative on 10 randomized F batches of shape (1024, 2048).

## 3. Anti-degeneracy backstops (v2)

**B1. Translation-symmetry collapse**: same as v1 — track val τ-MSE; retire if > 2.0× [D-24] baseline (R-c).

**B2. Flat-flux constant-prediction (D1/D2 carry-over risk)**: track Var($F_{\rm pred}$) over the sightline; retire if < 0.5× Var($F_{\rm truth}$) after burn-in 500 steps (R-d). Burn-in reduced from v1's 1000 → 500 (P3 absorption): network flat at 500 steps is already stuck.

**B3. P_F-amplitude-only collapse (REPLACED with coherence per S1)**: $P_F$ is phase-blind, so a network producing right-amplitude-wrong-phase flux would pass L1 perfectly while being uncorrelated with truth. Pearson $r$ (v1) is too weak — nearby Lyα sightlines share large-scale modes from the underlying cosmological density field, so baseline Pearson $r$ between unrelated rays can sit at 0.3–0.5 by chance (Croft+ 2002 §6). **R-e (v2)** uses cross-spectrum coherence:

$$|\gamma_{F_{\rm pred}, F_{\rm truth}}(k)|^2 = \frac{|S_{\rm pred, truth}(k)|^2}{S_{\rm pred, pred}(k) \cdot S_{\rm truth, truth}(k)}$$

Retire if median per-ray $|\gamma(k)|^2 \geq 0.5$ does **not** hold for at least 4 of 6 inertial-range k-bins at any step after burn-in 5k (Thrane & Romano 2013 §III for coherence framework).

**B4. λ-imbalance (REPLACED with GradNorm per S2)**: v1's fixed-λ-set-at-init + 5k-step retune was unstable. **v2 uses GradNorm** (Chen+ 2018) with $\alpha=0.12$ — the standard adaptive task-weighting scheme that maintains $|\nabla_\theta w_i \mathcal{L}_i|$ ratios near a learned-balanced target throughout training. No step-function retune; smooth continuous balancing. **B4 backstop now**: log $w_\tau / w_{P_F}$ ratio history at MLflow; retire if ratio exceeds 1000:1 in either direction at any step (run-away weight collapse — GradNorm's own pathological mode).

## 4. Pre-committed retire conditions (v2)

| Code | Trigger | Step window |
|:---|:---|:---|
| R-a | Loss NaN/Inf or training-loss divergence (loss > 10× initial) | within 1k steps |
| R-b | $P_F^{\rm pred}$ collapses to flat-zero or flat-constant (Var$_k$ $P_F^{\rm pred}$ < 0.1 × Var$_k$ $P_F^{\rm truth}$ over inertial range) | within 5k steps |
| R-c | val $\tau$-MSE > 2.0× [D-24] baseline | any step after burn-in 1k |
| R-d | Var($F_{\rm pred}$) < 0.5 × Var($F_{\rm truth}$) | any step after burn-in 500 |
| R-e | **median per-ray $|\gamma(k)|^2 < 0.5$ in $<\!4/6$ inertial-range k-bins** | any step after 5k |
| R-f | **$|\Delta P_F/P_F| > 0.20$ at step 25k, OR not monotone-decreasing over prior 5k steps (slope $\geq$ 0)** | step 25k |
| R-g | GradNorm weight ratio $w_\tau / w_{P_F}$ exceeds 1000:1 in either direction | any step |
| R-h | total wallclock exceeds 1.5× design envelope | n/a |

K3-tightened R-f: at step 25k the residual must be **below half the original 4× baseline** ($< 0.20$ absolute) **and trending in the right direction** (slope < 0 over the prior 5k-step window). A model stalled at 0.20 retires; a model that overshoots-and-stalls retires.

## 5. Success criteria (v2 — pre-committed cross-physics tally)

**Single-cell P1-T3 success (gate-7 absorption)**:

1. No retire condition triggered through 50k steps.
2. Step-50k inertial-range $|\Delta P_F/P_F| < 0.10$ ([D-13] gate).
3. Step-50k KS distance < 0.10 ([D-13] gate); step-50k mean-flux within $\pm 2\sigma$ of Kirkman+2007 anchor.

If 1–3 hold → **escalate to 4-physics × T3 sweep** under the v2-pre-committed multi-cell tally.

**Pre-committed 4-physics × T3 sweep tally** (S4 absorption — locked before sweep dispatch):

| Cell-tally | Verdict |
|:---|:---|
| $\geq 3/4$ cells with step-50k $|\Delta P_F/P_F| < 0.10$ | **CLOSE — L1 is a working loss formulation** → publish + run T4 full schedule |
| $2/4$ cells with step-50k $|\Delta P_F/P_F| < 0.10$ | **partial-with-physics-analysis-required** → which physics breaks L1? → separate analysis sprint |
| $\leq 1/4$ cells with step-50k $|\Delta P_F/P_F| < 0.10$ | **retire-as-loss-tweak-insufficient** → proceed to L2 or [D-53] per next-sprint PI design |

If single-cell 1–2 hold but 3 fails → partial success at the single-cell level; surface to user before sweep decision.

## 6. Compute envelope (v2)

- **Single-cell L1 sprint at P1-T3** ($n_{\rm rays}=1024$, 50k steps): 1 A30 GPU at Juno HPC, $\sim 7$ hr wallclock $\approx \$11$ (T3 is ~7× T1 per [D-23] linear-VRAM scaling)
- **Smokes**: memory smoke (5 steps, 1 min on T3 batch); host smoke (200 steps, 12 min on host GPU at T3) — runs locally before Juno dispatch
- **Full retire path** (R-a..R-h triggered before 50k): wallclock varies; R-h cap is 10.5 hr
- **4-physics × T3 sweep (only if single-cell closes)**: 4 cells × 7 hr ≈ 28 hr Juno A30 ≈ \$42

Total worst-case (single-cell L1 fail): $11; total best-case + sweep: $53. Well within budget.

## 7. Dispatch sequence (gates)

| Gate | Actor | Deliverable |
|:---|:---|:---|
| Gate-1 | PI | Design v1 (BANKED) |
| Gate-2 | defense-panel | NEEDS-WORK verdict (RETURNED with K1/K2/K3 + S1/S2/S3/S4/S5) |
| Gate-3 | PI | **THIS v2 absorption** — NON-PROVISIONAL per R15 clause (b) |
| Gate-4 | core-implementer | Land `src/training/p_flux_loss.py` (differentiable torch P_F estimator + log-MSE loss + GradNorm wrapper) + wire into `experiments/nerf/pipeline.py` + **5+ unit tests INCLUDING explicit `tests/test_torch_pf_estimator_equivalence.py` asserting 1e-6 / 1e-4 equivalence to `compute_p_flux` over 10 randomized batches** + memory smoke + host smoke (200 steps) |
| Gate-5 | PI | Sign-off on smoke results + retire-condition pre-check |
| Gate-6 | infrastructure-manager | Juno A30 dispatch (50k steps single-cell P1-T3); MLflow log includes per-step $w_\tau / w_{P_F}$ + inertial-range $|\Delta P_F/P_F|$ + median coherence |
| Gate-7 | PI | Absorption: write [D-60] CLOSE block with empirical observation + branch-routing per §5 tally |

## 8. References (v2 — cite-precedent absorbed per S5)

### Cited (load-bearing for the design)

- **[D-13]** (the three [D-13] gates including $P_F$ inertial-range residual)
- **[D-15]** ($P_F$ gate 10% bar, post-[D-36] retraction of external attribution)
- **[D-24]** (existing log1p+cap+mask supervision regime)
- **[D-35]** (mean-subtract vs mean-divide convention bug; load-bearing for K2 case-of-record on convention-pinning)
- **[D-37]-Ext 1 + 2** (R8–R23 rule-set; R10 retired-model-reuse contract — load-bearing for §1 "L1 differs from D1 on collapse axis but shares supervision-conflict risk" framing per S3; R15 NON-PROVISIONAL clause (b))
- **[D-39]** ($P_F$ binding-gate identification + saturation-band positive ID)
- **[D-40]** (sat-aware $P_F$ band loss RETIRED — primary case-of-record for L1's collapse-mode-axis differentiation)
- **[D-41]/[D-42]/[D-46]** (other 4-axis cascade closes)
- **[D-43]** (CVPR Step 5 CLOSED at HEAD `d7a23ca` — L1 is post-CVPR)
- **[D-53]** (supervision-target axis STILL OPEN; L1 attacks loss-tweak side, see §9)
- **[D-54]/[D-57]/[D-58]/[D-59]** (project-completion audit arc)
- **`src/analysis/p_flux.py`** (eval-time estimator that the training-time estimator must match — K2 load-bearing)
- **Chen et al. 2018 (GradNorm)** — *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, ICML; load-bearing for S2 absorption (multi-task weight balancing)
- **Thrane & Romano 2013** — *Sensitivity curves for searches for gravitational-wave backgrounds*, PRD 88, 124032; load-bearing for S1 absorption (cross-spectrum coherence framework distinguishing amplitude from phase agreement)
- **Croft et al. 2002** — *Toward a precise measurement of matter clustering: Lyα forest data at z=2-4*, ApJ 581, 20; load-bearing for S1 (sightline-correlation baseline for the coherence threshold)
- **D'Amour et al. 2020** — *Underspecification Presents Challenges for Credibility in Modern Machine Learning*, JMLR; load-bearing for §4.2 underspecification framing → motivates the L1 hypothesis
- **Lukić et al. 2015** — *The Lyman α forest in optically thin hydrodynamical simulations*, MNRAS 446, 3697; load-bearing for K1 (Sherwood $P_F$ convergence at $\gtrsim 5000$ skewers context)
- **[D-13] estimator definition references**: Walther et al. 2018 §3.2 (Hann-window $P_F$ on observational data); Boera et al. 2019 (Lomb-Scargle-on-observation precedent; not used here, but cite anchor)

### Adjacent ML+Lyα work (S5 absorption — cite-precedent surface, none uses Pk-MSE-as-loss)

- **Harrington et al. 2021 (Deep Forest)** — MNRAS 506, 5212; CNN reconstruction of Lyα forest from spectra. Closest prior reconstruction work. Uses per-bin flux MSE (not Pk-MSE). L1 is novel relative to this baseline.
- **Cabayol-Garcia et al. 2023** — arXiv:2305.19064; neural network *emulator* for Lyα 1D flux power spectrum given cosmological parameters. Pk is OUTPUT, not loss term. Tangential to L1.
- **Maitra et al. 2023 (LyαNNA)** — arXiv:2311.02167; field-level deep-learning inference for Lyα. Uses ResNet on full flux time-series for parameter inference; doesn't reconstruct 3D fields.
- **Villaescusa-Navarro et al. 2022 (CAMELS Multifield Dataset)** — ApJS 259, 61; dataset paper for 2D maps + 3D grids of cosmic fields. NOT a loss-term paper despite the panel's initial framing; no Pk-residual loss term in the published companion works at the time of WebSearch verification (2026-05-16).

**Cite-precedent verdict**: no published precedent for Pk-MSE-as-loss-term in Lyα-reconstruction. L1 is methodologically novel in this surface. The design honestly names this novelty rather than back-fitting cites.

## 9. Carry-forward (v2 — both L2 and [D-53] are candidate next sprints)

**If L1 closes (single-cell + 4-physics tally CLOSE)**:
- Run T4 full schedule under L1; write [D-XX] absorption demonstrating Pk-MSE-as-loss closes the [D-13] gate at publication-tier production
- Reframe the §4.2 paper finding: "underspecification is a real failure mode of [D-24]; adding the binding observable Pk directly to the supervision signal closes the gate"
- Post-CVPR paper or MNRAS submission target

**If L1 retires (any of R-a..R-h triggers, OR single-cell §5 fails, OR sweep ≤ 1/4)**:
- **Two candidate next sprints**: (A) **L2 — flux-domain loss** (replace τ-MSE with F-MSE); or (B) **[D-53] supervision-target axis sprint** (operate on the literal NeRF-side τ-transform axis with ≥ 2 alternative supervision-target formulations: Iršič+2017 flux-domain loss; Boera+2019 saturation-aware τ transform)
- **Sequencing decision is NOT pre-committed at this gate** — depends on the L1 retire mode (R-a/R-b/R-d → degenerate-collapse insight relevant to L2; R-c/R-e/R-f → genuine supervision-conflict insight relevant to [D-53])
- Out-of-scope-2 absorption (defense panel): the v1 framing that defaulted to L2 was inconsistent with [D-53] being load-bearing; v2 explicitly names both candidates

**If L1 partial-closes (2/4 in the sweep)**:
- Physics-analysis sprint: which feedback variant breaks L1? Is there a structural reason (e.g., AGN-driven outflows create kinematic signatures that the loss can't capture)?
- Sequence after the physics analysis: re-evaluate next sprint with the analysis result

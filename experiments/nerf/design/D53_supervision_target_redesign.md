# [D-53] Supervision-Target Redesign — Design Doc

## Status

- **PROVISIONAL** pending defense-panel design review (compute >$5-equiv Juno time AND any paper-section claim are panel-gated per [D-61] routing).
- **Activated 2026-05-23** per [D-61] gate-L2 close (LEDGER commit `d23ccc1`, R15 NON-PROVISIONAL).
- Author: support-researcher per [D-61] carry-forward.
- **Defense-panel verdict absorbed 2026-05-23** (commit `3074596`): (c) REJECTED for first-slot dispatch (see candidate (c) header for full verdict). (b) is **first-binding**: panel-bound selection. (a) **reworked per KILLER-1 + KILLER-2 + SERIOUS-3 absorption** (rename to Bolton+2008-tradition framing, triangular kernel bandwidth=bin_width/3 pre-commit, K2-equivalent re-spec'd to soft-vs-soft) and lifted from PROVISIONAL to **dispatch-eligible-for-later-slot pending PI sign-off** — (a) remains LATER-SLOT (not first-dispatch); (b) is first-binding per panel.
- Ranking among (a)/(b)/(c): (b) first, (a) later-slot, (c) rejected for first-slot.

## Context

[D-60] Sprint-L1 closed at [D-61] (commit `d23ccc1`) with the **L1 in-loss-function intervention class empirically exhausted** across 4 attempts (LR axis, warmup axis, per-task-clip [dead-lever per retune-3], reduction-op axis). All 4 R-b retired at step 200 with `var_pf_band_ratio ∈ [6.9e-7, 2.93e-6]` (0.63 log10-decade spread). The mechanism-targeted lever (mean reduction at L2) acted as predicted — per-task gradient ratio dropped ~10× from retune-3's 15599 → L2's 1429 — yet variance collapse was unchanged. **Per-task-ratio-as-causal-axis hypothesis FALSIFIED.**

Per [D-61] inference: the pathology lives **in the supervision-target structure**, NOT in any scalar weighting / LR / reduction-op choice over a fixed target. [D-53] candidates are the **first test of the supervision-target-redesign class** — they are NOT pre-justified as structurally addressing the upstream pathology. Verb-level discipline per [D-37]-ext rule 2 holds throughout this doc.

**Stop-gate (re-affirmed at [D-61]):** if the first [D-53] candidate dispatched also exhibits R-b pattern at step 200, close at **[D-62] direct-P_F-MSE-loss class exhausted** and escalate to architectural pivot (model-side capacity / inductive-bias, NOT loss-side). Do NOT iterate within [D-53] until 4-attempt depth without panel re-review.

## Prior-failure ledger (per [D-37]-ext rule 4 symmetric-disclosure)

All 4 attempts: P1-T1, `--gradnorm-full`, `--enable-l1-pf-loss`, step-200 R-b:`pf_pred_variance_collapse` retire on commit-stamped runs.

| Attempt | jobid | var_pf_band_ratio | log10 | per-task ratio | Lever axis | Source artifact |
|---------|-------|-------------------|-------|----------------|------------|-----------------|
| retune-1 | 201734 | 2.51e-6 | -5.60 | (not logged) | LR-axis (lr=1e-4 / wu=1000) | `cloud_runs/sprint_L1_retune1_201734/retire.json` |
| retune-2 | 201814 | 6.9e-7  | -6.16 | 20807 | LR-axis (lr=3e-5 / wu=2000) | `cloud_runs/sprint_L1_retune2_201814/` |
| retune-3 | 201856 | 1.75e-6 | -5.76 | 15599 | Per-task-clip (dead-lever; clip never engaged before retire) | `cloud_runs/sprint_L1_retune3_201856/` |
| L2 | 202109 | 2.93e-6 | -5.53 | **1429** | Reduction op (sum→mean) | `cloud_runs/sprint_L1_L2_202109/{driver.log,retire.json}` |

Closure verdict: **L1 in-loss-function class** = {LR axis, warmup axis, per-task gradient clip [dead-lever per retune-3], reduction-op axis}. 4 attempts, 4 R-b retires, 0.63 log10-decade spread. R13-compliant scope-locked closure. Architectural / hybrid-loss interventions ([D-53] candidate (c)) are explicitly NOT in the closed class.

---

## Candidate (a) — joint flux-PDF + P_F supervision (Bolton+2008-tradition PDF binning, this-project loss formulation)

### 0. Literature audit + honest framing per [D-37] rule (a)

**Important caveat (panel-rename absorption 2026-05-23, KILLER-1).** Prior v5 stub mislabeled this candidate "Iršič+2017-style flux-domain loss". Iršič et al. 2017 (`arXiv:1702.01761`) is the **XQ-100 P_F measurement paper** using covariance-weighted χ² for an observational power-spectrum measurement — NOT a NN loss-function formulation. The candidate is renamed throughout to **"joint flux-PDF + P_F supervision (Bolton+2008-tradition PDF binning, this-project loss formulation)"**. The actual published-precedent for joint flux-PDF + P_F constraints in cosmological inference is **Bolton+2008 MNRAS 386:1131** (20-50 bin PDF convention, §3) and **Lee+2015** (flux-PDF constraints from BOSS). There is **no direct citation of a published flux-PDF + P_F NN loss in that exact form for IGM-NeRF-class models** in the literature this researcher could locate. Closest precedents are summary-statistic-matching emulators (Cabayol-García et al. 2023 `arXiv:2305.19064`, Pedersen et al. emulator series) and field-level inference (Nayak/Maitra et al. 2023 `arXiv:2311.02167` "LyαNNA") — both **inference** machines targeting cosmological parameters from observed P_F + PDF, not **reconstruction** machines training a 3-D field against simulated PDF + P_F. Candidate (a) thus extrapolates from the Bolton-tradition measurement convention into a NN loss; the panel should weight the absence of a direct precedent.

### 1. Mechanism prediction

**Proposed property of the target that produces non-degenerate gradients on the diffuse-bin majority:** the joint (flux PDF, P_F) target replaces a single τ-MSE-derived scalar field with two **distribution-level** summary statistics. The flux PDF p(F) over the unit interval is histogram-binned (Bolton+2008 §3 convention, pre-committed **n_bins=30** unless future literature audit surfaces a different number); the gradient on each bin is non-degenerate so long as **any** voxel in the underlying sightline contributes to that flux bin under the differentiable Voigt path. The diffuse-bin majority — which carries τ ≪ 1, F ≈ 1 — concentrates near the PDF's F ≈ 1 bin, so a PDF-bin-MSE loss on that bin's height produces a coherent gradient signal across the diffuse-bin majority **rather than the per-voxel scalar imbalance that τ-MSE produces**.

**Panel-binding kernel/bandwidth pre-commit (defense-panel 2026-05-23, KILLER-1 absorption):** soft-binning kernel = **triangular**, bandwidth = **bin_width / 3**. Triangular has **compact support** (unlike Gaussian) so leakage from any input flux value is bounded to exactly the host bin + 2 adjacent bins; with bandwidth = bin_width/3, the leakage to each neighbor from a delta-input is **exactly 1/3 of the kernel mass** (the triangular kernel integrated over an adjacent bin of equal width).

**Bandwidth-preservation-of-variance-penalty audit (panel KILLER-1 absorption, demonstrating the constant-F PDF remains distinguishable from truth-PDF at the K2-equivalent tolerance):**

Consider the failure-mode geometry. A variance-collapsed solution produces a near-constant flux F ≈ F₀; under triangular soft-binning with bandwidth bin_width/3, this concentrates probability mass in the single bin containing F₀ plus exactly 1/3-mass leakage into each of its 2 neighbors — a 3-bin spike profile with weights (1/3, 1/3, 1/3) (or asymmetric if F₀ near a bin boundary; max leakage 1/3 per side regardless). With n_bins=30 this is a 3-bin / 30-bin = 10% support footprint at peak ~0.33 height per bin.

By contrast, the truth flux PDF at typical IGM z ≈ 4-5 (Bolton+2008 Fig. 3, Lee+2015 Fig. 9) spreads probability across **~10-20 of the 30 bins** with the dominant F ≈ 1 saturated-transmission bin carrying ~0.3-0.5 mass and a long left-tail to F ≈ 0 (Lyα absorbers). The L2 distance ‖p_const − p_truth‖₂ between a 3-bin spike (max bin height ~0.33) and a 10-20-bin distribution (max bin height ~0.3-0.5, support spanning 10× more bins) is **bounded from below by at least ~0.3 in L2 norm and ~0.5 in KL** — well above any K2-equivalent rtol=1e-4 floor. The constant-F basin is therefore **explicitly penalized**, not numerically obscured by kernel smearing. Gaussian kernel with σ ≳ bin_width/2 would smear the constant-F spike across ~6-8 bins and reduce the L2 separation below ~0.1 — which is why triangular bandwidth=bin_width/3 is the binding selector, not Gaussian.

**Why THIS target might rescue variance collapse where L1's τ-MSE-derived target did not (hedged):** the L1 sequence's failure mode was variance collapse of the predicted P_F band — the model converged on a near-constant flux that minimized τ-MSE locally but produced zero P_F variance. A PDF + P_F joint target **explicitly penalizes constant-flux solutions** per the bounded-from-below L2/KL separation above, so the geometric structure of the loss surface around the variance-collapsed basin is materially different. The literature does not specify this mechanism for neural reconstruction — it is the candidate's mechanism-prediction, not a citation.

**Hedge:** "candidate (a) is the first test of a flux-domain-distribution-supervision target in this project; we do NOT claim it structurally addresses the upstream pathology — that claim is reserved for empirical step-200 / step-1000 / step-5000 evidence."

### 2. Estimator-equivalence test specification

Pre-committed test analogous to [D-60] gate-4 K2 (`tests/test_torch_pf_estimator_equivalence.py:110-113` certifying `torch_p_flux` ≡ `compute_p_flux` at 1e-6/1e-4 over 10 batches).

**K2-equivalent for candidate (a) (re-spec per panel SERIOUS-3 absorption 2026-05-23):** a new `tests/test_torch_flux_pdf_estimator_equivalence.py` that asserts the **differentiable** `torch_soft_pdf(flux, n_bins=30, kernel='triangular', bandwidth=bin_width/3)` produces histogram-bin heights matching a **numpy soft-binning reference at the SAME kernel** `numpy_soft_pdf(flux, n_bins=30, kernel='triangular', bandwidth=bin_width/3)` at `rtol=1e-4 / atol=1e-6` over 10 randomized batches (same seed-sweep as K2). **Critical re-spec note**: the prior v5 stub specified soft-bin-torch vs hard-bin-numpy at rtol=1e-4 — panel verified (SERIOUS-3) that this is **structurally impossible to pass at tight tolerance** for any soft-binning kernel with non-trivial bandwidth (soft and hard binning genuinely disagree at the 1-10% level for any kernel wide enough to be differentiable). The re-spec'd test compares soft-binning torch against soft-binning numpy at the **same kernel + bandwidth**, which is the proper equivalence check (does the torch implementation faithfully reproduce the numpy reference of the same mathematical object) — rtol=1e-4 is then defensible. **Pre-flight gate:** test must PASS before any Juno dispatch. Joint-loss test runs P_F estimator (K2 already certified) and PDF estimator independently — joint scalar is just the weighted sum, no separate equivalence needed.

### 3. Pre-committed falsification criterion

**Same step-200 R-b pattern as L1 closes this candidate.** Specifically: if at step 200 the predicted P_F band variance ratio (`var_pf_band_ratio`) is `< 1e-3`, this candidate is **closed as R-b** — invokes the [D-62] stop-gate and routes to architectural pivot WITHOUT iterating to candidates (b)/(c).

**Secondary criteria (step 1000 and step 5000), softer, do NOT individually close the candidate but must all pass for PASS verdict:**
- Step 1000: `var_pf_band_ratio > 0.05` (one full decade above the L1 cluster's worst point).
- Step 5000: `var_pf_band_ratio > 0.3` (the [D-60] retune-gate threshold).
- Amendment B (carry from L2): `pf/tau ratio < 3.0` at step 5000 AND `loss_tau(5000) < 0.9 × loss_tau(1000)` (tau-loss moved).

**Symmetric-disclosure per [D-37]-ext rule 5:** the step-200 `var_pf_band_ratio < 1e-3` criterion is the same R-b pattern the L1 sequence retired on; we pre-commit to closing candidate (a) on the same signature.

### 4. Falsified-prior cascade hedged verbs (text-level audit)

- "Candidate (a) is the **first test** of the flux-domain-distribution supervision-target class."
- "Candidate (a) **may** rescue variance collapse via PDF-bin gradient structure — pending step-200 / step-1000 / step-5000 evidence."
- AVOIDED: "candidate (a) structurally addresses the upstream pathology" (forbidden verb per rule 2).
- AVOIDED: "Bolton+2008-tradition joint flux-PDF + P_F losses are known to rescue this failure mode in IGM neural-net training" (would require literature citation that does not exist per §0 — Bolton+2008 is a measurement-convention citation, not a NN-loss-precedent citation).

### 5. Compute envelope + smoke spec

- **Smoke**: ≤30 min Juno A30, P1-T1 (n_rays=64, microbatch=256), max_steps=200 (the falsification window). Same step-200 instrumentation as [D-60] (logs `var_pf_band_ratio`, per-task `w_ratio`, `loss_pf`, `loss_tau` at diag cadence {100, 200}).
- **CPU 1-step dry-run pre-flight** per L2 sbatch template (`scripts/submit_juno_sprint_L1_L2.sh:188-258`): `--max_steps 1` on CPU with `CUDA_VISIBLE_DEVICES=""` subshell, asserts (i) exit 0, (ii) no NaN/Inf in numeric fields, (iii) `Step 1/` line present, (iv) `Training finished.` line present. Aborts before Juno GPU on dry-run FAIL.
- **Sbatch derived from** `scripts/submit_juno_sprint_L1_L2.sh` with two changes: (i) loss flag `--enable-l1-fluxpdf-pf-joint` (new, panel-renamed per KILLER-1 from prior `--enable-l1-flux-domain`) replaces `--enable-l1-pf-loss`; (ii) wallclock relaxed from 24:00:00 to 04:00:00 since the smoke target is 200-step retire-or-pass, not 5k survival.
- **Pre-committed smoke FAIL routing**: R-b at step 200 → [D-62] stop-gate, escalate to architectural pivot per [D-61]. Do NOT auto-route to candidate (b) without panel re-review.
- **Fairshare**: P1-T1 200-step at A30 historically ~10-15 min wallclock under [D-60] cadence; well within Juno Fairshare envelope.

---

## Candidate (b) — k-space-normalized P_F target (panel PROBE-7 origin)

### 0. Literature audit + honest framing per [D-37] rule (a)

**Important caveat.** This researcher could not locate a **direct precedent** for a k-space-normalized P_F target as a neural-network loss in IGM ML literature. The closest related concept is the standard cosmology practice of **per-mode error-bar weighting** in chi-squared P_F fitting (Iršič+2017 §4 systematic-error budget; XQ-100 covariance matrix), but that is a chi-squared **fit-statistic** convention not a NN supervision target. Cabayol-García et al. 2023 (`arXiv:2305.19064`) emulator uses MSE on `log P_1D(k)` per k-bin (its sub-percent claim is per-k-bin), which is **closer** to candidate (b)'s intent than to candidate (a)'s — log-MSE in k-space is functionally a per-mode-normalized loss when the modes' dynamic range is large. The panel should treat candidate (b) as a **synthesis** of the Cabayol-García log-k-MSE convention with this project's PROBE-7 surfacing, NOT as a published-precedent design.

### 1. Mechanism prediction

**Proposed property of the target that produces non-degenerate gradients on the diffuse-bin majority:** the k-space-normalized P_F target re-weights the per-mode contribution to the loss by `1 / σ_k²` (per-mode variance), so high-variance modes (typically low-k, dominated by saturated absorbers) do not dominate the gradient over low-variance modes (typically high-k inertial-band modes, dominated by diffuse-bin contributions). In the current `pf_log_mse_loss` (`src/training/p_flux_loss.py:262`) the inertial-band sum/mean is taken on `(log P_pred − log P_truth)²` per bin; candidate (b) replaces this with `L = Σ_k (P_pred(k) − P_truth(k))² / σ_k²_truth(k)` where σ_k² is **truth-side** (NOT predicted-side — see panel pre-commit below).

**Panel-binding pre-commit on σ_k² estimator (defense-panel design review 2026-05-23, KILLER-1 absorption)**: σ_k² estimator is **truth-side, batch-sample, EMA-stabilized with decay 0.99**, with floor `σ²_floor = 0.01 × median_k(σ_k²_truth)` (relative, not the prior-absolute 1e-12). Truth-side eliminates chicken-and-egg (predicted-side at step 0 is degenerate: all rays initialized to ~same field → σ_k² ≈ 0 → 1/σ_k² → ∞ → gradient explodes). EMA decay 0.99 stabilizes batch noise. Relative floor prevents single-mode dominance (absolute 1e-12 is 6 OOM below typical P_F values ~10⁻⁵-10⁻³ s/km; any mode at floor would dominate loss by ~10⁸). **Honest disclosure**: with truth-side σ_k², candidate (b) is functionally "fixed per-mode reweighting" with EMA smoothing — NOT "adaptive" in the predicted-side sense. The mechanism remains valid (truth-side reweighting amplifies gradient on collapsing modes), but the "adaptive" framing is dropped.

**Why THIS target might rescue variance collapse where L1's τ-MSE-derived target did not (hedged):** the L1 sequence's pathology was variance collapse on the **predicted** P_F band — i.e., `var(P_pred(k))` → 0 across modes. A per-mode normalization by **truth-side** variance amplifies gradient signal on the high-k diffuse-band modes where the model is collapsing; this is mechanism-adjacent to the v3 KILLER-3 reduction-op intuition but operates **on the target structure, not the reduction operator**, so it is not in the L1 closed class.

**Hedge:** "candidate (b) is the first test of a k-space-normalized P_F target in this project; the mechanism prediction is **stronger than (a)'s but weaker than would-be-published precedent** because no IGM ML paper directly trains against this loss form."

### 2. Estimator-equivalence test specification

**K2-equivalent for candidate (b):** new `tests/test_torch_pf_knorm_estimator_equivalence.py` that asserts (i) the per-mode `σ_k²` estimator (`torch_per_mode_variance(batch_flux)`) matches a numpy reference at `rtol=1e-4 / atol=1e-6` over 10 batches; (ii) the full k-normalized loss reduces to the standard `pf_log_mse_loss` in the degenerate case `σ_k² = const ∀ k` (loss-form sanity check); (iii) gradient `∂L/∂flux` is finite and non-NaN under the smallest-allowed `σ_k²` floor — pre-commit (panel-bound 2026-05-23, PROBE absorption): **σ²_floor = 0.01 × median_k(σ_k²_truth)** (relative, NOT absolute 1e-12 which would let any sub-floor mode dominate loss by ~10⁸ given typical P_F values ~10⁻⁵-10⁻³ s/km).

K2 itself (`tests/test_torch_pf_estimator_equivalence.py`) is preserved by construction — `torch_p_flux` is unchanged; only the **downstream loss reduction** is modified.

### 3. Pre-committed falsification criterion

**Same step-200 R-b pattern as (a).** `var_pf_band_ratio < 1e-3` at step 200 closes this candidate as R-b → [D-62] stop-gate. Step 1000 / 5000 / Amendment B criteria same as (a).

**Additional candidate-(b)-specific FAIL trigger (panel-bound 2026-05-23, KILLER-2 absorption — replaces prior arbitrary `w_ratio > 50000` threshold)**: if at step 200, **per-mode gradient-magnitude inflation `max_k(|∂L/∂F|) / median_k(|∂L/∂F|) > 100`** AND `var_pf_band_ratio < 1e-3`, this surfaces as **"normalization-amplification-without-rescue"** — close immediately as R-b WITHOUT waiting for step 1000. The grad-magnitude inflation metric directly indexes the failure-mode the trigger is named for (per-mode gradient blowup driven by σ_k² floor edge cases), unlike the prior `w_ratio > 50000` which was an arbitrary multiple of L1's worst observed range without mechanism derivation.

### 4. Falsified-prior cascade hedged verbs (text-level audit)

- "Candidate (b) is the **first test** of a k-space-normalized P_F supervision-target."
- "Per-mode normalization **may** lift the diffuse-band gradient signal — pending empirical evidence."
- AVOIDED: "k-space normalization is the canonical fix for spectral-loss imbalance" (no published precedent per §0).
- AVOIDED: "structurally addresses the L1-class pathology."

### 5. Compute envelope + smoke spec

Identical compute envelope as (a): ≤30 min Juno A30 at P1-T1 with step-200 instrumentation; CPU 1-step dry-run pre-flight; sbatch derived from `scripts/submit_juno_sprint_L1_L2.sh` with `--pf-knorm-loss` new flag. Pre-committed smoke FAIL routing same as (a) per the [D-61] stop-gate.

---

## Candidate (c) — Hybrid τ + P_F per-task-weighted supervision [STATUS: EXPLORATORY — REJECTED for first-slot dispatch per defense-panel 2026-05-23]

**Panel verdict (binding, 2026-05-23)**: REJECTED for first-slot dispatch. Mechanism prediction is the weakest of the three candidates (§1 self-discloses 1-2 OOM gap to Chen+ 2018 / Sener-Koltun demonstrated regime — the same KILLER-2/SERIOUS-5 caveat the L1 v4→v5 cascade already absorbed and retracted). Implementation wallclock highest. Gradient-independence test as written (§2 "cosine similarity finite and bounded") is vacuously satisfiable. Candidate (c) is **eligible for later-slot dispatch ONLY** if (a) and (b) both R-b retire at step 200 AND defense-panel re-review approves either (i) reframe as exploratory architectural change with 5-decade-weaker mechanism claim than (a)/(b), OR (ii) synthetic-task evidence demonstrating chosen multi-task weighting handles 10³× imbalance before Juno commit. Panel-pre-committed selector if (c) later-dispatched: **Kendall uncertainty-weighting (arXiv:1705.07115)** — strongest mechanism match for "rescue from large per-task scale imbalance" among `{gradnorm-full, kendall-uncertainty, mgda-ub}`. The (c) text below is retained as design-record per [D-37] rule (a) honest-record (not paper-content propagation); it does NOT carry first-slot or auto-fallback status.

### 0. Literature audit + honest framing per [D-37] rule (a)

**Important architectural distinction.** Candidate (c) is **NOT** in the L1 closed class even though it superficially resembles GradNorm-on-two-tasks. The L1 sequence reweighted a single τ-MSE-derived target (where P_F was computed as a downstream function of τ then losses were combined with GradNorm weights `w_tau`, `w_pf`). Candidate (c) instead supervises the model on **τ in τ-space AND P_F as an independent statistic in flux-space**, where the two supervision targets are **architecturally separate quantities** with separate forward passes through the Voigt integrator. The hybrid weighting is then a multi-task formulation in the Chen+2018 GradNorm sense (`arXiv:1711.02257`), Kendall et al. 2018 uncertainty-weighting sense (`arXiv:1705.07115`), or Sener-Koltun 2018 MGDA sense (`arXiv:1810.04650`). Multi-task formulations in cosmology ML literature exist (e.g., Villaescusa-Navarro et al. CAMELS series) but **not specifically for IGM reconstruction with τ + P_F as the two heads** that this researcher located. This is candidate (c)'s positioning: distinct architectural class from L1, with multi-task ML literature support but no direct IGM precedent.

### 1. Mechanism prediction

**Proposed property of the target that produces non-degenerate gradients on the diffuse-bin majority:** two architecturally separate supervision heads provide **independent gradient signals** — the τ head provides local per-voxel gradient (well-studied; gives the diffuse-bin majority a dense gradient surface) while the P_F head provides global per-sightline gradient (penalizes variance collapse explicitly). The hybrid formulation differs from L1 in that L1's `loss_pf` was computed via the τ → flux differentiable path and its gradient back-propagated **through the same τ field** the `loss_tau` term supervised — so the two losses were not gradient-independent in the multi-task-learning sense. In candidate (c), the τ and P_F losses are computed from the same field forward but their backward gradients can be conditioned independently (per-task uncertainty learning Kendall+2018 or MGDA Sener-Koltun gradient deconfliction).

**Why THIS target might rescue variance collapse where L1's τ-MSE-derived target did not (hedged):** the L1 sequence's per-task ratio of ~20000 at sum-reduction (1429 at mean-reduction) reflects the magnitude mismatch when both losses sum over the same τ field; multi-task gradient deconfliction (Sener-Koltun 2018 §3 MGDA-UB) would identify a Pareto-stationary descent direction even at this ratio. **Honest caveat:** Chen+2018 §3.2 / §4 demonstrated regime is ~10× cross-task loss-scale imbalances (Table 1 NYUv2/CityScapes); 1429× and 20000× are 2-3 OOM above demonstrated regime, so multi-task literature does not strongly predict success at this scale either (this is the same caveat the L1 v4→v5 cascade absorbed at panel KILLER-2 / SERIOUS-5).

**Hedge:** "candidate (c) is the first test of an **architecturally-distinct dual-head supervision** in this project; it is NOT a continuation of L1's GradNorm-on-derived-quantities formulation. Mechanism prediction draws on multi-task-learning literature, NOT IGM-specific precedent."

### 2. Estimator-equivalence test specification

**K2-equivalent for candidate (c):** the τ-head equivalence is trivially preserved (τ-MSE is unchanged from baseline). The P_F-head equivalence is K2 itself (`tests/test_torch_pf_estimator_equivalence.py:110-113`) — unchanged from [D-60] gate-4. **What is NEW** is an architectural-faithfulness test: `tests/test_dual_head_gradient_independence.py` asserts that `∂loss_tau/∂model_params` and `∂loss_pf/∂model_params` can be computed independently (no implicit coupling through shared intermediate cached tensors) over 10 batches. PASS criterion: per-batch gradient cosine similarity is finite and bounded — i.e., the gradients are not perfectly aligned (which would indicate the dual-head formulation collapsed back to L1's single-target structure) and not perfectly anti-aligned (which would indicate a numerical-instability bug).

### 3. Pre-committed falsification criterion

**Same step-200 R-b pattern.** `var_pf_band_ratio < 1e-3` at step 200 closes this candidate as R-b → [D-62] stop-gate.

**Additional candidate-(c)-specific FAIL trigger:** if the dual-head formulation reproduces the L1 per-task `w_ratio` pattern at step 200 (i.e., w_ratio ∈ [1000, 25000], matching L1's range), this is **gradient-deconfliction-machinery-without-rescue** — close as R-b regardless of variance-collapse number, since the dual-head architectural distinction failed to manifest in the gradient dynamics.

### 4. Falsified-prior cascade hedged verbs (text-level audit)

- "Candidate (c) is the **first test** of an architecturally-distinct dual-head supervision in this project."
- "The architectural distinction from L1 **may** produce different gradient dynamics — pending step-200 evidence."
- AVOIDED: "candidate (c) inherits multi-task-learning success patterns from Chen+2018 / Kendall+2018 / Sener-Koltun 2018" — the demonstrated regime in those papers is 1-2 OOM below this project's observed loss-scale imbalance, so the inheritance claim is not warranted.
- AVOIDED: "structurally addresses the L1-class pathology."

### 5. Compute envelope + smoke spec

Same as (a) and (b): ≤30 min Juno A30 at P1-T1, step-200 instrumentation, CPU 1-step dry-run pre-flight. Sbatch derived from `scripts/submit_juno_sprint_L1_L2.sh` with `--enable-dual-head-tau-pf` new flag and `--multitask-weighting={gradnorm-full,kendall-uncertainty,mgda-ub}` selector (panel will pick one of the three for first test).

**Compute-envelope caveat per [D-37] rule (a):** candidate (c) is the **most implementation-heavy** of the three (requires dual-head forward refactor in `experiments/nerf/pipeline.py` + new multi-task weighting module) — implementation wallclock dominates compute wallclock for first attempt. Panel should weight this in ranking against (a) and (b) which are loss-module-only changes.

---

## Defense-panel readiness checklist (per design-stage 6 criteria)

| Criterion | (a) Iršič-style PDF + P_F | (b) k-space-normalized P_F | (c) Hybrid τ + P_F dual-head |
|-----------|---------------------------|---------------------------|------------------------------|
| 1. Mechanism prediction | DONE — diffuse-bin via PDF F≈1 bin gradient | DONE — per-mode-variance reweighting | DONE — gradient-independence via dual-head |
| 2. Estimator-equivalence test spec | DONE — new K2-equiv for soft-binning PDF | DONE — K2 preserved + per-mode-σ² equiv test | DONE — K2 preserved + new gradient-independence test |
| 3. Pre-committed falsification | DONE — step-200 var<1e-3 + (a)-specific | DONE — step-200 var<1e-3 + w_ratio>50k trigger | DONE — step-200 var<1e-3 + w_ratio-in-L1-range trigger |
| 4. Falsified-prior cascade verbs | DONE — text-level audited above | DONE — text-level audited above | DONE — text-level audited above |
| 5. Prior-failure ledger | DONE — table at doc top, all 4 attempts | DONE (shared with (a)) | DONE (shared with (a)) |
| 6. Compute envelope + smoke | DONE — derived from L2 sbatch | DONE — derived from L2 sbatch | PARTIAL — implementation wallclock unestimated (ranking input only, NOT dispatch blocker) |

## Recommended panel-review scope

Per [D-37]-ext, panel should focus attacks on:

1. **Mechanism-prediction weakness across all three candidates.** None has a published IGM-NeRF precedent. (a) leans on cosmology measurement-paper convention; (b) extrapolates from emulator log-k-MSE; (c) leans on general multi-task ML where the demonstrated regime is 1-2 OOM below this project's observed loss-scale imbalance.
2. **Ranking justification.** This doc deliberately does NOT pre-rank — panel should rank based on (i) mechanism-prediction strength, (ii) implementation complexity, (iii) [D-62] stop-gate cost (first candidate dispatched bears the close-on-R-b risk; subsequent candidates only run on panel re-review).
3. **First-candidate selection.** [D-61] stop-gate means the first dispatched candidate is the only one that runs without panel re-review. Panel should pre-commit which candidate is dispatched first based on highest mechanism-prediction confidence × lowest implementation wallclock.
4. **Soft-binning convention for candidate (a)** — PDF differentiability requires a kernel choice (Gaussian / triangular / sigmoid-sum); panel should specify the kernel + bandwidth before dispatch, since the soft-binning approximation directly affects the K2-equivalent test tolerance.
5. **σ_k² estimator choice for candidate (b)** — batch-sample-variance vs. analytical-theory-prediction vs. fixed-per-mode-prior. Panel pre-commits before dispatch.
6. **Multi-task weighting choice for candidate (c)** — `{gradnorm-full, kendall-uncertainty, mgda-ub}`. Panel pre-commits one for first attempt; the other two are NOT auto-fallbacks (would re-open within-[D-53] iteration that [D-61] stop-gate forbids).

**Binding-decision discipline (per PI re-review 2026-05-23, R15-cascade lockdown)**: Panel's pre-commitment of (i) first-candidate selection (item 3), (ii) kernel choice for (a) (item 4), (iii) σ_k² estimator for (b) (item 5), and (iv) multi-task weighting for (c) (item 6) are **BINDING**. PI does NOT re-open these decisions post-panel verdict; the only re-open mechanism is empirical (panel re-review on step-200 result of the first-dispatched candidate). This locks down against post-panel PI drift per R15-cascade discipline.

## Stop-gate criteria (panel-bound clarification 2026-05-23)

The uniform falsification criterion `var_pf_band_ratio < 1e-3 at step 200` is a **necessary** indicator but NOT sufficient. Panel-bound calibration for [D-37] honest-reporting:

- **Step 200**: PASS is **PROVISIONAL** — necessary condition, indicates the candidate broke from the L1 cluster's [6.9e-7, 2.93e-6] range, but does not establish rescue. Continue training.
- **Step 1000**: PASS is the **BINDING rescue verdict** — `var_pf_band_ratio > 0.05` at step 1000 establishes that the candidate has materially rescued the variance collapse (~4-5 OOM above L1 cluster + maintained past the warmup-zone diagnostic threshold per [D-60] v3 KILLER-1 analog).
- **Step 5000**: PASS is **generalization confirmation** — `var_pf_band_ratio > 0.3` at step 5000 establishes training-dynamics survival (analog of [D-60] gate-pilot 5k-survival bar from v3 §3 pre-commit).

**Disclosure for marginal outcomes per [D-37] rule (a)**: if a candidate's `var_pf_band_ratio > 1e-3` at step 200 but reverts to L1-range by step 1000, the candidate is reported as **"step-200 PROVISIONAL PASS, step-1000 BINDING FAIL"** — NOT as "ambiguous" or "needs more data." The step-1000 result is dispositive.

---

## Honest-framing notes per [D-37] rule (a)

- **All three candidates have thin published-IGM-precedent support.** (a)'s namesake (Iršič+2017) is a measurement paper not a loss paper; (b) has no direct IGM-NeRF precedent — closest is Cabayol-García+2023's log-k-MSE emulator loss; (c) leans on general multi-task ML but at a loss-scale imbalance 1-2 OOM above demonstrated multi-task regime. **No candidate is "the published fix" for the observed L1 pathology** — this is genuinely a class of first-tests.
- The [D-61] stop-gate at first-candidate-R-b is conservative-appropriate: given the thin literature support, defending a "[D-53] also exhausted" close on a single attempt requires panel pre-commitment, NOT a 4-attempt within-class sweep that would risk the [D-37]-ext rule 1 anti-degeneracy concern.
- **What is NOT in this design doc** (per [D-61] scope): architectural pivots (different model class — convolutional discriminator, transformer, etc.); different reconstruction targets outside the flux-domain entirely (e.g., direct density-field supervision); inference-machine reformulations à la Maitra/LyαNNA 2023. These are downstream gates if the [D-62] stop-gate fires.
- **What this doc CANNOT predict** per [D-61] / [D-37] honest-reporting: whether any of the three rescues the variance collapse. The 6 design-stage criteria certify the candidates are **ready for compute commitment**, not that they are likely to succeed. Panel re-review is the gate, and the empirical step-200 instrumentation is the final word.

---

*Carry-forward on panel APPROVE of one or more candidates: dispatch core-implementer to prototype the panel-selected first candidate as smoke harness on P1-T1 with same step-200 instrumentation as [D-60]. On panel HOLD: PI re-absorbs design, no Juno spend. On panel REJECT of all three: escalate immediately to architectural pivot per [D-62] stop-gate logic ([D-53] supervision-target-redesign class itself closed pre-empirically by panel verdict).*

# Project Diagnosis v2 — A5 Re-issue (close-out paper spine)

**Status**: branch-independent close-out diagnosis. Authored 2026-06-15 (owner, per [D-73]
amendment-1 §F item 6 / amendment-3 §J / amendment-4 §A5). Every quantitative claim is pinned
to a LEDGER decision or artifact path. This is the spine the latex-author turns into
`papers/shared/` atoms; the working title is the A3-compliant
**"Trainability and optimization-failure characterization of neural implicit IGM reconstruction
from Lyman-α sightlines at z = 0.3."**

The one (1d′)-gated number is the explicit-voxel-grid result (§5 slot, §6 branch dispositions).
Everything else here is final.

---

## 1. The corrected intervention record (A5 fixes)

The reconstruction target is the [D-13] gate set at fiducial P1, z = 0.3, n_rays = 1024:
|ΔP_F/P_F| < 10% over k_∥ ∈ [10^−2.5, 10^−1.5] s/km; ξ_ρ̂,ρ(r = 2 h⁻¹Mpc) > 0.6; KS(F-PDF) < 0.05.

The binding gate is P_F. It has not closed under any tested intervention. The honest enumeration
(replacing the loose "~14"): **12 interventions across four axes, plus 2 diagnostics.**

| # | Intervention | Axis | Outcome | Pin |
|---|---|---|---|---|
| 1 | sat-aware P_F band loss | loss (integrated) | FAIL, constant-prediction collapse (D1) | [D-40] |
| 2 | FGPA-tail per-pixel regularizer | loss (per-pixel) | FAIL, constant-prediction collapse (D2) | [D-41] |
| 3 | velocity-gradient conditioning | architecture-input | FAIL at smoke, density-head collapse (D3) | [D-42] |
| 4 | physics_id embedding joint training | data (pooled) | FAIL at smoke, trivial-flux collapse (D4) | [D-46] |
| 5–9 | L1/L2 P_F-MSE levers (lr×2, per-task-clip, reduction op, k-norm target) | loss | all retire to the var-collapse basin | [D-63]/[D-65] |
| 10 | fGPA-residual density prior | architecture/supervision-target | FALSIFIED at R_feas (120× under boundary) | [D-69] |
| 11 | direct ρ-MSE pretraining (γ) | supervision-target | FALSIFIED; active variance shrinkage **in 1 of 3 lr cells** (the other 2 UNKNOWN_NON_MONOTONIC — caveat per [D-70] am-2 K1) | [D-69] am-7 |
| 12 | skip-rich MLP body | architecture | FALSIFIED, n=10 Juno, all 10 seeds negative | [D-71] |
| A | linear log-ρ head (Softplus removed) | parameterization (output head) | COLLAPSE at probe scope | [D-73] am-2 |
| — | microbatch-size, physics-variant | (diagnostic, not intervention — [D-62] addendum) | — | [D-63] |

Two diagnostics (microbatch-size, physics-variant) are excluded from the intervention count per
the [D-62] addendum ruling; counting them as interventions was the over-extension A5 corrects.

**Pinned caveat (A5, load-bearing):** the (γ) "active variance shrinkage 1.6e-5 → 2.3e-6" claim is
the lr = 1e-4 single cell only; the 3-cell lr probe was 1-of-3 FAIL_SINKING, 2-of-3
UNKNOWN_NON_MONOTONIC ([D-70] am-2 K1; artifact `experiments/nerf/artifacts/d69_lr_probe/summary.json`).
Every citation of that result carries the 1-of-3 caveat.

---

## 2. Two failure modes, kept rigorously separate

The evidence describes two distinct phenomena. Welding them overstates the finding; the inference
boundary is binding ([D-73] am-2 §2, am-3).

**Mode A — the production flux-supervised model.** It trains. It passes KS in 3 of 4 physics cells
and mean-flux in 4 of 4 at single seed (2 of 4 under 5-seed bootstrap, [D-44]). It fails P_F by
36–42% in all 4 cells — 3.6–4.2× over the 10% bar ([D-39]). The mechanism is positively identified:
saturation-band under-fitting, cross-physics-consistent (R ∈ [18.87, 23.75], 4/4 cells > 12× the
floor). The healthy run's predicted-P_F band variance sits at ≈ 1.0 (matched to truth) from step
5000 to 50000 (A7, `experiments/nerf/artifacts/d73_a7_control/a7a_var_pf_control.json`). Mode A is a
**partial under-fit of a model that otherwise converges** — a quantitative deceleration, not a total
collapse (Rung-3: log-space gradients balanced R_log1p = 0.91 vs linear-flux R_linear = 0.18, a 5.5×
attenuation in saturation).

**Mode B — probe-scale collapse under direct supervision.** Under direct log-ρ-MSE supervision at
n_grid = 64, the field converges to the constant-mean basin: zero predicted variance, no structure.
The A1 head probe ([D-73] am-2) is the cleanest instance: removing the Softplus output head and
supervising the raw log-density directly produces the same collapse, head-invariantly — final
Var(ρ_θ)/Var(ρ_truth) = 2.5–6.6×10⁻⁶ (≈ 10⁴–10⁵× below the 0.1 escape bar), median |Pearson r| < 0.05
across all 9 cells, the 6 Softplus control cells statistically indistinguishable. The output-head
nonlinearity is acquitted as the cause; the collapse is upstream of the parameterization map, in the
optimization/loss landscape.

**Inference boundary:** A1/Mode-B does not test Mode A. It establishes head-invariance of the
Mode-B collapse at probe scope (n_grid = 64, direct log-ρ-MSE, P1, z = 0.3, 1000 steps). It provides
no direct evidence on whether Mode A's saturation deficit shares a root cause. The two are reported
in separate paragraphs.

---

## 3. The honest gaps (the integrity backbone)

These are stated plainly because they are load-bearing for the paper's credibility, and because
[D-37] requires the honest framing over the strengthening one.

**(a) The 3D ξ gate was never evaluated on the production MLP.** The [D-13] headline ξ gate is the
3D FFT-shell estimator (`src/analysis/cross_corr.py:compute_xi_pearson`). The production MLP is
supervised on 1D flux sightlines and produces no 3D ρ cube, so its ξ-of-record is the 1D-along-ray
r_ρ^log surrogate ([D-58]): median +0.029 on the pub-t1 P1-N64 checkpoint, +0.077 on a separate T3
checkpoint. [D-58] itself showed this surrogate *decouples* from ξ_3D(r = 2 Mpc/h) under
scale-dependent degradation. **Therefore the paper may not claim the MLP "fails all three gates."**
The correct statement: the MLP fails two of three directly-evaluated [D-13] gates (P_F, KS); the
third (3D ξ) was never evaluated on the MLP in its [D-13]-defined form (am-3 §H).

**(b) The healthy-run control is partial.** The 7-lever collapsed-basin retirements were read at
step 200; the A7 healthy-run control is available only from step 5000 (no step-200 checkpoint). So a
step-200 read of ≈ 10⁻⁶ is consistent with genuine collapse but not, by this control alone, exclusive
of a not-yet-trained transient. Every 7-lever citation carries this two-part caveat (am-3 §G).

**(c) The classical Wiener number is a self-anchored lower bound, not a floor.** The idealized
best-case Wiener reconstruction (after four corrections — unit-variance tracer, noise_rel = 1e-3,
74 px/ray CG, extended L-sweep) reaches ξ_3D(2 Mpc/h) ≥ **0.079** at L = 3, P1, z = 0.3
(`experiments/nerf/artifacts/wiener_baseline/a4prime_wiener.json`). It was still rising at L = 3
(the CPU RAM wall), so the optimum is un-pinned — the number is a **lower bound**, never a turned-over
optimum. The per-fix ladder (0.051 → 0.020 → 0.037 → 0.079) confirms the original 0.051 was partly an
over-regularization artifact (am-4 demote vindicated). It is reported self-anchored (R14): published
CLAMATO/TARDIS r-values are context only, never our bar. Two real low-bias contributors are disclosed:
z = 0.3 mean_F = 0.979 (genuinely low per-pixel S/N) and a real-space-pixel / redshift-space-flux
frame mismatch.

---

## 4. Scope: why z = 0.3, and what this does not claim

The fiducial is P1, z = 0.3. At z = 0.3 the mean transmitted flux is ⟨F⟩ ≈ 0.979 — roughly 2%
absorption — so the forest carries little per-pixel signal and saturated pixels are ≈ 0.11% of the
data (149/131072, Rung-3). This is the low-redshift, information-sparse regime (HST/COS territory),
not the z ≈ 2–3 regime where ground-based Lyα tomography operates.

**This characterization is scope-locked and does NOT claim neural IGM tomography is impossible.**
Published work succeeds at z ≈ 2–3, where the forest is information-rich: CLAMATO (Lee+ 2018) and
TARDIS (Horowitz+ 2019) reconstruct 3D density from sparse skewers; a 2024 MNRAS neural method
recovers optical-depth-weighted fields on Sherwood-family data at 4 ≤ z ≤ 5. The claim is narrower
and specific: **at z = 0.3, this method class hits a characterized wall, and the wall's location is
explained by the information content of the low-z forest, not by a deficiency of the architecture.**

---

## 5. The three-way ξ comparison

The explicit-grid result is the only neural 3D-ξ number the project can produce (the MLP cannot;
§3a). The comparison framework (am-3 §C(c)):

| Method | ξ_3D(2 h⁻¹Mpc) | Note |
|---|---|---|
| Classical Wiener (idealized) | ≥ 0.079 (lower bound, optimum un-pinned) | self-anchored; z=0.3 low-S/N + RSD-frame low-bias |
| Neural explicit voxel-grid (1d′) | **TBD** | the only neural 3D-ξ; G=192, plain-[D-24], one-lever |
| Production MLP | N/A (3D) — 1D surrogate ≈ +0.03/+0.08 | 1D-flux-supervised, no 3D cube |

The 0.6 bar provenance: estimator from Stark+ 2015; the 0.6 acceptance threshold is a project-side
adoption ([D-36] — the Stark attribution was retracted), not a Stark-quoted value. Both halves carry
on every citation.

---

## 6. The two pre-written (1d′) branch dispositions

The (1d′) explicit voxel-grid (four free grids matching the MLP's four free heads, G = 192, plain
[D-24] flux supervision, one-lever proven tol-0 at am-7) is the most expressive parameterization the
problem can be given. Its result determines which disposition lands. Both are valid [D-37]-rule-7 end
states; neither is spun.

**Branch A — (1d′) collapses** (trainability var_pf_band_ratio ≤ ~8.8×10⁻⁵ at step 5000, OR
PASS-trainability-but-P_F-fails). **Disposition: the characterization is complete.** Headline
(scope-locked, R8/R9): *at z = 0.3, P1, the P_F reconstruction deficit is invariant across the tested
representation classes — continuous coordinate-MLP, skip-rich MLP, and explicit free-per-voxel grid —
under flux supervision; the obstacle is the supervision regime and the information content of the
low-z forest, not the neural representation.* This is a successful result: a method class's failure
mode, named, localized (saturation-band + void-floor pinch), and bounded in scope. It is the honest
close. The paper's contribution is the characterization itself plus the falsification-disciplined
methodology; no over-reach to "impossible," only "characterized at this regime."

**Branch B — (1d′) escapes collapse and closes P_F** (var_pf > 1e-3 at step 5000 AND |ΔP_F/P_F| < 10%
at convergence). **Disposition: the project reopens.** The re-issue becomes a methods-paper-plus-
positive-result: an explicit field, forward-modeled through the differentiable Voigt integrator,
recovers the flux-relevant structure the implicit MLP could not — scope-locked to (192³, P1, z = 0.3),
NOT "the method class is solved." This triggers: the [D-74] JAX-at-v2-boundary evaluation (a scaling
build-out becomes the context where JAX's vmap/jit/memory wins and cheap higher-order autodiff pay
for a rewrite); and the conditional Rung-4 angled-sightline reopen ([D-73] §C). The true 3D ξ from
this run fills the §5 slot and anchors the three-way comparison with a real neural number.

---

## 7. What is banked regardless of branch

Independent of the (1d′) outcome, the project has produced: a validated differentiable RSD-convolved
Voigt forward model (Stage 2a, [D-57]-audited after the factor-12.9 damping-coefficient fix); a
positive mechanistic identification of the reconstruction failure (saturation-band deceleration +
void-floor collapse, the Rung-3 two-space quantification); substrate-scale information-budget findings
(4-class Sherwood feedback is 2-scalar-discriminable at 32³ ρ-crops, [D-51]; CNN-discriminable with a
24.3 pp margin at 48³, [D-56]); a corrected idealized classical Wiener baseline (§3c); and a
falsification-disciplined methodology demonstrated end-to-end across 12 interventions retired at
pre-committed stop conditions. These are valid scientific end-states under the 2026-06-01
documentation-pillars directive (decision-quality, not outcome-quality).

---

**Cover note (what is gated vs ready):**
- READY NOW (branch-independent): §1–§5, §7, and both §6 dispositions are final and pinned.
- (1d′)-GATED: only the §5 neural-3D-ξ cell and the §6 branch selection (A vs B) await the queued
  Juno run. The moment var_pf@5000 + the [D-13] gates land, the write-up is one number and one
  branch-selection away from a complete draft.

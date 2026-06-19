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
frame mismatch. **The frame mismatch is now quantified ([D-73] am-9 S7) and applies to BOTH the Wiener
and the neural (1d′) ξ numbers:** ξ scores a real-space density cube against real-space truth while
supervision/observation is redshift-space flux, with no RSD remap on either cube; at z = 0.3, P1 the
displacement is Δχ_rms = 1.30 h⁻¹Mpc, Δχ_p95 = 2.53 h⁻¹Mpc — comparable to / exceeding the r = 2 h⁻¹Mpc
gate point, so it depresses ξ(r = 2) independent of reconstruction quality. This is a named confound on
every ξ(r ≈ 2) number in §5, not a property of any one method.

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
| Neural explicit voxel-grid (1d′) | **0.0075** (≈ 25% of the 0.0298 achievable ceiling) | job 221335, G=192, plain-[D-24], one-lever; weak structure |
| Production MLP | N/A (3D) — 1D surrogate ≈ +0.03/+0.08 | 1D-flux-supervised, no 3D cube |

**The 0.6 acceptance bar is DEMOTED ([D-73] am-9 / [D-36] correction note, 2026-06-18).** Two
independent defects, both confirmed first-hand: **(S5) the threshold is unreachable under the
implemented estimator.** `compute_xi_pearson` normalizes by each field's *global* std and returns the
cross-*correlation function* ξ(r), which decays with r — NOT the per-lag Pearson coefficient the 0.6
bar assumes (≡1 for truth-vs-truth at every r). Measured: ξ(truth, truth) at r≈2 = **0.0298**, not ≈1.0,
so no field — not even a perfect reconstruction — can pass 0.6. The honest read is therefore
**ceiling-relative**: the grid recovers 0.0075/0.0298 ≈ 25% of the field's own achievable r≈2
autocorrelation, and sits *below* truth+100%-noise (0.0211) — genuinely weak structure recovery, but
the "0.0075 vs 0.6 → catastrophic fail" framing is INVALID. **(S7) frame mismatch** confounds *both* the
neural and Wiener numbers: ξ compares a real-space cube to real-space truth while supervision is
redshift-space flux, no RSD remap; at z=0.3 the displacement Δχ_rms = 1.30, Δχ_p95 = 2.53 h⁻¹Mpc is
≳ the r=2 gate scale, depressing ξ(r=2) independent of reconstruction quality. Provenance (unchanged):
the estimator is a *Stark-style FFT cross-power* (not "from Stark+2015"); the 0.6 threshold is a
project-side adoption ([D-36], Stark attribution retracted). **The ξ number is SUPPORTING, not
load-bearing — the decisive close-out evidence is K2 (degeneracy, §6), which is estimator-independent.**
A corrected per-lag coefficient ξ_cross/√(ξ_pp·ξ_tt) is banked as journal-length / future work (needs
the Juno grid cube and remains S7-confounded).

---

## 6. The two pre-written (1d′) branch dispositions

The (1d′) explicit voxel-grid (four free grids matching the MLP's four free heads, G = 192, plain
[D-24] flux supervision, one-lever proven tol-0 at am-7) is the maximal-capacity-within-this-study
parameterization (most degrees of freedom / least regularization; conditioning — e.g. hash-grid /
multigrid — is a separate untested axis). Its result determines which disposition lands. Both are valid
[D-37]-rule-7 end states; neither is spun.

**Branch A — SELECTED (job 221335, 2026-06-18, [D-73] am-9).** The trigger is a disjunction
(trainability var_pf_band_ratio ≤ ~8.8×10⁻⁵ at step 5000, OR PASS-trainability-but-P_F-fails). The run
fired the **PASS-trainability disjunct**: var_pf_band_ratio = 1.0959, flat step 5000→50000 — the grid
trained (escaped the Mode-B collapse, ~1000× over the bar) and then failed 3D-structure recovery.
*Honest disclosure:* the grid's `|ΔP_F/P_F|` was not separately tabulated; var_pf IS the
predicted-P_F-band-variance statistic (the §2 A7 observable), so the trainability clause is satisfied
directly and the "P_F-fails" arm is discharged by the absence of P_F-relevant structure recovery, not a
direct P_F measurement.

**Disposition: the characterization is complete.** The decisive, estimator-independent evidence is
**K2**: the TRUE sightline fields, forward-modeled through the SAME integrator under the exact [D-24]
loss (RSD applied identically, free tau_amp), score loss_data = 0.0101; the grid scored 0.0026 — it fits
the observed flux ~4× *better than the true field does*. (Non-trivial content: the 4× margin +
tau_amp-flatness, which rules out the [D-10]/[D-11]/[D-34] amplitude-calibration escape; the margin
includes integrator-induced slack — the 0.0101 floor is our FGPA-vs-RT error, not nature's.) Regime
note: var_pf = 1.0959 ≈ Mode-A's ≈1.0, so the grid sits in the **Mode-A basin** (saturation-band
partial-underfit), reproducing it at the most-expressive parameterization — *this* is the
representation-invariance content, scoped to P_F (the cross-representation claim belongs to the §1
intervention table, not to this single (1d′) run).

Headline (scope-locked, R8/R9; supersedes the prior draft): *At z = 0.3 (P1), a
maximal-capacity-within-this-study free-per-voxel field (192³), trained under the production flux
supervision, fits the observed flux better than the true field does through the same forward model
(0.0026 vs 0.0101, ~4×) while recovering only weak 3D structure (≈25% of the achievable ξ ceiling,
with real-vs-redshift-space frame mismatch a named additional confound). The z = 0.3 flux inverse
problem, under this FGPA forward model, is under-constrained: escaping the optimization collapse is
necessary but not sufficient, because the flux through this integrator does not determine the 3D field.
We do not separate problem-intrinsic from forward-model-induced under-determination at ⟨F⟩ = 0.979, and
make no claim beyond z = 0.3 — where the low-z forest is information-sparse (≈2% absorption), in contrast
to z ≈ 2–3 where CLAMATO/TARDIS succeed.* This is a successful result: a method class's failure mode,
named, localized, measured (K2 degeneracy), and bounded in scope. It is the honest close — the
contribution is the characterization plus the falsification-disciplined methodology; no over-reach to
"impossible," only "characterized at this regime."

**Branch B — NOT SELECTED (foreclosed by job 221335).** (var_pf > 1e-3 at step 5000 AND |ΔP_F/P_F| < 10%
at convergence.) Branch B required the explicit field to *recover the flux-relevant structure the MLP
could not*; the K2 degeneracy + weak ξ (§9c) show it did not (the grid fits flux *better* than truth yet
carries the wrong structure). The [D-74] JAX-at-v2 adoption and the Rung-4 angled-sightline reopen below
are therefore NOT triggered. **(Original Branch-B disposition, retained for the record:) the project reopens.** The re-issue becomes a methods-paper-plus-
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

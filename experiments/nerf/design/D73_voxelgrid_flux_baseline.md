# [D-73] §F item 2 — (1d′) Explicit Voxel-Grid + Flux-Supervision Design Doc

**Status**: PROVISIONAL (R15) — design-doc-of-record pending the single permitted
panel cycle (§6 K6 narrow-discharge gate). No dispatch until that cycle returns
APPROVE and infra-manager confirms Juno reachability (§5 dispatch-time gate).

**Provenance**: [D-73] §B (K5-retract → (1d′) replacement), §F item 2 (plan-of-record),
amendment-2 §3 (A1 COLLAPSE → (1d′) still runs under flux supervision, orthogonal to A1).
Review provenance (R6): PI-authored; defense-panel pre-review owed (one cycle, §6).

**One-line scope**: swap the coordinate-MLP ρ representation for an explicit
log10-ρ voxel grid, forward-model to flux through the **identical** [D-24] regime
and RSD-Voigt integrator as the production MLP, score on the [D-13] gates plus a
pre-trainability gate. This is a **parameterization-axis** test under Mode-A flux
supervision (R13), NOT a supervision-change and NOT Mode-B remediation.

---

## 1. The closure fork — RESOLVED: option (b)

**Decision**: (b) explicit log10-ρ voxel grid for the **density field only**;
T and X_HI tied to ρ via the fluctuating Gunn-Peterson approximation (FGPA);
v_pec NOT a free grid.

**Free fields**: ρ (grid). **Closed fields**: T = T0·(ρ/⟨ρ⟩)^(γ−1); X_HI from the
FGPA τ∝ρ^β scaling already embedded in the production forward model; v_pec recovered
from ρ via the same FGPA/RSD relation the production integrator consumes (linear-theory
peculiar velocity from the density field), defaulting to v_pec = 0 if that path is not
already wired in the production MLP route. v_pec is in either case a **fixed function
of ρ**, introducing zero free DOF.

**Rationale (stated, hedged)**:
- (b) is the stronger TARDIS-analog: T and X_HI tied to ρ by FGPA mirrors the
  classical Horowitz+2019 reconstruction assumption set; the test doubles as the
  missing TARDIS-analog baseline datum ([D-73] §B, A4-adjacent).
- (b) isolates exactly one degree of freedom (R10 one-lever): a PASS/FAIL is
  attributable to the ρ representation, not to a re-opened multi-field degeneracy.
  Option (a) (four free grids, 12 free DOF/voxel) reintroduces the
  constant-T / structured-ρ / structured-X_HI degeneracy the [D-42]/[D-47] arc
  mapped; a PASS under (a) could not be attributed to the ρ axis alone.
- The FGPA closure constants (T0, γ, β) are taken **unchanged from the production
  MLP forward model** — they are not re-fit here. This is load-bearing for one-lever
  cleanliness (see §3, §6).

**Pre-committed degeneracy audit (R3 / [D-37]-ext rule 3)**: under flux supervision
with the [D-24] saturated-absorber mask, `loss_data` is weakly informative on the
saturation-band voxels (the masked + capped bins). The question "what does this loss
leave unconstrained when loss_data is weakly informative on the diffuse-bin majority?"
is answered as: the explicit grid has per-voxel free ρ, so under flux supervision the
saturation-band voxels (masked) receive **no direct gradient** and the diffuse-forest
voxels receive gradient only through the unsaturated τ∈(0,10] bins. This is precisely
the regime the trainability gate (§4 (i)) is designed to probe — a grid that collapses
to constant-mean ρ under this gradient sparsity is the FAIL-trainability outcome, and
it is a pre-committed valid end-state, not a failure of the experiment.

---

## 2. Grid parameterization

- **Field**: a single dense tensor `log_rho_grid` of shape (G, G, G), G = 128 default.
  ≈ 2.10M free parameters (128³ = 2,097,152). Comparable in parameter count to the
  production MLP body; trivially fits A30 memory.
- **Resolution justification**: 128³ is coarse vs the production 768³ ρ-field. This is
  deliberate and pre-committed as a scope-lock: the test asks whether an explicit field
  can represent the flux-relevant structure *at all*, not whether it matches 768³.
  The [D-24] sightline geometry samples 16384 rays × n_bins along one axis; the
  flux-relevant longitudinal structure is band-limited to the [D-13] inertial range
  k_∥ ∈ [10^−2.5, 10^−1.5] s/km, which 128 voxels along the line of sight resolves
  with margin (the inertial band upper edge corresponds to ≳ a few voxels per mode at
  128³ over the 60 Mpc/h box). A trainability PASS at 128³ that fails P_F is NOT
  attributable to under-resolution at this band; a re-bake at higher G is the single
  permitted re-bake (§5) only if the panel flags a resolution confound at gate-construction.
- **Interpolation**: trilinear (`grid_sample`-equivalent or manual trilinear),
  autograd-compatible, no detached numpy in the ray-sampling path (CLAUDE.md contract).
  Ray sample points (the production integrator's quadrature nodes along each sightline)
  index into the grid by their normalized [0,1]³ box coordinates — identical coordinate
  normalization to the MLP input ([D-08] convention).
- **Output transform**: each voxel stores log10(ρ/⟨ρ⟩ + 1e-3) directly (same DENSITY_LOG_EPS
  convention as nerf.py:209 linear-log head). Conversion to linear ρ for the integrator is
  ρ_θ = clamp(10^v − 1e-3, min=0), the same `density_log_to_linear` map. Per A1
  (amendment-2): the output-map is NOT the obstacle — log10-per-voxel is chosen for
  parity with the production density representation and dynamic-range stability, not as
  an intervention.
- **Initialization**: voxels initialized to the **truth-field mean** log10(⟨ρ⟩/⟨ρ⟩ + 1e-3)
  = log10(1 + 1e-3) ≈ 4.3e-4 (i.e. ρ/⟨ρ⟩ ≈ 1 everywhere), plus small Gaussian noise
  (σ = 0.01 in log-space) to break symmetry. This is the constant-mean basin A1 collapsed
  into — initializing there makes the trainability gate a clean test of whether flux
  gradients can drive the grid OUT of constant-mean, which is the whole question.
  Initialization choice is logged as a pre-committed knob, not tuned.

---

## 3. Supervision + loss — exactly the [D-24] flux regime (one lever)

This is a **parameterization swap, NOT a supervision change**. Every loss-side element
is byte-for-byte the production MLP path:

- **Target**: redshift-space τ from `tauH1_2048_n16384_z0.300.dat` (first-half = RS per
  [D-06]/[D-24]), z = 0.3, P1.
- **Forward model**: the production RSD-convolved Voigt integrator (Stage 2a, re-validated),
  consuming the 4-field stack [ρ_θ (from grid), T(ρ), X_HI(ρ), v_pec(ρ)] exactly as the
  MLP route feeds [density, temp, h1_frac, vpec] to it. The integrator code path is
  UNCHANGED; only the producer of the field stack changes.
- **Saturated-absorber mask**: [D-24] item (1), τ_GT > 1e5 cores expanded to the τ_GT > 10
  connected component, identical mask, applied to data-loss AND the [D-11]/[D-21] two-pass
  mean-F reduction.
- **Forest cap**: τ_max = 10 ([D-24] item (2), LOCKED).
- **Loss form**: log1p MSE, L_data = ⟨(log(1+τ_pred^eff) − log(1+τ_GT^eff))²⟩_non-saturated
  ([D-24] item (3)), plus the [D-11] mean-flux anchor (λ_F = 1.0) at the [D-34]-corrected
  ⟨F⟩_obs(z=0.3) = 0.979 ± 0.005 — same anchor the production publication route uses.
- **The single lever (R10/R13)**: grid-vs-MLP. Nothing else moves. The FGPA closure
  constants (T0, γ, β), the integrator, the mask, the cap, the anchor, the loss form,
  the optimizer schedule are all inherited unchanged.
- **EXPLICITLY OUT OF SCOPE ([D-73] §D)**: the Rung-3 linear-flux-stratum-reweighted /
  saturation-band-reweighted loss. Folding it in would put two levers in one experiment
  (R10/R13 violation). Bounded-close caps new loss-engineering at zero. NOT included.

---

## 4. Pre-committed gate ladder

All thresholds pre-committed here ([D-37] symmetric-disclosure). Both gates reported
regardless of the other's outcome.

### (i) Trainability gate (primary, pre-science)
- **Observable**: `var_pf_band_ratio` = Var_k(P_F^pred) / Var_k(P_F^truth) over the
  [D-13] inertial band, computed by the EXACT production instrumentation at
  `pipeline.py:3047-3060` (r-averaged P_F over k ∈ [K_MIN_INERTIAL, K_MAX_INERTIAL],
  band-masked, float64 variance ratio). Definition-match to production is a panel-check
  item (§6).
- **Threshold**: `var_pf_band_ratio > 1e-3` at step 5000.
- **Margin rationale**: the [D-65]/[D-63] collapsed-basin ceiling is 2.93e-6 (job 202109,
  L2 reduction sum→mean). 1e-3 is ≥ 300× over that ceiling — a grid that clears it has
  produced flux-power structure the collapsed MLP basins never reached. **Healthy-run
  anchor (A7, 2026-06-10)**: the production pub-t1 run sits at var_pf_band_ratio ≈ 1.0
  at steps 5000–50000 (`d73_a7_control/a7a_var_pf_control.json`) — so the 1e-3 bar sits
  ~3 orders below the healthy/trained signature and ~300× above the collapsed basin, a
  defensible mid-band threshold. CAVEAT (strengthened per [D-73] amendment-3 §G — PARTIAL
  SERIOUS-2 discharge): the healthy-run control is available only from step 5000 onward (no
  step-200/1000 checkpoint), so a matched step-200 control does not exist. This means the
  SERIOUS-2 attack ("collapsed-basin retirements read at step 200 may be not-yet-trained, not
  genuine collapse") is only PARTIALLY discharged: the healthy run is confirmed at ≈1.0 by
  step 5000 (5+ orders above the ~1e-6 retirements), but a step-200 read of ~1e-6 is not
  directly excluded as a transient. The (1d′) trainability gate is read at step 5000 —
  INSIDE the confirmed-healthy window — so this caveat does NOT weaken the (1d′) gate itself;
  it bounds only what the prior 7-lever step-200 retirements can claim.
- **PASS / FAIL / MARGINAL**:
  - PASS: > 1e-3 at step 5000.
  - MARGINAL: ∈ [~8.8e-5, 1e-3) — above the collapsed ceiling by a clear margin but
    below the trainability bar; reported as "rises off the collapsed basin but does not
    reach trainability," logged, does not gate-PASS.
  - FAIL: ≤ ~8.8e-5 (within ~30× of the collapsed-basin ceiling) — the explicit grid
    ALSO collapses under flux supervision.

### (ii) Full [D-13] science gates (reported regardless of (i))
- (a) |ΔP_F/P_F| < 10% averaged over k_∥ ∈ [10^−2.5, 10^−1.5] s/km (Hann window, normalized δ_F).
- (b) ξ_ρ̂,ρ(r = 2 h⁻¹ Mpc) > 0.6, **3D FFT-shell estimator** (`src/analysis/cross_corr.py:compute_xi_pearson`)
  — the explicit grid IS a 3D ρ cube, so unlike the production MLP (1D-surface-supervised,
  no 3D cube; A7 could only report the 1D r_ρ^log surrogate) the (1d′) run CAN be scored on
  the true [D-13] 3D ξ. **[D-36] provenance disclosure** mandatory on every citation: the
  0.6 bar is a project-side adoption, NOT a Stark+2015-quoted value. **A4 anchor (2026-06-10)**:
  the idealized noiseless Wiener baseline reaches ξ_3D(2 Mpc/h) ≈ 0.05 on this geometry
  (`wiener_baseline/a4_wiener_baseline.json`) — the classical external bar; the (1d′) ξ is
  reported against both the 0.6 gate and the 0.05 Wiener anchor.
- (c) KS-distance on flux PDF < 0.05 over F ∈ [0.05, 0.95].

### Pre-committed outcome semantics (all three valid, publishable — decision-quality, rule 7)
1. **PASS trainability + close P_F (gate (a))** → the explicit field BREAKS the saturation
   deficit. Big result: reopens the project (conditional Rung-4 reopen per [D-73] §C).
   Verb stays at "the explicit-field parameterization breaks the deficit at (128³, P1, z=0.3,
   flux-supervised)" — scope-locked, NOT "the method class is solved."
2. **PASS trainability + P_F still fails** → the deficit is **supervision-coupled, not
   parameterization-bound**. Strengthens the close: even a free per-voxel field, given the
   flux gradient, cannot reproduce inertial-range P_F. This is the [D-39] saturation-band
   deficit attributed to the supervision regime, not the representation.
3. **FAIL trainability** → the explicit field ALSO collapses under flux supervision →
   strongest possible "supervision regime, not architecture" close: the collapse is
   representation-invariant under the [D-24] flux gradient.

None of the three is a process failure. The grading criterion is decision-quality at the
fork, not outcome-shape ([D-37]-ext rule 7).

---

## 5. Compute envelope

- **Steps**: 5000 to the trainability gate (i); if PASS, continue to the production-class
  schedule for the [D-13] gates (gate (ii) eval at the schedule's converged checkpoint).
  Step budget for the full [D-13] eval inherits the production tier-1 schedule (50k steps,
  [D-14]) capped by the wall-clock ceiling below.
- **Ray budget**: production sightline geometry — 16384 rays available; microbatch 1024
  with gradient accumulation ([D-14] memory plan) on A30 (24 GB). The grid forward is
  cheaper than the MLP forward (trilinear lookup vs 8-layer MLP), so per-step cost is
  bounded by the integrator, identical to production.
- **Expected Juno A30 wall-clock**: trainability gate (5000 steps) well under a few hours;
  full [D-13] schedule bounded by the hard cap.
- **HARD CAP ([D-73] §F item 7)**: 30 A30-hr total, **1 job + 1 re-bake**. The re-bake is
  reserved for a panel-flagged construction defect (e.g. a resolution confound forcing
  higher G, or a mask/closure-parity fix) — NOT for hyperparameter tuning (tuning would
  violate R8 cascade-close formality / the bounded-close zero-new-engineering cap).

**DISPATCH-TIME GATE (not a design-time blocker)**: Juno SSH/VPN reachability is
**UNVERIFIED** on the new machine (2026-06-10 machine change). infra-manager MUST confirm
Juno reachability per the `juno-hpc` skill BEFORE dispatch. This is flagged here so the
panel does not treat it as a design defect; it is an operational pre-flight item that
gates dispatch, not the design.

---

## 6. What the single permitted panel cycle must check (doubles as [D-71] §D/§F K6 narrow-discharge gate)

ONE cycle only (R-bank FREEZE / §F item 7 cap — one gate-construction panel cycle per
sub-30-A30-hr experiment). The cycle's object is the **gate construction**, not a readout.
Required checks:

1. **FGPA-closure choice (the load-bearing item)**: T, X_HI, v_pec are derived from ρ by
   the **identical** FGPA/RSD relations the production MLP path consumes — same T0, γ, β,
   same RSD treatment, same mask. No grid-specific second closure. This is what makes the
   one-lever attribution valid; if it fails, the whole comparison is confounded.
2. **One-lever cleanliness (R10)**: grid-vs-MLP is the ONLY changed axis; loss form, mask,
   cap, anchor, integrator, optimizer schedule all inherited unchanged; the Rung-3
   reweighted loss is confirmed ABSENT ([D-73] §D).
3. **Gate-ladder construction**: the 1e-3 / 300×-margin trainability threshold and the
   MARGINAL band; the three pre-committed outcome semantics; symmetric disclosure ([D-37]).
4. **var_pf_band_ratio definition-match**: the grid run's trainability observable is the
   EXACT `pipeline.py:3047-3060` instrumentation (r-averaged band P_F variance ratio,
   float64, [K_MIN_INERTIAL, K_MAX_INERTIAL] band) — no grid-specific re-definition.
   (R29-substance unit-chain check: the gate observable is a dimensionless variance ratio,
   truth-side denominator, same frame as the collapsed-basin ceiling AND the A7 healthy-run
   anchor (≈1.0) it is compared to — the 2.93e-6, 1e-3, and 1.0 are in the SAME units. The
   K6 unit-chain obligation is discharged by this same-frame construction, not by an external
   observational anchor, so no FGPA forward-model unit-transfer is crossed.)
5. **[D-36] ξ-provenance disclosure**: the 0.6 bar citation carries the project-side-adoption
   disclosure (not Stark-quoted); the A4 Wiener ξ ≈ 0.05 anchor carries its idealization
   caveats (noiseless, Gaussian prior, global-gain-fit).
6. **Re-bake scope**: the 1 re-bake is for a panel-flagged construction defect only, not tuning.

K6 narrow-discharge linkage: per [D-73] §B, this design-doc panel cycle IS the [D-71]
§D/§F K6 narrow-discharge gate. The K6 obligation discharges with an APPROVE on items 1-6;
a NEEDS-WORK returns the construction defect to the PI (no second cycle — the defect is
fixed and the re-bake budget absorbs at most one construction fix).

---

## 7. R-rule audit

- **R8/R9 (narrow-scope verbs)**: this is a parameterization-axis test, COLLAPSE / MARGINAL /
  PASS **at production scale (128³, P1, z=0.3, flux-supervised)** — NOT a method-class verdict
  in either direction. No outcome of this test falsifies or vindicates "the NeRF method class."
  Every citation carries the (128³, P1, z=0.3) scope-lock.
- **R10 (one-lever)**: grid-vs-MLP is the single lever. FGPA closure constants and all
  loss-side elements inherited unchanged. Rung-3 reweighted loss excluded ([D-73] §D). §6
  item 2 is the panel check on this.
- **R13 (scope re-verb)**: (1d′) is a Mode-A flux-supervised parameterization-axis test,
  NOT Mode-B remediation (A1 amendment-2 §3 confirms orthogonality: A1 tested the output-head
  axis under Mode-B direct-MSE; this tests the representation axis under Mode-A flux supervision).
- **R15 (PROVISIONAL by default)**: this design doc is PROVISIONAL pending the single panel
  cycle (§6). The PI sign-off on the gate construction is provisional until panel APPROVE OR
  an explicit PI-only deferred-panel annotation in §7 history. Downstream dispatches
  (core-implementer land, infra-manager Juno dispatch) inherit provisional status until lifted.
- **R29-substance (unit-chain)**: discharged internally — the trainability gate observable
  (var_pf_band_ratio), the collapsed-basin ceiling (2.93e-6), and the A7 healthy-run anchor
  (≈1.0) are in the SAME dimensionless truth-denominator frame; the 1e-3 bar is a
  project-internal multiple (≥300×) of that ceiling, NOT an external observational anchor
  crossing a nonlinear forward model. No FGPA/exp(−τ) unit-transfer is crossed by the gate
  threshold. (§6 item 4.)
- **R30 (grep-discipline)**: this session re-read nerf.py:204-218 (4-field output stack +
  density_log_to_linear), pipeline.py:3047-3073 (var_pf_band_ratio instrumentation),
  [D-24]/[D-13]/[D-39]/[D-65] LEDGER blocks, [D-73] §B/§D/§E + amendment-2.
- **R-bank FREEZE ([D-73] §F item 7)**: no new R-rules proposed by this doc. Sighting logs
  and hard auto-triggers stay live.
- **K6 narrow-discharge linkage**: §6 — the single panel cycle on this doc doubles as the
  [D-71] §D/§F K6 narrow-discharge gate ([D-73] §B).

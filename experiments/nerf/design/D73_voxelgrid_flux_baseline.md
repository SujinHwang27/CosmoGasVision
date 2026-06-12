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

## 1. The closure fork — RE-RESOLVED: option (a) (was option (b); corrected per (1d′)-construction panel KILLER-1, 2026-06-11)

**Decision**: (a) explicit log10 voxel grids for **all four fields** — ρ, T, X_HI,
v_pec — one independent grid per field, matching the production MLP's four free
output heads one-for-one.

**Free fields**: ρ (grid), T (grid), X_HI (grid), v_pec (grid). **Closed fields**: none.

**Rationale (stated, hedged) — corrected one-lever argument**:
- The production forward path is **four free fields**, not one. Verified this session at
  `nerf.py:204-218`: the MLP emits density, temp = softplus·1e4+1e3, h1_frac = sigmoid,
  vpec = tanh·500 as four independent output heads, consumed directly by
  `volume_render_physics` (`nerf.py:303-307`). There is **no FGPA closure in the
  production forward path**. The only FGPA-tail in the project ([D-41]) is a retired,
  smoke-FAILed, default-OFF soft regularizer whose z~3 Hui-Gnedin β=1.6/γ=−0.7 the
  project flags as mis-specified at z=0.3 (LEDGER S3 §587).
- Therefore the **one-lever match to production is option (a)**: four free grids vs four
  free MLP heads. Option (b) (ρ free, T/X_HI/v_pec FGPA-closed) would change **two** axes
  at once — the representation AND the field-DOF count (4-free → 1-free+closure) — so a
  (b) PASS/FAIL could not be attributed to the representation axis alone. Option (b)
  additionally imports the z~3 FGPA closure the project suspects is wrong at z=0.3,
  confounding every (b) gate verdict with a closure defect.
- **Anti-(a) argument STRUCK (was inverted)**: the prior text claimed option (a)
  "reintroduces the constant-T / structured-ρ / structured-X_HI degeneracy." That
  degeneracy is exactly what the production 4-free-head MLP **already carries**; matching
  it is *required* for one-lever attribution, not a defect introduced by (a). The [D-42]/
  [D-47] arc mapped that degeneracy as a property of the production representation — the
  (1d′) test must inherit it, not eliminate it, to isolate the representation axis.
- The **TARDIS-analog / FGPA-closed datum is a SEPARATE deliverable**, reported as an
  explicit two-axis result (representation swap + closure imposition), NEVER as the (1d′)
  one-lever test. It is out of scope for this bounded close-out (no new engineering budget;
  [D-73] §F item 7).

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

- **Field**: four independent dense tensors — `log_rho_grid`, `temp_grid`, `xhi_grid`,
  `vpec_grid`, each shape (G, G, G), G = 128 default. ≈ 4 × 2.10M = 8.39M free
  parameters total (4 × 128³). Comparable in parameter count to the production MLP body;
  trivially fits A30 memory. Each grid's output transform mirrors the corresponding
  production MLP head: log10(ρ/⟨ρ⟩ + 1e-3) for density (per §2 output-transform below);
  softplus·1e4+1e3 for T; sigmoid for X_HI; tanh·500 for v_pec — applied to the
  trilinearly-interpolated raw grid value, byte-for-byte the `nerf.py:214-216` head maps.
- **SERIOUS-2 RSD-parity (resolved by option (a))**: the free `vpec_grid` matches the
  production MLP's free vpec head feeding the RSD line-shift at `nerf.py:328`
  (v_source = vel_axis + vpec). The prior option-(b) v_pec=0 (or FGPA-derived v_pec)
  would have broken RSD-parity with the production path; the free v_pec grid restores it
  with zero closure assumption.
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
- **MANDATORY pre-dispatch voxels-per-shortest-mode calc (SERIOUS-4)**: before dispatch,
  compute the z=0.3 box-velocity-length → shortest-resolved-mode → voxels-per-mode at G=128
  over the 60 Mpc/h box for the [D-13] inertial-band upper edge k_∥ = 10^−1.5 s/km. If
  **voxels-per-shortest-mode ≲ 4**, START at G ≥ 192³ (not 128³) to preserve the single
  permitted re-bake (§5) for a genuine construction defect rather than spending it on an
  under-resolution confound the calc could have caught at design time. Record the calc
  (Hubble flow at z=0.3, box H(z), Nyquist) in the dispatch brief. A G≥192³ start triggered
  by this calc is NOT a re-bake — it is the correct initial resolution.
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
- **Margin rationale (re-anchored per (1d′)-construction panel SERIOUS-3)**: the bar is
  anchored PRIMARILY to the **A7 healthy production run** (`d73_a7_control/a7a_var_pf_control.json`),
  which sits at var_pf_band_ratio ≈ 1.0 at steps 5000–50000 — the frame-comparable,
  same-instrumentation, same-architecture-class trained signature. The 1e-3 bar sits ~3
  orders below the healthy/trained signature: a grid clearing it has produced flux-power
  structure within 3 decades of a healthy trained run. The collapsed-basin ceiling
  (2.93e-6, job 202109) is retained as a **SECONDARY sanity floor only** — the bar sits
  ≥300× above it — with the explicit caveat that 2.93e-6 is from a **different
  architecture + reduction** (L2 reduction sum→mean) and is therefore a cross-arch sanity
  reference, not the primary anchor. Anchoring primarily to A7 (frame-comparable) rather
  than to the cross-arch collapsed ceiling closes the SERIOUS-3 mismatch. CAVEAT (strengthened per [D-73] amendment-3 §G — PARTIAL
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
  the true [D-13] 3D ξ. **[D-36] / PROBE-6 provenance disclosure** mandatory on every ξ
  citation: the ξ estimator is from Stark+2015; the 0.6 threshold is a **project-side
  adoption per [D-36]**, NOT a Stark+2015-quoted value. Both halves (estimator-source AND
  threshold-provenance) carry on every citation. **A4 anchor — WITHDRAWN pending re-run
  (A4′), per A4-scrutiny panel 2026-06-11**: the prior ξ_3D(2 Mpc/h) ≈ 0.05 "best-case
  Wiener" anchor is **DEMOTED — not citable as an information floor**. As-run it was
  over-regularized (data tracer not unit-variance standardized → ~2.5× over-regularization)
  AND ran at noise_rel=0.05 (50× the claimed 1e-3 idealization; provenance contradiction),
  AND used 11 px/ray (LOS spacing coarser than the L=2 Mpc/h correlation length), AND its
  L-sweep was monotone to the boundary (optimum outside the window). A4′ re-run
  (support-researcher, CPU) is authorized to earn a real classical number with the panel
  fixes; until it lands, the (1d′) ξ is reported against the 0.6 [D-36]-provenance gate
  ONLY, with NO classical anchor. When A4′ lands, its self-anchored "validated best-case
  Wiener at z=0.3" value (R14 — never an external CLAMATO/TARDIS bar) is the anchor, with
  the z=0.3 / density-FFT-shell scope caveat attached.
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

1. **Four-free-field parity (the load-bearing item, re-resolved option (a))**: the four
   grids (ρ, T, X_HI, v_pec) feed the integrator through the **identical** output-head
   maps and the identical RSD/mask/integrator path the production 4-head MLP uses — same
   softplus/sigmoid/tanh transforms (`nerf.py:214-216`), same v_source RSD shift
   (`nerf.py:328`), same mask. There is NO FGPA closure on either side (the production
   path has none; verified `nerf.py:204-218`). This four-free-field parity is what makes
   the one-lever (grid-vs-MLP) attribution valid; if a grid-specific closure or transform
   crept in, the comparison would be confounded.
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
   disclosure (not Stark-quoted, estimator from Stark+2015); the A4 Wiener ξ anchor is
   WITHDRAWN pending the A4′ re-run (was over-regularized; see §4(ii)(b)) — until A4′ lands
   the (1d′) ξ carries NO classical anchor, only the 0.6 [D-36] gate.
6. **Re-bake scope**: the 1 re-bake is for a panel-flagged construction defect only, not tuning.

K6 narrow-discharge linkage: per [D-73] §B, this design-doc panel cycle IS the [D-71]
§D/§F K6 narrow-discharge gate. **K6 DISCHARGED 2026-06-11**: the single permitted cycle
returned NEEDS-WORK (KILLER-1, option-(b) two-lever), the PI applied the panel-authorized
mechanical fix (option (a), four free grids — edits (i)–(viii), [D-73] amendment-4) inside
the re-bake budget WITHOUT consuming a second cycle, and items 1-6 now hold under option (a).
The construction defect is fixed; no second cycle. The §6 item-1 FGPA-closure check is now
N/A (no closure — four free grids); item 2 one-lever cleanliness holds (grid-vs-MLP, four
free fields each side); items 3-6 unchanged.

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

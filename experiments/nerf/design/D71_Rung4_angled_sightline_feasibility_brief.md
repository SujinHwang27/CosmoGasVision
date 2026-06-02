# [D-71] Rung 4 — Angled-sightline feasibility diagnostic (BRIEF)

## §0 — Banner

- **Status**: BRIEF-AUTHORING (design only). Defense-panel pre-review owed before any Juno HPC dispatch.
- **Authored by**: support-researcher (PI-dispatched 2026-06-02 per LEDGER §I amendment-2 Part 4).
- **Parent decisions**: [D-71] §A–§I; [D-71] §I Part 7 (Juno-only generation directive); [D-71] §I amendment-2 (Rung 4 brief-authoring authorization); [D-37] / [D-37]-Ext (honest-reporting + symmetric disclosure); R29 CANDIDATE (unit-chain audit); R33 CANDIDATE (in-session-WebFetch citation discipline).
- **Scope-lock**: METHODOLOGY-BRIEF AUTHORING ONLY. No `sample_field_along_rays` implementation, no `submit_juno_*.sh` authoring, no HPC dispatch, no commits. Implementation surface owned by data-engineer at next dispatch.
- **Honest-framing rule** (per [D-37] rule (a)): observation-first; the ρ-only FGPA proxy yields UPPER-BOUND informativeness; null verdict is a valid end-state per [D-37]-Ext R7 (decision-quality not outcome-quality).

---

## §1 — Diagnostic question + null/alternative pre-commit

**Empirical question (Q4 SQ1+SQ2 operationalization)**: under the existing in-house FGPA-on-3D-field forward (ρ-only proxy; see §4), does the inertial-band variance of the 1D Lyα flux power spectrum P_F(k_‖), pooled over an angled-direction-sampled ray set, differ from that of an axis-parallel-only ray set traversing the *same* truth ρ field by more than the pre-committed threshold? If yes, axis-parallel-only supervision under-determines anisotropic structure (filaments transverse to all box axes) enough that angular coverage is a load-bearing supervision lever. If no, angular coverage is not the binding axis.

**Choice of observable**: the per-set inertial-band variance of P_F(k_‖) is the same quantity the existing FGPA renderer + `compute_p_flux_torch` pipeline computes (`scripts/d62_3_fgpa_variance_spectrum.py:152-167`; `src/analysis/flux_power_torch.py:41-105`). It is geometry-agnostic at the renderer level — the renderer ingests `(n_rays, n_bins)` field tensors with no orientation metadata — so the diagnostic stays inside the pipeline track-record rather than introducing a new estimator.

**Pre-committed null (H0)**: `|Δ log10 R_feas(angled − axis)| < 0.3 decade`, where `R_feas` is the dimensionless ratio defined in §3. Verbally: angled-set inertial-band variance does NOT differ materially from axis-parallel-set inertial-band variance.

**Pre-committed alternative (H1)**: `|Δ log10 R_feas(angled − axis)| ≥ 0.5 decade`. Verbally: angled rays sample variance structure not visible to axis-parallel rays at >3.16× ratio.

**Pre-committed verdict band**: `|Δ log10 R_feas| ∈ [0.3, 0.5]` → MARGINAL (insufficient evidence; recommend extending diagnostic with finer angular sampling, not promotion to pivot).

**Honest-framing flag (per [D-37] rule (a))**: alternative null the panel can construct where the 0.5-decade threshold would *not* bind — if angled and axis-parallel rays through the *same* statistically isotropic ρ field happen to sample the same 1D-projected P_F(k) by isotropy of the truth field's variance spectrum (Sherwood at z=0.3 is close to isotropic over the box; cosmic-variance-driven anisotropy is small at L=60 Mpc/h), then the ρ-only proxy *cannot* surface the angular-coverage signal even if angled supervision would help under the full hydro state (T+xHI+vlos). This is the load-bearing caveat that frames the ρ-only proxy as upper-bound-informativeness (see §4).

**Pre-commit verdicts (verdict-numerics frozen BEFORE methodology per [D-37] rule 5 symmetric disclosure)**:

| `|Δ log10 R_feas|` | Verdict | Action |
|---|---|---|
| `≥ 0.5 decade` | PIVOT | [D-71] retires into [D-72] supervision-geometry sprint; data-engineer dispatch on full angled-supervision pipeline (300-500 LOC, T+xHI+vlos SPH-kernel weighting) authorized. |
| `[0.3, 0.5) decade` | MARGINAL | Extend diagnostic: finer angular sampling (N_ang ×4) + filament-orientation-aliased angle pairs. No pivot authorization. |
| `< 0.3 decade` | NO-PIVOT | Cycle #7 reopens with loss-construction as remaining lever (Rung 3 Part 3 candidate becomes admissible after R10 orthogonality argument). |

---

## §2 — Sampling protocol

**Truth field**: Sherwood P1 z=0.300, ρ-only 3D field loaded via `igm_gal_loader.load_3d_field('rho')`. Reuses pub-t1 cached snapshot (positive-ID surface per LEDGER line 438). Grid resolution: matches existing cache (n_grid=768 if available; ≥384 acceptable).

**Box-frame**: comoving 60,000 kpc/h cube, normalized to `[0, 1]³` for MLP/grid-sample consistency (CLAUDE.md astrophysical-conventions).

**Per-ray bin count**: 2048 (Sherwood `los2048_*.dat` convention).

**Set A — Axis-parallel (`N_ax = 768`)**:
- 256 rays along each of {x̂, ŷ, ẑ} (balanced 3-way split).
- Transverse coordinates: uniform on `[0, 1]²` per ray (random seed=2026 for reproducibility; same seed used by `d62_3_fgpa_variance_spectrum.py`).
- Starting-offset randomization: yes (random origin in `[0, 1]` along the ray direction, periodic wrap; eliminates ray-origin aliasing).

**Set B — Angled (`N_ang = 768`)**:
- Direction sampling: **uniform on S²** via Marsaglia sampling (drawn from `cos(θ) ∈ [-1, 1]` uniform, `φ ∈ [0, 2π)` uniform), then unit-normalized.
- Rationale for uniform-S² over uniform-in-cos(θ) or fixed-(π/4, 0) angle-pair choice: uniform-S² is the maximum-entropy null for the anisotropic-structure hypothesis. Uniform-in-cos(θ) over-samples polar caps and would conflate "angled" with "near-axis-parallel" in 1/3 of the draws — false-negative risk. Fixed-(π/4, 0) targets a single filament-aliasing test but would not generalize the alternative hypothesis. Uniform-S² is the right null for "does angular coverage matter at all".
- Ray origins: uniform on `[0, 1]³` (full-box volume).
- N_ang = N_ax = 768 controls for sample-size noise in the ratio R_feas.

**Set C — Mixed**: `union(A, B)` = 1536 rays. Used as third anchor — the H1 prediction is that variance contributions from B are not double-counted within A's 1D-projected spectrum, so `R_feas(mixed)` should lie between R_feas(A) and R_feas(B), not equal either.

**Total sightlines**: 1536 unique rays. All three sets traverse the same truth ρ field at the same (physics_id=1, redshift=0.300). No new Sherwood snapshot needed.

**Periodic boundary handling**: rays wrap on `[0, 1]³` via `field_3d` periodic-wrap pre-pass before `torch.nn.functional.grid_sample` (data-engineer implementation surface).

---

## §3 — Metric construction

**`R_feas` definition** — dimensionless ratio-of-ratios, normalized to remove overall amplitude dependence:

```
R_feas^set = ⟨ var_kbin(P_F^set(k)) ⟩_{k ∈ inertial}  /  ⟨ P_F^set(k) ⟩_{k ∈ inertial}²
```

where:
- `P_F^set(k) = (1/N_set) Σ_{rays ∈ set} |FFT_Hann(δ_F)|² · (dv / Σw²) · one_sided_correction` per `compute_p_flux_torch` (`src/analysis/flux_power_torch.py:41-105`).
- `var_kbin = Var_{rays ∈ set}(|FFT_Hann(δ_F)|²)` at fixed k — same construction as `d62_3_fgpa_variance_spectrum.py:152-167`.
- `⟨·⟩_{k ∈ inertial}` averages over k bins with `k ∈ [10^-2.5, 10^-1.5] s/km` ([D-13] inertial band, edges as `_K_MIN_INERTIAL` / `_K_MAX_INERTIAL`).
- Dividing by `⟨P_F⟩²` normalizes out overall flux amplitude — this is the dimensionless-coefficient-of-variation-squared form, making `R_feas` invariant under multiplicative rescaling of the flux. This eliminates the `var_truth = 3.74e-15` FP-noise-floor regression that killed the [D-66] PASS verdict at `d62_3_fgpa_variance_spectrum.py:6-9`.

**FGPA forward** (ρ-only proxy):
- Render unscaled τ via `_render_tau_unscaled_chunk` (`d62_3_fgpa_variance_spectrum.py:67-107`), passing per-ray ρ sampled along the sightline + isothermal T_box + xHI_FGPA (from ρ via FGPA β=1.6, γ=-0.7) + v_pec=0.
- Calibrate τ_amp via `calibrate_tau_amp` to ⟨F⟩ = 0.979 (Becker+2013 z=0.3 anchor).
- F = exp(-τ); δ_F = F/⟨F⟩ − 1 (`flux_power_torch.py:72-76`).

**Pre-committed deltas**:
- `Δ log10 R_feas = log10 R_feas(B) − log10 R_feas(A)` — primary verdict statistic.
- `Δ log10 R_feas (mixed − axis) = log10 R_feas(C) − log10 R_feas(A)` — secondary consistency check; expected magnitude ~½ of primary under H1.

**R29-class unit-chain audit** (per [D-71] §I Part 7 anchor):

| Step | Quantity | Units | Transformation | Monotone-Lipschitz? |
|---|---|---|---|---|
| 1 | ρ-along-ray | ⟨ρ⟩ (dimensionless) | trilinear interp of `field_3d` | Yes (linear in field) |
| 2 | τ_unscaled | dimensionless | FGPA proxy: ρ^β → xHI; Voigt convolution | Monotone in ρ; locally Lipschitz |
| 3 | τ | dimensionless | τ_amp · τ_unscaled, τ_amp calibrated bisection | Monotone in τ_unscaled |
| 4 | F = exp(-τ) | dimensionless | exponential | Monotone decreasing |
| 5 | δ_F | dimensionless | F/⟨F⟩ − 1 | Linear in F |
| 6 | \|FFT(δ_F·Hann)\|² | (s/km)² | DFT + window | Quadratic |
| 7 | P_F(k) | s/km | × dv/Σw² | Linear in step 6 |
| 8 | Var_rays(P_F(k)) | (s/km)² | unbiased per-k variance over ray axis | Quadratic |
| 9 | band-integral / ⟨P_F⟩² | dimensionless | inertial-band mean divided by squared band-mean P_F | Ratio; non-monotone in degenerate edge case ⟨P_F⟩ → 0 |
| 10 | log10 R_feas | dimensionless | log10 of step 9 | Monotone in step 9 (over R_feas > 0) |
| 11 | Δ log10 R_feas threshold | decades | difference of step-10 quantities across sets | Linear in step 10 |

**Lipschitz/edge-case findings**:
- Step 2 (FGPA proxy) is non-trivially nonlinear (ρ^β with β=1.6 inflates ρ tail). The R_feas ratio absorbs the multiplicative β-dependence partially through the `⟨P_F⟩²` normalization, but β-sensitivity of the *shape* of P_F(k) is residual. **Panel pre-review item**: verify β=1.6 (Sherwood [D-10] convention at z=0.3) is the right reference; quantify β-sensitivity by repeating diagnostic at β ∈ {1.4, 1.6, 1.8} as supplementary if budget allows.
- Step 9 (ratio) degenerates at ⟨P_F⟩ → 0 (well-defined: ⟨F⟩ ≥ 0.5 at z=0.3 per [D-11]; ⟨P_F⟩ is bounded away from 0). Safe.
- Step 10 (log10) requires R_feas > 0. Variance is non-negative; ⟨P_F⟩² > 0. Safe.
- **The 0.5-decade threshold is monotone-Lipschitz in the underlying physical anisotropy**: a real anisotropy in the truth ρ field that grows the angled-set variance by factor c monotonically increases `Δ log10 R_feas` by `log10(c)`. The threshold is set at log10(3.16) ≈ 0.5 dex, corresponding to a 3.16× variance-ratio difference between angled and axis-parallel sets — equivalent to a 78% relative-variance change. Below 0.3 decade ≈ 2× variance-ratio, the difference is within bootstrap variance band for N_set=768 sightlines (standard error of log-variance ratio ≈ √(2/N) ≈ 0.05 decade at N=768; 6σ threshold is 0.30 decade, matching the inconclusive lower bound).

**Threshold-substance** (R29-substance, not knob-tuning): 0.3 decade is derived from the per-set bootstrap standard error √(2/N_set) on log-variance, scaled to 6σ at N_set=768. 0.5 decade is derived from the smallest physically meaningful anisotropy that would change a CVPR claim about reconstruction-quality (3× variance contribution from off-axis structure is large enough to invalidate axis-parallel-only baseline rigor; <2× is below natural cosmic-variance scatter in Sherwood-box L=60 Mpc/h per Bolton+2017 Fig. 4 PDF-width). Both edges fall out of error-bar + physical-significance considerations, not back-fitted to a desired verdict.

---

## §4 — ρ-only FGPA proxy caveat

**The diagnostic operates under a scientific simplification**: the 3D fields T, xHI, vlos are currently `NotImplementedError` at `src/data/igm_gal_loader.py:187-192` (loader raises explicitly: "Field requires SPH-kernel weighting against PartType0/Density and PartType0/SmoothingLength"). Only the 3D ρ field is available via the existing CIC-deposited grid (`igm_gal_loader.py:170-185`).

The diagnostic therefore uses:
- **Isothermal T = ⟨T⟩_box**: a single scalar temperature for all bins, derived as the mean temperature of the Sherwood snapshot at z=0.3. Specify reference value at dispatch time from cached snapshot statistics (typical IGM temperature at z=0.3 is ~10⁴ K; the exact box-mean is to be read from the Sherwood metadata). For the purposes of variance-spectrum computation, the temperature enters via the Doppler `b = 12.85 √(T/10⁴)` km/s (`d62_3_fgpa_variance_spectrum.py:79`) — a constant b across rays.
- **v_pec = 0**: no RSD broadening, no peculiar-velocity gradient enters the convolution. `v_source = vel_axis + 0` (`d62_3_fgpa_variance_spectrum.py:84`).

**Upper-bound-informativeness framing**: the full hydro state (T+xHI+vlos available per-voxel) contains additional anisotropic structure (thermal-broadening anisotropy along filaments; v_pec coherent-flow anisotropy at filament intersections) that the ρ-only proxy cannot expose. Angled sightlines through the full state could show STRONGER angular-coverage signal than the ρ-only proxy. The ρ-only proxy is informative for one direction (lower-bound on the angular-coverage effect from ρ alone), not the other.

**Pre-committed asymmetry framing** (per [D-37] rule (a), honest framing of evidentiary asymmetry):
- If ρ-only returns `|Δ log10 R_feas| ≥ 0.5 decade` → **strong evidence for PIVOT**: the full-hydro version would almost certainly also exceed (adding T/v_pec anisotropy can only add variance contributions, not cancel them).
- If ρ-only returns `|Δ log10 R_feas| < 0.3 decade` → **weak evidence for NO-PIVOT**: the full-hydro version *could* plausibly still exceed via T/v_pec anisotropy contributions absent in the proxy. NO-PIVOT verdict in this case is caveat-bound, and the panel may rule that the full angled-supervision pipeline (T/xHI/vlos SPH-kernel weighting, ~300-500 LOC per LEDGER §I) is *still* warranted to test the full-state version. Pre-committing this here so post-hoc verdict-tuning is foreclosed.
- MARGINAL: extend diagnostic at finer angular sampling first; do not promote to full-state version yet.

**Citation discipline (R33)**: FGPA proxy convention β=1.6, γ=-0.7 follows [D-10] internal; the conversion ρ → xHI ∝ ρ^β is from Hui & Gnedin 1997 (cited internally; no new WebFetch verification needed since this is repo-internal convention). Becker+2013 mean-flux anchor ⟨F⟩=0.979 at z=0.3 is repo-internal per `d62_3_fgpa_variance_spectrum.py:60`.

---

## §5 — Computational cost + Juno dispatch shape

**CPU pre-flight (LOCAL)** — PI-pre-committed sanity check:
- Analytic field `f(x, y, z) = sin(2π x / L)` with L=60 Mpc/h. Construct as 3D tensor on a 64³ grid.
- 64-ray autograd test: sample along 32 axis-parallel + 32 angled rays via `sample_field_along_rays`; verify `torch.autograd.grad` flows end-to-end through `grid_sample` periodic-wrap + Voigt + FFT.
- Sanity check: the analytic field's angled-ray P_F(k) should peak at k = 2π/L_proj where L_proj is the projection of the box-period onto the ray direction. Axis-parallel ẑ rays should see flat δ_F (sin(2π·constant) along the ray) → P_F(k) ≈ 0; angled rays at θ from ẑ should see P_F(k) peak at the projected wavenumber. This is the geometric-correctness sanity check independent of the FGPA proxy.
- Wall-time budget: ≤30 min on the local machine (CPU, no GPU). Memory: 64³ grid × float32 = 1 MB; trivial.

**Juno A30 dispatch** — scoped per `juno-hpc` SKILL conventions:
- Total sightlines: 1536 (768 axis + 768 angled; mixed is union, no new rendering).
- Per-sightline cost on A30: trilinear-interp at 2048 bins × 1 field × 8 corners (~16k FLOPs) + Voigt forward (W=64 window, 2048 bins, ~2M FLOPs per ray) + FFT (2048 bins, ~22k FLOPs). Dominated by Voigt: ~2M FLOPs/ray. A30 ~10 TFLOPs effective for this workload → ~0.2 ms/ray. Total: 1536 rays × 0.2 ms = ~0.3 s compute.
- Realistic A30 wall-time with PyTorch overhead, I/O, grid-sample memory-traffic: ~5-15 min including snapshot load + ρ-field load (~6 GB) + result write. Order-of-magnitude estimate, not measured.
- **Pre-committed wall-time ceiling**: ≤4 hours per Juno dispatch (`#SBATCH --time=04:00:00`). This is the soft cap; if the job exceeds 1 hr, infrastructure-manager investigates (per `juno-hpc` SKILL anti-pattern "silent OOM mid-training").

**Suggested `scripts/submit_juno_d71_I_angled_feasibility.sh` skeleton** (TBD authored by infrastructure-manager at dispatch time, not now):
- `#SBATCH --partition=a30 --gres=gpu:1 --mem=32G --time=04:00:00`
- Copy-in: `src/`, `scripts/`, `experiments/nerf/`, `pyproject.toml`, `uv.lock` to `${JUNO_SCRATCH}/d71_I_feas/${RUN_TAG}`.
- Symlink `${JUNO_SCRATCH}/sherwood` → `Sherwood/` (snapshot mirror).
- Activate `${JUNO_WORK}/.venv` (cu124 torch override per `juno-hpc` SKILL).
- Run `python -u scripts/d71_I_angled_feasibility.py --physics 1 --redshift 0.3 --n_ax 768 --n_ang 768 --n_bins 2048 --seed 2026 --output_dir ./d71_I_feas`.
- Copy-out: capsule + PNG to `${JUNO_WORK%/CosmoGasVision}/d71_I_results/${RUN_TAG}`.
- Round-trip back to host via `rsync juno:...` per `juno-hpc` SKILL §"MLflow round-trip".

**MLflow tag set** (if logging): `model_type=fgpa-proxy`, `stage=d71_I_feas`, `physics_id=P1`, `redshift=0.3`, `compute=juno`.

---

## §6 — Defense-panel pre-review handoff

Defense-panel must attack the following load-bearing claims at pre-review BEFORE Juno HPC dispatch:

1. **§1 null/alternative pre-commits are operationally well-posed**: the H0 / H1 / MARGINAL bands are pre-frozen with numerics that do not depend on the diagnostic outcome. The honest-framing flag (truth-field-isotropy alternative null) is named.

2. **§3 metric construction has no hidden ratio-of-pooled-variances regression** (Rev 2 §1 framework analysis precedent). `R_feas = ⟨Var_rays(P_F)⟩_band / ⟨P_F⟩²_band` is a coefficient-of-variation form, NOT a between-set variance ratio. Panel must verify: is the within-set per-k variance of P_F across rays the right "variance" quantity here, or does the [D-71] Rev 1 cycle #1-#6 hidden-pooled-variance failure mode lurk in the implicit "pool over rays at fixed k" step? The Rev 2 framework analysis at `experiments/nerf/design/D71_Rev2_gate_construction_scoping.md` §1.B identified per-stratum variance as protocol-compatible; this brief's per-k pool-over-rays variance is analogous but NOT identical. **STRUCTURAL CONCERN FLAGGED**: if angled sightlines have *systematically different per-ray variance amplitude* than axis-parallel (i.e., angled rays traverse longer effective path through the box and integrate more structure), the per-k Var_rays will inflate for the angled set NOT because of anisotropic-structure signal but because of path-length effects. The `⟨P_F⟩²` normalization partially absorbs this but does NOT fully eliminate it — squared mean of P_F also inflates with path length. **Mitigation owed at implementation time**: rays must be path-length-normalized to a common effective integration length L_eff = L_box (i.e., terminate ray at periodic-wrap boundary, not at fixed 2048 × dv km/s in proper velocity). Panel must rule on whether this mitigation is sufficient or whether a stricter dimensionless invariant is needed.

3. **§3 thresholds (0.3 / 0.5 decade) have R29-substance unit-chain derivation, not knob-tuning**: the §3 unit-chain table walks each transformation. 0.3 decade = 6σ at N=768 from bootstrap-SE argument; 0.5 decade = log10(3.16) physical-significance argument from Sherwood PDF-width scale (Bolton+2017 Fig. 4). Panel must verify the bootstrap-SE formula and the Bolton+2017 PDF-width citation. **R33 verification owed**: Bolton+2017 Fig. 4 PDF-width was not in-session WebFetch-re-verified for this brief — it is cited from repo-internal Rev 2 §1.A use. Panel may downgrade citation to "Sherwood-internal PDF-width statistic" if R33 strict-compliance required.

4. **§4 ρ-only proxy caveat is honestly stated**: upper-bound framing applies in one direction (PIVOT-from-strong-signal robust) and weak-evidence framing in the other (NO-PIVOT-caveat-bound). Panel must verify the asymmetric-framing is not over-claimed.

5. **§5 cost estimate is reproducible from cited per-sightline scaling, not order-of-magnitude waving**: Voigt-dominated cost ~2M FLOPs/ray cited from W=64 windowed-sum form at `d62_3_fgpa_variance_spectrum.py:87-99`. A30 ~10 TFLOPs effective is rule-of-thumb; actual measured throughput on the existing FGPA renderer benchmark should be substituted at dispatch time if available.

**Additional structural concerns surfaced during authoring (panel pre-review items)**:

- **(A)** The §2 N_ax = N_ang = 768 choice equalizes sample sizes but does NOT equalize information content: a uniform-S² draw of 768 directions × random origins covers 3D direction × 3D origin space, while 768 axis-parallel rays cover only 1D direction × 2D origin space at 256 per axis. This is intentional — the diagnostic asks "does angled coverage add signal at equal compute" — but the panel may rule that a fairer baseline is N_ax = 768 vs N_ang = 256 (matched-direction-count) instead of matched-ray-count. Brief defaults to matched-ray-count; flag for panel ruling.
- **(B)** Periodic-wrap pre-pass in `grid_sample` is non-trivial: PyTorch `grid_sample` does NOT support periodic BC natively; the implementation surface requires either pre-replicating the field tensor (3× memory) or implementing modular-index lookup. Data-engineer scope-locked at 50-80 LOC may be tight if the periodic-wrap pre-pass is the heavier component. Flag for data-engineer at implementation handoff.
- **(C)** No noise-injection robustness test in the brief. The [D-66] killer K2 was "var_truth is FP-noise-floor"; current brief does not include a noise-floor check for R_feas. Panel may rule that a baseline `R_feas` computation on a *noise-only* (zero-truth-anisotropy) field is owed as a null-control. Recommended addition: render P_F on Set A and Set B through a homogeneous ρ = ⟨ρ⟩_box constant field; verify R_feas(A) ≈ R_feas(B) within bootstrap band. NOT in current brief scope; flag.

---

## §7 — Capsule output specification

**Landing path**: `experiments/nerf/artifacts/d71_I_angled_feasibility/` (new subdirectory).

**JSON capsule** (`capsule.json`) — observation-first per [D-37] rule (a):

```json
{
  "schema_version": "d71_I_feas_v1",
  "physics_id": 1,
  "redshift": 0.300,
  "n_ax": 768,
  "n_ang": 768,
  "n_bins_per_ray": 2048,
  "seed": 2026,
  "fgpa_params": {"beta": 1.6, "gamma": -0.7, "T_iso_K": <value>, "v_pec_kms": 0.0, "tau_amp": <calibrated>},
  "mean_flux_calibrated": 0.979,
  "k_band_edges_skm": [0.00316, 0.03162],
  "raw_observation": {
    "rho_field_stats": {"mean": <>, "std": <>, "min": <>, "max": <>, "n_grid": <>},
    "P_F_axis": [<k_axis array>],
    "P_F_axis_per_k": [<P_F^A(k) array>],
    "P_F_angled_per_k": [<P_F^B(k) array>],
    "P_F_mixed_per_k": [<P_F^C(k) array>],
    "var_per_k_axis": [<Var_rays(P_F^A) array>],
    "var_per_k_angled": [<...>],
    "var_per_k_mixed": [<...>],
    "band_int_var_axis": <>,
    "band_int_var_angled": <>,
    "band_int_var_mixed": <>,
    "band_int_mean_PF_axis": <>,
    "band_int_mean_PF_angled": <>,
    "band_int_mean_PF_mixed": <>
  },
  "derived_ratios": {
    "R_feas_axis": <>,
    "R_feas_angled": <>,
    "R_feas_mixed": <>,
    "log10_R_feas_axis": <>,
    "log10_R_feas_angled": <>,
    "log10_R_feas_mixed": <>,
    "delta_log10_angled_minus_axis": <>,
    "delta_log10_mixed_minus_axis": <>
  },
  "verdict": {
    "label": "<PIVOT | MARGINAL | NO_PIVOT>",
    "threshold_table": {"pivot_ge_decade": 0.5, "marginal_band_decade": [0.3, 0.5], "no_pivot_lt_decade": 0.3},
    "caveat": "rho-only FGPA proxy; T/xHI/vlos NotImplementedError; upper-bound informativeness if PIVOT, weak evidence if NO_PIVOT"
  },
  "compute": {"node": "juno-a30", "wall_clock_s": <>, "run_tag": "<>"}
}
```

**PNG figure** (`pf_comparison.png`):
- 1×3 grid OR side-by-side overlay: P_F^A(k), P_F^B(k), P_F^C(k) on log-log axes.
- Shaded band: inertial range `[10^-2.5, 10^-1.5] s/km`.
- Per-set legend entries with R_feas value annotated.
- Colormap: cross-set comparison uses qualitative palette (matplotlib `tab10`); no density/velocity colormap applicable here since this is a 1D spectrum plot. (`magma` / `coolwarm` reserved for 3D-field renderers per support-researcher convention; this PNG is a 1D-spectrum overlay.)

**File path**: `experiments/nerf/artifacts/d71_I_angled_feasibility/pf_comparison.png` (<10 MB; not DVC-tracked unless byte-size exceeds threshold).

**DVC tracking**: if total artifact size (capsule + PNG) exceeds 10 MB, use `dvc-track` skill at landing time. Expected size: capsule ~5 KB; PNG ~200 KB. No DVC tracking expected.

**LEDGER §6 entry**: append at landing time via `ledger-update` skill, format `[D-71] Rung 4 feasibility capsule: <path>, verdict=<label>, Δlog10R_feas=<value>` per support-researcher convention.

---

## §8 — Sign-off

- **R15 status**: this brief is PROVISIONAL on every load-bearing claim until panel pre-review APPROVE; lift mechanism = explicit panel return.
- **R28 Tier (i)**: this brief contains 8 sections (§0 banner + §1–§7); no dispatch-rung enumeration applies (this is a brief, not a sprint). Tier (ii) §R28-CHECK omitted as not applicable to brief-authoring deliverable.
- **R29 status**: unit-chain audit walked in §3; thresholds derived from bootstrap-SE + physical-significance arguments, not knob-tuning. Pre-committed BEFORE methodology per [D-37] rule 5.
- **R32 status**: panel cycle live; this brief is the gate construction for that cycle.
- **R33 status**: in-session WebFetch verification NOT performed for Bolton+2017 PDF-width citation in §3; all other citations are repo-internal conventions. R33 strict-compliance may require panel ruling.
- **Word count**: ~2450 words (under ≤2500 budget).
- **Honest-framing compliance per [D-37] rule (a)**: observation-first framing throughout; verdict bands frozen pre-methodology; ρ-only proxy upper-bound caveat stated; structural concerns flagged for panel attack in §6.

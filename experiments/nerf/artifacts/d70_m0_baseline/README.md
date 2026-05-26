# d70_m0_baseline/ — diagnostic-only frozen-init noise-floor reference

**Status (Rev 3, 2026-05-25)**: This artifact is **frozen-init noise-floor reference, NOT the M0 PASS bar**. Per PI ruling absorbed in `experiments/nerf/design/D70_stage1_architectural_reframe_scoping.md` §0.7 Finding 1, the Rev 2 M0 gate spec of `> μ_frozen + 2σ_frozen` (= 3.867e-06) is **retired**. M0 at Rev 3 is a **direction-of-motion gate** (option δ): per-seed monotonicity of `Var_ratio(t) = Var(ρ_θ)/Var(ρ_truth)` across t ∈ {0, 100, 250, 500}, aggregate PASS ≥ 8/10 seeds.

## Diagnostic role of `baseline.json`

The JSON characterizes the *upper-2σ tail of the random-init variance-ratio distribution* for the current IGMNeRF architecture (8×256, single mid-skip, ReLU body, Softplus head; pre-(1b) wiring). Numbers retained for two diagnostic uses only:

1. **Source of `σ_frozen = 7.278e-07`**, which feeds the Rev 3 M0 statistical tolerance `ε_tol = 0.1 × σ_frozen / √N_crops = 7.28e-09` for the monotonicity clause.
2. **Sanity-check anchor for the (γ) constant-collapse-attractor narrative**: the (γ) lr=1e-4 trained floor at 2.25e-6 sits *below* μ_frozen = 2.412e-06, which is the evidence that training was anti-correlated with the quantity to grow — the empirical motivation for retiring threshold gates in favor of the direction-of-motion gate.

## Why the bar was retired

A (1b) variant whose trained variance lands just above μ + 2σ would PASS the Rev 2 bar without proving the constant-collapse attractor was escaped — it would only prove training didn't destroy more than ~2σ of init-chance variance. PI judged this insufficient discrimination: the pathology to discriminate is *"training is anti-correlated with the quantity we want to grow"*, which is a trajectory-shape statement, not an endpoint-magnitude statement.

## Rev 4 augmentation (2026-05-25, R28 PROVISIONAL)

The defense panel REJECTed D70 Rev 3 on K-A grounds: the Rev 3 `ε_tol = 0.1 × σ_frozen / √N_crops` formula confused three distinct statistics — it lifted `σ_frozen` (between-seed std of per-seed crop-MEANS) and divided by √N_crops as if it were a within-seed crop-to-crop SE, which it is not. The Rev 4 defense path is **Spearman ρ + permutation p-value** (no SE knob); within-seed SD therefore stops being a gate knob but remains useful as *honest disclosure* and is now recorded.

S-D additionally requires per-bin breakdown of the variance ratio to detect high-density-tail collapse hidden by aggregate-Var-ratio averaging.

New fields written by `scripts/d70_m0_frozen_init_baseline.py`:

| Field | Meaning |
| --- | --- |
| `crop_ratios_per_seed` | Raw 100-element per-seed crop ratio arrays (keys `seed_0` … `seed_9`). |
| `sigma_within_per_seed` | `std(crop_ratios_seed_i, ddof=1)` per seed — within-seed crop-to-crop dispersion. |
| `sigma_within_median`, `sigma_within_worst` | Honest-disclosure summary across the 10 seeds. **NOT used as a gate knob.** |
| `per_bin_ratios` | Per-seed per-bin mean Var(ρ_θ\|bin)/Var(ρ_truth\|bin) for bins {A void, B mean, C mid, D high-density tail}. |
| `per_bin_aggregate` | Per-bin {mean, std, n_seeds_valid, median_samples_per_crop, min_samples_per_crop} across the 10 seeds. |
| `bin_definition` | Bin edges on `log10(rho_truth / mean(rho))`: A ≤ -1; -1 < B ≤ +0.3; +0.3 < C ≤ +1.0; D > +1.0. Crops with < 5 samples in a bin contribute NaN to that bin. |
| `mean_rho_truth` | Mean of the n=768 ρ field used to define the over-density axis. |

Pre-existing fields (`ratios_per_seed`, `mu_frozen`, `sigma_frozen`, `m0_pass_bar`) are preserved so prior references continue to resolve.

## References

- `experiments/nerf/design/D70_stage1_architectural_reframe_scoping.md` §0.7 (Rev 3 PI absorption), §2 M0 critical clause, §6 KILLER attack item 4.
- `scripts/d70_m0_frozen_init_baseline.py` — generator script (currently untracked; R30 procedural confirm pending LEDGER write).
- `baseline.json` `notes` field — original Rev-2-era usage note; superseded by this README's diagnostic-only ruling.

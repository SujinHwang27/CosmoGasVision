# Benchmark-design survey of record (support-researcher, 2026-07-23)

**Question:** can CosmoGasVision inherit an existing Lyα-tomography benchmark, or must it design its own?
**Verdict: FORMALIZE OUR OWN — there is nothing to inherit.** No public (mock spectra, truth field, scoring code) triplet exists anywhere in the field; every method paper rolls its own mocks, noise, and metrics. Inherit the field's *conventions* for comparability; copy hosting/split *patterns* from CT and CAMELS.

## 1. Existing practice (verified per paper)

| Work | Mock data | Metrics | Noise | Public? |
|---|---|---|---|---|
| CLAMATO / Stark+2015 (arXiv:1504.03290, 1710.02894, 2109.09660) | DM-only TreePM 2560³/256 h⁻¹Mpc + FGPA, z≈2.3–2.5 | δ_F map comparisons; void/protocluster completeness; Wiener L⊥=2.5/L∥=2.0 | S/N≈4 per Å template (Lee+2014) | **dachshund Wiener code public** (github.com/caseywstark/dachshund); CLAMATO DR2 on Zenodo CC-BY (real data, NO truth pairs, no scoring code) |
| TARDIS I (arXiv:1903.09049) | same TreePM+FGPA class, z≈2.5 | tidal-eigenvalue Pearson r @2 h⁻¹Mpc (0.95); cosmic-web volume overlap (~80%) | power-law S/N (α=2.7) + continuum-error model σ_c=0.205/(S/N)+0.015 | no |
| TARDIS II (arXiv:2007.15994) | FastPM 256³/128 h⁻¹Mpc DM-only, z≈2.5 | density Pearson r @2 h⁻¹Mpc (0.72 joint / 0.56 Lyα-only); r_c(k); web classes 51–73% | S/N∈[2,10] power-law + continuum | no |
| **DeepCHART** (arXiv:2507.00135) | **own GADGET-3 runs with Sherwood-Relics physics** (NOT the Bolton+2017 Sherwood 60 Mpc/h suite): (40 h⁻¹Mpc)³ 2×512³ full hydro, z=2.5; 10 boxes, 9 train/1 test | voxel Pearson r on log DM density @1 & 2 h⁻¹Mpc (ρ≈0.77 current / 0.90 future); P(k) k=0.7–5; PDF | P(SNR)∝SNR⁻³·⁶, SNR∈[2,10], R=2500 (PFS); d⊥=2.4/1.0 | **code public** (github.com/soumak-maitra/DeepCHART); no data pairs |
| Porqueres+ 2019/2020 (arXiv:1907.02973, 2005.12928) | PT gravity + FGPA self-mocks (no hydro), 128 h⁻¹Mpc, z=2.5 | posterior-mean Pearson r (>0.8 near rays); P(k); posterior predictive | Gaussian σ=0.03 | no |
| Deep Forest (arXiv:2009.10673) | P-GADGET hydro 400 h⁻¹Mpc, z=3; **1D only** | τ RMSE | Gaussian S/N∈{2.5,5,10} | code public |
| ORCA (arXiv:2102.12306) | **Nyx full hydro** 100 h⁻¹Mpc, z≈2.05–2.55 | tidal-eigenvalue r (0.48–0.70); web volume 58.0%; "equivalent to 30–40% more sightlines" framing | S/N 1.4–10 PFS-like | no |
| LyAl-Net (arXiv:2303.17939) | Horizon-noAGN hydro | emulator (DM→forest), not tomography | — | — |

Cross-cutting: (a) classical methods validate on DM-only+FGPA; ML methods on full hydro; (b) **every paper is z≈2.3–3 — nothing at low z**; (c) no paper uses more than one feedback variant; (d) universal metric core = smoothed (log-)density/δ_F Pearson @1–2 h⁻¹Mpc + P(k)/r_c(k) + cosmic-web classes.

## 2. Shared benchmarks in Lyα tomography: NONE
DESI Lyα mock challenges are BAO/P1D-only (arXiv:2401.00303, 2404.03004, 2503.14741) — no 3D map-recovery task. CLAMATO DR2 = real data without truth. Chaves-Montero+ ML-Lyα review (arXiv:2605.22489, HTML-scanned) has no benchmark/evaluation-standardization section and does not survey 3D reconstruction. Sherwood has a public sightline release (Nottingham; Bolton+2017 arXiv:1605.03462) — raw material, not a benchmark.

## 3. Templates worth copying
- **LoDoPaB-CT** (Sci Data 8:109): fixed forward operator + fixed noise; frozen train/val/test + hidden challenge split; fixed metrics; leaderboard; Zenodo CC-BY. Lesson: *sampling operator and noise model are part of the benchmark definition.*
- **Libeskind+2018** (arXiv:1705.03021): 12 methods, one shared box — the "N methods, one dataset" model. DES Y3 mass-map comparison (Jeffrey+2021); kappaTNG public map suite.
- **CAMELS-CMD** (arXiv:2109.10915, 2201.01300): the direct precedent for releasing (input, truth) pairs ACROSS physics variants with loaders + standard tasks. **No CAMELS-style release exists for Lyα (spectra, truth) pairs — the niche is open.**

## 4. Verdict detail

**Inherit as conventions (verbatim, for comparability):** (1) voxel Pearson r on smoothed log-density at 1 and 2 h⁻¹Mpc as headline scales (makes our r_s readable against DeepCHART 0.77/0.90 and TARDIS II 0.72/0.56); (2) mean sightline separation d⊥ as the sampling parameter, reported at d⊥∈{1.0, 2.4} h⁻¹Mpc canonical points; (3) community noise bundle: P(SNR)∝SNR^−α (α∈[2.7,3.6]), SNR∈[2,10], Gaussian per-pixel + TARDIS continuum-error σ_c=0.205/(S/N)+0.015 + LSF (R=2500 PFS convention; COS-appropriate R for z=0.3); (4) dachshund as the runnable classical baseline.

**What [D-75] already has that NO prior work has:** pre-registered win conditions; acceptance-tested estimators; control fields (phase-randomized truth; low-pass ladder giving r_s↔scale meaning; sampling-operator control); stochasticity r(k) + NCCF; dual real/redshift-space framing.

**What it lacks for benchmark status:** observational-realism noise bundle; multiple realizations (we have 4 physics × 1 seed; quote sub-volume jackknives minimum); cosmic-web classification tier; a z≈2.5 config to touch the existing literature; public hosting + frozen splits + scoring code + license.

**Ranked top-5 additions (effort → payoff):**
1. Standard-convention noise + geometry package (α∈{2.7,3.6} SNR ladders, continuum error, LSF; d⊥∈{1.0,2.4}) — **small-medium**; scores become readable against TARDIS/ORCA/DeepCHART.
2. In-house reference baseline entries on our exact splits: dachshund Wiener + retrained DeepCHART — **medium**; the leaderboard ships with its first rows.
3. Cosmic-web classification tier (tidal-eigenvalue Pearson + per-class volume-overlap confusion @2 h⁻¹Mpc) — **small**; the first metric that community's reviewers look for.
4. Public release engineering: (spectra, truth) pairs ×4 physics + frozen splits incl. hidden challenge set + pip-installable scoring + Zenodo DOI CC-BY-4.0 — **medium-large**; the difference between "evaluation section" and "benchmark paper"; first CAMELS-style pair release in Lyα.
5. Generalization axes: cross-physics protocol (train on one variant, score on the other three — the unique selling point) + a z≈2.5 snapshot config — **medium-large**; connects the benchmark to the entire existing literature.

**[D-37] honest-framing note:** the z=0.3 + multi-physics niche is genuinely open; the corollary is that no external number exists to compare our z=0.3 scores against verbatim — comparability comes only through adopted conventions + self-run public baselines.

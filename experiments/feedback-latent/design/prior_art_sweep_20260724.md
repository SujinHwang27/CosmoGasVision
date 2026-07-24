# Prior-art sweep of record — exp/feedback-latent ([F-01] founding input)

**Sweep date:** 2026-07-24 (support-researcher). **Question:** is the conjunction — auto-decoder physics vectors inside a coordinate field, trained truth-supervised on a shared-IC feedback-variant suite, with latent-decoded delta maps validated against TRUE paired difference maps, plus ∂spectrum/∂latent through a differentiable Voigt renderer — unclaimed, and which components are derivative?

**Bottom line: proceed. The conjunction is open; the components are not.** Novelty verbs must attach to the *conjunction and the validation standard*, never to "first feedback latent." The two 2025/26 CAMELS-latent papers must be cited and differentiated in the first paragraph that uses the word "latent."

---

## Front 1 — CAMELS-family emulator/latent work (the headline is already claimed, twice)

- **Lin, S. et al., "One latent to fit them all," ApJL 996, L41 (2026), arXiv:2509.01881.** β-TCVAE over 5,072 CAMELS sims / 4 galaxy-formation models learns a **2D latent of baryonic feedback**, redshift- and cosmology-independent, disentangling SN vs AGN axes. **Owns the headline phrase-space.** Differences from us: learned from matter power suppression / transfer functions (a summary statistic) via an encoder over thousands of sims; no coordinate-field container, no field-level delta-map validation, no Lyα, no spectra.
- **Liu, M.-S. & Cuesta[-Lazaro], C., "Continuous Representations of Baryonic Feedback...," NeurIPS ML4PS 2025.** Shared continuous latent of simulator physics extended to **fields** (stellar mass, gas density, temperature, pressure) across CAMELS suites — the closest field-level relative. Aim: inference robustness/marginalization; no auto-decoder physics vectors, no truth-paired delta maps, no spectra gradients. (Workshop paper; verify author spelling before citing.)
- **Sharma, D. et al., MNRAS 538, 1415 (2025), arXiv:2401.15891 (GPemu):** field-level emulator of baryonic effects conditioned on *known* feedback strength — template for readout-2 minus the latent-discovery and truth-paired validation.
- CAMELS parameter-regression line (Villaescusa-Navarro+2021; CMD, arXiv:2201.01300) regresses known (A_SN1/2, A_AGN1/2), not an unsupervised latent.
- **Shared-seed differencing is standard, NOT a novelty:** van Daalen+2011 (OWLS), CAMELS 1P design (one-parameter-at-a-time at fixed IC seed), baryonification (Schneider & Teyssier 2015). "Shared ICs cancel cosmic variance" is an inherited design virtue.

## Front 2 — Conditioned neural fields / auto-decoders (architecture is derivative by design)

- **DeepSDF (Park+ CVPR 2019, arXiv:1901.05103):** the auto-decoder — per-instance latent codes optimized jointly with a shared decoder. Our "4 learnable physics vectors" = DeepSDF with 4 instances.
- **SIREN (Sitzmann+2020), Instant-NGP hash grids (Müller+2022):** the trunk options (explicitly NOT the falsified ReLU+PE MLP lineage).
- Sci-vis already conditions INRs on simulation parameters across ensembles: FA-INR (arXiv:2506.06858), VDL-Surrogate (arXiv:2207.13091), INR-for-ensembles (arXiv:2504.00904).
- Cosmology conditioning via style params: field-level N-body emulator (arXiv:2206.04594, NECOLA); **HyPhy (Horowitz+2022, arXiv:2106.12675)** conditionally generates hydro fields (incl. Lyα) from DM via a VAE.
- **No cosmology paper found using a DeepSDF-style auto-decoder physics code inside a coordinate field over a physics-variant suite** — that specific composition appears open.

## Front 3 — Feedback imprints on the low-z Lyα forest (occupied at the summary-statistic level, incl. our exact suite)

- **Bolton+2017, MNRAS 464, 897** — the Sherwood suite paper (our data), includes the feedback variants.
- **Nasir, Bolton et al. 2017, MNRAS 471, 1056, arXiv:1706.04790** — *"The effect of stellar and AGN feedback on the low-redshift Lyα forest in the Sherwood simulation suite."* **The exact question, the exact suite.** Statistics: CDDF, line-width (velocity-width) distribution. **Finding of record: feedback as implemented has only a SMALL effect on those statistics.** (Correction to memory: Nasir+2017, not a section of Bolton+2017.)
- Viel+2017 (b-parameter); Christiansen+2020 (Simba AGN jets vs Photon Underproduction); **the Tillman/Pirecki CAMELS-Simba line at low z** (arXiv:2204.09712; 2210.02467; 2307.06360; 2410.05383 P1D; 2509.18260) — flux PDF, P1D, CDDF, line widths, mean flux. **None extracts a learned/latent representation; direct parameter variation only** (confirmed for the newest, Pirecki+).

## Front 4 — Differentiable-renderer sensitivity in cosmology

- **TARDIS (Horowitz+2019; II 2021):** differentiable Lyα forward model, gradients w.r.t. ICs/density — already our baseline. A reviewer knows "differentiable Lyα rendering" is TARDIS's home turf.
- **THALAS (Ding, Horowitz, Lukić 2024, arXiv:2407.16009):** public "fully differentiable" map (baryon density, T, v) → Lyα optical depth in real & redshift space. Our renderer is not unique as an artifact.
- **BORG-Lyα (Porqueres+2019/2020):** differentiable field-level Lyα inference, gradients w.r.t. ICs.
- d(observable)/d(parameter) exists in lensing (Lanzieri+2023 DLL, arXiv:2305.07531; JAX-COSMO).
- **No prior computation of ∂(spectrum)/∂(feedback parameter or latent) via autodiff through a Voigt renderer found.** Readout 3 appears unclaimed; both ingredients (THALAS, DLL) pre-exist separately — only the composition is ours.

---

## The 5 most dangerous citations (must be answered)

1. **Lin+2026 ApJL (arXiv:2509.01881)** — owns "feedback latent." We must frame as complementary (field+spectra-native, truth-paired, renderer-differentiable) or read as a 4-sim rediscovery.
2. **Nasir+2017 MNRAS 471, 1056** — same suite, same question, *small-effect* result. Direct reviewer question: "What does the latent recover that Nasir's statistics missed?" If our delta maps only re-express their small effects, the contribution is method, not science.
3. **Liu & Cuesta 2025 ML4PS** — field-level continuous feedback representation across suites; closest methodological relative; timestamps the idea.
4. **Sharma+2025 MNRAS 538, 1415 (GPemu)** — field-level conditioned decoding of baryonic effects.
5. **Ding, Horowitz & Lukić 2024 (THALAS)** — public differentiable Lyα renderer (alt slot-5 if science-first: Tillman+2023 ApJL 945, L17).

## Does "the feedback latent of the Sherwood suite" survive?

**Yes — with that exact genitive and three honest pins.** Mental model: we do not discover feedback's latent space; we build a **4-point chart of one suite's subgrid ladder**, at fixed cosmic structure, one box, one redshift.

1. **n=4 variants → latent "geometry" is 4 points.** Ordering/collinearity are weak evidence alone. **d=1-vs-d=8 must run strictly as the pre-registered fit-quality comparison** (4 points always *embed* in 1D; the question is whether 1D *decodes* as well). d=8 gives more latent dims than training instances — reviewers will flag it; pre-empt with the fit-quality protocol.
2. **The claim no prior work can make: delta maps decoded from the latent, validated against TRUE same-IC difference maps.** The CAMELS-latent papers structurally cannot do this on LH sets and did not on 1P. **This is the paper's spine.**
3. **∂spectrum/∂z_p through the Voigt renderer** is the genuinely unclaimed readout and the survey-facing deliverable — frame as sensitivity of observables to *this suite's feedback ladder*, not to "feedback" in general. Interpolated latents decode hypothetical fields — descriptive only.

**Sources:** arXiv:2509.01881 · ML4PS 2025 #362 · arXiv:2401.15891 · arXiv:1901.05103 · arXiv:2106.12675 · arXiv:1706.04790 · Bolton+2017 MNRAS 464,897 · arXiv:1610.02046 · Tillman/Pirecki 2204.09712/2210.02467/2307.06360/2410.05383/2509.18260 · TARDIS II · arXiv:2005.12928 · THALAS arXiv:2407.16009 · Lanzieri+ arXiv:2305.07531 · FA-INR 2506.06858 · VDL-Surrogate 2207.13091.

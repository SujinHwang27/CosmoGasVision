# [U-04] Founding stage-gate ratification + Stage-1 design spec of record (PI, 2026-07-23)

> Committed verbatim by the coordinator as the ratification of record. Sign-off status: **R15 PROVISIONAL by default — PI-only, deferred panel review**; a narrow defense-panel pre-review is owed before the first Juno dispatch (Stage 2), not before Stage 1 ($0, local, no data-product claims).

**Verdict: RATIFY [U-01] gates WITH AMENDMENTS (3 rulings + 1 disclosed adjustment to the G2 bar's operative form). Stage 1 dispatch AUTHORIZED.**

**In-session re-verification block (R26).** [U-03] numbers re-verified against `experiments/nerf/artifacts/d75_rescore/d75_scores.json`: grid r_s(σ=2, real) = 0.59537 (octant SE 0.0196); MLP = 0.27458 (SE 0.0518); Wiener best-L = L3 both frames, 0.16515 (SE 0.0312); primary frame `"real"`; five-condition pair machinery confirmed in the `pairs` block. Claims SURVIVE. Drift flag: the [D-75] spec markdown is absent from THIS branch's tree — it lives on `exp/nerf` (commits 57985bb/753a478/f80f7df); ratification performed against the executable spec (driver + `nccf.py` + scores JSON). RESOLVED as branch-visibility, not loss.

**Prior-failure ledger (Ext-2 rule 4).** [D-40] integrated-stat loss → amplitude-shrink; [D-41] self-anchored regularizer → constant-prediction collapse; [D-42] input-hint → density-head collapse; [D-46] data-axis → D4; [D-73] K2 flux likelihood does not identify the field. Verb discipline: this track is the **first test of cross-example prior injection at z=0.3** — a candidate, not "structurally immune", not "expected to win".

---

## 1. Gate ratification

**G1 (estimator acceptance — hard prerequisite to any Stage-2 scoring).** [D-75] acceptance suite via `src/analysis/nccf.py` unchanged: (a) truth-vs-truth NCCF = 1.0 ± 1e-6 at every valid r bin (validity: min(C_xx, C_yy) > 0.01·C_xx(0)); (b) truth-vs-truth r_s(σ) = 1.0 ± 1e-6 at σ ∈ {1,2,4}; (c) NEW: both re-run under the held-out-region scoring mask (mask-restricted scoring path is new code; inherits G1, not grandfathered); (d) low-pass ladder monotone in k_c. Degenerate-variance policy (std < 1e-12 → UNDEFINED, never 0) verbatim.

**G2 (methods win — five-condition test, three pairs, held-out region only).**
- Frame ruling: **real-frame primary** (matches [D-75]); z-space diagnostic column; conditions 4/5 keep both-frames sign checks.
- Primary metric: r_s(σ=2 h⁻¹Mpc) on x = log₁₀(max(ρ/⟨ρ⟩, 1e-3)), 192³ lattice, **test region of the [D-49] split only**, U-Net evaluated at the fixed 1024-ray eval pattern (64-ray secondary column for MLP-lineage comparability).
- Five conditions per ordered pair A>B (identical to [D-75] §7 B-ii with two disclosed held-out-geometry adaptations, both recorded in the output JSON deviations block): (1) Δr_s ≥ 0.10; (2) Fisher-z paired t over 8 congruent sub-blocks of the test slab (2 axis-0 × 2×2 transverse; replaces full-box octants), df=7, t_crit=2.365, mean Δz > 0; (3) block bootstrap n_boot=1000, full-thickness transverse column blocks ~(slab×24×24), ≈64 blocks (wider CIs, conservative), CI95 excludes zero; (4) Wiener sign consistent both frames (null where no Wiener column); (5) no frame sign reversal.
- Pairs: `unet>grid`, `unet>wiener_L3`, `unet>mlp` (+ reverses for tie detection).
- **Operative-bar adjustment (binding):** 0.595 is a full-box number; all four columns (unet, grid, mlp, wiener_L3) are RE-SCORED on the identical test-region mask from the existing md5/sha256-pinned cubes (mechanical, CPU, support-researcher). G2 runs on those held-out numbers; 0.595 stays the cited full-box context anchor. Direction is conservative against the U-Net (baselines had test-slab likelihood access; the U-Net never sees test-slab truth).
- Overfitting-leak diagnostic (mandatory column, not a gate): r_s stratified by [D-49] `distance_to_train_region`.

**G3 (cross-physics generalization).** Train on 3 physics, evaluate held-out 4th (same test region, same suite). Provisional tolerance: absolute drop Δr_s(σ=2) ≤ 0.05 vs the 4-physics-trained model on the same physics/region. Flagged self-anchored (R14-fragile as headline); generalization gate, not win bar; finalized at the Stage-3 spec with panel review. Fold choice deferred.

**Outcome cells (all pre-committed, symmetric disclosure):**

| Cell | Condition (held-out, five-condition machinery) | Disposition |
|---|---|---|
| U0 | G1 fails, or scoring-path acceptance fails, or artifact-identity mismatch | Fail-closed; audit first |
| U-A | `unet>grid` fires (and `unet>wiener`, `unet>mlp` fire) | **Paper-grade win** — prior adds ≥0.10 beyond the measured likelihood-only ceiling; paper re-licensed per benefit gate |
| U-A′ | `unet>grid` fires but `unet>wiener` or `unet>mlp` does NOT | Transitivity anomaly → fail-closed audit before any claim |
| U-B | `unet>mlp` + `unet>wiener` fire; tie with grid | Prior ≈ likelihood ceiling, adds nothing beyond; publishable as extension of the under-constraint characterization; no win verb |
| U-C | `unet>mlp` fires; `grid>unet` fires | Under-ceiling; diagnostic follow-up (crop size, capacity, ray curriculum, budget) before interpretation |
| U-D | `unet>mlp` does not fire, or `mlp>unet` fires | **Implementation-suspect**; Stage-2 close BLOCKED; debug checklist before science |
| U-E | Win only at σ=4 | Scoped large-scale-only claim; verb ceiling |
| U-F | Condition-5 frame sign reversal sole failure | Frame-artifact audit; fail-closed |
| U-G | Zero-input control r_s ≥ 0.5 × actual | "Input-ignoring" flag: network not using rays; blocks any inversion claim |
| U-H | Held-out-physics drop > 0.05 | Prior physics-brittle; claims scoped "within-suite, trained-physics" |
| U-I | Trainability/overfit process failure pre-eval | Process finding; symmetric disclosure per [D-37] |

---

## 2. Stage-1 design spec (pair-manufacture plumbing)

Scope: data plumbing only. No training loop, no model, no GPU. Hard cap: $0 cloud, zero Juno, ≤1 CPU-hr aggregate truth-cube production, ≤30 min unit-test wallclock.

**(a) Sampler + resolution ruling: 192³ scoring-pitch lattice (312.5 kpc/h voxels), crop 64³ (= 20 h⁻¹Mpc).** Rationale: scoring lattice IS 192³ (removes regrid confound); gate scales need ≥15–20 Mpc/h context; [D-49] train region ~134 voxels thick at 192 pitch → ~70 distinct axis-0 offsets at 64³ (96³ halves diversity; 128³ leaves ~6 — rejected; revisitable as Stage-2 ablation iff U-C). Memory trivial (~2 MB/example). Random positions in [D-49] train region; **strict straddle rejection reused unchanged**; per-example fresh random ray subset, count log-uniform in [64, 1024]; 90° transverse rotations/flips; all 4 physics. Zero-ray crops rejected; rejection rate logged. Eval: fixed 1024-ray pattern (primary) + fixed 64-ray (secondary), test region only, sliding-window full-box inference masked to test voxels.

**(b) Rasterizer + input-variable ruling: input = flux decrement δ_F = 1 − F, NOT τ.** F is the bounded observable; τ unbounded (saturated spikes would dominate activations and reintroduce cap/log choices in the input); DeepCHART precedent is flux-domain; at z=0.3 the signal is small dips on ⟨F⟩≈0.98 which δ_F represents natively near zero. Fixed pre-registered global scale constant (≈×50; final value pinned by core-implementer in the Stage-1 exit artifact). **Per-crop or data-dependent normalization FORBIDDEN** (destroys cross-physics amplitude information). Channels: ch1 = scaled δ_F along ray voxels (zero elsewhere); ch2 = binary ray mask. Bin-to-voxel: loader's canonical `vel_axis` mapping; nearest-voxel (floor); ~10.7 bins/voxel → mean of assigned bins; multiple rays in a voxel → mean, mask stays 1. Must handle all three ray axes (mixed-axis production sightlines: 334/331/359 x/y/z in first 1024). Disclosed loss: voxel-averaging discards sub-voxel line-profile detail; multi-sub-bin-channel variant = pre-registered Stage-2 ablation candidate. Frame note: flux is z-space, truth real-space; the implicit frame mapping is a capability under test, policed by condition 5 + z-space column.

**(c) Target ruling: x = log₁₀(max(ρ/⟨ρ⟩, 1e-3))** — exactly the [D-75] scoring variable, base 10, floor 1e-3 (truth clamped fraction 4.1e-5, negligible). P1 target crops bit-exact from the sha256-pinned `truth_real_192.npy`. P2–P4 192³ cubes produced by the IDENTICAL producer path as the [D-75] scoring cube ([D-48] cache + [D-50] chunked CIC + 768→192 mean-pool), each with JSON provenance manifest (producer commit, sha256, mean ∈ [0.95, 1.05]).

**(d) Stage-1 exit criteria.** S1: unit tests PASS — crop containment + straddle rejection; ray-axis handling ×3; synthetic-ray round-trip (constant-F ray → exact on-path values, exact zeros off-path); bin-to-voxel mean equivalence; mask correctness; target transform matches scoring `x_transform` ≤1e-12; seeded determinism. S2: P1 target-crop bit-exactness vs pinned cube (sha256 + sampled-crop equality). S3: one rendered figure per physics (central slab: δ_F channel, mask, target x) under `experiments/unet-inversion/artifacts/stage1/`; PI eyeball at the Stage-1 gate. S4 (reported, not gated): per-crop ray-occupancy stats at n_rays ∈ {64, 1024} + zero-ray rejection rate.

**(e) Ownership + order.** (1) **data-engineer FIRST** — first deliverable BEFORE anything touches Sherwood data: current-session data-locality enumeration (R15 clause (c); the 2026-05-13c rule): empirical audit of `Sherwood/Physics{1..4}_*/los2048…`+`tauH1…`, `SherwoodIGM_gal/` extraction state per physics, sha256 of `truth_real_192.npy` vs `d75_scores.json`, grid/mlp/wiener cube md5s vs `d75_scores.json`. Then P2–P4 truth cubes + manifests; then the truth-crop provider over the [D-49] split at n=192 (in `src/data/`). (2) **core-implementer SECOND** (after the truth provider): sampler, rasterizer, dataset class, unit tests, viz script. (3) **support-researcher THIRD, parallel after the locality audit** (Stage-2 gate prep, not a Stage-1 blocker): held-out-region-restricted re-scoring of grid/mlp/wiener_L3 → operative G2 bars JSON under `experiments/unet-inversion/artifacts/`; plus the [U-02]-owed Chaves-Montero 2026 (arXiv:2605.22489) field-level-inference scan (blocks novelty claims, not Stage 1).

---

## 3. Anti-degeneracy audit (Ext-2 rule 3 — what log-ρ MSE on crops leaves unconstrained, stated in advance)

1. **Amplitude/variance**: MSE → posterior-mean → variance compression + small-scale power suppression; r_s is affine-invariant so the gate cannot see it — mandatory descriptive columns: predicted/truth variance ratio + P(k) ratio; every win claim carries the amplitude-compression disclosure.
2. **Diffuse-bin majority**: x-MSE dominated by void/sheet bulk; high-density peaks weakly constrained → expect truncated high-x tails; control: predicted-vs-truth x-PDF overlay.
3. **Input-ignoring solution**: nothing forces ray use; smooth generic-prior output is a valid low-loss basin — control: zero-input + shuffled-ray inference; cell U-G fires at ≥0.5× actual.
4. **Memorization/leakage**: one box, repeated truth crops → [D-49] disjoint test region + strict straddle rejection + distance-stratified readout.
5. **Physics averaging**: joint 4-physics training fits an average prior → per-physics eval columns; G3 measures brittleness.
6. **Frame slack**: real-frame loss with z-space inputs can absorb axis-dependent displacement → mixed-axis rays randomize; condition 5 + z-space column police.

---

## 4. Compute posture

Local smoke ladder pre-Stage-2: (s1) unit tests CPU; (s2) overfit-one-batch, 50 steps, loss must drop ≥10×; (s3) 500-step 2-physics mini-run MPS/CPU with the contract assertion live. **R20 twin-gate owed before HPC (both):** (i) behavioral integration test on the real dataset→rasterize→forward→loss path — asserts `loss.grad_fn is not None`, weights move ≥1e-6 over ≥2 optimizer steps; (ii) contract assertion at step ~100 inside the real loop, outside try/except, raising loud: prediction std on fixed val batch > 0.01 in x-units AND loss(100) < loss(1). **Juno reachability re-verification OWED (infrastructure-manager) before Stage-2 planning**, with the standard cost block (partition, hrs/run, $/run, ceiling, auto-stop, artifact lifecycle).

---

## 5. §R28-CHECK

Landing artifacts (9): A1 locality report · A2 P2–P4 truth cubes + manifests · A3 truth-crop provider · A4 sampler · A5 rasterizer · A6 dataset class · A7 unit tests · A8 viz script + 4 figures · A9 held-out baseline re-score JSON. Rungs (9): R1=A1 → R2=A2 → R3=A3 → R4=A4 → R5=A5 → R6=A6 → R7=A7 → R8=A8 → R9=A9 (support-researcher, parallel after R1). 9 ≥ 9 ✓. Grouped into 3 agent dispatches (data-engineer R1–R3; core-implementer R4–R8; support-researcher R9). PROVISIONAL lift = coordinator commit of this ratification.

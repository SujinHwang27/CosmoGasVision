# [U-06] Stage-2 design spec — U-Net training + evaluation (PI, 2026-07-23)

> Committed verbatim by the coordinator. NOTE: the PI draft carried the label "[U-05]"; renumbered to **[U-06]** at commit — [U-05] was already assigned to the benchmark survey. No content change.
> Sign-off status: **R15 PROVISIONAL — PI-only.** Two lifts owed before any Juno dispatch (§(e)); the local smoke ladder (s1–s3, $0, CPU/MPS) is dispatchable under this PROVISIONAL. Review trail per Ext-2 rule 6: gates >$5 compute → defense-panel pre-review REQUIRED before HPC.

**Prior-failure ledger (Ext-2 rule 4).** [D-40] integrated-stat loss → amplitude shrink; [D-41] self-anchored regularizer → constant-prediction collapse; [D-42] input-hint → density-head collapse; [D-73] K2 flux likelihood does not identify the field; **new, R9 2026-07-23**: per-scene MLP held-out recovery consistent with zero — the ceiling this track must beat is the grid's 0.5758, and the likelihood-only floor is effectively null on the slab. Verb discipline: Stage 2 is the **first test of cross-example prior injection at z=0.3** — a candidate. No outcome verb until G2 runs.

**Anti-degeneracy line item (Ext-2 rule 3) — what full-crop x-MSE leaves unconstrained here.** Input occupancy is 1.5%–22% of voxels; the MSE is dominated by off-ray voxels, so the top-hazard basin is the **input-ignoring generic-prior inpainter** (a smooth web-prior output scores nonzero r_s against any Sherwood-like target — this is precisely what the elevated masked phase-rand null measures). Police: (1) zero-input + shuffled-ray controls from the FIRST smoke (cell U-G at ≥0.5× actual); (2) null-band comparison (§(c)); (3) prediction-std contract floor (R20(ii)). Second basin: posterior-mean variance compression (r_s is affine-invariant and cannot see it) — police: variance-ratio + P(k)-ratio + x-PDF descriptive columns on every readout. Third: physics averaging — per-physics eval columns from the first multi-physics run.

## (a) Architecture ruling

`src/models/unet3d.py` — 3D U-Net, **4 levels (3 downsamplings), base 32 channels, doubling to 256** at the bottleneck. Per level: 2 × [Conv3d 3³ → GroupNorm(8) → SiLU]; down = stride-2 conv; up = nearest-upsample + 3³ conv, skip concatenation; head = 1×1×1 conv to 1 channel, **no output activation** (x spans ≈ [−3, +3.6]). Input (2, 64³): ch0 = 12.5·δ_F on ray voxels, ch1 = binary mask (Stage-1 rasterizer, unchanged). GroupNorm not BatchNorm (batch ≤ 4 locally). **Parameter budget ≈ 5.5 M** (within the 5–25 M envelope). Receptive field ≈ 60–90 voxels — covers the 64³ crop. Capacity increase (base 48/64) is a pre-registered U-C diagnostic knob, not a default.

**Loss: MSE on x over ALL crop voxels** (per [U-01]; the off-ray majority IS the inpainting task). **Optimizer: AdamW, lr 3e-4, wd 1e-4, betas default, grad-clip 1.0.** LR schedule: constant for s2/s3 smoke; cosine with 500-step warmup reserved for the Juno run. fp32 throughout locally (MPS bf16 not trusted). Seed 42 primary.

## (b) Training protocol — smoke ladder (local, $0)

Memory arithmetic (MPS/CPU): per example, input 2.10 MB + target 1.05 MB; activations dominate (level-1 maps 33.5 MB per retained tensor, ≈0.4–1 GB per example with autograd). **Batch 4 ≈ ≤4 GB total** — fits 16 GB unified memory; CPU fallback batch 1–2 with grad-accum to effective 4. Expected s3 wallclock: 500 steps × ~1–3 s ≈ 10–25 min MPS.

- **s1** — Stage-1 tests (green) + the **R20(i) behavioral integration test** (`tests/test_unet_training_contract.py`): real `UNetPairDataset` → rasterize → forward → MSE; asserts `loss.grad_fn is not None and loss.requires_grad`; asserts max-abs weight change ≥ 1e-6 over ≥2 optimizer steps. No HPC dispatch without this test existing and green.
- **s2** — overfit-one-batch: one fixed batch of 2 examples (seed 42), lr 1e-3 constant, 50 steps. **Gate: loss(50) ≤ 0.1 × loss(0).** May run to 200 steps as diagnostic; the gate reads at 50.
- **s3** — 500-step mini-run: P1+P2 sources, batch 4, lr 3e-4, n_rays log-uniform [64, 1024], augmentation on, dataset length 4096. **R20(ii) contract assertion at step 100, inside the loop, OUTSIDE try/except, raising loud: prediction std on a fixed val batch > 0.01 in x-units AND loss(100) < loss(1).** Fixed val batch: 8 examples, seed 4242, augment off.

**MLflow contract:** experiment `CosmoGasVision/unet-inversion`; run names `Stage2-OverfitOneBatch`, `Stage2-MiniRun500`, later `Stage2-TrainFull`; mandatory tags `model_type=unet3d`, `stage=2`, `physics_id`, `redshift=0.3`, plus `seed`, `delta_f_scale=12.5`, `crop_size=64`, `base_channels=32`, git commit. **`nullcontext` fallback mandatory** (tracker-403 precedent); all metrics mirrored to a local CSV under `experiments/unet-inversion/artifacts/stage2/`.

## (c) Eval protocol

- **Inference:** sliding-window over the full 192³ box, 64³ windows, **stride 32, uniform averaging on overlaps**; rays = fixed eval patterns pre-registered as **file-order sightlines [0, 1024)** (primary) and **[0, 64)** (secondary, MLP-lineage comparability). Each window rasterizes its intersecting subset; no augmentation at eval.
- **Scoring:** exactly the R9 conventions, imported not re-implemented — smooth full periodic cube FIRST (σ ∈ {1,2,4}), then restrict to the [D-49] test mask (`region_voxel_interval('test', 192)`, runtime-asserted); Pearson primary, Spearman column; real frame primary, z-space column (truth_zspace md5-pinned); 8-block means/SEs per the R9 geometry. Every readout quoted against the N=200 null band.
- **Null-band protocol (pre-registered, from the Stage-1 gate ruling):** N = 200 phase-randomized realizations of `truth_real_192`, seeds `default_rng([20260726, i])` i = 0..199 (banked seed-20260726 draw = realization 0 for lineage); each scored on the identical mask, both frames, σ ∈ {1,2,4}, Pearson + Spearman; quote median + [2.5, 97.5] percentile band per (σ, frame, metric). **"Above null" = above the 97.5th percentile.** Artifact `stage2/null_band_n200.json`. The single-draw 0.1185 is retired as a null level.
- **G2 pair tests:** the [U-04] five-condition machinery against the R9 operative bars (grid 0.5758 / wiener_L3 0.0863 / mlp 0.0727) — run at Stage-2 close (post-Juno), not at smoke. Mandatory diagnostic column: r_s stratified by `distance_to_train_region`. Note of record: only masked-vs-masked comparisons are admissible; `unet>mlp` is a weak floor (MLP held-out consistent with zero) — interpretation weight sits on `unet>grid` and the U-B tie boundary.
- **Controls, wired from the FIRST smoke eval (cell U-G):** (z1) all-zero input; (z2) mask-only (ch0 zeroed, real mask); (z3) shuffled-ray (permute δ_F assignments across rays, geometry intact). U-G fires if any control r_s ≥ 0.5 × actual.
- **Anti-degeneracy descriptive columns on every eval:** var(pred)/var(truth) on the test mask (unsmoothed and σ=2); P(k) ratio of predicted vs truth x-field (full box); predicted-vs-truth x-PDF overlay (test mask).

## (d) Pre-registered smoke gates (numbers)

| Gate | Threshold | On fail |
|---|---|---|
| s1 integration test | grad_fn present; weight movement ≥ 1e-6 / 2 steps | Fix before anything else |
| s2 overfit | loss(50) ≤ 0.1 × loss(0) | STOP; architecture/optimization debug |
| s3 contract (step 100) | pred std > 0.01 x-units AND loss(100) < loss(1) | Loud AssertionError; STOP |
| s3 loss trend (step 500) | smoothed train MSE(500) ≤ 0.7 × MSE(10) | STOP; lr/batch debug |
| s3 quick masked eval (step 500, trained physics, 1024-ray pattern) | **GREEN — authorize Juno planning: r_s(σ=2, real, masked) ≥ 0.20 AND > null 97.5th pct AND U-G controls clean.** AMBER — r_s in [null-97.5, 0.20): ONE local retune iteration (lr/batch/steps ≤ 2000), then re-read. RED — r_s ≤ null 97.5th pct, or var ratio < 0.01 (collapse), or U-G fires: STOP; anti-degeneracy audit before any HPC. | Per cell |

Honest framing pre-commit: 500 steps ≈ 2000 examples; GREEN at 0.20 is deliberately modest (the bar to beat at full training is 0.5758 — 0.20 at smoke only licenses spending compute, it is not a result). A RED here is a valid end state per Ext-2 rule 7 and routes to U-D/U-I disposition, not to gate-bending.

## (e) Pre-Juno blockers (both BLOCK any HPC dispatch; neither blocks s1–s3)

- **B1 — narrow defense-panel pre-review of THIS spec** (the R15 lift owed since [U-04]; scope: gate construction §(d) incl. unit-chain check on the null-band/threshold framing, anti-degeneracy audit sufficiency, architecture/loss rulings).
- **B2 — infrastructure-manager Juno block**: reachability re-verification + the standard cost block — partition, hrs/run, $/run, total ceiling, auto-stop, artifact lifecycle. Envelope: single GPU, ≤12 hr primary run, + one seed-repeat contingency; reject any infra plan missing a cost-block element.

## (f) §R28-CHECK (Tier ii)

Landing artifacts (9): A1 `src/models/unet3d.py` · A2 `experiments/unet-inversion/pipeline.py` (training loop + MLflow contract + step-100 contract assertion) · A3 `tests/test_unet_training_contract.py` · A4 eval harness (sliding-window inference + masked scorer importing R9 conventions + U-G controls + descriptive columns) · A5 `stage2/null_band_n200.json` · A6 s2 record · A7 s3 record + quick masked eval readout · A8 B1 panel verdict record · A9 B2 infra reachability + cost block.
Rungs (9): R1=A1 → R2=A2 → R3=A3 → R4=A6 → R5=A7 (core-implementer, sequential); R6=A4, R7=A5 (support-researcher; A5 parallel immediately); R8=A8 (panel, after spec commit, before Juno); R9=A9 (infrastructure-manager, parallel). 9 ≥ 9 ✓. Juno dispatch requires A1–A9 all green.

---

## Appendix — Stage-1 gate verdict + R9 absorption rulings (PI, same review)

**Stage-1: APPROVE — PASS on S1–S4** (16+12 tests green; S2 bit-exactness asserted; all four S3 figures eyeballed; S4 occupancy 1.55%@64 / 22.1%@1024, rejection 0/400). Deviations ratified: δ_F scale ×12.5 measurement-rules (asinh stretch = pre-registered ablation candidate requiring spec amendment, not a silent switch); byte-identical LOS geometry across physics NOTED with three implications — helps G3 integrity (physics not identifiable from mask), removes occupancy confounder, and **scopes G3: shared ICs across variants ⇒ any G3 claim is "within-suite, shared-seed physics-response generalization", not structure generalization**; loader refactor ratified (regression-gated, stash-verified pre-existing test failure); R1-uncommitted accepted with **Caveat 1**: the six Stage-1 JSONs must be force-added (or relocated under design/records/) and the P2–P4 cubes DVC-tracked before Stage-2 close. **Caveat 2**: G1(c) re-scoped by amendment to the metrics actually used in masked scoring (r_s both frames + phase-rand null); masked NCCF does not exist and is not grandfathered — any future masked-NCCF readout must pass G1 first.

**R9 rulings:** (i) MLP verb of record: "the per-scene MLP's held-out-slab recovery is statistically consistent with zero (0.073 ± 0.094), spatially patchy (block r −0.446 to +0.397)" — NOT "memorization" (no train/test protocol existed; it fit flux over the full box); full-box 0.275 was never a generalization number; only masked-vs-masked comparisons admissible henceforth. (ii) Null = distribution, not the single draw (protocol in §(c)); the single-draw 0.1185 retired (wiener_L3 also sits below it — no conclusion from one realization). (iii) Operative G2 bars CONFIRMED: grid 0.5758 (block SE 0.0443) / wiener_L3 0.0863 (best-L unchanged on mask) / mlp 0.0727; five-condition machinery unchanged; 0.5954 full-box = context anchor only. Banked: grid held-out ≈ full-box (|Δ| ≤ 0.02 all σ) — spatially uniform recovery, legitimating it as the slab ceiling.

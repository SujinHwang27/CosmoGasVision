# Sprint-4 design — 3D ResNet truth-baseline classifier ([D-47] option-C step 1)

**Status**: 🛠 Draft → implementation, 2026-05-12, main-thread (core-implementer dispatch with main-thread fallback per the sprint-1/2/3 permission pattern).
**Predecessor**: Sprint-3 [D-50] CIC chunked-scatter refactor (HEAD `1513999`). Builds on sprint-1 [D-48] disk cache (HEAD `923458f`) and sprint-2 [D-49] held-out region split (HEAD `4ff68fe`).
**Blocking**: sprint-5 ([D-47] option-C step 2 — reconstructed-baseline classifier) cannot dispatch until sprint-4 closes; sprint-5 reuses this sprint's architecture + training protocol verbatim with `crops` swapped from truth-CIC to NeRF-sampled.
**Downstream**: [D-47] gap measurement Δ̂(r) = Â_truth(r) − Â_recon(r); [D-15] empirical 85% bar lower-bounded by Â_truth(r).
**Decision number**: [D-51] (PENDING in §3 at dispatch; DONE block appended at sprint close).

---

## 1. Why this sprint exists

Per **[D-47]** option-C hybrid Stage 3 framing: the headline scientific quantity is the **gap** between (i) a 3D classifier's accuracy on *truth* ρ crops and (ii) the same classifier on *reconstructed* (NeRF-sampled) ρ crops, conditional on distance-to-train-region r. The truth-baseline (this sprint) supplies (i) — the empirical ceiling Â_truth(r). The reconstructed-baseline (sprint-5) supplies (ii). The gap is the pre-registered Stage 3 [D-15]-bar measurement.

Per **[D-12]** + **[D-46] Addendum 1** + sprint-2 [D-49]: the classifier evaluates on the **held-out region only** (val ∪ test). This sprint inherits the [D-49] strict-rejection straddle policy and the periodic 1D distance metric; it does not change the split scheme.

The truth-baseline by itself is **not a paper claim** — it's a measurement instrument. Its purpose is to anchor the gap. Its required property is *trustworthy measurement protocol*, not *high accuracy*.

## 2. Implicit-NeRF vs. voxel-substrate boundary

CosmoGasVision is a NeRF project: the IGM-NeRF F_θ(**x**) → (ρ, T, X_HI, v_pec) is *implicit*, supervised against τ(v) sightlines through the differentiable Voigt integrator. During NeRF training, the loss is computed at sample points along sightlines — there is no voxel grid in the loss path.

This sprint and the next operate at a different layer: **the Stage 3 classifier consumes voxel grids by architectural choice** (3D CNN locality + translation-equivariance are the right inductive biases for "which feedback variant did this small volume come from?"). The voxel grid is the *measurement substrate*, not the model representation. The pipeline:

- **Truth side (this sprint)** — Sherwood ground-truth ρ field is *natively* on a 768³ voxel grid (CIC-deposited from the particle table; [D-48] cache + [D-50] chunked-scatter deposition). Crop → 3D ResNet → Â_truth(r).
- **Reconstructed side (sprint-5)** — the trained NeRF F_θ will be **sampled on a 768³ voxel grid** (one forward pass per voxel center), producing a rasterized NeRF ρ field of the same shape as truth. Same crop pipeline, same classifier → Â_recon(r). F_θ remains implicit; the voxelization is just discretization for downstream measurement, analogous to marching-cubes on an implicit SDF.

The gap Δ̂(r) is apples-to-apples because both sides feed the same classifier the same input format. Sprint-4 itself **does not touch NeRF** — it operates entirely on the CIC truth grid.

## 3. Architecture — 3D ResNet-18 (4-class head)

**Backbone**: ResNet-18 3D — BasicBlock × {2,2,2,2} with 3D conv/BN/ReLU; channel widths halved from the video-defaults to fit the cubic single-channel scalar-field input:

| Stage | Channels | Output spatial (crop=32) |
|---|---|---|
| stem (3D conv k=7, stride=2) | 32 | 16³ |
| block1 (×2, stride=1) | 32 | 16³ |
| block2 (×2, first stride=2) | 64 | 8³ |
| block3 (×2, first stride=2) | 128 | 4³ |
| block4 (×2, first stride=2) | 256 | 2³ |
| global avg pool (3D) + FC(4) | 4 | — |

Target parameter budget **8–12M**. If the halved-channel variant lands <8M, bump block3/block4 to 192/384.

**File**: NEW `src/models/cnn3d.py` exporting `class ResNet3D` + factory `resnet18_3d_4class(in_channels=1, num_classes=4) -> nn.Module`. No edits to `src/models/nerf.py`.

## 4. Crop size — 32 voxels = 2.5 h⁻¹ Mpc

At n_grid=768, voxel side = 60/768 = 0.078 h⁻¹ Mpc.

| crop | physical extent | comment |
|---|---|---|
| 16 | 1.25 h⁻¹ Mpc | **below** the [D-13] ξ_{ρ̂,ρ}(r=2 h⁻¹ Mpc) gate radius; risks the classifier solving from sub-gate features |
| **32** | **2.5 h⁻¹ Mpc** | **matches** the [D-13] gate scale; classifier sees the same spatial regime the reconstruction is judged on |
| 64 | 5.0 h⁻¹ Mpc | exceeds gate scale (fine scientifically) but: doubles VRAM, drops sprint-2 acceptance rate from ~11% to ~6.7% |

**Locked**: `crop_size = 32`. Ablation across {16, 32, 64} deferred to follow-up if Â_overall fails gate (a) sanity floor (<0.50) — in that branch the natural next move is to test scale-dependence before redesigning the classifier.

## 5. Training config

| Knob | Value | Source |
|---|---|---|
| Optimizer | AdamW(β=(0.9, 0.999), wd=1e-4) | standard 3D ResNet recipe |
| LR schedule | warmup 0→3e-4 over 1 epoch, cosine 3e-4→3e-6 over total | classifier regime; matches the shape of [D-14]'s schedule |
| Batch size | 16 crops × 4 physics interleaved = **64 effective** (4-class balanced per batch); fallback bs=32 + grad-accum 2 if VRAM-limited | crop=32 fp32 occupies ~131 KiB; bs=64 ≈ 8 MiB activations + ~50 MiB conv intermediates |
| Epochs | 30, early-stop on val-loss plateau, patience=5 | classifier convergence on N≈20k samples is typically 20–40 epochs |
| Augmentation | random 3D axis flip (×3 axes, p=0.5 each) + random 90° rotation about a random axis (p=0.5). **No scaling/cropping/noise.** | sim-data is exact; only valid augmentations are isometries of the periodic box (octahedral group of cubic symmetry) |
| Regularization | wd=1e-4 (above); dropout=0.0 (BN provides regularization) | standard 3D ResNet practice |
| Loss | `nn.CrossEntropyLoss` on 4-way logits + integer physics_id labels (0..3 ← {P1, P2, P3, P4}) | per [D-47] step-1 spec |
| Split | [D-49] default: `axis=0, train_x_max=0.7, val_x_max=0.85` (`HeldoutSplitScheme()`) | sprint-2 default; no revision |
| Determinism | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic data-loader workers | gate (c) end-to-end determinism |

**Augmentation correctness**: augmentations operate on the (D, H, W) crop **after** it is drawn from the split-sampler, not on the underlying full ρ field. This preserves the [D-49] crop-determinism contract — the same `(seed, scheme)` produces the same crop *before augmentation*, and the augmentation pipeline is independently seeded per-sample.

## 6. Datasets — counts, seeds, sampling

Per physics × per region:

| split | n_crops per physics | total (× 4 physics) | source region call |
|---|---|---|---|
| train | 5,000 | 20,000 | `region="train"`, `seed=42` |
| val | 1,000 | 4,000 | `region="val"`, `seed=142` |
| test | 2,000 | 8,000 | `region="test"`, `seed=242` |

Test draws are made with `region="test"` (not `"heldout"`) so the returned `distance` reflects pure distance-to-train-edge, not a mix of val + test distances.

**Class balance**: equal counts per physics → already balanced. `CrossEntropyLoss` is unweighted.

**Sampling**: rejection-sampling per [D-49] (deterministic on `seed`). No grid-aligned sampling.

**Crop budget feasibility**: 4 × (5k + 1k + 2k) = 32k crops × ~6 ms/crop (post-[D-48] cache) ≈ **~3 min crop extraction** total. CIC at n_grid=768 amortized across 4 physics via [D-50] disk cache (one-time ~5.5 min cold per physics; subsequent calls are 1–2 s warm hits).

## 7. Headline metric — pre-registered Â(r) estimator

**Definition** (per [D-47]): Â(r) = conditional classification accuracy on test crops whose `distance_to_train_region` (returned per-crop by `extract_rho_crops_split`) falls in the r-bin around r.

**r-binning policy — equal-occupancy quintiles, val-set-fixed edges**:

1. After training completes, run the trained classifier on the *val set* and record the distance distribution `d_val[0..3999]`.
2. Compute 5 quintile edges from `d_val`: `q = [d_val.min(), p20, p40, p60, p80, d_val.max()]` (where p_k is the k-th percentile).
3. Write `q` to `experiments/nerf/artifacts/eval/sprint4/r_bin_edges.json` with timestamp + sha256 of the array. **This file MUST be written before any test-set prediction is logged.**
4. Run the trained classifier on the test set. For each test crop, bin by its `distance` using `q`. Accuracy within each bin is Â(quintile_k).
5. Report headline triplet Â(r_25), Â(r_50), Â(r_75) at the quintile-2/3/4 bin centers (i.e., bins containing the empirical 25th, 50th, 75th percentile points).

**Why val-set-fixed edges, not test-set**: classic pre-registration anti-degeneracy. Re-binning on test would allow (intentionally or not) bin-shaping that flatters Â. Locking edges to val (which is split-disjoint from test) breaks that degeneracy. The audit trail is the file mtime + sha256 logged before test inference.

**Bootstrap CI policy — pre-committed**:
- Resamples: **N = 1,000**
- Resample unit: **the crop** (not the sightline — there are no sightlines in this measurement; the input is a 3D volume). Distinct from the [D-44] sightline-unit convention which applies to P_F.
- α = 0.05 → 95% CI from [0.025, 0.975] quantiles of the bootstrap distribution.
- Resampling is per-r-bin: for bin k, resample with replacement from the N_k test crops in bin k, recompute accuracy, repeat 1,000 times.
- **Block bootstrap deferred** (sprint-2 §10 explicitly defers it; requires pilot correlation-length measurement on Sherwood ρ field). Ordinary bootstrap noted as a known approximation; block-bootstrap is a follow-up before paper publication of the gap.

**Output artifact shape** (one JSON file `experiments/nerf/artifacts/eval/sprint4/headline.json`):
```json
{
  "Â_overall": {"mean": ..., "ci_low": ..., "ci_high": ...},
  "Â_quintile_1": {"r_center": ..., "n_crops": ..., "mean": ..., "ci_low": ..., "ci_high": ...},
  "Â_quintile_2": {...},
  "Â_quintile_3": {...},
  "Â_quintile_4": {...},
  "Â_quintile_5": {...},
  "r_bin_edges": [...],
  "r_bin_edges_sha256": "...",
  "r_bin_edges_written_at_utc": "..."
}
```

## 8. Pre-committed gate — sprint-4 DONE iff ALL PASS

Mirrors the [D-50] 5-criterion table.

| # | Gate | Threshold | Interpretation of failure |
|---|---|---|---|
| **(a)** | **Sanity floor on overall test accuracy** | Â_overall 95% CI lower bound > 0.50 (= 2× the chance baseline of 0.25) | Data-pipeline bug (label-crop misalignment), augmentation bug (labels leak through transforms), or crop=32 is too small to capture the physics signature. Investigate; sprint blocked. |
| **(b)** | **r_50 well-definedness** | Test set has ≥ 200 crops in the quintile containing r_50; bootstrap CI half-width at quintile-3 < 0.05 (5 percentage points) | Procedural thinness; raise `n_crops_test` until the gate clears. Not a science blocker. |
| **(c)** | **Split-determinism end-to-end** | Two independent runs at `(seed_train=42, seed_val=142, seed_test=242, scheme=DEFAULT_SCHEME)` produce: (i) byte-identical train/val/test crop sets per `np.array_equal`, (ii) bit-identical model predictions to fp32, (iii) Â at all 5 quintiles identical to 1e-7 | If non-deterministic, [D-12] anti-leakage audit hook is broken. Critical fix-blocker. |
| **(d)** | **Â(r) smoothness / no-wild-oscillation** | The 5-quintile Â(r) curve is monotone, OR varies by < 0.10 (10 percentage points) across all 5 bins. No sign pre-committed for monotonicity — truth-baseline is *expected* to be roughly r-invariant since truth ρ has no train/test asymmetry. | Spikes (e.g., quintile-2→quintile-3 drops 30 pp, quintile-3→quintile-4 rises 30 pp) flag data leakage through the boundary. Sprint blocked, escalate to PI. |
| **(e)** | **Trivial-baseline backstop** — **TWO baselines, both must trail the 3D ResNet by ≥ 5 pp** | (e₁) **mean-overdensity baseline**: 1-scalar `crop.mean()` → FC(4). Must achieve Â_overall ≤ Â_overall(ResNet) − 0.05. (e₂) **mean+variance baseline**: 2-scalar `[crop.mean(), crop.var()]` → FC(4). Must achieve Â_overall ≤ Â_overall(ResNet) − 0.05. | If either trivial baseline matches the 3D ResNet within 5 pp, the task is dominated by low-order moments and the 3D ResNet is measuring a low-dimensional summary statistic, not 3D structure. The [D-47] gap measurement on the 3D-structure axis is then decorative. **Anti-degeneracy hook explicit per [D-37]-ext rule 3.** |

**Distinguishing success metric from gate criteria**:
- **Success metric** = the reported Â(r_25), Â(r_50), Â(r_75) triplet with CIs. The [D-15] empirical ceiling — there is no pre-set "pass/fail bar" on the value itself; whatever the truth-baseline gets is the ceiling.
- **Gate criteria** (a)–(e) = procedural correctness of the measurement. The sprint is DONE when the measurement is trustworthy, not when the number is high.

## 9. Anti-degeneracy hooks

Inheriting [D-37]-ext rule 3 discipline: "what does this measurement leave unconstrained when the obvious headline number is high?"

| # | Failure mode | How caught | Where |
|---|---|---|---|
| AD-1 | **Periodic-boundary leakage** — train crops near `x=0` are spatially adjacent (mod n_grid) to test crops near `x=1.0`; periodic wrap could leak the same voxels into both | Already prevented by sprint-2 [D-49] strict-rejection straddle policy. **Add unit test** in NEW `tests/test_split_anti_leakage.py` confirming no test-set crop's voxel index range intersects any train-set crop's range modulo n_grid. | `tests/test_split_anti_leakage.py` |
| AD-2 | **Crop-edge artifact identification** — the network solves the task by reading padding artifacts at the cube boundary | Per-physics 4×4 confusion matrix as a required artifact. If off-diagonal accuracy is asymmetric in a way correlated with crop sampling location, edge artifacts are the cause. | `experiments/nerf/artifacts/eval/sprint4/confusion_matrix.png` + `confusion_matrix.json` |
| AD-3 | **Trivial-statistic identification** — 4 physics distinguishable from low-order moments alone | Gate (e₁) mean-overdensity baseline + (e₂) mean+variance baseline. Both must trail the 3D ResNet by ≥ 5 pp. | Implementation in `src/analysis/conditional_accuracy.py`; outputs `baseline_mean_overdensity.json` + `baseline_mean_variance.json` |
| AD-4 | **Augmentation label leakage** — buggy per-physics augmentation seed leaks rotation distribution as a label proxy | Augmentation RNG seeded per-sample, independent of label. NEW unit test confirms `random.choice` of transform parameters is invariant across labels at fixed sample-index. Also implicitly caught by gate (e): a 1-scalar baseline is rotation-invariant, so if it fails the inequality, augmentation leakage cannot be the cause. | `tests/test_augmentation_label_independence.py` |

## 10. Hedged framing — verbs to use and avoid

When the [D-51] DONE block is written:

✅ "We train a 3D ResNet-18 classifier as a measurement instrument for the [D-47] empirical ceiling. The conditional accuracy Â(r) is reported per the pre-registered estimator. The truth-baseline yields ceiling Â_overall = X.XX [CI]; this anchors the [D-15] 85% bar."

❌ Avoid: "state-of-the-art 3D classifier", "the classifier successfully identifies feedback physics", "our 3D ResNet achieves X% accuracy" — verbs that promote the instrument to a contribution.

⚠️ Specifically: if Â(r) comes out >0.95 across all r, the temptation will be to frame this as "truth ρ is highly classifiable" — which is true but trivial. The **honest** framing per [D-37] rule 5 is symmetric: the high accuracy is *necessary* (otherwise the [D-47] gap measurement has no signal) but *not sufficient* for any Stage 3 conclusion. Sprint-5 (reconstructed-baseline) is what closes the scientific question.

## 11. API surface, files, dispatch plan

**New files**:

| File | Purpose | LOC budget |
|---|---|---|
| `src/models/cnn3d.py` | `class ResNet3D`, factory `resnet18_3d_4class`, 3D BasicBlock | ~250 |
| `src/analysis/conditional_accuracy.py` | `compute_conditional_accuracy(predictions, labels, distances, r_bin_edges)`, `bootstrap_accuracy_ci`, `compute_quintile_edges`, `eval_trivial_baselines` | ~300 |
| `experiments/nerf/pipeline.py` | (extend existing) — new function `run_sprint4_truth_baseline(...)` that wires data → model → train → eval → artifacts | ~200 added |
| `tests/test_cnn3d.py` | shape contract, parameter count, forward determinism, augmentation correctness | ~150 |
| `tests/test_conditional_accuracy.py` | bin-edge determinism, bootstrap reproducibility, trivial-baseline correctness on synthetic inputs | ~200 |
| `tests/test_split_anti_leakage.py` (AD-1) | confirm no train/test voxel-index intersection mod n_grid | ~80 |
| `tests/test_augmentation_label_independence.py` (AD-4) | augmentation RNG independence | ~80 |

**Output artifacts** (DVC-track if > 10 MB):

- `experiments/nerf/artifacts/sprint4/checkpoints/resnet18_3d_4class_best.pt` — trained model (~50 MB, DVC-tracked)
- `experiments/nerf/artifacts/eval/sprint4/r_bin_edges.json` (pre-test-inference write)
- `experiments/nerf/artifacts/eval/sprint4/headline.json` (Â triplet + CIs)
- `experiments/nerf/artifacts/eval/sprint4/confusion_matrix.png` + `.json`
- `experiments/nerf/artifacts/eval/sprint4/baseline_mean_overdensity.json`
- `experiments/nerf/artifacts/eval/sprint4/baseline_mean_variance.json`
- `experiments/nerf/artifacts/eval/sprint4/training_log.csv` (per-epoch loss + val acc)

**MLflow contract** (per `.claude/skills/mlflow-run`):
- Experiment: `CosmoGasVision/Stage3-TruthBaseline`
- Run name: `Sprint4-TruthBaseline-3DResNet18`
- Mandatory tags: `model_type=resnet18_3d`, `stage=Stage3-truth-baseline`, `physics_id=all`, `redshift=0.300`

**Owner**: core-implementer subagent (primary); main-thread fallback per the sprint-1/2/3 deny pattern. The test files are write-OK; subagent denies on `src/models/` are unlikely but the fallback is well-rehearsed.

**Defense-panel review**: **DEFERRED** per [D-37]-ext rule 6 with explicit annotation — the truth-baseline is an instrument, not a paper-claim surface. Panel review will be MANDATORY before sprint-5 dispatch (which produces the Δ̂(r) gap that anchors the paper claim). This sprint's r-binning policy + bootstrap-unit choice + 5-gate spec are pre-registered in this document, so sprint-5 panel review cannot retroactively re-shape the protocol. Annotation is recorded in the [D-51] entry's "Review trail".

**Smoke gate (5-step memory smoke)**: run 5 SGD steps on bs=64 crops with `torch.cuda.max_memory_allocated()` tracking. Pass = no OOM and loss decreasing across 5 steps. Cost ~30 s on local GPU.

**Wallclock**:
- Implementation: 4–6 hr (model + analysis + tests + pipeline driver)
- Smoke: 30 s
- Full train: ~2.5 hr on local 16 GB GPU; ~1 hr on `ml.g5.xlarge` if cloud-dispatched
- Bootstrap eval: ~5 min CPU after training
- **Sprint total**: one working session, host-side
- **$ cost**: $0 local; ~$1.20 if SageMaker spot

## 12. Carry-forward + references

**Carry-forward** (to be appended to [D-51] DONE block when sprint closes):
- Block-bootstrap on Â(r) before paper publication of Δ̂(r) (sprint-5+; requires pilot correlation-length on Sherwood ρ)
- Optional: crop-size ablation across {16, 32, 64} if gate (a) fails or for paper-revision robustness check
- Optional: axis ablation (`axis=1, 2`) on the [D-49] split to falsify "split direction doesn't change the headline"

**References**:
- [D-12]: cross-physics protocol + anti-leakage rule
- [D-13]: ξ_{ρ̂,ρ}(r) gate definition (sets the 2 h⁻¹ Mpc anchor for crop_size)
- [D-15]: Stage 3 85% bar
- [D-37]-ext: hedged framing (rule 2), anti-degeneracy audit (rule 3), honest reporting (rule 5), deferred-panel annotation (rule 6)
- [D-44]: bootstrap convention (sightline-unit; distinct from this sprint's crop-unit)
- [D-46]: physics_id-conditioned reconstruction MLP (sprint-5+ consumer)
- [D-47]: option-C hybrid Stage 3 framing; truth-baseline = step 1, reconstructed-baseline = step 2
- [D-48]: ρ-field disk cache (sprint-1; enables fast multi-physics crop draws)
- [D-49]: held-out region split (sprint-2; this sprint's data partition)
- [D-50]: CIC chunked-scatter refactor (sprint-3; enables n_grid=768 deposition)

**Predecessor HEADs**:
- Sprint-1 [D-48] code: `923458f`
- Sprint-1 [D-48] LEDGER: `14d51c2`
- Sprint-2 [D-49] code+design+tests: `4ff68fe`
- Sprint-2 [D-49] paper fixes: `0d72b12`
- Sprint-3 [D-50] code+ledger: `1513999`

Sprint-4 commit will reference `1513999` in its `Predecessor:` line.

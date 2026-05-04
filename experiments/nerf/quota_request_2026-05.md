# AWS Compute Quota Request — CosmoGasVision NeRF Track

**Date prepared**: 2026-05-04
**Project**: CosmoGasVision — 3D Intergalactic-Medium gas-density reconstruction from Lyman-alpha sightlines (Sherwood Simulation Suite, Bolton+ 2017)
**Track**: NeRF (continuous MLP IGM field; primary methodology)
**Target cosmology**: redshift z = 0.3, 60 Mpc/h comoving box, 4 feedback-physics variants
**Submission target**: CVPR (paper draft in `paper_cvpr/`, methods + experiments sections current as of 2026-05-04)
**Region**: `us-east-1`
**AWS account**: same as `cgv-infrastructure-user` IAM principal currently in use

---

## 1. Summary of request

We request the following service-quota actions on AWS SageMaker training (region `us-east-1`):

| Quota dimension | Current limit | Requested limit | Priority |
|:---|---:|---:|:---|
| `ml.g5.xlarge` for training job usage (**spot**) | 0 (request submitted, pending) | **4** | Required (blocks Tier 4) |
| `ml.g5.xlarge` for training job usage (**on-demand**) | 4 | **8** (uplift, optional) | Nice-to-have (burst headroom) |

The on-demand uplift is optional. The spot quota is the gating item.

---

## 2. What we have spent and validated

All numbers below are reconciled against `aws sagemaker describe-training-job` outputs and the project's MLflow tracker; full traceability is in `experiments/nerf/LEDGER.md` §6 / §6.5 / §6.6.

- **Verified spend on `ml.g5.xlarge` on-demand to date**: **~\$21.24** (~21.1 GPU-hours).
- **Validated artifact**: a 16-cell micro-grid (`stage=2b-microsweep-d24`, 4 physics × 4 sightline densities) plus per-physics tier-1 production runs at 50,000 optimization steps. Loss descent ratios in [0.558, 0.749] (vs ≤0.85 pass criterion), no NaN/Inf, `mean_F` consistent across 16 cells in [0.9238, 0.9302] vs the Danforth+ 2016 anchor target 0.877.
- **VRAM linearity model validated** across `accum_steps ∈ {1, 4, 64}`: peak GPU memory 2.82 / 11.20 / 11.23 / 11.77 GB at tiers 1-4, all comfortably under the A10G 24 GB budget.
- **Throughput anchor**: 0.119 s/step at tier 1 (n_rays = 64), measured across 5 independent 50k-step runs (P1 production + 4 cost-survey cells; spread <0.5%).

The implementation is physics-defensible (logged decisions D-06 through D-24 in the LEDGER, including a 2026-05-04 defense-panel verification cascade), the cloud contract is closed (SageMaker + ECR + IAM + GPU + MLflow round-trip), and we have direct measurements for every parameter that drives cost.

---

## 3. What we are requesting and why

The CVPR-headline contribution is the **survey-headline degradation curve**: reconstruction quality as a function of sightline density across `n_rays ∈ {16384, 1024, 256, 64}`, repeated for four cosmological-feedback variants (no feedback / stellar wind / wind+AGN / wind+strong-AGN). 16 production cells; 12 already feasible under the existing on-demand quota; 4 blocked.

### Per-tier production cost (extrapolated from validated anchors)

| Tier | n_rays | wall-clock per cell | on-demand $/cell | spot $/cell | × 4 physics on-demand | × 4 physics spot |
|:---|---:|---:|---:|---:|---:|---:|
| T1 | 64 | 1.65 hr | \$1.66 | \$0.50 | \$6.64 (already spent) | — |
| T2 | 256 | 2.78 hr | \$2.80 | \$0.84 | **\$11.20** | \$3.36 |
| T3 | 1024 | 4.31 hr | \$4.34 | \$1.30 | **\$17.34** | \$5.20 |
| T4 | 16384 | 8.25 days | \$199 | \$60 | (impractical) | **\$239** |

**T4 on-demand is operationally untenable** (single-cell wall-clock >8 days; on-demand pricing pushes the 4-physics T4 sweep to ~\$800). Spot pricing on the same instance family makes the same sweep ~\$240 and brings the wall-clock inside a CVPR-revision cycle through 4-way parallelism.

### Forward budget envelope

- **Forward minimum** (Batch 2 + Batch 3, on-demand, no T4): **\$28.54**
- **Forward complete** (Batch 2 + Batch 3 + Tier 4 on spot): **\$247.56**
- **Recommended ceiling with 50% contingency** for re-runs and post-review revisions: **\$800-\$1,000**

The contingency is sized to absorb (a) one micro-grid re-run if a methodological amendment lands during peer review, (b) the cosmological-evaluator validation runs (P_F(k), Pearson cross-correlation, flux-PDF KS), and (c) one round of post-fix re-dispatches.

---

## 4. Justification

The four T4 cells are the **only** rows of the [D-13] degradation curve that exercise the dense-sightline regime — without them the headline plot has a missing column at every physics, which removes the paper's central scaling claim. The cost is governed by `n_rays × accum_steps`; we have already proven the implementation fits the A10G memory budget at this scale (micro-grid T4: peak 11.77 GB / 24 GB), so this is purely a quota constraint, not a hardware or implementation question.

The `ml.g5.xlarge` spot family was selected because: (i) on-demand at this tier is operationally infeasible on a CVPR timeline, (ii) the four cells are independent (one per physics seed), so spot interruptions are recoverable with checkpoint resume (already implemented; checkpoints persisted to `s3://cosmo-gas-vision-storage/stage2b-checkpoints/` every 10k / 25k steps), and (iii) the same instance family is the validated platform for tiers 1-3, eliminating cross-instance numerical-equivalence risk.

---

## 5. Operational safeguards in place

- **Cost ceiling**: per-tier `max_steps` schedule pinned in [D-23]; runs cannot exceed the schedule without a documented amendment.
- **Checkpointing**: every 10,000 / 25,000 / 50,000 steps to S3.
- **Memory safety**: the [D-23] sub-clause gate requires every new microbatch value to predict peak VRAM under 21.6 GB before dispatch; one prior failure (microbatch math error, ~\$0.04 sunk) is documented in the LEDGER.
- **Audit trail**: every billable second is reconciled against `aws sagemaker describe-training-job` and recorded in `experiments/nerf/LEDGER.md` §6.6.

---

## 6. Contact

[Name], [Role]
[Email]
GitHub branch: `exp/nerf` at this repository

Supporting artifacts available on request: the 16-cell micro-grid matrix, the τ_max sensitivity gate (PASS at ~100-180× margin), the bring-up bug log, and the full decision log [D-01]…[D-24].

"""Compute gate-5 absorption statistics for [D-56] sprint-5 (c') from per-crop JSONLs.

Outputs:
  - sigma_seed_resnet_emp, sigma_seed_mvsk_emp     (std across 5 seed accuracies)
  - rho_emp                                        (within-seed mean paired ResNet-vs-MVSK indicator correlation)
  - rho_seed_emp                                   (mean pairwise between-seed correlation of ResNet correctness)
  - AD-5 margin bootstrap 95% CI (per-seed margins, n=5 bootstrap)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path("cloud_runs/Sprint5cprime-48cube-5seed-8992c12-20260516-111221-d8928b/eval")
SEEDS = [42, 142, 242, 342, 442]

# Per-seed indicator vectors
resnet = {}
mvsk = {}
for s in SEEDS:
    rs, ms = [], []
    with open(ROOT / f"per_crop_seed_{s}.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            rs.append(int(rec["resnet_correct"]))
            ms.append(int(rec["mvsk_correct"]))
    resnet[s] = np.array(rs, dtype=np.int8)
    mvsk[s]   = np.array(ms, dtype=np.int8)

# Per-seed accuracies (sanity vs headline)
per_seed_acc_resnet = np.array([resnet[s].mean() for s in SEEDS])
per_seed_acc_mvsk   = np.array([mvsk[s].mean() for s in SEEDS])
print(f"per_seed_acc_resnet: {per_seed_acc_resnet}")
print(f"per_seed_acc_mvsk:   {per_seed_acc_mvsk}")
print(f"seed_avg_resnet:     {per_seed_acc_resnet.mean():.6f}")
print(f"seed_avg_mvsk@48:    {per_seed_acc_mvsk.mean():.6f}")

# sigma_seed_emp = sample std across seed-level accuracies
sigma_seed_resnet_emp = per_seed_acc_resnet.std(ddof=1)
sigma_seed_mvsk_emp   = per_seed_acc_mvsk.std(ddof=1)
print(f"\nsigma_seed_resnet_emp (sample std n=5): {sigma_seed_resnet_emp:.6f}")
print(f"sigma_seed_mvsk_emp   (sample std n=5): {sigma_seed_mvsk_emp:.6f}")

# rho_emp = within-seed mean Pearson correlation between resnet-correct and mvsk-correct, averaged over seeds
within_corrs = []
for s in SEEDS:
    r, m = resnet[s].astype(float), mvsk[s].astype(float)
    # guard against zero variance
    if r.std() == 0 or m.std() == 0:
        continue
    rho = np.corrcoef(r, m)[0, 1]
    within_corrs.append(rho)
    print(f"  seed {s}: within-seed rho(resnet, mvsk) = {rho:.6f}")
rho_emp = float(np.mean(within_corrs))
print(f"rho_emp (mean within-seed paired-classifier corr): {rho_emp:.6f}")

# rho_seed_emp = mean pairwise between-seed correlation of resnet indicators (10 pairs)
between_corrs = []
for i, si in enumerate(SEEDS):
    for sj in SEEDS[i+1:]:
        r_i, r_j = resnet[si].astype(float), resnet[sj].astype(float)
        rho = np.corrcoef(r_i, r_j)[0, 1]
        between_corrs.append(rho)
        print(f"  pair ({si},{sj}): between-seed rho(resnet_i, resnet_j) = {rho:.6f}")
rho_seed_emp = float(np.mean(between_corrs))
print(f"rho_seed_emp (mean pairwise between-seed corr, 10 pairs): {rho_seed_emp:.6f}")

# AD-5 margin per seed and bootstrap CI
margins = per_seed_acc_resnet - per_seed_acc_mvsk
print(f"\nper_seed_margins (pp): {margins * 100}")
print(f"mean margin: {margins.mean():.6f} ({margins.mean()*100:.2f} pp)")
print(f"sample std of margin (n=5, ddof=1): {margins.std(ddof=1):.6f}")

# Bootstrap CI on seed-averaged margin (resample 5 seeds with replacement, B=10000)
rng = np.random.default_rng(20260516)
B = 10000
boot_means = np.empty(B)
for b in range(B):
    idx = rng.integers(0, 5, size=5)
    boot_means[b] = margins[idx].mean()
lo, hi = np.percentile(boot_means, [2.5, 97.5])
print(f"\nAD-5 margin 95% bootstrap CI (B={B}, resample 5 seeds w/ replacement):")
print(f"  point estimate: {margins.mean()*100:.2f} pp")
print(f"  95% CI: [{lo*100:.2f}, {hi*100:.2f}] pp")
print(f"  lower-CI > 10pp ? {lo*100 > 10}")
print(f"  lower-CI > 0   ? {lo*100 > 0}")

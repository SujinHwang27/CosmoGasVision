"""Â(r) conditional-accuracy estimator + crop-unit bootstrap + audit-trail.

Sprint-4 [D-51] implementation per the design doc at
``experiments/nerf/design/sprint4_truth_baseline.md`` \xa77 + \xa78.

The headline scientific quantity is the triplet Â(r_25), Â(r_50), Â(r_75)
with crop-unit bootstrap 95% CIs, conditional on
``distance_to_train_region``. Bin edges are equal-occupancy quintiles
computed on the **val** distance distribution and pre-committed to a
JSON artifact (with sha256 + timestamp) before test inference, to
foreclose post-hoc bin reshaping that would compromise the
[D-47]-pre-registered estimator.

This is the *measurement instrument* surface — NOT a paper claim. See
[D-37]-ext rule 2 and design doc \xa710 for verb discipline.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Number of equal-occupancy quintile bins; the headline triplet
# r_25/r_50/r_75 corresponds to the bin containing the 25th/50th/75th
# percentile of the val distance distribution (quintile 2/3/4 of 5).
N_QUINTILE_BINS = 5
HEADLINE_QUINTILE_INDICES = {25: 1, 50: 2, 75: 3}  # 0-indexed bin id

DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_ALPHA = 0.05


# -------------------------------------------------------- bin construction

def compute_quintile_edges(val_distances: np.ndarray) -> np.ndarray:
    """Equal-occupancy quintile edges from the val-set distance vector.

    Returns 6 floats: ``[min, p20, p40, p60, p80, max + eps]``. The final
    edge is nudged by a small epsilon to ensure the bin-assignment
    routine includes the maximum-distance sample in the final bin.

    Args:
        val_distances: 1D array of test-region distances from the val
            split. MUST come from val, not test, to satisfy the
            pre-registration discipline.
    """
    if val_distances.ndim != 1:
        raise ValueError(f"val_distances must be 1D, got shape {val_distances.shape}")
    if val_distances.size == 0:
        raise ValueError("val_distances is empty")
    if not np.isfinite(val_distances).all():
        raise ValueError("val_distances contains non-finite values")

    percentiles = np.percentile(
        val_distances, [0, 20, 40, 60, 80, 100], method="linear",
    )
    # Nudge the max edge so np.digitize includes the maximum sample
    # in the top bin. We use float32 precision as the nudge unit since
    # downstream distances may be float32.
    eps = max(float(np.finfo(np.float32).eps) * (1.0 + abs(percentiles[-1])), 1e-10)
    percentiles[-1] = percentiles[-1] + eps
    return percentiles.astype(np.float64)


def bin_indices_from_distances(
    distances: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Assign each distance to a bin in ``[0, N_QUINTILE_BINS)``.

    Distances below ``edges[0]`` are clamped to bin 0; distances at or
    above ``edges[-1]`` are clamped to ``N_QUINTILE_BINS - 1``. This
    correctly handles test-set distances that fall slightly outside the
    val-set empirical range without producing out-of-bounds indices.
    """
    if edges.size != N_QUINTILE_BINS + 1:
        raise ValueError(
            f"edges must have {N_QUINTILE_BINS + 1} entries, got {edges.size}"
        )
    # np.digitize with right=False returns bin index in [1, len(edges)-1]
    # for values in [edges[0], edges[-1]). We subtract 1 and clamp.
    bins = np.digitize(distances, edges[1:-1], right=False)
    # Clamp explicitly to valid range
    bins = np.clip(bins, 0, N_QUINTILE_BINS - 1)
    return bins.astype(np.int64)


def write_r_bin_edges_artifact(
    edges: np.ndarray,
    out_path: Path,
    run_metadata: Optional[dict] = None,
) -> str:
    """Write the pre-committed r_bin_edges.json artifact.

    Returns the sha256 of the edges array (as the canonical audit-trail
    digest). The file mtime + the returned sha256 are the inputs to the
    pre-registration check at gate evaluation time.

    Should be called BEFORE any test-set prediction is logged, so that
    a defense-panel review can verify the file's mtime precedes the
    test-eval timestamp.
    """
    edges_bytes = np.asarray(edges, dtype=np.float64).tobytes()
    sha256 = hashlib.sha256(edges_bytes).hexdigest()
    payload = {
        "edges": [float(e) for e in edges],
        "edges_sha256": sha256,
        "n_bins": N_QUINTILE_BINS,
        "headline_quintile_indices": HEADLINE_QUINTILE_INDICES,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": (
            "Sprint-4 [D-51] equal-occupancy quintile edges from val-set "
            "distance distribution per design doc \xa77 / [D-47] estimator."
        ),
    }
    if run_metadata is not None:
        payload["run_metadata"] = run_metadata
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return sha256


def load_r_bin_edges_artifact(in_path: Path) -> tuple[np.ndarray, str]:
    """Load the pre-committed edges + verify the recorded sha256 matches
    a fresh hash of the loaded array. Returns ``(edges, sha256)`` on
    success; raises ``RuntimeError`` if the digest mismatches (this is
    the audit-trail integrity check)."""
    with open(in_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    edges = np.asarray(payload["edges"], dtype=np.float64)
    recorded_sha = payload["edges_sha256"]
    fresh_sha = hashlib.sha256(edges.tobytes()).hexdigest()
    if fresh_sha != recorded_sha:
        raise RuntimeError(
            f"r_bin_edges.json integrity check failed: recorded sha256 "
            f"{recorded_sha} != fresh sha256 {fresh_sha}"
        )
    return edges, recorded_sha


# ---------------------------------------------------- accuracy estimator

@dataclass(frozen=True)
class BinResult:
    bin_index: int
    n_crops: int
    accuracy: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    r_center: Optional[float] = None  # geometric center of the bin

    def as_dict(self) -> dict:
        d = {
            "bin_index": self.bin_index,
            "n_crops": self.n_crops,
            "accuracy": self.accuracy,
        }
        if self.ci_low is not None:
            d["ci_low"] = self.ci_low
        if self.ci_high is not None:
            d["ci_high"] = self.ci_high
        if self.r_center is not None:
            d["r_center"] = self.r_center
        return d


def compute_conditional_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    edges: np.ndarray,
) -> dict:
    """Point estimate of Â per bin + overall (no bootstrap CI).

    Args:
        predictions: 1D array of argmax predictions per crop.
        labels: 1D array of ground-truth labels per crop.
        distances: 1D array of crop distances-to-train-region.
        edges: ``N_QUINTILE_BINS + 1`` bin edges from
            ``compute_quintile_edges`` on the val set.
    """
    if not (predictions.shape == labels.shape == distances.shape):
        raise ValueError(
            f"predictions/labels/distances shape mismatch: "
            f"{predictions.shape} / {labels.shape} / {distances.shape}"
        )
    bins = bin_indices_from_distances(distances, edges)
    correct = predictions == labels

    per_bin: list[BinResult] = []
    for k in range(N_QUINTILE_BINS):
        mask = bins == k
        n_k = int(mask.sum())
        if n_k > 0:
            acc_k = float(correct[mask].mean())
            r_center = float(0.5 * (edges[k] + edges[k + 1]))
        else:
            acc_k = float("nan")
            r_center = float(0.5 * (edges[k] + edges[k + 1]))
        per_bin.append(BinResult(
            bin_index=k, n_crops=n_k, accuracy=acc_k, r_center=r_center,
        ))

    overall = BinResult(
        bin_index=-1,
        n_crops=int(correct.size),
        accuracy=float(correct.mean()),
    )
    return {
        "overall": overall.as_dict(),
        "per_bin": [b.as_dict() for b in per_bin],
    }


def bootstrap_accuracy_ci(
    predictions: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    edges: np.ndarray,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> dict:
    """Crop-unit bootstrap of Â per bin + overall.

    Per design doc \xa77: resample with replacement from the test crops
    in each bin (and overall), recompute accuracy, repeat N_bootstrap
    times. Report the [α/2, 1-α/2] quantiles as the 95% CI.

    Distinct from the [D-44] sightline-unit bootstrap convention (P_F).
    Block-bootstrap deferred to follow-up; this is ordinary bootstrap.
    """
    if n_bootstrap < 100:
        raise ValueError(
            f"n_bootstrap={n_bootstrap} is too small; use >=100 (recommended 1000)"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    bins = bin_indices_from_distances(distances, edges)
    correct = (predictions == labels).astype(np.float64)
    rng = np.random.default_rng(seed)

    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    per_bin: list[BinResult] = []
    for k in range(N_QUINTILE_BINS):
        mask = bins == k
        n_k = int(mask.sum())
        r_center = float(0.5 * (edges[k] + edges[k + 1]))
        if n_k == 0:
            per_bin.append(BinResult(
                bin_index=k, n_crops=0,
                accuracy=float("nan"),
                ci_low=float("nan"),
                ci_high=float("nan"),
                r_center=r_center,
            ))
            continue
        bin_correct = correct[mask]
        acc_point = float(bin_correct.mean())
        # Bootstrap
        boot_samples = rng.choice(bin_correct, size=(n_bootstrap, n_k), replace=True)
        boot_accs = boot_samples.mean(axis=1)
        ci_low = float(np.quantile(boot_accs, q_lo))
        ci_high = float(np.quantile(boot_accs, q_hi))
        per_bin.append(BinResult(
            bin_index=k, n_crops=n_k,
            accuracy=acc_point, ci_low=ci_low, ci_high=ci_high,
            r_center=r_center,
        ))

    # Overall (marginal)
    overall_correct = correct
    n_overall = int(overall_correct.size)
    overall_point = float(overall_correct.mean())
    boot_samples_all = rng.choice(
        overall_correct, size=(n_bootstrap, n_overall), replace=True,
    )
    boot_accs_all = boot_samples_all.mean(axis=1)
    overall_ci_low = float(np.quantile(boot_accs_all, q_lo))
    overall_ci_high = float(np.quantile(boot_accs_all, q_hi))
    overall = BinResult(
        bin_index=-1,
        n_crops=n_overall,
        accuracy=overall_point,
        ci_low=overall_ci_low,
        ci_high=overall_ci_high,
    )
    return {
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "overall": overall.as_dict(),
        "per_bin": [b.as_dict() for b in per_bin],
    }


def headline_triplet(bootstrap_result: dict) -> dict:
    """Extract Â(r_25), Â(r_50), Â(r_75) from a bootstrap result.

    Returns a dict keyed by ``"25"``, ``"50"``, ``"75"`` -> per-bin dict.
    Quintile-2/3/4 map to the bins containing the 25th/50th/75th
    percentile of the val distance distribution.
    """
    per_bin = bootstrap_result["per_bin"]
    out = {}
    for p_label, bin_idx in HEADLINE_QUINTILE_INDICES.items():
        out[str(p_label)] = per_bin[bin_idx]
    return out


# ------------------------------------------------ trivial-baseline eval

def evaluate_trivial_baseline_accuracy(
    baseline_predictions: np.ndarray,
    baseline_labels: np.ndarray,
    resnet_overall_accuracy: float,
    name: str,
    margin_pp: float = 0.05,
) -> dict:
    """Apply gate (e) — the trivial baseline must trail the 3D ResNet
    by at least ``margin_pp`` (default 0.05 = 5 percentage points).

    Returns a dict with the baseline accuracy, the margin observed,
    and a boolean ``pass`` indicating whether the gate is satisfied.
    """
    baseline_acc = float((baseline_predictions == baseline_labels).mean())
    observed_margin = float(resnet_overall_accuracy - baseline_acc)
    return {
        "baseline_name": name,
        "baseline_accuracy": baseline_acc,
        "resnet_accuracy": resnet_overall_accuracy,
        "observed_margin_pp": observed_margin,
        "required_margin_pp": margin_pp,
        "pass": bool(observed_margin >= margin_pp),
    }

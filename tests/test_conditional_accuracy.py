"""Sprint-4 [D-51] unit tests for ``src/analysis/conditional_accuracy.py``.

Covers:
  - quintile edge construction from val distances
  - bin-index assignment determinism + boundary handling
  - bootstrap CI reproducibility under fixed seed
  - r_bin_edges.json audit-trail integrity (sha256 match)
  - trivial-baseline gate-(e) logic
  - headline triplet extraction

Run:
    PYTHONPATH=. uv run pytest tests/test_conditional_accuracy.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.conditional_accuracy import (
    HEADLINE_QUINTILE_INDICES,
    N_QUINTILE_BINS,
    bin_indices_from_distances,
    bootstrap_accuracy_ci,
    compute_conditional_accuracy,
    compute_quintile_edges,
    evaluate_trivial_baseline_accuracy,
    headline_triplet,
    load_r_bin_edges_artifact,
    write_r_bin_edges_artifact,
)


# ---------------------------------------------------------- bin construction

def test_quintile_edges_equal_occupancy_on_uniform_distribution():
    """Equal-occupancy on a uniform sample -> evenly-spaced edges."""
    rng = np.random.default_rng(0)
    distances = rng.uniform(0.0, 0.15, size=4000)
    edges = compute_quintile_edges(distances)
    assert edges.size == N_QUINTILE_BINS + 1
    # Each bin should contain ~800 samples (within a few %)
    bins = bin_indices_from_distances(distances, edges)
    counts = np.bincount(bins, minlength=N_QUINTILE_BINS)
    for k, n_k in enumerate(counts):
        assert 700 <= n_k <= 900, f"bin {k} has {n_k} samples; expected ~800"


def test_quintile_edges_rejects_non_finite_input():
    with pytest.raises(ValueError, match="non-finite"):
        compute_quintile_edges(np.array([0.0, 0.1, np.nan, 0.05]))


def test_quintile_edges_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        compute_quintile_edges(np.array([], dtype=np.float64))


def test_bin_indices_clamps_outside_range():
    """Test-set distances slightly outside the val range must be assigned
    to the nearest valid bin, not raise an out-of-bounds error."""
    rng = np.random.default_rng(1)
    val_d = rng.uniform(0.01, 0.10, size=200)
    edges = compute_quintile_edges(val_d)
    # Make some test distances outside the val range
    test_d = np.array([0.0, 0.005, 0.12, 0.50])
    bins = bin_indices_from_distances(test_d, edges)
    # 0.0 and 0.005 should clamp to bin 0
    assert bins[0] == 0 and bins[1] == 0
    # 0.12 and 0.50 should clamp to bin 4 (top)
    assert bins[2] == N_QUINTILE_BINS - 1
    assert bins[3] == N_QUINTILE_BINS - 1
    assert (bins >= 0).all() and (bins < N_QUINTILE_BINS).all()


def test_bin_indices_assignment_consistent_with_edges():
    """Spot-check: a distance equal to ``edges[k]`` lands in bin k."""
    edges = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.15])
    distances = np.array([0.0, 0.02, 0.03, 0.05, 0.06, 0.149])
    bins = bin_indices_from_distances(distances, edges)
    # 0.0 -> bin 0, 0.02 -> bin 0, 0.03 -> bin 1, 0.05 -> bin 1,
    # 0.06 -> bin 2, 0.149 -> bin 4
    expected = np.array([0, 0, 1, 1, 2, 4])
    np.testing.assert_array_equal(bins, expected)


# ------------------------------------------------------------ point accuracy

def test_compute_conditional_accuracy_perfect_predictions():
    """All correct -> 1.0 per bin + overall."""
    n = 1000
    rng = np.random.default_rng(0)
    distances = rng.uniform(0, 0.15, size=n)
    edges = compute_quintile_edges(distances)
    labels = rng.integers(0, 4, size=n)
    predictions = labels.copy()  # perfect
    result = compute_conditional_accuracy(predictions, labels, distances, edges)
    assert result["overall"]["accuracy"] == pytest.approx(1.0)
    for b in result["per_bin"]:
        if b["n_crops"] > 0:
            assert b["accuracy"] == pytest.approx(1.0)


def test_compute_conditional_accuracy_chance_baseline_4class():
    """Random predictions on a 4-class problem -> Â ~ 0.25 marginally."""
    n = 10000
    rng = np.random.default_rng(0)
    distances = rng.uniform(0, 0.15, size=n)
    edges = compute_quintile_edges(distances)
    labels = rng.integers(0, 4, size=n)
    predictions = rng.integers(0, 4, size=n)
    result = compute_conditional_accuracy(predictions, labels, distances, edges)
    # 4-class chance baseline is 0.25 (with ~1% slack at this N)
    assert 0.22 <= result["overall"]["accuracy"] <= 0.28


# ----------------------------------------------------------- bootstrap CI

def test_bootstrap_ci_reproducible_under_fixed_seed():
    rng = np.random.default_rng(0)
    n = 1500
    distances = rng.uniform(0, 0.15, size=n)
    edges = compute_quintile_edges(distances)
    labels = rng.integers(0, 4, size=n)
    predictions = labels.copy()
    # Add some error so CI isn't degenerate
    flip_idx = rng.choice(n, size=300, replace=False)
    predictions[flip_idx] = (predictions[flip_idx] + 1) % 4

    r1 = bootstrap_accuracy_ci(predictions, labels, distances, edges, n_bootstrap=200, seed=42)
    r2 = bootstrap_accuracy_ci(predictions, labels, distances, edges, n_bootstrap=200, seed=42)
    assert r1["overall"]["ci_low"] == r2["overall"]["ci_low"]
    assert r1["overall"]["ci_high"] == r2["overall"]["ci_high"]
    for b1, b2 in zip(r1["per_bin"], r2["per_bin"]):
        if b1["n_crops"] > 0:
            assert b1["ci_low"] == b2["ci_low"]
            assert b1["ci_high"] == b2["ci_high"]


def test_bootstrap_ci_covers_point_estimate():
    """The CI should bracket the point estimate (acc in [ci_low, ci_high])."""
    rng = np.random.default_rng(0)
    n = 2000
    distances = rng.uniform(0, 0.15, size=n)
    edges = compute_quintile_edges(distances)
    labels = rng.integers(0, 4, size=n)
    predictions = labels.copy()
    flip_idx = rng.choice(n, size=400, replace=False)
    predictions[flip_idx] = (predictions[flip_idx] + 1) % 4

    r = bootstrap_accuracy_ci(predictions, labels, distances, edges, n_bootstrap=500, seed=0)
    o = r["overall"]
    assert o["ci_low"] <= o["accuracy"] <= o["ci_high"], (
        f"point {o['accuracy']:.4f} outside CI [{o['ci_low']:.4f}, {o['ci_high']:.4f}]"
    )


def test_bootstrap_rejects_too_few_resamples():
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_accuracy_ci(
            np.array([0]), np.array([0]),
            np.array([0.05]), np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.15]),
            n_bootstrap=50,
        )


# ----------------------------------------------------- audit-trail integrity

def test_r_bin_edges_json_roundtrip_integrity(tmp_path):
    """Write + load round-trip with sha256 audit-trail intact."""
    edges = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.151])
    out = tmp_path / "r_bin_edges.json"
    sha = write_r_bin_edges_artifact(edges, out, run_metadata={"run_id": "test"})
    assert out.is_file()
    loaded_edges, loaded_sha = load_r_bin_edges_artifact(out)
    np.testing.assert_array_equal(loaded_edges, edges)
    assert loaded_sha == sha


def test_r_bin_edges_json_detects_tampering(tmp_path):
    """If someone edits the edges after the JSON is written, the
    integrity check raises."""
    edges = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.151])
    out = tmp_path / "r_bin_edges.json"
    write_r_bin_edges_artifact(edges, out)
    # Tamper: change one edge
    payload = json.loads(out.read_text())
    payload["edges"][2] = 0.05
    out.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="integrity check failed"):
        load_r_bin_edges_artifact(out)


# --------------------------------------------------------- headline triplet

def test_headline_triplet_picks_correct_bins():
    """Headline maps to quintile indices 1, 2, 3 (r_25/r_50/r_75)."""
    rng = np.random.default_rng(0)
    n = 2000
    distances = rng.uniform(0, 0.15, size=n)
    edges = compute_quintile_edges(distances)
    labels = rng.integers(0, 4, size=n)
    predictions = labels.copy()
    r = bootstrap_accuracy_ci(predictions, labels, distances, edges, n_bootstrap=200, seed=0)
    triplet = headline_triplet(r)
    assert set(triplet.keys()) == {"25", "50", "75"}
    assert triplet["25"]["bin_index"] == HEADLINE_QUINTILE_INDICES[25]
    assert triplet["50"]["bin_index"] == HEADLINE_QUINTILE_INDICES[50]
    assert triplet["75"]["bin_index"] == HEADLINE_QUINTILE_INDICES[75]


# ----------------------------------------------------------- gate (e)

def test_trivial_baseline_gate_pass_when_resnet_beats_by_margin():
    rng = np.random.default_rng(0)
    n = 1000
    labels = rng.integers(0, 4, size=n)
    # Baseline gets 50% (random-ish on a non-trivial signal)
    base_preds = labels.copy()
    base_preds[rng.choice(n, size=500, replace=False)] = rng.integers(0, 4, size=500)
    resnet_acc = 0.80
    res = evaluate_trivial_baseline_accuracy(
        base_preds, labels, resnet_acc, name="mean-only", margin_pp=0.05,
    )
    assert res["pass"] is True
    assert res["observed_margin_pp"] > 0.05


def test_trivial_baseline_gate_fail_when_too_close():
    rng = np.random.default_rng(0)
    n = 1000
    labels = rng.integers(0, 4, size=n)
    # Baseline matches resnet within 2 pp -> fail
    base_preds = labels.copy()
    base_preds[rng.choice(n, size=200, replace=False)] = rng.integers(0, 4, size=200)
    # Baseline acc is ~0.80
    resnet_acc = 0.82  # only 2 pp above baseline
    res = evaluate_trivial_baseline_accuracy(
        base_preds, labels, resnet_acc, name="mean-only", margin_pp=0.05,
    )
    assert res["pass"] is False
    assert res["observed_margin_pp"] < 0.05

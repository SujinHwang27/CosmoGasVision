"""Tests for scripts/d70_wilcoxon_gate.py — MLflow-direct read path.

Uses a hand-rolled FakeClient that mimics the MlflowClient surface area
the harness consumes: get_experiment_by_name, search_runs,
get_metric_history. No real mlflow imports are exercised here; the
harness's injectable ``client`` param threads the fake through.
"""
from __future__ import annotations

import math
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pytest

# Make ./scripts importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import d70_wilcoxon_gate as gate  # noqa: E402


# ---------------------------------------------------------------------------
# Fake MLflow surface
# ---------------------------------------------------------------------------

@dataclass
class FakeMetric:
    step: int
    value: float
    timestamp: int = 0


@dataclass
class FakeRunInfo:
    run_id: str


@dataclass
class FakeRunData:
    tags: Dict[str, str]


@dataclass
class FakeRun:
    info: FakeRunInfo
    data: FakeRunData
    # metrics indexed by key -> list[FakeMetric]
    _metrics: Dict[str, List[FakeMetric]] = field(default_factory=dict)


@dataclass
class FakeExperiment:
    experiment_id: str = "0"
    name: str = "CosmoGasVision/NeRF"


class FakeClient:
    def __init__(self, runs: List[FakeRun],
                 experiment_name: str = "CosmoGasVision/NeRF"):
        self._runs = runs
        self._exp = FakeExperiment(name=experiment_name)
        self._exp_present = True

    def disable_experiment(self):
        self._exp_present = False

    def get_experiment_by_name(self, name: str):
        if not self._exp_present or name != self._exp.name:
            return None
        return self._exp

    def search_runs(self, experiment_ids, filter_string, max_results=100):
        # Apply a minimal subset of the MLflow filter language sufficient
        # for the harness's canonical filter:
        #   tags.K = 'V' AND tags.K2 = 'V2' AND ...
        clauses = [c.strip() for c in filter_string.split(" AND ")]
        wanted: Dict[str, str] = {}
        for c in clauses:
            assert c.startswith("tags."), f"unsupported filter: {c}"
            k, v = c[len("tags."):].split(" = ")
            wanted[k.strip()] = v.strip().strip("'")
        out = []
        for r in self._runs:
            if all(r.data.tags.get(k) == v for k, v in wanted.items()):
                out.append(r)
        return out[:max_results]

    def get_metric_history(self, run_id: str, key: str):
        for r in self._runs:
            if r.info.run_id == run_id:
                return list(r._metrics.get(key, []))
        return []


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

def _make_run(seed: int, var_pred: float, var_truth: float,
              bin_d: Optional[float] = 1.0,
              bin_b: Optional[float] = 1.0,
              arch: str = gate.TAG_ARCH_VALUE,
              juno_batch: str = gate.TAG_JUNO_BATCH_VALUE,
              stage: str = gate.TAG_STAGE_VALUE,
              max_steps: int = 500) -> FakeRun:
    tags = {
        gate.TAG_STAGE: stage,
        gate.TAG_ARCH: arch,
        gate.TAG_JUNO_BATCH: juno_batch,
        gate.TAG_SEED: str(seed),
        "model_type": "nerf",
    }
    metrics: Dict[str, List[FakeMetric]] = {
        gate.METRIC_VAR_PRED: [FakeMetric(step=max_steps, value=var_pred)],
        gate.METRIC_VAR_TRUTH: [FakeMetric(step=max_steps, value=var_truth)],
    }
    if bin_d is not None:
        metrics[gate.METRIC_BIN_D] = [FakeMetric(step=max_steps, value=bin_d)]
    if bin_b is not None:
        metrics[gate.METRIC_BIN_B] = [FakeMetric(step=max_steps, value=bin_b)]
    return FakeRun(
        info=FakeRunInfo(run_id=f"run-{seed:02d}"),
        data=FakeRunData(tags=tags),
        _metrics=metrics,
    )


def _build_runs(deltas: List[float],
                bin_d_offset_per_seed: Optional[List[float]] = None,
                truth_var: float = 1.0,
                max_steps: int = 500) -> List[FakeRun]:
    """One run per seed. R_real = 1.0 + delta ⇒ var_pred = truth_var * R_real."""
    runs = []
    for i, d in enumerate(deltas):
        r_real = 1.0 + d
        vp = truth_var * r_real
        bd = (1.0 + (bin_d_offset_per_seed[i] if bin_d_offset_per_seed else 0.0))
        runs.append(_make_run(seed=i, var_pred=vp, var_truth=truth_var,
                              bin_d=bd, bin_b=1.0, max_steps=max_steps))
    return runs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pass_case_strong_improve():
    """Synthetic deltas ~ N(+2σ, σ²) ⇒ Wilcoxon should PASS at α=0.025."""
    rng = np.random.default_rng(42)
    sigma = 0.1
    deltas = rng.normal(loc=2 * sigma, scale=sigma, size=10).tolist()
    runs = _build_runs(deltas)
    client = FakeClient(runs)
    result = gate.run_gate(
        tracking_uri="http://fake",
        experiment_name="CosmoGasVision/NeRF",
        max_steps=500,
        client=client,
    )
    assert result["verdict"] == "PASS", result
    assert result["p_stage1"] is not None and result["p_stage1"] <= 0.025


def test_fail_case_negative_drift():
    """Synthetic deltas ~ N(−σ, σ²) ⇒ FAIL."""
    rng = np.random.default_rng(7)
    sigma = 0.1
    deltas = rng.normal(loc=-sigma, scale=sigma, size=10).tolist()
    runs = _build_runs(deltas)
    client = FakeClient(runs)
    result = gate.run_gate(
        tracking_uri="http://fake",
        experiment_name="CosmoGasVision/NeRF",
        max_steps=500,
        client=client,
    )
    assert result["verdict"] == "FAIL", result
    assert result["p_stage1"] > 0.05


def test_boundary_marginal_then_rebake_to_n20():
    """Deltas at +0.5σ at n=10 may land MARGINAL; at n=20 lower p resolves
    PASS/FAIL deterministically."""
    rng = np.random.default_rng(1234)
    sigma = 0.1
    # Use 10 borderline draws first.
    deltas_10 = rng.normal(loc=0.5 * sigma, scale=sigma, size=10).tolist()
    runs10 = _build_runs(deltas_10)
    client10 = FakeClient(runs10)
    result10 = gate.run_gate(
        tracking_uri="http://fake",
        experiment_name="CosmoGasVision/NeRF",
        max_steps=500,
        client=client10,
    )
    # At this effect size + small n, the gate should NOT return PASS.
    # We accept either MARGINAL or FAIL — the boundary case spec is that
    # PASS is rare here, and MARGINAL triggers the re-bake path.
    assert result10["verdict"] in {"MARGINAL", "FAIL"}, result10

    # Stage-2 path: rebake to n=20 with same draw distribution.
    deltas_20 = (
        deltas_10
        + rng.normal(loc=0.5 * sigma, scale=sigma, size=10).tolist()
    )
    runs20 = _build_runs(deltas_20)
    client20 = FakeClient(runs20)
    result20 = gate.run_gate(
        tracking_uri="http://fake",
        experiment_name="CosmoGasVision/NeRF",
        max_steps=500,
        stage2=True,
        client=client20,
    )
    assert result20["verdict"] in {"PASS", "FAIL"}, result20
    assert result20["p_stage2"] is not None


def test_empty_experiment_raises_loud():
    """R20 twin-gate: empty MLflow experiment must raise AssertionError,
    never silently return PASS."""
    client = FakeClient([])
    with pytest.raises(AssertionError, match="0 matching runs"):
        gate.run_gate(
            tracking_uri="http://fake",
            experiment_name="CosmoGasVision/NeRF",
            max_steps=500,
            client=client,
        )


def test_dry_run_asserts_minimum_n():
    """--dry-run should raise AssertionError when <N_STAGE1 runs exist."""
    # 3 runs only — well below n=10.
    rng = np.random.default_rng(0)
    deltas = rng.normal(loc=0.2, scale=0.1, size=3).tolist()
    runs = _build_runs(deltas)
    client = FakeClient(runs)
    with pytest.raises(AssertionError, match=r"refusing to compute"):
        gate.run_gate(
            tracking_uri="http://fake",
            experiment_name="CosmoGasVision/NeRF",
            max_steps=500,
            dry_run=True,
            client=client,
        )


def test_insufficient_coverage_bin_d():
    """When >20% seeds have missing Bin-D metric, sub-clause (ii) flags
    INSUFFICIENT-COVERAGE."""
    rng = np.random.default_rng(99)
    sigma = 0.1
    deltas = rng.normal(loc=2 * sigma, scale=sigma, size=10).tolist()
    # Drop Bin-D from 5 out of 10 seeds (50% excluded > 20% threshold).
    runs = []
    for i, d in enumerate(deltas):
        bd = None if i < 5 else 1.0
        runs.append(
            _make_run(seed=i, var_pred=1.0 + d, var_truth=1.0,
                      bin_d=bd, bin_b=1.0)
        )
    client = FakeClient(runs)
    result = gate.run_gate(
        tracking_uri="http://fake",
        experiment_name="CosmoGasVision/NeRF",
        max_steps=500,
        client=client,
    )
    assert result["bin_d"]["verdict"] == "INSUFFICIENT-COVERAGE", result["bin_d"]
    assert result["bin_d"]["excluded_fraction"] > 0.20


def test_filter_string_rejects_non_stage1a_runs():
    """A run tagged with stage='2b' must NOT be returned by the Stage 1a
    filter even if other tags happen to match."""
    rng = np.random.default_rng(5)
    deltas = rng.normal(loc=0.2, scale=0.1, size=10).tolist()
    valid_runs = _build_runs(deltas)
    polluter = _make_run(seed=99, var_pred=10.0, var_truth=1.0,
                         stage="2b")  # wrong stage tag
    client = FakeClient(valid_runs + [polluter])
    result = gate.run_gate(
        tracking_uri="http://fake",
        experiment_name="CosmoGasVision/NeRF",
        max_steps=500,
        client=client,
    )
    # The polluter must be excluded.
    assert "99" not in result["per_seed"], result["per_seed"].keys()
    assert result["n_runs_matched"] == 10


def test_seed_pairing_preserved():
    """Per-seed dict must be keyed correctly and round-trip through
    extract → wilcoxon."""
    rng = np.random.default_rng(31)
    deltas = rng.normal(loc=2 * 0.1, scale=0.1, size=10).tolist()
    runs = _build_runs(deltas)
    client = FakeClient(runs)
    result = gate.run_gate(
        tracking_uri="http://fake",
        experiment_name="CosmoGasVision/NeRF",
        max_steps=500,
        client=client,
    )
    assert set(result["per_seed"].keys()) == {str(i) for i in range(10)}
    for i, d in enumerate(deltas):
        assert result["per_seed"][str(i)]["delta"] == pytest.approx(d, rel=1e-9)


def test_missing_metric_skips_seed_gracefully():
    """A run with no m3_var_pred_log metric must yield delta=None and
    not crash the harness."""
    rng = np.random.default_rng(13)
    deltas = rng.normal(loc=2 * 0.1, scale=0.1, size=10).tolist()
    runs = _build_runs(deltas)
    # Wipe the var_pred metric on seed 3.
    runs[3]._metrics.pop(gate.METRIC_VAR_PRED)
    client = FakeClient(runs)
    result = gate.run_gate(
        tracking_uri="http://fake",
        experiment_name="CosmoGasVision/NeRF",
        max_steps=500,
        client=client,
    )
    assert result["per_seed"]["3"]["delta"] is None
    # 9 valid deltas remain; Wilcoxon should still produce a verdict.
    assert result["n_valid_deltas"] == 9
    assert result["verdict"] in {"PASS", "MARGINAL", "FAIL", "INSUFFICIENT-N"}

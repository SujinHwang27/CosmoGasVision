"""Tests for the outbound data-export boundary (src/export.py).

Fast tests use SYNTHETIC fixtures to exercise the write/validate logic, the
provenance-sidecar contract, and the pure statistics helpers — no real
Sherwood/ binaries and no real checkpoints. Any test that touches the real
sim data or a real production checkpoint is marked ``@pytest.mark.slow``.

Owner: data-engineer. Contract: .claude/skills/data-export/SKILL.md.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src import export as X


# --------------------------------------------------------------------------- #
# Provenance / write contract
# --------------------------------------------------------------------------- #
def _read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_export_root_constant():
    assert X.EXPORT_ROOT == "results/exports"


def test_write_export_csv_and_sidecar(tmp_path):
    rows = [
        {"a": 1.5, "b": 2.0},
        {"a": np.float64(3.25), "b": 4.0},
    ]
    artifact, sidecar = X.write_export(
        out_dir=tmp_path,
        filename="unit.csv",
        fieldnames=["a", "b"],
        rows=rows,
        producing_fn="test.fn",
        source_data_path="synthetic",
        physics_id=1,
        caveat="synthetic caveat",
    )
    assert artifact.exists() and sidecar.exists()
    # CSV round-trips, header present, no comment lines.
    text = artifact.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "a,b"
    assert not any(ln.lstrip().startswith("#") for ln in text.splitlines())
    got = _read_csv(artifact)
    assert len(got) == 2
    assert float(got[0]["a"]) == 1.5

    # Sidecar has the mandatory fields + a non-'unknown' git SHA.
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["git"]["commit"] != "unknown"
    for key in ("producing_function", "source_data_path", "n_rays_eval",
                "eval_seed", "physics_id", "redshift", "box_kpc_h", "caveat"):
        assert key in meta
    assert meta["n_rays_eval"] == X.N_RAYS_EVAL == 1024
    assert meta["eval_seed"] == X.EVAL_SEED == 42
    assert meta["redshift"] == 0.3


def test_write_export_full_float64_precision(tmp_path):
    # An irrational-ish float must survive to full repr precision (no rounding).
    val = float(np.pi) * 1e-7
    artifact, _ = X.write_export(
        out_dir=tmp_path,
        filename="prec.csv",
        fieldnames=["x"],
        rows=[{"x": val}],
        producing_fn="test.fn",
        source_data_path="synthetic",
        physics_id=1,
        caveat="precision",
    )
    got = _read_csv(artifact)
    assert float(got[0]["x"]) == val  # bit-exact round-trip


def test_write_export_json_payload(tmp_path):
    artifact, sidecar = X.write_export(
        out_dir=tmp_path,
        filename="payload.json",
        fieldnames=None,
        rows=None,
        producing_fn="test.fn",
        source_data_path="synthetic",
        physics_id=[1, 2],
        caveat="json",
        json_payload={"k": [1, 2, 3]},
    )
    assert json.loads(artifact.read_text())["k"] == [1, 2, 3]
    assert sidecar.exists()


def test_write_export_rejects_unknown_git(tmp_path, monkeypatch):
    monkeypatch.setattr(
        X, "get_git_info",
        lambda: {"commit": "unknown", "branch": "x", "dirty": "False", "timestamp": "t"},
    )
    with pytest.raises(RuntimeError, match="unknown"):
        X.write_export(
            out_dir=tmp_path, filename="x.csv", fieldnames=["a"],
            rows=[{"a": 1.0}], producing_fn="f", source_data_path="s",
            physics_id=1, caveat="c",
        )


# --------------------------------------------------------------------------- #
# Validation guards
# --------------------------------------------------------------------------- #
def test_validate_flux_accepts_in_bounds():
    X._validate_flux("ok", np.array([0.0, 0.5, 1.0], dtype=np.float64))


def test_validate_flux_rejects_out_of_bounds():
    with pytest.raises(AssertionError):
        X._validate_flux("hi", np.array([0.5, 1.5], dtype=np.float64))
    with pytest.raises(AssertionError):
        X._validate_flux("lo", np.array([-0.01, 0.5], dtype=np.float64))


def test_validate_flux_rejects_nan():
    with pytest.raises(AssertionError):
        X._validate_flux("nan", np.array([0.5, np.nan], dtype=np.float64))


def test_validate_pf_rejects_negative():
    with pytest.raises(AssertionError):
        X._validate_pf("pf", np.array([1.0, -2.0, np.nan], dtype=np.float64))


def test_validate_pf_allows_nan_bins():
    # Empty log-k bins legitimately return NaN; the guard tolerates NaN, only
    # negative finite values fail.
    X._validate_pf("pf", np.array([1.0, np.nan, 2.0], dtype=np.float64))


def test_validate_rows_rejects_nonfinite():
    with pytest.raises(AssertionError):
        X._validate_rows("r", [{"a": 1.0}, {"a": float("inf")}])


def test_validate_rows_rejects_empty():
    with pytest.raises(AssertionError):
        X._validate_rows("r", [])


# --------------------------------------------------------------------------- #
# Deterministic selection + band residual (pure)
# --------------------------------------------------------------------------- #
def test_seed42_selection_deterministic_and_sorted():
    a = X._seed42_selection(1024)
    b = X._seed42_selection(1024)
    assert np.array_equal(a, b)
    assert (np.diff(a) > 0).all()  # strictly increasing (sorted, unique)
    assert a.shape == (1024,)
    # Representative-ray index (sel[0]) is stable at 67 for this project's draw.
    assert int(a[0]) == 67


def test_pf_band_residual_matches_published_definition():
    # Band = [10^-2.5, 10^-1.5]; residual = mean(|mlp-truth|/truth) over the band.
    centers = np.array([1e-3, 10 ** -2.0, 10 ** -1.8, 1e-1], dtype=np.float64)
    pf_truth = np.array([9.9, 2.0, 4.0, 9.9], dtype=np.float64)
    pf_mlp = np.array([9.9, 3.0, 5.0, 9.9], dtype=np.float64)
    # in-band bins: indices 1 and 2. residuals: 0.5 and 0.25 -> mean 0.375
    r = X._pf_band_residual(centers, pf_mlp, pf_truth)
    assert r == pytest.approx(0.375, abs=1e-12)


def test_pf_band_residual_handles_no_in_band():
    centers = np.array([1e-3, 1e-1], dtype=np.float64)  # both outside band
    r = X._pf_band_residual(centers, np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert np.isnan(r)


# --------------------------------------------------------------------------- #
# Figure 2a — mean-flux table (banked; no render). Uses a synthetic bootstrap
# JSON so it does NOT depend on the real d44 artifact being present.
# --------------------------------------------------------------------------- #
def _synthetic_d44(tmp_path: Path) -> str:
    payload = {
        "results_full": {
            c: {
                "meanF_mean": 0.97,
                "meanF_q16": 0.965,
                "meanF_q84": q84,
                "meanF_per_seed": [0.97, 0.968, 0.972, 0.969, 0.966],
                "KS_mean": 0.04,
            }
            for c, q84 in (("P1", 0.9690), ("P2", 0.9724), ("P3", 0.9760), ("P4", 0.9770))
        }
    }
    p = tmp_path / "d44.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_export_mean_flux_table_2of4_gate_logic(tmp_path):
    d44 = _synthetic_d44(tmp_path)
    res = X.export_mean_flux_table(out_dir=str(tmp_path), bootstrap_json=d44)
    assert res["n_rows"] == 4
    # P1 (q84=0.969) and P2 (q84=0.9724) are below the 0.974 gate lower edge -> FAIL;
    # P3/P4 (q84 >= 0.974) -> PASS. => 2/4 PASS, reproducing the LEDGER verdict.
    assert res["n_bootstrap_pass_of_4"] == 2
    assert res["cells_ci_below_gate_band"] == ["P1", "P2"]
    # Honesty: all four q84 sit below the 0.979 point anchor.
    assert res["cells_ci_below_point_anchor"] == ["P1", "P2", "P3", "P4"]

    rows = _read_csv(Path(res["artifact"]))
    assert [r["physics"] for r in rows] == ["P1", "P2", "P3", "P4"]
    # seed=42 anchors are the LEDGER values, not the synthetic bootstrap.
    assert float(rows[0]["mean_F_pred_seed42"]) == pytest.approx(0.97895)

    meta = json.loads(Path(res["sidecar"]).read_text())
    assert meta["mean_F_obs"] == 0.979
    assert meta["mean_F_obs_err"] == 0.005
    # Verb-ceiling: caveat must carry the 2/4 softening, never claim 4/4.
    assert "2/4" in meta["caveat"] or "2 of 4" in meta["caveat"] or "only 2/4" in meta["caveat"]


def test_seed42_anchors_match_ledger():
    # Guard against silent drift of the hardcoded LEDGER anchors.
    assert X.SEED42_ANCHORS["P1"]["P_F_residual"] == 0.4155
    assert X.SEED42_ANCHORS["P1"]["mean_F"] == 0.97895
    assert X.SEED42_ANCHORS["P1"]["KS"] == 0.0325
    assert X.MEAN_F_OBS == 0.979 and X.MEAN_F_OBS_ERR == 0.005


def test_pf_band_edges():
    assert X.PF_BAND_LO == pytest.approx(10 ** -2.5)
    assert X.PF_BAND_HI == pytest.approx(10 ** -1.5)


# --------------------------------------------------------------------------- #
# SLOW: real Sherwood/ + real production checkpoint. These reproduce the
# published banked numbers and are the integration guard.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_export_pf_miss_reproduces_banked_p1(tmp_path):
    res = X.export_pf_miss(out_dir=str(tmp_path), cells=("P1",))
    # Reproduces the banked seed=42 P1 band residual 0.4155 to within tolerance.
    assert res["p1_band_residual"] == pytest.approx(0.4155, abs=0.01)
    curve = _read_csv(Path(res["artifact"]))
    assert len(curve) > 0
    for r in curve:
        assert float(r["P_F_mlp"]) >= 0.0
        assert float(r["P_F_truth"]) >= 0.0
    assert Path(res["band_table_artifact"]).exists()


@pytest.mark.slow
def test_export_flux_pdf_reproduces_banked_ks_p1(tmp_path):
    res = X.export_flux_pdf(out_dir=str(tmp_path), cell="P1")
    assert res["ks_exported_cell"] == pytest.approx(0.0325, abs=0.005)
    rows = _read_csv(Path(res["artifact"]))
    assert len(rows) == 49
    for r in rows:
        assert float(r["pdf_mlp"]) >= 0.0 and float(r["pdf_truth"]) >= 0.0


@pytest.mark.slow
def test_export_single_sightline_bounds_and_index(tmp_path):
    res = X.export_single_sightline(out_dir=str(tmp_path), cell="P1")
    assert res["representative_ray_global_index"] == 67
    rows = _read_csv(Path(res["artifact"]))
    assert len(rows) == 2048
    for r in rows:
        fm, ft = float(r["F_mlp"]), float(r["F_truth"])
        assert 0.0 <= fm <= 1.0 + 1e-9
        assert 0.0 <= ft <= 1.0 + 1e-9

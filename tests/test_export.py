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


# --------------------------------------------------------------------------- #
# ep04 "the-direct-attack" batch (banked [D-40] exports). Synthetic fixtures;
# no dependence on the real sat_aware_hypc artifacts.
# --------------------------------------------------------------------------- #
def _synthetic_d40_per_bin(tmp_path: Path, nan_in_band: bool = False,
                           band=(10 ** -2.5, 10 ** -1.5)) -> str:
    k = [1e-3, 4e-3, 6e-3, 1e-2, 2e-2, 5e-2]
    p_truth = [float("nan"), 2.0, 1.5, 1.0, 0.5, 0.1]
    if nan_in_band:
        p_truth[2] = float("nan")
    p_pred = [t * 0.4 for t in p_truth]  # exact proportionality -> pearson 1.0
    ratio = [p / t if t == t else float("nan") for p, t in zip(p_pred, p_truth)]
    rel = [r - 1.0 if r == r else float("nan") for r in ratio]
    in_band = [band[0] <= x <= band[1] for x in k]
    payload = {
        "run_id": "synthetic",
        "ckpt_path": "synthetic.pt",
        "pf_band_s_per_km": list(band),
        "k_axis": k,
        "P_truth": p_truth,
        "P_pred": p_pred,
        "P_pred_over_P_truth": ratio,
        "rel_diff_per_bin": rel,
        "in_band_mask": in_band,
        "diagnostics": {
            "log10_std_pred_in_band": 0.25,
            "log10_std_truth_in_band": 0.25,
            "band_ratio_mean": 0.4,
            "band_ratio_std": 0.0,
            "pearson_log_in_band": 1.0,
            "mean_abs_rel_diff_in_band": 0.6,
        },
        "hypothesis_c_verdict": "synthetic",
    }
    p = tmp_path / "d40_per_bin.json"
    p.write_text(json.dumps(payload))
    return str(p)


def _synthetic_d40_baseline(tmp_path: Path) -> str:
    payload = {
        c: {"log_std_pred": 0.3, "log_std_truth": 0.2, "ratio_mean": rm,
            "ratio_std": 0.4, "pearson_log": 0.85}
        for c, rm in (("P1", 0.98), ("P2", 0.97), ("P3", 0.74), ("P4", 0.79))
    }
    p = tmp_path / "d40_baseline.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_export_d40_pf_per_bin_drops_nan_and_reproduces_diags(tmp_path):
    src = _synthetic_d40_per_bin(tmp_path)
    res = X.export_d40_pf_per_bin(out_dir=str(tmp_path), per_bin_json=src)
    rows = _read_csv(Path(res["artifact"]))
    assert len(rows) == 5  # one out-of-band NaN bin dropped
    assert [int(r["in_gate_band"]) for r in rows] == [1, 1, 1, 1, 0]
    assert res["pearson_log_rederived"] == pytest.approx(1.0)
    meta = json.loads(Path(res["sidecar"]).read_text())
    assert meta["n_nan_bins_dropped"] == 1
    # Honesty: the caveat names the signature, not constant collapse.
    assert "amplitude-shrink" in meta["caveat"]
    assert "shape preservation" in meta["caveat"]


def test_export_d40_pf_per_bin_refuses_nan_inside_band(tmp_path):
    src = _synthetic_d40_per_bin(tmp_path, nan_in_band=True)
    with pytest.raises(AssertionError, match="inside the gate band"):
        X.export_d40_pf_per_bin(out_dir=str(tmp_path), per_bin_json=src)


def test_export_d40_pf_per_bin_rejects_wrong_band(tmp_path):
    src = _synthetic_d40_per_bin(tmp_path, band=(1e-3, 1e-1))
    with pytest.raises(AssertionError, match="band edges"):
        X.export_d40_pf_per_bin(out_dir=str(tmp_path), per_bin_json=src)


def test_export_d40_verdict_table_banked_values(tmp_path):
    res = X.export_d40_verdict_table(out_dir=str(tmp_path))
    assert res["n_rows"] == 5
    assert res["pf_worsening_pct"] == pytest.approx(37.35, abs=0.05)
    rows = {r["metric"]: r for r in _read_csv(Path(res["artifact"]))}
    assert float(rows["pf_residual_band_mean"]["sat_aware_run"]) == 0.5707
    assert float(rows["ks_distance"]["sat_aware_run"]) == 0.1888
    assert float(rows["train_loss_pf_band_final"]["sat_aware_run"]) == 6.77e-06
    meta = json.loads(Path(res["sidecar"]).read_text())
    # Honesty: endpoints-only rule + not-a-win framing must ride the caveat.
    assert "ENDPOINTS" in meta["caveat"] or "endpoints" in meta["caveat"]
    assert "NOT a win" in meta["caveat"]
    assert meta["train_steps"] == 12500


def test_export_d40_shape_amplitude_summary(tmp_path):
    src = _synthetic_d40_per_bin(tmp_path)
    base = _synthetic_d40_baseline(tmp_path)
    res = X.export_d40_shape_amplitude_summary(
        out_dir=str(tmp_path), per_bin_json=src, baseline_json=base)
    rows = _read_csv(Path(res["artifact"]))
    assert len(rows) == 5
    assert rows[0]["model"] == "sat_aware_intervention"
    assert float(rows[0]["amplitude_ratio_mean"]) == pytest.approx(0.4)
    assert [r["physics"] for r in rows[1:]] == ["P1", "P2", "P3", "P4"]
    meta = json.loads(Path(res["sidecar"]).read_text())
    # Honesty: baseline-only scope of the cross-physics rows.
    assert "P1 only" in meta["caveat"]


def test_export_d40_run_config_discipline_framing(tmp_path):
    res = X.export_d40_run_config(out_dir=str(tmp_path))
    rows = {r["key"]: r["value"] for r in _read_csv(Path(res["artifact"]))}
    assert rows["train_steps"] == "12500"
    assert rows["physics"] == "P1 (fiducial) only"
    meta = json.loads(Path(res["sidecar"]).read_text())
    assert "argued, not tested" in meta["caveat"]


def test_export_d40_scrub_gate_on_consumer_surfaces(tmp_path):
    """No internal identifiers on any consumer-facing surface (caveats, CSV
    headers/values). Internal tags are confined to internal_lineage."""
    src = _synthetic_d40_per_bin(tmp_path)
    base = _synthetic_d40_baseline(tmp_path)
    outs = [
        X.export_d40_verdict_table(out_dir=str(tmp_path)),
        X.export_d40_pf_per_bin(out_dir=str(tmp_path), per_bin_json=src),
        X.export_d40_shape_amplitude_summary(out_dir=str(tmp_path),
                                             per_bin_json=src, baseline_json=base),
        X.export_d40_run_config(out_dir=str(tmp_path)),
    ]
    barred = ("[D-", "LEDGER", "pub-t1", "Tier-1", "Tier-2", "Juno", "Sprint")
    for res in outs:
        meta = json.loads(Path(res["sidecar"]).read_text())
        csv_text = Path(res["artifact"]).read_text()
        for tok in barred:
            assert tok not in meta["caveat"], (res["artifact"], tok)
            assert tok not in csv_text, (res["artifact"], tok)


# --------------------------------------------------------------------------- #
# ep05 "the-physics-constraint" batch (banked [D-41] exports + local-tracker
# re-read). Synthetic fixtures; no dependence on the real mlflow.db/artifacts.
# --------------------------------------------------------------------------- #
def _synthetic_mlflow_db(tmp_path: Path, break_endpoint: bool = False,
                         drop_step: bool = False) -> str:
    import sqlite3
    p = tmp_path / "mlflow_synth.db"
    con = sqlite3.connect(str(p))
    con.execute("create table metrics (key text, value real, timestamp int, "
                "run_uuid text, step int)")
    run = X.D41_SMOKE_RUN_ID
    n = 50
    for step in range(1, n + 1):
        t = (step - 1) / (n - 1)
        vals = {
            "loss_fgpa_tail": 6.367 * (1 - t) + 0.3085 * t,
            "mean_flux_pred": 0.8687 * (1 - t) + 1.0 * t,
            "tau_amp": 1.0 * (1 - t) + 0.9906 * t,
            "loss_data": 0.02, "loss_meanF": 0.01, "loss": 0.5,
        }
        if break_endpoint and step == n:
            vals["mean_flux_pred"] = 0.95
        for k, v in vals.items():
            if drop_step and k == "tau_amp" and step == 25:
                continue
            con.execute("insert into metrics values (?,?,?,?,?)", (k, v, 0, run, step))
    con.commit()
    con.close()
    return str(p)


def _synthetic_d41_collapse(tmp_path: Path, tamper: bool = False) -> str:
    def block(lo, med, hi):
        return {"min": lo, "median": med, "mean": med, "max": hi}
    payload = {
        "truth": {"density": block(0.007, 0.145, 6456.5),
                  "X_HI": block(7.6e-11, 6.0e-07, 2.3e-04),
                  "n_HI": block(8.8e-12, 8.4e-08, 4.1e-02),
                  "n_HI_lt_1e9_frac": 0.004},
        "fgpa_tail_tier1_pred": {"density": block(68.3, 71.5, 74.7),
                                 "X_HI": block(2.1e-05, 3.3e-05, 5.2e-05),
                                 "n_HI": block(1.6e-03, 2.4e-03, 3.6e-03),
                                 "n_HI_lt_1e9_frac": 0.0},
        "collapse_ratio_med": {"density": 71.5 / 0.145,
                               "X_HI": 3.3e-05 / 6.0e-07,
                               "n_HI": (2.4e-03 / 8.4e-08) if not tamper else 999.0},
    }
    p = tmp_path / "d41_collapse.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_export_d41_smoke_trace_reads_tracker(tmp_path):
    db = _synthetic_mlflow_db(tmp_path)
    res = X.export_d41_smoke_trace(out_dir=str(tmp_path), db_path=db)
    rows = _read_csv(Path(res["artifact"]))
    assert len(rows) == 50
    assert float(rows[-1]["mean_flux_pred"]) == 1.0
    assert res["descent_factor"] == pytest.approx(6.367 / 0.3085, rel=1e-6)
    meta = json.loads(Path(res["sidecar"]).read_text())
    # Honesty: the tell's licensed reading + smoke-vs-ratification split.
    assert "fingerprint of collapse" in meta["caveat"]
    assert "RATIFIED" in meta["caveat"]


def test_export_d41_smoke_trace_rejects_endpoint_drift(tmp_path):
    db = _synthetic_mlflow_db(tmp_path, break_endpoint=True)
    with pytest.raises(AssertionError, match="disagree with"):
        X.export_d41_smoke_trace(out_dir=str(tmp_path), db_path=db)


def test_export_d41_smoke_trace_rejects_missing_steps(tmp_path):
    db = _synthetic_mlflow_db(tmp_path, drop_step=True)
    with pytest.raises(AssertionError, match="missing steps"):
        X.export_d41_smoke_trace(out_dir=str(tmp_path), db_path=db)


def test_export_d41_collapse_signature(tmp_path):
    src = _synthetic_d41_collapse(tmp_path)
    res = X.export_d41_collapse_signature(out_dir=str(tmp_path), collapse_json=src)
    rows = _read_csv(Path(res["artifact"]))
    assert [r["field"] for r in rows] == ["density", "X_HI", "n_HI"]
    assert float(rows[0]["pred_median"]) == pytest.approx(71.5)
    meta = json.loads(Path(res["sidecar"]).read_text())
    # Honesty: attribution to the confirmation run + corrected mechanism.
    assert "CONFIRMATION" in meta["caveat"]
    assert "CONSTANT-PREDICTION COLLAPSE" in meta["caveat"]
    assert "empirically" in meta["caveat"]


def test_export_d41_collapse_signature_rejects_tampered_ratio(tmp_path):
    src = _synthetic_d41_collapse(tmp_path, tamper=True)
    with pytest.raises(AssertionError, match="disagrees with banked"):
        X.export_d41_collapse_signature(out_dir=str(tmp_path), collapse_json=src)


def test_export_d41_verdict_and_config_honesty(tmp_path):
    res3 = X.export_d41_verdict_table(out_dir=str(tmp_path))
    meta3 = json.loads(Path(res3["sidecar"]).read_text())
    assert "EMPTY" in json.dumps(_read_csv(Path(res3["artifact"])))
    assert "do NOT quote" in meta3["caveat"]
    res4 = X.export_d41_run_config(out_dir=str(tmp_path))
    meta4 = json.loads(Path(res4["sidecar"]).read_text())
    assert "measure the relation from truth" in meta4["caveat"]


def test_export_d41_scrub_gate_on_consumer_surfaces(tmp_path):
    db = _synthetic_mlflow_db(tmp_path)
    src = _synthetic_d41_collapse(tmp_path)
    outs = [
        X.export_d41_smoke_trace(out_dir=str(tmp_path), db_path=db),
        X.export_d41_collapse_signature(out_dir=str(tmp_path), collapse_json=src),
        X.export_d41_verdict_table(out_dir=str(tmp_path)),
        X.export_d41_run_config(out_dir=str(tmp_path)),
    ]
    barred = ("[D-", "LEDGER", "pub-t1", "Tier-1", "Tier-2", "tier1", "Juno", "Sprint", "197381")
    for res in outs:
        meta = json.loads(Path(res["sidecar"]).read_text())
        csv_text = Path(res["artifact"]).read_text()
        for tok in barred:
            assert tok not in meta["caveat"], (res["artifact"], tok)
            assert tok not in csv_text, (res["artifact"], tok)

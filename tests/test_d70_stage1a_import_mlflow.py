"""Tests for scripts/d70_stage1a_import_mlflow.py.

Strategy: build a real source MLflow file-store under a tmpdir containing 2
"seeds", point a real destination file-store at another tmpdir, then invoke
the import script's ``main`` programmatically. This avoids mocking mlflow's
internals (which has historically been brittle) and exercises the actual
replay path end-to-end.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient  # noqa: E402

# Make ``scripts/`` importable so we can call main() directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import d70_stage1a_import_mlflow as importer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: build n=2 source stores under <input_dir>/seed_<i>/mlflow/
# ---------------------------------------------------------------------------

def _build_source_seed_store(seed_root: Path, seed: int) -> None:
    """Create one Stage-1a-shaped file-store under ``seed_root/mlflow/``.

    Mirrors the layout produced by submit_juno_stage1a_1b.sh after PCV copy.
    """
    store = seed_root / "mlflow"
    store.mkdir(parents=True, exist_ok=True)
    uri = store.resolve().as_uri()

    # Use a fresh mlflow context bound to this store. mlflow's globals are
    # process-wide, so we set/reset for each call.
    mlflow.set_tracking_uri(uri)
    exp_name = "CosmoGasVision/NeRF"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=f"Stage1a-1b-SkipRichMLP-P1-N768-S{seed}"):
        mlflow.set_tags({
            "model_type": "nerf",
            "stage": "1a-density-pretrain",
            "physics_id": "1",
            "redshift": "0.3",
            "body_arch": "skip-rich-mlp",
            "compute": "juno",
            "juno_batch": "stage1a-1b-skiprich",
            "seed": str(seed),
            "stage_substep": "1a-(1b)",
        })
        mlflow.log_param("max_steps", "500")
        mlflow.log_metric("m3_var_pred_log", 0.5 + 0.01 * seed, step=500)
        mlflow.log_metric("m3_var_truth_log", 0.5, step=500)
        mlflow.log_metric("s3_log_mse_bin_D", 0.1 + 0.01 * seed, step=500)


@pytest.fixture
def two_seed_input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "stage1a_results"
    input_dir.mkdir()
    for seed in (0, 1):
        _build_source_seed_store(input_dir / f"seed_{seed}", seed)
    # Reset the global tracking URI so subsequent test logic isn't pinned to
    # the last seed's source store.
    mlflow.set_tracking_uri("")
    return input_dir


@pytest.fixture
def dst_uri(tmp_path: Path) -> str:
    dst_store = tmp_path / "host_mlruns"
    dst_store.mkdir()
    return dst_store.resolve().as_uri()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_imports_two_seeds(two_seed_input: Path, dst_uri: str, capsys):
    rc = importer.main([
        "--input-dir", str(two_seed_input),
        "--tracking-uri", dst_uri,
        "--experiment", "CosmoGasVision/NeRF",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "D70_IMPORT_REPLAY_RUNS_FOUND=2" in captured
    assert "D70_IMPORT_REPLAY_RUNS_IMPORTED=2" in captured
    assert "D70_IMPORT_REPLAY_RUNS_SKIPPED=0" in captured
    assert "D70_IMPORT_REPLAY_STATUS=OK" in captured

    # Verify dst tracker actually holds both runs with the expected tags + metrics.
    client = MlflowClient(tracking_uri=dst_uri)
    exp = client.get_experiment_by_name("CosmoGasVision/NeRF")
    assert exp is not None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            "tags.stage = '1a-density-pretrain' AND "
            "tags.body_arch = 'skip-rich-mlp' AND "
            "tags.juno_batch = 'stage1a-1b-skiprich'"
        ),
    )
    assert len(runs) == 2
    seeds = sorted(r.data.tags.get("seed") for r in runs)
    assert seeds == ["0", "1"]
    for r in runs:
        assert "m3_var_pred_log" in r.data.metrics
        assert "m3_var_truth_log" in r.data.metrics
        assert "s3_log_mse_bin_D" in r.data.metrics
        assert r.data.tags.get("imported_from_juno") == "true"


def test_idempotency_second_run_skips(two_seed_input: Path, dst_uri: str, capsys):
    # First import.
    rc1 = importer.main([
        "--input-dir", str(two_seed_input),
        "--tracking-uri", dst_uri,
    ])
    assert rc1 == 0
    capsys.readouterr()  # discard

    # Second import on the same input — should skip both.
    rc2 = importer.main([
        "--input-dir", str(two_seed_input),
        "--tracking-uri", dst_uri,
    ])
    assert rc2 == 0
    captured = capsys.readouterr().out
    assert "D70_IMPORT_REPLAY_RUNS_FOUND=2" in captured
    assert "D70_IMPORT_REPLAY_RUNS_IMPORTED=0" in captured
    assert "D70_IMPORT_REPLAY_RUNS_SKIPPED=2" in captured
    assert "D70_IMPORT_REPLAY_STATUS=OK" in captured


def test_empty_input_dir_raises_loud(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AssertionError, match="No MLflow stores found"):
        importer.main([
            "--input-dir", str(empty),
            "--tracking-uri", (tmp_path / "dst").as_uri(),
        ])


def test_dry_run_does_not_write(two_seed_input: Path, tmp_path: Path, capsys):
    dst_store = tmp_path / "dst_dryrun"
    dst_store.mkdir()
    dst_uri_local = dst_store.resolve().as_uri()

    rc = importer.main([
        "--input-dir", str(two_seed_input),
        "--tracking-uri", dst_uri_local,
        "--dry-run",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "D70_IMPORT_REPLAY_RUNS_FOUND=2" in captured
    assert "D70_IMPORT_REPLAY_RUNS_IMPORTED=0" in captured
    assert "D70_IMPORT_REPLAY_RUNS_SKIPPED=0" in captured
    assert "would consider" in captured

    # Destination must remain empty: no experiment created.
    client = MlflowClient(tracking_uri=dst_uri_local)
    exp = client.get_experiment_by_name("CosmoGasVision/NeRF")
    assert exp is None, "dry-run must not create the destination experiment"

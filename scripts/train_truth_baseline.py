"""Sprint-4 [D-51] truth-baseline 3D ResNet training + Â(r) evaluation.

Per the design doc at ``experiments/nerf/design/sprint4_truth_baseline.md``
(\xa7\xa71-12). Produces:

  - Trained checkpoint at experiments/nerf/artifacts/sprint4/checkpoints/
  - Per-bin Â(r) with crop-unit 1k-bootstrap 95% CIs (headline.json)
  - r_bin_edges.json (pre-test-inference audit trail)
  - Per-physics 4x4 confusion matrix (json + png)
  - Trivial-baseline accuracies (mean-overdensity, mean+variance)
  - 5-gate evaluation against the design doc \xa78 pre-committed spec

Two modes:
  --smoke  : 5-step memory smoke, no full eval (validates the wiring +
             checks for OOM on the local GPU; gate-(c) determinism check)
  default  : full 30-epoch train on 20k/4k/8k crops with all 5 gates

Run:
    PYTHONPATH=. python -u scripts/train_truth_baseline.py --smoke
    PYTHONPATH=. python -u scripts/train_truth_baseline.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.analysis.conditional_accuracy import (
    DEFAULT_BLOCK_SIZE_VOXELS,
    GATE_E_ESCALATION_BAND_LOW,
    GATE_E_MARGIN_PP,
    block_bootstrap_accuracy_ci,
    bootstrap_accuracy_ci,
    compute_quintile_edges,
    evaluate_trivial_baseline_accuracy,
    headline_triplet,
    write_r_bin_edges_artifact,
)
from src.data.augment3d import AugmentConfig, DEFAULT_AUGMENT, augment_batch
from src.data.loader import (
    DEFAULT_SCHEME,
    SherwoodLoader,
    _RHO_FIELD_CACHE,
)
from src.models.cnn3d import (
    MeanOverdensityBaseline,
    MeanVarianceBaseline,
    resnet18_3d_4class,
)


def _inject_synthetic_rho_fields(n_grid: int, redshift: float) -> None:
    """Inject deterministic per-physics synthetic rho fields into the
    in-memory cache so wiring-smoke runs do not require Sherwood
    IGM_gal data for P2/P3/P4 (only P1 is locally available on most
    hosts). Each physics gets a distinct seed so the truth-baseline
    classifier has a real signal to learn at the wiring-smoke layer.
    """
    for physics_id in (1, 2, 3, 4):
        rng = np.random.default_rng(seed=0xA51FF7 + physics_id)
        rho = rng.lognormal(mean=0.0, sigma=0.5 + 0.1 * physics_id,
                            size=(n_grid, n_grid, n_grid)).astype(np.float32)
        rho /= rho.mean()
        _RHO_FIELD_CACHE[(physics_id, round(redshift, 3), n_grid)] = rho

ART_ROOT = Path(_REPO) / "experiments/nerf/artifacts"
SPRINT4_DIR = ART_ROOT / "sprint4"
EVAL_DIR = ART_ROOT / "eval" / "sprint4"
SPRINT4_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = SPRINT4_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="5-step memory smoke; no full eval.")
    ap.add_argument("--crop_size", type=int, default=32)
    ap.add_argument("--n_grid", type=int, default=768)
    ap.add_argument("--n_crops_train", type=int, default=5000,
                    help="Per physics (4 physics -> total = 4 * n_crops_train).")
    ap.add_argument("--n_crops_val", type=int, default=1000)
    ap.add_argument("--n_crops_test", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr_max", type=float, default=3e-4)
    ap.add_argument("--lr_min", type=float, default=3e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--seed_train", type=int, default=42)
    ap.add_argument("--seed_val", type=int, default=142)
    ap.add_argument("--seed_test", type=int, default=242)
    ap.add_argument("--seed_model", type=int, default=42)
    ap.add_argument("--seed_aug", type=int, default=42)
    ap.add_argument("--redshift", type=float, default=0.300)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-augment", action="store_true",
                    help="Disable augmentation (e.g., for the determinism gate-(c) check).")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    return ap.parse_args()


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def build_lr_schedule(optimizer, lr_max: float, lr_min: float,
                      warmup_steps: int, total_steps: int):
    """Linear warmup 0 -> lr_max, then cosine decay lr_max -> lr_min."""
    decay_steps = max(1, total_steps - warmup_steps)
    min_ratio = lr_min / lr_max if lr_max > 0 else 0.0
    import math as _math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def draw_split_crops(
    loader: SherwoodLoader,
    region: str,
    n_per_physics: int,
    crop_size: int,
    n_grid: int,
    seed: int,
    return_positions: bool = False,
):
    """Draw `n_per_physics` crops per physics from a region. Returns
    (crops, labels, distances) concatenated across the 4 physics. When
    ``return_positions=True``, additionally returns the per-crop CIC
    corner positions (n_total, 3) int64 — required for the [D-52]
    amendment 8 block-bootstrap CI.
    """
    all_crops, all_labels, all_dists = [], [], []
    all_positions: list[np.ndarray] = []
    for physics_id in (1, 2, 3, 4):
        result = loader.extract_rho_crops_split(
            physics_id=physics_id,
            redshift=0.300,
            crop_size=crop_size,
            n_crops=n_per_physics,
            region=region,
            scheme=DEFAULT_SCHEME,
            seed=seed,
            n_grid=n_grid,
            return_positions=return_positions,
        )
        if return_positions:
            crops, labels, dists, positions = result
            all_positions.append(np.asarray(positions, dtype=np.int64))
        else:
            crops, labels, dists = result
        # Labels in loader come back as physics_id (1..4); shift to 0..3.
        all_crops.append(crops)
        all_labels.append((labels - 1).to(torch.long))
        all_dists.append(np.asarray(dists, dtype=np.float64))
    if return_positions:
        return (
            torch.cat(all_crops, dim=0),
            torch.cat(all_labels, dim=0),
            np.concatenate(all_dists),
            np.concatenate(all_positions, axis=0),
        )
    return (
        torch.cat(all_crops, dim=0),
        torch.cat(all_labels, dim=0),
        np.concatenate(all_dists),
    )


def train_one_epoch(
    model: nn.Module,
    crops: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    epoch: int,
    batch_size: int,
    device: torch.device,
    augment_cfg: AugmentConfig,
    aug_base_seed: int,
    rng: np.random.Generator,
) -> float:
    """One epoch of stochastic gradient descent. Returns mean loss."""
    model.train()
    n = crops.shape[0]
    order = rng.permutation(n)
    losses: list[float] = []
    for start in range(0, n, batch_size):
        sel = order[start : start + batch_size]
        sample_indices = torch.from_numpy(sel.astype(np.int64))
        batch = crops[sel].to(device)
        y = labels[sel].to(device)
        if augment_cfg.enabled:
            batch = augment_batch(
                batch, epoch=epoch, sample_indices=sample_indices,
                base_seed=aug_base_seed, config=augment_cfg,
            )
        logits = model(batch)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def eval_predictions(
    model: nn.Module,
    crops: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (predictions, ground_truth, mean_loss)."""
    model.eval()
    n = crops.shape[0]
    preds, truths, losses = [], [], []
    crit = nn.CrossEntropyLoss(reduction="mean")
    for start in range(0, n, batch_size):
        batch = crops[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        logits = model(batch)
        loss = crit(logits, y)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        truths.append(y.cpu().numpy())
        losses.append(float(loss.item()))
    return (
        np.concatenate(preds),
        np.concatenate(truths),
        float(np.mean(losses)),
    )


def compute_confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int = 4,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(predictions.astype(int), labels.astype(int)):
        cm[t, p] += 1
    return cm


def train_trivial_baseline(
    factory,
    crops_train: torch.Tensor, labels_train: torch.Tensor,
    crops_val: torch.Tensor, labels_val: torch.Tensor,
    epochs: int, batch_size: int, lr: float, device: torch.device,
    seed: int,
) -> nn.Module:
    """Train a trivial baseline (1-scalar or 2-scalar) -> FC(4) per
    gate (e). Standard simple training loop; no augmentation."""
    torch.manual_seed(seed)
    model = factory(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(crops_train.shape[0])
        for start in range(0, crops_train.shape[0], batch_size):
            sel = order[start : start + batch_size]
            batch = crops_train[sel].to(device)
            y = labels_train[sel].to(device)
            logits = model(batch)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    run_id = f"sprint4_{int(time.time())}"
    print(f"[sprint4] run_id={run_id} device={device}")
    print(f"[sprint4] mode={'SMOKE' if args.smoke else 'FULL'}")

    if args.smoke:
        # Smoke overrides for fast wiring check. Sizes are clamped low
        # but large enough to populate every quintile in the test set
        # (so the block-bootstrap doesn't return all-NaN per-bin CIs).
        args.n_crops_train = 32
        args.n_crops_val = 16
        args.n_crops_test = 24
        args.epochs = 1
        args.batch_size = 8
        args.n_bootstrap = 100
        args.early_stop_patience = 999  # disable
        args.crop_size = 16  # smaller for memory; must fit the val/test region
        args.n_grid = 128   # val/test regions ~19 voxels wide at this n_grid
        # Inject synthetic per-physics rho fields so the wiring smoke
        # does NOT require Sherwood IGM_gal data for P2/P3/P4 (only P1
        # is locally CIC-cached on this host). The full-mode (no
        # --smoke) path still uses real Sherwood data via the loader.
        print(f"[sprint4] SMOKE: injecting synthetic per-physics rho fields "
              f"at n_grid={args.n_grid} (skips Sherwood IGM_gal I/O)")
        _inject_synthetic_rho_fields(n_grid=args.n_grid, redshift=args.redshift)

    # ----- 1) Build datasets -------------------------------------------
    print(f"[sprint4] loading rho crops "
          f"({args.n_crops_train} train + {args.n_crops_val} val + "
          f"{args.n_crops_test} test, per physics x 4 physics) ...")
    t_load = time.perf_counter()
    repo_root = Path(_REPO)
    loader = SherwoodLoader(data_root=str(repo_root / "Sherwood"))

    crops_train, labels_train, _ = draw_split_crops(
        loader, region="train", n_per_physics=args.n_crops_train,
        crop_size=args.crop_size, n_grid=args.n_grid, seed=args.seed_train,
    )
    crops_val, labels_val, dist_val = draw_split_crops(
        loader, region="val", n_per_physics=args.n_crops_val,
        crop_size=args.crop_size, n_grid=args.n_grid, seed=args.seed_val,
    )
    # Test set: pull positions too — needed for the [D-52] amendment 8
    # block-bootstrap CI (positions are voxel-unit CIC corners on the
    # n_grid field, axes (0, 1, 2)). Train/val do not need positions.
    crops_test, labels_test, dist_test, positions_test = draw_split_crops(
        loader, region="test", n_per_physics=args.n_crops_test,
        crop_size=args.crop_size, n_grid=args.n_grid, seed=args.seed_test,
        return_positions=True,
    )
    print(
        f"[sprint4] dataset built in {time.perf_counter()-t_load:.1f}s - "
        f"train {tuple(crops_train.shape)} / val {tuple(crops_val.shape)} / "
        f"test {tuple(crops_test.shape)}"
    )

    # ----- 2) Model + optimizer ----------------------------------------
    torch.manual_seed(args.seed_model)
    model = resnet18_3d_4class(in_channels=1, num_classes=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sprint4] model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_max, betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    batches_per_epoch = (crops_train.shape[0] + args.batch_size - 1) // args.batch_size
    total_steps = batches_per_epoch * args.epochs
    warmup_steps = batches_per_epoch * args.warmup_epochs
    scheduler = build_lr_schedule(
        optimizer, args.lr_max, args.lr_min, warmup_steps, total_steps,
    )

    augment_cfg = AugmentConfig(enabled=not args.no_augment)

    # ----- 3) Training loop --------------------------------------------
    training_log_path = EVAL_DIR / f"{run_id}_training_log.csv"
    rng = np.random.default_rng(args.seed_train + 7777)
    best_val_loss = float("inf")
    patience_left = args.early_stop_patience
    train_history = []

    with open(training_log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc",
                         "lr", "elapsed_s"])
        t_train_start = time.perf_counter()

        for epoch in range(args.epochs):
            t_epoch_start = time.perf_counter()
            train_loss = train_one_epoch(
                model, crops_train, labels_train, optimizer, scheduler,
                criterion, epoch=epoch, batch_size=args.batch_size,
                device=device, augment_cfg=augment_cfg,
                aug_base_seed=args.seed_aug, rng=rng,
            )
            val_preds, val_truths, val_loss = eval_predictions(
                model, crops_val, labels_val, args.batch_size, device,
            )
            val_acc = float((val_preds == val_truths).mean())
            cur_lr = float(optimizer.param_groups[0]["lr"])
            elapsed_s = time.perf_counter() - t_train_start
            writer.writerow([epoch, train_loss, val_loss, val_acc, cur_lr, elapsed_s])
            train_history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_acc": val_acc, "lr": cur_lr,
                "epoch_seconds": time.perf_counter() - t_epoch_start,
            })
            print(
                f"[sprint4] epoch {epoch+1}/{args.epochs}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f} lr={cur_lr:.2e} "
                f"({time.perf_counter()-t_epoch_start:.1f}s)"
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_left = args.early_stop_patience
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, CKPT_DIR / f"resnet18_3d_4class_best.pt")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[sprint4] early-stopping at epoch {epoch+1}")
                    break

    t_train_total = time.perf_counter() - t_train_start
    print(f"[sprint4] training done in {t_train_total:.1f}s")

    # Note: smoke mode now FALLS THROUGH to the full-eval path so the
    # [D-52] amendment-compliant headline.json fields (outcome_branch,
    # power_calibration, deliverable_framing, scope_disclosure,
    # prior_work_cites, block-bootstrap CI) are exercised by the wiring
    # smoke. Sizes are clamped above (n_crops=4/4/4 per physics,
    # n_bootstrap=100). A separate {run_id}_smoke.json marker is still
    # written below to confirm smoke completion + the new field
    # surfaces.

    # ----- 4) Load best checkpoint + write r_bin_edges -----------------
    ckpt_path = CKPT_DIR / "resnet18_3d_4class_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    edges = compute_quintile_edges(dist_val)
    edges_path = EVAL_DIR / f"{run_id}_r_bin_edges.json"
    edges_sha = write_r_bin_edges_artifact(
        edges, edges_path,
        run_metadata={"run_id": run_id, "epoch_best": ckpt["epoch"]},
    )
    print(f"[sprint4] r_bin_edges sha256={edges_sha[:16]}... at {edges_path}")

    # ----- 5) Test inference + bootstrap -------------------------------
    test_preds, test_truths, test_loss = eval_predictions(
        model, crops_test, labels_test, args.batch_size, device,
    )
    boot = bootstrap_accuracy_ci(
        test_preds, test_truths, dist_test, edges,
        n_bootstrap=args.n_bootstrap, alpha=args.alpha,
        seed=args.seed_test + 13,
    )
    triplet = headline_triplet(boot)

    # [D-52] amendment 8: block-bootstrap CI side-by-side with the IID
    # crop-unit bootstrap (Politis & Romano 1994; Norberg et al. 2009).
    # Block size = 64 voxels at n_grid=768 ~= 5 h^{-1} Mpc ~= 2x [D-13]
    # gate scale, applied on the (axis_1, axis_2) plane perpendicular to
    # the [D-49] held-out axis-0 slab.
    print(f"[sprint4] running block-bootstrap CI side-by-side "
          f"(block_size_voxels={DEFAULT_BLOCK_SIZE_VOXELS}, "
          f"n_bootstrap={args.n_bootstrap}) ...")
    block_boot = block_bootstrap_accuracy_ci(
        test_preds, test_truths, dist_test, edges,
        crop_positions=positions_test,
        block_size_voxels=DEFAULT_BLOCK_SIZE_VOXELS,
        n_bootstrap=args.n_bootstrap, alpha=args.alpha,
        seed=args.seed_test + 17,
    )
    block_triplet = headline_triplet(block_boot)

    # Side-by-side comparison: if max |ord_ci_low - block_ci_low| across
    # the 5 quintiles > 0.02, the block-bootstrap is the headline.
    diffs = []
    for k in range(5):
        ord_lo = boot["per_bin"][k].get("ci_low")
        blk_lo = block_boot["per_bin"][k].get("ci_low")
        if ord_lo is not None and blk_lo is not None \
                and np.isfinite(ord_lo) and np.isfinite(blk_lo):
            diffs.append(abs(ord_lo - blk_lo))
    max_ci_low_diff = float(max(diffs)) if diffs else float("nan")
    use_block_as_headline = bool(
        np.isfinite(max_ci_low_diff) and max_ci_low_diff > 0.02
    )
    bootstrap_comparison = {
        "max_ci_low_abs_diff_across_quintiles": max_ci_low_diff,
        "threshold_for_block_as_headline": 0.02,
        "block_bootstrap_is_headline": use_block_as_headline,
        "note": (
            "Per [D-52] amendment 8: if max |ord_ci_low - block_ci_low| "
            "across the 5 quintiles > 0.02, the block-bootstrap CI is the "
            "headline number (the IID bootstrap underestimates CI width on "
            "the spatially-correlated Sherwood rho field). Otherwise the "
            "two CIs are reported side-by-side without re-ranking."
        ),
    }
    if use_block_as_headline:
        print(f"[sprint4] block-bootstrap CI is the headline "
              f"(max ci_low abs diff = {max_ci_low_diff:.4f} > 0.02; "
              "IID 1k-bootstrap underestimates CI width on this "
              "spatially-correlated population).")
    else:
        print(f"[sprint4] IID 1k-bootstrap and block-bootstrap CIs agree "
              f"within 0.02 (max ci_low abs diff = {max_ci_low_diff:.4f}); "
              "reporting both side-by-side without re-ranking.")

    cm = compute_confusion_matrix(test_preds, test_truths, num_classes=4)

    # ----- 6) Trivial baselines (gate (e)) -----------------------------
    print("[sprint4] training trivial baselines for gate (e) ...")
    mean_baseline = train_trivial_baseline(
        MeanOverdensityBaseline,
        crops_train, labels_train, crops_val, labels_val,
        epochs=max(5, args.epochs // 3), batch_size=args.batch_size,
        lr=1e-3, device=device, seed=args.seed_model + 1,
    )
    mean_var_baseline = train_trivial_baseline(
        MeanVarianceBaseline,
        crops_train, labels_train, crops_val, labels_val,
        epochs=max(5, args.epochs // 3), batch_size=args.batch_size,
        lr=1e-3, device=device, seed=args.seed_model + 2,
    )
    mean_preds, _, _ = eval_predictions(
        mean_baseline, crops_test, labels_test, args.batch_size, device,
    )
    mv_preds, _, _ = eval_predictions(
        mean_var_baseline, crops_test, labels_test, args.batch_size, device,
    )

    resnet_acc = boot["overall"]["accuracy"]
    # [D-52] amendment 6: AD-5 gate-(e) threshold tightened 5pp -> 10pp.
    # Margins in [GATE_E_ESCALATION_BAND_LOW, GATE_E_MARGIN_PP) route to
    # capacity-matched-MLP follow-on escalation (flagged in log + json).
    gate_e1 = evaluate_trivial_baseline_accuracy(
        mean_preds, test_truths, resnet_acc, name="mean-overdensity",
        margin_pp=GATE_E_MARGIN_PP,
        escalation_band_low_pp=GATE_E_ESCALATION_BAND_LOW,
    )
    gate_e2 = evaluate_trivial_baseline_accuracy(
        mv_preds, test_truths, resnet_acc, name="mean+variance",
        margin_pp=GATE_E_MARGIN_PP,
        escalation_band_low_pp=GATE_E_ESCALATION_BAND_LOW,
    )
    for gate_label, gate_dict in (("mean-overdensity", gate_e1),
                                  ("mean+variance", gate_e2)):
        if gate_dict.get("escalate_to_capacity_matched_mlp"):
            print(
                f"[sprint4] AD-5 ESCALATION FLAG (per [D-52] amendment 6): "
                f"{gate_label} margin = "
                f"{gate_dict['observed_margin_pp']:.3f} lands in "
                f"[{GATE_E_ESCALATION_BAND_LOW:.2f}, {GATE_E_MARGIN_PP:.2f}) "
                "pp band; escalate to capacity-matched-MLP follow-on "
                "(4th-moment + spectral-energy baseline, ~100 params; "
                "Loshchilov 2019 precedent) before paper text ships."
            )

    # ----- 7) Five-gate evaluation -------------------------------------
    # (a) sanity floor: overall CI lower bound > 0.50
    ci_low_overall = boot["overall"]["ci_low"]
    gate_a = {
        "pass": bool(ci_low_overall > 0.50),
        "threshold": "overall CI lower bound > 0.50",
        "observed_ci_low": ci_low_overall,
        "observed_accuracy": resnet_acc,
    }
    # (b) r_50 well-definedness: bin 2 has >= 200 crops AND CI half-width < 0.05
    bin2 = boot["per_bin"][2]
    half_width = (bin2.get("ci_high", float("nan")) - bin2.get("ci_low", float("nan"))) / 2.0
    gate_b = {
        "pass": bool(bin2.get("n_crops", 0) >= 200 and half_width < 0.05),
        "threshold": "r_50 bin n_crops >= 200 AND CI half-width < 0.05",
        "observed_n_crops": bin2.get("n_crops", 0),
        "observed_half_width": half_width,
    }
    # (c) determinism: tested separately (run twice and compare); flag here
    gate_c = {
        "pass": None,
        "note": "End-to-end determinism check requires a separate run; "
                "verify by re-invoking train_truth_baseline.py with same "
                "seeds and comparing predictions for bit-identity.",
    }
    # (d) smoothness: max - min across the 5 quintiles
    per_bin_accs = [b["accuracy"] for b in boot["per_bin"] if b.get("accuracy") is not None]
    finite_accs = [a for a in per_bin_accs if np.isfinite(a)]
    N_bins = len(per_bin_accs)
    if len(finite_accs) == N_bins and N_bins > 0:
        smoothness_range = float(max(finite_accs) - min(finite_accs))
    else:
        smoothness_range = float("nan")
    # Monotone or range < 0.10
    if len(finite_accs) >= 5:
        is_monotone = all(
            finite_accs[i] <= finite_accs[i+1] for i in range(len(finite_accs)-1)
        ) or all(
            finite_accs[i] >= finite_accs[i+1] for i in range(len(finite_accs)-1)
        )
    else:
        is_monotone = False
    gate_d = {
        "pass": bool(is_monotone or (np.isfinite(smoothness_range) and smoothness_range < 0.10)),
        "threshold": "monotone OR max-min range < 0.10",
        "observed_range": smoothness_range,
        "observed_per_bin_accs": finite_accs,
        "is_monotone": is_monotone,
    }
    # (e) trivial baselines — AD-5 [D-52] amendment 6 (5pp -> 10pp).
    gate_e = {
        "pass": bool(gate_e1["pass"] and gate_e2["pass"]),
        "threshold": (
            f"both trivial baselines trail ResNet by >= "
            f"{int(GATE_E_MARGIN_PP * 100)} pp "
            "(AD-5, [D-52] amendment 6 — tightened from 5 pp under "
            "R13 scope-lock re-verbing audit)"
        ),
        "required_margin_pp": GATE_E_MARGIN_PP,
        "escalation_band_low_pp": GATE_E_ESCALATION_BAND_LOW,
        "mean_overdensity": gate_e1,
        "mean_variance": gate_e2,
        "escalate_to_capacity_matched_mlp": bool(
            gate_e1.get("escalate_to_capacity_matched_mlp")
            or gate_e2.get("escalate_to_capacity_matched_mlp")
        ),
    }

    gates = {
        "gate_a_sanity_floor": gate_a,
        "gate_b_r50_well_defined": gate_b,
        "gate_c_determinism": gate_c,
        "gate_d_smoothness": gate_d,
        "gate_e_trivial_baseline": gate_e,
    }

    # ----- 7b) [D-52] 4-branch outcome routing -------------------------
    # Per [D-52] amendment 7 pre-committed routing. The decision rule
    # selects ONE of four branches; the driver continues running and
    # produces all numbers in every branch — the routing only affects
    # reporting framing in §3 paper text. Implementation note: the
    # block-bootstrap-as-headline override (amendment 8) is applied here
    # — when use_block_as_headline is true, the r_50 CI used for
    # branching is taken from the block-bootstrap result.
    headline_per_bin = block_boot["per_bin"] if use_block_as_headline else boot["per_bin"]
    r50_bin = headline_per_bin[2]
    r50_ci_low = r50_bin.get("ci_low", float("nan"))
    process_failure_reasons = []
    if gate_a.get("pass") is False:
        process_failure_reasons.append("gate-(a) sanity-floor FAIL")
    # gate_c is DEFERRED (separate-run determinism check); not a
    # synchronous process-failure trigger here.
    # AD-1 anti-leakage lives in tests/test_split_anti_leakage.py and is
    # checked at test-suite time; recorded here as a top-level marker
    # for completeness.
    if not np.isfinite(r50_ci_low):
        process_failure_reasons.append("r_50 CI lower bound non-finite (training divergence proxy)")

    ad5_pass = bool(gate_e["pass"])

    if process_failure_reasons:
        outcome_branch = "process_failure"
        outcome_note = (
            "sprint-4 measurement infrastructure failed pre-condition(s): "
            + "; ".join(process_failure_reasons)
            + ". No A_truth(r) value publishable for this submission cycle."
        )
    elif not ad5_pass and gate_e1.get("observed_margin_pp", 0.0) < GATE_E_ESCALATION_BAND_LOW \
            and gate_e2.get("observed_margin_pp", 0.0) < GATE_E_ESCALATION_BAND_LOW:
        # Both baselines closer than 5 pp -> hard ceiling-disqualified.
        outcome_branch = "ceiling_disqualified"
        outcome_note = (
            "AD-5 FAIL at >= 10 pp margin (and both trivial baselines "
            "within the 5 pp band): the deep model cannot beat "
            "capacity-matched trivials at >= 10 pp on Sherwood 4-class "
            "crops at 32^3; A_truth(r) reading is anchored to low-order "
            "moment structure, NOT to feedback-specific 3D content. "
            "Substantive null result, sec 4 follow-on caveat only."
        )
    elif r50_ci_low > 0.87 and ad5_pass:
        outcome_branch = "above_bar"
        outcome_note = (
            "1k-bootstrap CI lower bound at r_50 > 0.87 AND AD-5 PASS "
            "at >= 10 pp margin -> above-bar reporting per [D-52] "
            "success-criterion branch (i)."
        )
    elif 0.85 <= r50_ci_low <= 0.87:
        outcome_branch = "indistinguishable_from_bar"
        outcome_note = (
            "1k-bootstrap CI lower bound at r_50 in [0.85, 0.87]; "
            "result reports as indistinguishable from the [D-15] 0.85 bar "
            "at this n (MDE 0.05 at 80% power per [D-52] amendment 5)."
        )
    elif r50_ci_low < 0.85 and ad5_pass:
        outcome_branch = "below_bar_with_AD5_pass"
        outcome_note = (
            f"1k-bootstrap CI lower bound at r_50 = {r50_ci_low:.4f} < 0.85 "
            "AND AD-5 PASS at >= 10 pp margin -> below empirical reference "
            f"at value {r50_ci_low:.4f} per [D-37]-ext rule 5 "
            "symmetric-honesty; A_truth(r) is a probe-classifier "
            "discriminability lower bound, the IGM-feedback "
            "discriminability ceiling may be higher under different "
            "architectures or supervision regimes."
        )
    else:
        # AD-5 fail with one baseline in the escalation band -> ceiling-disqualified.
        outcome_branch = "ceiling_disqualified"
        outcome_note = (
            "AD-5 FAIL at >= 10 pp margin per [D-52] amendment 7 "
            "ceiling-disqualification routing. Substantive null result, "
            "sec 4 follow-on caveat only."
        )
    print(f"[sprint4] [D-52] outcome routing: {outcome_branch}")
    print(f"[sprint4] {outcome_note}")

    # ----- 7c) Power calibration block ---------------------------------
    # Per [D-52] amendment 5: pre-committed MDE / Wilson-score
    # CI half-width / indistinguishable-from-bar band, documented
    # alongside the empirical results.
    power_calibration = {
        "mde_at_80pct_power": 0.05,
        "wilson_score_ci_halfwidth_per_quintile": 0.035,
        "indistinguishable_from_bar_band": [0.85, 0.87],
        "test_set_n_per_physics": int(args.n_crops_test),
        "test_set_n_total": int(crops_test.shape[0]),
        "note": (
            "Per [D-52] amendment 5: at test-set n ~= 2k crops/physics x "
            "4 physics ~= 8k total in the [D-49] axis=0 held-out region, "
            "MDE on A_truth(r) at 80% power ~= 0.05; the gate "
            "'lower bound > 0.85' is detectable above-or-below only when "
            "p_hat >= 0.87, leaving a 2-pp band where the result reports "
            "as 'indistinguishable from the 0.85 bar at this n'. "
            "Per-quintile (n ~= 400) the Wilson-score 95% CI half-width "
            "is ~3.5 pp on a binary classifier at p=0.85."
        ),
    }

    # ----- 8) Write headline.json --------------------------------------
    # [D-52] amendment 3: probe-classifier discriminability lower-bound
    # framing in field names (NOT "ceiling"). Per Alain & Bengio 2017 +
    # Theunissen 2003, classifier accuracy is a lower bound on task
    # discriminability, not a ceiling on it.
    headline = {
        "run_id": run_id,
        "spec": "Sprint-4 [D-51] truth-baseline 3D ResNet per design doc "
                "\xa7\xa71-12 + [D-52] post-pre-review amendments (2026-05-13b)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": t_train_total,
        "epochs_completed": ckpt["epoch"] + 1,
        "model_params": n_params,
        "device": str(device),
        "best_val_loss": best_val_loss,
        "best_val_acc": ckpt["val_acc"],
        "test_loss": test_loss,
        # ---- [D-52] amendment 7: 4-branch outcome routing (top-level)
        "outcome_branch": outcome_branch,
        "outcome_note": outcome_note,
        # ---- [D-52] amendment 3: framing fields (top-level)
        "deliverable_framing": (
            "probe-classifier discriminability lower bound on Sherwood "
            "physics-recipe signature at z=0.3 in truth rho field at "
            "crop=32^3"
        ),
        "scope_disclosure": (
            "the 0.85 bar is project-internal per [D-36]; no external "
            "observational anchor"
        ),
        "prior_work_cites": [
            "bolton2017sherwood (1D flux-stat scale)",
            "irsic2017lyman (cross-physics P_F differential)",
        ],
        # ---- [D-52] amendment 5: power-calibration block
        "power_calibration": power_calibration,
        # ---- [D-52] amendment 8: side-by-side bootstrap CIs
        "test_overall_ordinary_bootstrap": boot["overall"],
        "test_per_bin_ordinary_bootstrap": boot["per_bin"],
        "test_overall_block_bootstrap": block_boot["overall"],
        "test_per_bin_block_bootstrap": block_boot["per_bin"],
        "headline_triplet_ordinary_bootstrap": triplet,
        "headline_triplet_block_bootstrap": block_triplet,
        "block_bootstrap_metadata": {
            "block_size_voxels": block_boot.get("block_size_voxels"),
            "n_blocks": block_boot.get("n_blocks"),
            "mean_crops_per_block": block_boot.get("mean_crops_per_block"),
            "cite": "Politis & Romano 1994; Norberg et al. 2009",
        },
        "bootstrap_comparison": bootstrap_comparison,
        # ---- legacy aliases retained for back-compat with sprint-4 readers
        "test_overall": boot["overall"],
        "test_per_bin": boot["per_bin"],
        "headline_triplet": triplet,
        # ---- standard fields
        "r_bin_edges": [float(e) for e in edges],
        "r_bin_edges_sha256": edges_sha,
        "confusion_matrix": cm.tolist(),
        "trivial_baselines": {
            "mean_overdensity": gate_e1,
            "mean_variance": gate_e2,
        },
        "gates": gates,
    }
    out_path = EVAL_DIR / f"{run_id}_headline.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(headline, fh, indent=2, default=str)
    cm_path = EVAL_DIR / f"{run_id}_confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as fh:
        json.dump({"confusion_matrix": cm.tolist(),
                   "row_label": "ground truth physics_id",
                   "col_label": "predicted physics_id"}, fh, indent=2)

    # ----- 9) Summary print --------------------------------------------
    # ASCII-only ("A_hat" not "Â"); the Windows console cp949
    # codepage cannot encode the Latin-1 hat.
    print("\n[sprint4] HEADLINE A_hat(r) triplet "
          "(probe-classifier discriminability lower bound, per [D-52] "
          "amendment 3 -- Alain & Bengio 2017; Theunissen 2003):")
    print("  ordinary 1k-bootstrap | block-bootstrap (block=64 vox)")
    for k in ("25", "50", "75"):
        v_ord = triplet[k]
        v_blk = block_triplet[k]
        ord_ci = (f"[{v_ord['ci_low']:.3f}, {v_ord['ci_high']:.3f}]"
                  if v_ord.get("ci_low") is not None else "[n/a]")
        blk_ci = (f"[{v_blk['ci_low']:.3f}, {v_blk['ci_high']:.3f}]"
                  if v_blk.get("ci_low") is not None else "[n/a]")
        print(f"  r_{k}: A_hat = {v_ord['accuracy']:.4f}  "
              f"ord {ord_ci}  blk {blk_ci}  "
              f"(r_center = {v_ord.get('r_center', float('nan')):.4f}, "
              f"n_crops = {v_ord['n_crops']})")
    o = boot["overall"]
    o_blk = block_boot["overall"]
    print(f"  overall: A_hat = {o['accuracy']:.4f}  "
          f"ord [{o['ci_low']:.3f}, {o['ci_high']:.3f}]  "
          f"blk [{o_blk['ci_low']:.3f}, {o_blk['ci_high']:.3f}]  "
          f"(n_crops = {o['n_crops']})")
    print(f"\n[sprint4] [D-52] outcome routing: {outcome_branch}")
    print(f"  {outcome_note}")
    print(f"\n[sprint4] 5-gate summary:")
    for name, g in gates.items():
        if g["pass"] is None:
            verdict = "DEFERRED"
        elif g["pass"]:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        print(f"  {name}: {verdict}")
    print(f"\n[sprint4] artifacts:")
    print(f"  headline      : {out_path}")
    print(f"  r_bin_edges   : {edges_path}")
    print(f"  confusion mat : {cm_path}")
    print(f"  training log  : {training_log_path}")
    print(f"  ckpt          : {ckpt_path}")

    if args.smoke:
        # Smoke marker — companion to the full headline.json, used by
        # the wiring-smoke acceptance criterion: confirms the smoke
        # path ran end-to-end through the [D-52] amendment fields.
        smoke_summary = {
            "run_id": run_id,
            "mode": "smoke",
            "smoke_steps_completed": True,
            "no_oom": True,
            "model_params": n_params,
            "train_history": train_history,
            "headline_path": str(out_path),
            "outcome_branch": outcome_branch,
            "new_fields_present": {
                "outcome_branch": "outcome_branch" in headline,
                "power_calibration": "power_calibration" in headline,
                "deliverable_framing": "deliverable_framing" in headline,
                "scope_disclosure": "scope_disclosure" in headline,
                "prior_work_cites": "prior_work_cites" in headline,
                "block_bootstrap_overall": (
                    "test_overall_block_bootstrap" in headline
                ),
                "block_bootstrap_per_bin": (
                    "test_per_bin_block_bootstrap" in headline
                ),
                "bootstrap_comparison": "bootstrap_comparison" in headline,
            },
        }
        smoke_path = EVAL_DIR / f"{run_id}_smoke.json"
        with open(smoke_path, "w", encoding="utf-8") as fh:
            json.dump(smoke_summary, fh, indent=2, default=str)
        print(f"[sprint4] smoke OK - wrote {smoke_path}")
        # Smoke does not gate-fail (wiring is the only criterion).
        return 0

    return 0 if all(g.get("pass") for g in gates.values() if g["pass"] is not None) else 1


if __name__ == "__main__":
    sys.exit(main())

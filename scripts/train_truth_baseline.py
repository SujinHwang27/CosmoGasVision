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
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Draw `n_per_physics` crops per physics from a region. Returns
    (crops, labels, distances) concatenated across the 4 physics.
    """
    all_crops, all_labels, all_dists = [], [], []
    for physics_id in (1, 2, 3, 4):
        crops, labels, dists = loader.extract_rho_crops_split(
            physics_id=physics_id,
            redshift=0.300,
            crop_size=crop_size,
            n_crops=n_per_physics,
            region=region,
            scheme=DEFAULT_SCHEME,
            seed=seed,
            n_grid=n_grid,
        )
        # Labels in loader come back as physics_id (1..4); shift to 0..3.
        all_crops.append(crops)
        all_labels.append((labels - 1).to(torch.long))
        all_dists.append(np.asarray(dists, dtype=np.float64))
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
        # Smoke overrides for fast wiring check
        args.n_crops_train = 8
        args.n_crops_val = 4
        args.n_crops_test = 4
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
    crops_test, labels_test, dist_test = draw_split_crops(
        loader, region="test", n_per_physics=args.n_crops_test,
        crop_size=args.crop_size, n_grid=args.n_grid, seed=args.seed_test,
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

    if args.smoke:
        print("[sprint4] SMOKE mode: skipping full eval + gates")
        smoke_summary = {
            "run_id": run_id,
            "mode": "smoke",
            "smoke_steps_completed": True,
            "no_oom": True,
            "model_params": n_params,
            "train_history": train_history,
        }
        with open(EVAL_DIR / f"{run_id}_smoke.json", "w", encoding="utf-8") as fh:
            json.dump(smoke_summary, fh, indent=2, default=str)
        print(f"[sprint4] smoke OK - wrote {EVAL_DIR/f'{run_id}_smoke.json'}")
        return 0

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
    gate_e1 = evaluate_trivial_baseline_accuracy(
        mean_preds, test_truths, resnet_acc, name="mean-overdensity",
    )
    gate_e2 = evaluate_trivial_baseline_accuracy(
        mv_preds, test_truths, resnet_acc, name="mean+variance",
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
    # (e) trivial baselines
    gate_e = {
        "pass": bool(gate_e1["pass"] and gate_e2["pass"]),
        "threshold": "both trivial baselines trail ResNet by >= 5 pp",
        "mean_overdensity": gate_e1,
        "mean_variance": gate_e2,
    }

    gates = {
        "gate_a_sanity_floor": gate_a,
        "gate_b_r50_well_defined": gate_b,
        "gate_c_determinism": gate_c,
        "gate_d_smoothness": gate_d,
        "gate_e_trivial_baseline": gate_e,
    }

    # ----- 8) Write headline.json --------------------------------------
    headline = {
        "run_id": run_id,
        "spec": "Sprint-4 [D-51] truth-baseline 3D ResNet per design doc \xa7\xa71-12",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": t_train_total,
        "epochs_completed": ckpt["epoch"] + 1,
        "model_params": n_params,
        "device": str(device),
        "best_val_loss": best_val_loss,
        "best_val_acc": ckpt["val_acc"],
        "test_loss": test_loss,
        "test_overall": boot["overall"],
        "test_per_bin": boot["per_bin"],
        "headline_triplet": triplet,
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
    print("\n[sprint4] HEADLINE Â(r) triplet:")
    for k, v in triplet.items():
        ci = f"[{v['ci_low']:.3f}, {v['ci_high']:.3f}]" if v.get("ci_low") is not None else "[n/a]"
        print(f"  r_{k}: Â = {v['accuracy']:.4f} {ci}  (r_center = {v.get('r_center', float('nan')):.4f}, n_crops = {v['n_crops']})")
    o = boot["overall"]
    print(f"  overall: Â = {o['accuracy']:.4f}  CI [{o['ci_low']:.3f}, {o['ci_high']:.3f}]  (n_crops = {o['n_crops']})")
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
    return 0 if all(g.get("pass") for g in gates.values() if g["pass"] is not None) else 1


if __name__ == "__main__":
    sys.exit(main())

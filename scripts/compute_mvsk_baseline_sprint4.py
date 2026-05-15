"""
Compute the [mean, var, skew, kurtosis] 4-scalar baseline classifier accuracy
on the sprint-4 32^3 test set.

Design-doc obligation: A4 (sprint5_cprime_48cube.md v3 §6 gate-(e)).

Disclosure rule:
    MVSK-at-32^3 >= 0.42 -> threshold tightened vs sprint-4 [mean, var]
                            baseline of 0.368; disclose in §6 footnote +
                            branch-iv interpretation. NOT a blocker.
    MVSK-at-32^3 <  0.42 -> threshold preserved; no additional disclosure.

This script expects cached test/train crop tensors (shape [N, 32, 32, 32])
plus integer labels. If cached crops are absent in the sprint-4 disk artifact,
it surfaces a scope-decision BLOCKER and writes a disposition JSON.

Usage:
    python -u scripts/compute_mvsk_baseline_sprint4.py \
        --sprint4-dir cloud_runs/Sprint4-30ep-a13dce8-20260514-110740-e72fca \
        --out experiments/nerf/artifacts/eval/sprint5_cprime/\
mvsk_baseline_sprint4.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np


def _moments(x: np.ndarray) -> np.ndarray:
    """4-scalar [mean, var, skew, kurtosis (excess)] per crop.

    x: shape [N, D, H, W]
    returns: shape [N, 4]
    """
    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    mu = flat.mean(axis=1)
    var = flat.var(axis=1)
    std = np.sqrt(var) + 1e-12
    centered = flat - mu[:, None]
    skew = (centered ** 3).mean(axis=1) / (std ** 3)
    kurt = (centered ** 4).mean(axis=1) / (std ** 4) - 3.0
    return np.stack([mu, var, skew, kurt], axis=1)


def _train_eval_fc(
    train_feats: np.ndarray,
    train_y: np.ndarray,
    test_feats: np.ndarray,
    test_y: np.ndarray,
    n_classes: int = 4,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Train FC(4 -> 64 -> n_classes) on CPU; return (test_acc, confusion)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    # standardize features per-column on train stats
    mu = train_feats.mean(axis=0, keepdims=True)
    sd = train_feats.std(axis=0, keepdims=True) + 1e-8
    tr = (train_feats - mu) / sd
    te = (test_feats - mu) / sd

    tr_t = torch.from_numpy(tr.astype(np.float32))
    tr_y = torch.from_numpy(train_y.astype(np.int64))
    te_t = torch.from_numpy(te.astype(np.float32))
    te_y_np = test_y.astype(np.int64)

    net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, n_classes))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = tr_t.shape[0]
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = net(tr_t[idx])
            loss = loss_fn(logits, tr_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred = net(te_t).argmax(dim=1).cpu().numpy()
    acc = float((pred == te_y_np).mean())
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for y_true, y_pred in zip(te_y_np, pred):
        cm[y_true, y_pred] += 1
    return acc, cm


def _load_cached_crops(sprint4_dir: Path) -> dict | None:
    """Look for cached train/test crop tensors + labels under sprint4_dir.
    Returns dict with keys {train_x, train_y, test_x, test_y} or None.
    """
    candidates = [
        ("train_crops.npz", "test_crops.npz"),
        ("crops_train.npz", "crops_test.npz"),
    ]
    for tr_name, te_name in candidates:
        tr = sprint4_dir / "data" / tr_name
        te = sprint4_dir / "data" / te_name
        if tr.exists() and te.exists():
            ztr = np.load(tr)
            zte = np.load(te)
            return {
                "train_x": ztr["x"],
                "train_y": ztr["y"],
                "test_x": zte["x"],
                "test_y": zte["y"],
                "cache_source": f"{tr}, {te}",
            }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprint4-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    out_payload: dict = {
        "source": "sprint4 test crops at 32^3 on [D-49] split",
        "computed_at": _dt.date.today().isoformat(),
        "design_doc_obligation": (
            "A4 (§6 gate-(e) sprint5_cprime_48cube.md v3) — "
            "MVSK 4-scalar pre-flight"
        ),
        "mean_var_baseline_at_32cube_reference": 0.368,
        "disclosure_threshold": 0.42,
    }

    cached = _load_cached_crops(args.sprint4_dir)
    if cached is None:
        out_payload.update(
            {
                "status": "BLOCKER",
                "blocker_kind": "scope_decision",
                "blocker_reason": (
                    "Sprint-4 disk artifact contains only checkpoints + eval "
                    "JSONs + training log; no cached train/test crop tensors. "
                    "Computing the MVSK 4-scalar accuracy on the sprint-4 32^3 "
                    "test set requires either (a) re-extraction from Sherwood "
                    "rho at n_grid=768 on [D-49] axis=0 held-out region using "
                    "[D-50] CIC chunked-scatter at the same seeds (~6 min + "
                    "~30 sec FC train), (b) reading crop sha256 from a manifest "
                    "and pulling cached crops if such a manifest existed (it "
                    "does not in this artifact), or (c) deferring the A4 "
                    "pre-flight to the core-implementer (c′) dispatch which "
                    "will extract crops anyway."
                ),
                "scope_decision_options": ["(a) extract+train", "(b) cached pull (unavailable)", "(c) defer to (c′)"],
                "recommendation": (
                    "option (c) — A4 is a disclosure-only gate, not a blocker; "
                    "deferring it to the (c′) refactor avoids duplicate crop "
                    "extraction. The (c′) dispatch will produce a 48^3 MVSK "
                    "baseline at the same time as the 32^3 disclosure number "
                    "if instrumented to log both."
                ),
            }
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        print("[BLOCKER] no cached crops under sprint4 disk artifact")
        print(f"  wrote {args.out}")
        return 0

    train_feats = _moments(cached["train_x"])
    test_feats = _moments(cached["test_x"])
    acc, cm = _train_eval_fc(
        train_feats, cached["train_y"], test_feats, cached["test_y"]
    )
    delta_pp = acc - 0.368
    interp = "threshold tightened" if acc >= 0.42 else "threshold preserved"

    out_payload.update(
        {
            "status": "OK",
            "mvsk_at_32cube": acc,
            "delta_pp": delta_pp,
            "interpretation": interp,
            "confusion_matrix": cm.tolist(),
            "n_test": int(cached["test_x"].shape[0]),
            "n_train": int(cached["train_x"].shape[0]),
            "cache_source": cached["cache_source"],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"MVSK acc = {acc:.4f} (delta = {delta_pp:+.4f} pp); {interp}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
D70 Pre-flight A: frozen-init pre-activation histogram at the 8 ReLU
application sites in src/models/nerf.py.

Per spec: hook the 4 Linears in self.layers1 and 4 Linears in self.layers2;
capture the PRE-ReLU activation (= Linear output) and report per-site
dead-fraction = frac{units <= 0} over (batch * hidden_dim).

Mirrors the canonical model-construction in experiments/nerf/pipeline.py
(lines 695, 1042, 1807): IGMNeRF(hidden_dim=256, num_layers=8, L=10),
baseline (no velocity-gradient conditioning, no physics_embedding).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.nerf import IGMNeRF  # noqa: E402


SEEDS = [0, 1, 2]
N_COORDS = 4096
HIDDEN_DIM = 256
NUM_LAYERS = 8
L_PE = 10
OUT_DIR = REPO_ROOT / "cloud_runs" / "d70_preflight_A_dying_relu"
OUT_JSON = OUT_DIR / "hist.json"
OUT_PNG = OUT_DIR / "hist.png"
DEAD_THRESHOLD = 0.05


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        return "unknown"


def run_one_seed(seed: int):
    """Returns (site_dead_fractions[list[float] len 8], site_preacts[list[np.ndarray]])."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = IGMNeRF(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        L=L_PE,
    )
    model.eval()

    # 8 hook sites: layers1[0..3], layers2[0..3]
    site_modules = list(model.layers1) + list(model.layers2)
    assert len(site_modules) == 8, (
        f"Expected 8 Linear sites (4 layers1 + 4 layers2); got {len(site_modules)}"
    )

    captured: list[torch.Tensor | None] = [None] * 8
    hooks = []
    for i, m in enumerate(site_modules):
        def make_hook(idx):
            def hook(_mod, _inp, out):
                # out is the Linear output == PRE-ReLU activation
                captured[idx] = out.detach()
            return hook
        hooks.append(m.register_forward_hook(make_hook(i)))

    # Forward batch of (1, N, 3) since model expects (..., 3)
    coords = torch.rand(1, N_COORDS, 3)
    with torch.no_grad():
        _ = model(coords)

    for h in hooks:
        h.remove()

    dead_fracs = []
    preacts_flat = []
    for i, t in enumerate(captured):
        assert t is not None, f"site {i} not captured"
        flat = t.reshape(-1).cpu().numpy()
        dead_fracs.append(float((flat <= 0).mean()))
        preacts_flat.append(flat)

    return dead_fracs, preacts_flat


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    site_dead_per_seed: dict[str, list[float]] = {}
    # Keep preacts only for seed 0 for the plot (representative).
    preacts_seed0: list[np.ndarray] | None = None

    for seed in SEEDS:
        dead_fracs, preacts = run_one_seed(seed)
        site_dead_per_seed[f"seed_{seed}"] = dead_fracs
        if seed == 0:
            preacts_seed0 = preacts

    arr = np.array([site_dead_per_seed[f"seed_{s}"] for s in SEEDS])  # (3, 8)
    max_per_site = arr.max(axis=0).tolist()
    max_overall = float(arr.max())
    flat_idx = int(arr.argmax())
    seed_idx, site_idx = divmod(flat_idx, 8)
    site_names = [f"layers1[{i}]" for i in range(4)] + [f"layers2[{i}]" for i in range(4)]
    any_above = bool(max_overall > DEAD_THRESHOLD)

    out = {
        "seeds": SEEDS,
        "n_coords": N_COORDS,
        "hidden_dim": HIDDEN_DIM,
        "site_dead_fractions": site_dead_per_seed,
        "max_dead_fraction_across_seeds_per_site": max_per_site,
        "any_site_above_5_percent": any_above,
        "site_with_max_dead_fraction": {
            "site_index": site_idx,
            "site_name": site_names[site_idx],
            "max_dead_frac": max_overall,
        },
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # Plot 8-subplot grid using seed-0 preacts.
    assert preacts_seed0 is not None
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for i, ax in enumerate(axes.flat):
        data = preacts_seed0[i]
        ax.hist(data, bins=80, color="steelblue", alpha=0.85)
        ax.axvline(0.0, color="red", lw=1.2)
        df0 = site_dead_per_seed["seed_0"][i]
        ax.set_title(f"{site_names[i]}  dead={df0:.3f}", fontsize=10)
        ax.set_xlabel("pre-activation")
        ax.set_ylabel("count")
    fig.suptitle(
        f"D70 pre-flight A: pre-ReLU activations at frozen init (seed 0); "
        f"max dead frac across seeds = {max_overall:.3f} at {site_names[site_idx]}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)

    mechanism_supported = any_above
    print(
        f"dying-ReLU pre-flight: max_dead_frac={max_overall:.2f} at site {site_idx} "
        f"(seed {SEEDS[seed_idx]}); mechanism_supported={mechanism_supported}"
    )


if __name__ == "__main__":
    main()

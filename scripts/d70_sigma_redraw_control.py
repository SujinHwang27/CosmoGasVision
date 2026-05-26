"""
D70 σ_redraw control harness (per S4 footnote debt).

Spec
----
Fix a checkpoint from pre-flight C seed-0 (or an equivalent recently-trained
short pretrain), then perform 100 independent re-draws of the dead-fraction
measurement using a FRESH independent ``torch.Generator`` per redraw
(NOT the global RNG; the S4 origin trace at
``scripts/d70_preflight_C_relu_trajectory.py:139`` used global RNG via
``torch.rand`` — which is the confound this script isolates).

Reports:
  - std of dead-fraction across 100 re-draws (per site, and aggregate)
  - PASS if (σ_redraw / drift) < 0.2, where drift = 17-21pp (mid 19pp) from
    pre-flight C — i.e., LLN-confound contribution to the observed drift is
    structurally negligible.
  - FAIL otherwise → pre-flight C must be re-run with a fixed measurement-
    coord set.

CPU only, ~5 min budget.

Out-of-scope
------------
Does NOT touch the training-time drift estimate or pre-flight C results.
This is purely a measurement-noise floor probe.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.nerf import IGMNeRF  # noqa: E402

# --- Config -----------------------------------------------------------------
N_REDRAWS = 100
N_MEASURE_COORDS = 4096
HIDDEN_DIM = 256
NUM_LAYERS = 8
L_PE = 10

# Drift from pre-flight C seed-0 (17–21pp band per dispatch brief). Use mid.
PREFLIGHT_C_DRIFT_PP = 0.19  # 19 percentage points
SIGMA_RATIO_THRESHOLD = 0.20  # PASS if σ_redraw / drift < 0.20

# Short pretrain so the model is in a meaningful (drifted) state for the
# noise-floor probe. Mirrors pre-flight C's seed-0 training recipe minimally;
# we don't need byte-identical checkpoints, only a "deep enough" state where
# the dead-fraction has had time to drift.
N_PRETRAIN_STEPS = 100   # small but past the warmup regime
LR = 1e-4
MICROBATCH = 1024
SEED_TRAIN = 0
PRETRAIN_LOG_EPS = 1.0e-3

OUT_DIR = REPO_ROOT / "cloud_runs" / "d70_sigma_redraw_control"
OUT_JSON = OUT_DIR / "result.json"


def measure_dead_fracs_with_generator(model, generator, device):
    """Forward-hook dead-frac measurement with a CALLER-PROVIDED Generator.

    Mirrors d70_preflight_C measure_dead_fracs() except `torch.rand` is
    replaced with `torch.rand(..., generator=generator)` so each redraw is
    independent of global RNG state.
    """
    model.eval()
    site_modules = list(model.layers1) + list(model.layers2)
    n_sites = len(site_modules)
    captured = [None] * n_sites
    hooks = []
    for i, m in enumerate(site_modules):
        def make_hook(idx):
            def hook(_mod, _inp, out):
                captured[idx] = out.detach()
            return hook
        hooks.append(m.register_forward_hook(make_hook(i)))

    coords = torch.rand(
        1, N_MEASURE_COORDS, 3,
        generator=generator, device=device,
    )
    with torch.no_grad():
        _ = model(coords)
    for h in hooks:
        h.remove()

    dead = []
    for t in captured:
        flat = t.reshape(-1)
        dead.append(float((flat <= 0).float().mean().item()))
    model.train()
    return dead


def _build_drifted_model(device):
    """Train a small model briefly so dead-fraction has had time to drift
    off 0.5 — gives the σ_redraw probe a realistic operating point."""
    torch.manual_seed(SEED_TRAIN)
    np.random.seed(SEED_TRAIN)
    model = IGMNeRF(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, L=L_PE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED_TRAIN + 10_000)

    for _ in range(N_PRETRAIN_STEPS):
        # Synthetic surrogate: coords uniform [0,1]^3, rho_truth ~ lognormal.
        # This is a NOISE-FLOOR probe; the truth distribution is immaterial to
        # whether redraw variance is small relative to the trained-state drift.
        coords = torch.rand(MICROBATCH, 3, generator=gen, device=device)
        # Lognormal-ish truth via exp(N(0,1.5)) on a fresh generator draw.
        z = torch.randn(MICROBATCH, generator=gen, device=device) * 1.5
        rho_truth = torch.exp(z)
        out = model(coords.unsqueeze(0))
        rho_theta = out[0, :, 0]
        diff = torch.log10(rho_theta + PRETRAIN_LOG_EPS) - torch.log10(rho_truth + PRETRAIN_LOG_EPS)
        loss = (diff * diff).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    t0 = time.time()
    print(f"[d70_sigma_redraw] building drifted model (seed={SEED_TRAIN}, "
          f"{N_PRETRAIN_STEPS} steps, lr={LR})")
    model = _build_drifted_model(device)
    print(f"[d70_sigma_redraw] model trained in {time.time()-t0:.1f}s")

    # 100 independent re-draws with fresh Generators (seeded distinctly).
    print(f"[d70_sigma_redraw] running {N_REDRAWS} re-draws "
          f"(N_MEASURE_COORDS={N_MEASURE_COORDS}) with fresh torch.Generators")
    per_site_values = []  # list of length N_REDRAWS, each is list of 8 floats
    t1 = time.time()
    for r in range(N_REDRAWS):
        gen = torch.Generator(device=device)
        # Distinct seed per redraw; offset to avoid collision with training gen.
        gen.manual_seed(2_000_000 + r)
        dead = measure_dead_fracs_with_generator(model, gen, device)
        per_site_values.append(dead)

    arr = np.array(per_site_values)  # (N_REDRAWS, n_sites)
    per_site_std = arr.std(axis=0)
    per_site_mean = arr.mean(axis=0)

    # Aggregate σ_redraw: take the max across sites (worst case) — this is
    # the conservative estimate for the LLN-confound check.
    sigma_redraw_max = float(per_site_std.max())
    sigma_redraw_mean = float(per_site_std.mean())

    drift = PREFLIGHT_C_DRIFT_PP
    ratio_max = sigma_redraw_max / drift
    ratio_mean = sigma_redraw_mean / drift

    # PASS spec: σ_redraw / drift < 0.20 (use worst-site σ for the gate)
    verdict = "PASS" if ratio_max < SIGMA_RATIO_THRESHOLD else "FAIL"

    elapsed = time.time() - t1
    print(f"[d70_sigma_redraw] redraws done in {elapsed:.1f}s")
    print(f"[d70_sigma_redraw] per_site_mean = {per_site_mean.tolist()}")
    print(f"[d70_sigma_redraw] per_site_std  = {per_site_std.tolist()}")
    print(f"[d70_sigma_redraw] sigma_redraw_max  = {sigma_redraw_max:.6f}")
    print(f"[d70_sigma_redraw] sigma_redraw_mean = {sigma_redraw_mean:.6f}")
    print(f"[d70_sigma_redraw] drift (pre-flight C)  = {drift:.4f}")
    print(f"[d70_sigma_redraw] ratio_max  = {ratio_max:.4f}  (PASS if < {SIGMA_RATIO_THRESHOLD})")
    print(f"[d70_sigma_redraw] ratio_mean = {ratio_mean:.4f}")
    print(f"[d70_sigma_redraw] VERDICT={verdict}")

    out = {
        "spec": (
            "100 independent torch.Generator re-draws of dead-fraction "
            "measurement; gates σ_redraw / drift < 0.2 (worst-site σ)."
        ),
        "n_redraws": N_REDRAWS,
        "n_measure_coords": N_MEASURE_COORDS,
        "n_pretrain_steps": N_PRETRAIN_STEPS,
        "lr": LR,
        "preflight_c_drift_pp": drift,
        "sigma_ratio_threshold": SIGMA_RATIO_THRESHOLD,
        "per_site_mean": per_site_mean.tolist(),
        "per_site_std": per_site_std.tolist(),
        "sigma_redraw_max": sigma_redraw_max,
        "sigma_redraw_mean": sigma_redraw_mean,
        "ratio_max": ratio_max,
        "ratio_mean": ratio_mean,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[d70_sigma_redraw] wrote {OUT_JSON}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

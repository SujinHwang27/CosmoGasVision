"""Stage-1 smoke driver for the exp/feedback-latent SIREN auto-decoder ([F-02] A5+A6).

Runs the §4(c) smoke ladder (s2 overfit / s3 mini-run + step-100 contract) and the
§4(d) anti-collapse controls (c1 shared-z / c2 swap-test / c3 shuffled-assignment /
c4 label-shuffle-null), plus a lambda_z sweep whose pin is DERIVED (derivation-at-
spec-time): lambda_z is chosen where c1 separates while conditioned Q does not
degrade. Every threshold is anchored to a RECORDED floor written to the results
JSON. SMOKE ONLY: small sample counts, CPU/MPS, no data-product / paper claims.

Run:  PYTHONPATH=. .venv/bin/python experiments/feedback-latent/stage1_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from src.analysis.nccf import gaussian_smooth_periodic, pearson
from src.data.feedback_cube_provider import FeedbackCubeProvider
from src.models.feedback_field import FeedbackField

# --------------------------------------------------------------------------- #
# Constants of record
# --------------------------------------------------------------------------- #
EXPERIMENT_NAME = "CosmoGasVision/feedback-latent"
RUN_NAME = "Stage1-Smoke"
MANDATORY_TAGS = {
    "model_type": "conditioned-field",
    "stage": "Stage1-Smoke",
    "physics_id": "1,2,3,4",
    "redshift": "0.3",
}

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
RESULTS_JSON = ARTIFACT_DIR / "stage1_smoke_results.json"

BOX_MPC_H = 60.0
SCORE_N = 64                      # downsampled full-box scoring grid (periodic-safe)
SIGMA_MPC_H = 2.0                 # r_s smoothing scale (sigma=2 h^-1 Mpc)
# [D-49] split mirrored on the 64^3 scoring grid: val = [int(.7*64), int(.85*64))
VAL_LO, VAL_HI = int(0.70 * SCORE_N), int(0.85 * SCORE_N)   # [44, 54)

D_MODEL = 8
MINI_STEPS = 500
S2_STEPS = 200
BATCH_PER_VARIANT = 1024
LR = 1e-4                        # F0 audit (stage1_lr_audit.json): 1e-3 too high for
#                                  SIREN(omega0=30) — stuck at mean-floor; 1e-4 overfits
#                                  a fixed batch to ~6e-4. lr is a run-config, gates unchanged.
SEEDS = [0, 1, 2]
LAMBDA_SWEEP = [0.0, 1e-4, 1e-3, 1e-2]
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# MLflow (mandatory nullcontext fallback)
# --------------------------------------------------------------------------- #
def _mlflow_run():
    try:
        import mlflow

        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow.start_run(run_name=RUN_NAME, tags=MANDATORY_TAGS)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] MLflow unavailable ({exc!r}); nullcontext.", flush=True)
        return nullcontext()


# --------------------------------------------------------------------------- #
# scoring cubes (full-box 64^3 periodic smoothing, pearson restricted to val)
# --------------------------------------------------------------------------- #
def _full_x_cube(prov: FeedbackCubeProvider, p: int) -> np.ndarray:
    slabs = [prov.region_cube(p, r)[0] for r in ("train", "val", "test")]
    return np.concatenate(slabs, axis=0)          # (192,192,192)


def _downsample(x192: np.ndarray, n: int = SCORE_N) -> np.ndarray:
    f = x192.shape[0] // n
    return x192.reshape(n, f, n, f, n, f).mean(axis=(1, 3, 5))


def _score_grid_coords() -> torch.Tensor:
    ax = (np.arange(SCORE_N) + 0.5) / SCORE_N * 2.0 - 1.0     # cell centers -> [-1,1]
    ii, jj, kk = np.meshgrid(ax, ax, ax, indexing="ij")
    grid = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
    return torch.as_tensor(grid, dtype=torch.float32, device=DEVICE)


def _decode_cube(model: FeedbackField, coords: torch.Tensor,
                 z: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(coords, z=z).cpu().numpy().reshape(SCORE_N, SCORE_N, SCORE_N)
    model.train()
    return out


def _r_s(dec_cube: np.ndarray, truth_cube: np.ndarray) -> float:
    ds = gaussian_smooth_periodic(dec_cube, BOX_MPC_H, SIGMA_MPC_H)
    ts = gaussian_smooth_periodic(truth_cube, BOX_MPC_H, SIGMA_MPC_H)
    return pearson(ds[VAL_LO:VAL_HI], ts[VAL_LO:VAL_HI])


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _fixed_batch(prov, variants, n, seed):
    coords, targets, idx = [], [], []
    for vi, p in enumerate(variants):
        c, x = prov.sample(p, "train", n, seed=seed + p)
        coords.append(c)
        targets.append(x)
        idx.append(torch.full((n,), vi, dtype=torch.long))
    return (torch.cat(coords).to(DEVICE), torch.cat(targets).to(DEVICE),
            torch.cat(idx).to(DEVICE))


def _train(prov, variants, steps, seed, lambda_z, shared_z=False,
           label_shuffle=False, fixed_batch=None, record_at=()):
    """Train a conditioned (or shared-z / label-shuffled) model.

    Returns (model, {step: loss}, code_z[n_variants,d]).
    shared_z: one shared code for all variants (via explicit-z forward, c1).
    label_shuffle: random variant labels per example (c4 null).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FeedbackField(d=D_MODEL, n_variants=len(variants)).to(DEVICE)
    shared_code = None
    if shared_z:
        shared_code = torch.nn.Parameter(torch.zeros(D_MODEL, device=DEVICE))
        params = list(model.net.parameters()) + list(model.out_layer.parameters()) + [shared_code]
    else:
        params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    loss_hist = {}
    rng = np.random.default_rng(seed + 777)
    for step in range(1, steps + 1):
        if fixed_batch is not None:
            coords, targets, idx = fixed_batch
        else:
            coords, targets, idx = _fixed_batch(prov, variants, BATCH_PER_VARIANT,
                                                seed=step * 13 + seed)
        if label_shuffle:
            perm = torch.as_tensor(rng.integers(0, len(variants), size=idx.shape[0]),
                                   dtype=torch.long, device=DEVICE)
            idx = perm
        opt.zero_grad()
        if shared_z:
            pred = model(coords, z=shared_code)
            reg = model.code_prior_explicit(shared_code, lambda_z)
        else:
            pred = model(coords, variant_idx=idx)
            reg = model.code_prior(idx, lambda_z)
        loss = (pred - targets).pow(2).mean() + reg
        loss.backward()
        opt.step()
        if step == 1 or step in record_at or step == steps:
            loss_hist[step] = float(loss.detach())

    if shared_z:
        code_z = shared_code.detach().unsqueeze(0).repeat(len(variants), 1)
    else:
        code_z = model.codes.weight.detach().clone()
    return model, loss_hist, code_z


def _Q_per_variant(model, coords, code_z, truth_cubes, variants,
                   assign=None):
    """r_s(sigma=2) per variant; assign maps variant slot -> code row (for swap/shuffle)."""
    Q = {}
    for vi, p in enumerate(variants):
        row = vi if assign is None else assign[vi]
        dec = _decode_cube(model, coords, code_z[row])
        Q[p] = _r_s(dec, truth_cubes[p])
    return Q


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prov = FeedbackCubeProvider()
    variants = prov.loaded_variants
    print(f"[smoke] loaded variants {variants}", flush=True)
    assert len(variants) >= 2, "need >=2 cubes"

    coords = _score_grid_coords()
    truth_cubes = {p: _downsample(_full_x_cube(prov, p)) for p in variants}

    results = {
        "spec": "f02_founding_ratification.md §4(c) ladder + §4(d) controls",
        "scoring": {
            "grid_n": SCORE_N, "box_mpc_h": BOX_MPC_H, "sigma_mpc_h": SIGMA_MPC_H,
            "estimator": "nccf.gaussian_smooth_periodic + nccf.pearson (r_s)",
            "val_voxel_bounds_on_grid": [VAL_LO, VAL_HI],
            "note": "full-box 64^3 periodic smoothing (periodic-safe); pearson "
                    "restricted to [D-49] val slab; truth block-mean downsampled 192->64.",
        },
        "config": {"d": D_MODEL, "lr": LR, "batch_per_variant": BATCH_PER_VARIANT,
                   "mini_steps": MINI_STEPS, "s2_steps": S2_STEPS, "seeds": SEEDS,
                   "variants": variants},
    }

    with _mlflow_run():
        # ------------------------------------------------------------------ #
        # s2 — overfit one fixed batch (2 variants), gates at 50 / 200
        # ------------------------------------------------------------------ #
        s2_vars = variants[:2]
        fb = _fixed_batch(prov, s2_vars, BATCH_PER_VARIANT, seed=42)
        _, targets, idx = fb
        # recorded floors: per-variant-mean predictor (mean-floor) + target_std
        mean_floor_pred = torch.empty_like(targets)
        for vi in range(len(s2_vars)):
            m = idx == vi
            mean_floor_pred[m] = targets[m].mean()
        mse_mean_floor = float((targets - mean_floor_pred).pow(2).mean())
        target_std = float(targets.std())

        model_s2, hist_s2, _ = _train(prov, s2_vars, S2_STEPS, seed=0, lambda_z=0.0,
                                      fixed_batch=fb, record_at=(50,))
        with torch.no_grad():
            pred50_model, _, _ = model_s2, None, None  # model already at step 200
        # re-evaluate pred_std at 50 by retraining-free proxy: use final-model pred
        # (we captured loss@50; for pred_std@50 re-run a 50-step model)
        model_s2_50, _, _ = _train(prov, s2_vars, 50, seed=0, lambda_z=0.0, fixed_batch=fb)
        with torch.no_grad():
            pred_std_50 = float(model_s2_50(fb[0], variant_idx=fb[2]).std())

        loss50 = hist_s2.get(50)
        loss200 = hist_s2.get(S2_STEPS)
        s2 = {
            "recorded_floors": {"mse_mean_floor": mse_mean_floor,
                                "target_std": target_std},
            "loss_50": loss50, "loss_200": loss200, "pred_std_50": pred_std_50,
            "gate_i_nontriviality": {"value": loss50, "bar": 0.9 * mse_mean_floor,
                                     "pass": loss50 < 0.9 * mse_mean_floor},
            "gate_ii_anticollapse": {"value": pred_std_50, "bar": 0.1 * target_std,
                                     "pass": pred_std_50 > 0.1 * target_std},
            "gate_iii_memorization": {"value": loss200, "bar": 0.5 * mse_mean_floor,
                                      "pass": loss200 <= 0.5 * mse_mean_floor},
        }
        s2["GREEN"] = all(s2[k]["pass"] for k in
                          ("gate_i_nontriviality", "gate_ii_anticollapse",
                           "gate_iii_memorization"))
        results["s2"] = s2
        print(f"[smoke] s2 GREEN={s2['GREEN']} loss50={loss50:.4f} "
              f"floor={mse_mean_floor:.4f} pred_std50={pred_std_50:.4f}", flush=True)

        # ------------------------------------------------------------------ #
        # s3 — mini-run (2 variants, >=500 steps) + LOUD step-100 contract
        # ------------------------------------------------------------------ #
        # fixed val batch for the pred_std contract
        vc, vx = prov.sample(s3_p := s2_vars[0], "val", 2048, seed=999)
        vc = vc.to(DEVICE)
        model_s3, hist_s3, _ = _train(prov, s2_vars, MINI_STEPS, seed=0, lambda_z=0.0,
                                      record_at=(100,))
        with torch.no_grad():
            pred_std_val = float(model_s3(vc, variant_idx=torch.zeros(vc.shape[0],
                                          dtype=torch.long, device=DEVICE)).std())
        loss1, loss100 = hist_s3[1], hist_s3[100]
        # LOUD contract assertion — OUTSIDE any try/except, raises on breach
        assert pred_std_val > 0.01 and loss100 < loss1, (
            f"S3 step-100 contract BREACH: pred_std_val={pred_std_val:.4f} "
            f"(>0.01?), loss100={loss100:.4f} < loss1={loss1:.4f}?")
        results["s3"] = {
            "loss_1": loss1, "loss_100": loss100, "loss_final": hist_s3[MINI_STEPS],
            "pred_std_val": pred_std_val,
            "step100_contract": {"pred_std_val_gt_0p01": pred_std_val > 0.01,
                                 "loss100_lt_loss1": loss100 < loss1, "pass": True},
            "GREEN": True,
        }
        print(f"[smoke] s3 GREEN loss1={loss1:.4f} loss100={loss100:.4f} "
              f"pred_std_val={pred_std_val:.4f}", flush=True)

        # ------------------------------------------------------------------ #
        # lambda_z sweep — pin where c1 separates while conditioned Q holds
        # ------------------------------------------------------------------ #
        sweep = []
        for lz in LAMBDA_SWEEP:
            m_c, _, z_c = _train(prov, variants, MINI_STEPS, seed=0, lambda_z=lz)
            Qc = _Q_per_variant(m_c, coords, z_c, truth_cubes, variants)
            m_s, _, z_s = _train(prov, variants, MINI_STEPS, seed=0, lambda_z=lz,
                                 shared_z=True)
            Qs = _Q_per_variant(m_s, coords, z_s, truth_cubes, variants)
            Qc_mean = float(np.mean(list(Qc.values())))
            Qs_mean = float(np.mean(list(Qs.values())))
            sweep.append({"lambda_z": lz, "Q_conditioned_mean": Qc_mean,
                          "Q_shared_mean": Qs_mean, "c1_separation": Qc_mean - Qs_mean,
                          "Q_conditioned_per_variant": Qc})
            print(f"[smoke] sweep lz={lz:g} Qc={Qc_mean:.4f} Qs={Qs_mean:.4f} "
                  f"sep={Qc_mean - Qs_mean:.4f}", flush=True)

        best_Qc = max(s["Q_conditioned_mean"] for s in sweep)
        eligible = [s for s in sweep if s["Q_conditioned_mean"] >= best_Qc - 0.02]
        pinned = max(eligible, key=lambda s: s["c1_separation"])
        lambda_z = pinned["lambda_z"]
        results["lambda_z_sweep"] = {
            "ladder": sweep,
            "pinned_lambda_z": lambda_z,
            "rationale": ("pinned at the lambda_z with largest c1 separation among "
                          "those whose conditioned Q is within 0.02 of the best "
                          "conditioned Q (fit not degraded). Derivation-at-spec-time; "
                          "no magic default."),
            "Q_degradation_tolerance": 0.02,
        }
        print(f"[smoke] pinned lambda_z={lambda_z:g}", flush=True)

        # ------------------------------------------------------------------ #
        # controls c1/c2 at >=3 seeds with pinned lambda_z
        # ------------------------------------------------------------------ #
        Qc_seeds, Qshared_seeds, Qswap_seeds = [], [], []
        cond_models = {}
        for sd in SEEDS:
            m_c, _, z_c = _train(prov, variants, MINI_STEPS, seed=sd, lambda_z=lambda_z)
            cond_models[sd] = (m_c, z_c)
            Qc = _Q_per_variant(m_c, coords, z_c, truth_cubes, variants)
            Qc_seeds.append(np.mean(list(Qc.values())))

            m_s, _, z_s = _train(prov, variants, MINI_STEPS, seed=sd, lambda_z=lambda_z,
                                 shared_z=True)
            Qs = _Q_per_variant(m_s, coords, z_s, truth_cubes, variants)
            Qshared_seeds.append(np.mean(list(Qs.values())))

            # c2 swap: decode variant i with mean of j!=i codes' r_s
            swap_deg = []
            for vi, p in enumerate(variants):
                others = [j for j in range(len(variants)) if j != vi]
                q_swaps = [_r_s(_decode_cube(m_c, coords, z_c[j]), truth_cubes[p])
                           for j in others]
                swap_deg.append(Qc[p] - float(np.mean(q_swaps)))
            Qswap_seeds.append(np.mean(swap_deg))

        Qc_seeds = np.array(Qc_seeds)
        seed_sd = float(Qc_seeds.std(ddof=1))
        bar = 2.0 * seed_sd
        c1_sep = float(Qc_seeds.mean() - np.mean(Qshared_seeds))
        c2_deg = float(np.mean(Qswap_seeds))
        results["controls"] = {
            "seed_SD_of_Q_conditioned": seed_sd,
            "bar_2x_seedSD": bar,
            "c1_shared_z": {"Q_conditioned_mean": float(Qc_seeds.mean()),
                            "Q_shared_mean": float(np.mean(Qshared_seeds)),
                            "separation": c1_sep, "bar": bar, "pass": c1_sep > bar},
            "c2_swap_test": {"mean_degradation": c2_deg, "bar": bar,
                             "pass": c2_deg > bar},
        }
        print(f"[smoke] c1 sep={c1_sep:.4f} c2 deg={c2_deg:.4f} bar(2sd)={bar:.4f}",
              flush=True)

        # ------------------------------------------------------------------ #
        # c3 shuffled-assignment (diagnostic) + c4 label-shuffle null (diagnostic)
        # ------------------------------------------------------------------ #
        m_c0, z_c0 = cond_models[SEEDS[0]]
        rngp = np.random.default_rng(0)
        perm = list(rngp.permutation(len(variants)))
        while perm == list(range(len(variants))):
            perm = list(rngp.permutation(len(variants)))
        Q_correct = _Q_per_variant(m_c0, coords, z_c0, truth_cubes, variants)
        Q_shuffled = _Q_per_variant(m_c0, coords, z_c0, truth_cubes, variants,
                                    assign=perm)
        # pairwise code separation: conditioned vs label-shuffle null
        def _pair_dist(z):
            zt = z if isinstance(z, torch.Tensor) else torch.as_tensor(z)
            ds = [float((zt[i] - zt[j]).norm()) for i in range(zt.shape[0])
                  for j in range(i + 1, zt.shape[0])]
            return ds
        m_c4, _, z_c4 = _train(prov, variants, MINI_STEPS, seed=0, lambda_z=lambda_z,
                               label_shuffle=True)
        results["controls"]["c3_shuffled_assignment"] = {
            "permutation": perm,
            "Q_correct_mean": float(np.mean(list(Q_correct.values()))),
            "Q_shuffled_mean": float(np.mean(list(Q_shuffled.values()))),
            "degrades": float(np.mean(list(Q_correct.values())))
                        > float(np.mean(list(Q_shuffled.values()))),
            "note": "diagnostic",
        }
        results["controls"]["c4_label_shuffle_null"] = {
            "conditioned_pairwise_code_L2": _pair_dist(z_c0),
            "labelshuffle_pairwise_code_L2": _pair_dist(z_c4),
            "conditioned_mean_pair_L2": float(np.mean(_pair_dist(z_c0))),
            "labelshuffle_mean_pair_L2": float(np.mean(_pair_dist(z_c4))),
            "note": "diagnostic: under shuffled labels z_p should NOT separate "
                    "(labelshuffle_mean should be << conditioned_mean).",
        }

        # ------------------------------------------------------------------ #
        # per-variant central-slab figures (decoded vs truth x)
        # ------------------------------------------------------------------ #
        try:
            fig_paths = _render_figures(m_c0, coords, z_c0, truth_cubes, variants)
            results["figures"] = fig_paths
        except Exception as exc:  # noqa: BLE001 — figures are non-critical; never lose results
            results["figures"] = {"error": repr(exc)}
            print(f"[smoke] figure rendering failed (non-fatal): {exc!r}", flush=True)

        # log key metrics to MLflow if available
        try:
            import mlflow
            if mlflow.active_run() is not None:
                mlflow.log_metrics({
                    "s2_loss50": loss50, "s2_loss200": loss200,
                    "s3_pred_std_val": pred_std_val,
                    "pinned_lambda_z": lambda_z,
                    "c1_separation": c1_sep, "c2_degradation": c2_deg,
                    "seed_SD": seed_sd,
                })
        except Exception:  # noqa: BLE001
            pass

    def _jsonable(o):  # numpy scalars/arrays -> native python (int64 etc.)
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=_jsonable))
    print(f"[smoke] wrote {RESULTS_JSON}", flush=True)
    print(f"[smoke] SUMMARY s2GREEN={results['s2']['GREEN']} "
          f"s3GREEN={results['s3']['GREEN']} lambda_z={lambda_z:g} "
          f"c1={c1_sep:.4f}/pass={results['controls']['c1_shared_z']['pass']} "
          f"c2={c2_deg:.4f}/pass={results['controls']['c2_swap_test']['pass']}",
          flush=True)
    return 0


def _render_figures(model, coords, code_z, truth_cubes, variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid = SCORE_N // 2
    paths = {}
    for vi, p in enumerate(variants):
        dec = _decode_cube(model, coords, code_z[vi])
        truth = truth_cubes[p]
        vmin = float(min(dec[mid].min(), truth[mid].min()))
        vmax = float(max(dec[mid].max(), truth[mid].max()))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(truth[mid], vmin=vmin, vmax=vmax, cmap="viridis")
        ax[0].set_title(f"P{p} truth x (slab)")
        ax[1].imshow(dec[mid], vmin=vmin, vmax=vmax, cmap="viridis")
        ax[1].set_title(f"P{p} decoded x")
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        out = ARTIFACT_DIR / f"stage1_slab_P{p}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        paths[f"P{p}"] = str(out)
    return paths


if __name__ == "__main__":
    sys.exit(main())

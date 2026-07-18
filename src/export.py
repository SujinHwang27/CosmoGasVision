"""CosmoGasVision outbound data-export boundary.

This module is the ONE audited boundary through which CosmoGasVision data
leaves the project for external consumers (per the ``data-export`` skill,
``.claude/skills/data-export/SKILL.md``). Every export here:

- ingests simulation truth via the canonical :class:`SherwoodLoader` (never
  raw ``np.load`` on sim data);
- loads model predictions from a torch checkpoint + the existing differentiable
  renderer (:func:`volume_render_physics`) — a checkpoint is NOT sim data, so
  ``torch.load`` is correct there;
- is deterministic, type-hinted, and computes in float64;
- carries a ``_validate_*`` bounds/finiteness guard;
- writes its artifact PLUS a ``.provenance.json`` sidecar (git-stamped, with an
  honest [D-73]-verb-ceiling caveat) via :func:`write_export`.

The first consumer serviced is ``selements-website`` (public-facing curation
site); its landing dir is ``results/exports/selements-website/``.

CLAIM-BEARING EXPORTS inherit the [D-73] close-out verb-ceiling. The production
MLP is a CHARACTERIZATION of the optimization/identifiability wall at z=0.3 — it
"fails two of three directly-evaluated [D-13] gates" (P_F + flux-PDF KS); the 3D
xi gate was NEVER evaluated on the MLP in its defined form, so "fails all three
gates" is BARRED. See SKILL.md + ``experiments/nerf/LEDGER.md`` §3 [D-73].
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.analysis.flux_pdf import compute_flux_pdf, ks_distance
from src.analysis.p_flux import compute_p_flux
from src.data.loader import SherwoodLoader
from src.models.nerf import IGMNeRF, volume_render_physics
from src.utils.provenance import get_git_info

# --------------------------------------------------------------------------- #
# Canonical landing root (per SKILL.md). One subdir per consumer.
# --------------------------------------------------------------------------- #
EXPORT_ROOT = "results/exports"

# --------------------------------------------------------------------------- #
# Fiducial [D-13] evaluation geometry. These are EVAL-geometry constants: the
# production MLP was TRAINED at n_rays=64 but all published numbers (P_F, KS,
# mean_F) are at eval n_rays=1024 / eval_seed=42. (Caveat 1 in the request:
# NEVER "trained on 1024 dense sightlines".)
# --------------------------------------------------------------------------- #
N_RAYS_EVAL: int = 1024
EVAL_SEED: int = 42
REDSHIFT: float = 0.3
BOX_KPC_H: float = 60000.0  # 60 Mpc/h box; documented in CLAUDE.md astro conventions.
N_RAYS_AVAIL: int = 16384  # num_los in the los2048_n16384 sightline file.

# Production pub-t1 model constructor geometry (from each run's MLflow params;
# identical across P1-P4 — only the checkpoint / tau_amp differ per physics).
_HIDDEN_DIM: int = 256
_NUM_LAYERS: int = 8
_L_FOURIER: int = 10
_TAU_MAX: float = 10.0
_RENDER_WINDOW: int = 64

# Flux-PDF / KS restriction window (Bolton+2008 / Lee+2015 convention; [D-13]).
F_RANGE: Tuple[float, float] = (0.05, 0.95)

# The four pub-t1 production checkpoints (P1 fiducial + P2-P4). Read tau_amp
# from each checkpoint's log_tau_amp, NEVER hardcode across physics.
PUB_T1_CHECKPOINTS: Dict[str, str] = {
    "P1": "cloud_runs/pub-t1-extracted/P1-N64-S0-1778430089-7f65fe/checkpoints/step_050000.pt",
    "P2": "cloud_runs/pub-t1-extracted/P2-N64-S0-1778430089-0f7fc8/checkpoints/step_050000.pt",
    "P3": "cloud_runs/pub-t1-extracted/P3-N64-S0-1778430089-b9dad4/checkpoints/step_050000.pt",
    "P4": "cloud_runs/pub-t1-extracted/P4-N64-S0-1778430089-b18fc5/checkpoints/step_050000.pt",
}
PHYSICS_ID: Dict[str, int] = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

# Banked artifacts (READ, do not recompute).
D44_BOOTSTRAP_JSON: str = (
    "experiments/nerf/artifacts/eval/d44_bootstrap/d44_bootstrap_KS_meanF.json"
)

# seed=42 single-seed per-cell anchors (LEDGER §3 gate table, lines 329-332).
# The P_F band-mean residual is the published |ΔP_F/P_F| aggregate; the export
# RE-RUNS P_F and asserts reproduction of this value within tolerance.
SEED42_ANCHORS: Dict[str, Dict[str, float]] = {
    "P1": {"mean_F": 0.97895, "KS": 0.0325, "P_F_residual": 0.4155},
    "P2": {"mean_F": 0.97536, "KS": 0.0742, "P_F_residual": 0.3757},
    "P3": {"mean_F": 0.97819, "KS": 0.0408, "P_F_residual": 0.3591},
    "P4": {"mean_F": 0.98154, "KS": 0.0389, "P_F_residual": 0.3613},
}

# Kirkman+2007 observed mean-flux anchor at z=0.3 ([D-34]; LEDGER line 193).
MEAN_F_OBS: float = 0.979
MEAN_F_OBS_ERR: float = 0.005

# [D-13] mean_F PASS gate band (LEDGER §3 gate table header). The [D-44]
# bootstrap "2/4 PASS" verdict is evaluated against THIS band (a CI whose q84
# reaches into [MEANF_GATE_LO, MEANF_GATE_HI] passes; one whose q84 is below
# MEANF_GATE_LO fails), NOT against the point value 0.979. Getting this right is
# what makes the honest 2/4 statement (P1/P2 fail, P3/P4 marginal) reproduce.
MEANF_GATE_LO: float = 0.974
MEANF_GATE_HI: float = 0.984

# [D-13] P_F band for the aggregate |ΔP_F/P_F| residual, k_|| in
# [10^-2.5, 10^-1.5] s/km.
PF_BAND_LO: float = 10 ** -2.5
PF_BAND_HI: float = 10 ** -1.5


# =========================================================================== #
# Shared render / eval machinery (mirrors scripts/d73_a7_control_recompute.py
# + scripts/eval_partial_d13.py; the renderer is NOT reinvented here).
# =========================================================================== #
def _seed42_selection(n_rays_eval: int = N_RAYS_EVAL) -> np.ndarray:
    """Deterministic held-out ray selection: the exact [D-13] eval draw.

    Reproduces ``rng = np.random.default_rng(seed=42); sel =
    rng.choice(N_RAYS_AVAIL, size=n_rays_eval, replace=False); sel.sort()`` —
    the same draw at scripts/eval_partial_d13.py:73-75 that produced every
    published P_F / KS / mean_F number. Returned sorted ascending; index 0 is
    the deterministic "representative ray" for the single-sightline figure.
    """
    rng = np.random.default_rng(seed=EVAL_SEED)
    sel = rng.choice(N_RAYS_AVAIL, size=int(n_rays_eval), replace=False)
    sel.sort()
    return sel


def _load_pub_t1_model(ckpt_path: str) -> Tuple[IGMNeRF, float]:
    """Load a pub-t1 production checkpoint -> (model.eval(), tau_amp float64).

    ``torch.load`` is correct here (a checkpoint is not sim data). tau_amp is
    ``exp(log_tau_amp)`` read from the checkpoint per physics — never hardcoded.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = IGMNeRF(hidden_dim=_HIDDEN_DIM, num_layers=_NUM_LAYERS, L=_L_FOURIER)
    model.load_state_dict(state["model_state"])
    model.eval()
    tau_amp = float(np.exp(np.float64(state["log_tau_amp"])))
    return model, tau_amp


def _render_flux_mlp_and_truth(
    physics_cell: str,
    n_rays_eval: int = N_RAYS_EVAL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Render MLP flux + load truth flux on the held-out seed=42 rays.

    Truth-side tau via the canonical :class:`SherwoodLoader` path (never raw
    ``np.load`` on sim data). Model-side tau via the checkpoint + the existing
    :func:`volume_render_physics`.

    RENDER CONVENTION — reproduces the PUBLISHED [D-13] numbers EXACTLY.
    The banked P_F=0.4155 / KS / mean_F numbers were produced by
    ``scripts/eval_partial_d13.py`` -> ``_render_tau_for_model``, which calls
    ``volume_render_physics(model, coords, vel_axis)`` with ``tau_amp=None``
    (=> the renderer default 1.0) and NO tau_max clamp on the rendered tau.
    Empirically (verified 2026-07-09): this convention reproduces the published
    P1 band residual 0.4155 to 4 decimals; applying the checkpoint tau_amp
    (1.0455) + a tau_max clamp instead yields 0.4414 (the a7-recompute
    convention — a DIFFERENT, self-consistent number, NOT the published one).
    We therefore render at tau_amp=1.0, no clamp, to be byte-faithful to the
    figure the website consumes. The per-physics checkpoint tau_amp IS still
    read from ``log_tau_amp`` and surfaced (never hardcoded) for the record.

    Returns
    -------
    F_mlp : (n_rays, n_bins) float64  (= exp(-tau_pred), tau_amp=1.0, no clamp)
    F_truth : (n_rays, n_bins) float64  (= exp(-tau_truth))
    vel_axis : (n_bins,) float64 (km/s)
    sel : (n_rays,) int64 — the sorted seed=42 ray indices used.
    tau_amp_ckpt : float — exp(log_tau_amp) read from the checkpoint (recorded
        for provenance; NOT applied to the rendered flux, per the published
        convention above).
    """
    ckpt = PUB_T1_CHECKPOINTS[physics_cell]
    phys_id = PHYSICS_ID[physics_cell]

    loader = SherwoodLoader("Sherwood")
    sl = loader.load_sightlines(phys_id, REDSHIFT)
    tau_truth_full = np.asarray(sl["tau_h1"], dtype=np.float64)
    vel_axis = np.asarray(sl["vel_axis"], dtype=np.float64)
    box_kpc_h = float(sl["header"]["box_kpc_h"])
    coords_world = loader.get_world_coordinates(sl)  # (N, n_bins, 3) world kpc/h

    sel = _seed42_selection(n_rays_eval)
    tau_truth = tau_truth_full[sel]
    coords = torch.tensor(coords_world[sel] / box_kpc_h, dtype=torch.float32)
    vel_axis_t = torch.tensor(vel_axis, dtype=torch.float32)

    model, tau_amp_ckpt = _load_pub_t1_model(ckpt)
    with torch.no_grad():
        # Published convention: tau_amp defaults to 1.0, no tau_max clamp.
        tau_pred = volume_render_physics(
            model, coords, vel_axis=vel_axis_t, window=_RENDER_WINDOW, z=REDSHIFT,
        )
    tau_pred_np = tau_pred.cpu().numpy().astype(np.float64)

    F_mlp = np.exp(-tau_pred_np)
    F_truth = np.exp(-tau_truth)
    return F_mlp, F_truth, vel_axis, sel.astype(np.int64), float(tau_amp_ckpt)


def _pf_band_residual(
    centers: np.ndarray, pf_mlp: np.ndarray, pf_truth: np.ndarray
) -> float:
    """Band-mean |ΔP_F/P_F| over k_|| in [10^-2.5, 10^-1.5] s/km ([D-13]).

    This is the published aggregate residual (LEDGER gate table): the mean over
    in-band k of |P_F_mlp - P_F_truth| / P_F_truth. NOT "the power spectrum is
    4x off" — it is a band-mean fractional error (caveat 4).
    """
    band = (centers >= PF_BAND_LO) & (centers <= PF_BAND_HI)
    band &= np.isfinite(pf_mlp) & np.isfinite(pf_truth) & (pf_truth > 0)
    if not band.any():
        return float("nan")
    frac = np.abs(pf_mlp[band] - pf_truth[band]) / pf_truth[band]
    return float(np.mean(frac.astype(np.float64)))


# =========================================================================== #
# Validation guards
# =========================================================================== #
def _validate_flux(name: str, F: np.ndarray) -> None:
    if not np.isfinite(F).all():
        raise AssertionError(f"{name}: non-finite flux values present.")
    fmin, fmax = float(np.min(F)), float(np.max(F))
    if fmin < 0.0 or fmax > 1.0 + 1e-9:
        raise AssertionError(f"{name}: flux out of [0,1]: min={fmin}, max={fmax}.")


def _validate_pf(name: str, pf: np.ndarray) -> None:
    finite = pf[np.isfinite(pf)]
    if finite.size and float(np.min(finite)) < 0.0:
        raise AssertionError(f"{name}: negative P_F present ({float(np.min(finite))}).")


def _validate_rows(name: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise AssertionError(f"{name}: empty row set.")
    for i, r in enumerate(rows):
        for k, v in r.items():
            if isinstance(v, float) and not np.isfinite(v):
                raise AssertionError(f"{name}: non-finite value at row {i} key {k!r}.")


# =========================================================================== #
# Write helpers (tidy CSV, no comment lines; provenance sidecar).
# =========================================================================== #
def _write_csv(path: Path, fieldnames: List[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Write a tidy CSV with full float64 precision (repr; no rounding)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def write_export(
    out_dir: Path,
    filename: str,
    fieldnames: Optional[List[str]],
    rows: Optional[Sequence[Dict[str, Any]]],
    producing_fn: str,
    source_data_path: str,
    physics_id: Any,
    caveat: str,
    extra_sidecar: Optional[Dict[str, Any]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Write an artifact (CSV or JSON) + its git-stamped provenance sidecar.

    A non-``unknown`` git SHA is REQUIRED. Raises if git is unavailable so an
    orphan (unprovenanced) export can never ship.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / filename

    if json_payload is not None:
        with open(artifact, "w", encoding="utf-8") as fh:
            json.dump(json_payload, fh, indent=2)
    else:
        assert fieldnames is not None and rows is not None
        _write_csv(artifact, fieldnames, rows)

    git = get_git_info()
    if git.get("commit", "unknown") == "unknown":
        raise RuntimeError(
            "provenance git SHA resolved to 'unknown'; refusing to ship an "
            "unprovenanced export (SKILL.md mandatory requirement)."
        )
    sidecar: Dict[str, Any] = {
        "artifact": filename,
        "producing_function": producing_fn,
        "source_data_path": source_data_path,
        "git": git,
        "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_rays_eval": N_RAYS_EVAL,
        "eval_seed": EVAL_SEED,
        "physics_id": physics_id,
        "redshift": REDSHIFT,
        "box_kpc_h": BOX_KPC_H,
        "caveat": caveat,
    }
    if extra_sidecar:
        sidecar.update(extra_sidecar)
    sidecar_path = out_dir / (filename + ".provenance.json")
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
    return artifact, sidecar_path


# =========================================================================== #
# Figure 1 — pf-miss: RE-RUN P_F(k_||) MLP vs truth at P1 (+ optional P2-P4).
# =========================================================================== #
def export_pf_miss(
    out_dir: str,
    cells: Sequence[str] = ("P1",),
    n_rays_eval: int = N_RAYS_EVAL,
    pf_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Small-scale flux-power miss: P_F(k_||) MLP vs truth over the measured
    k_|| range at P1, plus the band-mean |ΔP_F/P_F| residual per cell.

    Scientific purpose: show the production MLP misses the [D-13] small-scale
    flux-power gate. The band-mean |ΔP_F/P_F| in k_|| in [10^-2.5, 10^-1.5]
    s/km is ~0.42 at P1 (4.2x over the 10% gate) — i.e. "misses the gate by
    ~4x the tolerance", NOT "the power spectrum is 4x off".

    The full P_F(k_||) curve columns are emitted for the FIRST cell in
    ``cells`` (P1 fiducial). The band residual scalar is emitted for every
    requested cell. The P1 band residual is asserted to reproduce the banked
    0.4155 within ``pf_tolerance``.
    """
    cells = list(cells)
    out = Path(out_dir)
    lead_cell = cells[0]

    curve_rows: List[Dict[str, Any]] = []
    band_scalars: Dict[str, Dict[str, float]] = {}
    tau_amp_ckpt: Dict[str, float] = {}
    lead_centers = lead_pf_mlp = lead_pf_truth = None

    for cell in cells:
        F_mlp, F_truth, vel_axis, _sel, tau_amp = _render_flux_mlp_and_truth(cell, n_rays_eval)
        tau_amp_ckpt[cell] = tau_amp
        _validate_flux(f"pf-miss/{cell}/F_mlp", F_mlp)
        _validate_flux(f"pf-miss/{cell}/F_truth", F_truth)

        centers, pf_mlp = compute_p_flux(F_mlp, vel_axis)
        _, pf_truth = compute_p_flux(F_truth, vel_axis)
        _validate_pf(f"pf-miss/{cell}/P_F_mlp", pf_mlp)
        _validate_pf(f"pf-miss/{cell}/P_F_truth", pf_truth)

        residual = _pf_band_residual(centers, pf_mlp, pf_truth)
        band_scalars[cell] = {
            "abs_delta_PF_over_PF_in_band": residual,
            "banked_seed42_P_F_residual": SEED42_ANCHORS[cell]["P_F_residual"],
        }

        if cell == lead_cell:
            lead_centers, lead_pf_mlp, lead_pf_truth = centers, pf_mlp, pf_truth
            # compute_p_flux returns NaN for EMPTY log-k bins (documented). The
            # "measured k_|| range" is the set of bins with finite P_F in BOTH
            # curves; we emit only those (no NaN in a downstream-parseable CSV),
            # honestly restricting to the measured range rather than padding.
            measured = np.isfinite(pf_mlp) & np.isfinite(pf_truth)
            for k, pm, pt in zip(centers[measured], pf_mlp[measured], pf_truth[measured]):
                curve_rows.append({
                    "k_parallel_s_per_km": float(k),
                    "P_F_mlp": float(pm),
                    "P_F_truth": float(pt),
                })

    # Reproduction assertion for P1 (mandatory): band residual must reproduce
    # the banked 0.4155 aggregate within tolerance.
    p1_residual = band_scalars["P1"]["abs_delta_PF_over_PF_in_band"]
    banked = SEED42_ANCHORS["P1"]["P_F_residual"]
    if not np.isfinite(p1_residual):
        raise AssertionError("pf-miss: P1 band residual is non-finite.")
    if abs(p1_residual - banked) > pf_tolerance:
        raise AssertionError(
            f"pf-miss: P1 reproduced |ΔP_F/P_F|={p1_residual:.4f} deviates from "
            f"banked {banked:.4f} by more than tol={pf_tolerance:.4f}."
        )

    _validate_rows("pf-miss/curve", curve_rows)
    caveat = (
        "RE-RUN at the [D-13] fiducial eval geometry (n_rays=1024, eval_seed=42; "
        "transverse separation 60/sqrt(1024)=1.875 ~1.9 h^-1 Mpc, denser than "
        "CLAMATO's 2.37 h^-1 Mpc). The production MLP was TRAINED at n_rays=64 — "
        "this is an EVAL-geometry statement, NEVER 'trained on 1024 dense "
        "sightlines'. The band-mean |dP_F/P_F| ~0.42 (P1) is 4.2x the [D-13] 10% "
        "small-scale flux-power gate: the MLP MISSES THE SMALL-SCALE FLUX-POWER "
        "GATE BY ~4x THE TOLERANCE (a ~42% band-mean fractional error), NOT 'the "
        "power spectrum is 4x off'. Single realization / fixed cosmology (Sherwood, "
        "one 60 cMpc/h box); z=0.3 scope-lock (no claim beyond z=0.3; CLAMATO/"
        "TARDIS succeed at z~2-3). This is a characterization of the under-"
        "constrained z=0.3 flux inverse problem under this FGPA forward model."
    )
    measured_k = [r["k_parallel_s_per_km"] for r in curve_rows]
    extra = {
        "band_k_parallel_s_per_km": [PF_BAND_LO, PF_BAND_HI],
        "band_residual_per_cell": band_scalars,
        "abs_delta_PF_over_PF_in_band": p1_residual,
        "measured_k_range_s_per_km": [min(measured_k), max(measured_k)] if measured_k else [],
        "n_measured_kbins": len(curve_rows),
        "n_empty_kbins_dropped": int(len(lead_centers) - len(curve_rows)),
        "empty_kbin_note": (
            "compute_p_flux returns NaN for empty log-k bins; the CSV emits only "
            "the finite (measured) bins. Dropped bins carried no rays in that "
            "log-k interval — not a numerical failure."
        ),
        "reproduces_banked_seed42_P_F_residual_P1": {
            "reproduced": p1_residual,
            "banked": banked,
            "abs_diff": abs(p1_residual - banked),
            "tolerance": pf_tolerance,
            "within_tolerance": True,
        },
        "cells_exported": cells,
        "pf_binning": "src.analysis.p_flux.compute_p_flux defaults (k in [1e-3,1e-1], 20 log bins)",
        "render_convention": (
            "published eval_partial_d13 convention: tau_amp=1.0 (renderer "
            "default), no tau_max clamp — reproduces banked 0.4155 to 4 dp."
        ),
        "checkpoint_tau_amp_per_cell": tau_amp_ckpt,
        "checkpoint_tau_amp_note": (
            "tau_amp read from each checkpoint's log_tau_amp (never hardcoded), "
            "recorded here; NOT applied to the rendered flux under the published "
            "P_F convention. Applying it (+clamp) is the a7-recompute convention "
            "and yields 0.4414 for P1 — a different, self-consistent number."
        ),
    }
    artifact, sidecar = write_export(
        out_dir=out,
        filename="fig1-pf-miss.csv",
        fieldnames=["k_parallel_s_per_km", "P_F_mlp", "P_F_truth"],
        rows=curve_rows,
        producing_fn="src.export.export_pf_miss",
        source_data_path=(
            "Sherwood/Physics{1..N}_*/{los2048,tauH1}_n16384_z0.300.dat (truth) "
            "+ cloud_runs/pub-t1-extracted/P*-N64-*/checkpoints/step_050000.pt (MLP)"
        ),
        physics_id=[PHYSICS_ID[c] for c in cells],
        caveat=caveat,
        extra_sidecar=extra,
    )

    # Companion "4/4 cells" band table: one row per exported cell. The claim
    # this supports is "P_F 0/4 PASS: |dP_F/P_F| 36-42%, 3.6x-4.2x over the 10%
    # gate, 4/4 cells" (LEDGER §3). Emits only the cells actually rendered this
    # run; deferred cells are simply absent (honest — no fabricated rows).
    table_rows = [
        {
            "physics": c,
            "abs_delta_PF_over_PF_in_band": float(
                band_scalars[c]["abs_delta_PF_over_PF_in_band"]
            ),
            "banked_seed42_P_F_residual": float(
                band_scalars[c]["banked_seed42_P_F_residual"]
            ),
        }
        for c in cells
    ]
    _validate_rows("pf-miss/band-table", table_rows)
    table_caveat = (
        "P_F small-scale flux-power band-mean residual |dP_F/P_F| over k_|| in "
        "[10^-2.5, 10^-1.5] s/km, per physics cell, at n_rays=1024/eval_seed=42 "
        "(EVAL geometry; MLP TRAINED at n_rays=64). All exported cells fail the "
        "[D-13] 10% gate at 3.6x-4.2x the tolerance (0/4 PASS) — 'misses the "
        "small-scale flux-power gate by ~4x the tolerance', NOT 'the power "
        "spectrum is 4x off'. Single realization / fixed cosmology; z=0.3 "
        "scope-lock. This is a characterization of the under-constrained z=0.3 "
        "flux inverse problem under this FGPA forward model."
    )
    table_artifact, table_sidecar = write_export(
        out_dir=out,
        filename="fig1-band-table.csv",
        fieldnames=["physics", "abs_delta_PF_over_PF_in_band", "banked_seed42_P_F_residual"],
        rows=table_rows,
        producing_fn="src.export.export_pf_miss (companion band table)",
        source_data_path=(
            "Sherwood/Physics{1..4}_*/{los2048,tauH1}_n16384_z0.300.dat (truth) "
            "+ cloud_runs/pub-t1-extracted/P*-N64-*/checkpoints/step_050000.pt (MLP)"
        ),
        physics_id=[PHYSICS_ID[c] for c in cells],
        caveat=table_caveat,
        extra_sidecar={
            "band_k_parallel_s_per_km": [PF_BAND_LO, PF_BAND_HI],
            "pf_gate_tolerance": 0.10,
            "cells_exported": cells,
            "cells_deferred": [c for c in ("P1", "P2", "P3", "P4") if c not in cells],
            "render_convention": "published eval_partial_d13: tau_amp=1.0, no tau_max clamp",
            "checkpoint_tau_amp_per_cell": tau_amp_ckpt,
        },
    )

    return {
        "artifact": str(artifact),
        "sidecar": str(sidecar),
        "band_table_artifact": str(table_artifact),
        "band_table_sidecar": str(table_sidecar),
        "n_curve_rows": len(curve_rows),
        "p1_band_residual": p1_residual,
        "banked_p1_residual": banked,
        "band_scalars": band_scalars,
    }


# =========================================================================== #
# Figure 2a — mean-flux table: ASSEMBLE FROM BANKED (no recompute).
# =========================================================================== #
def export_mean_flux_table(
    out_dir: str,
    bootstrap_json: str = D44_BOOTSTRAP_JSON,
) -> Dict[str, Any]:
    """Mean transmitted flux per physics: seed=42 anchor + [D-44] 5-seed
    sightline-bootstrap CI, assembled entirely from BANKED artifacts.

    Reads the [D-44] bootstrap JSON (meanF_mean / q16 / q84) + the LEDGER
    seed=42 per-cell mean_F anchors. Does NOT recompute. The Kirkman+2007
    observed anchor <F>=0.979+/-0.005 is carried in the sidecar.
    """
    with open(bootstrap_json, "r", encoding="utf-8") as fh:
        d44 = json.load(fh)
    results_full = d44["results_full"]

    rows: List[Dict[str, Any]] = []
    for cell in ("P1", "P2", "P3", "P4"):
        r = results_full[cell]
        rows.append({
            "physics": cell,
            "mean_F_pred_seed42": float(SEED42_ANCHORS[cell]["mean_F"]),
            "mean_F_bootstrap_mean": float(r["meanF_mean"]),
            "mean_F_bootstrap_lo": float(r["meanF_q16"]),
            "mean_F_bootstrap_hi": float(r["meanF_q84"]),
        })
    _validate_rows("mean-flux-table", rows)

    # Honest softening (caveat 2): seed=42 is 4/4 PASS in-band; the 5-seed
    # sightline bootstrap is 2/4 against the [D-13] mean_F gate band
    # [0.974, 0.984]. A cell PASSES the bootstrap gate if its CI q84 reaches
    # the band lower edge (0.974); it FAILS if q84 < 0.974 (CI entirely below
    # the band). Per LEDGER §3: P1/P2 CIs lie below the band (FAIL); P3/P4 are
    # marginal at q84 (PASS). This reproduces the LEDGER's 2/4 verdict — NOT a
    # test against the point value 0.979, which all four q84 sit below.
    bootstrap_gate = {}  # cell -> "PASS" | "FAIL(CI-below-band)"
    ci_below_band = []
    ci_below_point = []
    for cell in ("P1", "P2", "P3", "P4"):
        r = results_full[cell]
        q84 = float(r["meanF_q84"])
        if q84 < MEANF_GATE_LO:
            bootstrap_gate[cell] = "FAIL(CI-below-band)"
            ci_below_band.append(cell)
        else:
            bootstrap_gate[cell] = "PASS(marginal-at-q84)"
        if q84 < MEAN_F_OBS:
            ci_below_point.append(cell)
    n_bootstrap_pass = sum(1 for v in bootstrap_gate.values() if v.startswith("PASS"))

    caveat = (
        "mean_F is 4/4 PASS at the seed=42 single-seed eval but only 2/4 at the "
        "[D-44] 5-seed sightline bootstrap against the [D-13] gate band "
        "[0.974, 0.984]: P1 and P2 bootstrap CIs lie ENTIRELY BELOW the band "
        "(q84 < 0.974); P3/P4 are marginal (q84 just inside the band). All four "
        "bootstrap CIs sit below the Kirkman+2007 point anchor <F>=0.979. Any "
        "'reproduces the average flux' claim MUST carry this softening. mean_F_obs "
        "= 0.979 +/- 0.005 (Kirkman+2007, z=0.3). Single realization / fixed "
        "cosmology (Sherwood, one 60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "mean_F_obs": MEAN_F_OBS,
        "mean_F_obs_err": MEAN_F_OBS_ERR,
        "mean_F_obs_source": "Kirkman et al. 2007, MNRAS 376:1227 ([D-34])",
        "meanF_gate_band": [MEANF_GATE_LO, MEANF_GATE_HI],
        "bootstrap_spec": "[D-44] K=5 seeds {42-46}, sightline-level bootstrap, n_rays_eval=1024",
        "bootstrap_source_json": bootstrap_json,
        "bootstrap_gate_verdict_per_cell": bootstrap_gate,
        "n_bootstrap_pass_of_4": n_bootstrap_pass,
        "cells_bootstrap_ci_below_gate_band": ci_below_band,
        "cells_bootstrap_ci_below_point_anchor": ci_below_point,
        "seed42_anchor_source": "experiments/nerf/LEDGER.md §3 gate table (lines 329-332)",
        "meanF_per_seed": {c: results_full[c]["meanF_per_seed"] for c in ("P1", "P2", "P3", "P4")},
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig2a-mean-flux-table.csv",
        fieldnames=[
            "physics", "mean_F_pred_seed42",
            "mean_F_bootstrap_mean", "mean_F_bootstrap_lo", "mean_F_bootstrap_hi",
        ],
        rows=rows,
        producing_fn="src.export.export_mean_flux_table",
        source_data_path=bootstrap_json + " + experiments/nerf/LEDGER.md §3 gate table",
        physics_id=[1, 2, 3, 4],
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {
        "artifact": str(artifact),
        "sidecar": str(sidecar),
        "n_rows": len(rows),
        "n_bootstrap_pass_of_4": n_bootstrap_pass,
        "cells_ci_below_gate_band": ci_below_band,
        "cells_ci_below_point_anchor": ci_below_point,
    }


# =========================================================================== #
# Figure 2b — flux-pdf: RE-RUN histograms at P1 (+ banked KS scalars P1-P4).
# =========================================================================== #
def export_flux_pdf(
    out_dir: str,
    cell: str = "P1",
    n_rays_eval: int = N_RAYS_EVAL,
    bootstrap_json: str = D44_BOOTSTRAP_JSON,
) -> Dict[str, Any]:
    """Flux PDF p(F) MLP vs truth restricted to F in [0.05, 0.95] at P1, plus
    the KS distance per physics.

    The histogram arrays (F_bin_center, pdf_mlp, pdf_truth) are RE-RUN for the
    requested cell (P1). The per-physics KS scalars are the banked seed=42
    LEDGER anchors + the [D-44] bootstrap KS means (read, not recomputed).
    """
    F_mlp, F_truth, _vel_axis, _sel, tau_amp_ckpt = _render_flux_mlp_and_truth(cell, n_rays_eval)
    _validate_flux(f"flux-pdf/{cell}/F_mlp", F_mlp)
    _validate_flux(f"flux-pdf/{cell}/F_truth", F_truth)

    # Histogram edges over the [D-13] KS restriction window F in [0.05, 0.95].
    f_lo, f_hi = F_RANGE
    F_bins = np.linspace(f_lo, f_hi, 50)
    centers_mlp, pdf_mlp = compute_flux_pdf(F_mlp, F_bins)
    centers_truth, pdf_truth = compute_flux_pdf(F_truth, F_bins)
    assert np.allclose(centers_mlp, centers_truth)

    # KS distance re-computed for the exported cell (self-consistency); banked
    # KS scalars carried for all four cells in the sidecar.
    ks_cell = ks_distance(F_mlp, F_truth, F_range=F_RANGE)

    rows: List[Dict[str, Any]] = [
        {"F_bin_center": float(c), "pdf_mlp": float(pm), "pdf_truth": float(pt)}
        for c, pm, pt in zip(centers_mlp, pdf_mlp, pdf_truth)
    ]
    _validate_rows("flux-pdf/hist", rows)

    with open(bootstrap_json, "r", encoding="utf-8") as fh:
        d44 = json.load(fh)
    ks_banked = {
        c: {
            "KS_seed42": float(SEED42_ANCHORS[c]["KS"]),
            "KS_bootstrap_mean": float(d44["results_full"][c]["KS_mean"]),
        }
        for c in ("P1", "P2", "P3", "P4")
    }

    caveat = (
        f"RE-RUN histograms at n_rays=1024, eval_seed=42, F in [{f_lo},{f_hi}] "
        "(Bolton+2008/Lee+2015 window). KS is 3/4 PASS at the [D-13] gate (P2 "
        "fails at 0.0742). This is one of the two directly-evaluated [D-13] gates "
        "the production MLP is scored on (P_F fails 4/4; flux-PDF KS 3/4 pass); "
        "the 3D xi gate was NEVER evaluated on the MLP in its defined form, so "
        "'fails all three gates' is BARRED. Single realization / fixed cosmology "
        "(Sherwood, one 60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "F_range": list(F_RANGE),
        "n_bins_hist": len(rows),
        "KS_distance_exported_cell": ks_cell,
        "KS_per_physics_banked": ks_banked,
        "banked_source": bootstrap_json + " + LEDGER §3 gate table",
        "pdf_normalization": "density over the [0.05,0.95] window (integrates to 1)",
        "render_convention": "published eval_partial_d13: tau_amp=1.0, no tau_max clamp",
        "checkpoint_tau_amp": tau_amp_ckpt,
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig2b-flux-pdf.csv",
        fieldnames=["F_bin_center", "pdf_mlp", "pdf_truth"],
        rows=rows,
        producing_fn="src.export.export_flux_pdf",
        source_data_path=(
            f"Sherwood/Physics{PHYSICS_ID[cell]}_*/{{los2048,tauH1}}_n16384_z0.300.dat "
            f"(truth) + {PUB_T1_CHECKPOINTS[cell]} (MLP)"
        ),
        physics_id=PHYSICS_ID[cell],
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {
        "artifact": str(artifact),
        "sidecar": str(sidecar),
        "n_hist_bins": len(rows),
        "ks_exported_cell": ks_cell,
        "ks_banked": ks_banked,
    }


# =========================================================================== #
# Figure 3 — single representative sightline (optional).
# =========================================================================== #
def export_single_sightline(
    out_dir: str,
    cell: str = "P1",
    n_rays_eval: int = N_RAYS_EVAL,
) -> Dict[str, Any]:
    """One representative P1 sightline: v (km/s), F_mlp, F_truth.

    Determinism: the ray is the FIRST index of the sorted seed=42 selection
    (sel[0]). Labelled clearly as ONE representative ray — not a typical
    spectrum.
    """
    F_mlp, F_truth, vel_axis, sel, tau_amp_ckpt = _render_flux_mlp_and_truth(cell, n_rays_eval)
    _validate_flux(f"single-sightline/{cell}/F_mlp", F_mlp)
    _validate_flux(f"single-sightline/{cell}/F_truth", F_truth)

    ray_local = 0  # first row of the sorted selection
    ray_global = int(sel[ray_local])
    f_mlp_ray = F_mlp[ray_local]
    f_truth_ray = F_truth[ray_local]

    rows: List[Dict[str, Any]] = [
        {"v_km_per_s": float(v), "F_mlp": float(fm), "F_truth": float(ft)}
        for v, fm, ft in zip(vel_axis, f_mlp_ray, f_truth_ray)
    ]
    _validate_rows("single-sightline", rows)

    caveat = (
        f"ONE REPRESENTATIVE RAY (global sightline index {ray_global} = first of "
        "the sorted seed=42 n_rays=1024 selection), NOT a typical spectrum and "
        "NOT hand-picked for agreement. Eval geometry n_rays=1024, eval_seed=42; "
        "the production MLP was TRAINED at n_rays=64. Single realization / fixed "
        "cosmology (Sherwood, one 60 cMpc/h box); z=0.3 scope-lock. Illustrative "
        "of the under-constrained z=0.3 flux inverse problem under this FGPA "
        "forward model; a single agreeing/ disagreeing ray is not a gate."
    )
    extra = {
        "representative_ray_global_index": ray_global,
        "representative_ray_selection_rule": "sel[0] of sorted seed=42 selection",
        "n_bins": len(rows),
        "render_convention": "published eval_partial_d13: tau_amp=1.0, no tau_max clamp",
        "checkpoint_tau_amp": tau_amp_ckpt,
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig3-single-sightline.csv",
        fieldnames=["v_km_per_s", "F_mlp", "F_truth"],
        rows=rows,
        producing_fn="src.export.export_single_sightline",
        source_data_path=(
            f"Sherwood/Physics{PHYSICS_ID[cell]}_*/{{los2048,tauH1}}_n16384_z0.300.dat "
            f"(truth) + {PUB_T1_CHECKPOINTS[cell]} (MLP)"
        ),
        physics_id=PHYSICS_ID[cell],
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {
        "artifact": str(artifact),
        "sidecar": str(sidecar),
        "n_bins": len(rows),
        "representative_ray_global_index": ray_global,
    }


# =========================================================================== #
# ep04 "the-direct-attack" batch — the [D-40] saturation-aware P_F band-loss
# intervention and its D1 (amplitude-shrink, shape-preserved) collapse.
# All four exports are BANKED-source: they read decision-record scalars and the
# on-disk per-bin diagnostic JSONs; nothing is recomputed from sim data.
# =========================================================================== #

# Banked per-bin diagnostic artifacts ([D-40] Addendum 1; READ, do not recompute).
D40_PF_PER_BIN_JSON: str = (
    "experiments/nerf/artifacts/eval/sat_aware_hypc/"
    "87dcf9e63564465489f770266fcec197_pf_per_bin.json"
)
D40_BASELINE_PER_BIN_JSON: str = (
    "experiments/nerf/artifacts/eval/sat_aware_hypc/pubt1_xphysics_pf_perbin.json"
)

# [D-40] verdict-of-record scalars (decision record 2026-05-10 + Addendum 1).
# The training-time loss_pf_band values are ENDPOINTS ONLY, at the precision the
# decision record banked them: the full per-step trajectory did not survive the
# compute-site round-trip (the local MLflow store holds only the 50-step smoke
# run ea2eec25...; the compute-site file store for run 87dcf9e6... was never
# imported; the banked step-10000 checkpoint carries no loss history).
D40_VERDICT_OF_RECORD: Dict[str, Any] = {
    "pf_residual_band_mean": {"sat_aware": 0.5707, "reference": 0.4155,
                              "reference_source": "production baseline P1 (50k-step schedule)",
                              "gate": 0.10},
    "ks_distance": {"sat_aware": 0.1888, "reference": 0.0325,
                    "reference_source": "production baseline P1 (50k-step schedule)",
                    "gate": 0.05},
    "loss_data_at_step_10000": {"sat_aware": 0.0100, "reference": 0.0025,
                                "reference_source": "preview run at matched step count",
                                "gate": None},
    "train_loss_pf_band_start": {"sat_aware": 0.99, "reference": None,
                                 "reference_source": None, "gate": None},
    "train_loss_pf_band_final": {"sat_aware": 6.77e-06, "reference": None,
                                 "reference_source": None, "gate": None},
}
D40_TRAIN_STEPS: int = 12500

# [D-40] intervention spec of record (three added loss terms + run config +
# pre-committed stop discipline).
D40_RUN_CONFIG: List[Tuple[str, Any]] = [
    ("intervention", "three loss terms added to the production objective"),
    ("loss_term_1", "saturation-band weighting of the data loss (weight 3.0)"),
    ("loss_term_2", "rank-order pairwise surrogate (weight 0.1, 512 pairs)"),
    ("loss_term_3", "band-integrated relative flux-power residual (weight 1.0)"),
    ("physics", "P1 (fiducial) only"),
    ("train_n_rays", 64),
    ("train_microbatch", 1024),
    ("train_steps", 12500),
    ("train_seed", 0),
    ("mean_flux_anchor", 0.979),
    ("eval_n_rays", 1024),
    ("eval_seed", 42),
    ("stop_rule", "pre-committed FAIL criterion fixed before dispatch"),
    ("follow_up_cancelled", "all-four-physics full-schedule stage cancelled on falsification"),
    ("cost_saved_estimate", "~$4-6 / ~6.6 GPU-hr / ~1h37m wall"),
]


def _read_banked_json(path: str) -> Dict[str, Any]:
    """Read a banked diagnostic JSON (tolerates NaN literals per json spec ext)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def export_d40_verdict_table(out_dir: str) -> Dict[str, Any]:
    """ep04 fig1: the training-vs-eval contrast as a verdict table (BANKED).

    The episode's decisive datum: the added training term collapsed five orders
    of magnitude (0.99 -> 6.77e-6 over 12,500 steps) while the eval flux-power
    residual WORSENED (0.5707 vs the 0.4155 production baseline, +37.4%) and
    the flux-PDF KS gate broke (0.1888 vs the 0.05 gate). Endpoints only; the
    per-step trajectory is not recoverable (see D40_VERDICT_OF_RECORD note).
    """
    rows: List[Dict[str, Any]] = []
    for metric, d in D40_VERDICT_OF_RECORD.items():
        rows.append({
            "metric": metric,
            "sat_aware_run": d["sat_aware"],
            "reference_value": d["reference"] if d["reference"] is not None else "",
            "reference_source": d["reference_source"] or "",
            "gate": d["gate"] if d["gate"] is not None else "",
        })
    _validate_rows("d40-verdict-table", rows)

    pf = D40_VERDICT_OF_RECORD["pf_residual_band_mean"]
    worsening_pct = 100.0 * (pf["sat_aware"] - pf["reference"]) / pf["reference"]

    caveat = (
        "Verdict table of record for the saturation-aware flux-power band-loss "
        "intervention (P1 only, single seed, 12,500 steps at minimal-intervention "
        "weights). The training-time band-loss values are ENDPOINTS of record "
        "(0.99 -> 6.77e-6); the full per-step trajectory did not survive the "
        "compute-site round-trip, so the five-orders descent must be cited as "
        "endpoints, never drawn as a curve. The five-orders descent is NOT a "
        "win: eval flux-power residual worsened +37.4% vs the production "
        "baseline and the flux-PDF KS gate broke (3.8x over). Reference values "
        "for P_F/KS come from the 50k-step production baseline (the intervention "
        "ran 12,500 steps; the FAIL verdict is the pre-committed criterion -- "
        "verdict-level, with the schedule difference disclosed); loss_data is "
        "compared at matched step count. Single realization / fixed cosmology "
        "(Sherwood, one 60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "train_steps": D40_TRAIN_STEPS,
        "pf_worsening_pct_vs_baseline": worsening_pct,
        "ks_over_gate_factor": D40_VERDICT_OF_RECORD["ks_distance"]["sat_aware"] / 0.05,
        "trajectory_recoverability": (
            "endpoints only -- local tracker holds the 50-step smoke, not this "
            "run; compute-site metric store never round-tripped; banked "
            "step-10000 checkpoint has no loss history"
        ),
        "internal_lineage": "[D-40] + Addendum 1; MLflow run 87dcf9e6...; compute job 197319 (train) / 197328 (eval)",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig1-verdict-table.csv",
        fieldnames=["metric", "sat_aware_run", "reference_value", "reference_source", "gate"],
        rows=rows,
        producing_fn="src.export.export_d40_verdict_table",
        source_data_path="decision record (banked scalars; see internal_lineage)",
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows),
            "pf_worsening_pct": worsening_pct}


def export_d40_pf_per_bin(
    out_dir: str,
    per_bin_json: str = D40_PF_PER_BIN_JSON,
) -> Dict[str, Any]:
    """ep04 fig2: per-bin P_F table behind the D1 mechanism (BANKED JSON).

    Ships k, P_F_pred (sat-aware), P_F_truth, pred/truth ratio, relative
    difference, and the in-gate-band flag. Window-null NaN bins are dropped
    (matching the ep03 fig1-pf-miss.csv k-axis so the site can overlay the
    production-baseline curve directly). Re-derives the banked headline
    diagnostics from the shipped rows and asserts consistency.
    """
    d = _read_banked_json(per_bin_json)
    k = np.asarray(d["k_axis"], dtype=np.float64)
    p_pred = np.asarray(d["P_pred"], dtype=np.float64)
    p_truth = np.asarray(d["P_truth"], dtype=np.float64)
    ratio = np.asarray(d["P_pred_over_P_truth"], dtype=np.float64)
    rel = np.asarray(d["rel_diff_per_bin"], dtype=np.float64)
    in_band = np.asarray(d["in_band_mask"], dtype=bool)
    band_lo, band_hi = (float(x) for x in d["pf_band_s_per_km"])

    if not np.isclose(band_lo, PF_BAND_LO) or not np.isclose(band_hi, PF_BAND_HI):
        raise AssertionError("banked band edges disagree with the published P_F band.")

    keep = np.isfinite(p_truth) & np.isfinite(p_pred)
    if in_band[~keep].any():
        raise AssertionError("a window-null bin lies inside the gate band; refusing to drop it.")
    k, p_pred, p_truth, ratio, rel, in_band = (
        a[keep] for a in (k, p_pred, p_truth, ratio, rel, in_band)
    )
    _validate_pf("d40-per-bin P_pred", p_pred)
    _validate_pf("d40-per-bin P_truth", p_truth)
    if not (np.diff(k) > 0).all() or float(k[0]) <= 0.0:
        raise AssertionError("k axis not positive/ascending after NaN drop.")

    # Consistency: re-derive the banked headline diagnostics from shipped rows.
    diag = d["diagnostics"]
    band_rel = np.abs(rel[in_band])
    if not np.isclose(float(band_rel.mean()), float(diag["mean_abs_rel_diff_in_band"]), atol=1e-12):
        raise AssertionError("re-derived in-band |rel diff| mean disagrees with banked diagnostics.")
    lp, lt = np.log10(p_pred[in_band]), np.log10(p_truth[in_band])
    pearson = float(np.corrcoef(lp, lt)[0, 1])
    if not np.isclose(pearson, float(diag["pearson_log_in_band"]), atol=1e-9):
        raise AssertionError("re-derived in-band log-Pearson disagrees with banked diagnostics.")
    if not np.isclose(float(ratio[in_band].mean()), float(diag["band_ratio_mean"]), atol=1e-12):
        raise AssertionError("re-derived in-band ratio mean disagrees with banked diagnostics.")

    rows = [
        {
            "k_parallel_s_per_km": float(k[i]),
            "P_F_sat_aware": float(p_pred[i]),
            "P_F_truth": float(p_truth[i]),
            "pred_over_truth": float(ratio[i]),
            "rel_diff": float(rel[i]),
            "in_gate_band": int(in_band[i]),
        }
        for i in range(k.size)
    ]
    _validate_rows("d40-pf-per-bin", rows)

    caveat = (
        "Per-bin flux power P_F(k_par) for the saturation-aware intervention "
        "run vs truth (P1, eval n_rays=1024/seed=42). The mechanism table "
        "behind the failure signature: in the gate band the model preserves the "
        "SHAPE of P_F (log-space Pearson 0.8346) while rendering it at ~0.43x "
        "the true amplitude (band ratio mean 0.4293) -- amplitude-shrink with "
        "shape preservation, NOT a constant-prediction collapse, and NOT the "
        "flux->1 transparency collapse of the later interventions. Three "
        "window-null NaN bins dropped (k-axis matches the ep03 fig1-pf-miss.csv "
        "17-bin axis; overlay the production baseline from that file -- do not "
        "re-ship it). Single realization / fixed cosmology (Sherwood, one "
        "60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "pf_band_s_per_km": [band_lo, band_hi],
        "n_bins_shipped": int(k.size),
        "n_nan_bins_dropped": int((~keep).sum()),
        "diagnostics_banked": diag,
        "pearson_log_rederived": pearson,
        "internal_lineage": (
            "[D-40] Addendum 1 per-bin dump (scripts/diag_pf_per_bin.py); "
            "verdict string of record: " + str(d.get("hypothesis_c_verdict"))
        ),
        "checkpoint_of_record": d.get("ckpt_path"),
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig2-pf-per-bin.csv",
        fieldnames=["k_parallel_s_per_km", "P_F_sat_aware", "P_F_truth",
                    "pred_over_truth", "rel_diff", "in_gate_band"],
        rows=rows,
        producing_fn="src.export.export_d40_pf_per_bin",
        source_data_path=per_bin_json,
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows),
            "pearson_log_rederived": pearson}


def export_d40_shape_amplitude_summary(
    out_dir: str,
    per_bin_json: str = D40_PF_PER_BIN_JSON,
    baseline_json: str = D40_BASELINE_PER_BIN_JSON,
) -> Dict[str, Any]:
    """ep04 fig2 companion: shape-vs-amplitude summary, intervention vs the
    production baseline across all four physics (BANKED JSONs).

    The one-table form of the D1 signature: the intervention preserves shape
    (Pearson 0.83) at ~0.43x amplitude with LOW bin-to-bin scatter (0.12),
    while the production baseline holds amplitude ratios ~0.74-0.98 with HIGH
    scatter (0.31-0.46) -- the intervention traded amplitude for the very term
    it was told to minimize.
    """
    sat = _read_banked_json(per_bin_json)["diagnostics"]
    base = _read_banked_json(baseline_json)

    rows: List[Dict[str, Any]] = [{
        "model": "sat_aware_intervention",
        "physics": "P1",
        "pearson_log_in_band": float(sat["pearson_log_in_band"]),
        "amplitude_ratio_mean": float(sat["band_ratio_mean"]),
        "amplitude_ratio_std": float(sat["band_ratio_std"]),
        "log10_std_pred_in_band": float(sat["log10_std_pred_in_band"]),
        "log10_std_truth_in_band": float(sat["log10_std_truth_in_band"]),
    }]
    for cell in ("P1", "P2", "P3", "P4"):
        b = base[cell]
        rows.append({
            "model": "production_baseline",
            "physics": cell,
            "pearson_log_in_band": float(b["pearson_log"]),
            "amplitude_ratio_mean": float(b["ratio_mean"]),
            "amplitude_ratio_std": float(b["ratio_std"]),
            "log10_std_pred_in_band": float(b["log_std_pred"]),
            "log10_std_truth_in_band": float(b["log_std_truth"]),
        })
    _validate_rows("d40-shape-amplitude", rows)

    caveat = (
        "Shape-vs-amplitude summary of the failure signature: the intervention "
        "run (P1) preserves the P_F shape (log-Pearson 0.8346) at a uniformly "
        "suppressed ~0.43x amplitude (scatter 0.12), while the production "
        "baseline holds amplitude ratio ~0.74-0.98 (scatter 0.31-0.46) across "
        "all four feedback variants. The signature is amplitude-shrink WITH "
        "shape preservation -- a scale distortion, not a constant collapse. The "
        "intervention was run on P1 only: the cross-physics rows describe the "
        "BASELINE only, and nothing here licenses a claim about how the "
        "intervention would behave on P2-P4. Single realization / fixed "
        "cosmology (Sherwood, one 60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "internal_lineage": "[D-40] Addendum 1 diagnostics + cross-physics baseline per-bin summary",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig2-shape-amplitude-summary.csv",
        fieldnames=["model", "physics", "pearson_log_in_band", "amplitude_ratio_mean",
                    "amplitude_ratio_std", "log10_std_pred_in_band", "log10_std_truth_in_band"],
        rows=rows,
        producing_fn="src.export.export_d40_shape_amplitude_summary",
        source_data_path=f"{per_bin_json} + {baseline_json}",
        physics_id=[1, 2, 3, 4],
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d40_run_config(out_dir: str) -> Dict[str, Any]:
    """ep04 spec readout: the intervention's three loss terms, run config, and
    the pre-committed stop discipline (BANKED from the decision record)."""
    rows = [{"key": k, "value": str(v)} for k, v in D40_RUN_CONFIG]
    _validate_rows("d40-run-config", rows)
    caveat = (
        "Intervention spec of record: three loss terms added to the unchanged "
        "production objective, run on the fiducial physics variant only at a "
        "reduced schedule (12,500 steps vs the production 50k), under a "
        "pre-committed FAIL criterion fixed before dispatch. On falsification "
        "the all-four-physics full-schedule follow-up was cancelled (~$4-6 "
        "saved). The story is the discipline: a small pre-registered test "
        "retired the intervention family -- not a big experiment failing. "
        "Weight retuning was NOT attempted: the decision record argues the "
        "degeneracy is in the loss SHAPE (argued, not tested)."
    )
    extra = {"internal_lineage": "[D-40] Spec + Verdict + Tier-2 cancellation clauses"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="spec-run-config.csv",
        fieldnames=["key", "value"],
        rows=rows,
        producing_fn="src.export.export_d40_run_config",
        physics_id=1,
        source_data_path="decision record (banked spec; see internal_lineage)",
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


# =========================================================================== #
# ep05 "the-physics-constraint" batch — the [D-41] per-pixel FGPA-tail
# regularizer and its D2 constant-prediction collapse. One RE-READ artifact
# (the 50-step smoke trace, read from the LOCAL MLflow store — this run
# executed on the host, so unlike ep04 the trajectory is recoverable) and
# three BANKED artifacts. The verdict history of record: smoke FAIL (the
# mean-F tell) -> retrospective challenge -> user-authorized Tier-1
# confirmation (Juno 197381, 25k steps) which RATIFIED the verdict and
# yielded the field-level collapse diagnosis; the mechanism narrative was
# then corrected from "drove absorption to zero" to constant-prediction
# collapse (the [D-42-meta] Addendum 1 item 4+13 record).
# =========================================================================== #

MLFLOW_DB: str = "mlflow.db"
D41_SMOKE_RUN_ID: str = "21fb45cb3cd54793911824950591d44c"
D41_COLLAPSE_JSON: str = (
    "experiments/nerf/artifacts/eval/cleanup_pass/"
    "item4_n_HI_distribution_tier1_fgpatail.json"
)
D41_ITEMS_268_JSON: str = "experiments/nerf/artifacts/eval/cleanup_pass/items_2_6_8.json"

# Banked smoke endpoints (decision-record step table; consistency-asserted
# against the tracker re-read).
D41_SMOKE_ENDPOINTS: Dict[str, Tuple[float, float]] = {
    "loss_fgpa_tail": (6.367, 0.3085),
    "mean_flux_pred": (0.8687, 1.0000),
    "tau_amp": (1.0000, 0.9906),
}

# Banked Tier-1 ratification scalars (retrospective Addendum record).
D41_TIER1: Dict[str, Any] = {
    "pf_residual_mean": 0.999965,
    "pf_residual_baseline": 0.4155,
    "ks_distance_raw": 0.0,
    "ks_reading": "degenerate empty sample (constant F~1 leaves no pixels in the [0.05,0.95] analysis window) -- NOT a pass",
    "steps": 25000,
    "wallclock": "55:46",
    "cost_spent": "~$1.50 paid GPU",
}

D41_RUN_CONFIG: List[Tuple[str, Any]] = [
    ("intervention", "per-voxel physics prior: penalize deviation of the network's (density, temperature) -> absorption relation from the FGPA tail scaling on diffuse bins"),
    ("scaling_form", "tau proportional to Delta^1.6 T^-0.7 (Hui & Gnedin 1997; exponents frozen)"),
    ("regularizer_weight", 0.1),
    ("huber_delta", 0.5),
    ("valid_tau_ceiling", 0.5),
    ("anchor_constant_C_log_units", -9.2151),
    ("anchor_constant_source", "truth-side cache; valid mask 131072/131072 source bins (100%)"),
    ("exponents_truth_fit_later", "beta_emp 1.67 (4% off frozen 1.6); gamma_emp -0.41 (40% off frozen -0.7); scatter 0.139 dex over 7.58M voxels -- verdict unaffected, lesson banked"),
    ("physics", "P1 (fiducial) only"),
    ("smoke_run", "50 steps, n_rays=64, microbatch=32, seed 0, anchor 0.979, host CPU, minutes of compute, no paid dispatch"),
    ("smoke_verdict", "FAIL on the mean-flux tell; full-scale stage initially skipped (~$1.50 / ~50 min paid GPU saved)"),
    ("full_scale_confirmation", "one full-scale confirmation run later executed (user-authorized during the retrospective review): 25,000 steps, ~56 min wallclock, ~$1.50 -- verdict RATIFIED at scale"),
    ("stop_rule", "no paid dispatch on a failing smoke; the confirmation run was review-mandated, not a retry"),
]


def _read_mlflow_metric_series(
    db_path: str, run_id: str, keys: Sequence[str]
) -> Dict[str, Dict[int, float]]:
    """Read per-step metric series for one run from a local MLflow SQLite
    store (read-only; deterministic ORDER BY step)."""
    import sqlite3

    series: Dict[str, Dict[int, float]] = {k: {} for k in keys}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "select key, step, value from metrics where run_uuid = ? and key in ({}) "
            "order by step".format(",".join("?" * len(keys))),
            [run_id, *keys],
        )
        for key, step, value in cur.fetchall():
            series[key][int(step)] = float(value)
    finally:
        con.close()
    return series


def export_d41_smoke_trace(
    out_dir: str,
    db_path: str = MLFLOW_DB,
    run_id: str = D41_SMOKE_RUN_ID,
) -> Dict[str, Any]:
    """ep05 fig1: the 50-step smoke trace, RE-READ from the local tracker.

    The descent-vs-tell figure: the regularizer term falls ~20x while the
    predicted mean flux climbs to exactly 1.0000 (perfect transparency) and
    the amplitude guard holds. Endpoints are consistency-asserted against the
    banked decision-record step table.
    """
    keys = ["loss_fgpa_tail", "mean_flux_pred", "loss_data", "loss_meanF",
            "tau_amp", "loss"]
    series = _read_mlflow_metric_series(db_path, run_id, keys)
    steps = sorted(series["loss_fgpa_tail"])
    if not steps or steps[0] != 1 or steps[-1] != len(steps):
        raise AssertionError("smoke trace incomplete: expected contiguous steps from 1.")
    for k in keys:
        if sorted(series[k]) != steps:
            raise AssertionError(f"metric {k!r} missing steps vs loss_fgpa_tail.")

    for k, (start, end) in D41_SMOKE_ENDPOINTS.items():
        got_start, got_end = series[k][steps[0]], series[k][steps[-1]]
        if abs(got_start - start) > 5e-4 * max(1.0, abs(start)) or \
           abs(got_end - end) > 5e-4 * max(1.0, abs(end)):
            raise AssertionError(
                f"{k}: tracker endpoints ({got_start}, {got_end}) disagree with "
                f"banked record ({start}, {end})."
            )

    rows = [
        {"step": s, **{k: series[k][s] for k in keys}}
        for s in steps
    ]
    _validate_rows("d41-smoke-trace", rows)
    F = np.array([r["mean_flux_pred"] for r in rows])
    _validate_flux("d41-smoke-trace mean_flux_pred", F)

    caveat = (
        "50-step host smoke trace for the per-pixel physics-prior intervention "
        "(P1, single seed), re-read from the experiment tracker. The added "
        "term descends ~20x (6.37 -> 0.31) while the predicted mean flux "
        "climbs to 1.0000 EXACTLY -- perfectly transparent gas. The 1.0000 "
        "looks like passing and is the fingerprint of collapse: the score the "
        "anchor reports is quietly satisfied by a universe with nothing in it. "
        "The amplitude guard held (0.99), so the escape was through the "
        "fields, not the calibration. Smoke-scale evidence: the verdict this "
        "trace triggered was later RATIFIED by a full-scale confirmation run "
        "(see fig3); cite the smoke as the trigger, the confirmation as the "
        "ratification. Single realization / fixed cosmology (Sherwood, one "
        "60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "n_steps": len(steps),
        "descent_factor_loss_fgpa_tail": series["loss_fgpa_tail"][steps[0]] / series["loss_fgpa_tail"][steps[-1]],
        "internal_lineage": f"[D-41] smoke; MLflow run {run_id} (local store); cloud_runs/fgpa_tail_smoke.log",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig1-smoke-trace.csv",
        fieldnames=["step", *keys],
        rows=rows,
        producing_fn="src.export.export_d41_smoke_trace",
        source_data_path=f"{db_path} (local MLflow store, run {run_id})",
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows),
            "descent_factor": extra["descent_factor_loss_fgpa_tail"]}


def export_d41_collapse_signature(
    out_dir: str,
    collapse_json: str = D41_COLLAPSE_JSON,
) -> Dict[str, Any]:
    """ep05 fig2: the field-level collapse signature (BANKED Tier-1 diagnostic).

    Truth vs predicted per-field distribution summaries from the confirmation
    run's checkpoint: every field collapsed to a near-constant (density
    68.3-74.7 around 71.5; X_HI ~3.3e-5), and predicted n_HI is ~28,000x
    LARGER than the truth median -- the empirical refutation of the initial
    "drove absorption to zero" reading.
    """
    d = _read_banked_json(collapse_json)
    truth, pred, ratio = d["truth"], d["fgpa_tail_tier1_pred"], d["collapse_ratio_med"]

    rows: List[Dict[str, Any]] = []
    for field in ("density", "X_HI", "n_HI"):
        t, p = truth[field], pred[field]
        if not (p["min"] > 0 and t["min"] > 0):
            raise AssertionError(f"{field}: non-positive field values in banked diagnostic.")
        rederived = p["median"] / t["median"]
        if abs(rederived - ratio[field]) > 1e-9 * abs(ratio[field]):
            raise AssertionError(f"{field}: re-derived median ratio disagrees with banked.")
        rows.append({
            "field": field,
            "truth_min": float(t["min"]), "truth_median": float(t["median"]),
            "truth_max": float(t["max"]),
            "pred_min": float(p["min"]), "pred_median": float(p["median"]),
            "pred_max": float(p["max"]),
            "pred_over_truth_median": float(ratio[field]),
        })
    _validate_rows("d41-collapse-signature", rows)

    caveat = (
        "Field-level collapse signature, measured on the full-scale "
        "CONFIRMATION run's checkpoint (25,000 steps) -- not on the 50-step "
        "smoke; always attribute these numbers to the confirmation run. Truth "
        "spans decades in every field; the prediction is a near-constant in "
        "every field (density 68.3-74.7 around 71.5 -- 494x the truth median "
        "of 0.145; X_HI ~3.3e-5). Predicted n_HI is ~28,000x LARGER than the "
        "truth median: nothing was driven to zero. The licensed mechanism of "
        "record is CONSTANT-PREDICTION COLLAPSE -- constants trivially "
        "satisfy the enforced physics relation (one equation, three "
        "constants), and the mean-flux anchor pulls the resulting constant "
        "absorption toward transparency. The initial 'network drove "
        "absorption to zero, evading the anchor' reading is empirically "
        "wrong and may be narrated only as the corrected first guess. Single "
        "realization / fixed cosmology (Sherwood, one 60 cMpc/h box); z=0.3 "
        "scope-lock."
    )
    extra = {
        "frac_n_HI_below_1e-9": {
            "truth": truth["n_HI_lt_1e9_frac"],
            "prediction": pred["n_HI_lt_1e9_frac"],
        },
        "internal_lineage": (
            "[D-42-meta] Addendum 1 items 4+13; Tier-1 Juno job 197381 "
            "(RUN_TAG P1-N64-S0-1778508750-7f0c7e), eval job 197387; "
            "checkpoint cloud_runs/fgpa-tail-tier1-P1-step25k.pt"
        ),
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig2-collapse-signature.csv",
        fieldnames=["field", "truth_min", "truth_median", "truth_max",
                    "pred_min", "pred_median", "pred_max", "pred_over_truth_median"],
        rows=rows,
        producing_fn="src.export.export_d41_collapse_signature",
        source_data_path=collapse_json,
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d41_verdict_table(out_dir: str) -> Dict[str, Any]:
    """ep05 fig3: the two-stage verdict table (BANKED) -- smoke trigger +
    full-scale ratification."""
    rows: List[Dict[str, Any]] = [
        {"stage": "smoke", "metric": "regularizer_term_start", "value": 6.367,
         "reference": "", "reading": "training start"},
        {"stage": "smoke", "metric": "regularizer_term_end_step50", "value": 0.3085,
         "reference": "", "reading": "~20x descent in 50 steps"},
        {"stage": "smoke", "metric": "mean_flux_pred_end", "value": 1.0000,
         "reference": "anchor 0.979", "reading": "the tell: exact transparency, fingerprint of collapse"},
        {"stage": "smoke", "metric": "tau_amp_end", "value": 0.9906,
         "reference": "guard range", "reading": "calibration guard held; escape was through the fields"},
        {"stage": "confirmation", "metric": "pf_residual_band_mean", "value": 0.999965,
         "reference": "production baseline 0.4155 (verdict-level only; no multiple licensed against a ceiling-saturated residual)",
         "reading": "residual pinned at its ~1.0 ceiling: essentially no flux structure at all"},
        {"stage": "confirmation", "metric": "ks_distance", "value": 0.0,
         "reference": "gate 0.05",
         "reading": "a zero that means EMPTY, not perfect: constant F~1 leaves no pixels in the analysis window"},
    ]
    _validate_rows("d41-verdict-table", rows)
    caveat = (
        "Two-stage verdict of record. The 50-step smoke triggered the FAIL "
        "(on the mean-flux tell) and the full-scale stage was skipped under "
        "the standing stop rule; the retrospective review then challenged the "
        "smoke-scale verdict and one full-scale confirmation run was "
        "authorized, which RATIFIED it: flux-power residual pinned at ~1.0 "
        "(vs 0.4155 baseline -- state it as the residual ceiling, i.e. no "
        "flux structure; do NOT quote a 'times worse' multiple) and a KS "
        "'0' that is a degenerate empty sample, not a pass. Single "
        "realization / fixed cosmology; z=0.3 scope-lock."
    )
    extra = {"confirmation_run": D41_TIER1,
             "internal_lineage": "[D-41] verdict + [D-42-meta] Addendum 1 items 4+13 (K4/D3 settlement)"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig3-verdict-table.csv",
        fieldnames=["stage", "metric", "value", "reference", "reading"],
        rows=rows,
        producing_fn="src.export.export_d41_verdict_table",
        source_data_path="decision record (banked scalars; see internal_lineage)",
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d41_run_config(out_dir: str) -> Dict[str, Any]:
    """ep05 spec readout (BANKED): regularizer form, config, two-stage stop
    discipline, and the frozen-exponents confession."""
    rows = [{"key": k, "value": str(v)} for k, v in D41_RUN_CONFIG]
    _validate_rows("d41-run-config", rows)
    caveat = (
        "Intervention spec of record. The exponents were frozen from the "
        "Hui & Gnedin 1997 scaling; a later fit to the simulation's own truth "
        "found the temperature exponent materially off (-0.41 vs the frozen "
        "-0.7) -- the verdict is unaffected (the collapse is a separate "
        "mechanism), but the lesson was banked: measure the relation from "
        "truth before freezing textbook values. Cost framing: the smoke cost "
        "minutes of host CPU and no paid dispatch; skipping the full stage "
        "saved ~$1.50 of paid GPU; the later review-mandated confirmation run "
        "spent ~$1.50 to ratify the verdict at scale. The story is the "
        "discipline: a cheap pre-committed test called it, and the review "
        "refused to let a smoke-scale verdict stand unchallenged."
    )
    extra = {
        "smoke_device_evidence": (
            "banked smoke log cloud_runs/fgpa_tail_smoke.log, first line: "
            "'Using device: cpu' (per-gate C3 proviso: CPU wording retained "
            "with banked citation)"
        ),
        "internal_lineage": "[D-41] Spec + Verdict; [D-42-meta] item 8 (beta/gamma truth fit) + item 13 (Tier-1 authorization)",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="spec-run-config.csv",
        fieldnames=["key", "value"],
        rows=rows,
        producing_fn="src.export.export_d41_run_config",
        source_data_path="decision record (banked spec; see internal_lineage)",
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


# =========================================================================== #
# ep06 "the-planted-clue" batch — the [D-42] velocity-gradient conditioning
# intervention (architecture-INPUT axis; loss UNCHANGED) and its D3
# asymmetric-head collapse. Gate values are BANKED from the [D-42] Addendum
# authoritative readout (the on-disk gates JSON is header-only — an MLflow
# loss-history wiring gap; see D42_GATES_JSON_BROKEN). The scalar smoke trace
# is RE-READ from the local tracker (it survived, like ep05); the per-HEAD
# spread trace did NOT survive (only the step-50 endpoints are of record).
# Per the ep05 finding, verdict/gate CSVs ship a reader-facing `label` column.
# =========================================================================== #

D42_SMOKE_RUN_ID: str = "63e6990b085b46258a52530989b2edfc"
D42_GATES_JSON_BROKEN: str = (
    "experiments/nerf/artifacts/eval/d42_smoke/d42_smoke_P1_gates.json"
)

# The six pre-committed smoke gates (authoritative [D-42] Addendum readout).
# label = reader-facing; every value is banked from the decision record.
D42_GATE_TABLE: List[Dict[str, Any]] = [
    {"gate": "1", "label": "No NaN anywhere",
     "measured": "all finite", "floor_or_threshold": "no NaN/Inf", "verdict": "PASS"},
    {"gate": "2", "label": "Training loss descends",
     "measured": "ratio 0.349 (0.0318 -> 0.0111)", "floor_or_threshold": "final/initial < 0.85",
     "verdict": "PASS"},
    {"gate": "3", "label": "Mean flux stays near the anchor",
     "measured": "1.0000 (drift 0.021 from 0.979)", "floor_or_threshold": "within 0.05 of 0.979",
     "verdict": "PASS (but see the tell)"},
    {"gate": "4", "label": "Absorption-amplitude knob stable",
     "measured": "0.9924", "floor_or_threshold": "within [0.5, 2.0]", "verdict": "PASS"},
    {"gate": "5", "label": "Density field keeps spread (anti-collapse floor)",
     "measured": "0.0071", "floor_or_threshold": "spread > 1.45", "verdict": "FAIL (~200x below floor)"},
    {"gate": "6", "label": "Neutral-fraction field keeps spread (anti-collapse floor)",
     "measured": "0.0332", "floor_or_threshold": "spread > 6e-5", "verdict": "PASS (~550x above floor)"},
]

# Per-head asymmetric-collapse signature (step-50 endpoints, banked from the
# [D-42] Addendum mechanism paragraphs). Values the record gives as "approx"
# (median ~0.0003, ~4.4e-3) are flagged approximate in the sidecar.
D42_HEAD_ASYMMETRY: List[Dict[str, Any]] = [
    {"head": "density", "label": "Density head (the one the clue was fed to)",
     "spread_measured": 0.0071, "spread_floor": 1.45, "spread_verdict": "FAIL",
     "pred_median_approx": 0.0003, "truth_median": 0.145,
     "pred_min": 0.0000, "pred_max": 0.0071,
     "outcome": "collapsed to a sliver near zero (truth median ~500x the predicted median)"},
    {"head": "X_HI", "label": "Neutral-fraction head",
     "spread_measured": 0.0332, "spread_floor": 6e-5, "spread_verdict": "PASS",
     "pred_median_approx": 4.4e-3, "truth_median": 6.0e-7,
     "pred_min": 4.8e-11, "pred_max": 3.3e-2,
     "outcome": "kept spatial structure (spread matches the truth dynamic range; absolute level ~7300x high)"},
]

D42_RUN_CONFIG: List[Tuple[str, Any]] = [
    ("intervention", "condition the density head on the line-of-sight velocity gradient (an architecture-INPUT change; the lesson is UNCHANGED)"),
    ("conditioning_signal", "d(v_pec)/d(chi) computed once from the TRUE Sherwood gas by finite difference, standardized, FROZEN (detached -- no gradient to the network, so the network cannot alter it)"),
    ("why_truth_anchored", "unlike the previous episode's regularizer, the clue is fixed against truth; the network cannot satisfy it by shrinking its own outputs"),
    ("loss_form", "UNCHANGED from the production baseline (same data loss, same mean-flux anchor, same amplitude prior); the only change is one extra input feature"),
    ("gate_discipline", "first spec written under the recalibrated discipline: hedged design verbs, six pre-committed smoke gates, and an anti-degeneracy audit table"),
    ("density_floor_origin", "the density-spread and X_HI-spread floors were born from the previous episode's constant-collapse: value-agnostic spread floors set at 10x (density) and 100x (neutral fraction) the truth medians; the neutral-fraction floor also sits 1.8x above that episode's collapse value"),
    ("physics", "P1 (fiducial) only"),
    ("smoke_run", "50 steps, n_rays=64, microbatch=32, seed 0, anchor 0.979, minutes on the host machine (device: cpu per the run log), no paid dispatch"),
    ("verdict", "5/6 gates PASS; the density-spread floor FAILED at 0.0071 vs 1.45 -> retired at smoke"),
    ("cost", "saved ~$1.50 of paid GPU (the full-scale stage was never dispatched); spent ~$0 paid -- minutes of host compute"),
    ("stop_rule", "any single pre-committed gate FAIL stops paid dispatch; no confirmation run was run this time (the floor breach was unambiguous)"),
]


def export_d42_gate_table(out_dir: str) -> Dict[str, Any]:
    """ep06 fig1: the six pre-committed smoke gates (BANKED authoritative
    readout; the on-disk JSON is header-only). Ships a reader-facing `label`
    column per the ep05 finding."""
    rows = list(D42_GATE_TABLE)
    _validate_rows("d42-gate-table", rows)
    n_pass = sum(1 for r in rows if r["verdict"].startswith("PASS"))
    n_fail = sum(1 for r in rows if r["verdict"].startswith("FAIL"))
    if (n_pass, n_fail) != (5, 1):
        raise AssertionError(f"gate tally {n_pass}P/{n_fail}F != banked 5P/1F.")

    caveat = (
        "The six pre-committed smoke gates, and the debut of the pre-registered "
        "gate discipline. Values are the AUTHORITATIVE decision-record readout: "
        "the on-disk gates JSON is HEADER-ONLY -- every gate reads pass=false "
        "with 'loss_history not provided', an experiment-tracker wiring gap, NOT "
        "the real result (path in the provenance lineage) -- so this table is "
        "exported from the six-gate table banked in the decision record, not "
        "from that file. 5/6 PASS; the density-spread gate FAILED at "
        "0.0071 against its 1.45 floor (~200x below). The mean-flux gate 'PASSED' "
        "at 1.0000 -- but that pass is the tell: a wide-thresholded check (within "
        "0.05 of 0.979) let perfectly transparent gas slip through as if healthy "
        "(see the head-asymmetry artifact). Single realization / fixed cosmology "
        "(Sherwood, one 60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "n_pass": n_pass, "n_fail": n_fail,
        "on_disk_json_status": "header-only/empty (tracker loss-history wiring gap); NOT the readout",
        "internal_lineage": (
            "[D-42] Addendum 1 six-gate table (authoritative); MLflow run "
            f"{D42_SMOKE_RUN_ID}; broken JSON {D42_GATES_JSON_BROKEN}"
        ),
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig1-gate-table.csv",
        fieldnames=["gate", "label", "measured", "floor_or_threshold", "verdict"],
        rows=rows,
        producing_fn="src.export.export_d42_gate_table",
        source_data_path="decision record (banked authoritative readout; the on-disk JSON is header-only)",
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows),
            "n_pass": n_pass, "n_fail": n_fail}


def export_d42_head_asymmetry(out_dir: str) -> Dict[str, Any]:
    """ep06 fig2: the per-head asymmetric-collapse signature -- the density head
    dead near zero while the X_HI head kept structure (BANKED step-50
    endpoints; the per-head spread trace did not survive)."""
    rows: List[Dict[str, Any]] = []
    for h in D42_HEAD_ASYMMETRY:
        floor_ratio = h["spread_floor"] / h["spread_measured"]
        rows.append({
            "head": h["head"], "label": h["label"],
            "spread_measured": float(h["spread_measured"]),
            "spread_floor": float(h["spread_floor"]),
            "spread_verdict": h["spread_verdict"],
            "spread_vs_floor": (f"{floor_ratio:.0f}x below floor" if h["spread_verdict"] == "FAIL"
                                else f"{1.0/floor_ratio:.0f}x above floor"),
            "pred_median_approx": float(h["pred_median_approx"]),
            "truth_median": float(h["truth_median"]),
            "outcome": h["outcome"],
        })
    _validate_rows("d42-head-asymmetry", rows)
    # The two heads must disagree on verdict -- that IS the signature.
    verdicts = {r["spread_verdict"] for r in rows}
    if verdicts != {"PASS", "FAIL"}:
        raise AssertionError(f"head-asymmetry expects one PASS + one FAIL; got {verdicts}.")

    caveat = (
        "The D3 asymmetric-head collapse, measured at step 50. Distinct from the "
        "two earlier failures: the first shrank a structured answer (amplitude "
        "wrong, shape kept); the second abandoned structure everywhere (all "
        "fields constant); this one is PARTIAL -- collapse hiding in a single "
        "head of a multi-head machine. The density head -- the very head the "
        "velocity-gradient clue was fed to -- collapsed to a sliver near zero "
        "(spread 0.0071 vs its 1.45 floor; predicted median ~0.0003 vs truth "
        "0.145), while the neutral-fraction head kept spatial structure (spread "
        "0.0332, ~550x above its own floor). The density collapse to near-zero "
        "is what drove the mean flux to 1.0000: near-zero density means near-zero "
        "absorption means transparent gas. These are step-50 endpoint values -- "
        "the per-head spread was not logged per step, so there is no trace to "
        "draw. Medians marked _approx are the decision record's approximate "
        "values. Single realization / fixed "
        "cosmology (Sherwood, one 60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "step_of_measurement": 50,
        "trace_availability": "step-50 endpoints only; the per-head spread was not logged per step",
        "internal_lineage": "[D-42] Addendum 1 mechanism paragraphs (density-head / X_HI-head readout)",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig2-head-asymmetry.csv",
        fieldnames=["head", "label", "spread_measured", "spread_floor", "spread_verdict",
                    "spread_vs_floor", "pred_median_approx", "truth_median", "outcome"],
        rows=rows,
        producing_fn="src.export.export_d42_head_asymmetry",
        source_data_path="decision record (banked step-50 per-head readout)",
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d42_smoke_trace(
    out_dir: str,
    db_path: str = MLFLOW_DB,
    run_id: str = D42_SMOKE_RUN_ID,
) -> Dict[str, Any]:
    """ep06 fig3: the scalar 50-step smoke trace, RE-READ from the local
    tracker (it survived). The mean-flux path climbing to 1.0000 is the
    plottable story; the per-head spread trace did NOT survive (see fig2)."""
    keys = ["loss", "loss_data", "loss_meanF", "mean_flux_pred", "tau_amp"]
    series = _read_mlflow_metric_series(db_path, run_id, keys)
    steps = sorted(series["loss"])
    if not steps or steps[0] != 1 or steps[-1] != len(steps):
        raise AssertionError("smoke trace incomplete: expected contiguous steps from 1.")
    for k in keys:
        if sorted(series[k]) != steps:
            raise AssertionError(f"metric {k!r} missing steps vs loss.")

    # Consistency vs banked endpoints (the [D-42] Addendum gate table).
    if abs(series["loss"][steps[0]] - 0.0318) > 5e-4:
        raise AssertionError(f"loss step1 {series['loss'][steps[0]]} != banked 0.0318.")
    if abs(series["loss"][steps[-1]] - 0.0111) > 5e-4:
        raise AssertionError(f"loss step50 {series['loss'][steps[-1]]} != banked 0.0111.")
    if abs(series["mean_flux_pred"][steps[-1]] - 1.0) > 5e-4:
        raise AssertionError(f"mean_F step50 {series['mean_flux_pred'][steps[-1]]} != banked 1.0000.")
    if abs(series["tau_amp"][steps[-1]] - 0.9924) > 1e-3:
        raise AssertionError(f"tau_amp step50 {series['tau_amp'][steps[-1]]} != banked 0.9924.")

    rows = [{"step": s, "loss_total": series["loss"][s], "loss_data": series["loss_data"][s],
             "loss_mean_flux_term": series["loss_meanF"][s],
             "mean_flux_pred": series["mean_flux_pred"][s], "tau_amp": series["tau_amp"][s]}
            for s in steps]
    _validate_rows("d42-smoke-trace", rows)
    _validate_flux("d42-smoke-trace mean_flux_pred", np.array([r["mean_flux_pred"] for r in rows]))

    caveat = (
        "50-step host smoke trace for the velocity-gradient conditioning "
        "intervention (P1, single seed), re-read from the experiment tracker. "
        "The plottable story is the mean flux climbing from 0.874 THROUGH the "
        "0.979 anchor and landing on 1.0000 -- the tell, for the second time in "
        "the arc, but this time it slipped past a wide-thresholded gate as a "
        "'pass'. The density collapse that CAUSED the transparency is not "
        "visible in this scalar trace: the per-head spread was not logged per "
        "step, so it is a step-50 endpoint only (see fig2-head-asymmetry). "
        "Single realization / fixed cosmology (Sherwood, one 60 cMpc/h box); "
        "z=0.3 scope-lock."
    )
    extra = {
        "n_steps": len(steps),
        "per_head_spread_trace": "NOT available (endpoint-only at step 50; see fig2)",
        "internal_lineage": f"[D-42] Addendum 1 smoke; MLflow run {run_id} (local store); cloud_runs/d42_smoke_P1/smoke_stdout.log",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig3-smoke-trace.csv",
        fieldnames=["step", "loss_total", "loss_data", "loss_mean_flux_term",
                    "mean_flux_pred", "tau_amp"],
        rows=rows,
        producing_fn="src.export.export_d42_smoke_trace",
        source_data_path=f"{db_path} (local MLflow store, run {run_id})",
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d42_run_config(out_dir: str) -> Dict[str, Any]:
    """ep06 spec readout (BANKED): the conditioning input, the unchanged loss,
    the six gates + floor provenance, config, and the cost framing."""
    rows = [{"key": k, "value": str(v)} for k, v in D42_RUN_CONFIG]
    _validate_rows("d42-run-config", rows)
    caveat = (
        "Intervention spec of record. The change is to the machine's INPUT, not "
        "its lesson: the loss is byte-for-byte the production objective, and the "
        "network is handed one extra feature -- the line-of-sight velocity "
        "gradient taken from the simulation's own truth, frozen so the network "
        "cannot alter it. This was the first spec under the recalibrated "
        "discipline (hedged verbs, pre-committed gates, an anti-degeneracy "
        "audit). The honest edge: the audit did NOT anticipate a per-head "
        "collapse -- it imagined a constant-everywhere collapse like the "
        "previous episode's -- and it was the collapse-agnostic density-spread "
        "FLOOR, inherited from that episode, that caught this different shape. "
        "Cost: saved ~$1.50 of paid GPU by not dispatching the full stage; spent "
        "~$0 paid (minutes of host compute); no confirmation run this time -- the "
        "floor breach was unambiguous. The story is the discipline."
    )
    extra = {"internal_lineage": "[D-42] Spec (math contract, smoke gates, audit table) + Addendum 1 verdict"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="spec-run-config.csv",
        fieldnames=["key", "value"],
        rows=rows,
        producing_fn="src.export.export_d42_run_config",
        source_data_path="decision record (banked spec; see internal_lineage)",
        physics_id=1,
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


# =========================================================================== #
# ep07 "all-four-at-once" batch — the [D-46] physics_id-embedding joint
# training (data axis) and its D4 combined-trivial-collapse-with-active-
# embedding, retired at a 50-step P-mixed host smoke; the epic that closes
# the four-axis cascade. UNLIKE ep06, the on-disk gates JSON is HEALTHY
# (full loss_history + structured per-gate readouts + the complete pairwise
# embedding-distance matrix) — every artifact here is RE-READ from it, with
# consistency asserts against the decision-record Addendum table. The
# per-step mean-flux trace was NOT logged (endpoints only, per physics).
# =========================================================================== #

D46_GATES_JSON: str = (
    "experiments/nerf/artifacts/eval/d46_smoke/d46_smoke_1778675261_gates.json"
)
D46_PHYSICS_LABELS: List[str] = ["P1 (no feedback)", "P2 (stellar)", "P3 (AGN)", "P4 (strong AGN)"]

# Banked consistency anchors (decision-record Addendum gate table).
D46_BANKED: Dict[str, Any] = {
    "gate2_ratio": 1.0036, "gate4_tau_amp": 0.991,
    "gate5_spreads": [6.4e-5, 7.5e-6, 2.0e-4, 8.9e-7],
    "gate7_max_l2": 7.045, "n_pass": 4, "n_fail": 3,
}


def _read_d46_gates(gates_json: str) -> Dict[str, Any]:
    d = _read_banked_json(gates_json)
    g = d["gates"]
    # Consistency vs the banked Addendum table.
    if abs(g["gate_2_loss_descent"]["observed_ratio"] - D46_BANKED["gate2_ratio"]) > 5e-4:
        raise AssertionError("gate-2 ratio disagrees with the banked record.")
    if abs(g["gate_4_tau_amp_window"]["observed"] - D46_BANKED["gate4_tau_amp"]) > 5e-3:
        raise AssertionError("gate-4 tau_amp disagrees with the banked record.")
    for got, banked in zip(g["gate_5_density_spread"]["observed_spread_per_physics"],
                           D46_BANKED["gate5_spreads"]):
        if abs(got - banked) > 0.1 * banked:
            raise AssertionError("gate-5 spreads disagree with the banked record.")
    if abs(g["gate_7_embedding_nondegeneracy"]["observed_max"] - D46_BANKED["gate7_max_l2"]) > 1e-2:
        raise AssertionError("gate-7 max L2 disagrees with the banked record.")
    n_pass = sum(1 for v in g.values() if v["pass"])
    if (n_pass, len(g) - n_pass) != (D46_BANKED["n_pass"], D46_BANKED["n_fail"]):
        raise AssertionError("gate tally disagrees with the banked 4 PASS / 3 FAIL.")
    return d


def export_d46_gate_table(out_dir: str, gates_json: str = D46_GATES_JSON) -> Dict[str, Any]:
    """ep07 fig1: the seven pre-committed smoke gates, RE-READ from the healthy
    on-disk gates JSON (consistency-asserted against the decision record)."""
    d = _read_d46_gates(gates_json)
    g = d["gates"]
    rows: List[Dict[str, Any]] = [
        {"gate": "1", "label": "No NaN anywhere", "measured": "all finite",
         "floor_or_threshold": "no NaN/Inf", "verdict": "PASS"},
        {"gate": "2", "label": "Training loss keeps descending",
         "measured": f"ratio {g['gate_2_loss_descent']['observed_ratio']:.4f} "
                     f"({g['gate_2_loss_descent']['observed_loss_ref']:.5f} at step 10 -> "
                     f"{g['gate_2_loss_descent']['observed_loss_end']:.5f} at step 50)",
         "floor_or_threshold": "step-50/step-10 < 0.85",
         "verdict": "FAIL (flat after step 10)"},
        {"gate": "3", "label": "Mean flux inside a healthy window and at least a hair away from 1.000 (hardened after the last two collapses)",
         "measured": "per-variant [1.000, 1.000, 1.000, 1.000]",
         "floor_or_threshold": "within [0.5, 0.99] AND at least 0.001 away from 1.000",
         "verdict": "FAIL (the tell, in four-part unison -- and this time the gate caught it)"},
        {"gate": "4", "label": "Absorption-amplitude knob stable",
         "measured": f"{g['gate_4_tau_amp_window']['observed']:.4f}",
         "floor_or_threshold": "within [0.5, 2.0]", "verdict": "PASS"},
        {"gate": "5", "label": "Density field keeps spread, per variant (anti-collapse floor)",
         "measured": "[6.4e-5, 7.5e-6, 2.0e-4, 8.9e-7]",
         "floor_or_threshold": "spread > 1.45 each",
         "verdict": "FAIL (7,264x to 1,624,925x below floor -- all four dead)"},
        {"gate": "6", "label": "Neutral-fraction field keeps spread, per variant (anti-collapse floor)",
         "measured": "[3.5e-3, 1.0e-3, 6.8e-3, 2.9e-4]",
         "floor_or_threshold": "spread > 6e-5 each",
         "verdict": "PASS (4.9x-113x above floor)"},
        {"gate": "7", "label": "The four physics labels stay distinct in the learned embedding (new this episode)",
         "measured": f"max pairwise distance {g['gate_7_embedding_nondegeneracy']['observed_max']:.3f} (all six pairs 4.6-7.0)",
         "floor_or_threshold": "max pairwise distance > 0.1",
         "verdict": "PASS (distinct codes -- and that is the strange part)"},
    ]
    _validate_rows("d46-gate-table", rows)
    caveat = (
        "The seven pre-committed smoke gates for the pooled-physics "
        "intervention; the gate set hardened again this episode (gate 3 now "
        "tests the mean flux away from BOTH the anchor and 1.000 -- the "
        "banked lesson of the last two collapses -- and gate 7, embedding "
        "distinctness, is new). 3/7 FAIL: the loss went flat after step 10 "
        "(ratio 1.0036), the mean flux landed on 1.000 in all four variants "
        "at once (the tell's third appearance -- caught cleanly this time), "
        "and every density head collapsed between ~7,000x and ~1,600,000x "
        "below the spread floor. "
        "4/7 PASS -- including gate 7: the four labels stayed genuinely "
        "distinct while all four routes led to the same transparent gas. "
        "Values re-read from the on-disk gate readout (healthy this run) and "
        "consistency-checked against the decision record. Single realization "
        "/ fixed cosmology (Sherwood, one 60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "n_pass": 4, "n_fail": 3,
        "run_config": {k: d[k] for k in ("steps_completed", "training_seconds",
                                         "model_params", "hidden_dim",
                                         "n_rays_pooled", "microbatch", "seed")},
        "internal_lineage": f"[D-46] Addendum 2 gate table; {D46_GATES_JSON}; scripts/d46_50step_host_smoke.py",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig1-gate-table.csv",
        fieldnames=["gate", "label", "measured", "floor_or_threshold", "verdict"],
        rows=rows,
        producing_fn="src.export.export_d46_gate_table",
        source_data_path=gates_json,
        physics_id=[1, 2, 3, 4],
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d46_d4_signature(out_dir: str, gates_json: str = D46_GATES_JSON) -> Dict[str, Any]:
    """ep07 fig2a: the D4 per-variant signature -- dead density heads, alive
    X_HI heads, the tell in unison. RE-READ from the gates JSON."""
    d = _read_d46_gates(gates_json)
    g = d["gates"]
    dens = g["gate_5_density_spread"]["observed_spread_per_physics"]
    xhi = g["gate_6_xhi_spread"]["observed_spread_per_physics"]
    mf = g["gate_3_mean_F_window"]["observed_per_physics"]
    rows: List[Dict[str, Any]] = []
    for i, label in enumerate(D46_PHYSICS_LABELS):
        rows.append({
            "physics": f"P{i+1}", "label": label,
            "density_spread": float(dens[i]), "density_floor": 1.45,
            "density_vs_floor": f"{1.45/dens[i]:,.0f}x below floor",
            "xhi_spread": float(xhi[i]), "xhi_floor": 6e-5,
            "xhi_vs_floor": f"{xhi[i]/6e-5:.1f}x above floor",
            "mean_flux": float(mf[i]),
        })
    _validate_rows("d46-d4-signature", rows)
    if not all(r["density_spread"] < 1.45 and r["xhi_spread"] > 6e-5 for r in rows):
        raise AssertionError("D4 signature expects all density FAIL + all X_HI PASS.")
    caveat = (
        "The D4 signature, per physics variant, at step 50: in every one of "
        "the four variants the density head collapsed to a near-uniform "
        "near-zero sliver (spreads 8.9e-7 to 2.0e-4 against a 1.45 floor -- "
        "four dead heads) while the neutral-fraction head kept structure "
        "(spreads 4.9x to 113x above its floor) and the mean flux landed on "
        "1.000 -- the transparent-gas tell, in four-part unison. Read "
        "alongside the embedding distances (fig2b): the four variants were "
        "given genuinely distinct learned labels, and all four arrived at "
        "the same trivial solution. Step-50 endpoints (the per-step per-head "
        "trace was not logged). Single realization / fixed cosmology "
        "(Sherwood, one 60 cMpc/h box); z=0.3 scope-lock."
    )
    extra = {
        "density_min_per_physics": g["gate_5_density_spread"]["observed_min_per_physics"],
        "density_max_per_physics": g["gate_5_density_spread"]["observed_max_per_physics"],
        "xhi_min_per_physics": g["gate_6_xhi_spread"]["observed_min_per_physics"],
        "xhi_max_per_physics": g["gate_6_xhi_spread"]["observed_max_per_physics"],
        "internal_lineage": f"[D-46] Addendum 2 (gates 3/5/6); {D46_GATES_JSON}",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig2a-d4-signature.csv",
        fieldnames=["physics", "label", "density_spread", "density_floor", "density_vs_floor",
                    "xhi_spread", "xhi_floor", "xhi_vs_floor", "mean_flux"],
        rows=rows,
        producing_fn="src.export.export_d46_d4_signature",
        source_data_path=gates_json,
        physics_id=[1, 2, 3, 4],
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d46_embedding_distances(out_dir: str, gates_json: str = D46_GATES_JSON) -> Dict[str, Any]:
    """ep07 fig2b: the full pairwise embedding-distance matrix -- the ALIVE
    half of D4 (four distinct codes). RE-READ from the gates JSON."""
    d = _read_d46_gates(gates_json)
    pairs = d["gates"]["gate_7_embedding_nondegeneracy"]["observed_pairwise"]
    if len(pairs) != 6:
        raise AssertionError(f"expected 6 embedding pairs, got {len(pairs)}.")
    rows = [{
        "pair": f"P{p['p_i']+1}-P{p['p_j']+1}",
        "label": f"{D46_PHYSICS_LABELS[p['p_i']]} vs {D46_PHYSICS_LABELS[p['p_j']]}",
        "embedding_distance_l2": float(p["l2"]),
        "floor": 0.1,
    } for p in pairs]
    _validate_rows("d46-embedding-distances", rows)
    caveat = (
        "All six pairwise distances between the four learned 16-dimensional "
        "physics codes at step 50: 4.6 to 7.0, every pair 46x-70x above the "
        "0.1 distinctness floor. This is the ALIVE half of the failure "
        "signature: the network did not ignore the labels -- it learned four "
        "genuinely distinct codes and routed all four into the same "
        "trivial-flux solution (fig2a). Licensed reading stops there: "
        "distinct codes at the conditioning layer certify only that the "
        "labels were not collapsed -- the record's own lesson is that this "
        "gate alone cannot certify the codes were doing physics-relevant "
        "work. Single realization / fixed cosmology; z=0.3 scope-lock."
    )
    extra = {"internal_lineage": f"[D-46] Addendum 2 (gate 7); {D46_GATES_JSON}"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig2b-embedding-distances.csv",
        fieldnames=["pair", "label", "embedding_distance_l2", "floor"],
        rows=rows,
        producing_fn="src.export.export_d46_embedding_distances",
        source_data_path=gates_json,
        physics_id=[1, 2, 3, 4],
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d46_loss_trace(out_dir: str, gates_json: str = D46_GATES_JSON) -> Dict[str, Any]:
    """ep07 fig3: the 50-step loss history -- the flat-after-step-10 story.
    RE-READ from the gates JSON. The per-step mean-flux trace was NOT logged
    (endpoints only); this is the one plottable trajectory."""
    d = _read_d46_gates(gates_json)
    hist = d["loss_history"]
    if len(hist) != 50:
        raise AssertionError(f"expected 50 loss steps, got {len(hist)}.")
    rows = [{"step": i + 1, "loss_total": float(v)} for i, v in enumerate(hist)]
    _validate_rows("d46-loss-trace", rows)
    caveat = (
        "The 50-step training-loss history for the pooled-physics smoke: the "
        "loss reaches ~0.013 by step 10 and stays there -- step-50/step-10 "
        "ratio 1.0036 against the <0.85 descent gate; the wiggle is noise "
        "around a flat line, not descent. This is the one per-step trajectory "
        "that was logged: the per-step mean flux was NOT (its per-variant "
        "step-50 endpoints, all 1.000, are in fig1/fig2a) -- so the tell is "
        "cited as endpoints, and only this loss curve is drawn. Single "
        "realization / fixed cosmology; z=0.3 scope-lock."
    )
    extra = {
        "gate2_ratio": d["gates"]["gate_2_loss_descent"]["observed_ratio"],
        "mean_flux_trace": "NOT logged per step; per-variant endpoints only (see fig1/fig2a)",
        "internal_lineage": f"[D-46] Addendum 2 (gate 2); {D46_GATES_JSON}",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="fig3-loss-trace.csv",
        fieldnames=["step", "loss_total"],
        rows=rows,
        producing_fn="src.export.export_d46_loss_trace",
        source_data_path=gates_json,
        physics_id=[1, 2, 3, 4],
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


D46_RUN_CONFIG: List[Tuple[str, Any]] = [
    ("intervention", "pool all four Sherwood feedback variants into ONE network, with a learned 16-dimensional code per variant appended to the input (the data axis; loss UNCHANGED)"),
    ("rationale", "the deficit had looked the same in all four variants from the start; read that sameness as a signal -- pooling multiplies the strong-absorption examples each gradient step sees at fixed sightline budget"),
    ("design_scale", "full design: 1,024 sightlines per variant (4,096 pooled), interleaved per microbatch -- never dispatched"),
    ("smoke_scale", "the 50-step smoke pooled 1,024 sightlines TOTAL across the four variants (microbatch 64, seed 0, 498,244 params)"),
    ("gate_discipline", "seven pre-committed gates; gate 3 HARDENED per the banked lesson of the last two collapses (mean flux tested away from BOTH the anchor and 1.000); gate 7 (label distinctness) NEW, added by the audit against the one residual risk it named"),
    ("audit_residual_risk", "the audit named 'the network ignores the labels' as the residual risk and built gate 7 to catch it; the failure arrived at a surface downstream of the labels instead -- gate 7 passed while the density heads died"),
    ("verdict", "3/7 gates FAIL (flat loss; the tell in four-part unison; all four density heads ~10,000x below the spread floor) -> retired at smoke"),
    ("what_ran", "50 steps, 196.6 s, on the host machine's GPU; no paid dispatch"),
    ("cost", "spent ~$0 paid (minutes of host compute); the paid full-scale ladder (~$1.50 first stage, ~$10-15 if escalated) was cancelled unrun"),
    ("stop_rule", "any single gate FAIL retires the intervention at smoke; the follow-up full-scale sprint was cancelled per the pre-committed stop"),
]


def export_d46_run_config(out_dir: str) -> Dict[str, Any]:
    """ep07 spec readout (BANKED): pooling design, gate hardening lineage,
    stop discipline, cost."""
    rows = [{"key": k, "value": str(v)} for k, v in D46_RUN_CONFIG]
    _validate_rows("d46-run-config", rows)
    caveat = (
        "Intervention spec of record for the fourth counterfactual, the data "
        "axis. Two scale numbers must never be conflated: the DESIGN pooled "
        "1,024 sightlines per variant (4,096 total), but the smoke that "
        "retired it pooled 1,024 total -- the design scale was never "
        "dispatched. The loss is byte-unchanged; the intervention is the "
        "data distribution plus a learned label. The gate set is itself part "
        "of the story: gate 3 is the previous episodes' tell-lesson written "
        "into procedure, and it fired correctly this time; gate 7 is the "
        "audit's named residual risk made into a check, and it passed -- the "
        "failure came from a surface the audit did not name. Cost stated the "
        "right way around: ~$0 paid spent; the paid ladder cancelled unrun. "
        "Single realization / fixed cosmology; z=0.3 scope-lock."
    )
    extra = {"internal_lineage": "[D-46] design spec (hypothesis, math contract, smoke gates, falsification rules) + Addendum 2 verdict; [D-53] hook"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir),
        filename="spec-run-config.csv",
        fieldnames=["key", "value"],
        rows=rows,
        producing_fn="src.export.export_d46_run_config",
        source_data_path="decision record (banked spec; see internal_lineage)",
        physics_id=[1, 2, 3, 4],
        caveat=caveat,
        extra_sidecar=extra,
    )
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


# =========================================================================== #
# ep08 "changing-the-target" batch — the GROUPED trainability-wall epic:
# the direct flux-power supervision arc, the seven-lever collapsed-basin
# cluster, and the three closing probes. The caveat-densest batch of the arc.
# fig1/fig2/spec BANKED from the decision record ([D-60] arc, [D-63]/[D-65]
# tables); fig3 RE-READ from the three on-disk verdict artifacts. The
# skip-rich rows carry the record's AMENDED readings (primary-observable
# swap; Wilcoxon direction; init-confound demotion of the per-seed deltas).
# =========================================================================== #

D63_CLUSTER_JSON_SOURCES: str = "decision record 5-attempt table + 2 diagnostic absorptions"
D69_FGPA_VERDICT_JSON: str = "experiments/nerf/artifacts/d62_3_stage1_verdict.json"
D69_LRPROBE_SUMMARY_JSON: str = "experiments/nerf/artifacts/d69_lr_probe/summary.json"
D71_SKIPRICH_VERDICT_JSON: str = "cloud_runs/stage1a-1b-skiprich-203337-verdict.json"

D60_ARC_ROWS: List[Dict[str, Any]] = [
    {"beat": "1", "label": "First dispatch of the direct flux-power objective",
     "what_happened": "retired mid-run at step 2,271: the task-balancing weights ran away (ratio ~1,154:1) under a SIMPLIFIED variant of the published balancing method (the full method segfaulted on the local development host, so this dispatch ran the simplified variant; the full method later ran on the cluster in the pilot chain)",
     "licensed_status": "instrument falsified, hypothesis narrowed -- the simplified balancing variant failed, NOT the direct-target idea itself"},
    {"beat": "2", "label": "Pilot chain: two wiring bugs found and fixed",
     "what_happened": "a silent 5,000-step null (the new term was never wired into the gradients), caught by a deliberately-broken test run; then a computation-graph break, caught the same way; then a clean pilot",
     "licensed_status": "the first 'result' was no result at all -- a bug, not evidence; the honesty beat is that the project proved its own null wrong before believing anything"},
    {"beat": "3", "label": "The clean pilot's verdict",
     "what_happened": "retired at step 200 under the pre-committed stop gate: the predicted flux-power variance had collapsed to 0.0063 of truth",
     "licensed_status": "real evidence, not a bug -- the first sighting of the collapsed basin"},
]

# The seven-lever cluster ([D-63] 5-attempt table + [D-65] 2 diagnostics),
# plus the healthy reference row (A7a control, production run).
D63_CLUSTER_ROWS: List[Dict[str, Any]] = [
    {"lever": "retune-1", "label": "Learning rate, first retune", "what_varied": "lr=1e-4, warmup 1000",
     "var_pf_band_ratio": 2.51e-6, "log10": -5.60, "per_task_ratio": "not logged"},
    {"lever": "retune-2", "label": "Learning rate, second retune", "what_varied": "lr=3e-5, warmup 2000",
     "var_pf_band_ratio": 6.9e-7, "log10": -6.16, "per_task_ratio": "20,807"},
    {"lever": "retune-3", "label": "Per-task gradient clipping", "what_varied": "per-task clip (dead lever)",
     "var_pf_band_ratio": 1.75e-6, "log10": -5.76, "per_task_ratio": "15,599"},
    {"lever": "L2", "label": "Loss reduction operator", "what_varied": "sum -> mean reduction",
     "var_pf_band_ratio": 2.93e-6, "log10": -5.53, "per_task_ratio": "1,429"},
    {"lever": "k-norm", "label": "The supervision target itself, k-normalized", "what_varied": "per-mode-normalized flux-power target",
     "var_pf_band_ratio": 7.46e-7, "log10": -6.13, "per_task_ratio": "1.008"},
    {"lever": "diag-mb", "label": "Microbatch size", "what_varied": "microbatch=1024",
     "var_pf_band_ratio": 7.63e-7, "log10": -6.12, "per_task_ratio": "1.010"},
    {"lever": "diag-P2", "label": "Physics variant", "what_varied": "P2 instead of P1",
     "var_pf_band_ratio": 3.74e-7, "log10": -6.43, "per_task_ratio": "1.008"},
]


def export_d60_direct_target_arc(out_dir: str) -> Dict[str, Any]:
    """ep08 fig1: the direct-target opening movement (BANKED)."""
    rows = list(D60_ARC_ROWS)
    _validate_rows("d60-direct-target-arc", rows)
    caveat = (
        "The opening movement of the supervision-target campaign: the network "
        "scored directly on the log flux power spectrum, balanced against the "
        "production data loss by a published two-task balancing method -- run "
        "in a SIMPLIFIED variant on the first dispatch because the full method "
        "segfaulted on the local development host; the full method ran on the "
        "cluster in the later pilot chain. Licensed status language is per-run and "
        "binding: the first dispatch falsified the simplified balancing "
        "INSTRUMENT and narrowed the hypothesis -- 'the direct-target idea "
        "was falsified' is BARRED at this beat (and stays barred until the "
        "retune cap exhausted, and then only at the scope of this campaign). "
        "The wiring-bug chain is narratable as an honesty beat: the first "
        "apparent null was a silent bug, caught by deliberately breaking the "
        "code to prove the test could see anything at all. Single realization "
        "/ fixed cosmology; z=0.3 scope-lock."
    )
    extra = {"internal_lineage": "[D-60] + gate-8/pilot/retune addenda; Juno 201587 (retire step 2271, R-g w_ratio ~1154:1); pilot chain 201602/201607/201669/201712 (clean pilot R-b step 200, var_pf_band_ratio 0.0063); GradNorm = Chen et al. 2018 (simplified proxy G_i = w_i|L_i|, alpha=0.12; full second-order path unit-tested but segfaulted on the dispatch host)"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig1-direct-target-arc.csv",
        fieldnames=["beat", "label", "what_happened", "licensed_status"],
        rows=rows, producing_fn="src.export.export_d60_direct_target_arc",
        source_data_path="decision record (banked arc; see internal_lineage)",
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d63_collapsed_basin_cluster(out_dir: str) -> Dict[str, Any]:
    """ep08 fig2: the seven-lever collapsed-basin cluster (BANKED from the
    decision-record tables), with the healthy reference carried in the
    sidecar UNDER its mandatory two-part caveat."""
    rows = list(D63_CLUSTER_ROWS)
    _validate_rows("d63-collapsed-basin-cluster", rows)
    vals = [r["var_pf_band_ratio"] for r in rows]
    band_decades = float(np.log10(max(vals)) - np.log10(min(vals)))
    if not 0.85 < band_decades < 0.95:
        raise AssertionError(f"cluster band {band_decades:.2f} decades != banked ~0.89.")
    caveat = (
        "The seven-lever collapsed-basin cluster: seven runs, each varying a "
        "different lever -- learning rate (twice), per-task clipping, the "
        "loss reduction operator, a re-normalized supervision target, "
        "microbatch size, and the physics variant -- and every one retired at "
        "the same step-200 stop gate with the predicted flux-power variance "
        "collapsed to between 3.7e-7 and 2.9e-6 of truth: a 0.89-decade band "
        "across every lever pulled. The sharpest single datum: the "
        "re-normalized target fixed the two-task gradient imbalance from "
        "~20,800:1 to 1.008 -- more than four decades -- and the collapse did "
        "not move. MANDATORY two-part caveat on every citation of this "
        "figure: healthy production runs hold this ratio near 1.0 from step "
        "5,000 onward, and the retired runs were read at step 200 with NO "
        "matched step-200 healthy reading banked -- a collapsed floor against "
        "a healthy plateau, not a matched-step comparison. Inference of "
        "record: per-task balance is NECESSARY BUT NOT SUFFICIENT; the "
        "pathology lives upstream of loss construction, in the structure of "
        "the supervision target itself. NOT licensed: 'the loss-construction "
        "class was exhausted' -- one named alternative was never tested, only "
        "de-prioritized on this evidence. Single realization / fixed "
        "cosmology; z=0.3 scope-lock."
    )
    extra = {
        "band_decades_rederived": band_decades,
        "healthy_reference": {
            "value": "var_pf_band_ratio ~ 1.0 from step 5000 (production run, recomputed control)",
            "two_part_caveat": "healthy runs hold ~1.0 from step 5000; retirements were read at step 200; no matched step-200 healthy control exists",
            "source": "experiments/nerf/artifacts/d73_a7_control/a7a_var_pf_control.json",
        },
        "knorm_datum": "per-task gradient ratio 20,807 -> 1.008 (~4.3 decades) with variance collapse unchanged (step-200 same-run pair from the authoritative seven-attempt table; the cut's '~22,000 -> 0.98' is a different, banked step-100 pair from the first, instrumentation-blocked dispatch, welded there to a variance outcome that belongs to the re-dispatch -- flagged for amendment)",
        "internal_lineage": "[D-63] 5-attempt table (201734/201814/201856/202109/202289) + [D-65] diagnostics (202291/202292); R8 re-verbed form binding",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig2-collapsed-basin-cluster.csv",
        fieldnames=["lever", "label", "what_varied", "var_pf_band_ratio", "log10", "per_task_ratio"],
        rows=rows, producing_fn="src.export.export_d63_collapsed_basin_cluster",
        source_data_path=D63_CLUSTER_JSON_SOURCES,
        physics_id=[1, 2], caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows),
            "band_decades": band_decades}


def export_d69_closing_probes(
    out_dir: str,
    fgpa_json: str = D69_FGPA_VERDICT_JSON,
    lrprobe_json: str = D69_LRPROBE_SUMMARY_JSON,
    skiprich_json: str = D71_SKIPRICH_VERDICT_JSON,
) -> Dict[str, Any]:
    """ep08 fig3: the three closing probes (RE-READ from the on-disk verdict
    artifacts, cross-checked against the banked record; skip-rich rows carry
    the record's AMENDED readings)."""
    fgpa = _read_banked_json(fgpa_json)
    if fgpa["verdict"] != "FAIL" or abs(fgpa["R_feas"] - 8.334e-3) > 1e-5:
        raise AssertionError("fGPA-residual verdict JSON disagrees with the banked record.")
    lr = _read_banked_json(lrprobe_json)
    verdicts = lr["per_cell_verdicts"]
    if verdicts.count("FAIL_SINKING") != 1 or len(verdicts) != 3:
        raise AssertionError("lr-probe verdicts disagree with the banked 1-of-3 record.")
    sk = _read_banked_json(skiprich_json)
    deltas = [sk["per_seed"][str(i)]["delta"] for i in range(10)]
    if not all(d < 0 for d in deltas):
        raise AssertionError("skip-rich deltas disagree with the banked 10/10-negative record.")
    var_ratios = [sk["per_seed"][str(i)]["var_pred"] / sk["per_seed"][str(i)]["var_truth"]
                  for i in range(10)]

    rows: List[Dict[str, Any]] = [
        {"probe": "feasibility", "label": "Could a physics-residual target even see the structure? (feasibility gate, no training run)",
         "headline_value": f"discrimination ratio {fgpa['R_feas']:.2e} vs a FAIL boundary of 1.0 (~120x below)",
         "verdict": "FALSIFIED before any training",
         "licensed_reading": "the physics-residual candidate target cannot tell structured truth from structureless noise in the band that matters; it was retired on a computed feasibility check -- no training run"},
        {"probe": "pretraining-lr", "label": "Direct density pretraining, three learning rates (the probe behind the mandatory 1-of-3 caveat)",
         "headline_value": "1 of 3 cells actively shrinking; 2 of 3 non-monotonic and unreadable",
         "verdict": "FROZEN by the pre-committed matrix (no cell passed)",
         "licensed_reading": "every citation carries the count: one cell of three showed active shrinkage; the project's own first 'all cells failing' framing was convicted in review and corrected"},
        {"probe": "skip-rich-body", "label": "A skip-rich network body under density pretraining, ten seeds (the architecture probe)",
         "headline_value": f"predicted-to-true variance ratio ~{np.median(var_ratios):.0e} at every seed (a ~3-decade deficit); all 10 per-seed deltas negative",
         "verdict": "FALSIFIED (narrow scope)",
         "licensed_reading": "primary observable is the ~3-decade variance deficit at step 500 (the per-seed deltas are init-confounded and demoted to a direction indicator); the improvement test returned p=1.0 -- maximally unsupported improvement, NOT a statistical test of regression; scope-locked to this architecture, this supervision, P1 z=0.3, 500 steps, an inherited learning rate -- it does NOT say the architecture axis fails"},
    ]
    _validate_rows("d69-closing-probes", rows)
    caveat = (
        "The three probes that closed the campaign, each read from its "
        "on-disk verdict artifact. The feasibility gate retired a candidate "
        "target on a computed feasibility check, before any training. The "
        "pretraining probe carries its "
        "MANDATORY count on every citation: 1 of 3 cells actively shrinking, "
        "2 of 3 unreadable -- and the corrected-in-review history of its own "
        "over-framing is narratable as an honesty beat. The skip-rich probe "
        "carries the record's amended readings: the primary observable is the "
        "~3-decade variance deficit (per-seed deltas init-confounded, "
        "demoted); the p=1.0 tests improvement, never regression; and the "
        "falsification is scope-locked -- one architecture under one "
        "supervision is not the architecture axis. Single realization / "
        "fixed cosmology; z=0.3 scope-lock."
    )
    extra = {
        "skiprich_var_ratios_per_seed": var_ratios,
        "skiprich_deltas_per_seed": deltas,
        "skiprich_bin_ratio": "strong-absorption bins mis-fit 10-13x worse than moderate bins (void-floor-saturation pathology)",
        "lrprobe_cell_verdicts": verdicts,
        "internal_lineage": (
            f"[D-69] base + am-7 ({lrprobe_json}); {fgpa_json}; [D-71] §A-§H + "
            f"AMENDMENT BLOCK ({skiprich_json}); [D-70] am-2 K1 (the 1-of-3 correction)"
        ),
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig3-closing-probes.csv",
        fieldnames=["probe", "label", "headline_value", "verdict", "licensed_reading"],
        rows=rows, producing_fn="src.export.export_d69_closing_probes",
        source_data_path=f"{fgpa_json} + {lrprobe_json} + {skiprich_json}",
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


D60_CAMPAIGN_CONFIG: List[Tuple[str, Any]] = [
    ("what_changed", "the supervision TARGET: score the network directly on the log flux power spectrum (later: on a re-normalized version; later still: on the density field itself via pretraining)"),
    ("balancing", "the two-task weighting used a published gradient-balancing method, run in a SIMPLIFIED variant on the first dispatch (the full method segfaulted on the local development host; unit-tested in isolation, it ran on the cluster in the later pilot chain)"),
    ("stop_discipline", "every paid run carried a pre-committed step-200 variance stop gate and a hard step cap; most runs retired at step 200 -- minutes into budgets of hours"),
    ("where_it_ran", "paid runs on the compute cluster; the feasibility gate and the pretraining probe on the host machine (minutes, no paid dispatch)"),
    ("cost_framing", "NO campaign-total dollar figure is banked for this group -- do not quote one; per-run budgets were small and capped, and most runs retired at tiny fractions of budget; the licensed form is qualitative: a sequence of small capped runs, each stopped by a pre-committed gate"),
    ("what_the_campaign_established", "a collapsed basin robust to every lever pulled on the loss side; per-task balance necessary but not sufficient; the pathology upstream, in the supervision-target structure"),
    ("what_it_did_not_establish", "that the direct-target idea is impossible (the first falsification was of a simplified instrument); that the loss-construction class is exhausted (one named alternative was de-prioritized, never tested); that the architecture axis fails (one architecture was falsified, narrowly)"),
]


def export_d60_campaign_config(out_dir: str) -> Dict[str, Any]:
    """ep08 spec readout (BANKED): what changing-the-target meant, the
    discipline, the honest cost framing."""
    rows = [{"key": k, "value": str(v)} for k, v in D60_CAMPAIGN_CONFIG]
    _validate_rows("d60-campaign-config", rows)
    caveat = (
        "Campaign spec of record for the grouped supervision-target epic. "
        "Two premise corrections are binding: (1) there is NO banked "
        "campaign-total dollar figure -- do not quote one; the licensed cost "
        "framing is qualitative (small capped runs, pre-committed stops, most "
        "retired minutes into their budgets). (2) The healthy variance "
        "reference (~1.0 from step 5000) comes from the production-run "
        "control artifact, NOT from the frozen-init calibration file -- and "
        "it always travels with its two-part caveat. Single realization / "
        "fixed cosmology; z=0.3 scope-lock."
    )
    extra = {"internal_lineage": "[D-60]/[D-62]/[D-63]/[D-64]/[D-65]/[D-67]/[D-68]/[D-69]/[D-70]/[D-71] governance chain; healthy ref = d73_a7_control/a7a_var_pf_control.json (NOT d70_m0_baseline/baseline.json, which is the frozen-init M0 calibration)"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="spec-campaign-config.csv",
        fieldnames=["key", "value"],
        rows=rows, producing_fn="src.export.export_d60_campaign_config",
        source_data_path="decision record (banked campaign spec; see internal_lineage)",
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


# =========================================================================== #
# ep09 "removing-the-clamp" batch — the [D-73] §E A1 head probe: remove the
# bounded density head, supervise log10(rho) directly, read the
# pre-registered ESCAPE/COLLAPSE verdict. Small and clean: one experiment,
# one pre-registered gate, verdict COLLAPSE head-invariantly. All figure
# artifacts RE-READ from experiments/nerf/artifacts/d73_a1_head_probe/.
# HAZARD (shipped as a premise correction): the per-cell JSONs carry a
# legacy harness `verdict` field from a VOIDED verdict matrix (AM-1) — the
# gate verdict is computed SOLELY from the variance/correlation trajectories;
# the exports below exclude that field.
# =========================================================================== #

D73_A1_DIR: str = "experiments/nerf/artifacts/d73_a1_head_probe"
D73_A1_SUMMARY: str = f"{D73_A1_DIR}/d73_summary.json"
D73_A1_ESCAPE_BAR: float = 0.1
D73_A1_PEARSON_BAR: float = 0.2


def _read_a1_cells(a1_dir: str) -> List[Dict[str, Any]]:
    import glob as _glob
    cells = []
    for path in sorted(_glob.glob(f"{a1_dir}/cell_*.json")):
        name = Path(path).name
        if name.startswith("._"):
            continue
        c = _read_banked_json(path)
        tr = c["var_ratio_trajectory"]
        final = tr[-1]
        cells.append({
            "head": c.get("head", "softplus"),
            "lr": float(c["lr_max"]), "seed": int(c["seed"]),
            "final_step": int(final["step"]),
            "final_var_ratio": float(final["ratio"]),
            "final_pearson_r": float(final["pearson_r"]),
        })
    return cells


def export_d73a1_per_cell_verdicts(out_dir: str, a1_dir: str = D73_A1_DIR) -> Dict[str, Any]:
    """ep09 fig1: all 15 in-directory cells (9 unclamped + 6 clamped
    controls), final variance ratio + correlation vs the ESCAPE bar."""
    cells = _read_a1_cells(a1_dir)
    linlog = [c for c in cells if c["head"] == "linear-log"]
    ctrl = [c for c in cells if c["head"] != "linear-log"]
    if len(linlog) != 9 or len(ctrl) != 6:
        raise AssertionError(f"expected 9 unclamped + 6 control cells; got {len(linlog)}+{len(ctrl)}.")
    lo = min(c["final_var_ratio"] for c in linlog)
    hi = max(c["final_var_ratio"] for c in linlog)
    if not (2.0e-6 < lo and hi < 7.0e-6):
        raise AssertionError(f"unclamped final ratios [{lo:.2e}, {hi:.2e}] outside the banked 2.5e-6-6.6e-6 record.")
    if any(abs(c["final_pearson_r"]) >= 0.2 for c in cells):
        raise AssertionError("a cell shows |pearson| >= 0.2 at the final step, contradicting the banked readout.")

    rows = [{
        "head": ("unclamped (linear on log density)" if c["head"] == "linear-log"
                 else "clamped control (bounded head)"),
        "lr": c["lr"], "seed": c["seed"],
        "final_var_ratio": c["final_var_ratio"],
        "final_pearson_r": c["final_pearson_r"],
        "escape_bar": D73_A1_ESCAPE_BAR,
        "distance_below_bar": f"{D73_A1_ESCAPE_BAR / c['final_var_ratio']:,.0f}x",
    } for c in sorted(cells, key=lambda c: (c["head"] != "linear-log", c["lr"], c["seed"]))]
    _validate_rows("d73a1-per-cell-verdicts", rows)
    caveat = (
        "All fifteen probe cells at the final recorded step: nine unclamped "
        "(the bounded density head replaced by a linear head on the log of "
        "the raw quantity) and six clamped controls. Every unclamped cell "
        "sits ~10,000x-100,000x below the 0.1 escape bar (finals 2.5e-6 to "
        "6.6e-6), correlations never leave the noise floor, and the clamped "
        "controls are statistically indistinguishable -- the two populations "
        "are the same population. The verdict of record is COLLAPSE, "
        "head-invariantly; the clamp is ACQUITTED at probe scope. Three "
        "additional legacy clamped seed-0 cells exist in the older probe "
        "directory (cite-only) and are not re-shipped. SCOPE-LOCK on every "
        "use: direct density supervision, reduced grid, 1,000 steps -- this "
        "probe does NOT test the flux-supervised regime. Single realization "
        "/ fixed cosmology; z=0.3 scope-lock."
    )
    extra = {
        "legacy_verdict_field_warning": (
            "per-cell JSONs carry a harness-legacy 'verdict' field from a "
            "VOIDED verdict matrix -- it is NOT the probe verdict and is "
            "excluded from this export; the gate verdict is computed solely "
            "from the variance/correlation trajectories"
        ),
        "internal_lineage": f"[D-73] §E + am-1 (AM-1 artifact isolation/voiding, AM-3 read-only control) + am-2 readout; {a1_dir}/cell_*.json; legacy seed-0 controls in d69_lr_probe/ (cite-only)",
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig1-per-cell-verdicts.csv",
        fieldnames=["head", "lr", "seed", "final_var_ratio", "final_pearson_r",
                    "escape_bar", "distance_below_bar"],
        rows=rows, producing_fn="src.export.export_d73a1_per_cell_verdicts",
        source_data_path=f"{a1_dir}/cell_*.json",
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d73a1_median_trajectories(out_dir: str, summary_json: str = D73_A1_SUMMARY) -> Dict[str, Any]:
    """ep09 fig2: the median-across-seeds trajectories per learning-rate cell
    (six recorded steps, 0..1000) -- variance ratio and correlation together,
    the two conditions the pre-registered gate tests jointly."""
    d = _read_banked_json(summary_json)
    if d["verdict"] != "COLLAPSE" or d["escape_hits"]:
        raise AssertionError("summary verdict disagrees with the banked COLLAPSE record.")
    rows: List[Dict[str, Any]] = []
    for lr_label, traj in d["median_trajectories_by_lr_cell"].items():
        for pt in traj:
            rows.append({
                "lr_cell": lr_label, "step": int(pt["step"]),
                "median_var_ratio": float(pt["median_ratio"]),
                "median_pearson_r": float(pt["median_pearson_r"]),
                "n_seeds": int(pt["n_seeds"]),
            })
    _validate_rows("d73a1-median-trajectories", rows)
    if max(r["median_var_ratio"] for r in rows) >= D73_A1_ESCAPE_BAR:
        raise AssertionError("a median trajectory point clears the escape bar, contradicting COLLAPSE.")
    caveat = (
        "Median-across-seeds trajectories for the unclamped cells at the six "
        "recorded steps (0 to 1,000), variance ratio and correlation "
        "together -- the two conditions the pre-registered gate tests "
        "jointly at the same step. No trajectory approaches either bar "
        "(ratio 0.1; correlation 0.2): the highest median variance ratio "
        "ever recorded is ~6e-5 at step 100, decaying afterward, and the "
        "median correlation never leaves the noise floor (null 95% bound "
        "~0.06). A real per-step trace survived for this probe -- draw it; "
        "endpoints-only rules do not apply here. Single realization / fixed "
        "cosmology; z=0.3 scope-lock."
    )
    extra = {"internal_lineage": f"[D-73] §E gate computed from these trajectories; {summary_json}"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig2-median-trajectories.csv",
        fieldnames=["lr_cell", "step", "median_var_ratio", "median_pearson_r", "n_seeds"],
        rows=rows, producing_fn="src.export.export_d73a1_median_trajectories",
        source_data_path=summary_json,
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


D73_A1_GATE_SPEC: List[Dict[str, Any]] = [
    {"element": "the disjunct", "label": "ESCAPE or COLLAPSE, both informative",
     "spec": "ESCAPE iff, in at least one learning-rate cell, some recorded step has median variance ratio > 0.1 AND median correlation >= 0.2 at the SAME step; otherwise COLLAPSE",
     "why": "either outcome decides something: ESCAPE convicts the clamp; COLLAPSE acquits it and points upstream"},
    {"element": "the spurious-fire guard", "label": "Ratio and structure must fire together",
     "spec": "a ratio pass with a correlation fail at every such step is logged ESCAPE-UNSTABLE and routed as COLLAPSE",
     "why": "guards two false-fire channels named in advance: an unclamped head at initialization can clear the ratio bar with zero structure, and a few extreme voxels can dominate the variance"},
    {"element": "the null anchor", "label": "The correlation bar sits above chance",
     "spec": "correlation bar 0.2 vs a null 95% bound of ~0.06 on the fixed probe sample",
     "why": "an escape must show structure a random field cannot"},
    {"element": "the voided legacy verdict", "label": "The reused harness's old verdict matrix does not rule",
     "spec": "per-cell files carry a legacy 'verdict' field from the older probe harness; it is VOID for this gate -- the verdict is computed solely from the recorded trajectories",
     "why": "reusing an instrument must not mean inheriting its old rules"},
]


def export_d73a1_gate_spec(out_dir: str) -> Dict[str, Any]:
    """ep09 fig3: the pre-registered gate, as quotable spec rows (BANKED)."""
    rows = list(D73_A1_GATE_SPEC)
    _validate_rows("d73a1-gate-spec", rows)
    caveat = (
        "The pre-registered gate, written and panel-amended before the run. "
        "Both outcomes were declared informative in advance -- an escape "
        "would convict the clamp, a collapse acquits it -- so the probe "
        "could not produce a spin-dependent result. The spurious-fire guard "
        "is the arc's checks-that-learn motif continued: the gate names, in "
        "advance, two specific ways a naive threshold could false-fire, and "
        "requires variance AND structure at the same recorded step. Single "
        "realization / fixed cosmology; z=0.3 scope-lock."
    )
    extra = {"internal_lineage": "[D-73] §E rule verbatim + amendment-1 AM-1 (voiding), AM-2 (guard + null bound), AM-4 (frame correction)"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig3-gate-spec.csv",
        fieldnames=["element", "label", "spec", "why"],
        rows=rows, producing_fn="src.export.export_d73a1_gate_spec",
        source_data_path="decision record (banked pre-registration; see internal_lineage)",
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


D73_A1_PROBE_CONFIG: List[Tuple[str, Any]] = [
    ("what_removing_the_clamp_means", "the bounded (always-positive) density head is replaced by a plain linear head whose output IS the log of the raw density -- no clamp, no floor, nothing for the optimizer to hide behind"),
    ("supervision", "direct density supervision (the probe regime): the network is scored on the density field itself, not on the flux"),
    ("config", "reduced grid (64 per side), fiducial variant P1, z=0.3, 1,000 steps per cell, 3 learning rates x 3 seeds unclamped + 6 clamped controls"),
    ("where_it_ran", "the host machine, ~5-6 seconds per cell (one first-run outlier ~2.5 minutes); the whole probe fits far inside its pre-committed 1 CPU-hour cap"),
    ("cost", "spent: about four minutes of host compute across all fifteen cells, $0 paid; nothing was saved or skipped -- this probe was always host-scale by design"),
    ("verdict", "COLLAPSE, head-invariantly: all nine unclamped cells ~10,000x-100,000x below the escape bar; the six clamped controls indistinguishable"),
    ("what_it_unlocks", "the root-cause sentence, at exactly this scope: the collapse is upstream of the output clamp, in the optimization/loss landscape -- under direct density supervision, at this grid, for this step count"),
    ("what_it_does_not_test", "the flux-supervised regime (the production lesson); no sentence about the flux lesson may cite this probe"),
]


def export_d73a1_probe_config(out_dir: str) -> Dict[str, Any]:
    """ep09 spec readout (BANKED)."""
    rows = [{"key": k, "value": str(v)} for k, v in D73_A1_PROBE_CONFIG]
    _validate_rows("d73a1-probe-config", rows)
    caveat = (
        "Probe spec of record. The scope-lock is the hardest-working line in "
        "this episode: this probe supervises the density DIRECTLY, at a "
        "reduced grid, for a bounded step count -- it does not touch the "
        "flux-supervised regime, and the acquittal it delivers is an "
        "acquittal of the clamp AS THE CAUSE OF THIS COLLAPSE, at this "
        "scope; never 'the head plays no role anywhere.' Cost is stated "
        "as-spent (host-only, $0 paid, always host-scale by design -- there "
        "was no paid stage to save here). Single realization / fixed "
        "cosmology; z=0.3 scope-lock."
    )
    extra = {"internal_lineage": "[D-73] §E pre-registration + Pulse close-out row (root-cause sentence scope); harness scripts/d69_lr_sensitivity_probe.py --head linear-log; head site src/models/nerf.py:180"}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="spec-probe-config.csv",
        fieldnames=["key", "value"],
        rows=rows, producing_fn="src.export.export_d73a1_probe_config",
        source_data_path="decision record (banked probe spec; see internal_lineage)",
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


# =========================================================================== #
# ep10 "the-grid-probe" batch ([D-73] (1d') free-voxel-grid close-out).
# fig1/fig3/fig4/spec RE-READ from banked run artifacts (P_F sharpener,
# verification battery, in-job xi profile, Wiener L-sweep, healthy control,
# and the run's own DVC-preserved metric store); nothing recomputed from
# checkpoints. The heaviest verb-ceiling in the arc rides these caveats:
# under-constraint verbs are licensed HERE and only here (K2), always with
# "under this FGPA forward model" + z=0.3 + the integrator-slack caveat.
# =========================================================================== #

D73_GRID_RUN_DIR: str = (
    "cloud_runs/d73-1dprime-voxel192-P1-z0.3-c6f3aed-20260618-000035-8b4e90"
)
D73_GRID_SHARPENER: str = "experiments/nerf/artifacts/d73_1dprime/pf_sharpener.json"
D73_GRID_BATTERY: str = "experiments/nerf/artifacts/d73_1dprime/verification_battery.json"
D73_GRID_XI_INJOB: str = f"{D73_GRID_RUN_DIR}/eval/xi_3d_injob.json"
D73_GRID_METRIC_FILE: str = (
    f"{D73_GRID_RUN_DIR}/mlflow/254171605862827701/"
    "257cb3e536244eeea4c7e6fdf41a3432/metrics/l1_var_pf_band_ratio"
)
D73_GRID_RUN_META: str = (
    f"{D73_GRID_RUN_DIR}/mlflow/254171605862827701/"
    "257cb3e536244eeea4c7e6fdf41a3432/meta.yaml"
)
D73_A7_CONTROL: str = "experiments/nerf/artifacts/d73_a7_control/a7a_var_pf_control.json"
D73_WIENER_LSWEEP: str = "experiments/nerf/artifacts/wiener_baseline/a4prime_wiener_lsweep.json"

D73_GRID_PF_GATE: float = 0.10
D73_GRID_MLP_PF_BASELINE: float = 0.4155  # == \PFresidGridBaseline (eval n_rays=1024 on a net TRAINED at 64).
D73_GRID_TRAJ_SAMPLE_STEPS: Tuple[int, ...] = (
    1, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000,
    7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000,
)


def export_d73grid_flux_gates(
    out_dir: str, sharpener_json: str = D73_GRID_SHARPENER
) -> Dict[str, Any]:
    """ep10 fig1: the flux-power gate verdicts — the grid PASSES the gate the
    production network failed. Verdicts comparable; magnitudes NOT."""
    d = _read_banked_json(sharpener_json)
    resid = float(d["mean_abs_rel_diff_in_band"])
    if not (0.034 < resid < 0.036 and bool(d["PASS"])):
        raise AssertionError(f"sharpener residual {resid} / PASS flag disagree with the banked 0.0352 PASS.")
    if abs(float(d["gate"]) - D73_GRID_PF_GATE) > 1e-12:
        raise AssertionError("sharpener gate != banked 0.10.")
    if abs(float(d["mlp_pubt1_baseline"]) - D73_GRID_MLP_PF_BASELINE) > 1e-6:
        raise AssertionError("baseline in sharpener artifact drifted from the banked 0.4155.")
    if int(d["n_rays"]) != 1024 or int(d["n_kbins_in_band"]) != 10:
        raise AssertionError("sharpener eval geometry drifted from the banked n_rays=1024 / 10-bin band.")

    rows = [
        {"label": "free voxel grid (this episode's run)",
         "band_mean_flux_power_residual": resid,
         "gate": D73_GRID_PF_GATE, "verdict": "PASS",
         "training_sightlines": 1024,
         "note": ("scored under the same convention and the same 10% bar as the "
                  "production network; band-mean over the ten measured "
                  "wavenumber bins")},
        {"label": "production network (the arc's baseline)",
         "band_mean_flux_power_residual": D73_GRID_MLP_PF_BASELINE,
         "gate": D73_GRID_PF_GATE, "verdict": "FAIL",
         "training_sightlines": 64,
         "note": ("verdicts are directly comparable (same convention, same bar); "
                  "the residual magnitudes are NOT a controlled same-configuration "
                  "contrast — never quote a ratio between these two numbers")},
    ]
    _validate_rows("d73grid-flux-gates", rows)
    caveat = (
        "The gate finally opens: the free voxel grid's band-mean flux-power "
        "residual is 0.0352 against the 0.10 gate — PASS — where the "
        "production network failed at 0.4155. BINDING config caveat: the "
        "grid trained on 1,024 sightlines (the gate-definition evaluation "
        "density); the production network trained on 64. Same evaluation "
        "convention and same bar, so the PASS/FAIL VERDICTS are directly "
        "comparable; the residual MAGNITUDES are not a controlled "
        "same-configuration contrast and no ratio between them may be "
        "quoted ('the grid passed where the network failed', never 'the "
        "grid's error was N times smaller'). 'Closes both flux gates' is an "
        "enumerated claim: the trainability check and this flux-power gate, "
        "and only those — no verdict is banked for this run on the "
        "mean-flux or pixel-distribution gates. Single realization / fixed "
        "cosmology; z=0.3 scope-lock."
    )
    extra = {
        "grid_mean_flux_context": {
            "F_pred_mean": float(d["F_pred_mean"]), "F_truth_mean": float(d["F_truth_mean"]),
            "bar": ("context only — NO gate verdict is banked on the mean flux for this "
                    "run; do not narrate these two numbers as a pass or a fail; stating "
                    "the gap as unscored context ('the grid's mean transmitted flux sits "
                    "at 0.928 against the truth's 0.979') is licensed — only a pass/fail "
                    "reading is barred"),
        },
        "internal_lineage": (
            "[D-73] am-10 §10a/§10e; scripts/d73_pf_sharpener.py on step_050000.pt "
            "(Juno job 221335, commit c6f3aed); baseline = \\PFresidPone pub-t1 P1 "
            "(eval n_rays=1024/eval_seed=42 on a net TRAINED at n_rays=64)"
        ),
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig1-flux-gates.csv",
        fieldnames=["label", "band_mean_flux_power_residual", "gate", "verdict",
                    "training_sightlines", "note"],
        rows=rows, producing_fn="src.export.export_d73grid_flux_gates",
        source_data_path=sharpener_json,
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d73grid_trainability(
    out_dir: str,
    metric_file: str = D73_GRID_METRIC_FILE,
    a7_control_json: str = D73_A7_CONTROL,
) -> Dict[str, Any]:
    """ep10 fig2: the trainability trace — the grid's per-step variance ratio
    (full 50k-step store, downsampled), the healthy production reference at
    its checkpoint cadence, and the prior campaign's collapsed floor."""
    series: Dict[int, float] = {}
    with open(metric_file, "r", encoding="utf-8") as fh:
        for line in fh:
            _ts, val, step = line.split()
            series[int(step)] = float(val)
    if len(series) != 50000:
        raise AssertionError(f"metric store has {len(series)} points, not the banked 50,000.")
    tail = [v for s, v in series.items() if s >= 5000]
    if not (1.0958 <= min(tail) and max(tail) <= 1.0971):
        raise AssertionError(
            f"from-5k band [{min(tail):.4f}, {max(tail):.4f}] drifted from the banked flat 1.0959–1.0970.")
    if abs(series[50000] - 1.0959) > 5e-4:
        raise AssertionError("final var_pf_band_ratio drifted from the banked 1.0959.")

    rows: List[Dict[str, Any]] = []
    for s in D73_GRID_TRAJ_SAMPLE_STEPS:
        label = ""
        if s == 1:
            label = ("near-zero start = the pre-committed spatially-uniform "
                     "initialization, not a collapse verdict")
        elif s == 50000:
            label = "flat from step 5,000 to 50,000 (band 1.0959–1.0970)"
        rows.append({"series": "free voxel grid (this run)", "step": s,
                     "var_pf_band_ratio": series[s], "label": label})

    a7 = _read_banked_json(a7_control_json)
    for pt in a7["var_pf_trajectory"]:
        v = float(pt["var_pf_band_ratio"])
        if not 0.97 < v < 1.08:
            raise AssertionError(f"healthy control point {v} outside the banked ~1.0 plateau.")
        rows.append({"series": "healthy production reference", "step": int(pt["step"]),
                     "var_pf_band_ratio": v, "label": ""})

    for r in D63_CLUSTER_ROWS:
        rows.append({"series": "collapsed floor, prior campaign (read at step 200)",
                     "step": 200, "var_pf_band_ratio": float(r["var_pf_band_ratio"]),
                     "label": str(r["label"])})
    _validate_rows("d73grid-trainability", rows)
    caveat = (
        "The trainability verdict: the grid's predicted-to-true flux-power "
        "band-variance ratio rises from its (spatially uniform, "
        "pre-committed) initialization to ~1.096 and holds FLAT from step "
        "5,000 to 50,000 (recorded band 1.0959–1.0970) — the grid trains "
        "where the previous campaign's runs collapsed, and 'escaping the "
        "collapse' is licensed language for exactly this observable. TWO "
        "mandatory caveats. (1) Cadence: the collapsed-floor rows were read "
        "at their step-200 stop gate; the healthy reference exists only from "
        "step 5,000 (checkpoint cadence) — a collapsed floor against later "
        "plateaus, not a matched-step comparison. (2) The near-zero early "
        "steps reflect the smooth initialization, NOT a visit to the prior "
        "campaign's basin — the configurations differ (different "
        "representation, different sightline density, and a different "
        "training objective: the collapsed-floor runs were trained to match "
        "the flux power spectrum directly, while the grid was trained on the "
        "production flux lesson with this variance ratio recorded as a "
        "passive readout); do not narrate 'it started in the basin and "
        "climbed out'. Single realization / fixed cosmology; z=0.3 "
        "scope-lock."
    )
    extra = {"internal_lineage": (
        "[D-73] am-9 §9a (var_pf_band_ratio=1.0959 flat 5k→50k); metric "
        "l1_var_pf_band_ratio from the run's DVC-preserved MLflow file-store "
        "(run 257cb3e5, Juno job 221335); healthy ref = A7(a) control "
        "recompute (pub-t1 P1 seed-0 checkpoints); floor rows = [D-63]/[D-65] "
        "seven-lever table"
    )}
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig2-trainability-trace.csv",
        fieldnames=["series", "step", "var_pf_band_ratio", "label"],
        rows=rows, producing_fn="src.export.export_d73grid_trainability",
        source_data_path=f"{metric_file} + {a7_control_json}",
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d73grid_k2(
    out_dir: str, battery_json: str = D73_GRID_BATTERY
) -> Dict[str, Any]:
    """ep10 fig3: the decisive datum — the TRUE fields, forward-modeled
    through the SAME integrator, fit the observed flux ~4x WORSE than the
    grid does. Under-constraint verbs are licensed by this row set only."""
    d = _read_banked_json(battery_json)
    k2 = d["K2"]
    grid_loss = float(k2["grid_loss_data"])
    truth_best = float(k2["truth_loss_best"])
    truth_amp1 = float(k2["truth_loss_at_amp1"])
    truth_gridamp = float(k2["truth_loss_at_grid_amp"])
    if abs(grid_loss - 0.0026) > 1e-6:
        raise AssertionError("grid loss drifted from the banked 0.0026.")
    if not (0.0100 < truth_best <= truth_gridamp <= truth_amp1 < 0.0102):
        raise AssertionError("truth-loss sweep values drifted from the banked ~0.0101 flat band.")
    margin = truth_best / grid_loss
    if not 3.7 < margin < 4.0:
        raise AssertionError(f"margin quotient {margin:.2f} outside the banked '~4x'.")
    amp_spread = truth_amp1 - truth_best
    if amp_spread > 1e-5:
        raise AssertionError("amplitude sweep no longer flat (spread > 1e-5).")

    rows = [
        {"label": "the true fields, best free amplitude in the sweep",
         "field_through_integrator": "truth (density, temperature, neutral fraction, velocity at the exact ray points)",
         "amplitude_setting": float(k2["best_tau_amp"]),
         "flux_data_loss": truth_best,
         "note": ("'best' lands at the sweep edge; the three banked points "
                  "agree to ~5e-6 and the recorded sweep verdict is flat "
                  "across [0.2, 4] — amplitude calibration cannot close the "
                  "gap")},
        {"label": "the true fields, amplitude fixed at 1",
         "field_through_integrator": "truth", "amplitude_setting": 1.0,
         "flux_data_loss": truth_amp1, "note": ""},
        {"label": "the true fields, at the grid's own learned amplitude",
         "field_through_integrator": "truth", "amplitude_setting": 1.526,
         "flux_data_loss": truth_gridamp, "note": ""},
        {"label": "the free voxel grid, as trained",
         "field_through_integrator": "the grid's four fields",
         "amplitude_setting": 1.526, "flux_data_loss": grid_loss,
         "note": "about four times better than the true field through the same instrument"},
    ]
    _validate_rows("d73grid-k2", rows)
    caveat = (
        "The decisive datum, and the arc's heaviest caveat set rides it. "
        "The TRUE fields, forward-modeled through the SAME integrator under "
        "the SAME loss with a free amplitude, score 0.0101 — flat across the "
        "swept amplitude range [0.2, 4] — while the grid achieved 0.0026: "
        "the grid fits the observed flux about four times better than the "
        "truth does ('about four times', never a more precise multiple). "
        "What this LICENSES, exactly: minimizing this flux loss does not "
        "identify the true 3D field — the z=0.3 flux inverse problem, under "
        "this forward model, is under-constrained. THE quotable caveat "
        "(mandatory beside the 4x): the truth's 0.0101 residual is OUR OWN "
        "forward model's error floor, not nature's — the margin includes "
        "slack the grid can exploit and the truth cannot, so 'four times "
        "better than truth' must never be read as 'four times closer to "
        "reality'. The non-trivial content is the MARGIN plus the "
        "amplitude-FLATNESS (which rules out the amplitude-calibration "
        "escape), not bare 'grid beats truth' — the grid is the argmin of "
        "this loss, so beating any single field is near-guaranteed by "
        "construction. NOT licensed: that the flux intrinsically cannot "
        "determine the field; any claim beyond z=0.3; separating "
        "problem-intrinsic from forward-model-induced under-determination "
        "(entangled at 98% mean transmission). Single realization / fixed "
        "cosmology; z=0.3 scope-lock."
    )
    extra = {
        "margin_quotient_rederived": margin,
        "internal_lineage": (
            "[D-73] am-9 §9b (K2, panel-discharged); "
            "scripts/d73_k2_verification_battery.py variant (a) truth-sightline "
            "fields at exact ray points; loss matched to pipeline.py:2617-2646; "
            "RSD applied identically on both sides (no frame mismatch enters K2)"
        ),
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig3-truth-vs-grid.csv",
        fieldnames=["label", "field_through_integrator", "amplitude_setting",
                    "flux_data_loss", "note"],
        rows=rows, producing_fn="src.export.export_d73grid_k2",
        source_data_path=battery_json,
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows),
            "margin": margin}


def export_d73grid_xi(
    out_dir: str,
    xi_injob_json: str = D73_GRID_XI_INJOB,
    battery_json: str = D73_GRID_BATTERY,
    wiener_lsweep_json: str = D73_WIENER_LSWEEP,
) -> Dict[str, Any]:
    """ep10 fig4: the structure it actually holds — the grid's 3D correlation
    profile, ceiling-relative, with the demoted bar and the classical
    lower-bound reference."""
    xi = _read_banked_json(xi_injob_json)
    bat = _read_banked_json(battery_json)
    ws = _read_banked_json(wiener_lsweep_json)

    ceiling = float(bat["S5"]["xi_truth_vs_truth_at_r2"])
    noise = bat["S5"]["xi_perturbed_at_r2"]
    grid_at_gate = float(xi["xi_at_gate_r"])
    if abs(ceiling - 0.0298) > 5e-4 or abs(grid_at_gate - 0.0075) > 5e-4:
        raise AssertionError("ceiling / grid gate-scale xi drifted from the banked 0.0298 / 0.0075.")
    frac = grid_at_gate / ceiling
    if not 0.23 < frac < 0.27:
        raise AssertionError(f"ceiling fraction {frac:.3f} outside the banked ~25%.")
    if not grid_at_gate < float(noise["noise_std_1.0x"]):
        raise AssertionError("grid no longer sits below the truth+100%-noise reference.")
    peak = ws["peak"]
    if abs(float(peak["xi_3d_peak_at_r2"]) - 0.0789) > 5e-4 or not peak["peak_at_window_boundary"]:
        raise AssertionError("Wiener lower bound drifted from the banked ≥0.079 (rising at the wall).")

    rows: List[Dict[str, Any]] = []
    r_centers = [float(r) for r in xi["r_centers_mpc_h"]]
    xi_vals = [float(v) for v in xi["xi"]]
    # The banked gate-scale reading is SOLELY the shell whose value the run's
    # own eval banked as xi_at_gate_r (the 1.75-centred shell); the equidistant
    # 2.25 shell carries no banked gate status (PI gate B1).
    gate_idx = min(range(len(xi_vals)), key=lambda i: abs(xi_vals[i] - grid_at_gate))
    for i, (rc, val) in enumerate(zip(r_centers, xi_vals)):
        label = ""
        if i == gate_idx:
            label = ("the gate-scale reading: ~25% of the achievable ceiling, "
                     "below the truth-plus-noise reference")
        elif abs(rc - float(xi["gate_r_mpc_h"])) <= abs(r_centers[gate_idx] - float(xi["gate_r_mpc_h"])):
            label = ("adjacent shell, shown for the profile only — the banked "
                     "gate-scale reading is the 1.75-centred shell")
        rows.append({"series": "free voxel grid vs truth", "r_mpc_h": rc,
                     "xi": val, "label": label})
    rows += [
        {"series": "truth vs itself (the achievable ceiling)", "r_mpc_h": 2.0,
         "xi": ceiling,
         "label": ("even a PERFECT reconstruction scores only this under the "
                   "estimator as implemented — the old 0.6 bar is unreachable")},
        {"series": "truth + 25% noise", "r_mpc_h": 2.0,
         "xi": float(noise["noise_std_0.25x"]), "label": ""},
        {"series": "truth + 50% noise", "r_mpc_h": 2.0,
         "xi": float(noise["noise_std_0.5x"]), "label": ""},
        {"series": "truth + 100% noise", "r_mpc_h": 2.0,
         "xi": float(noise["noise_std_1.0x"]),
         "label": "the grid sits below this reference"},
    ]
    for pt in ws["L_sweep"]:
        rows.append({
            "series": "classical linear reconstruction (smoothing-length sweep)",
            "r_mpc_h": 2.0, "xi": float(pt["xi_at_r2_interp"]),
            "label": (f"smoothing length {pt['L_mpc_h']:g} Mpc/h"
                      + ("; still RISING at the compute wall — the best case is a "
                         "LOWER BOUND, never an information floor"
                         if float(pt["L_mpc_h"]) == 3.0 else "")),
        })
    rows.append({"series": "the demoted acceptance bar", "r_mpc_h": 2.0, "xi": 0.6,
                 "label": ("unreachable as implemented: no field, even truth vs "
                           "itself, can pass it — retired as a headline, shown "
                           "only to explain the demotion")})
    _validate_rows("d73grid-xi", rows)
    caveat = (
        "The structure the grid actually holds — stated ONCE, neutrally, "
        "ceiling-relative. The estimator returns a correlation FUNCTION "
        "(decays with distance), not the per-distance coefficient its old "
        "0.6 acceptance bar assumed, so even truth-vs-itself scores 0.0298 "
        "at the gate scale: the bar is unreachable as implemented and is "
        "retired as a headline. Against the achievable ceiling the grid "
        "recovers ~25% (0.0075 of 0.0298) and sits below the "
        "truth-plus-100%-noise reference (0.0211): 3D structure recovery is "
        "genuinely WEAK — but 'catastrophic fail vs 0.6' is a retired "
        "framing and must not appear. A real-space-versus-redshift-space "
        "frame mismatch (line-of-sight displacement ~1.3–2.5 Mpc/h, "
        "comparable to the 2 Mpc/h gate scale) depresses EVERY number on "
        "this figure's family — the classical reference included — "
        "independent of reconstruction error. The classical linear "
        "reconstruction's 0.079 is a self-anchored LOWER BOUND (still "
        "rising at the compute wall): 'a classical method's best case we "
        "could reach', never 'the floor any method must beat', and NOT "
        "configuration-matched to the grid (different reconstruction "
        "support and smoothing) — no 'classical beats neural by N times' "
        "reading is licensed. This whole figure is SUPPORTING evidence; "
        "the flux-loss comparison carries the verdict. Single realization "
        "/ fixed cosmology; z=0.3 scope-lock."
    )
    extra = {
        "ceiling_fraction_rederived": frac,
        "gate_scale_note": ("the grid profile's gate-scale reading is the shell "
                            "centered at r=1.75 Mpc/h (bin nearest the 2 Mpc/h "
                            "gate radius), as banked in the run's own eval"),
        "internal_lineage": (
            "[D-73] am-9 §9c (S5 unreachable-as-implemented + S7 frame confound; "
            "0.6-gate DEMOTED; [D-36] correction note); grid profile = in-job "
            "eval xi_3d_injob.json (job 221335); ceiling/noise = "
            "verification_battery.json S5; classical = A4' Wiener L-sweep "
            "(R14 self-anchored, L≥3.5 RAM-bound)"
        ),
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="fig4-xi-ceiling-relative.csv",
        fieldnames=["series", "r_mpc_h", "xi", "label"],
        rows=rows, producing_fn="src.export.export_d73grid_xi",
        source_data_path=f"{xi_injob_json} + {battery_json} + {wiener_lsweep_json}",
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}


def export_d73grid_probe_config(
    out_dir: str,
    sharpener_json: str = D73_GRID_SHARPENER,
    run_meta_yaml: str = D73_GRID_RUN_META,
) -> Dict[str, Any]:
    """ep10 spec readout: what the grid probe is, its pre-commitments, scale,
    and the as-banked cost shape (no dollar figure is banked for this run)."""
    d = _read_banked_json(sharpener_json)
    start_ms = end_ms = None
    with open(run_meta_yaml, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("start_time:"):
                start_ms = int(line.split(":", 1)[1].strip())
            elif line.startswith("end_time:"):
                end_ms = int(line.split(":", 1)[1].strip())
    if start_ms is None or end_ms is None:
        raise AssertionError("run meta lacks start/end timestamps.")
    wall_h = (end_ms - start_ms) / 3.6e6
    if not 6.5 < wall_h < 8.0:
        raise AssertionError(f"recorded wall {wall_h:.2f} h drifted from the banked ~7.2 h.")

    rows_kv: List[Tuple[str, str]] = [
        ("what_the_probe_is",
         "replace the network entirely with the most expressive field this study "
         "allows: four free voxel grids, one per physical field (log density, "
         "temperature, neutral fraction, velocity) — a value free at every one of "
         "192-cubed cells, about 28 million free parameters, nothing shared "
         "between locations"),
        ("the_lesson",
         "the production flux objective, byte-identical: it was proven before "
         "dispatch that the sequence of numbers the optimizer sees is "
         "bit-for-bit identical to the production loss (the trainability "
         "readout is a detached diagnostic that touches no gradient)"),
        ("initialization",
         "pre-committed before the run: an absorption-rich start with initial "
         "mean transmitted flux 0.9270 — no post-hoc initialization tuning"),
        ("sightlines",
         "1,024 sightlines from the same simulated survey — the density the "
         "study's gates are defined at; the production network TRAINED on 64, "
         "which is one reason grid-vs-network residual magnitudes are never a "
         "controlled contrast"),
        ("run_scale",
         f"50,000 steps, one job, one datacenter GPU; about "
         f"{wall_h:.1f} recorded wall-clock hours (run start to end, in-job "
         "evaluation included)"),
        ("cost",
         "one GPU job inside a pre-committed 30-GPU-hour budget, about a "
         "quarter of it used; no dollar figure is banked for this dispatch — "
         "print the budget shape, not a price"),
        ("mean_flux_context",
         f"the run's evaluated mean transmitted flux is "
         f"{float(d['F_pred_mean']):.4f} vs truth {float(d['F_truth_mean']):.4f}; "
         "context only — NO verdict is banked on the mean-flux or "
         "pixel-distribution gates for this run"),
        ("what_it_settles",
         "at z=0.3, the most expressive field this study allows closes both "
         "flux gates the study defines and fits the observed flux better than "
         "the true field does through the same instrument — yet recovers only "
         "weak 3D structure: the z=0.3 flux inverse problem, under this "
         "forward model, is under-constrained; escaping the collapse was "
         "necessary but not sufficient"),
        ("what_it_does_not_settle",
         "whether the wall is the universe's information content or our own "
         "approximate forward model (not separated at 98% mean transmission); "
         "anything beyond z=0.3 (where the forest is dense, at z≈2–3, "
         "tomography demonstrably works); denser-sightline tiers (not run — "
         "the study rests on the 1,024-sightline probe)"),
    ]
    rows = [{"key": k, "value": v} for k, v in rows_kv]
    _validate_rows("d73grid-probe-config", rows)
    caveat = (
        "Probe spec of record for the arc's final experiment. The triple "
        "hedge is mandatory on every under-constraint sentence: under this "
        "forward model, at this redshift (z=0.3), on one simulated "
        "universe. The licensed claim boundary: minimizing this flux loss "
        "does not identify the true field — NEVER that the flux "
        "intrinsically cannot determine it, and the problem-intrinsic vs "
        "forward-model-induced split is explicitly NOT separated. Cost is "
        "stated as the banked budget shape (a pre-committed 30-GPU-hour "
        "cap, about a quarter used); no dollar figure is banked for this "
        "run and none may be printed. The grid is 192 cells per side — "
        "never any other mesh number. Single realization / fixed "
        "cosmology; z=0.3 scope-lock."
    )
    extra = {
        "recorded_wall_hours": wall_h,
        "internal_lineage": (
            "[D-73] (1d') option (a) four free grids (am-4 K6 discharge); G=192 "
            "angular-k ruling + init_xhi=0.2 pre-commit (am-5/am-7, tol-0 "
            "one-lever proof); Juno a30 job 221335 (commit c6f3aed), cap ≤30 "
            "A30-hr; wall from the run's MLflow meta start/end; F means from "
            "pf_sharpener.json"
        ),
    }
    artifact, sidecar = write_export(
        out_dir=Path(out_dir), filename="spec-probe-config.csv",
        fieldnames=["key", "value"],
        rows=rows, producing_fn="src.export.export_d73grid_probe_config",
        source_data_path=f"{sharpener_json} + {run_meta_yaml}",
        physics_id=1, caveat=caveat, extra_sidecar=extra)
    return {"artifact": str(artifact), "sidecar": str(sidecar), "n_rows": len(rows)}

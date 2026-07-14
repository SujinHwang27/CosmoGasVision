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
         "reference": "production baseline 0.4155",
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

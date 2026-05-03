"""Cosmological evaluation metrics for IGM tomography (Stage 2b).

Top-level [D-13] gating functions (the only names re-exported here):

- ``compute_xi_pearson`` — Pearson density-density cross-correlation xi(r),
  bounded in [-1, 1]; gating threshold xi(r=2 h^-1 Mpc) > 0.6.
- ``compute_p_flux``     — Hann-windowed 1D flux power P_F(k_||) on F.
- ``compute_flux_pdf``   — flux PDF p(F).
- ``ks_distance``        — KS distance on raw flux samples, F in [0.05, 0.95].

Diagnostic / legacy entrypoints remain importable from their submodules
(``compute_xi_covariance``, ``compute_xi_rho``, ``compute_xi_cross``,
``compute_PF_1d``, ``compute_Pdelta_3d``, ``compute_F_PDF``,
``ks_distance_pdf``) but are intentionally NOT re-exported at package
top level — they are not [D-13] gates.
"""

# [D-13] gating functions — canonical top-level API.
from .cross_corr import compute_xi_pearson
from .p_flux import compute_p_flux
from .flux_pdf import compute_flux_pdf, ks_distance

__all__ = [
    "compute_xi_pearson",
    "compute_p_flux",
    "compute_flux_pdf",
    "ks_distance",
]

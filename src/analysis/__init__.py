"""Cosmological evaluation metrics for IGM tomography (Stage 2b).

Spec-compliant top-level entrypoints (per [D-13] gating criteria):
- ``compute_p_flux``    — 1D flux power P_F(k_||) on transmitted flux F
- ``compute_xi_rho``    — density-density cross-correlation xi(r)
- ``compute_flux_pdf``  — flux PDF p(F)
- ``ks_distance``       — KS distance between two flux samples

Legacy entrypoints (used by ``stage2b_report.py`` and earlier tests):
- ``compute_PF_1d``, ``compute_Pdelta_3d``, ``compute_xi_cross``,
  ``compute_F_PDF``.
"""

# Spec-compliant API (preferred for new callers and the CVPR analysis pass).
from .p_flux import compute_p_flux
from .cross_corr import compute_xi_rho
from .flux_pdf import compute_flux_pdf, ks_distance

# Legacy API kept for stage2b_report.py + existing GRF tests.
from .flux_power import compute_PF_1d
from .density_power import compute_Pdelta_3d
from .cross_corr import compute_xi_cross
from .flux_pdf import compute_F_PDF, ks_distance_pdf

__all__ = [
    # spec-compliant
    "compute_p_flux",
    "compute_xi_rho",
    "compute_flux_pdf",
    "ks_distance",
    # legacy
    "compute_PF_1d",
    "compute_Pdelta_3d",
    "compute_xi_cross",
    "compute_F_PDF",
    "ks_distance_pdf",
]

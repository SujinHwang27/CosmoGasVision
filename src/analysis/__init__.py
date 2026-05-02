"""Cosmological evaluation metrics for IGM tomography (Stage 2b).

Five orthogonal probes per LEDGER §5:
- 1D flux power P_F(k_||)        -> flux_power.compute_PF_1d
- 3D density auto-power P_d(k)   -> density_power.compute_Pdelta_3d
- Density cross-correlation xi   -> cross_corr.compute_xi_cross
- Flux PDF + KS distance         -> flux_pdf.compute_F_PDF, ks_distance
- Stage 2b orchestrator          -> stage2b_report.generate_report
"""

from .flux_power import compute_PF_1d
from .density_power import compute_Pdelta_3d
from .cross_corr import compute_xi_cross
from .flux_pdf import compute_F_PDF, ks_distance

__all__ = [
    "compute_PF_1d",
    "compute_Pdelta_3d",
    "compute_xi_cross",
    "compute_F_PDF",
    "ks_distance",
]

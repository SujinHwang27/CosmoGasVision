"""Training-side modules (loss formulations, multi-task weighting wrappers).

This package collects training-graph utilities that are loss-formulation-axis
rather than model-axis. Sprint-L1 (direct P_F MSE loss test per design v2 at
`experiments/nerf/design/sprint_L1_direct_pf_loss.md`) introduces `p_flux_loss`
here: a differentiable torch reimplementation of the eval-side P_F estimator
(`src.analysis.p_flux.compute_p_flux`), a log-MSE loss over the [D-13]
inertial band, and a GradNorm (Chen+ 2018) multi-task weight wrapper.
"""

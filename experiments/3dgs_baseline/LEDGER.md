# Experiment Ledger: CosmoGasVision (3DGS Baseline Track)

> **⚠️ DEPRECATED REPOSITORY-WIDE — user directive 2026-05-11.** We are no longer pursuing the 3DGS track. This LEDGER is preserved in-repo for git history only. No new work, no MLflow runs, no DVC tracking, no agent dispatches. The CVPR paper baselines are TARDIS (Horowitz+2019) and Wiener filtering (Stark+2015); 3DGS is not cited or compared in the submission. Active track is NeRF at `experiments/nerf/LEDGER.md`.

This Ledger serves as the Single Source of Truth for the 3D Gaussian Splatting baseline, consolidating the project plan, technical decision log, and session history into a context-aware command center.

---

## 1. The Pulse (Progress & Roadmap)

| Stage | Focus Area | Status | Target Metric | Purpose |
|:--- |:--- |:--- |:--- |:--- |
| **Stage 1** | Seed Extraction (Peak-finding) | ✅ **DONE** | 16,915 seeds (P3, z0.3) | Initialize Gaussian positions. |
| **Stage 2** | 3D Gaussian Fitting | 🚀 **NEXT** | MSE Flux Reconstruction | Recov. 3D fields supervised by LOS. |
| **Stage 3** | Physics Model Classification | ⏳ **PENDING** | Model Accuracy > 80% | Discriminate AGN vs SN feedback. |

### ✅ Completed Milestones
- **2026-03-26**: Finalized the `stage1_seed_extraction.py` pipeline (now at `experiments/3dgs_baseline/pipeline.py`).
- **2026-03-26**: Generated initial point cloud (16,915 seeds) from Physics1, z=0.3 snapshot.
- **2026-03-27**: Established branch isolation via the unified **Command Center** workflow.

---

## 2. The Logic (Decision Log)

- **[D-01] Shift from COLMAP to Seed Extraction**: Given sightlines serve as "cameras," we extract Gaussian seeds from density peaks (`xaxis`, `yaxis`, `zaxis` + `posaxis`) instead of using standard SfM point clouds.
- **[D-02] Differentiable Ray Integrator**: Adopted ray-marching integration (NeRF-style) rather than tile-based rasterization to properly model the Lyman-alpha optical depth.
- **[D-03] Comparison target (TARDIS)**: Decided on **Horowitz et al. (2019) TARDIS** as the primary classical Wiener-filter comparison baseline.
- **[D-04] GPU Requirement**: Confirmed that while Stage 1 is CPU-feasible (1-4hr), Stage 2 (full snapshot fitting) is **infeasible on CPU** and requires T4/P100 cloud accelerators.

---

## 3. The Data (Lineage & Governance)

**Box Size**: 60,000 kpc/h (60 Mpc/h).  
**Hardware Profile**: Intel i7-1165G7, 64 GB RAM.

| Data Type | Primary Source | Implementation Logic |
|:--- |:--- |:--- |
| **LOS Tensors** | `tauH1_*.dat` | Supervision for flux reconstruction (Stage 2). |
| **Snapshots** | `snap_*.hdf5` | Ground-truth for PSNR/SSIM field validation. |
| **Halo Catalog** | `halolist_012.dat` | Context for seed extraction at high-density nodes. |

### Data Mapping (HDF5)
- **Physics 1** (No feedback): `planck1_60_768_z{redshift}`
- **Physics 2** (Stellar winds): `planck1_60_768_ps13_z{redshift}`
- **Physics 3** (Wind + AGN): `planck1_60_768_ps13agn_z{redshift}`
- **Physics 4** (Strong AGN): `planck1_60_768_ps13agn_strong_z{redshift}`

---

## 4. Implementation Strategy (Detailed)

### **Seed Extraction (Stage 1)**
- Each sightline is treated as a camera.
- Extracts density/H1 peaks to initialize Gaussian centers.
- Pilot run (P3, z=0.3) successfully generated 16,915 seeds in ~120 min.

### **Gaussian Fitting (Stage 2)**
- Parameterizes Gaussians with: Position, Covariance (Filament Anisotropy), Opacity (Density), Temp, H1frac, vpec.
- **Loss**: `‖τ_rendered − τ_H1_ground_truth‖`.

---

## 5. Session History & Handoff

### **Session Snapshot: March 27, 2026**
- **Refactor Completed**: Branch isolation established. Global documentation moved to DVC-tracked `docs/` bucket.
- **Architecture Isolated**: Overview formally defines **Track B (3DGS)** as the explicit baseline competing with the NeRF-based primary model.
- **Handoff**: 3DGS Ledger focus narrowed strictly to baseline metrics and seed extraction logs. Mermaid diagram removed here but preserved in the global overview and the NeRF track.

### **Immediate Next Steps**
1.  **Bulk Seed Extraction**: Run `pipeline.py` for redshifts z=0.1, 2.2, and 2.4 across all physics models.
2.  **Rasterizer Development**: Implement the 3D-Gaussian-to-1D-LOS projection kernel in PyTorch.
3.  **DVC Sync**: Track the generated `.npy` seed clouds to prevent bloated Git commits.

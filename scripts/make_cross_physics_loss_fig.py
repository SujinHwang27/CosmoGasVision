"""Cross-physics loss-vs-step convergence figure (paper Sec.3 figure).

Reads the per-step `loss_data` history from each of the four P{1,2,3,4}-T2
MLflow file-stores under cloud_runs/batch2-extracted/, overlays them on a
log-y loss-vs-step plot, and writes the figure to
paper_cvpr/figures/cross_physics_convergence.png.

The figure carries the PI-required headline: under the [D-24] log1p+cap+mask
loss, the four feedback variants converge symmetrically — the cross-physics
loss spread at step 1 (~factor 1.4) is preserved through the run with no
divergent trajectory, in sharp contrast to the raw-tau MSE regime which had
a step-1 cross-physics spread of ~10^10x (see micro-grid §3 numerics).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BATCH2 = ROOT / "cloud_runs" / "batch2-extracted"
OUT = ROOT / "paper_cvpr" / "figures" / "cross_physics_convergence.png"

CELLS = {
    "P1 (no fdbk)":  ("P1-N256-S0-1778118060-66d2cc", "C0"),
    "P2 (wind)":     ("P2-N256-S0-1778126385-2d0cc8", "C1"),
    "P3 (wind+AGN)": ("P3-N256-S0-1778128623-0e1ea0", "C2"),
    "P4 (str. AGN)": ("P4-N256-S0-1778136958-5295a6", "C3"),
}


def load_loss_data(cell_dir: str) -> tuple[np.ndarray, np.ndarray]:
    base = BATCH2 / cell_dir / "mlflow"
    # mlflow/<exp_id>/<run_id>/metrics/loss_data
    candidates = list(base.glob("*/*/metrics/loss_data"))
    candidates = [c for c in candidates if c.parent.parent.parent.name != "0"]
    if not candidates:
        raise FileNotFoundError(f"no loss_data metric file under {base}")
    arr = np.loadtxt(candidates[0])  # cols: timestamp_ms, value, step
    if arr.ndim == 1:
        arr = arr[None, :]
    steps = arr[:, 2].astype(int)
    values = arr[:, 1]
    order = np.argsort(steps)
    return steps[order], values[order]


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    final_vals = {}
    for label, (cell_dir, color) in CELLS.items():
        steps, values = load_loss_data(cell_dir)
        ax.semilogy(steps, values, lw=1.2, color=color, label=label)
        final_vals[label] = values[-1]
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\mathcal{L}_{\text{data}}$ (log1p+cap+mask)")
    ax.set_xlim(0, 25_000)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    print("Final loss_data per physics:")
    for label, v in final_vals.items():
        print(f"  {label:14s} = {v:.5f}")
    spread = max(final_vals.values()) - min(final_vals.values())
    rel = spread / np.mean(list(final_vals.values()))
    print(f"final spread: {spread:.5f} ({rel*100:.1f}% relative)")


if __name__ == "__main__":
    main()

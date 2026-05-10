"""Build the S5/S7 ablation overlay figure for paper_cvpr/sec/2_method.tex (per [D-38]).

Inputs: four cloud_runs/<TAG>/mlflow/ file-stores, each containing one MLflow run with
per-step metric history (loss_data, mean_flux_pred, ...).

Outputs:
  paper_cvpr/figures/s5s7_loss_curves.png
  paper_cvpr/figures/s5s7_loss_curves.pdf

Caption / framing is PI-dictated (see dispatch); this script renders the figure only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

REPO = Path(__file__).resolve().parent.parent

# (label, dir-tag, color, linestyle)
CELLS = [
    ("full ($\\tau_{\\max}{=}10$, mask on)",            "ablation-s5s7-full-1778343832-042792",           "k",  "-"),
    ("no-cap ($\\tau_{\\max}{=}10^9$, mask on)",        "ablation-s5s7-no-cap-1778354453-99549c",         "C3", "--"),
    ("no-mask ($\\tau_{\\max}{=}10$, mask off)",        "ablation-s5s7-no-mask-1778365090-b4a09a",        "C0", "--"),
    ("no-cap-no-mask",                                  "ablation-s5s7-no-cap-no-mask-1778375695-de69f0", "C4", ":"),
]

ANCHOR_BROKEN = 0.877   # the broken-anchor target this batch was trained against
ANCHOR_FIXED  = 0.979   # the corrected [D-34] target (publication-class)


def _client_for(tag: str) -> tuple[MlflowClient, str]:
    """Return (client, run_id) for the single run inside cloud_runs/<tag>/mlflow/."""
    store = (REPO / "cloud_runs" / tag / "mlflow").as_uri()
    client = MlflowClient(tracking_uri=store)
    for exp in client.search_experiments():
        if exp.experiment_id == "0":
            # default empty experiment; skip
            continue
        runs = client.search_runs(exp.experiment_id, max_results=10)
        for run in runs:
            return client, run.info.run_id
    # fall back: try experiment_id "0" too
    for exp in client.search_experiments():
        runs = client.search_runs(exp.experiment_id, max_results=10)
        for run in runs:
            return client, run.info.run_id
    raise RuntimeError(f"no run found in {tag}")


def get_history(tag: str, metric: str):
    client, run_id = _client_for(tag)
    history = client.get_metric_history(run_id, metric)
    if not history:
        raise RuntimeError(f"no history for {tag}/{metric}")
    history = sorted(history, key=lambda m: m.step)
    steps = [m.step for m in history]
    vals  = [m.value for m in history]
    return steps, vals


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    final_data = {}
    final_meanF = {}
    peak_data = {}

    for label, tag, color, ls in CELLS:
        steps_d, vals_d = get_history(tag, "loss_data")
        steps_f, vals_f = get_history(tag, "mean_flux_pred")

        axL.plot(steps_d, vals_d, color=color, linestyle=ls, linewidth=1.4, label=label)
        axR.plot(steps_f, vals_f, color=color, linestyle=ls, linewidth=1.4, label=label)

        final_data[tag] = vals_d[-1]
        final_meanF[tag] = vals_f[-1]
        peak_data[tag] = (steps_d[max(range(len(vals_d)), key=lambda i: vals_d[i])],
                          max(vals_d))

    # Reference horizontal lines on the right panel
    axR.axhline(ANCHOR_BROKEN, color="0.55", linestyle=":", linewidth=1.2,
                label=f"broken anchor $\\langle F\\rangle_{{\\rm obs}}{{=}}{ANCHOR_BROKEN}$")
    axR.axhline(ANCHOR_FIXED,  color="green", linestyle=":", linewidth=1.2,
                label=f"corrected [D-34] target ${ANCHOR_FIXED}$")

    # Use log-x to expose the early-warmup divergence in the data-loss panel
    axL.set_xscale("log")
    axL.set_xlim(left=1)
    axL.set_xlabel("training step")
    axL.set_ylabel(r"$\mathcal{L}_{\rm data}$ (data MSE)")
    axL.set_title(r"(a) data MSE $\mathcal{L}_{\rm data}$ vs. step")
    axL.legend(loc="upper right", framealpha=0.95)

    axR.set_xscale("log")
    axR.set_xlim(left=1)
    axR.set_ylim(0.86, 0.99)
    axR.set_xlabel("training step")
    axR.set_ylabel(r"$\langle\hat F\rangle$ (mean predicted flux)")
    axR.set_title(r"(b) mean predicted flux $\langle\hat F\rangle$ vs. step")
    axR.legend(loc="center right", framealpha=0.95)

    fig.tight_layout()

    out_dir = REPO / "paper_cvpr" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "s5s7_loss_curves.png"
    out_pdf = out_dir / "s5s7_loss_curves.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Cross-check printout
    print("=" * 72)
    print("S5/S7 ablation final-state numbers (cross-check vs. dispatch table)")
    print("=" * 72)
    print(f"{'cell':<40s} {'L_data_final':>14s} {'<F>_final':>12s} {'L_data_peak (step)':>22s}")
    for label, tag, _, _ in CELLS:
        ld = final_data[tag]
        mf = final_meanF[tag]
        ps, pv = peak_data[tag]
        print(f"{tag:<40s} {ld:>14.6f} {mf:>12.4f} {pv:>14.6f} (s={ps})")
    print("=" * 72)
    print(f"figure saved -> {out_png}")
    print(f"figure saved -> {out_pdf}")


if __name__ == "__main__":
    main()

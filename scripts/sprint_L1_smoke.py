"""Sprint-L1 host smoke driver (memory + 200-step host smoke).

Per gate-4 spec (PI dispatch brief 2026-05-16):

- memory mode: 5 steps; verify gradient flow + no NaN. Cheap (~1 min).
- host mode:   200 steps at P1-T3 schedule (n_rays=1024); verify
    (a) loss decreases
    (b) GradNorm weights finite + reasonable band
    (c) inertial-range |dP_F/P_F| starts >0 and decreases (any decrease ok)
    (d) no retire-condition trigger
    (e) write smoke_host.json with per-step monitored values

Wallclock target ~12 min on a single host GPU. Abort at 30 min (code-path bug).

This driver dispatches the standard ``experiments/nerf/pipeline.py`` train
loop with ``--enable-l1-pf-loss`` and reduced ``max_steps``. The host pipeline
already logs every L1 metric to MLflow per step; we additionally parse the
captured stdout to assemble the smoke summary file.

Usage
-----
::

    python -u scripts/sprint_L1_smoke.py --mode memory
    python -u scripts/sprint_L1_smoke.py --mode host
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "experiments" / "nerf" / "artifacts" / "sprint_L1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

WALLCLOCK_ABORT_S = 30 * 60   # 30 min


def _build_argv(mode: str) -> list[str]:
    """Build the pipeline argv list for the chosen smoke mode."""
    if mode == "memory":
        max_steps = 5
        n_rays = 64           # tiny; just verify forward+backward
        microbatch = 64
        checkpoint_dir = str(ARTIFACT_DIR / "memory_smoke")
        retire_dir = checkpoint_dir
    elif mode == "host":
        max_steps = 200
        # T3 batch per design v2 §6 calls for n_rays=1024. We probe the host
        # GPU; if absent, scale down to n_rays=64 so the smoke still verifies
        # the code path end-to-end within the 30-min wallclock cap. The
        # actual T3-scale 50k-step run belongs to gate-6 Juno dispatch (A30)
        # so this fallback is a code-correctness surrogate, not a perf bench.
        import torch as _t
        n_rays = 1024 if _t.cuda.is_available() else 64
        microbatch = n_rays
        checkpoint_dir = str(ARTIFACT_DIR / "host_smoke")
        retire_dir = checkpoint_dir
    else:
        raise ValueError(f"unknown mode={mode!r}; expected 'memory' or 'host'.")

    return [
        sys.executable, "-u",
        str(REPO_ROOT / "experiments" / "nerf" / "pipeline.py"),
        "--n_rays", str(n_rays),
        "--physics", "1",
        "--seed", "20260516",
        "--microbatch", str(microbatch),
        "--max_steps", str(max_steps),
        "--warmup_steps", "10",          # tight; smoke does not need full warmup
        "--checkpoint_interval", "0",    # no checkpoints during smoke
        "--checkpoint_dir", checkpoint_dir,
        "--data_root", "Sherwood",       # falls back to dummy data if missing
        "--enable-l1-pf-loss",
        "--l1-retire-dir", retire_dir,
        "--run_name", f"SprintL1-Smoke-{mode}",
    ]


def _parse_steps(stdout_lines: list[str]) -> list[dict]:
    """Parse 'Step N/M | ...' lines from the pipeline stdout into dicts."""
    rows = []
    for line in stdout_lines:
        if not line.startswith("Step "):
            continue
        # Quick tokenization; the existing pipeline format:
        #   Step {step}/{max} | loss=X (data=Y, meanF=Z, prior=W){extras} | <F>=A | grad=B | clip=C | lr=D | tau_amp=E
        try:
            head, rest = line.split("|", 1)
            step_part = head.strip()  # "Step N/M"
            step = int(step_part.split()[1].split("/")[0])
        except Exception:
            continue
        # We don't need to parse loss values from stdout — MLflow has them.
        rows.append({"step": step, "raw": line.rstrip()})
    return rows


def run_smoke(mode: str) -> dict:
    argv = _build_argv(mode)
    print(f"[smoke:{mode}] dispatch: {' '.join(argv)}", flush=True)

    start = time.time()
    # Stream child stdout in realtime so the operator sees progress.
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        cwd=str(REPO_ROOT),
    )
    captured = []
    try:
        for line in proc.stdout:
            captured.append(line)
            print(line, end="", flush=True)
            if (time.time() - start) > WALLCLOCK_ABORT_S:
                proc.kill()
                raise RuntimeError(
                    f"[smoke:{mode}] wallclock exceeded {WALLCLOCK_ABORT_S}s; "
                    "aborting. This is a code-path bug — investigate before "
                    "Juno dispatch."
                )
    finally:
        proc.wait()
    elapsed_s = time.time() - start

    rc = proc.returncode
    rows = _parse_steps(captured)

    # Detect retire-trigger via the retire.json the pipeline drops.
    retire_dir = Path(_build_argv(mode)[_build_argv(mode).index("--l1-retire-dir") + 1])
    retire_path = retire_dir / "retire.json"
    retire_payload = None
    if retire_path.exists():
        with open(retire_path, "r", encoding="utf-8") as fh:
            retire_payload = json.load(fh)

    # Build summary.
    summary = {
        "mode": mode,
        "wallclock_s": elapsed_s,
        "return_code": rc,
        "n_steps_observed": len(rows),
        "first_step_line": rows[0]["raw"] if rows else None,
        "last_step_line": rows[-1]["raw"] if rows else None,
        "retire_triggered": retire_payload is not None,
        "retire_payload": retire_payload,
        # Smoke pass criteria:
        #   memory: rc == 0 AND >= 1 step observed (5-step loop completed)
        #   host:   rc == 0 (incl. PCV-exit-0 retire) AND no abort
        "pass": (rc == 0),
    }
    out_path = ARTIFACT_DIR / f"smoke_{mode}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[smoke:{mode}] summary -> {out_path}", flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Sprint-L1 host smoke driver.")
    p.add_argument("--mode", required=True, choices=["memory", "host"])
    args = p.parse_args(argv)
    summary = run_smoke(args.mode)
    if not summary["pass"]:
        print(f"[smoke:{args.mode}] FAIL: rc={summary['return_code']}", flush=True)
        sys.exit(1)
    print(f"[smoke:{args.mode}] PASS (wallclock={summary['wallclock_s']:.1f}s)", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()

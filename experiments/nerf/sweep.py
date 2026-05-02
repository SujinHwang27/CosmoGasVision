"""Stage 2b ablation matrix sweep emitter.

Iterates the 4x4 cartesian product of ``--n_rays {16384, 1024, 256, 64}`` x
``--physics {1, 2, 3, 4}`` from [D-13]. A single seed is used here; the
matrix-of-seeds is Stage 2b+1 work.

By default, prints the local CLI invocation per matrix point (16 lines).
With ``--remote``, prints the equivalent ``scripts/sagemaker_stage2b_launch.py``
invocation. Print-only, matching the SageMaker launcher's safety posture.
This module never calls ``subprocess.run``.
"""

from __future__ import annotations

import argparse
import itertools
import sys


N_RAYS_AXIS = [16384, 1024, 256, 64]
PHYSICS_AXIS = [1, 2, 3, 4]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Emit Stage 2b 4x4 ablation matrix invocations (print-only).",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="Single seed for this sweep. Default 0.")
    p.add_argument("--max_steps", type=int, default=50000,
                   help="Forwarded to pipeline.py / launcher.")
    p.add_argument("--remote", action="store_true",
                   help="Emit SageMaker launcher invocations instead of "
                        "local pipeline.py invocations.")
    p.add_argument("--mlflow_uri", type=str,
                   default="http://127.0.0.1:5000",
                   help="MLflow tracking URI (only used in --remote mode).")
    p.add_argument("--dry-run", action="store_true",
                   help="Alias for default print-only behavior; provided "
                        "for explicit invocation in scripts and CI.")
    return p.parse_args(argv)


def build_local_cmd(physics: int, n_rays: int, seed: int,
                    max_steps: int) -> str:
    run_name = f"Stage2b-Ablation-P{physics}-N{n_rays}-S{seed}"
    return (
        f"python -u experiments/nerf/pipeline.py "
        f"--n_rays {n_rays} --physics {physics} --seed {seed} "
        f"--max_steps {max_steps} --run_name {run_name}"
    )


def build_remote_cmd(physics: int, n_rays: int, seed: int,
                     max_steps: int, mlflow_uri: str) -> str:
    return (
        f"python scripts/sagemaker_stage2b_launch.py "
        f"--n_rays {n_rays} --physics {physics} --seed {seed} "
        f"--max_steps {max_steps} --mlflow_uri {mlflow_uri}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    # The cartesian product is fixed by [D-13]. Order is (n_rays outer,
    # physics inner) so the printed matrix groups by sightline density —
    # that's the headline axis of the degradation curve.
    points = list(itertools.product(N_RAYS_AXIS, PHYSICS_AXIS))
    assert len(points) == 16, "Stage 2b matrix must be 16 points."

    for n_rays, physics in points:
        if args.remote:
            print(build_remote_cmd(
                physics, n_rays, args.seed, args.max_steps, args.mlflow_uri,
            ))
        else:
            print(build_local_cmd(
                physics, n_rays, args.seed, args.max_steps,
            ))

    print(f"\n# {len(points)} invocation(s) emitted "
          f"({'remote' if args.remote else 'local'} mode). "
          f"No commands were executed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

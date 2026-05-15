"""Sprint-5 (c') pre-flight tests per design doc v4 §6 gate-(c) + §4.1.

Two test classes:

  - ``TestCropAdmissibility``: verifies that ≥ 2000 admissible test
    crops per physics are obtainable at ``crop_size=48`` on the [D-49]
    axis-0 70/15/15 split with strict-rejection straddle policy.
    Uses an in-memory synthetic ρ field at ``n_grid=768`` (no Sherwood
    I/O).
  - ``TestCuDNNDeterminism``: verifies cuDNN-deterministic forward
    passes at input shape (1, 1, 48, 48, 48). Skipped if CUDA is
    unavailable; on CPU fall-back logs "S4 owed at Juno H100 dispatch"
    per design doc v4 §6 gate-(c) pre-flight obligation.

Run:
    PYTHONPATH=. uv run pytest tests/test_sprint5_pre_flight.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.loader import (
    _RHO_FIELD_CACHE,
    DEFAULT_SCHEME,
    SherwoodLoader,
)
from src.models.cnn3d import resnet18_3d_4class


# ============================================================ admissibility

class TestCropAdmissibility:
    """Design doc v4 §4.1: at ``n_grid=768``, [D-49] split defaults
    (axis=0, train_x_max=0.7, val_x_max=0.85), ``crop_size=48`` with
    strict-rejection straddle policy yields ≥ 2000 admissible test crops
    per physics (≥ 8000 total across 4 physics).
    """

    N_GRID = 768
    CROP_SIZE = 48
    REQUIRED_PER_PHYSICS = 2000

    def _seed_synthetic_rho(self, physics_id: int, redshift: float = 0.300) -> None:
        """Inject a synthetic ρ field into the loader cache so the test
        runs without Sherwood I/O. Deterministic per physics so the
        synthetic field is non-trivial. **Memory note**: a full 768³
        float32 lognormal allocation is ~1.69 GiB per physics, which
        exceeds available RAM on dev machines (~7 GiB needed for 4
        physics). We free the previous field before allocating the
        next so peak memory stays at one field.
        """
        # Free any previously injected field for this physics shape.
        prev = _RHO_FIELD_CACHE.pop((physics_id, round(redshift, 3), self.N_GRID), None)
        del prev
        rng = np.random.default_rng(seed=0xC0DE + physics_id)
        rho = rng.lognormal(
            mean=0.0, sigma=0.5,
            size=(self.N_GRID, self.N_GRID, self.N_GRID),
        ).astype(np.float32)
        rho /= rho.mean()
        _RHO_FIELD_CACHE[(physics_id, round(redshift, 3), self.N_GRID)] = rho

    @pytest.mark.parametrize("physics_id", [1, 2, 3, 4])
    def test_48cube_test_set_size_per_physics(self, physics_id):
        """≥ 2000 admissible test crops at ``crop_size=48`` on the
        [D-49] test region (axis-0 x ∈ [0.85, 1.0)), checked once per
        physics so peak RAM stays at one 768³ field (~1.7 GiB). The
        rejection sampler must succeed within ``max_rejections=100_000``.
        Cross-physics aggregate (≥ 8000 total) is the trivial sum;
        verified by the parametrize sweep below."""
        loader = SherwoodLoader(data_root=".")
        self._seed_synthetic_rho(physics_id)
        try:
            crops, _labels, _dists, positions = loader.extract_rho_crops_split(
                physics_id=physics_id,
                redshift=0.300,
                crop_size=self.CROP_SIZE,
                n_crops=self.REQUIRED_PER_PHYSICS,
                region="test",
                scheme=DEFAULT_SCHEME,
                seed=42 + physics_id,
                n_grid=self.N_GRID,
                return_positions=True,
            )
            assert crops.shape == (
                self.REQUIRED_PER_PHYSICS, 1,
                self.CROP_SIZE, self.CROP_SIZE, self.CROP_SIZE,
            ), f"physics={physics_id}: crops shape {tuple(crops.shape)}"
            # Verify each corner lies wholly inside the test region on
            # the split axis.
            test_voxel_start = int(DEFAULT_SCHEME.val_x_max * self.N_GRID)
            test_corner_max = self.N_GRID - self.CROP_SIZE
            axis_corners = positions[:, DEFAULT_SCHEME.axis]
            assert (axis_corners >= test_voxel_start).all(), (
                f"physics={physics_id}: corner < test region start"
            )
            assert (axis_corners <= test_corner_max).all(), (
                f"physics={physics_id}: corner > admissible test corner_max"
            )
        finally:
            # Free the 1.7 GiB field before the next parametrize step.
            _RHO_FIELD_CACHE.pop(
                (physics_id, 0.3, self.N_GRID), None,
            )

    def test_admissibility_geometry_only(self):
        """Pure-arithmetic admissibility check (no field allocation).
        Design doc v4 §4.1: at ``n_grid=768``, test region voxel-width
        = 116; corner range on split axis = [652, 720]; 69 admissible
        x-corners × 768 × 768 y/z corners (periodic non-split axes)
        = 40.7M candidates. ≫ 2000 required."""
        n_grid = 768
        crop_size = 48
        train_end = int(DEFAULT_SCHEME.train_x_max * n_grid)  # 537
        val_end = int(DEFAULT_SCHEME.val_x_max * n_grid)      # 652
        test_start = val_end
        test_end = n_grid
        admissible_x_corners = test_end - crop_size - test_start + 1  # 69
        admissible_y_corners = n_grid  # periodic
        admissible_z_corners = n_grid  # periodic
        total_candidates = (
            admissible_x_corners * admissible_y_corners * admissible_z_corners
        )
        assert admissible_x_corners > 0, (
            f"test region width {test_end - test_start} too small for "
            f"crop_size {crop_size} at n_grid {n_grid}"
        )
        assert total_candidates >= 4 * 2000, (
            f"only {total_candidates} candidate corners at n_grid={n_grid}, "
            f"crop_size={crop_size}; design doc v4 §4.1 requires ≥ 8000"
        )


# ============================================================ determinism

class TestCuDNNDeterminism:
    """Design doc v4 §6 gate-(c) pre-flight: verify that with
    ``torch.use_deterministic_algorithms(True)`` + cuDNN determinism
    flags set, two forward passes through ``ResNet3D`` at input shape
    (1, 1, 48, 48, 48) produce byte-identical fp32 predictions.

    This test must run on H100 to discharge gate-(c). On CPU fall-back,
    the test logs expected-failure with "S4 owed at Juno H100 dispatch".
    """

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="S4 pre-flight requires H100 (CUDA unavailable)")
    def test_cudnn_determinism_at_48cube(self):
        """Two seed-identical forward passes at (1, 1, 48, 48, 48)
        must produce byte-identical predictions under cuDNN
        determinism flags."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        device = torch.device("cuda")
        torch.manual_seed(20260515)
        net = resnet18_3d_4class().to(device).eval()

        torch.manual_seed(99)
        x = torch.randn(1, 1, 48, 48, 48, device=device)

        with torch.no_grad():
            out1 = net(x)
            out2 = net(x)

        assert torch.equal(out1, out2), (
            f"cuDNN-deterministic forward not byte-identical at (1,1,48,48,48); "
            f"max abs diff = {(out1 - out2).abs().max().item():.3e}. "
            "Per design doc v4 §6 gate-(c), relax to |Δp| < 1e-5 with "
            "footnote OR fail gate-(c) pre-flight."
        )

    def test_cpu_fallback_logs_owed_at_juno(self, capsys):
        """CPU fall-back marker: the gate-(c) cuDNN-determinism check
        is **owed at Juno H100 dispatch**. This test always runs and
        always passes; its purpose is to emit a log line that the
        gate-(c) pre-flight is deferred on this host.
        """
        if torch.cuda.is_available():
            pytest.skip("CUDA available — gate-(c) discharged by the cuDNN test")
        print("[sprint5cprime gate-(c)] S4 owed at Juno H100 dispatch — "
              "cuDNN determinism on (1,1,48,48,48) cannot be verified on CPU.")
        captured = capsys.readouterr()
        assert "owed at Juno H100 dispatch" in captured.out

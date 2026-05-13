# Sprint-2 design — held-out region spatial split for `SherwoodLoader`

**Status**: 🛠 Draft → implementation, 2026-05-12, main-thread (PI brief + data-engineer brief, both delegated to main thread per permission constraint resolution).
**Predecessor**: Sprint-1 [D-48] ρ-field disk cache (HEAD `923458f` + `14d51c2`).
**Blocking**: [D-46] Tier-1 dispatch (per [D-46] Addendum 1); [D-47] option-C step-2 (reconstructed-baseline classifier evaluation on held-out crops).
**Downstream**: Sprint-4 (3D ResNet truth-baseline) consumes this for [D-47] option-C step-1 — truth-baseline trains/evaluates on the same partition the eventual reconstructed-baseline evaluates on, so the gap is apples-to-apples.

---

## 1. Why a held-out region exists at all

Per **[D-12]** (cross-physics protocol): the Stage 3 classifier must not see the physics label leaked via the reconstruction. [D-12] originally rejected the conditional-`physics_id`-embedding alternative for this reason; **[D-46]** walks part of that back by pooling 4 physics into one MLP with a `physics_id` embedding, on the data-axis hypothesis that high-τ peaks quadruple per gradient step.

[D-46] Addendum 1 + the defense-panel verdict (METHODS-2 attack) made the held-out-region requirement explicit: the reconstruction MLP trains on `region = train` sightlines only; Stage 3's feedback classifier evaluates on `region ∈ {val, test}` crops the reconstruction never saw. This is the only way [D-46]'s data-axis intervention survives [D-12]'s anti-leakage rule.

Per **[D-47]** option-C hybrid Stage 3 framing, the truth-baseline (3D ResNet on ground-truth ρ crops) runs on the same partition, so the reconstructed-baseline's accuracy is reported against an honest empirical ceiling.

## 2. Scheme — slab along the x-axis

We adopt a **contiguous slab geometry** along the lowest-index axis. The 60 Mpc/h Sherwood box has periodic BC on every axis; partitioning along one axis only (rather than 3D corner-quadrants) gives:

- a maximally-simple boundary topology (one slab plane per partition edge);
- straightforward `distance_to_train_region(coord)` (periodic 1D distance, not max-over-axes);
- a clear "far from train" tail at $r ≈ 0.5$ \* (held-out fraction), reachable for the [D-47] $r_{25}/r_{50}/r_{75}$ pre-registered estimator.

**Default scheme** (`HeldoutSplitScheme()`):
| Region | Axis-0 (x) fraction | Voxel index range at `n_grid=768` |
|--------|---------------------|-----------------------------------|
| train  | `[0, 0.7)`          | `[0, 538)`                        |
| val    | `[0.7, 0.85)`       | `[538, 653)`                      |
| test   | `[0.85, 1.0)`       | `[653, 768)`                      |

Other axes (y, z) are unconstrained — every region spans the full box on those axes. This means:
- train region volume = 70% of box;
- val region volume = 15% of box;
- test region volume = 15% of box;
- total held-out (val + test) volume = 30% of box.

70/15/15 chosen to:
- give the reconstruction MLP a generous training volume;
- leave enough held-out for two non-trivial classifier evaluations (val for hyperparameter selection, test for the headline number);
- keep the smallest region (val or test) ≥ 10% so n_crops budgets are not starved.

**Knobs**: `train_x_max` and `val_x_max` are tunable; `axis ∈ {0, 1, 2}` allows ablating axis choice. A future ablation can swap to `axis=1` or `axis=2` to falsify "the split direction does not change the headline."

**Rejected alternatives** (with reasons):

- *3D corner quadrant* (e.g. train = $[0, 0.7]^3$, held-out = complement): simpler set-theoretically, but train volume = 34% — wastes 66% of box on held-out, starving the reconstruction MLP. Also breaks `distance_to_train` smoothness (corners of the held-out region are far in 3D from train, but the L∞-distance is not strictly monotonic in physical distance).
- *Axis-bisection (50/50)*: cleaner symmetry but halves the reconstruction's training data, which would re-open [D-12]'s "is the reconstruction even seeing the modes that matter?" critique.
- *Three-axis slab union* (held-out = ($x > 0.7$) ∨ ($y > 0.7$) ∨ ($z > 0.7$)): held-out becomes a topologically weird shell. Not adopted.

## 3. Distance metric — periodic 1D along the split axis

`distance_to_train_region(coord_normalized, scheme)` returns the **shortest periodic 1D distance** from `coord_normalized[scheme.axis]` to the train interval `[0, scheme.train_x_max]`:

- If `coord_normalized[axis] ∈ [0, train_x_max]`: distance = 0.
- Otherwise (coord ∈ held-out): distance = `min(coord - train_x_max, 1 - coord)` — i.e. closer of (a) reach back across the lower train boundary, (b) reach forward across the periodic seam back to `x = 0`.

For the default `train_x_max = 0.7`:
- Held-out boundary at $x = 0.7$ → distance = 0 (at the seam).
- Periodic seam at $x = 1.0 \equiv 0.0$ → distance = 0 (the other train boundary, via wraparound).
- Midpoint of held-out at $x = 0.85$ → distance = `min(0.15, 0.15) = 0.15` (maximal $r$).

So distance $r \in [0, 0.15]$ for the default scheme; the [D-47] $r_{25}/r_{50}/r_{75}$ percentiles are computed over the empirical distribution of held-out crop centers' distances, NOT over $[0, 0.15]$ analytically — the empirical distribution is what defines the conditional-accuracy estimator.

Non-split axes do not contribute (y, z displacements are always 0 away from train).

## 4. Region mask — discrete classifier

`region_mask(coord_normalized, scheme)` returns `"train"`, `"val"`, or `"test"` by axis-0 bucketing:
- `coord[axis] < train_x_max` → `"train"`
- `train_x_max ≤ coord[axis] < val_x_max` → `"val"`
- `val_x_max ≤ coord[axis] < 1.0` → `"test"`

(Right-open intervals everywhere; coord must be in `[0, 1)` per the project's coord convention.)

## 5. Crop straddle policy — STRICT (reject straddles)

A crop has voxel index range `[corner_axis, corner_axis + crop_size) (mod n_grid)` per axis. The crop is **wholly in region R** iff:
- For the split axis (`scheme.axis`): the full unwrapped index range `[corner_axis, corner_axis + crop_size)` is contained in R's voxel range AND **does not wrap modulo n_grid**.
  - "does not wrap" is critical: if `corner + crop_size > n_grid` the crop wraps from box edge back to $x = 0$ (train), so a "test" assignment is leaky.
- For non-split axes (y, z): no constraint (wraparound is fine since y and z don't split).

Crops that straddle the region boundary OR wrap around the periodic seam are **rejected**, not relabelled.

**Why strict over relaxed**: The defense-panel **COSMO-1** attack specifically named seam leakage as the killer-class concern. Relabelling straddling crops to "boundary" or "ambiguous" would still permit the classifier to learn from seam features that the reconstruction MLP also saw — re-opening the [D-12] attack vector. Strict rejection trades a sampling-efficiency cost (more rejected draws near the seam) for a guarantee that the boundary is sharp.

**Sampling efficiency at default scheme**:
- Acceptance rate for `region="train"` at `crop_size=32, n_grid=768`: roughly $(\text{train\_x\_max} - \text{crop\_size}/\text{n\_grid}) = (0.7 - 0.042) = 0.658$ along x-axis × 1.0 × 1.0 on y, z = **65.8%**.
- For `region="val"`: $(\text{val\_x\_max} - \text{train\_x\_max} - \text{crop\_size}/\text{n\_grid}) = (0.15 - 0.042) = 0.108$ → **10.8%** along x.
- For `region="test"`: same as val → **10.8%** along x.

For `crop_size = 64, n_grid = 768`: val/test acceptance drops to $(0.15 - 0.083) ≈ 6.7\%$ — still workable; ~15 corner draws per accepted crop. For `crop_size = 128` the val/test rate is $(0.15 - 0.167) < 0$: impossible. We document this constraint and refuse to sample when `crop_size / n_grid > min(region_fraction)`.

`max_rejections = 100_000` cap on the rejection loop; raise `RuntimeError` if hit (signals a pathological scheme/crop_size combo).

## 6. Determinism contract

- Same `(seed, scheme, region, crop_size, n_crops, physics_id, redshift, n_grid)` → byte-identical output.
- The rejection loop uses **one** `np.random.default_rng(seed)`; rejected draws advance the RNG state, so adding a 7th crop to an n_crops=6 batch returns the same 6 crops + 1 new (it does not reorder).
- `np.random.default_rng(seed)` is created per call (no global RNG mutation), matching `extract_rho_crops`.

## 7. API surface

```python
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
import torch

Region = Literal["train", "val", "test", "heldout"]

@dataclass(frozen=True)
class HeldoutSplitScheme:
    """Geometry of the train / val / test partition of the periodic box.

    The split is a contiguous slab along axis 0 (x) by default. train + val + test
    must cover [0, 1] exactly; train is always [0, train_x_max].
    """
    train_x_max: float = 0.7
    val_x_max: float = 0.85
    axis: int = 0  # 0=x, 1=y, 2=z

DEFAULT_SCHEME = HeldoutSplitScheme()


def region_mask(
    coord_normalized: np.ndarray | float,
    scheme: HeldoutSplitScheme = DEFAULT_SCHEME,
) -> str | np.ndarray:
    """Discrete region label for a single coord (shape (3,)) or batch (shape (N, 3)).

    Returns a Python str for the single-coord case, or an np.ndarray of dtype
    object/str for the batched case.
    """
    ...

def distance_to_train_region(
    coord_normalized: np.ndarray | float,
    scheme: HeldoutSplitScheme = DEFAULT_SCHEME,
) -> float | np.ndarray:
    """Shortest periodic distance along scheme.axis from the query coord to
    the train interval [0, scheme.train_x_max].

    0 inside train; range (0, 0.5 * (1 - scheme.train_x_max)] outside train.
    """
    ...

# On SherwoodLoader:
def extract_rho_crops_split(
    self,
    physics_id: int,
    redshift: float,
    crop_size: int,
    n_crops: int,
    region: Region,
    scheme: HeldoutSplitScheme = DEFAULT_SCHEME,
    seed: int = 42,
    n_grid: int = 768,
    max_rejections: int = 100_000,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Like extract_rho_crops, but only returns crops whose voxel support
    is wholly within `region`. Straddling crops are rejected (not relabelled).

    Returns
    -------
    crops  : (n_crops, 1, crop_size, crop_size, crop_size) float32 overdensity
    labels : (n_crops,) long, physics_id (broadcast)
    distances : (n_crops,) float32, distance_to_train_region per crop CENTER
                in normalized box coords. Always 0 for region="train"; > 0
                for region ∈ {val, test, heldout}.
    """
    ...
```

`region="heldout"` is a convenience: it accepts crops that are wholly in val OR wholly in test (i.e., wholly outside train), without distinguishing which.

## 8. What the [D-47] pre-registered estimator expects from this

[D-47] specifies:
- $\hat A(r)$ at $r = \{r_{25}, r_{50}, r_{75}\}$: conditional accuracy at percentile distances of held-out crops to the train region. **This skill returns the per-crop distances; downstream code (sprint-4 truth-baseline trainer) computes the percentiles and the $\hat A(r)$ estimator.**
- Moving-block bootstrap on crop spatial position: the spatial position is the crop corner (returned implicitly via the deterministic `seed`); the block length is empirical and outside this sprint.
- Negative-result trigger if $\hat A(r_{25}) - \hat A(r_{75}) > 0.10$: downstream interpretation, not a loader concern.

What this sprint enables: **the right denominator** for the estimator. Crops cannot leak from train into the eval set.

## 9. Test plan — `tests/test_heldout_split.py`

Six tests, all at `n_grid = 32` or `64` for fast CI (the disk cache from sprint-1 makes a real CIC at `n_grid=64` a one-time ~60 s cost, paid once and amortised across the file).

1. **`test_region_mask_at_axis_interior_and_boundary`**: `region_mask` returns correct label at `x = 0.5` (train), `x = 0.78` (val), `x = 0.95` (test), and at boundaries `x = 0.7` (val, right-open), `x = 0.85` (test, right-open). y, z values are irrelevant.
2. **`test_distance_to_train_region_zero_inside_train_positive_outside`**: distance is exactly 0 for any `x ∈ [0, 0.7)`, and strictly positive for `x ∈ (0.7, 1.0)`. At `x = 0.85` (midpoint of held-out), distance = 0.15.
3. **`test_distance_to_train_region_periodic`**: distance from `x = 0.99` is `0.01` (close via wraparound to `x = 0`), NOT `0.29` (the non-periodic distance to `x = 0.7`).
4. **`test_extract_split_returns_only_crops_in_region`**: for each `region ∈ {train, val, test}`, sample `n_crops = 16` at `crop_size = 4, n_grid = 32`, then assert that for every returned crop, the per-axis voxel index range lies wholly within the region's voxel range AND does not wrap.
5. **`test_straddling_crops_rejected_not_relabelled`**: at a crop_size that would force ≥ 1 straddle if sampled greedily (e.g., `crop_size=4, n_grid=32, scheme.train_x_max=0.7` → train voxel range [0, 22), straddling corners at indices 19, 20, 21 produce crops that extend into [val] — these must NOT be returned with `region="train"`).
6. **`test_determinism_under_seed_and_scheme`**: two calls with identical args produce byte-identical crops + distances. Two calls with the same args but different `seed` produce different corners.

In addition: at the end of the test file, **assert pre-existing tests still pass** is a CI concern, not an in-file assertion. The runner will run all three test files (`test_rho_crop_extraction.py`, `test_rho_disk_cache.py`, `test_heldout_split.py`) and verify green.

## 10. Carry-forward / out-of-scope

- **The 3D ResNet truth-baseline itself** (sprint-4) consumes this output. Architecture, training, evaluation are out of scope for sprint-2.
- **Conditional-accuracy $\hat A(r)$ estimator implementation**: consumes `distances` returned by `extract_rho_crops_split`; separate task.
- **Moving-block bootstrap**: separate task (sprint-4 or later).
- **Multi-z held-out sets**: this sprint pins `z = 0.3` per [D-46]/[D-47] z=0.3 scope.
- **Held-out region as a 3D shape**: out of scope; the slab is what [D-47] specifies.
- **Stale `*.tmp.npy` sweep**: PI minor ask from sprint-1, blocked on `Sherwood/**` deny rule. Carry-forward.

## 11. References

- [D-12] — anti-leakage rule (this sprint operationalizes the mandatory mitigation).
- [D-15] — Stage 3 85% bar (this sprint defines the legitimate evaluation set).
- [D-46] / [D-46] Addendum 1 — physics_id embedding + held-out region requirement.
- [D-47] — option-C hybrid Stage 3 framing + pre-registered $\hat A(r)$ estimator.
- [D-48] — disk cache (enables fast CI iteration on this sprint).
- Defense panel verdict 2026-05-11 — COSMO-1, METHODS-2 attacks (strict rejection is the response).

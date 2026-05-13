"""[D-50] gate (a) driver — cold + warm CIC at n_grid=768 with peak-RSS
tracking. Equivalent to ``test_cache_hit_timing_gate`` in
``tests/test_rho_disk_cache.py`` but augmented with an external psutil
poll so we can report achieved peak resident memory (required to satisfy
the LEDGER \xa73 [D-50] gate (a) "host-budget" wording, which the slow
pytest gate does not measure).

Cold + warm both target physics_id=1, z=0.300, n_grid=768 with the same
crop knobs as the slow pytest gate (crop_size=8, n_crops=2, seed=42).

Cache directory is pinned to D:\\tmp\\cosmogasvision_d50\\cache_n768 so we
do not pollute the canonical Sherwood/.rho_field_cache/ during gate
evaluation.

Exit codes:
    0  cold and warm both within gate budgets
    1  cold exceeded 15-min ceiling
    2  warm exceeded 15-s budget
    3  unexpected exception during the run
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

CACHE_DIR = Path(r"D:\tmp\cosmogasvision_d50\cache_n768")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["COSMOGAS_RHO_CACHE_DIR"] = str(CACHE_DIR)

COLD_BUDGET_S = 15 * 60     # 15 min ceiling for the cold CIC deposition
WARM_BUDGET_S = 15.0        # 15 s budget for the warm disk-cache hit
SAMPLE_INTERVAL_S = 0.5

import psutil  # noqa: E402

proc = psutil.Process()
_peak = {"rss_bytes": 0}
_stop = threading.Event()


def _rss_sampler() -> None:
    while not _stop.is_set():
        try:
            rss = proc.memory_info().rss
        except psutil.Error:
            break
        if rss > _peak["rss_bytes"]:
            _peak["rss_bytes"] = rss
        time.sleep(SAMPLE_INTERVAL_S)


def _start_sampler() -> threading.Thread:
    _peak["rss_bytes"] = proc.memory_info().rss
    _stop.clear()
    t = threading.Thread(target=_rss_sampler, daemon=True)
    t.start()
    return t


def _stop_sampler(t: threading.Thread) -> int:
    _stop.set()
    t.join(timeout=2.0)
    return _peak["rss_bytes"]


def _gib(b: int) -> float:
    return b / (1024 ** 3)


def main() -> int:
    print(f"[D-50] cache dir       : {CACHE_DIR}")
    print(f"[D-50] cold budget     : {COLD_BUDGET_S:.0f}s")
    print(f"[D-50] warm budget     : {WARM_BUDGET_S:.0f}s")
    print(f"[D-50] sample interval : {SAMPLE_INTERVAL_S}s")

    print("[D-50] importing SherwoodLoader ...", flush=True)
    from src.data.loader import SherwoodLoader, _RHO_FIELD_CACHE

    repo_root = Path(__file__).resolve().parent.parent
    loader = SherwoodLoader(data_root=str(repo_root / "Sherwood"))

    # If a prior gate run produced a valid cache entry, delete it so we
    # exercise the cold-CIC path. This is the only place we delete from
    # CACHE_DIR; the canonical Sherwood/.rho_field_cache/ is left alone.
    for f in CACHE_DIR.glob("rho_field_p1_z0.300_n768.*"):
        f.unlink()
        print(f"[D-50] removed prior cache file: {f.name}")

    _RHO_FIELD_CACHE.clear()

    # --------------------------------------------------------- cold
    print("[D-50] cold call (CIC deposition at n_grid=768) ...", flush=True)
    sampler = _start_sampler()
    t0 = time.perf_counter()
    try:
        _ = loader.extract_rho_crops(
            physics_id=1,
            redshift=0.300,
            crop_size=8,
            n_crops=2,
            seed=42,
            n_grid=768,
        )
    except Exception:
        _stop_sampler(sampler)
        traceback.print_exc()
        return 3
    cold_s = time.perf_counter() - t0
    cold_peak = _stop_sampler(sampler)

    print(f"[D-50] cold seconds    : {cold_s:.2f}s")
    print(f"[D-50] cold peak RSS   : {_gib(cold_peak):.3f} GiB ({cold_peak} B)")

    if cold_s >= COLD_BUDGET_S:
        print(
            f"[D-50] FAIL: cold {cold_s:.1f}s exceeded ceiling {COLD_BUDGET_S}s",
            file=sys.stderr,
        )
        return 1

    # --------------------------------------------------------- warm
    _RHO_FIELD_CACHE.clear()

    print("[D-50] warm call (disk-cache hit at n_grid=768) ...", flush=True)
    sampler = _start_sampler()
    t0 = time.perf_counter()
    try:
        _ = loader.extract_rho_crops(
            physics_id=1,
            redshift=0.300,
            crop_size=8,
            n_crops=2,
            seed=42,
            n_grid=768,
        )
    except Exception:
        _stop_sampler(sampler)
        traceback.print_exc()
        return 3
    warm_s = time.perf_counter() - t0
    warm_peak = _stop_sampler(sampler)

    print(f"[D-50] warm seconds    : {warm_s:.2f}s")
    print(f"[D-50] warm peak RSS   : {_gib(warm_peak):.3f} GiB ({warm_peak} B)")

    if warm_s > WARM_BUDGET_S:
        print(
            f"[D-50] FAIL: warm {warm_s:.2f}s exceeded budget {WARM_BUDGET_S}s",
            file=sys.stderr,
        )
        return 2

    print(
        f"[D-50] PASS - cold={cold_s:.1f}s peak={_gib(cold_peak):.2f}GiB | "
        f"warm={warm_s:.2f}s peak={_gib(warm_peak):.2f}GiB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

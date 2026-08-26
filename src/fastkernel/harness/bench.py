"""Benchmark protocol (identical for reference and candidate):

warm-up runs -> busy loop for `ramp_seconds` so GPU clocks come up -> `repeats` timed runs, each
CUDA-synchronised -> median (primary), min, mean, p90, std. Peak VRAM via max_memory_allocated.
"""
from __future__ import annotations

import statistics
import time
from typing import Any


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_callable(fn, *, warmup: int = 5, repeats: int = 50, ramp_seconds: float = 1.0) -> dict[str, Any]:
    import torch
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        _sync()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ramp_seconds:
            fn()
        _sync()
        samples: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            _sync()
            samples.append((time.perf_counter() - t0) * 1e3)
    samples_sorted = sorted(samples)
    return {
        "median_ms": statistics.median(samples), "min_ms": samples_sorted[0], "max_ms": samples_sorted[-1],
        "mean_ms": statistics.fmean(samples), "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "p90_ms": samples_sorted[min(len(samples) - 1, int(0.9 * len(samples)))], "repeats": repeats, "samples_ms": samples,
    }


def peak_memory_mb() -> float | None:
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e6


def reset_peak_memory() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

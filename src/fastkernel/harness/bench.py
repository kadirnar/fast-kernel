"""Benchmark protocol (identical for reference and candidate).

Absolute timing: warm-up runs -> busy loop for `ramp_seconds` so GPU clocks come up -> `repeats`
timed runs, each CUDA-synchronised -> median (primary), min, mean, p90, std.

Absolute milliseconds are only comparable *within one process*. Clock/thermal state, driver state
and whatever else shares the GPU drift between sessions, so comparing a candidate measured now
against an incumbent measured an hour ago charges that drift to the candidate. `compare_callables`
therefore times two callables **interleaved, in one process**, and reports the paired ratio: drift
moves both members of a pair the same way and cancels out. That ratio -- not a raw millisecond
count -- is what the accept/reject decision should be built on.
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_once(fn: Callable[[], Any], inner: int) -> float:
    """Milliseconds for one sample: `inner` back-to-back calls, one sync, divided by `inner`.

    Batching amortises the sync and the CPU wake-up that follows it over `inner` calls, which is
    the dominant per-sample jitter for workloads of a few milliseconds.
    """
    t0 = time.perf_counter()
    for _ in range(inner):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1e3 / inner


def time_callable(fn, *, warmup: int = 5, repeats: int = 50, ramp_seconds: float = 1.0, inner: int = 1) -> dict[str, Any]:
    import torch
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        _sync()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ramp_seconds:
            fn()
        _sync()
        samples: list[float] = [_time_once(fn, inner) for _ in range(repeats)]
    samples_sorted = sorted(samples)
    return {
        "median_ms": statistics.median(samples), "min_ms": samples_sorted[0], "max_ms": samples_sorted[-1],
        "mean_ms": statistics.fmean(samples), "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "p90_ms": samples_sorted[min(len(samples) - 1, int(0.9 * len(samples)))], "repeats": repeats,
        "inner": inner, "samples_ms": samples,
    }


def _median_uncertainty(values: list[float]) -> float:
    """Half-width of a ~95 % interval for the median (asymptotic: 1.2533 * sigma / sqrt(n))."""
    n = len(values)
    if n < 2:
        return float("inf")
    return 2.0 * 1.2533 * statistics.pstdev(values) / math.sqrt(n)


def auto_inner(fn, *, target_ms: float = 5.0, cap: int = 32) -> int:
    """Calls per timed sample so that one sample lasts about `target_ms`.

    A single 2 ms call timed with perf_counter is mostly CPU wake-up jitter; batching a few calls
    per sample pushes that jitter below the signal without changing what is measured.
    """
    import torch
    with torch.inference_mode():
        for _ in range(2):
            fn()
        _sync()
        probe = [_time_once(fn, 1) for _ in range(5)]
    med = statistics.median(probe)
    if med <= 0:
        return 1
    return max(1, min(cap, int(math.ceil(target_ms / med))))


def compare_callables(fn_a, fn_b, *, warmup: int = 5, pairs: int = 20, ramp_seconds: float = 1.0,
                      inner: int | None = None, target_ms: float = 5.0) -> dict[str, Any]:
    """Time `fn_a` and `fn_b` interleaved and return the paired ratio a/b (>1 means b is faster).

    Each pair runs both callables back to back, and the order alternates (a,b then b,a) so that any
    advantage of running second -- warmer caches, a clock that is still ramping -- cancels across
    pairs. `ratio_uncertainty` is the half-width of a ~95 % interval on the median ratio; a
    difference smaller than that is not measurable with this many pairs, and is the honest noise
    floor for an accept/reject decision.

    `inner` is chosen per callable by default. An optimised candidate is often several times faster
    than the reference it is compared against, and the shorter call carries proportionally more of
    the fixed per-sample jitter; batching each side to about `target_ms` puts them on equal footing
    instead of letting the faster one dominate the uncertainty.
    """
    import torch
    inner_a = inner if isinstance(inner, int) else auto_inner(fn_a, target_ms=target_ms)
    inner_b = inner if isinstance(inner, int) else auto_inner(fn_b, target_ms=target_ms)
    with torch.inference_mode():
        for _ in range(warmup):
            fn_a()
            fn_b()
        _sync()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ramp_seconds:
            fn_a()
            fn_b()
        _sync()
        a_samples: list[float] = []
        b_samples: list[float] = []
        ratios: list[float] = []
        for i in range(pairs):
            if i % 2 == 0:
                ta = _time_once(fn_a, inner_a)
                tb = _time_once(fn_b, inner_b)
            else:
                tb = _time_once(fn_b, inner_b)
                ta = _time_once(fn_a, inner_a)
            a_samples.append(ta)
            b_samples.append(tb)
            if tb > 0:
                ratios.append(ta / tb)
    ratio = statistics.median(ratios) if ratios else float("nan")
    unc = _median_uncertainty(ratios) if ratios else float("inf")
    return {
        "ratio": ratio,                                  # a / b  (>1 => b is faster than a)
        "ratio_uncertainty": unc / ratio if ratio else float("inf"),   # relative, ~95 % half-width
        "ratio_spread": (statistics.pstdev(ratios) / ratio) if len(ratios) > 1 and ratio else 0.0,
        "a_median_ms": statistics.median(a_samples) if a_samples else None,
        "b_median_ms": statistics.median(b_samples) if b_samples else None,
        "pairs": len(ratios), "inner_a": inner_a, "inner_b": inner_b, "ratios": ratios,
    }


def self_noise(fn, *, warmup: int = 5, pairs: int = 20, ramp_seconds: float = 1.0, inner: int = 1) -> dict[str, Any]:
    """Measure the harness against itself: the same callable on both sides of `compare_callables`.

    The true ratio is exactly 1, so whatever the comparison reports instead is measurement error.
    The noise floor is the larger of the residual bias and the median's uncertainty -- an
    improvement below it cannot be told apart from noise on this machine.
    """
    cmp = compare_callables(fn, fn, warmup=warmup, pairs=pairs, ramp_seconds=ramp_seconds, inner=inner)
    bias = abs(cmp["ratio"] - 1.0) if cmp["ratio"] == cmp["ratio"] else float("inf")
    cmp["noise"] = max(bias, cmp["ratio_uncertainty"])
    cmp["bias"] = bias
    return cmp


def peak_memory_mb() -> float | None:
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e6


def reset_peak_memory() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

from __future__ import annotations

import importlib
import os
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

from ..util import now_iso


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def device_capabilities(microbench: bool = True) -> dict[str, Any]:
    """GPU facts + tiny measured roofline numbers (bandwidth, TFLOPS, launch latency)."""
    info: dict[str, Any] = {
        "probed_at": now_iso(), "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": None, "cuda": None, "cudnn": None, "device": None, "name": None,
    }
    try:
        import torch
    except ImportError as exc:
        info["error"] = f"torch not importable: {exc}"
        return info
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    info["cudnn"] = _safe(lambda: torch.backends.cudnn.version())
    if not torch.cuda.is_available():
        info["device"] = "cpu"
        info["name"] = platform.processor() or "cpu"
        return info
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    info.update({
        "device": "cuda", "name": props.name, "compute_capability": f"{cap[0]}.{cap[1]}", "sm_count": props.multi_processor_count,
        "total_memory_gb": round(props.total_memory / 1e9, 2), "device_count": torch.cuda.device_count(),
        "driver": _safe(lambda: torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None),
        "l2_cache_mb": _safe(lambda: round(props.L2_cache_size / 1e6, 1)),
        "max_shared_memory_per_block_kb": _safe(lambda: round(props.shared_memory_per_block_optin / 1024)),
        "regs_per_multiprocessor": _safe(lambda: props.regs_per_multiprocessor),
    })
    if not microbench:
        return info
    try:
        torch.cuda.synchronize()
        # bandwidth: device-to-device copy of 256 MB
        n = 64 * 1024 * 1024
        a = torch.empty(n, dtype=torch.float32, device="cuda")
        b = torch.empty_like(a)
        for _ in range(3):
            b.copy_(a)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            b.copy_(a)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        info["measured_bandwidth_gbs"] = round(2 * a.numel() * 4 / dt / 1e9, 1)
        del a, b
        # bf16 tensor-core matmul
        m = 4096
        x = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            x @ y
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            x @ y
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        info["measured_bf16_tflops"] = round(2 * m ** 3 / dt / 1e12, 1)
        xf, yf = x.float(), y.float()
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for _ in range(2):
                xf @ yf
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                xf @ yf
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 5
            info["measured_fp32_tflops"] = round(2 * m ** 3 / dt / 1e12, 1)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        del x, y, xf, yf
        # launch latency: many tiny kernels
        z = torch.zeros(32, device="cuda")
        for _ in range(100):
            z.add_(1.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(2000):
            z.add_(1.0)
        torch.cuda.synchronize()
        info["launch_latency_us"] = round((time.perf_counter() - t0) / 2000 * 1e6, 2)
        info["ridge_flop_per_byte"] = round(info["measured_bf16_tflops"] * 1e12 / (info["measured_bandwidth_gbs"] * 1e9), 1)
    except Exception as exc:  # noqa: BLE001
        info["microbench_error"] = str(exc)
    finally:
        _safe(lambda: torch.cuda.empty_cache())
    return info


# CUDA C++ is the implementation backend; `cuda-graphs` captures it and `hub-kernels` is a
# pre-built CUDA kernel, which is still CUDA. The tile DSLs are not probed because they are not
# offered -- see the playbook's module docstring for the measurements behind that.
BACKENDS = ["cuda-cpp", "cuda-graphs", "hub-kernels"]
_MODULES = {
    "cuda-cpp": "fastkernel.backends.cuda_cpp",
    "cuda-graphs": "fastkernel.backends.graphs",
    "hub-kernels": "fastkernel.backends.hub_kernels",
}


def probe_all(names: list[str] | None = None, compile_test: bool = True) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in names or BACKENDS:
        module = importlib.import_module(_MODULES[name])
        started = time.perf_counter()
        try:
            result = module.probe(compile_test=compile_test)
        except Exception as exc:  # noqa: BLE001
            result = {"available": False, "compiled": False, "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("available", False)
        result.setdefault("compiled", False)
        result["probe_seconds"] = round(time.perf_counter() - started, 2)
        result["notes"] = getattr(module, "NOTES", "")
        out[name] = result
    return out


def env_summary() -> dict[str, Any]:
    keys = ["CUDA_HOME", "CUDA_PATH", "TRITON_CACHE_DIR", "TORCH_CUDA_ARCH_LIST", "TILELANG_TARGET", "PATH"]
    return {k: os.environ.get(k) for k in keys if os.environ.get(k)}

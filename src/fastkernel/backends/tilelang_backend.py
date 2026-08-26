"""TileLang: tile-level DSL (TVM based) for GEMM/attention/conv with explicit pipelines."""
from __future__ import annotations

from typing import Any

from .cuda_cpp import ensure_cuda_home

NOTES = (
    "TileLang kernels: @tilelang.jit + T.prim_func with T.Kernel grid, T.alloc_shared/T.alloc_fragment, T.copy, T.gemm, "
    "T.Pipelined. Needs nvcc or NVRTC for the measured compute capability; call ensure_cuda_home() before importing. "
    "Use tilelang.compile(func, out_idx=[...], target='cuda', execution_backend='tvm_ffi'). If nvcc rejects the host compiler "
    "(gcc newer than the CUDA release supports), install a newer self-contained toolchain: `fast-kernel toolchain install --cuda 13.3`."
)


def probe(compile_test: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "compiled": False, "version": None}
    ensure_cuda_home()
    try:
        import tilelang
    except ImportError as exc:
        result["error"] = f"tilelang not importable: {exc}. Fix: uv pip install tilelang"
        return result
    result["available"] = True
    result["version"] = getattr(tilelang, "__version__", None)
    if not compile_test:
        return result
    try:
        import tilelang.language as T
        import torch
        if not torch.cuda.is_available():
            result["error"] = "no CUDA device"
            return result
        n, block = 4096, 256

        @T.prim_func
        def add_kernel(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
            with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
                for i in T.Parallel(block):
                    idx = bx * block + i
                    if idx < n:
                        C[idx] = A[idx] + B[idx]

        last_error = None
        for kwargs in ({"target": "cuda", "execution_backend": "tvm_ffi"}, {"target": "cuda"}, {}):
            try:
                kernel = tilelang.compile(add_kernel, out_idx=[2], **kwargs)
                a = torch.randn(n, device="cuda")
                b = torch.randn(n, device="cuda")
                c = kernel(a, b)
                torch.cuda.synchronize()
                result["compiled"] = bool(torch.allclose(c, a + b))
                result["compile_kwargs"] = kwargs
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if not result["compiled"] and last_error is not None:
            raise last_error
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {_error_summary(str(exc))}"
        if _host_compiler_problem(str(exc)):
            result["fix"] = "host compiler too new for this nvcc: run `fast-kernel toolchain install --cuda 13.3` (self-contained wheels) and re-probe"
    return result


def _error_summary(message: str, limit: int = 700) -> str:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    errors = [line for line in lines if "error" in line.lower() and "warning" not in line.lower()]
    text = " | ".join(errors[:6]) if errors else " | ".join(lines[-4:])
    return text[:limit]


def _host_compiler_problem(message: str) -> bool:
    return any(key in message for key in ("unsupported GNU version", "cudafe++' died", "is not allowed", "__builtin_is_virtual_base_of",
                                          "expected a type specifier", "exception specification is incompatible"))

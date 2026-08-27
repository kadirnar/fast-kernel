"""Triton: the default backend for fused kernels (compiles in seconds, ships ptxas)."""
from __future__ import annotations

from typing import Any

NOTES = (
    "Preferred backend for fusion, norms, elementwise chains, implicit-GEMM conv, attention and quantizer search. "
    "Use @triton.autotune with configs keyed on shapes; persist winners under candidate/tuned/. Cache dir: TRITON_CACHE_DIR."
)


def probe(compile_test: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "compiled": False, "version": None}
    from ..util import ensure_package
    ensure_package("triton")
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        result["error"] = f"triton not importable even after auto-install: {exc}"
        return result
    result["available"] = True
    result["version"] = getattr(triton, "__version__", None)
    if not compile_test:
        return result
    try:
        import torch
        if not torch.cuda.is_available():
            result["error"] = "no CUDA device"
            return result

        @triton.jit
        def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        x = torch.randn(4096, device="cuda")
        y = torch.randn(4096, device="cuda")
        out = torch.empty_like(x)
        _add[(4096 // 1024,)](x, y, out, x.numel(), BLOCK=1024)
        torch.cuda.synchronize()
        result["compiled"] = bool(torch.allclose(out, x + y))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
    return result

"""CuTe DSL (nvidia-cutlass-dsl): Python-authored CUTLASS/CuTe kernels with explicit layouts."""

from typing import Any

from .cuda_cpp import ensure_cuda_home

NOTES = (
    "CuTe DSL: `import cutlass, cutlass.cute as cute`; @cute.kernel device functions, @cute.jit host launchers, "
    "cute.Tensor layouts, TMA/MMA atoms on recent GPUs. Convert torch tensors with cutlass.torch / from_dlpack. "
    "Best for hand-scheduled GEMM/attention; measure against Triton before committing."
)


def probe(compile_test: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "compiled": False, "version": None}
    ensure_cuda_home()
    from ..util import ensure_package
    ensure_package("cutlass", "nvidia-cutlass-dsl")
    try:
        import cutlass
        import cutlass.cute as cute  # noqa: F401
    except ImportError as exc:
        result["error"] = f"cutlass DSL not importable even after auto-install: {exc}"
        return result
    result["available"] = True
    result["version"] = getattr(cutlass, "__version__", None)
    if not compile_test:
        return result
    try:
        import torch
        from cutlass.cute.runtime import from_dlpack
        if not torch.cuda.is_available():
            result["error"] = "no CUDA device"
            return result

        @cute.kernel
        def add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            bdim, _, _ = cute.arch.block_dim()
            idx = bidx * bdim + tidx
            if idx < cute.size(gA):
                gC[idx] = gA[idx] + gB[idx]

        @cute.jit
        def launch(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
            n = cute.size(mA)
            add_kernel(mA, mB, mC).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1))

        a = torch.randn(4096, device="cuda")
        b = torch.randn(4096, device="cuda")
        c = torch.empty_like(a)
        launch(from_dlpack(a), from_dlpack(b), from_dlpack(c))
        torch.cuda.synchronize()
        result["compiled"] = bool(torch.allclose(c, a + b))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
    return result

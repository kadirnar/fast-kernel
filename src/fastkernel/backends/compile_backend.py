"""torch.compile (inductor) probe and helpers."""
from __future__ import annotations

from typing import Any

NOTES = (
    "torch.compile(fn, mode='max-autotune-no-cudagraphs' | 'reduce-overhead', dynamic=False, fullgraph=False). "
    "Compile time is excluded from latency but recompiles on new shapes are not; mark_static / pad to shape buckets."
)


def probe(compile_test: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "compiled": False, "version": None}
    try:
        import torch
    except ImportError as exc:
        result["error"] = str(exc)
        return result
    result["available"] = hasattr(torch, "compile")
    result["version"] = torch.__version__
    if not compile_test or not result["available"]:
        return result
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def fn(x, y):
            return torch.nn.functional.silu(x) * y + 1.0

        compiled = torch.compile(fn, dynamic=False)
        x = torch.randn(1024, device=device)
        y = torch.randn(1024, device=device)
        out = compiled(x, y)
        result["compiled"] = bool(torch.allclose(out, fn(x, y), atol=1e-5))
        result["has_triton"] = _has_triton()
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
    return result


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False

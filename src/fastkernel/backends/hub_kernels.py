"""Hugging Face `kernels` hub: pre-built kernels (flash-attn, activations, norms) without a compiler."""
from __future__ import annotations

from typing import Any

NOTES = (
    "from kernels import get_kernel; k = get_kernel('kernels-community/activation'); k.silu_and_mul(out, x). "
    "Kernels are downloaded once (HF cache) and matched to the torch/CUDA build; useful when the local toolchain "
    "cannot build a backend."
)


def probe(compile_test: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "compiled": False, "version": None}
    from ..util import ensure_package
    ensure_package("kernels")
    try:
        import kernels
    except ImportError as exc:
        result["error"] = f"kernels not importable even after auto-install: {exc}"
        return result
    result["available"] = True
    result["version"] = getattr(kernels, "__version__", None)
    if not compile_test:
        return result
    try:
        import os
        from pathlib import Path
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        cached = sorted(p.name for p in cache.glob("kernels--*")) if cache.exists() else []
        result["cached_kernels"] = cached
        result["compiled"] = True  # nothing to compile locally
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result

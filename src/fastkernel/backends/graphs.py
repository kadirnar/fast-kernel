"""CUDA graph capture helper: turn any callable with static shapes into a single replay.

    graphed = Graphed(fn, example_inputs)   # captures once
    out = graphed(*inputs)                  # copies into static buffers, replays, returns static outputs

Outputs are the graph's static buffers: clone them if you keep them across calls.
"""
from __future__ import annotations

from typing import Any

NOTES = (
    "Whole-workload or per-stage CUDA graphs remove launch latency and Python overhead. Requirements: static shapes "
    "(pad/bucket), no CPU sync inside (.item(), data-dependent control flow), static input/output buffers, warm-up on a "
    "side stream before capture. One graph per shape bucket; fall back to eager for others."
)


EAGER = False   # when True, Graphed.__call__ runs the captured function eagerly (used by the profiler for module attribution)


class eager_mode:
    """Context manager: run every Graphed callable eagerly so the profiler can attribute kernels to modules."""

    def __enter__(self):
        global EAGER
        self._prev = EAGER
        EAGER = True
        return self

    def __exit__(self, *exc):
        global EAGER
        EAGER = self._prev
        return False


class Graphed:
    def __init__(self, fn, example_args: tuple[Any, ...] = (), example_kwargs: dict[str, Any] | None = None,
                 warmup: int = 3, pool=None):
        import torch
        self.fn = fn
        self.args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in example_args)
        self.kwargs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in (example_kwargs or {}).items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(warmup):
                self.fn(*self.args, **self.kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph, pool=pool):
            self.out = self.fn(*self.args, **self.kwargs)
        torch.cuda.synchronize()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch
        if EAGER:
            with torch.inference_mode():
                return self.fn(*args, **kwargs)
        for dst, src in zip(self.args, args, strict=False):
            if isinstance(dst, torch.Tensor):
                dst.copy_(src, non_blocking=True)
        for key, src in kwargs.items():
            dst = self.kwargs.get(key)
            if isinstance(dst, torch.Tensor):
                dst.copy_(src, non_blocking=True)
        self.graph.replay()
        return self.out


class ShapeBucketedGraphs:
    """Keep one captured graph per input shape signature; capture lazily on first sight."""

    def __init__(self, fn, max_graphs: int = 16):
        self.fn = fn
        self.graphs: dict[tuple, Graphed] = {}
        self.max_graphs = max_graphs

    @staticmethod
    def signature(args, kwargs):
        import torch
        parts = []
        for a in list(args) + list(kwargs.values()):
            parts.append((tuple(a.shape), str(a.dtype)) if isinstance(a, torch.Tensor) else repr(a))
        return tuple(parts)

    def __call__(self, *args, **kwargs):
        sig = self.signature(args, kwargs)
        graphed = self.graphs.get(sig)
        if graphed is None:
            if len(self.graphs) >= self.max_graphs:
                return self.fn(*args, **kwargs)
            graphed = Graphed(self.fn, args, kwargs)
            self.graphs[sig] = graphed
        return graphed(*args, **kwargs)


def probe(compile_test: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "compiled": False, "version": None}
    try:
        import torch
    except ImportError as exc:
        result["error"] = str(exc)
        return result
    result["available"] = torch.cuda.is_available()
    result["version"] = torch.__version__
    if not result["available"]:
        result["error"] = "no CUDA device"
        return result
    if not compile_test:
        return result
    try:
        x = torch.randn(1024, device="cuda")
        g = Graphed(lambda t: torch.nn.functional.gelu(t) * 2, (x,))
        y = torch.randn(1024, device="cuda")
        out = g(y)
        torch.cuda.synchronize()
        result["compiled"] = bool(torch.allclose(out, torch.nn.functional.gelu(y) * 2))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
    return result

"""Kernel-level tracing with per-module attribution.

We push a `torch.profiler.record_function` scope for every nn.Module call (forward pre/post hooks),
run the workload under the profiler, export the chrome trace and join CUDA kernels to the innermost
user annotation that was active when their launching CPU op ran. The result attributes every
microsecond of GPU time to a module path, tells how many kernels the workload launches, and how
much of the wall time the GPU was actually busy (launch-bound detection).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from typing import Any

OP_CATEGORY_RULES: list[tuple[tuple[str, ...], str]] = [
    (("scaled_dot_product", "flash", "efficient_attention", "sdpa", "attention"), "attention"),
    (("cudnn_convolution", "convolution", "conv1d", "conv2d", "conv3d", "conv_transpose", "slow_conv", "mkldnn_conv"), "conv"),
    (("addmm", "bmm", "baddbmm", "matmul", "linear", "mm", "gemm", "cublas"), "gemm"),
    (("layer_norm", "rms_norm", "group_norm", "batch_norm", "native_norm", "instance_norm", "norm"), "norm"),
    (("cdist", "codebook", "quantize", "vq"), "quantizer"),
    (("softmax", "argmin", "argmax", "topk", "sum", "mean", "cumsum", "logsumexp", "amax", "amin", "min", "max", "var", "std", "reduce"), "reduction"),
    (("embedding", "index_select", "gather", "scatter", "index_put", "index", "take"), "indexing"),
    (("copy_", "_to_copy", "contiguous", "cat", "stack", "pad", "clone", "transpose", "permute", "slice", "narrow", "view", "reshape", "expand", "fill_", "zeros", "ones", "arange", "memcpy", "memset", "elementwise_kernel_copy"), "memory-movement"),
    (("add", "mul", "sub", "div", "silu", "gelu", "elu", "relu", "sigmoid", "tanh", "exp", "log", "pow", "rsqrt", "sqrt", "where", "clamp", "neg", "abs", "sin", "cos", "erf", "lerp", "addcmul", "mul_", "add_", "elementwise", "vectorized", "unrolled", "activation", "gate", "rope", "rotary"), "elementwise"),
]


def categorize(name: str) -> str:
    text = name.lower()
    for keys, category in OP_CATEGORY_RULES:
        if any(key in text for key in keys):
            return category
    return "other"


def _hook_all_modules(root: Any, module_paths: dict[int, str]) -> list[Any]:
    """Attach record_function scopes to every module; return hook handles."""
    import torch

    handles = []
    stacks: dict[int, list[Any]] = defaultdict(list)

    def make_pre(path: str, cls: str):
        def pre_hook(module, args, kwargs=None):
            scope = torch.profiler.record_function(f"fk::{path}::{cls}")
            scope.__enter__()
            stacks[id(module)].append(scope)
        return pre_hook

    def post_hook(module, args, output):
        stack = stacks.get(id(module))
        if stack:
            stack.pop().__exit__(None, None, None)

    for name, module in root.named_modules():
        path = name or "<root>"
        module_paths[id(module)] = path
        cls = type(module).__name__
        handles.append(module.register_forward_pre_hook(make_pre(path, cls)))
        handles.append(module.register_forward_hook(post_hook))
    return handles


def _capture_shapes(root: Any, shapes: dict[str, dict[str, Any]]) -> list[Any]:
    """Record input/output/parameter shapes per module (one call is enough)."""
    import torch

    handles = []

    def describe(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}
        if isinstance(value, (list, tuple)):
            return [describe(v) for v in value[:4]]
        if isinstance(value, dict):
            return {k: describe(v) for k, v in list(value.items())[:6]}
        return None

    for name, module in root.named_modules():
        path = name or "<root>"

        def hook(module, args, output, path=path):
            if path in shapes:
                shapes[path]["calls"] += 1
                return
            params = {n: list(p.shape) for n, p in module.named_parameters(recurse=False)}
            shapes[path] = {
                "class": type(module).__name__,
                "inputs": [describe(a) for a in args[:4]],
                "output": describe(output),
                "params": params,
                "attrs": {k: getattr(module, k) for k in ("kernel_size", "stride", "padding", "dilation", "groups", "in_channels",
                                                          "out_channels", "in_features", "out_features", "num_heads", "eps")
                          if hasattr(module, k) and isinstance(getattr(module, k), (int, float, tuple, list))},
                "calls": 1,
            }
        handles.append(module.register_forward_hook(hook))
    return handles


def profile_workload(model: Any, run_fn, inputs: dict[str, Any], hooks_root: Any, *, warmup: int = 2,
                     trace_path: str | None = None, attribution_context=None) -> dict[str, Any]:
    """attribution_context: optional context manager active only during the profiled run (e.g. graphs.eager_mode(),
    so CUDA-graph replays are traced as their captured kernels with module attribution); wall time is measured outside it."""
    import contextlib

    import torch
    from torch.profiler import ProfilerActivity, profile
    attribution_context = attribution_context or contextlib.nullcontext

    device_sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    shapes: dict[str, dict[str, Any]] = {}
    module_paths: dict[int, str] = {}
    with torch.inference_mode():
        # Shapes are captured through module forward hooks, so this call must run the *eager* path:
        # a candidate that replays CUDA graphs never enters forward(), the hooks never fire, and every
        # target would be left without shapes -- and therefore without a roofline estimate (SOL = 0 %,
        # headroom collapses to raw share). Same attribution context as the profiled run below.
        shape_handles = _capture_shapes(hooks_root, shapes)
        try:
            with attribution_context():
                run_fn(model, inputs)
        finally:
            for h in shape_handles:
                h.remove()
        for _ in range(max(0, warmup - 1)):
            run_fn(model, inputs)
        device_sync()
        handles = _hook_all_modules(hooks_root, module_paths)
        try:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            with attribution_context():
                run_fn(model, inputs)          # warm the attribution path too (first eager call may allocate)
                device_sync()
                t0 = time.perf_counter()
                with profile(activities=activities, record_shapes=False, with_stack=True) as prof:
                    run_fn(model, inputs)
                    device_sync()
                wall_ms = (time.perf_counter() - t0) * 1e3
        finally:
            for h in handles:
                h.remove()
    # Wall time without the profiler (the profiler adds CPU overhead).  This used to be ONE
    # un-warmed call, which put the denominator of `gpu_busy_ratio` at whatever clock state and
    # first-call cost the GPU happened to be in: on campaigns/mimi it read 1.686 ms against a
    # benchmarked median of 1.430, so the ratio reported 83.6 % busy where the truth against
    # measured latency was 98.6 % -- and agents then hunt 16 % of idle that does not exist.
    with torch.inference_mode():
        for _ in range(3):
            run_fn(model, inputs)
        device_sync()
        walls = []
        for _ in range(7):
            t0 = time.perf_counter()
            run_fn(model, inputs)
            device_sync()
            walls.append((time.perf_counter() - t0) * 1e3)
        walls.sort()
        plain_wall_ms = walls[len(walls) // 2]

    fd, path = tempfile.mkstemp(suffix=".json", prefix="fk-trace-")
    os.close(fd)
    prof.export_chrome_trace(path)
    try:
        with open(path, encoding="utf-8") as fh:
            trace = json.load(fh)
    finally:
        if trace_path:
            try:
                shutil.move(path, trace_path)      # /tmp may be another filesystem
            except OSError:
                os.unlink(path)
        else:
            os.unlink(path)
    result = _analyze_trace(trace, shapes, class_table=_class_line_table(hooks_root))
    result["wall_ms_profiled"] = wall_ms
    result["wall_ms"] = plain_wall_ms
    busy = result["gpu_busy_ms"]
    # Kept for the dashboard, but `evaluate` overrides it with the benchmark's own median once
    # that exists: a ratio is only as good as its denominator, and the benchmark measures the
    # same call under the protocol the campaign's verdicts are made on.
    result["gpu_busy_ratio"] = (busy / plain_wall_ms) if plain_wall_ms > 0 else None
    result["launch_bound"] = bool(result["gpu_busy_ratio"] is not None and result["gpu_busy_ratio"] < 0.6)
    result["avg_kernel_us"] = (busy * 1e3 / result["kernel_count"]) if result["kernel_count"] else None
    return result


def _class_line_table(root: Any) -> dict[str, list[tuple[int, int, str]]]:
    """(file -> [(start, end, ClassName)]) for every nn.Module class in the model, so that Python frames of
    non-forward methods (encode / quantize / generate ...) can still be attributed to a module class."""
    import inspect
    table: dict[str, list[tuple[int, int, str]]] = {}
    seen: set[type] = set()
    for module in root.modules():
        for cls in type(module).__mro__:
            if cls in seen or cls.__module__ in ("torch.nn.modules.module", "builtins"):
                continue
            seen.add(cls)
            try:
                file = inspect.getsourcefile(cls)
                lines, start = inspect.getsourcelines(cls)
            except (OSError, TypeError):
                continue
            if not file:
                continue
            table.setdefault(file, []).append((start, start + len(lines), cls.__name__))
    for entries in table.values():
        entries.sort(key=lambda e: (e[1] - e[0]))   # innermost (shortest) range first
    return table


_FRAME_RE = re.compile(r"^(.*)\((\d+)\): (.*)$")


class _FrameIndex:
    """Attribute timestamps to the innermost active Python frame (per thread) using the `python_function`
    events the profiler emits with with_stack=True. Frames nest properly per thread, so one sweep in time
    order with a stack answers every query."""

    def __init__(self, events: list[dict[str, Any]], class_table: dict[str, list[tuple[int, int, str]]]):
        self.by_tid: dict[Any, list[tuple[float, float, str]]] = {}
        for ev in events:
            if ev.get("ph") != "X" or ev.get("cat") != "python_function":
                continue
            self.by_tid.setdefault(ev.get("tid"), []).append((ev["ts"], ev["ts"] + ev.get("dur", 0), ev.get("name", "")))
        for frames in self.by_tid.values():
            frames.sort(key=lambda f: (f[0], -f[1]))
        self.class_table = class_table
        self._by_base: dict[str, list[str]] = {}
        for file in class_table:
            self._by_base.setdefault(os.path.basename(file), []).append(file)
        self._cache: dict[str, tuple[str, str] | None] = {}

    def _table_file(self, frame_file: str) -> str | None:
        """Frame paths are often relative to site-packages; match the class table by path suffix."""
        for candidate in self._by_base.get(os.path.basename(frame_file), []):
            if candidate == frame_file or candidate.endswith(os.sep + frame_file) or frame_file.endswith(os.sep + candidate):
                return candidate
        return None

    def _classify(self, name: str) -> tuple[str, str] | None:
        if name in self._cache:
            return self._cache[name]
        result: tuple[str, str] | None = None
        match = _FRAME_RE.match(name)
        if match and not match.group(3).startswith("<"):   # <genexpr>/<listcomp>/<lambda> frames have unreliable spans
            frame_file, line, func = match.group(1), int(match.group(2)), match.group(3)
            file = self._table_file(frame_file)
            if file is None and ("/candidate/" in frame_file or frame_file.startswith("candidate/")):
                # agent-owned code launching work outside the reference modules (CUDA-graph replays, custom ops)
                result = (f"candidate:{func}", f"candidate:{func}")
            elif file is not None:
                result = (f"{os.path.basename(file)}:{func}", "")
                for c_start, c_end, cls in self.class_table[file]:
                    if c_start <= line <= c_end:
                        result = (f"{cls}.{func}", cls)
                        break
        self._cache[name] = result
        return result

    def resolve_many(self, queries: list[tuple[float, Any, Any]]) -> dict[Any, tuple[str, str]]:
        """queries: [(ts, tid, key)] -> {key: (path, class)} for those inside model code."""
        out: dict[Any, tuple[str, str]] = {}
        by_tid: dict[Any, list[tuple[float, Any]]] = {}
        for ts, tid, key in queries:
            by_tid.setdefault(tid, []).append((ts, key))
        for tid, items in by_tid.items():
            frames = self.by_tid.get(tid, [])
            items.sort(key=lambda q: q[0])
            stack: list[tuple[float, float, str]] = []
            fi = 0
            for ts, key in items:
                while fi < len(frames) and frames[fi][0] <= ts:
                    stack.append(frames[fi])
                    fi += 1
                while stack and stack[-1][1] < ts:
                    stack.pop()
                best_file = None
                for frame in reversed(stack):
                    hit = self._classify(frame[2])
                    if hit is None:
                        continue
                    if hit[1]:
                        out[key] = hit
                        break
                    if best_file is None:
                        best_file = hit
                else:
                    if best_file is not None:
                        out[key] = best_file
        return out


def _analyze_trace(trace: dict[str, Any], shapes: dict[str, dict[str, Any]],
                   class_table: dict[str, list[tuple[int, int, str]]] | None = None) -> dict[str, Any]:
    events = trace.get("traceEvents", [])
    frame_index = _FrameIndex(events, class_table or {})
    annotations: list[dict[str, Any]] = []
    cpu_ops: dict[Any, dict[str, Any]] = {}
    kernels: list[dict[str, Any]] = []
    runtime_by_corr: dict[Any, dict[str, Any]] = {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if cat in ("user_annotation", "gpu_user_annotation") and name.startswith("fk::"):
            if cat == "user_annotation":
                annotations.append(ev)
        elif cat == "cpu_op":
            ext = (ev.get("args") or {}).get("External id")
            if ext is not None:
                cpu_ops.setdefault(ext, ev)
        elif cat == "cuda_runtime":
            corr = (ev.get("args") or {}).get("correlation")
            if corr is not None:
                runtime_by_corr[corr] = ev
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(ev)
    annotations.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))

    def innermost_annotation(ts: float, tid: Any) -> dict[str, Any] | None:
        best = None
        for ann in annotations:
            if ann.get("tid") != tid:
                continue
            start = ann["ts"]
            end = start + ann.get("dur", 0)
            if start <= ts <= end:
                if best is None or ann.get("dur", 0) <= best.get("dur", 0):
                    best = ann
            elif start > ts:
                break
        return best

    per_module: dict[str, dict[str, Any]] = {}
    per_kernel: dict[str, dict[str, Any]] = {}
    per_op: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[float, float]] = []
    total_gpu_us = 0.0
    # resolve kernels that fall outside every forward() scope through the Python call stack
    pending: list[tuple[float, Any, int]] = []
    for idx, kern in enumerate(kernels):
        args = kern.get("args") or {}
        anchor = cpu_ops.get(args.get("External id")) or runtime_by_corr.get(args.get("correlation"))
        if anchor is not None and innermost_annotation(anchor["ts"], anchor.get("tid")) is None:
            pending.append((anchor["ts"], anchor.get("tid"), idx))
    resolved = frame_index.resolve_many(pending)
    for idx, kern in enumerate(kernels):
        dur = float(kern.get("dur", 0))
        total_gpu_us += dur
        intervals.append((kern["ts"], kern["ts"] + dur))
        args = kern.get("args") or {}
        ext = args.get("External id")
        corr = args.get("correlation")
        cpu_op = cpu_ops.get(ext)
        launcher = runtime_by_corr.get(corr)
        op_name = cpu_op.get("name") if cpu_op else "<unknown>"
        anchor = cpu_op or launcher
        module_path, module_cls = "<unattributed>", ""
        if anchor is not None:
            ann = innermost_annotation(anchor["ts"], anchor.get("tid"))
            if ann is not None:
                _, module_path, module_cls = ann["name"].split("::", 2)
            elif idx in resolved:
                module_path, module_cls = resolved[idx]
        category = categorize(op_name if op_name != "<unknown>" else kern.get("name", ""))
        if category == "other":
            category = categorize(kern.get("name", ""))
        mod = per_module.setdefault(module_path, {"path": module_path, "class": module_cls, "gpu_us": 0.0,
                                                  "kernel_count": 0, "ops": defaultdict(float), "categories": defaultdict(float)})
        mod["gpu_us"] += dur
        mod["kernel_count"] += 1
        mod["ops"][op_name] += dur
        mod["categories"][category] += dur
        kname = kern.get("name", "")
        k = per_kernel.setdefault(kname, {"name": kname, "count": 0, "gpu_us": 0.0, "category": category})
        k["count"] += 1
        k["gpu_us"] += dur
        o = per_op.setdefault(op_name, {"op": op_name, "count": 0, "gpu_us": 0.0, "category": category})
        o["count"] += 1
        o["gpu_us"] += dur
    # union of kernel intervals = GPU busy time
    intervals.sort()
    busy = 0.0
    cur_start, cur_end = None, None
    for start, end in intervals:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        busy += cur_end - cur_start

    modules_out = []
    for mod in per_module.values():
        top_cat = max(mod["categories"].items(), key=lambda kv: kv[1])[0] if mod["categories"] else "other"
        shape_info = shapes.get(mod["path"], {})
        modules_out.append({
            "path": mod["path"], "class": mod["class"] or shape_info.get("class", ""), "gpu_us": round(mod["gpu_us"], 2),
            "kernel_count": mod["kernel_count"], "category": top_cat,
            "categories": {k: round(v, 2) for k, v in sorted(mod["categories"].items(), key=lambda kv: -kv[1])},
            "ops": {k: round(v, 2) for k, v in sorted(mod["ops"].items(), key=lambda kv: -kv[1])[:8]},
            "shapes": shape_info,
            "fraction": (mod["gpu_us"] / total_gpu_us) if total_gpu_us else 0.0,
        })
    modules_out.sort(key=lambda m: -m["gpu_us"])
    kernels_out = sorted(per_kernel.values(), key=lambda k: -k["gpu_us"])[:60]
    ops_out = sorted(per_op.values(), key=lambda k: -k["gpu_us"])[:60]
    for item in kernels_out + ops_out:
        item["gpu_us"] = round(item["gpu_us"], 2)
    return {
        "kernel_count": len(kernels), "gpu_time_ms": total_gpu_us / 1e3, "gpu_busy_ms": busy / 1e3,
        "modules": modules_out, "kernels": kernels_out, "ops": ops_out,
        "module_shapes_count": len(shapes),
    }

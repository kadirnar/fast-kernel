"""Roofline-style classification of hotspots: compute-, memory- or latency-bound."""
from __future__ import annotations

import math
from typing import Any

DTYPE_BYTES = {"float32": 4, "float": 4, "bfloat16": 2, "float16": 2, "half": 2, "int8": 1, "uint8": 1,
               "int32": 4, "int64": 8, "long": 8, "bool": 1, "float8_e4m3fn": 1}


def _numel(shape: list[int] | None) -> int:
    if not shape:
        return 0
    return int(math.prod(shape))


def _dtype_bytes(desc: dict[str, Any] | None) -> int:
    if not desc or not isinstance(desc, dict):
        return 4
    return DTYPE_BYTES.get(desc.get("dtype", "float32"), 4)


def _first_tensor(descs: Any) -> dict[str, Any] | None:
    if isinstance(descs, dict) and "shape" in descs:
        return descs
    if isinstance(descs, (list, tuple)):
        for item in descs:
            found = _first_tensor(item)
            if found:
                return found
    if isinstance(descs, dict):
        for item in descs.values():
            found = _first_tensor(item)
            if found:
                return found
    return None


def estimate_flops_bytes(module: dict[str, Any]) -> tuple[float, float]:
    """Rough FLOPs / bytes for one call of a module from its recorded shapes."""
    shapes = module.get("shapes") or {}
    cls = (shapes.get("class") or module.get("class") or "").lower()
    inp = _first_tensor(shapes.get("inputs"))
    out = _first_tensor(shapes.get("output"))
    params = shapes.get("params") or {}
    attrs = shapes.get("attrs") or {}
    calls = max(1, int(shapes.get("calls", 1)))
    in_n, out_n = _numel(inp["shape"]) if inp else 0, _numel(out["shape"]) if out else 0
    in_b, out_b = _dtype_bytes(inp), _dtype_bytes(out)
    param_bytes = sum(_numel(s) for s in params.values()) * (in_b if in_b else 4)
    flops, byts = 0.0, 0.0
    weight = params.get("weight")
    if "linear" in cls and weight and inp:
        k = weight[1] if len(weight) > 1 else weight[0]
        n = weight[0]
        m = in_n // max(1, k)
        flops = 2.0 * m * n * k
        byts = (m * k + k * n + m * n) * in_b
    elif ("conv" in cls) and weight and out:
        if "transpose" in cls:
            kernel_prod = _numel(weight[2:]) if len(weight) > 2 else 1
            groups = int(attrs.get("groups", 1) or 1)
            flops = 2.0 * in_n * (weight[1] * groups) * kernel_prod
        else:
            kernel_prod = _numel(weight[2:]) if len(weight) > 2 else 1
            groups = int(attrs.get("groups", 1) or 1)
            cin_per_group = weight[1]
            flops = 2.0 * out_n * cin_per_group * kernel_prod
        byts = in_n * in_b + out_n * out_b + param_bytes
    elif "attention" in cls or "attn" in cls:
        # projections + QK^T + PV; assume (B, T, D)
        if inp and len(inp["shape"]) >= 3:
            b, t, d = inp["shape"][0], inp["shape"][1], inp["shape"][-1]
            flops = 2.0 * b * t * d * d * 4 + 4.0 * b * t * t * d
            byts = in_n * in_b * 6 + param_bytes
    elif "embedding" in cls:
        byts = out_n * out_b
    elif any(key in cls for key in ("norm", "act", "silu", "gelu", "elu", "relu", "sigmoid", "dropout", "rotary", "rope")):
        flops = 8.0 * max(in_n, out_n)
        byts = (in_n * in_b + out_n * out_b) + param_bytes
    elif "mlp" in cls or "feedforward" in cls or "ffn" in cls:
        total_params = sum(_numel(s) for s in params.values())
        if inp and total_params:
            tokens = in_n // max(1, inp["shape"][-1])
            flops = 2.0 * tokens * total_params
            byts = total_params * in_b + in_n * in_b + out_n * out_b
    else:
        byts = (in_n * in_b + out_n * out_b) + param_bytes
        flops = 2.0 * max(in_n, out_n)
    return flops * calls, byts * calls


def classify_module(module: dict[str, Any], device: dict[str, Any]) -> dict[str, Any]:
    flops, byts = estimate_flops_bytes(module)
    peak_flops = float(device.get("measured_bf16_tflops") or device.get("measured_fp32_tflops") or 50.0) * 1e12
    peak_bw = float(device.get("measured_bandwidth_gbs") or 500.0) * 1e9
    launch_us = float(device.get("launch_latency_us") or 4.0)
    ridge = peak_flops / peak_bw
    gpu_us = float(module.get("gpu_us", 0.0))
    kernel_count = int(module.get("kernel_count", 0) or 0)
    avg_kernel_us = gpu_us / kernel_count if kernel_count else 0.0
    intensity = (flops / byts) if byts else 0.0
    achieved_tflops = (flops / (gpu_us * 1e-6)) / 1e12 if gpu_us > 0 and flops else 0.0
    achieved_gbs = (byts / (gpu_us * 1e-6)) / 1e9 if gpu_us > 0 and byts else 0.0
    if kernel_count and avg_kernel_us < 2.5 * launch_us:
        bound = "latency"
    elif intensity and intensity > ridge:
        bound = "compute"
    else:
        bound = "memory"
    category = module.get("category", "other")
    if category == "other" and "quant" in (module.get("class", "") or "").lower():
        category = "quantizer"
    return {
        **module, "category": category, "boundness": bound, "flops": flops, "bytes": byts,
        "arith_intensity": intensity, "ridge_intensity": ridge, "avg_kernel_us": avg_kernel_us,
        "achieved_tflops": achieved_tflops, "achieved_gbs": achieved_gbs,
        "pct_peak_compute": (achieved_tflops * 1e12 / peak_flops * 100) if flops else None,
        "pct_peak_bandwidth": (achieved_gbs * 1e9 / peak_bw * 100) if byts else None,
    }

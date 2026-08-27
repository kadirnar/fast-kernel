from fastkernel.playbook import BY_ID, CATEGORIES, TECHNIQUES, techniques_for
from fastkernel.profiling.classify import classify_module, estimate_flops_bytes
from fastkernel.profiling.plan import render_plan_md
from fastkernel.profiling.rank import build_targets, technique_matrix
from fastkernel.profiling.trace import categorize

DEVICE = {"name": "test-gpu", "measured_bandwidth_gbs": 800.0, "measured_bf16_tflops": 200.0, "launch_latency_us": 4.0,
          "compute_capability": "12.0", "sm_count": 70, "total_memory_gb": 16}


def linear_module(path, gpu_us, kernels, calls=1):
    return {"path": path, "class": "Linear", "gpu_us": gpu_us, "kernel_count": kernels, "category": "gemm",
            "shapes": {"class": "Linear", "inputs": [{"shape": [1, 512, 1024], "dtype": "bfloat16"}],
                       "output": {"shape": [1, 512, 4096], "dtype": "bfloat16"}, "params": {"weight": [4096, 1024]}, "calls": calls}}


def test_playbook_consistency():
    assert len({t.id for t in TECHNIQUES}) == len(TECHNIQUES)
    for tech in TECHNIQUES:
        assert all(c in CATEGORIES for c in tech.applies_to), tech.id
        assert tech.expected_speedup > 1.0
    assert BY_ID["cuda-graphs"].tier == 0
    assert techniques_for("launch-bound", "latency")[0].id == "cuda-graphs"
    assert any(t.id == "fused-quantizer" for t in techniques_for("quantizer", "memory"))


def test_categorize_ops():
    assert categorize("aten::addmm") == "gemm"
    assert categorize("aten::cudnn_convolution") == "conv"
    assert categorize("aten::_scaled_dot_product_flash_attention") == "attention"
    assert categorize("aten::native_layer_norm") == "norm"
    assert categorize("aten::argmin") == "reduction"
    assert categorize("aten::embedding") == "indexing"
    assert categorize("aten::_to_copy") == "memory-movement"
    assert categorize("aten::silu") == "elementwise"
    assert categorize("something_weird") == "other"


def test_flops_and_classification():
    mod = linear_module("layers.0.fc", 100.0, 1)
    flops, byts = estimate_flops_bytes(mod)
    assert flops == 2.0 * 512 * 4096 * 1024
    assert byts > 0
    cls = classify_module(mod, DEVICE)
    assert cls["boundness"] in ("compute", "memory")
    tiny = classify_module(linear_module("layers.0.small", 5.0, 3), DEVICE)
    assert tiny["boundness"] == "latency"


def test_build_targets_amdahl_and_launch_bound():
    profile = {"wall_ms": 10.0, "gpu_busy_ms": 3.0, "kernel_count": 900, "avg_kernel_us": 3.3,
               "modules": [linear_module(f"layers.{i}.fc", 100.0, 4) for i in range(4)]
               + [{"path": "quant", "class": "Codebook", "gpu_us": 600.0, "kernel_count": 300, "category": "reduction", "shapes": {}}]}
    hints = [{"symbol": "Codebook", "category": "quantizer", "note": "fuse"}]
    targets = build_targets(profile, DEVICE, history=[], hints=hints)
    assert targets[0]["category"] == "launch-bound"
    assert targets[0]["fraction"] > 0.6
    groups = {t["class"]: t for t in targets}
    assert groups["Linear"]["instance_count"] == 4
    assert groups["Codebook"]["category"] == "quantizer"
    assert groups["Codebook"]["techniques"][0]["status"] == "untried"
    assert all(t["amdahl_gain"] <= t["fraction"] for t in targets)
    # history demotes tried techniques
    hist = [{"number": 1, "status": "discard", "target": groups["Codebook"]["id"], "techniques": ["fused-quantizer"]},
            {"number": 2, "status": "keep", "target": groups["Linear"]["id"], "techniques": ["epilogue-fusion"]}]
    matrix = technique_matrix(hist)
    assert matrix[(groups["Codebook"]["id"], "fused-quantizer")] == "rejected"
    targets2 = build_targets(profile, DEVICE, history=hist, hints=hints)
    cb = next(t for t in targets2 if t["class"] == "Codebook")
    statuses = {t["id"]: t["status"] for t in cb["techniques"]}
    assert statuses["fused-quantizer"] == "rejected"
    assert cb["techniques"][0]["status"] == "untried"


def test_render_plan():
    profile = {"wall_ms": 2.0, "gpu_busy_ms": 1.9, "kernel_count": 10, "avg_kernel_us": 190.0,
               "modules": [linear_module("fc", 1900.0, 10)]}
    targets = build_targets(profile, DEVICE)
    payload = {"generated_at": "now", "workload": "w", "experiment": 3, "summary": {"wall_ms": 2.0, "gpu_busy_ms": 1.9, "gpu_time_ms": 1.9,
               "gpu_busy_ratio": 0.95, "kernel_count": 10, "avg_kernel_us": 190.0, "launch_bound": False}, "device": DEVICE,
               "targets": targets, "top_kernels": [{"name": "k", "count": 1, "gpu_us": 1.0, "category": "gemm"}], "top_ops": []}
    md = render_plan_md(payload, "notes here", {"triton": {"compiled": True, "version": "3.7"}})
    assert "GPU-BOUND" in md and "Linear" in md and "notes here" in md and "triton" in md
    # agent-facing PLAN.md must not prescribe a technique or predict a speedup
    assert "techniques to try" not in md and "expected speedup" not in md

"""The optimization playbook: a technique catalogue the ranker uses to recommend what to try.

It generalizes AutoKernel's six tiers (block-size tuning, memory access, compute/mixed precision,
advanced (split-K, persistent), architecture-specific, kernel-specific tricks) to whole-model
inference, where the biggest wins usually come from *launch/latency* problems (CUDA graphs,
fusion, fewer kernels) rather than from peak-FLOPS GEMM tuning.

Each technique declares which hotspot categories it applies to, the backends that can implement
it, the expected speedup on the *targeted fraction* (used by the Amdahl ranking), risk, and the
skill that documents how to do it. Nothing here is a hardware restriction: expected gains are
priors that measurements overwrite.

**CUDA C++ is the implementation backend.**  A kernel written here is hand-written CUDA (plus
`cuda-graphs` for capture and `torch` for the ops that stay stock); the DSL backends are not
offered.  The reason is measured rather than stylistic: the wins that remain in a mature campaign
come from *fusion granularity* -- a whole resnet block or a whole quantizer stage in one launch, so
its intermediates never reach global memory -- and that needs control a tile DSL's automatic
pipeline does not give.  In `campaigns/mimi`, folding an ELU into a convolution's shared-memory
staging cost 3.6x under Triton because `cp.async` is a DMA and cannot transform a value in flight,
so the activation must land, be read back, and a barrier must sit inside the very pipeline that
exists to hide latency.  Adopting a CUDA-fused lineage of the same model, with one launch per
resnet block and one per RVQ stage, was worth **+13.8 %** in a single experiment where a season of
tile tuning on the DSL tree had run out at 70-98 % of every measured floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Hotspot categories produced by profiling/classify.py
CATEGORIES = [
    "launch-bound",      # whole-workload: GPU idle between many tiny kernels / python overhead
    "gemm",              # linear / matmul / bmm
    "conv",              # conv1d / conv2d / conv_transpose
    "attention",         # sdpa / qk^T softmax pv
    "norm",              # layernorm / rmsnorm / groupnorm / batchnorm
    "elementwise",       # activations, residual adds, casts, muls, gating
    "reduction",         # softmax, argmin/argmax, cdist, topk, sums
    "indexing",          # embedding / gather / scatter / index_select
    "memory-movement",   # copy / contiguous / cat / pad / transpose / slicing
    "quantizer",         # codebook search (VQ/RVQ), nearest-neighbour distance+argmin
    "sequential",        # dependent chains (RVQ stages, autoregressive decode steps)
    "other",
]

# CUDA C++ is the implementation backend; `torch` and `cuda-graphs` remain because "leave it stock"
# and "capture it" are legitimate answers, and `hub-kernels` because a pre-built CUDA kernel is
# still CUDA. The tile DSLs are deliberately absent -- see the module docstring.
BACKENDS = ["torch", "cuda-graphs", "cuda-cpp", "hub-kernels"]


@dataclass(frozen=True)
class Technique:
    id: str
    title: str
    tier: int
    applies_to: tuple[str, ...]
    boundness: tuple[str, ...]          # compute | memory | latency | any
    backends: tuple[str, ...]
    expected_speedup: float             # on the targeted fraction; prior only
    risk: str                           # low | medium | high
    skill: str
    summary: str
    steps: tuple[str, ...] = field(default_factory=tuple)


TECHNIQUES: list[Technique] = [
    # ---- Tier 0: whole-workload structure (usually the biggest wins for inference) -----
    Technique("cuda-graphs", "Capture the workload (or each stage) into CUDA graphs", 0,
              ("launch-bound", "sequential"), ("latency", "any"), ("cuda-graphs", "cuda-cpp"), 3.0, "low", "cuda-graphs",
              "Removes per-kernel launch latency and Python overhead; one replay instead of N launches. Requires static "
              "shapes and static input/output buffers (one graph per shape bucket).",
              ("Make the forward shape-static (pad to buckets); allocate static I/O buffers.",
               "Warm up on a side stream, capture with torch.cuda.CUDAGraph, replay in the candidate apply().",
               "Keep a per-shape cache; fall back to eager for shapes outside the buckets.")),
    Technique("kernel-count-reduction", "Reduce kernel count: fuse elementwise chains, remove copies/casts", 0,
              ("launch-bound", "elementwise", "memory-movement"), ("latency", "memory"), ("cuda-cpp",),
              2.0, "low", "cuda-cpp-kernels",
              "When the GPU is idle most of the wall time, each removed launch is worth its launch latency (~3-8 us) plus its "
              "DRAM round trip. Fuse activation+residual+cast into the producing GEMM/conv epilogue.",
              ("Count launches per forward (fast-kernel profile reports it).",
               "Replace chains like norm->linear->act->add with one fused kernel or an epilogue.")),
    Technique("weight-prepack", "Pre-cast / pre-transpose / pre-fold weights at load time", 0,
              ("gemm", "conv", "norm", "memory-movement"), ("memory", "latency"), ("torch", "cuda-cpp"), 1.3, "low", "cuda-cpp-kernels",
              "Fold BatchNorm into conv, pre-cast weights to bf16/fp16, pre-transpose to the layout the kernel wants, "
              "merge Q/K/V or gate/up projections into one GEMM.",
              ("Do it once in apply(); never inside the forward.",)),
    Technique("dtype-policy", "Lower-precision compute with fp32 accumulation where the oracle tolerates it", 0,
              ("gemm", "conv", "attention"), ("compute", "memory"), ("torch", "cuda-cpp"), 1.7, "medium",
              "numerical-verification",
              "bf16/fp16 tensor cores and half the bytes. Keep residual streams / accumulators fp32; verify with the gates "
              "(strict policies may forbid this).",
              ("Try TF32 first (allow_tf32), then bf16 weights with fp32 activations, then full bf16.",)),
    # ---- Tier 1-2: memory & block tuning ---------------------------------------------------
    Technique("block-tuning", "Block-size / num_warps / num_stages sweep (autotune)", 1,
              ("gemm", "conv", "attention", "norm", "reduction"), ("compute", "memory"), ("cuda-cpp",),
              1.3, "low", "cuda-cpp-kernels",
              "Sweep the tile constants (BLOCK_M/N/K in powers of two, 16..256, rectangular tiles), the block size (64..512 "
              "threads) and the pipeline depth. Cache the winning config per shape.",
              ("Sweep the tile constants of the CUDA kernel per shape; persist winners to candidate/tuned/*.json.",)),
    Technique("memory-access", "Coalescing, vectorized loads, channels-last, L2 swizzle, prefetch", 2,
              ("gemm", "conv", "norm", "elementwise", "reduction", "indexing"), ("memory",), ("cuda-cpp",),
              1.3, "low", "cuda-cpp-kernels",
              "Ensure contiguous access along the fastest dimension; vectorize with float4/__ldg; transpose operands or switch "
              "to channels-last; swizzle block indices so neighbouring tiles share L2; pipeline global->shared with cp.async.",
              ()),
    # ---- Tier 3: compute -------------------------------------------------------------------
    Technique("epilogue-fusion", "Fuse bias/activation/residual/cast/quant into the GEMM or conv epilogue", 3,
              ("gemm", "conv", "elementwise"), ("memory", "latency"), ("cuda-cpp",), 1.5, "low",
              "cuda-cpp-kernels", "Avoid writing and re-reading the intermediate; typical for MLP (act*gate), conv+ELU, residual adds.",
              ()),
    Technique("implicit-gemm-conv", "Implicit-GEMM convolution on tensor cores", 3,
              ("conv",), ("compute", "memory"), ("cuda-cpp",), 2.0, "medium", "cuda-cpp-kernels",
              "Express conv1d/conv2d as a GEMM over (out_positions) x (in_channels*kernel) without materializing im2col; fuse "
              "causal padding, ELU/SiLU epilogues; split-K for narrow outputs.",
              ()),
    Technique("fused-attention", "Fused attention (flash-style online softmax) or the right SDPA backend", 3,
              ("attention",), ("compute", "memory", "latency"), ("torch", "cuda-cpp", "hub-kernels"), 1.6, "medium",
              "cuda-cpp-kernels", "For short sequences a single fused kernel (QKV proj + attention + out proj) beats 6 launches; for "
              "long ones use flash attention. Handle causal / sliding-window / RoPE inside.",
              ()),
    Technique("fused-norm", "Fused RMSNorm/LayerNorm (+ optional residual and quant) kernels", 3,
              ("norm",), ("memory", "latency"), ("cuda-cpp", "hub-kernels"), 2.5, "low", "cuda-cpp-kernels",
              "One pass, row per program, Welford or sum-of-squares in fp32; fuse the following projection's input cast.", ()),
    Technique("fused-quantizer", "Fused codebook distance + argmin (VQ / RVQ search)", 3,
              ("quantizer", "reduction", "sequential"), ("memory", "latency"), ("cuda-cpp",), 3.0, "medium",
              "cuda-cpp-kernels",
              "Never materialize cdist: compute -2 x.c + |c|^2 on tensor cores in a coarse pass, re-rank the top-k exactly in fp32, "
              "run all residual stages in one persistent kernel with a grid barrier.",
              ()),
    Technique("fused-elementwise", "Fuse activation / gating / residual chains into one kernel", 3,
              ("elementwise", "memory-movement"), ("memory", "latency"), ("cuda-cpp",), 2.0, "low", "cuda-cpp-kernels",
              "Each unfused elementwise op is a full DRAM round trip; SiLU*gate, GELU, ELU, residual add, RoPE apply are all "
              "cheap to fuse.", ()),
    # ---- Tier 4: advanced ------------------------------------------------------------------
    Technique("split-k", "Split-K / stream-K for skinny GEMMs and small batch", 4,
              ("gemm", "conv"), ("compute", "latency"), ("cuda-cpp",), 1.4, "medium",
              "cuda-cpp-kernels", "When M*N tiles cannot fill the SMs, split the reduction across CTAs and reduce in fp32.", ()),
    Technique("persistent-kernel", "Persistent / megakernel with grid barriers for dependent stages", 4,
              ("sequential", "launch-bound", "quantizer"), ("latency",), ("cuda-cpp",), 1.5, "high", "cuda-cpp-kernels",
              "One launch, all phases; only pays off when the per-phase barrier cost is below the launch latency it replaces "
              "(measure! on some GPUs a 4-kernel version wins).", ()),
    Technique("warp-specialization", "Warp specialization, TMA, async copies", 4,
              ("gemm", "attention", "conv"), ("compute", "memory"), ("cuda-cpp",), 1.3, "high",
              "cuda-cpp-kernels", "Producer/consumer warps, TMA loads, cp.async pipelines; large tiles and deep pipelines.", ()),
    # ---- Tier 5: architecture-specific -----------------------------------------------------
    Technique("arch-tuning", "Architecture-specific tuning (SM count, smem size, tensor-core shapes)", 5,
              ("gemm", "conv", "attention"), ("compute",), ("cuda-cpp",), 1.15, "medium",
              "cuda-cpp-kernels", "Re-tune tiles for the SM count and shared memory budget of the *measured* device; check "
              "capabilities.json rather than assuming a generation.", ()),
    # ---- Tier 6: kernel-specific tricks ----------------------------------------------------
    Technique("indexing-fusion", "Fuse embedding/gather with the consumer; avoid index_select copies", 6,
              ("indexing", "memory-movement"), ("memory", "latency"), ("cuda-cpp",), 1.5, "low", "cuda-cpp-kernels", "", ()),
    Technique("decode-step-fusion", "Fuse the autoregressive decode step (norm+proj+attn+mlp) per layer", 6,
              ("sequential", "launch-bound", "gemm"), ("latency", "memory"), ("cuda-graphs", "cuda-cpp"), 2.5,
              "medium", "cuda-graphs", "Decode is launch bound: capture the whole step in a CUDA graph with a static KV cache; "
              "fuse GEMV-shaped projections.", ()),
    Technique("hub-kernel", "Use a pre-built kernel from the Hugging Face kernels hub", 6,
              ("attention", "norm", "elementwise", "gemm"), ("any",), ("cuda-cpp", "hub-kernels"), 1.5, "low", "hub-kernels",
              "kernels.get_kernel('kernels-community/...') gives tested flash-attn / activation / norm kernels without a "
              "compiler on the box.", ()),
]

BY_ID = {t.id: t for t in TECHNIQUES}


def techniques_for(category: str, boundness: str) -> list[Technique]:
    out = []
    for tech in TECHNIQUES:
        if category not in tech.applies_to:
            continue
        if "any" in tech.boundness or boundness in tech.boundness:
            out.append(tech)
    out.sort(key=lambda t: (t.tier, -t.expected_speedup))
    return out


def expected_speedup(category: str, boundness: str) -> float:
    techs = techniques_for(category, boundness)
    if not techs:
        return 1.2
    return max(t.expected_speedup for t in techs)

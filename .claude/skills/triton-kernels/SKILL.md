---
name: triton-kernels
description: How to write, autotune and integrate Triton kernels for fast-kernel candidates - fused norms, elementwise/gating chains, epilogue-fused GEMMs, implicit-GEMM convolutions, fused attention, codebook argmin, persistent kernels. Use when implementing any Triton kernel.
argument-hint: [kernel-kind]
allowed-tools: Bash(fast-kernel templates), Read, Edit, Write, Glob, Grep
---

# Triton kernels for fast-kernel

Start from a template (`fast-kernel templates` lists paths; copy into `candidate/kernels/`):
`triton_rmsnorm.py` (row-per-program norm + residual), `triton_fused_silu_mul.py`, `triton_matmul.py`
(autotuned bf16 GEMM, fused bias/act/residual epilogue, grouped tile order), `triton_causal_conv1d.py`
(depthwise causal conv + gating; LFM2 ShortConv), `triton_codebook_argmin.py` (fused -2x·c + |c|² argmin
on tensor cores; Mimi RVQ).

## Recipe

1. **Contract first**: write the torch reference of exactly what the kernel replaces (shapes, dtypes,
   strides, masking, padding) and a 5-line self-test (`if __name__ == "__main__"`), with seeds and
   `torch.testing.assert_close` at the tolerance your gate policy allows.
2. **Grid & tiles**: one program per row for norms/softmax (BLOCK = next_power_of_2(cols), num_warps 4-8);
   2-D tiles for GEMM/conv (start 64×64×32, 4 warps, 3 stages) with `@triton.autotune(configs, key=[M,N,K])`.
   Persist tuned configs (`candidate/tuned/*.json`) so timed runs never include autotuning.
3. **Numerics**: load in the storage dtype, compute/accumulate in fp32 (`tl.dot(a, b, acc)`), cast in the
   epilogue. For strict policies keep reductions in fp32 and avoid `tl.atomic_add` (non-deterministic).
4. **Fusion**: put bias, activation (silu/gelu/elu), residual add, LayerScale, cast, quantization into the
   epilogue; pass `HAS_*: tl.constexpr` flags rather than branching at runtime.
5. **Memory**: coalesce along the contiguous dim; `.contiguous()` once outside the hot loop or better,
   make the producer emit the layout you want (channels-last for convs); avoid `tl.trans` on big tiles.
6. **Small-M / latency-bound shapes** (decode, T≈13 frames): the count of launches dominates — fuse
   whole blocks (LN+QKV+RoPE, attn+O+residual, LN+FC1+act, FC2+residual) even if each kernel is "inefficient".
7. **Sequential chains** (RVQ stages): first one launch per stage, then a persistent kernel with a grid
   barrier (`tl.atomic_add` on a counter + spin) only if measured better.
8. **Integration**: replace `module.forward` (keep signature) or swap the nn.Module in `apply(model, ctx)`;
   count invocations for `report()`.

## Pitfalls
`tl.arange` needs power-of-2 sizes; masks on every load/store at edges; `num_stages>2` needs sm80+;
`triton.next_power_of_2`; int64 offsets for > 2^31 elements; the Triton cache dir (`TRITON_CACHE_DIR`)
should live under `.fast-kernel/` to keep experiments isolated; compile time is excluded from latency
but must not leak into timed runs (warm up every shape).

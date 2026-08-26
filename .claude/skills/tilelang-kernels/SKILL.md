---
name: tilelang-kernels
description: How to write TileLang (tile-ai) kernels for fast-kernel candidates - pipelined GEMM/attention/conv with explicit shared memory and fragments, target auto-selection, nvcc/NVRTC setup. Use when Triton tiles are not enough or the playbook suggests tilelang.
argument-hint: [kernel-kind]
allowed-tools: Bash(fast-kernel templates), Bash(fast-kernel probe), Read, Edit, Write, Glob, Grep
---

# TileLang kernels

Template: `fastkernel/backends/templates/tilelang_gemm.py` (bf16 GEMM, `T.Kernel` grid, `T.alloc_shared`,
`T.alloc_fragment`, `T.Pipelined(..., num_stages=3)`, `T.copy`, `T.gemm`, `tilelang.compile(func, out_idx=[...])`).

## Setup
- `from fastkernel.backends.cuda_cpp import ensure_cuda_home; ensure_cuda_home()` before `import tilelang`
  (points CUDA_HOME/PATH at the pip-installed nvcc when there is no system toolkit).
- `fast-kernel probe` reports whether the probe kernel compiled on this machine and the exact error if not.
  Typical fixes: install `nvidia-cuda-nvcc` wheel, newer `tilelang`, set `TILELANG_TARGET`/use the NVRTC
  backend. Record the outcome in KNOWLEDGE.md; fall back to Triton for that target rather than stalling.

## Recipe
1. Express the op as tiles: block_M/N/K, threads (128 = 4 warps typical), shared buffers for A/B tiles,
   a fragment accumulator, a pipelined K loop.
2. Fuse epilogues in the `T.copy(C_l, C[...])` step (apply bias/activation on the fragment first).
3. For attention: flash-style loop over KV tiles with online softmax in fragments; for conv: implicit GEMM
   indexing in the copy.
4. Shapes are compile-time: build one kernel per shape bucket and cache the compiled objects in a dict
   keyed by shape (compilation is excluded from timing but must happen in `apply()` / warm-up).
5. Verify with the same self-test discipline as Triton; compare TFLOPS against the Triton template before
   adopting (on some consumer GPUs Triton wins for small tiles).

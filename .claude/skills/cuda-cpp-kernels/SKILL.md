---
name: cuda-cpp-kernels
description: How to write CUDA C++ kernels for fast-kernel candidates via torch.utils.cpp_extension.load_inline with the pip-installed nvcc - fused epilogues, warp-level reductions, grid barriers / persistent kernels, wmma/mma tensor cores. Use when a DSL cannot express the kernel.
argument-hint: [kernel-kind]
allowed-tools: Bash(fast-kernel templates), Bash(fast-kernel probe), Read, Edit, Write, Glob, Grep
---

# CUDA C++ kernels

Template: `fastkernel/backends/templates/cuda_cpp_template.py` — `load_cuda_inline(name, cuda_src, cpp_src,
functions, campaign_root)` compiles into `.fast-kernel/build/<name>` with `-O3` for the measured
`TORCH_CUDA_ARCH_LIST` (set by `ensure_cuda_home()`). Fast-math is deliberately NOT forced — under the
strict policy a kernel needs exact `rsqrt`/`div`/`fmad` and no denormal flush. Pass
`extra_cuda_cflags=["--use_fast_math"]` yourself only when the numerics genuinely tolerate it.

## Recipe
1. Prototype: `TORCH_CHECK` dtypes/contiguity, launch on `at::cuda::getCurrentCUDAStream()`, one kernel
   per fused op; grid-stride loops for elementwise, warp shuffles (`__shfl_xor_sync`) for row reductions,
   shared-memory tiles with padding (`[32][33]`) for transposes/GEMM staging, `wmma`/`mma.sync` for tensor cores.
2. Persistent kernels: launch `num_SMs × k` blocks, loop over work items; a software grid barrier
   (atomic counter + `__threadfence()`) only when all blocks are co-resident (check occupancy).
3. Determinism: avoid `atomicAdd` on floats in strict policies; tree-reduce in shared memory.
4. Build once (torch caches by source hash); a changed source recompiles (~10-60 s, excluded from latency).
5. Verify with a self-test against torch at the gate tolerance; integrate by wrapping the extension
   function in `apply(model, ctx)`; `report()` counts calls.
Compile errors: read the first nvcc error, check `nvcc --version` vs torch's CUDA, and arch flags; the pip
wheel `nvidia-cuda-nvcc` provides nvcc without a system toolkit.

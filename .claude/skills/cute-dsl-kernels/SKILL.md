---
name: cute-dsl-kernels
description: How to write CuTe DSL (nvidia-cutlass-dsl) kernels in Python for fast-kernel candidates - layouts, @cute.kernel / @cute.jit, torch interop via from_dlpack, MMA/TMA atoms. Use when hand-scheduled GEMM/attention is needed.
argument-hint: [kernel-kind]
allowed-tools: Bash(fast-kernel templates), Bash(fast-kernel probe), Read, Edit, Write, Glob, Grep
---

# CuTe DSL kernels

Template: `fastkernel/backends/templates/cute_dsl_elementwise.py` (`@cute.kernel` device function,
`@cute.jit` launcher with `.launch(grid=..., block=...)`, `from_dlpack(torch_tensor)`).

## Recipe
1. `ensure_cuda_home()` then `import cutlass, cutlass.cute as cute`; `fast-kernel probe` shows if the probe
   compiled here (CuTe DSL needs a matching CUDA driver/toolkit; fix the environment before concluding).
2. Describe tensors with layouts (`cute.make_layout`, `cute.local_tile`, `cute.local_partition`) so that
   copies are vectorized and coalesced; prefer `cute.copy` with tiled copy atoms over scalar loops.
3. GEMM: tiled MMA atoms (`cute.nvgpu.warpgroup` / `mma_atom` for the measured architecture), shared
   memory staging with swizzled layouts, a K-pipeline with async copies (TMA on sm90+, cp.async earlier).
4. Compile once per shape bucket; keep the compiled callables in `apply()`; time only steady state.
5. Verify vs torch with the gate tolerance; compare with the Triton/TileLang versions and keep the fastest —
   measured on this GPU, not assumed from the architecture.

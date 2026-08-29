---
name: hotspot-analysis
description: How to read a fast-kernel profile (module attribution, kernel counts, GPU-busy ratio, roofline boundness) and turn it into the next hypothesis from the measured headroom. Use when deciding what to optimize next.
user-invocable: false
---

# Reading a profile

`profile.json` / `PLAN.md` contain: `wall_ms` (one call, no profiler), `gpu_busy_ms` (union of kernel
intervals), `gpu_busy_ratio`, `kernel_count`, `avg_kernel_us`, per-module rows (`path`, `class`, `gpu_us`,
`kernel_count`, `category`, `boundness`, `shapes`, FLOP/byte estimates, % of peak) and ranked `targets`
(module-class groups) with `fraction`, `sol_efficiency`, `headroom`, `kernel_count` and the tried/accepted counts.

## Rules

1. **GPU busy < 60 % of wall** → the workload is launch/Python bound. The first hypotheses are structural:
   CUDA graphs (whole call or per stage), fusion of elementwise chains into one CUDA kernel, removing the
   copies between them. Kernel count per call is the metric to drive down; expect wall ≈ launches × launch latency.
2. **Latency-bound module** (avg kernel < ~2.5 × launch latency) → fuse with neighbours; the per-kernel
   efficiency does not matter, the *count* does.
3. **Memory-bound** (arithmetic intensity < ridge) → fewer bytes: fuse to avoid intermediates, lower
   precision if the policy allows, channels-last / coalesced layouts, vectorized loads.
4. **Compute-bound** (intensity > ridge, % of peak low) → tensor cores via `wmma`/`mma.sync` (bf16/TF32 per
   policy), tile/block tuning, split-K when SMs are underfilled, deeper `cp.async` pipelining.
5. **Sequential chains** (RVQ stages, decode steps) → one persistent kernel or one CUDA graph for the chain.
6. **Headroom**: end-to-end headroom = share × (1 − SOL). A big share already at its roofline has little
   left to give; a low-SOL target is where real speed hides. The method is never prescribed — discover it.
   Re-rank after every keep — shares move.
7. **Technique matrix**: `techniques[].status` (untried / accepted / rejected / crash). Untried first,
   accepted can be iterated further, rejected only with a *new* reason, crash after fixing the cause.
8. Cross-check with `top_kernels`: cuDNN/cuBLAS kernel names, `elementwise_kernel`, `copy_`/`cat` counts
   reveal what the module rows hide (e.g. dozens of `_to_copy` casts = dtype churn).

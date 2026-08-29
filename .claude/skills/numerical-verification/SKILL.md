---
name: numerical-verification
description: Rules and techniques for keeping optimized kernels numerically faithful to the reference - accumulation dtypes, deterministic reductions, exact argmin with coarse+re-rank, tolerance policies, bisecting mismatches. Use when designing a kernel's numerics or debugging gate failures.
user-invocable: false
---

# Numerical verification

- The reference is fp32 with TF32 off (`torch.backends.cuda.matmul.allow_tf32=False`,
  `torch.backends.cudnn.allow_tf32=False`) unless the spec says otherwise (LLMs: bf16 reference).
- **strict** policy: discrete outputs (codes, tokens, classes) identical; floats allclose at the spec's
  tolerance; determinism bitwise. **tolerant**: statistical gates (code match ≥ 85 %, SNR, top-1 ≥ 97 %).
- Keep accumulators fp32; cast at the epilogue. Sum in a fixed order (tree/shared-memory reductions, no
  float atomics) for determinism.
- Exact argmin on tensor cores: fp16/bf16 coarse distances for all candidates, then exact fp32 distances
  for the top-k (k = 2..4) and pick the true minimum — identical codes at tensor-core speed. Ties: the
  reference's tie-break is "first index" (`argmin`); replicate it.
- Softmax/log-sum-exp online in fp32; RoPE in fp32 then cast; GELU tanh vs erf must match the reference
  variant; ELU/SiLU exactness (use `exp` not fast approximations under strict).
- Causal/streaming convs: reproduce the reference padding (left pad = kernel−1 for causal) and any
  `padding_total`/extra-padding logic exactly; check odd lengths (edge gates).
- CUDA graphs/compile: outputs are reused buffers — clone before comparing; RNG must not be in the path.
- Bisect: run reference and candidate side by side, hook module outputs (`register_forward_hook`),
  compare max-abs / rel error per module, find the first divergent module, then inspect strides,
  dtypes, masks, and boundary indices for that op.

## Replacing a vendor kernel bit-for-bit

Under `strict`, "correct" is not "close": your kernel must return the same bits as whatever cuBLAS /
cuDNN / ATen kernel the reference happens to dispatch to. Floating-point addition is not associative,
so this is a statement about **reduction order**, and the order is a property of that specific kernel.

- **Characterize it, do not guess.** Write a scratch script that reproduces the reference op alone and
  test candidate orderings against it with `torch.equal` (not `allclose` — allclose hides exactly the
  disagreement you are hunting). Sequential-K, split-K into fixed chunks, and interleaved splits all
  give different bits; find which one matches before writing the real kernel.
- **Verify the harness's own flags in every scratch script.** The reference runs with TF32 off; a
  scratch script that forgets `torch.backends.cudnn.allow_tf32 = False` and
  `torch.backends.cuda.matmul.allow_tf32 = False` compares your exact kernel against a *TF32*
  reference and blames your kernel for a mismatch it did not cause.
- **FMA contraction changes bits.** `a * b + c` fused into one FMA rounds once; separate multiply and
  add round twice. If the reference did not contract, block it: `__fmul_rn` / `__fadd_rn` (and
  `__fmaf_rn` where you *do* want the fused form). `-O3` alone lets nvcc contract freely, so state the
  intent per expression rather than relying on the flags.
- **Scaling by a power of two is exact**, so a factor of ±2^k can be folded into a precomputed
  constant without changing a single bit. Nothing else about a constant fold is safe.
- **Padding for parallelism is free accuracy-wise but not free of consequence**: vendor kernels often
  pad M to 128 and eat the wasted FLOPs because the padding buys occupancy. A hand-written kernel that
  "wastes nothing" can easily be slower for exactly that reason — check occupancy before concluding
  your arithmetic is the problem.
- **Layout traps that silently corrupt a supposedly exact kernel**: `reshape()` to the same shape
  returns the existing (possibly permuted, non-contiguous) view — call `.contiguous()` when a kernel
  assumes packing; broadcast operands such as `[1, T, D]` cos/sin tables need their batch stride
  forced to 0; a `register_forward_hook` that returns a value *replaces* the module output, so hooks
  used for observation must return `None`.
- Re-verify after every refactor with `torch.equal` on the isolated op. Gate-level allclose can pass
  while a kernel has quietly stopped being exact, and the next experiment inherits the drift.

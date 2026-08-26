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

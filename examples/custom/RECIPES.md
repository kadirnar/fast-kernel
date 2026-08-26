# Recipes for a custom torch module

1. `fast-kernel profile` first: if GPU busy < 60 % of wall time, start with `cuda-graphs` (whole forward,
   static shapes) or `torch-compile` (`reduce-overhead`).
2. Then the top target by Amdahl gain in PLAN.md: norms/activations/residual chains → fused Triton
   kernels; GEMMs → merged weights + epilogue fusion; convolutions → channels-last, cuDNN benchmark,
   implicit GEMM; attention → SDPA flash / fused kernel; reductions/argmin → fused kernels.
3. Re-profile after every keep; shares move.
Outputs must stay allclose to the fp32 reference (`gates.precision: strict`).

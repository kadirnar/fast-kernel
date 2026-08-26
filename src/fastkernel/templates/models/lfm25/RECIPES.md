# LFM2.5 recipes — ordered by expected payoff (strict policy: identical greedy tokens)

Architecture: 16 blocks = 10 `Lfm2ShortConv` (in_proj 3× → chunk B,C,x → B·x → causal depthwise conv1d
L=3 → C·y → out_proj) + 6 GQA attention blocks (32 heads / 8 KV heads, q/k RMSNorm, RoPE), SwiGLU MLP
(2048 → 12288), RMSNorm, vocab 65536, bf16. Primary workload: 64 greedy tokens after a 64-token prompt.

## 1. Static cache + CUDA-graph decode step  (`cuda-graphs`, `decode-step-fusion`)
Decode is a chain of GEMV-shaped launches per token: the GPU idles between them. Set
`model.generation_config.cache_implementation = "static"`, capture the per-token forward (fixed shapes,
static KV cache, `Graphed` per cache length bucket or `torch.compile(mode="reduce-overhead")` on
`model.forward`), keep the prefill eager. Expect the largest single win.

## 2. Fused RMSNorm into the next projection  (`fused-norm`)
`triton_rmsnorm.py` (row per program, fp32 statistics, optional residual add). Replace `operator_norm`,
`ffn_norm`, q/k norms; then fold the cast/quant of the following projection into it.

## 3. Merged gate/up projections with a fused silu·mul epilogue  (`weight-prepack`, `epilogue-fusion`)
`w1`/`w3` → one GEMM (`triton_matmul.py`, `act="silu"` on the gate half, multiply by the up half in the
epilogue); `triton_fused_silu_mul.py` is the fallback when the GEMM stays on cuBLAS.

## 4. One fused ShortConv kernel  (`fused-elementwise`, `implicit-gemm-conv`)
`triton_causal_conv1d.py` computes the depthwise causal conv with the C·y gate fused; add the B·x
pre-gate and the cache update for the single-token decode path. Keep the `causal_conv1d_fn`/`_update`
semantics exactly (left padding L−1, fp32 accumulation).

## 5. Prefill  (`fused-attention`, `block-tuning`, `hub-kernel`)
Prefill is GEMM/attention bound: SDPA flash backend (or `kernels-community/flash-attn2`), bf16 GEMM tile
tuning, fused MLP epilogues. Measure `prefill` separately — it is a secondary workload.

## Tokens must stay identical
bf16 reference; fp32 accumulation inside every kernel; deterministic reductions; the harness checks
greedy tokens (exact), top-1 (≥ 99.5 %) and top-5 overlap on the last 8 prefill positions.

# Mimi recipes — ordered by measured payoff (RTX 5070 Ti, transformers 5.16, torch 2.13, strict policy)

Each recipe is one experiment: implement under `candidate/`, then
`uv run fast-kernel eval -m "..." --technique <id> --target <id>`. Numbers are real measurements from
this harness; re-measure on your GPU rather than assuming them.

## Measured so far (1 s of 24 kHz audio, batch 1, identical codes throughout)

| # | recipe | result |
|--:|---|---|
| 0 | stock `MimiModel` fp32 | 18.86 ms, 1641 launches, GPU busy 23 % |
| 3 | CUDA graphs for `encode`/`decode` + host-side conv padding math | 5.14 ms (3.67×), 1436 launches, GPU busy 97 % |
| 4 | Triton fp32 implicit-GEMM `Conv1d` (deterministic split-K) for weight-bound layers | 3.89 ms (4.84×), 1282 launches |
| 5 | Triton fp32 fused RVQ codebook search (per-stage exact distances, first-index argmin) | 3.14 ms (6.0×), 768 launches |

## 1. CUDA graphs around encode / decode  (`cuda-graphs`) — done in #3

Capture blockers found and fixed: `MimiConv1d._get_extra_padding_for_conv1d` returns a CUDA scalar that
`F.pad` turns into `.item()` (host sync → capture invalidated); `padding_left/right` are *meta* tensors after
`from_pretrained` (never `int()` them — derive from `int(module.padding_total)`). The candidate replaces the
conv forward with Python-int padding arithmetic, then wraps `encode`/`decode` in one `Graphed` per input
shape and rebuilds `MimiEncoderOutput` / `MimiDecoderOutput` around cloned outputs.

## 2. Convolutions as implicit GEMM  (`implicit-gemm-conv`, `split-k`) — done in #4 for Conv1d

cuDNN's `precomputed_convolve_sgemm` is 5–10× off the weight-bandwidth floor for Mimi's small-T convs
(T_out ≤ 32, weights 4–33 MB). fp32 `tl.dot(..., input_precision="ieee")`, split-K into a partial buffer
summed in fixed order (deterministic, no atomics). **Next:** the same for `ConvTranspose1d` (decoder
upsampling, ~5 % share) and the stride-1 residual-block convs that were left on cuDNN; then fuse the
ELU that follows each conv into the epilogue (`epilogue-fusion`).

## 3. Residual vector quantizer  (`fused-quantizer`) — done in #5, one launch per stage

Per stage: exact fp32 `sum_d (r - e)^2` for all 2048 codes, block-wise min with first-index tie-break
(matches `torch.argmin`), then `r -= embed[code]`. **Next:** fuse the 32 sequential stages into one
persistent kernel with a grid barrier — measured 2× *slower* on this GPU in a hand-written version
(barrier ≈ launch latency), so try it only after cheaper ideas; the alternative is a tensor-core coarse
pass (bf16/fp16 −2x·c + |c|²) followed by an exact fp32 re-rank of the top-4 — identical codes, faster
distances (`triton_codebook_argmin.py` template).

## 4. Transformer blocks  (`fused-norm`, `fused-attention`, `epilogue-fusion`) — untried, ~6 % + 15 % share

T = 13 frames for 1 s: everything is latency bound; the count of launches matters, not FLOPs.
Per layer, replace ~25 launches by 4: LayerNorm+QKV(+RoPE) → attention+O-proj+residual(+LayerScale) →
LayerNorm+FC1+GELU → FC2(+residual). Start with the two LayerNorm fusions (`triton_rmsnorm.py` shows the
row-per-program pattern; LayerNorm adds the mean), then merge Q/K/V weights into one GEMM
(`weight-prepack`) with a `triton_matmul.py` epilogue.

## 5. Kernel-count reduction inside the graphs  (`kernel-count-reduction`) — untried

768 launches remain in the graph. `fast-kernel profile` → top kernels: `elementwise_kernel` copies,
`_to_copy` casts, `cat`/`pad`, ELU. Each removed launch is worth ~4 µs of GPU time here; fuse ELU +
residual + LayerScale into the producing kernel's epilogue, pre-pad inputs once, keep channels-last
buffers so no transposes are launched.

## 6. Precision (human decision only)

bf16 tensor cores in the quantizer/GEMMs give another ~2× in hand-written implementations but flip
near-tie codes (80–93 % identical codes, reconstruction quality unchanged). The strict policy forbids it;
if the human sets `gates.precision: tolerant` in GOAL.md, the gates become decode SNR ≥ 40 dB,
reconstruction within 0.25 dB, ≥ 80 % identical codes.

## Did not pay off here (measured in fast-mimi on the same GPU; re-measure before trusting)

persistent single-launch transformer (2× slower than 4 kernels); split-K "one tile per CTA" transformer;
INT8 weights (latency-bound, no gain); tf32x3 exact RVQ; fusing conv0 with the first residual block.

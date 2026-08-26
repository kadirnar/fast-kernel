# LFM2-Audio recipes — ordered by expected payoff (strict policy: identical text/audio tokens)

Architecture: LFM2.5 1.2B backbone + FastConformer encoder (115M) + Mimi-compatible 8-codebook decoder
driven by a 6-layer depthformer. Primary workload: TTS of one sentence (greedy-equivalent sampling);
secondary: ASR of 2 s of audio.

## 1. CUDA graphs on the per-step forwards  (`cuda-graphs`, `decode-step-fusion`)
Each generated frame runs the backbone once and the depthformer 8× (one per codebook): dozens of
GEMV-shaped launches. Capture the backbone step with a static cache and the depthformer loop as one
graph per step; keep shapes fixed with cache-length buckets.

## 2. Backbone kernels shared with the LFM2.5 campaign  (`fused-norm`, `epilogue-fusion`, ShortConv)
Copy the kept kernels from `campaigns/lfm25/candidate/kernels/` — same modules, same shapes.

## 3. Mimi decoder  (`implicit-gemm-conv`, `fused-quantizer`)
`processor.decode` runs the Mimi decoder: copy the kept convolution and codebook kernels from
`campaigns/mimi/candidate/kernels/`.

## 4. FastConformer encoder (ASR)  (`fused-attention`, `implicit-gemm-conv`)
Conv subsampling + relative-position attention + depthwise conv modules: fused attention with the
rel-pos bias, channels-last convs with fused activations.

## Tokens must stay identical
Sampling is seeded and greedy-equivalent; the harness compares text tokens, audio codes and the decoded
waveform (SNR ≥ 40 dB). Any kernel must keep fp32 accumulation and reference tie-breaks.

# YOLO recipes — ordered by expected payoff (strict policy: same boxes/confidences/classes)

Fused `DetectionModel` (Conv+BN folded), end-to-end NMS-free head, output (B, 300, 6). Primary workload:
batch 1 @ 640²; secondary: batch 8.

## 1. CUDA graph of the whole forward  (`cuda-graphs`)
Batch 1 launches hundreds of small conv/act/concat kernels; one graph per input shape removes the
launch gaps. Static input buffer, outputs cloned.

## 2. Layout and cuDNN settings  (`memory-access`, `weight-prepack`)
`channels_last` for the whole model, weights pre-converted once in `apply()`; `cudnn.benchmark` on the
static shape (tune in warm-up, not in timed runs).

## 3. Conv + SiLU epilogue fusion  (`epilogue-fusion`, `torch-compile`)
`torch.compile(mode="max-autotune-no-cudagraphs")` on the backbone fuses activations into inductor
kernels; or a Triton implicit-GEMM conv2d with the SiLU epilogue for the layers that dominate the profile.

## 4. Copy elimination  (`kernel-count-reduction`)
`Concat`/`Upsample` materialise copies; pre-allocate and write in place, fuse nearest-upsample into the
consuming conv's loads.

## 5. Batch 8  (`block-tuning`, `dtype-policy` — human decision)
At batch 8 the convs are compute bound: tile tuning; fp16/bf16 convs only if the human sets the tolerant
policy (boxes within 2 px, confidences within 1e-2).

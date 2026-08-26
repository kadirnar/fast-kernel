# Mimi optimization recipes (ordered by measured payoff)

These are concrete starting points, not instructions to skip measurement. Every step is one
experiment: implement in `candidate/`, run `fast-kernel eval -m "..." --technique <id> --target <id>`.

## 1. CUDA graphs around encode / decode  (`cuda-graphs`, expected 5-10x on the stock path)

```python
from fastkernel.backends.graphs import Graphed
import torch

def apply(model, ctx):
    enc = model.encode
    dec = model.decode
    cache = {}
    def graphed_encode(audio, padding_mask=None, **kw):
        key = ("enc", tuple(audio.shape))
        if key not in cache:
            cache[key] = Graphed(lambda a, m: enc(a, m).audio_codes, (audio, padding_mask))
        codes = cache[key](audio, padding_mask)
        return type(enc(audio, padding_mask))(audio_codes=codes)   # or a tiny namespace with .audio_codes
    ...
    model.encode = graphed_encode
    return model
```
Pitfalls: transformers' outputs are ModelOutput dataclasses -- build the same type (or an object with
the same attribute) around the graph's static buffers; clone outputs before they are reused by the next
call inside `roundtrip`; the padding mask is constant so it can be baked into the graph.
Capture blockers measured here: `MimiConv1d._get_extra_padding_for_conv1d` returns a CUDA scalar that
`F.pad` turns into `.item()` (host sync -> capture invalidated); its `padding_left/right` attributes are
meta tensors after `from_pretrained`. Patch the conv forward to do the padding arithmetic with Python
ints (`int(module.kernel_size)`, `int(module.stride)`, `int(module.padding_total)`) first.

## 2. Fused codebook search  (`fused-quantizer`, target `MimiEuclideanCodebook`)

`quantize(x)`: `dist = -2 x.c + |c|^2` per stage, `argmin`, `residual -= c[idx]`. Use
`fastkernel/backends/templates/triton_codebook_argmin.py` as the kernel; replace
`MimiEuclideanCodebook.quantize` for all 32 layers with one call each first, then fuse the 32 stages into
one persistent kernel (grid barrier) once the per-stage version is correct.
Strict policy needs exact fp32 distances: a fp16/bf16 coarse pass + exact fp32 re-rank of the top-k
(k=2..4) keeps codes identical while using tensor cores.

## 3. Transformer blocks  (`fused-attention`, `fused-norm`, `epilogue-fusion`)

For 1 s of audio the transformer sees T=13 frames: everything is latency bound. Fuse per layer:
LN+QKV(+RoPE) -> attention+O-proj+residual(+LayerScale) -> LN+FC1+GELU -> FC2(+residual), i.e. 4 launches
per layer instead of ~25. Split-K FC2 (intermediate 2048->512) helps at these tiny M.

## 4. SEANet convolutions  (`implicit-gemm-conv`, `epilogue-fusion`)

Causal Conv1d / ConvTranspose1d as implicit GEMM (channels-last), ELU fused into the epilogue,
autotuned per layer shape (cache under `candidate/tuned/`). Fold LayerScale/residual adds.

## What did not pay off on an RTX 5070 Ti (re-measure elsewhere)
persistent single-launch transformer; split-K one-tile-per-CTA transformer; INT8 weights;
tf32x3 exact RVQ; fusing conv0 with the first residual block.

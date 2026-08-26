"""LiquidAI LFM2.5 (transformers Lfm2ForCausalLM): hybrid short-conv + GQA attention decoder.

model_args.variant selects the checkpoint: 1.2B-Instruct (default), 1.2B-Base, 1.2B-Thinking, 350M, 230M.
"""
from __future__ import annotations

from .hf_causal_lm import HFCausalLMSpec

VARIANTS = {
    "1.2b": "LiquidAI/LFM2.5-1.2B-Instruct",
    "1.2b-instruct": "LiquidAI/LFM2.5-1.2B-Instruct",
    "1.2b-base": "LiquidAI/LFM2.5-1.2B-Base",
    "1.2b-thinking": "LiquidAI/LFM2.5-1.2B-Thinking",
    "350m": "LiquidAI/LFM2.5-350M",
    "230m": "LiquidAI/LFM2.5-230M",
}


class LFM25Spec(HFCausalLMSpec):
    name = "lfm25"
    display_name = "LiquidAI LFM2.5 (Lfm2ForCausalLM)"
    default_hub_id = "LiquidAI/LFM2.5-1.2B-Instruct"
    notes = """\
LFM2.5-1.2B: 16 blocks = 10 double-gated short-convolution blocks (Lfm2ShortConv: in_proj to 3x width,
chunk into B, C, x; B*x -> causal depthwise conv1d with L_cache=3 -> C*y -> out_proj) + 6 GQA attention
blocks (32 heads, 8 KV heads, q/k RMSNorm, RoPE), SwiGLU MLP (hidden 2048, intermediate 12288), RMSNorm,
vocab 65536, bf16. Decode (the primary workload) is a chain of GEMV-shaped projections per token: it is
launch/overhead bound at batch 1, so a static-KV-cache CUDA-graph decode step (or torch.compile
mode=reduce-overhead) is the first thing to try, followed by fusing RMSNorm into the next projection,
merging w1/w3 with a fused silu*mul epilogue, and a fused gating+depthwise-conv kernel for ShortConv.
Prefill is GEMM bound: bf16 tensor cores, fused attention (SDPA flash), fused MLP epilogues.
"""

    @property
    def hub_id(self) -> str:  # type: ignore[override]
        if self.args.get("hub_id"):
            return str(self.args["hub_id"])
        variant = str(self.args.get("variant", "1.2b")).lower()
        return VARIANTS.get(variant, self.default_hub_id)

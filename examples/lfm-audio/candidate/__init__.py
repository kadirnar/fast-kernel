"""Candidate for the LFM2-Audio campaign -- THE ONLY CODE THE AGENT EDITS.

apply(model, ctx) -> model
    `model` is the liquid_audio `LFM2AudioModel` (eval, on ctx.device). Return the optimized model; it must keep
    `generate_sequential(**chat_state, max_new_tokens=..., audio_temperature=..., audio_top_k=...)` and
    `generate_interleaved(...)` working. Typical targets: the Transformers `Lfm2` backbone modules inside the
    model (layers[i].conv / self_attn / feed_forward / operator_norm), the depthformer, the FastConformer
    encoder, and the Mimi decoder used by the processor.

report() -> dict   (optional evidence that the optimized path executed)
"""
from __future__ import annotations

_STATE = {"active": False, "kernels": [], "invocations": 0}


def apply(model, ctx):
    ctx.log("identity candidate: reference path unchanged")
    return model


def report() -> dict:
    return dict(_STATE)

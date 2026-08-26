"""Candidate for the LFM2.5 campaign -- THE ONLY CODE THE AGENT EDITS.

apply(model, ctx) -> model
    `model` is the reference `Lfm2ForCausalLM` (bf16, eval, on ctx.device). Return the optimized model. You
    may replace `model.model.layers[i].conv` / `.self_attn` / `.feed_forward` / `.operator_norm` modules,
    monkeypatch forwards, swap attention backends, pre-merge weights, or wrap `model.generate` / `model.forward`
    (static cache + CUDA graphs). `generate(input_ids, attention_mask, max_new_tokens, do_sample=False)` and
    `forward(input_ids, attention_mask, use_cache=False).logits` must keep working.

report() -> dict   (optional evidence that the optimized path executed)
"""
from __future__ import annotations

_STATE = {"active": False, "kernels": [], "invocations": 0}


def apply(model, ctx):
    ctx.log("identity candidate: reference path unchanged")
    return model


def report() -> dict:
    return dict(_STATE)

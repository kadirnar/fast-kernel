"""Candidate for the Mimi campaign -- THE ONLY CODE THE AGENT EDITS.

Contract
--------
apply(model, ctx) -> model
    Receives the freshly loaded reference `transformers.MimiModel` (fp32, eval, on ctx.device) and returns
    the optimized model object. It may: replace submodules, monkeypatch forward methods, wrap
    `model.encode` / `model.decode` (e.g. with CUDA graphs), pre-pack weights, register Triton/TileLang/
    CuTe/CUDA C++ kernels from `candidate/kernels/`. The returned object must keep the public API:
    `encode(audio, padding_mask).audio_codes` and `decode(codes, padding_mask).audio_values`.

report() -> dict   (optional)
    Evidence that the optimized path really executed, e.g. {"active": True, "kernels": [...], "invocations": n}.

Notes
-----
- ctx.strict tells you whether the gate policy is strict (exact codes). ctx.capabilities holds the probe
  results (which backends compile here). ctx.log("...") lines end up in the experiment record.
- Keep fp32 accumulation for the residual stream and the quantizer distances; bf16 weights are fine for
  the GEMMs in tolerant mode.
- Every experiment builds on the accepted incumbent; small, focused changes measure best.
"""
from __future__ import annotations

_STATE = {"active": False, "kernels": [], "invocations": 0}


def apply(model, ctx):
    ctx.log("identity candidate: reference path unchanged")
    return model


def report() -> dict:
    return dict(_STATE)

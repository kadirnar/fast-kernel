"""Candidate for the YOLO campaign -- THE ONLY CODE THE AGENT EDITS.

apply(model, ctx) -> model
    `model` is the fused Ultralytics `DetectionModel` (fp32, eval, on ctx.device). Return the optimized module;
    `model(images)` must still return the (B, 300, 6) predictions as its first output.
"""
from __future__ import annotations

_STATE = {"active": False, "kernels": [], "invocations": 0}


def apply(model, ctx):
    ctx.log("identity candidate: reference path unchanged")
    return model


def report() -> dict:
    return dict(_STATE)

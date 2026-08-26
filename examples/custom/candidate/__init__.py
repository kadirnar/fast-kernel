"""Candidate -- THE ONLY CODE THE AGENT EDITS.  apply(model, ctx) -> model ; report() -> dict (optional)."""
from __future__ import annotations

_STATE = {"active": False, "kernels": [], "invocations": 0}


def apply(model, ctx):
    ctx.log("identity candidate: reference path unchanged")
    return model


def report() -> dict:
    return dict(_STATE)

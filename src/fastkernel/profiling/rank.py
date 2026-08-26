"""Amdahl ranking of hotspot targets and technique recommendation.

Targets are *groups* of module instances that share a class and category (e.g. all 16
`Lfm2ShortConv` blocks) because one kernel fixes all of them. The score is the end-to-end gain
Amdahl's law promises for the group's fraction and the playbook's expected speedup:

    gain = fraction * (1 - 1 / expected_speedup)

A launch-bound workload additionally gets a whole-workload target (`launch-bound`) whose fraction
is the GPU idle share of wall time. Prior experiments (technique matrix) demote what was already
tried and rejected, and mark what was accepted.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from ..playbook import expected_speedup, techniques_for
from .classify import classify_module


def _target_id(key: str) -> str:
    return "t_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def build_targets(profile: dict[str, Any], device: dict[str, Any], history: list[dict[str, Any]] | None = None,
                  hints: list[dict[str, Any]] | None = None, max_targets: int = 24) -> list[dict[str, Any]]:
    history = history or []
    hints = hints or []
    modules = [classify_module(m, device) for m in profile.get("modules", [])]
    total_gpu_us = sum(m["gpu_us"] for m in modules) or 1.0
    wall_ms = float(profile.get("wall_ms") or 0.0)
    gpu_busy_ms = float(profile.get("gpu_busy_ms") or 0.0)

    groups: dict[str, dict[str, Any]] = {}
    for mod in modules:
        cls = mod.get("class") or (mod["path"].rsplit(".", 1)[-1] if mod["path"] not in ("<unattributed>", "<root>") else "module")
        if mod["path"] == "<unattributed>":
            key = f"unattributed::{mod['category']}"
        else:
            key = f"{cls}::{mod['category']}"
        grp = groups.setdefault(key, {"key": key, "class": cls, "category": mod["category"],
                                      "gpu_us": 0.0, "kernel_count": 0, "instances": [], "boundness": defaultdict(float),
                                      "flops": 0.0, "bytes": 0.0})
        grp["gpu_us"] += mod["gpu_us"]
        grp["kernel_count"] += mod["kernel_count"]
        grp["flops"] += mod["flops"]
        grp["bytes"] += mod["bytes"]
        grp["boundness"][mod["boundness"]] += mod["gpu_us"]
        if len(grp["instances"]) < 12:
            grp["instances"].append({"path": mod["path"], "gpu_us": mod["gpu_us"], "kernel_count": mod["kernel_count"],
                                     "shapes": _compact_shapes(mod.get("shapes") or {}), "boundness": mod["boundness"],
                                     "avg_kernel_us": round(mod["avg_kernel_us"], 2)})
    hint_by_symbol = {h.get("symbol", ""): h for h in hints}
    targets: list[dict[str, Any]] = []
    for grp in groups.values():
        bound = max(grp["boundness"].items(), key=lambda kv: kv[1])[0] if grp["boundness"] else "memory"
        fraction = grp["gpu_us"] / total_gpu_us
        category = grp["category"]
        hint = hint_by_symbol.get(grp["class"])
        if hint and hint.get("category"):
            category = hint["category"]
        exp = expected_speedup(category, bound)
        gain = fraction * (1 - 1 / exp)
        targets.append({
            "id": _target_id(grp["key"]), "title": f"{grp['class']} ({category}, {bound}-bound)", "class": grp["class"],
            "category": category, "boundness": bound, "fraction": fraction, "gpu_us": round(grp["gpu_us"], 2),
            "kernel_count": grp["kernel_count"], "instances": grp["instances"], "instance_count": len(grp["instances"]) if len(grp["instances"]) < 12 else sum(1 for m in modules if m["category"] == grp["category"] and (m.get("class") or m["path"].rsplit(".", 1)[-1]) == grp["class"]),
            "expected_speedup": exp, "amdahl_gain": gain, "scope": "module-group",
            "flops": grp["flops"], "bytes": grp["bytes"], "hint": (hint or {}).get("note", ""),
        })
    # whole-workload launch-bound target
    if wall_ms > 0:
        idle_fraction = max(0.0, 1.0 - gpu_busy_ms / wall_ms)
        if idle_fraction > 0.15:
            exp = expected_speedup("launch-bound", "latency")
            targets.append({
                "id": _target_id("workload::launch-bound"), "title": "Whole workload: launch/Python-overhead bound",
                "class": "<workload>", "category": "launch-bound", "boundness": "latency", "fraction": idle_fraction,
                "gpu_us": round((wall_ms - gpu_busy_ms) * 1e3, 1), "kernel_count": profile.get("kernel_count", 0),
                "instances": [], "instance_count": 1, "expected_speedup": exp,
                "amdahl_gain": idle_fraction * (1 - 1 / exp), "scope": "workload",
                "hint": f"GPU busy only {100 * (1 - idle_fraction):.0f}% of wall time across {profile.get('kernel_count', 0)} launches "
                        f"(avg kernel {profile.get('avg_kernel_us') or 0:.1f} us). CUDA graphs / fusion / torch.compile apply.",
            })
    matrix = technique_matrix(history)
    for target in targets:
        target["techniques"] = recommend(target, matrix)
        tried = [t for t in target["techniques"] if t["status"] != "untried"]
        target["attempts"] = len(tried)
        target["accepted"] = sum(1 for t in tried if t["status"] == "accepted")
        # demote targets whose best techniques all failed; never to zero (ideas change)
        rejected_share = (sum(1 for t in tried if t["status"] == "rejected") / max(1, len(target["techniques"])))
        target["score"] = target["amdahl_gain"] * (1.0 - 0.5 * rejected_share)
    targets.sort(key=lambda t: -t["score"])
    for rank, target in enumerate(targets, 1):
        target["rank"] = rank
    return targets[:max_targets]


def _compact_shapes(shapes: dict[str, Any]) -> dict[str, Any]:
    if not shapes:
        return {}
    return {"inputs": shapes.get("inputs"), "output": shapes.get("output"), "params": shapes.get("params"),
            "attrs": shapes.get("attrs"), "calls": shapes.get("calls")}


def technique_matrix(history: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """(target_id, technique_id) -> best status seen so far (accepted > rejected > crash)."""
    order = {"accepted": 3, "rejected": 2, "crash": 1}
    matrix: dict[tuple[str, str], str] = {}
    for exp in history:
        target = exp.get("target") or ""
        for tech in exp.get("techniques") or []:
            key = (target, tech)
            status = exp.get("status", "")
            if status == "keep":
                status = "accepted"
            elif status == "discard":
                status = "rejected"
            if status not in order:
                continue
            if order[status] > order.get(matrix.get(key, ""), 0):
                matrix[key] = status
    return matrix


def recommend(target: dict[str, Any], matrix: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    out = []
    for tech in techniques_for(target["category"], target["boundness"]):
        status = matrix.get((target["id"], tech.id), "untried")
        out.append({"id": tech.id, "title": tech.title, "tier": tech.tier, "backends": list(tech.backends),
                    "expected_speedup": tech.expected_speedup, "risk": tech.risk, "skill": tech.skill,
                    "status": status, "summary": tech.summary})
    # untried first (by tier), then accepted (can be iterated), then rejected
    out.sort(key=lambda t: ({"untried": 0, "accepted": 1, "crash": 2, "rejected": 3}[t["status"]], t["tier"], -t["expected_speedup"]))
    return out

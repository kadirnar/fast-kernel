"""Five-stage correctness harness (AutoKernel's stages generalised to whole models):

1. smoke        the primary workload runs and its outputs are finite / well formed
2. shapes       every workload's output structure and shapes match the reference
3. numerical    the model spec's comparison (exact codes, allclose, top-1 agreement, SNR ...) passes
4. determinism  running the candidate twice gives identical (or, by policy, tolerant) outputs
5. edge         edge-case workloads (short/odd/batched inputs) match the reference

A failing or missing mandatory gate can never become a performance success.
"""
from __future__ import annotations

import time
import traceback
from typing import Any

from ..models.spec import GateCheck, ModelSpec, Workload, _flatten


def _finite_checks(outputs: Any, prefix: str) -> list[GateCheck]:
    import torch
    checks = []
    leaves = _flatten(outputs, prefix)
    if not leaves:
        return [GateCheck(f"{prefix}/outputs", False, detail="no outputs produced")]
    for name, value in leaves:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                ok = bool(torch.isfinite(value).all().item())
                checks.append(GateCheck(f"{name}/finite", ok, detail="" if ok else "NaN/Inf"))
            else:
                checks.append(GateCheck(f"{name}/present", True, detail=f"{tuple(value.shape)} {value.dtype}"))
    return checks


def run_gates(spec: ModelSpec, candidate: Any, workloads: list[Workload], reference_outputs: dict[str, Any],
              inputs: dict[str, dict[str, Any]], edge_workloads: list[Workload], edge_reference: dict[str, Any],
              edge_inputs: dict[str, dict[str, Any]], stages: list[str], run_workload) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": True, "stages": {}, "checks_total": 0, "checks_failed": 0}
    primary = next((w for w in workloads if w.primary), workloads[0])
    outputs: dict[str, Any] = {}

    def stage(name: str, fn):
        if name not in stages:
            result["stages"][name] = {"passed": True, "skipped": True, "checks": []}
            return
        started = time.perf_counter()
        try:
            checks = fn()
            crashed = None
        except Exception as exc:  # noqa: BLE001
            crashed = traceback.format_exc()
            checks = [GateCheck(f"{name}/exception", False, detail=f"{type(exc).__name__}: {str(exc)[:300]}\n" + crashed[-1200:])]
        passed = all(c.passed for c in checks)
        result["stages"][name] = {"passed": passed, "skipped": False, "seconds": round(time.perf_counter() - started, 3),
                                  "checks": [c.to_dict() for c in checks], "crashed": bool(crashed)}
        result["checks_total"] += len(checks)
        result["checks_failed"] += sum(1 for c in checks if not c.passed)
        if not passed:
            result["passed"] = False

    def smoke():
        out = run_workload(candidate, primary, inputs[primary.name])
        outputs[primary.name] = out
        return _finite_checks(out, primary.name)

    def shapes():
        checks = []
        for w in workloads:
            if w.name not in outputs:
                outputs[w.name] = run_workload(candidate, w, inputs[w.name])
            ref_leaves = _flatten(reference_outputs[w.name], w.name)
            cand_leaves = _flatten(outputs[w.name], w.name)
            if len(ref_leaves) != len(cand_leaves):
                checks.append(GateCheck(f"{w.name}/structure", False, detail=f"{len(cand_leaves)} leaves vs {len(ref_leaves)}"))
                continue
            for (name, ref), (_, cand) in zip(ref_leaves, cand_leaves, strict=True):
                if hasattr(ref, "shape") and hasattr(cand, "shape"):
                    same = tuple(ref.shape) == tuple(cand.shape)
                    checks.append(GateCheck(f"{name}/shape", same, detail=f"{tuple(cand.shape)}" + ("" if same else f" != {tuple(ref.shape)}")))
        return checks

    def numerical():
        checks = []
        for w in workloads:
            if w.name not in outputs:
                outputs[w.name] = run_workload(candidate, w, inputs[w.name])
            if hasattr(spec, "observe_inputs"):
                spec.observe_inputs(w, inputs[w.name])
            checks.extend(spec.compare(w, reference_outputs[w.name], outputs[w.name]))
        return checks

    def determinism():
        first = outputs.get(primary.name) or run_workload(candidate, primary, inputs[primary.name])
        second = run_workload(candidate, primary, inputs[primary.name])
        return spec.compare_determinism(primary, first, second)

    def edge():
        checks = []
        for w in edge_workloads:
            out = run_workload(candidate, w, edge_inputs[w.name])
            if hasattr(spec, "observe_inputs"):
                spec.observe_inputs(w, edge_inputs[w.name])
            checks.extend(_finite_checks(out, w.name))
            checks.extend(spec.compare(w, edge_reference[w.name], out))
        return checks or [GateCheck("edge/none", True, detail="no edge workloads defined")]

    stage("smoke", smoke)
    if result["stages"].get("smoke", {}).get("passed", True):
        stage("shapes", shapes)
        stage("numerical", numerical)
        stage("determinism", determinism)
        stage("edge", edge)
    else:
        for name in ("shapes", "numerical", "determinism", "edge"):
            result["stages"][name] = {"passed": False, "skipped": True, "checks": [], "reason": "smoke failed"}
        result["passed"] = False
    passed_stages = sum(1 for s in result["stages"].values() if s.get("passed"))
    result["summary"] = f"{passed_stages}/{len(result['stages'])} stages passed, {result['checks_failed']} failed checks"
    result["failed_checks"] = [c for s in result["stages"].values() for c in s.get("checks", []) if not c["passed"]][:25]
    return result

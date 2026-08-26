"""Subprocess entry point that evaluates the *current* candidate tree of a campaign.

    python -m fastkernel.harness.run --campaign <root> --out <dir> [--no-profile] [--noise-check]

Writes gates.json, metrics.json, profile.json, candidate_report.json into --out and prints progress.
Exit code 0: evaluated (gates may have failed); 2: the candidate crashed; 3: harness/reference error.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..backends.base import device_capabilities
from ..backends.cuda_cpp import ensure_cuda_home
from ..config import load_goal
from ..models.spec import CandidateContext, candidate_report, load_spec
from ..profiling.rank import build_targets
from ..profiling.trace import profile_workload
from ..util import read_json, write_json
from .bench import peak_memory_mb, reset_peak_memory, time_callable
from .gates import run_gates

SEED = 20260826


def log(message: str) -> None:
    print(f"[harness {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _setup_torch(deterministic_tf32: bool = True):
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        if deterministic_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch


def _run_workload(model, workload, inputs):
    import torch
    with torch.inference_mode():
        out = workload.run(model, inputs)
    return _materialize(out)


def _materialize(tree: Any) -> Any:
    import torch
    if isinstance(tree, torch.Tensor):
        return tree.detach().clone()
    if isinstance(tree, dict):
        return {k: _materialize(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_materialize(v) for v in tree)
    if hasattr(tree, "items") and not isinstance(tree, (str, bytes)):
        try:
            return {k: _materialize(v) for k, v in tree.items() if v is not None}
        except Exception:  # noqa: BLE001
            return tree
    return tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--profile-only", action="store_true", help="skip gates and benchmark; only profile")
    parser.add_argument("--noise-check", action="store_true", help="benchmark the primary workload twice to estimate noise")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--history", default=None, help="JSON file with prior experiment records (technique matrix)")
    parser.add_argument("--workloads", default=None, help="comma separated subset")
    args = parser.parse_args(argv)

    root = Path(args.campaign).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    ensure_cuda_home()
    goal = load_goal(root / "GOAL.md")
    torch = _setup_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caps_path = root / "capabilities.json"
    caps = read_json(caps_path, None) or {}
    if not caps.get("device_info"):
        log("probing device capabilities (first run; `fast-kernel probe` also records backends)")
        caps["device_info"] = device_capabilities(microbench=True)
        write_json(caps_path, caps)
    device_info = caps.get("device_info", caps)

    started = time.perf_counter()
    result: dict[str, Any] = {"started_at": time.time(), "device": device_info.get("name"), "goal": goal.to_dict()}
    try:
        spec = load_spec(root, goal.model, goal.model_args, goal.gates)
        workloads = spec.workloads()
        if args.workloads:
            wanted = set(args.workloads.split(","))
            workloads = [w for w in workloads if w.name in wanted or w.primary]
        elif goal.workloads:
            wanted = set(goal.workloads)
            workloads = [w for w in workloads if w.name in wanted or w.primary]
        primary = spec.primary_workload(goal.primary_workload)
        for w in workloads:
            w.primary = w.name == primary.name
        edge_workloads = spec.edge_workloads() if "edge" in goal.gates.stages else []
        log(f"spec={spec.name} workloads={[w.name for w in workloads]} primary={primary.name} edge={[w.name for w in edge_workloads]}")

        # ---- reference --------------------------------------------------------------------
        log("loading reference model")
        reference = spec.load_reference()
        inputs = {w.name: w.make_inputs(device, SEED) for w in workloads}
        edge_inputs = {w.name: w.make_inputs(device, SEED + 1) for w in edge_workloads}
        log("computing reference outputs")
        reference_outputs = {w.name: _run_workload(reference, w, inputs[w.name]) for w in workloads}
        edge_reference = {w.name: _run_workload(reference, w, edge_inputs[w.name]) for w in edge_workloads}
        ref_bench: dict[str, Any] = {}
        if args.noise_check and not args.no_bench:
            log("reference timing (noise estimate)")
            fn = lambda m=reference: primary.run(m, inputs[primary.name])  # noqa: E731
            a = time_callable(fn, warmup=args.warmup or goal.bench.warmup, repeats=args.repeats or goal.bench.repeats,
                              ramp_seconds=goal.bench.ramp_seconds)
            b = time_callable(fn, warmup=2, repeats=args.repeats or goal.bench.repeats, ramp_seconds=0.2)
            noise = abs(a["median_ms"] - b["median_ms"]) / ((a["median_ms"] + b["median_ms"]) / 2)
            ref_bench = {"median_ms_a": a["median_ms"], "median_ms_b": b["median_ms"], "noise": noise}
            log(f"reference {a['median_ms']:.4f} / {b['median_ms']:.4f} ms -> noise {noise * 100:.2f}%")
        spec.free(reference)
        if getattr(spec, "reference_model", None) is not None:
            spec.reference_model = None
        del reference
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
        result["phase"] = "reference"
        write_json(out_dir / "metrics.json", result)
        log("REFERENCE/HARNESS ERROR\n" + result["error"])
        return 3

    # ---- candidate ----------------------------------------------------------------------
    ctx = CandidateContext(campaign_root=root, device=device, spec_name=spec.name, capabilities=caps, policy=goal.gates,
                           model_args=goal.model_args, workload_names=[w.name for w in workloads])
    try:
        log("building candidate (candidate.apply)")
        t0 = time.perf_counter()
        candidate = spec.load_candidate(ctx)
        result["candidate_build_seconds"] = round(time.perf_counter() - t0, 2)
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
        result["phase"] = "candidate-build"
        result["candidate_logs"] = ctx.logs
        write_json(out_dir / "metrics.json", result)
        log("CANDIDATE CRASHED while building\n" + result["error"])
        return 2

    # ---- gates --------------------------------------------------------------------------
    if args.profile_only:
        goal.gates.stages = []
        args.no_bench = True
    log("running correctness gates: " + (", ".join(goal.gates.stages) or "none"))
    gates = run_gates(spec, candidate, workloads, reference_outputs, inputs, edge_workloads, edge_reference, edge_inputs,
                      goal.gates.stages, _run_workload)
    write_json(out_dir / "gates.json", gates)
    log(f"gates: {'PASS' if gates['passed'] else 'FAIL'} ({gates['summary']})")
    for check in gates["failed_checks"][:8]:
        log(f"  failed: {check['name']}: {check['detail'][:200]}")
    del reference_outputs, edge_reference
    gc.collect()

    # ---- benchmark ----------------------------------------------------------------------
    metrics: dict[str, Any] = {"workloads": {}, "primary": primary.name, "reference": ref_bench}
    if not args.no_bench:
        try:
            for w in workloads:
                if not w.bench:
                    continue
                reset_peak_memory()
                fn = lambda w=w: w.run(candidate, inputs[w.name])  # noqa: E731
                stats = time_callable(fn, warmup=args.warmup or goal.bench.warmup, repeats=args.repeats or goal.bench.repeats,
                                      ramp_seconds=goal.bench.ramp_seconds)
                stats["peak_vram_mb"] = peak_memory_mb()
                stats.update(spec.derived_metrics(w, stats["median_ms"]))
                metrics["workloads"][w.name] = stats
                log(f"bench {w.name}: median {stats['median_ms']:.4f} ms (min {stats['min_ms']:.4f}, p90 {stats['p90_ms']:.4f})"
                    + "".join(f", {k} {v:.4g}" for k, v in stats.items() if k in ("rtf", "tokens_per_s", "fps")))
            p = metrics["workloads"].get(primary.name, {})
            metrics["latency_ms"] = p.get("median_ms")
            metrics["latency_min_ms"] = p.get("min_ms")
            metrics["peak_vram_mb"] = p.get("peak_vram_mb")
            for key in ("rtf", "tokens_per_s", "fps", "audio_x_realtime", "samples_per_s"):
                if key in p:
                    metrics[key] = p[key]
        except Exception:  # noqa: BLE001
            metrics["error"] = traceback.format_exc()
            log("BENCHMARK CRASHED\n" + metrics["error"])
    metrics["candidate_report"] = candidate_report(root)
    metrics["candidate_logs"] = ctx.logs
    metrics["gates_passed"] = gates["passed"]
    metrics["seconds"] = round(time.perf_counter() - started, 2)
    write_json(out_dir / "metrics.json", {**result, **metrics})

    # ---- profile ------------------------------------------------------------------------
    if not args.no_profile:
        try:
            log("profiling primary workload")
            from ..backends import graphs as _graphs
            prof = profile_workload(candidate, lambda m, i: primary.run(m, i), inputs[primary.name], spec.hooks_root(candidate),
                                    warmup=2, trace_path=str(out_dir / "trace.json"), attribution_context=_graphs.eager_mode)
            history = read_json(Path(args.history), []) if args.history else []
            targets = build_targets(prof, device_info, history=history, hints=spec.hotspot_hints())
            prof_out = {**{k: v for k, v in prof.items() if k != "modules"}, "modules": prof["modules"][:80], "targets": targets,
                        "workload": primary.name}
            write_json(out_dir / "profile.json", prof_out)
            log(f"profile: {prof['kernel_count']} kernels, wall {prof['wall_ms']:.3f} ms, GPU busy {prof['gpu_busy_ms']:.3f} ms "
                f"({(prof['gpu_busy_ratio'] or 0) * 100:.0f}%), top target: {targets[0]['title'] if targets else '-'}")
        except Exception:  # noqa: BLE001
            write_json(out_dir / "profile.json", {"error": traceback.format_exc()})
            log("PROFILE FAILED\n" + traceback.format_exc()[-800:])
    log(f"done in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

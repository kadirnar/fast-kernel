"""One experiment = evaluate the current candidate tree, decide keep/revert, record everything.

    record = run_experiment(campaign, "fuse RMSNorm into QKV projection", techniques=["fused-norm"], target="t_ab12")

Decision order (strict): crash -> revert; gates FAIL -> revert; improved by at least
max(min_improvement, measurement uncertainty) -> keep (commit, tag exp-N, promote incumbent);
improved but by less than that -> **bank** (commit, leave in candidate/, do not move the incumbent),
so small real wins accumulate until they are jointly large enough to measure instead of being thrown
away one at a time; equal or worse -> revert unless `simpler=True` and the loss is within the
threshold (autoresearch's simplicity rule).

"Improved" is measured by the anchored ratio when one is available: the candidate and the reference
model are timed interleaved in the same process, so drift between sessions cancels instead of being
charged to the candidate. See harness/bench.compare_callables.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from typing import Any

from .. import knowledge, results
from ..campaign import Campaign, Incumbent
from ..profiling.plan import write_plan
from ..util import GpuLock, fmt, now_iso, read_json, run, slugify, tail_text, write_json


def _metric_value(metrics: dict[str, Any], name: str, primary: str | None) -> float | None:
    if name in metrics and isinstance(metrics[name], (int, float)):
        return float(metrics[name])
    wl = (metrics.get("workloads") or {}).get(primary or metrics.get("primary") or "", {})
    for key in (name, name.replace("latency_ms", "median_ms")):
        if key in wl and isinstance(wl[key], (int, float)):
            return float(wl[key])
    return None


def _improvement(minimize: bool, incumbent: float | None, value: float | None) -> float | None:
    if incumbent is None or value is None or incumbent == 0:
        return None
    return (incumbent - value) / incumbent if minimize else (value - incumbent) / incumbent


def decide_improvement(goal, incumbent, metrics: dict[str, Any], value: float | None, *,
                       simpler: bool = False) -> dict[str, Any]:
    """How much faster is this candidate, is that measurable, and what should happen to it?

    Pure: no git, no files, no side effects -- the whole accept/bank/reject rule in one place.

    Two things decide it. First, *what is compared*: when both the candidate and the incumbent were
    timed against the same reference model interleaved in their own process (`metrics["anchor"]`),
    the ratio of those two ratios is used. Drift between sessions -- clocks, thermals, another
    process on the GPU -- moves a raw millisecond count but cancels out of the ratio, so it is no
    longer charged to the candidate. Second, *what counts as measurable*: the threshold is the
    combined uncertainty of the two ratio measurements, not a number frozen at baseline.

    A gain that is real but below that threshold is banked rather than discarded, so a campaign can
    accumulate several small wins and promote them together once they are jointly measurable.
    """
    anchor = metrics.get("anchor") or {}
    anchor_ratio = anchor.get("ratio")
    anchor_unc = float(anchor.get("ratio_uncertainty") or 0.0)
    raw_improvement = _improvement(goal.minimize, incumbent.value, value)
    improvement = raw_improvement
    threshold = max(goal.min_improvement, incumbent.noise_floor or 0.0)
    basis = "raw milliseconds vs the incumbent's stored value"
    contested = False
    if anchor_ratio and incumbent.anchor_ratio:
        improvement = anchor_ratio / incumbent.anchor_ratio - 1.0
        # two independent ratio measurements, so their uncertainties add in quadrature
        threshold = max(goal.min_improvement, math.hypot(anchor_unc, incumbent.anchor_uncertainty))
        basis = "anchored ratio vs the reference model"
        # The two bases measure the same change against different references, so they should agree
        # on its SIGN. When they do not, this run has not resolved the change, whatever magnitude
        # the anchored ratio reports -- and acting on it is how a campaign promotes a regression or
        # throws away a real win. Both happened in campaigns/mimi within one session: a change was
        # kept at +4.21 % anchored that the profiler put at +18.7 us of gpu_busy and five absolute
        # readings put at 1.45 % SLOWER, and its revert was then discarded at -1.07 % anchored while
        # the same run measured 1.406 against an incumbent of 1.432. Banking a contested reading
        # keeps the work without moving the number the campaign defends.
        if raw_improvement is not None and (improvement > 0) != (raw_improvement > 0):
            contested = True
    out: dict[str, Any] = {"anchor_ratio": anchor_ratio, "anchor_uncertainty": anchor_unc,
                           "improvement": improvement, "raw_improvement": raw_improvement,
                           "contested": contested, "threshold": threshold, "decision_basis": basis}
    if improvement is not None and improvement >= threshold:
        out["status"] = "keep"
        out["reason"] = (f"improved {improvement * 100:+.2f}% (threshold {threshold * 100:.2f}%, {basis})"
                         + (f" -- CONTESTED: raw milliseconds say {raw_improvement * 100:+.2f}%, so re-anchor and "
                            f"re-measure before building on this number" if contested else ""))
    elif simpler and improvement is not None and improvement >= -threshold:
        out["status"] = "keep"
        out["reason"] = f"kept as simpler code within noise ({improvement * 100:+.2f}%)"
    elif improvement is not None and improvement > 0 and incumbent.banked < goal.bench.max_banked:
        # Real, gate-clean, but smaller than this machine can resolve in one measurement.
        # Discarding it loses it for good; instead it is committed and the next experiment builds
        # on top of it. The incumbent -- the number the campaign defends -- only moves once the
        # accumulated tree clears the floor in a single measurement, so banking never claims a
        # speedup that was not measured.
        out["status"] = "bank"
        out["reason"] = (f"banked {improvement * 100:+.2f}%: real but under the {threshold * 100:.2f}% floor; "
                         f"kept in candidate/ for the next experiment to build on "
                         f"({incumbent.banked + 1}/{goal.bench.max_banked})")
    else:
        out["status"] = "discard"
        out["reason"] = ((f"no improvement: {improvement * 100:+.2f}% vs threshold {threshold * 100:.2f}%"
                          + (f" -- CONTESTED: raw milliseconds say {raw_improvement * 100:+.2f}%, so this run has not "
                             f"resolved it; re-anchor and retry before discarding the idea" if contested else ""))
                         if improvement is not None else "could not compare to incumbent")
    return out


def _busy_ratio(prof: dict[str, Any], value: float | None) -> float | None:
    """GPU-busy against the BENCHMARKED latency, not the profiler's own wall clock.

    The profile times its own call to get a denominator; the benchmark times the same call under
    the protocol every verdict is made on, with a clock ramp and a median. When both exist the
    benchmark's is the honest one. On campaigns/mimi the profile's denominator reported 83.6 %
    busy where the benchmark's gives 98.6 %, and the difference is not academic -- it is the
    difference between "16 % of the wall is reclaimable idle" and "there is nothing on the host
    left to win", which is the first thing an agent reads off `fast-kernel brief`.
    """
    busy = prof.get("gpu_busy_ms")
    if busy and value:
        return busy / float(value)
    return prof.get("gpu_busy_ratio")


def run_experiment(campaign: Campaign, description: str, *, baseline: bool = False, techniques: list[str] | None = None,
                   target: str | None = None, notes: str = "", timeout: float | None = None, simpler: bool = False,
                   force: bool = False, profile: bool | None = None, repeats: int | None = None, warmup: int | None = None,
                   workloads: list[str] | None = None, agent: str | None = None, quiet: bool = False) -> dict[str, Any] | None:
    goal = campaign.goal
    store = campaign.store
    campaign.ensure_git()
    techniques = list(techniques or [])
    agent = agent or os.environ.get("FK_AGENT") or "cli"

    number = store.next_experiment_number()
    if baseline and number != 0 and not force:
        print(f"error: baseline already recorded (next experiment is #{number}); use --force to re-baseline", file=sys.stderr)
        return None
    if baseline:
        number = 0 if not force else number
    incumbent = campaign.load_incumbent()
    if not baseline and incumbent.number < 0:
        print("error: no baseline yet -- run `fast-kernel baseline` first", file=sys.stderr)
        return None
    diff = "" if baseline else campaign.candidate_diff()
    if not baseline and not diff.strip() and not force:
        print("error: candidate/ is identical to the incumbent -- nothing to evaluate (use --force to re-measure)", file=sys.stderr)
        return None

    slug = slugify("baseline" if baseline else description)
    exp_dir = campaign.experiment_dir(number, slug)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "patch.diff").write_text(diff, encoding="utf-8")
    if notes:
        (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    record: dict[str, Any] = {
        "number": number, "name": slug, "status": "running", "description": description if not baseline else
        (description or "baseline: unmodified reference path"), "techniques": techniques, "target": target, "agent": agent,
        "parent": incumbent.number if not baseline else None, "parent_commit": incumbent.commit if not baseline else None,
        "created_at": now_iso(), "dir": str(exp_dir), "patch_lines": len(diff.splitlines()), "baseline": baseline,
        "primary_metric": goal.target_metric,
    }
    store.save_experiment(record)
    store.event("experiment.started", number=number, description=record["description"], techniques=techniques, target=target, agent=agent)
    store.set_agent("harness", "running", f"#{number} {record['description'][:80]}")

    history = store.list_experiments()
    history_path = exp_dir / "history.json"
    write_json(history_path, [{k: e.get(k) for k in ("number", "status", "target", "techniques")} for e in history])
    argv = [sys.executable, "-m", "fastkernel.harness.run", "--campaign", str(campaign.root), "--out", str(exp_dir),
            "--history", str(history_path)]
    if baseline:
        argv.append("--noise-check")
    do_profile = goal.bench.profile_every_experiment if profile is None else profile
    if not do_profile:
        argv.append("--no-profile")
    if repeats:
        argv += ["--repeats", str(repeats)]
    if warmup:
        argv += ["--warmup", str(warmup)]
    if workloads:
        argv += ["--workloads", ",".join(workloads)]
    env = {"PYTHONUNBUFFERED": "1", "FK_EXPERIMENT": str(number), "FK_CAMPAIGN": str(campaign.root)}
    if not quiet:
        print(f"== experiment #{number}: {record['description']}", flush=True)
    # One GPU, one measurement at a time: parallel agents may build and think concurrently, but the
    # harness only runs once the device is free, and the wait is not charged to this experiment.
    with GpuLock(timeout=goal.bench.timeout_seconds, log=lambda m: None if quiet else print(m, flush=True)) as lock:
        if lock.waited > 1.0:
            store.event("gpu.waited", number=number, seconds=round(lock.waited, 1))
        started = time.perf_counter()
        result = run(argv, cwd=campaign.root, timeout=timeout or goal.bench.timeout_seconds, env=env)
        duration = time.perf_counter() - started
    record["gpu_wait_s"] = round(lock.waited, 1)
    (exp_dir / "run.log").write_text(result.stdout + ("\n--- stderr ---\n" + result.stderr if result.stderr else ""), encoding="utf-8")

    metrics = read_json(exp_dir / "metrics.json", {}) or {}
    gates = read_json(exp_dir / "gates.json", {}) or {}
    prof = read_json(exp_dir / "profile.json", {}) or {}
    value = _metric_value(metrics, goal.target_metric, metrics.get("primary"))
    record.update({
        "duration_s": round(duration, 1), "returncode": result.returncode, "timed_out": result.timed_out,
        "primary_value": value, "metrics": _compact_metrics(metrics), "gates": _compact_gates(gates),
        "kernel_count": prof.get("kernel_count"), "gpu_busy_ratio": _busy_ratio(prof, value), "wall_ms": prof.get("wall_ms"),
        "top_targets": [{k: t.get(k) for k in ("id", "title", "fraction", "amdahl_gain", "sol_efficiency", "category", "boundness")} for t in (prof.get("targets") or [])[:6]],
        "candidate_report": metrics.get("candidate_report"), "peak_vram_mb": metrics.get("peak_vram_mb"),
        "candidate_logs": (metrics.get("candidate_logs") or [])[-20:],
    })
    for key in ("rtf", "tokens_per_s", "fps", "audio_x_realtime"):
        if key in metrics:
            record[key] = metrics[key]

    # ---- decision ---------------------------------------------------------------------
    reason = ""
    if result.timed_out:
        status, reason = "crash", f"timeout after {goal.bench.timeout_seconds:.0f} s"
    elif result.returncode == 3:
        status, reason = "error", "harness/reference error (see run.log) -- candidate left in place"
    elif result.returncode == 2 or metrics.get("error") or result.returncode not in (0,):
        status, reason = "crash", _crash_reason(metrics, result)
    elif not gates.get("passed"):
        failed = gates.get("failed_checks") or []
        status = "discard"
        reason = "gates failed: " + "; ".join(f"{c['name']} ({c['detail'][:80]})" for c in failed[:4])
    elif value is None:
        status, reason = "crash", f"target metric '{goal.target_metric}' missing from metrics"
    elif baseline:
        status, reason = "baseline", "baseline recorded"
    elif not diff.strip():
        # `--force` on an unchanged tree: this is a re-measurement of the incumbent itself, not a
        # proposal. Nothing to keep or revert -- but it does tell us the incumbent's own anchor
        # ratio, which is how a campaign that started before anchoring existed acquires one.
        status = "remeasure"
        anchor = metrics.get("anchor") or {}
        record["anchor_ratio"] = anchor.get("ratio")
        record["anchor_uncertainty"] = float(anchor.get("ratio_uncertainty") or 0.0)
        reason = (f"re-measured the incumbent: {value if value is None else f'{value:.4f}'} "
                  f"{goal.target_metric}" + (f", anchor {record['anchor_ratio']:.4f}x vs the reference"
                                             if record.get("anchor_ratio") else ""))
    else:
        verdict = decide_improvement(goal, incumbent, metrics, value, simpler=simpler)
        status, reason = verdict.pop("status"), verdict.pop("reason")
        record.update(verdict)
    record["status"] = status
    record["reason"] = reason

    baseline_rec = next((e for e in history if e.get("number") == 0), None) if not baseline else record
    base_value = (baseline_rec or {}).get("primary_value")
    if value and base_value:
        record["speedup_vs_baseline"] = base_value / value if goal.minimize else value / base_value
    if value and incumbent.value and not baseline:
        record["speedup_vs_incumbent"] = incumbent.value / value if goal.minimize else value / incumbent.value

    # ---- git / incumbent ----------------------------------------------------------------
    if status in ("keep", "baseline"):
        message = f"exp {number}: {record['description'][:72]}" if not baseline else "exp 0: baseline"
        commit = campaign.commit_candidate(message, tag=f"exp-{number}")
        record["commit"] = commit
        noise = incumbent.noise_floor
        if baseline:
            ref = metrics.get("reference") or {}
            noise = float(ref.get("noise") or 0.0)
            record["noise_floor"] = noise
        campaign.save_incumbent(Incumbent(number=number, commit=commit, value=value, metrics=record["metrics"],
                                          noise_floor=noise, anchor_ratio=record.get("anchor_ratio"),
                                          anchor_uncertainty=record.get("anchor_uncertainty") or 0.0, banked=0))
        store.event("incumbent.promoted", number=number, commit=commit, value=value, speedup_vs_baseline=record.get("speedup_vs_baseline"))
        if prof.get("targets"):
            try:
                from ..models.spec import load_spec
                spec_notes = load_spec(campaign.root, goal.model, goal.model_args, goal.gates).notes
            except Exception:  # noqa: BLE001
                spec_notes = ""
            caps = read_json(campaign.capabilities_path, {}) or {}
            write_plan(campaign.root, profile=prof, targets=prof["targets"], device=caps.get("device_info", {}),
                       workload=prof.get("workload", ""), experiment=number, spec_notes=spec_notes, backends=caps.get("backends"))
    elif status == "remeasure":
        record["commit"] = campaign.head()
        if record.get("anchor_ratio"):
            campaign.save_incumbent(Incumbent(number=incumbent.number, commit=incumbent.commit, value=incumbent.value,
                                              metrics=incumbent.metrics, noise_floor=incumbent.noise_floor,
                                              anchor_ratio=record["anchor_ratio"],
                                              anchor_uncertainty=record.get("anchor_uncertainty") or 0.0,
                                              banked=incumbent.banked))
    elif status == "bank":
        # Committed, so that a later discard's `git checkout -- candidate` restores the banked work
        # instead of silently deleting it. The incumbent (value, ratio) deliberately stays put: the
        # next experiment is still measured against the last number the campaign can defend.
        record["commit"] = campaign.commit_candidate(f"exp {number} (banked): {record['description'][:60]}")
        campaign.save_incumbent(Incumbent(number=incumbent.number, commit=incumbent.commit, value=incumbent.value,
                                          metrics=incumbent.metrics, noise_floor=incumbent.noise_floor,
                                          anchor_ratio=incumbent.anchor_ratio,
                                          anchor_uncertainty=incumbent.anchor_uncertainty,
                                          banked=incumbent.banked + 1))
    elif status in ("discard", "crash"):
        record["commit"] = campaign.head()
        campaign.restore_candidate()
    else:
        record["commit"] = campaign.head()

    # ---- campaign memory: structured reflexion + measured outcome (KernelAgent reflexion, KernelSkill
    # dual memory, and the failure class that drives adaptive error routing) --------------------------
    from .. import memory as _memory
    record["_target_obj"] = next((t for t in (prof.get("targets") or []) if t.get("id") == record.get("target")), None)
    refl = _memory.reflexion(record, incumbent_value=incumbent.value, minimize=goal.minimize, gates=gates, run_log=result.stdout)
    record.pop("_target_obj", None)
    record["reflexion"] = refl
    record["failure_class"] = refl.get("failure_class")
    if not baseline:
        _memory.record_outcome(campaign, refl)

    record["finished_at"] = now_iso()
    store.save_experiment(record)
    store.event("experiment.finished", number=number, status=status, reason=reason, value=value,
                improvement=record.get("improvement"), speedup_vs_baseline=record.get("speedup_vs_baseline"),
                kernel_count=record.get("kernel_count"), description=record["description"])
    store.set_agent("harness", "idle", f"#{number} {status}")
    results.append_row(campaign.results_path, exp=number, commit=record.get("commit", ""), status=status, metric=goal.target_metric,
                       value=value, speedup=record.get("speedup_vs_baseline"),
                       peak_vram_gb=(record.get("peak_vram_mb") or 0) / 1024 if record.get("peak_vram_mb") else None,
                       gates=(gates.get("summary") or reason)[:60], description=record["description"])
    knowledge.append_experiment(campaign.knowledge_path, record)
    write_json(exp_dir / "record.json", record)
    if not quiet:
        print(render_verdict(campaign, record, gates, result.stdout + result.stderr), flush=True)
    return record


_EXC_RE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning)\b.*")


def _exception_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        if _EXC_RE.match(line):
            return line
    return lines[-1] if lines else ""


def _crash_reason(metrics: dict[str, Any], result) -> str:
    err = metrics.get("error") or ""
    if err:
        return f"crash in {metrics.get('phase', 'candidate')}: {_exception_line(err)[:220]}"
    text = tail_text(result.stderr or result.stdout, 40)
    return f"crash (exit {result.returncode}): {_exception_line(text)[:220] or 'no output'}"


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, stats in (metrics.get("workloads") or {}).items():
        out[name] = {k: v for k, v in stats.items() if k != "samples_ms"}
    for key in ("latency_ms", "latency_min_ms", "peak_vram_mb", "rtf", "tokens_per_s", "fps", "audio_x_realtime", "primary", "seconds",
                "candidate_build_seconds"):
        if key in metrics:
            out[key] = metrics[key]
    if metrics.get("reference"):
        out["reference"] = metrics["reference"]
    return out


def _compact_gates(gates: dict[str, Any]) -> dict[str, Any]:
    if not gates:
        return {}
    return {"passed": gates.get("passed"), "summary": gates.get("summary"),
            "stages": {k: {"passed": v.get("passed"), "skipped": v.get("skipped"), "checks": len(v.get("checks", []))} for k, v in (gates.get("stages") or {}).items()},
            "failed_checks": gates.get("failed_checks", [])[:12],
            "all_checks": [c for s in (gates.get("stages") or {}).values() for c in s.get("checks", [])][:80]}


def render_verdict(campaign: Campaign, record: dict[str, Any], gates: dict[str, Any], log: str) -> str:
    goal = campaign.goal
    inc = campaign.load_incumbent()
    lines = [f"== experiment #{record['number']} [{record['status'].upper()}] {record['description']}", f"   reason: {record.get('reason', '')}"]
    if gates:
        stages = " ".join(f"{k}:{'ok' if v.get('passed') else ('skip' if v.get('skipped') else 'FAIL')}" for k, v in (gates.get("stages") or {}).items())
        lines.append(f"   gates: {stages}")
        for check in (gates.get("failed_checks") or [])[:6]:
            lines.append(f"     x {check['name']}: {check['detail'][:160]}")
    value = record.get("primary_value")
    extra = "".join(f", {k}={fmt(record[k])}" for k in ("rtf", "tokens_per_s", "fps") if record.get(k))
    lines.append(f"   {goal.target_metric}: {fmt(value)} (incumbent {fmt(inc.value)}, baseline speedup {fmt(record.get('speedup_vs_baseline'), 3)}x{extra})")
    if record.get("improvement") is not None:
        lines.append(f"   decided on {record.get('decision_basis', 'raw milliseconds')}: "
                     f"{record['improvement'] * 100:+.2f}% vs a {(record.get('threshold') or 0) * 100:.2f}% resolution limit"
                     + (f"; {inc.banked} banked improvement(s) waiting to be promoted" if inc.banked else ""))
    if record.get("kernel_count") is not None:
        lines.append(f"   kernels/launch: {record['kernel_count']}, GPU busy {fmt((record.get('gpu_busy_ratio') or 0) * 100, 3)}%, "
                     f"wall {fmt(record.get('wall_ms'))} ms, duration {record.get('duration_s')} s")
    if record.get("top_targets"):
        top = record["top_targets"][0]
        sol = top.get("sol_efficiency")
        sol_str = f", roofline {sol * 100:.0f}% ({(1 - sol) * 100:.0f}% headroom)" if sol is not None else ""
        lines.append(f"   biggest remaining hotspot: {top.get('title')} ({top.get('id')}) share {fmt((top.get('fraction') or 0) * 100, 3)}% of GPU time{sol_str}")
    refl = record.get("reflexion") or {}
    if record.get("failure_class"):
        from ..memory import failure_detail
        lines.append(f"   failure class: {record['failure_class']} -- {failure_detail(record['failure_class'])}")
    if refl.get("outcome"):
        lines.append(f"   reflexion: {refl['outcome']}")
    if record["status"] in ("crash", "error"):
        lines.append("   --- log tail ---")
        lines.extend("   " + line for line in tail_text(log, 25).splitlines())
    lines.append("FK_RESULT " + json.dumps({k: record.get(k) for k in ("number", "status", "reason", "primary_value", "improvement",
                                                                      "speedup_vs_baseline", "kernel_count", "commit")}))
    return "\n".join(lines)

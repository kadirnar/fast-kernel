"""Campaign memory: the measured outcomes of past experiments, retrievable by target signature.

This is a memory of MEASUREMENTS, never authored priors. It records what each experiment actually did
— which target, which techniques, the measured delta vs the incumbent, the verdict, and (on a crash)
the classified failure — and lets the loop retrieve the measured history of similar targets so it does
not repeat dead ends. Two horizons, in the spirit of KernelSkill's dual memory:

- long-term: every experiment on this target signature across the whole campaign (and, optionally,
  sibling campaigns of the same model) — a retrievable, evidence-grounded skill base;
- short-term: the *repair chain* for the exact target currently being worked — the ordered list of
  failures already seen, so the agent avoids cyclic re-attempts.

It also carries the reflexion (KernelAgent) and the failure classifier that drives adaptive error
routing (AKG's Conductor): the failure class is a *diagnosis*, not a prescribed fix.

Stdlib-only; no torch. Persisted as one JSON object per line in .fast-kernel/memory.jsonl.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MEMORY_FILE = "memory.jsonl"

# (compiled regex over the run log, failure class, one-line diagnosis). First match wins; order matters.
_FAILURE_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED", re.I), "oom",
     "ran out of device memory — reduce working set / tile size / batch, or free intermediates"),
    (re.compile(r"illegal memory access|misaligned address|invalid device pointer", re.I), "illegal-memory",
     "illegal/misaligned memory access — check bounds, strides, masks and .contiguous()"),
    (re.compile(r"No module named|ModuleNotFoundError|ImportError", re.I), "import",
     "a package is missing — install it (uv pip install ...) and retry"),
    (re.compile(r"nvcc|ptxas|cicc|cudafe|error: .*\.cu|failed to compile|RuntimeError: Error building extension|undefined symbol", re.I),
     "compile", "the kernel did not compile — read the compiler error; fix the source or the toolchain"),
    (re.compile(r"size mismatch|shape .*mismatch|must match|stride|expected .* but got|not contiguous|dimension out of range|number of dims", re.I),
     "shape", "a shape/stride/dtype mismatch — align shapes, strides and dtypes with the reference"),
    (re.compile(r"timed out|timeout|exceeded .* seconds", re.I), "timeout",
     "the experiment exceeded the harness timeout — make it faster to build/run or narrower"),
    (re.compile(r"NaN|Inf|not finite", re.I), "nan", "produced NaN/Inf — check accumulation and masks"),
]


# class -> one-line diagnosis, for a failure class that was stored earlier (no re-classification).
FAILURE_DETAIL: dict[str, str] = {cls: detail for _rule, cls, detail in _FAILURE_RULES}
FAILURE_DETAIL.update({
    "numerical": "the numerical gate failed against the frozen reference",
    "determinism": "two identical runs disagreed — remove non-deterministic reductions/atomics",
    "edge": "an edge workload failed — short, odd-length or batched inputs",
    "other": "unclassified failure — read the run log",
})


def failure_detail(failure_class: str | None) -> str:
    return FAILURE_DETAIL.get(failure_class or "", FAILURE_DETAIL["other"])


def target_signature(target: dict[str, Any] | None) -> str:
    """A coarse, stable key for 'the same kind of target': category:boundness:class."""
    t = target or {}
    return f"{t.get('category', '?')}:{t.get('boundness', '?')}:{t.get('class', t.get('title', '?'))}"


def classify_failure(run_log: str, gates: dict[str, Any] | None = None) -> dict[str, str]:
    """Diagnose a crash/failed experiment from its log and gate results. Returns {class, detail}."""
    log = run_log or ""
    # a failed correctness stage is a stronger signal than the raw log
    if gates and not gates.get("passed", True):
        for name in ("numerical", "determinism", "edge", "shapes"):
            st = (gates.get("stages") or {}).get(name) or {}
            if not st.get("skipped", False) and not st.get("passed", True):
                cls = {"numerical": "numerical", "determinism": "determinism", "edge": "edge", "shapes": "shape"}[name]
                return {"class": cls, "detail": f"the {name} gate failed against the frozen reference"}
    for rule, cls, detail in _FAILURE_RULES:
        if rule.search(log):
            return {"class": cls, "detail": detail}
    return {"class": "other", "detail": "unclassified failure — read the run log"}


def reflexion(record: dict[str, Any], *, incumbent_value: float | None, minimize: bool,
              gates: dict[str, Any] | None = None, run_log: str = "") -> dict[str, Any]:
    """A compact, structured self-analysis of one finished experiment (KernelAgent-style).

    Everything here is derived from measurement: the measured delta vs the incumbent, the verdict, and
    the failure class on a crash. No opinion about *which technique to try next* — that stays the
    agent's job, grounded in this measured record."""
    status = record.get("status", "")
    value = record.get("primary_value")
    delta_pct = None
    if value is not None and incumbent_value:
        delta_pct = ((incumbent_value - value) / incumbent_value * 100.0) if minimize \
            else ((value - incumbent_value) / incumbent_value * 100.0)
    failure = None
    if status in ("crash", "error") or (gates and not gates.get("passed", True)):
        failure = classify_failure(run_log, gates)
    if status == "keep":
        outcome = f"kept: improved {delta_pct:+.2f}%" if delta_pct is not None else "kept"
    elif status == "bank":
        # a real gain, kept in the tree, but too small for this machine to resolve on its own
        outcome = (f"banked: improved {delta_pct:+.2f}%, under the {record.get('threshold', 0) * 100:.2f}% floor"
                   if delta_pct is not None else "banked (real but below the noise floor)")
    elif status == "discard" and failure is None:
        outcome = (f"no gain ({delta_pct:+.2f}% vs {record.get('threshold', 0) * 100:.2f}% threshold)"
                   if delta_pct is not None else "discarded (below threshold)")
    elif failure is not None:
        outcome = f"{status}: {failure['class']} — {failure['detail']}"
    else:
        outcome = status or "unknown"
    return {
        "number": record.get("number"), "target": record.get("target"),
        "target_sig": target_signature(record.get("_target_obj")) if record.get("_target_obj") else record.get("target_sig"),
        "techniques": record.get("techniques") or [], "status": status,
        "hypothesis": (record.get("description") or "")[:200], "delta_pct": delta_pct,
        "failure_class": (failure or {}).get("class"), "outcome": outcome,
    }


def record_outcome(campaign: Any, reflexion_dict: dict[str, Any]) -> None:
    """Append one reflexion to the campaign's long-term memory (best-effort; never raises)."""
    try:
        path = Path(campaign.state_dir) / MEMORY_FILE
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(reflexion_dict, default=str) + "\n")
    except OSError:
        pass


def _read_memory(state_dir: Path) -> list[dict[str, Any]]:
    path = Path(state_dir) / MEMORY_FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _sibling_state_dirs(campaign: Any) -> list[Path]:
    """Other campaigns of the SAME model under the same campaigns/ dir — long-term cross-campaign memory."""
    try:
        model = campaign.goal.model
        campaigns_dir = Path(campaign.root).parent
        dirs = []
        for child in campaigns_dir.iterdir():
            sd = child / ".fast-kernel"
            if child.resolve() == Path(campaign.root).resolve() or not sd.is_dir():
                continue
            goal = child / "GOAL.md"
            if goal.exists() and f"model: {model}" in goal.read_text(encoding="utf-8", errors="ignore"):
                dirs.append(sd)
        return dirs
    except (OSError, AttributeError):
        return []


def retrieve(campaign: Any, target: dict[str, Any] | None, *, k: int = 8, cross_campaign: bool = True) -> dict[str, Any]:
    """Measured history relevant to `target`: the repair chain (failures for this exact target) and the
    outcomes of experiments on the same target signature (long-term skill base). Facts only."""
    sig = target_signature(target)
    tid = (target or {}).get("id")
    entries = _read_memory(campaign.state_dir)
    if cross_campaign:
        for sd in _sibling_state_dirs(campaign):
            entries.extend(_read_memory(sd))
    same_target = [e for e in entries if (tid and e.get("target") == tid) or e.get("target_sig") == sig]
    repair_chain = [e for e in same_target if e.get("failure_class")][-k:]
    similar = [e for e in same_target if e.get("status") in ("keep", "discard")][-k:]
    return {"signature": sig, "repair_chain": repair_chain, "similar": similar,
            "kept": [e for e in similar if e.get("status") == "keep"]}


def render_memory(mem: dict[str, Any]) -> str:
    """A compact, agent-readable block of measured facts (never a prescription)."""
    lines = [f"measured memory for {mem.get('signature', '?')} (facts from past experiments, not advice):"]
    if not mem.get("similar") and not mem.get("repair_chain"):
        lines.append("  (nothing measured yet for this kind of target)")
        return "\n".join(lines)
    for e in mem.get("similar", []):
        techs = ", ".join(e.get("techniques") or []) or "-"
        lines.append(f"  - exp #{e.get('number')} [{e.get('status')}] techniques=[{techs}]: {e.get('outcome')}")
    if mem.get("repair_chain"):
        lines.append("  repair chain (failures already seen for this target — do not repeat):")
        for e in mem["repair_chain"]:
            lines.append(f"  - exp #{e.get('number')}: {e.get('failure_class')} ({(e.get('techniques') or ['-'])[0]})")
    return "\n".join(lines)

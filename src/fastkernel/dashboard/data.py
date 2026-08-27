"""Read-only data assembly shared by the live server and the static report."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import knowledge
from ..campaign import Campaign
from ..util import read_json, tail_text

COMPACT_KEYS = ("number", "name", "status", "description", "techniques", "target", "agent", "parent", "commit", "created_at",
                "finished_at", "duration_s", "primary_value", "primary_metric", "improvement", "threshold", "speedup_vs_baseline",
                "speedup_vs_incumbent", "kernel_count", "gpu_busy_ratio", "wall_ms", "peak_vram_mb", "reason", "rtf", "tokens_per_s",
                "fps", "patch_lines", "baseline", "top_targets")


def compact_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    out = {k: exp.get(k) for k in COMPACT_KEYS}
    gates = exp.get("gates") or {}
    out["gates_passed"] = gates.get("passed")
    out["gates_summary"] = gates.get("summary")
    out["failed_checks"] = [c.get("name") for c in (gates.get("failed_checks") or [])[:5]]
    return out


def campaign_state(campaign: Campaign) -> dict[str, Any]:
    experiments = campaign.store.list_experiments()
    hotspots = read_json(campaign.hotspots_path, {}) or {}
    caps = read_json(campaign.capabilities_path, {}) or {}
    baseline = next((e for e in experiments if e.get("number") == 0), None)
    return {
        "summary": campaign.summary(),
        "experiments": [compact_experiment(e) for e in experiments],
        "hotspots": {"summary": hotspots.get("summary"), "targets": (hotspots.get("targets") or [])[:16], "generated_at": hotspots.get("generated_at"),
                     "workload": hotspots.get("workload"), "experiment": hotspots.get("experiment")},
        "baseline_targets": (baseline or {}).get("top_targets") or [],
        "capabilities": {"device": caps.get("device_info", {}), "backends": {k: {kk: v.get(kk) for kk in ("available", "compiled", "version", "error")}
                                                                         for k, v in (caps.get("backends") or {}).items()}},
        "agents": campaign.store.agents(),
        "leases": campaign.store.leases(),
        "insights": knowledge.read_insights(campaign.knowledge_path, 20),
        "last_event_id": campaign.store.last_event_id(),
    }


def experiment_detail(campaign: Campaign, number: int) -> dict[str, Any] | None:
    record = campaign.store.get_experiment(number)
    if not record:
        return None
    exp_dir = Path(record["dir"]) if record.get("dir") else None
    detail = dict(record)
    detail["compact"] = compact_experiment(record)
    if exp_dir is not None and exp_dir.exists():
        detail["patch"] = (exp_dir / "patch.diff").read_text(encoding="utf-8")[:60000] if (exp_dir / "patch.diff").exists() else ""
        detail["log_tail"] = tail_text((exp_dir / "run.log").read_text(encoding="utf-8", errors="replace"), 80) if (exp_dir / "run.log").exists() else ""
        prof = read_json(exp_dir / "profile.json", {}) or {}
        detail["profile"] = {"kernel_count": prof.get("kernel_count"), "wall_ms": prof.get("wall_ms"), "gpu_busy_ms": prof.get("gpu_busy_ms"),
                             "gpu_busy_ratio": prof.get("gpu_busy_ratio"), "targets": (prof.get("targets") or [])[:12],
                             "kernels": (prof.get("kernels") or [])[:15], "modules": (prof.get("modules") or [])[:20]}
        detail["notes"] = (exp_dir / "notes.md").read_text(encoding="utf-8") if (exp_dir / "notes.md").exists() else ""
    return detail


def campaigns_index(root: Path) -> list[dict[str, Any]]:
    out = []
    for campaign in Campaign.discover_all(root):
        try:
            out.append(campaign.summary())
        except Exception as exc:  # noqa: BLE001
            out.append({"name": campaign.name, "root": str(campaign.root), "error": str(exc)})
    return out

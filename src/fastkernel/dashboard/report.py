"""Export a self-contained HTML report (same UI as the live dashboard, data embedded, no server)."""
from __future__ import annotations

import json
from pathlib import Path

from ..campaign import Campaign
from .data import campaign_state, experiment_detail

STATIC = Path(__file__).parent / "static"


def build_report(campaign: Campaign, out: Path, max_details: int = 400) -> Path:
    state = campaign_state(campaign)
    details = {}
    for exp in state["experiments"][-max_details:]:
        detail = experiment_detail(campaign, int(exp["number"]))
        if detail:
            detail.pop("metrics", None)
            details[str(exp["number"])] = {k: detail.get(k) for k in ("compact", "gates", "profile", "patch", "log_tail", "notes", "metrics",
                                                                    "candidate_report", "candidate_logs")}
    events = campaign.store.events_after(max(0, campaign.store.last_event_id() - 400), limit=400)
    plan = campaign.plan_path.read_text(encoding="utf-8") if campaign.plan_path.exists() else ""
    know = campaign.knowledge_path.read_text(encoding="utf-8") if campaign.knowledge_path.exists() else ""
    data = {"static": True, "name": campaign.name, "state": state, "details": details, "events": events, "plan": plan, "knowledge": know,
            "goal": campaign.goal_path.read_text(encoding="utf-8")}
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    embedded = json.dumps(data, default=str).replace("</", "<\\/")
    html = html.replace('<link rel="stylesheet" href="/static/styles.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="/static/app.js"></script>', f"<script>window.__FK_DATA__ = {embedded};</script>\n<script>\n{js}\n</script>")
    out = Path(out)
    out.write_text(html, encoding="utf-8")
    return out

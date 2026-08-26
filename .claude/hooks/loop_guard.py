#!/usr/bin/env python3
"""Stop hook: keep the session iterating while a campaign loop is active (the 'ralph loop').

Blocks the stop with an instruction to run the next experiment, as long as the campaign made progress
since the last block (a new experiment was recorded). Two consecutive stops without progress let the
session end so a stuck agent cannot spin forever. `fast-kernel loop stop` / `fast-kernel stop` end it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import experiment_count, find_campaigns, last_experiment, project_dir, read_input  # noqa: E402

MAX_NO_PROGRESS = 3


def main() -> int:
    if os.environ.get("FK_HEADLESS") == "1":
        return 0    # `fast-kernel auto` / workers own the iteration count; the hook must not keep the session alive
    data = read_input()
    root = project_dir()
    cwd = Path(data.get("cwd") or root)
    campaigns = [c for c in {*find_campaigns(root), *find_campaigns(cwd)} if (c / ".fast-kernel" / "loop.active").exists()]
    campaigns = [c for c in campaigns if not (c / ".fast-kernel" / "stop").exists() and not (c / ".fast-kernel" / "paused").exists()]
    if not campaigns:
        return 0
    campaign = sorted(campaigns)[0]
    guard_path = campaign / ".fast-kernel" / "loop_guard.json"
    try:
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        guard = {"count": 0, "no_progress": 0}
    count = experiment_count(campaign)
    if count > int(guard.get("count", 0)):
        guard = {"count": count, "no_progress": 0}
    else:
        guard["no_progress"] = int(guard.get("no_progress", 0)) + 1
    guard["last_stop"] = time.time()
    guard_path.write_text(json.dumps(guard), encoding="utf-8")
    if guard["no_progress"] >= MAX_NO_PROGRESS:
        guard["no_progress"] = 0
        guard_path.write_text(json.dumps(guard), encoding="utf-8")
        print(f"fast-kernel loop guard: no new experiment in {MAX_NO_PROGRESS} consecutive stops; letting the session end. "
              f"Run `fast-kernel status` / `/fk-experiment` to continue.", file=sys.stderr)
        return 0
    last = last_experiment(campaign) or {}
    reason = (
        f"fast-kernel loop is ACTIVE for campaign `{campaign}` ({count} experiments so far; last: "
        f"#{last.get('number', '-')} [{last.get('status', '-')}] {str(last.get('description', ''))[:80]}). "
        f"Continue with the next experiment now: run `fast-kernel status --brief` and `fast-kernel ideas` in that directory, pick the "
        f"highest-Amdahl untried idea, edit only candidate/, then `fast-kernel eval -m ...` and `fast-kernel note ...`. "
        f"Do not ask whether to continue. To end the loop a human runs `fast-kernel loop stop`."
    )
    print(json.dumps({"decision": "block", "reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

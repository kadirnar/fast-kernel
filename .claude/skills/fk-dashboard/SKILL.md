---
name: fk-dashboard
description: Start (or point the user to) the fast-kernel live dashboard - every experiment on a live graph with lineage, hotspots, agents and events. Use for "show me the graph", "open the dashboard", "watch experiments live".
argument-hint: [--port 8765] [--root campaigns]
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Bash(curl -s http://127.0.0.1:*), Read
---

If `curl -s http://127.0.0.1:8765/api/campaigns` answers, the dashboard is already up: give the URL.
Otherwise start it in the background: `uv run fast-kernel dashboard --root campaigns --port 8765`
(add `--open` to launch a browser). The page streams `experiment.*`, `incumbent.promoted`, `agent.*`
and `note` events over SSE, and shows: latency per experiment with keep/discard/crash marks and the
incumbent line, speedup vs baseline, kernel launches, time-share by target (baseline vs now), the
experiment lineage tree, agent activity, the event log, and per-experiment details (gates, metrics,
profile, patch, log). For an offline copy: `/fk-report`.

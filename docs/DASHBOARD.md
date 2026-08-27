# Live dashboard and reports

`fast-kernel dashboard [--root campaigns] [--port 8765] [--open]` starts a zero-dependency server
(stdlib `ThreadingHTTPServer`). If the port is taken it picks the next free one and prints the URL.
It discovers every campaign under `--root` (a directory with `GOAL.md` + `candidate/`).

## What you see

- **KPI tiles**: incumbent metric with delta vs baseline and a sparkline, speedup vs baseline,
  experiment counts (keep / discard / crash), kernel launches per call (baseline -> now), GPU busy
  ratio, throughput (rtf / tokens per s / fps), active agents.
- **Every experiment**: the primary metric per experiment (log-scale toggle); marks coloured by
  status (keep = good, discard = neutral, crash = critical, harness error = warning, baseline = accent)
  with a 2 px surface ring; the incumbent as a step line; nearest-point hover tooltips; click opens
  the detail drawer.
- **Speedup vs baseline** and **kernel launches** step charts of the incumbent.
- **Where the time goes**: GPU-time share by target category for baseline vs the current incumbent,
  fixed categorical slots (7 + Other), legend + tooltips + the ranked-targets table as the table twin.
- **Ranked targets**: rank, class, boundness, share, roofline efficiency (SOL), measured headroom.
- **Lineage**: SVG tree — trunk of keeps, discards/crashes hanging off their parent incumbent.
- **Experiments table** (filterable) — the accessible twin of the chart; **Agents** (live state and
  last tool call from `fast-kernel auto` / workers / eval); **Insights** (KNOWLEDGE.md); **Event stream**.
- **Detail drawer**: verdict, per-workload timings, candidate report/logs, gates (stages + every
  check with value/threshold), profile (targets + top kernels), patch, log tail.
- **Controls**: start loop flag, pause, resume, stop, add a note. Theme toggle (light/dark/system).

## Live updates

`GET /api/c/<campaign>/stream?after=<event id>` is a Server-Sent Events stream polled from the
SQLite events table every 0.5 s (heartbeat every 15 s). The page re-fetches `/api/c/<campaign>/state`
on structural events and patches the agents panel from `agent.*` events without a refetch.
`EventSource` reconnects automatically.

## API

```
GET  /api/campaigns                         all campaigns under --root with summaries
GET  /api/c/<name>/state                    summary, compact experiments, hotspots, capabilities, agents, leases, insights
GET  /api/c/<name>/experiments/<n>          full record + gates + profile + patch + log tail
GET  /api/c/<name>/events?after=<id>        event backlog
GET  /api/c/<name>/plan                     PLAN.md, KNOWLEDGE.md, GOAL.md text
GET  /api/c/<name>/stream?after=<id>        SSE
POST /api/c/<name>/control {"action": "pause"|"resume"|"stop"|"start-loop"|"note", "text": ...}
```

## Static report

`fast-kernel report [--out report.html]` embeds the same data (state, per-experiment details, last
400 events, PLAN.md, KNOWLEDGE.md, GOAL.md) into the same single-page UI: one self-contained HTML
file, no server, no external assets. Share it or open it offline.

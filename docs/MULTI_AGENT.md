# Multi-agent operation

## Roles (Claude Code subagents, `.claude/agents/`)

| agent | owns | never |
|---|---|---|
| `fk-orchestrator` | the loop for one campaign | grading its own work |
| `fk-profiler` | hotspot evidence and the next hypothesis | editing code |
| `fk-kernel-engineer` | one focused change under `candidate/` + `fast-kernel eval` | editing the harness |
| `fk-verifier` | root-causing gate failures, minimal numerically exact fixes | loosening gates |
| `fk-benchmarker` | noise floor, protocol, "is it real?" | hand-editing results |
| `fk-reviewer` | reading the diff before an expensive eval | editing files |
| `fk-librarian` | references, templates, prior experiments | editing files |

The harness is the referee for all of them: acceptance is computed, not argued.

## Three ways to keep going forever

1. **Interactive, Stop-hook loop** — `/fk-optimize <model>` sets `.fast-kernel/loop.active`; the
   project's Stop hook (`.claude/hooks/loop_guard.py`) blocks the end of every turn with "run the next
   experiment" as long as new experiments keep landing (it yields after 3 consecutive stops without
   progress so a stuck agent cannot spin). `fast-kernel loop stop` ends it.
2. **Scheduled** — `/loop /fk-experiment` (self-paced) or `/loop 10m /fk-experiment` re-invokes the
   one-iteration skill; survives `--resume`.
3. **Headless** — `fast-kernel auto [--iterations N] [--model ...]` runs `claude -p` once per
   experiment with AGENTS.md appended to the system prompt, an allow-list of tools, and stream-json
   output that feeds the dashboard's agents panel and event log. `fast-kernel pause|resume|stop`.
   Headless sessions carry `FK_HEADLESS=1`, so the Stop hook yields to the driver's iteration count
   instead of forcing the session to continue.

## In-session parallel engineers (`/fk-parallel`)

Explore in parallel, measure serially. The orchestrator creates one private worktree per target
(`fast-kernel worktree create eng-<target>` → `<campaign>/.fast-kernel/worktrees/eng-<target>`, branch
`worker/eng-<target>` from the incumbent, with PLAN.md / KNOWLEDGE.md / incumbent.json copied
in), spawns one `fk-kernel-engineer` per worktree, and each engineer ends with
`fast-kernel propose -m "..." --technique ... --target ...` — which commits its `candidate/` changes and
writes the diff plus metadata to the main campaign's `.fast-kernel/inbox/`. The orchestrator then runs
`fast-kernel inbox`: every proposal is applied on the *current* incumbent, measured by the full harness
and kept or reverted, one after another. Patches that no longer apply are rejected with an event.

## Headless parallel workers (`fast-kernel auto --agents N`)

Each worker (`fast-kernel worker --name wN`) gets its own git worktree of the campaign
(`.fast-kernel/worktrees/wN`, branch `worker/wN` from the incumbent), copies of PLAN.md /
hotspots.json / KNOWLEDGE.md / incumbent.json, and **leases** one hotspot target (SQLite lease table)
so two workers do not attack the same module. Its accepted local experiment becomes a **proposal**:
the `candidate/` diff is written to `.fast-kernel/inbox/<worker>-<ts>.diff` with metadata. The main
loop (or `fast-kernel inbox`) applies proposals one at a time on top of the *current* incumbent and
re-evaluates them with the full harness — so the lineage stays a single chain of measured
improvements even when many agents explore concurrently. Conflicting patches are rejected with an event.

## Claude Agent SDK

`fastkernel/agents/driver.py` shells out to the `claude` CLI so it works with any subscription;
`claude-agent-sdk` (extra `agent`) can replace `run_iteration` with an in-process `query()` — the
prompt and tool allow-list are the same.

"""Headless loop: run the Claude Code CLI (`claude -p`) once per experiment, forever.

Each iteration is an independent `claude -p` invocation with the project's AGENTS.md appended to the
system prompt, restricted to the tools the loop needs, streamed as JSON so the dashboard shows what the
agent is doing. Between iterations the driver honours pause/stop flags and promotes worker proposals
from the inbox. This is the `fast-kernel auto` command; interactive users get the same loop through
the `/fk-optimize` skill or `/loop /fk-experiment`.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from ..campaign import Campaign
from ..util import now_iso, run, which
from .prompts import iteration_prompt

# After this many consecutive experiments with no accepted improvement the optimization is treated as
# exhausted and the autonomous loop stops itself -- it never asks a human whether to keep going.
CONVERGE_AFTER = 15

DEFAULT_ALLOWED_TOOLS = [
    "Read", "Edit", "Write", "MultiEdit", "Glob", "Grep", "LS",
    "Bash(fast-kernel *)", "Bash(fk *)", "Bash(uv run *)", "Bash(python *)", "Bash(python3 *)", "Bash(.venv/bin/python *)",
    "Bash(uv pip install *)", "Bash(pip install *)", "Bash(uv pip *)",
    "Bash(git diff *)", "Bash(git log *)", "Bash(git status *)", "Bash(git show *)", "Bash(cat *)", "Bash(ls *)", "Bash(head *)",
    "Bash(tail *)", "Bash(grep *)", "Bash(nvidia-smi *)", "Bash(cd *)", "Bash(sed -n *)", "Bash(wc *)",
]


def find_project_root(campaign_root: Path) -> Path:
    env = os.environ.get("FAST_KERNEL_HOME")
    if env and (Path(env) / ".claude").exists():
        return Path(env)
    for candidate in [campaign_root, *campaign_root.parents]:
        if (candidate / ".claude" / "skills" / "fk-experiment").exists() or (candidate / "AGENTS.md").exists():
            return candidate
    return campaign_root


def claude_binary() -> str | None:
    return which("claude") or (str(Path.home() / ".local" / "bin" / "claude") if (Path.home() / ".local" / "bin" / "claude").exists() else None)


def run_iteration(campaign: Campaign, *, prompt: str, model: str | None = None, max_turns: int = 80,
                  permission_mode: str = "acceptEdits", allowed_tools: list[str] | None = None, extra_args: list[str] | None = None,
                  agent_name: str = "claude", timeout: float = 3600.0, verbose: bool = True) -> dict[str, Any]:
    binary = claude_binary()
    if not binary:
        raise RuntimeError("claude CLI not found on PATH (install Claude Code: npm i -g @anthropic-ai/claude-code)")
    project_root = find_project_root(campaign.root)
    agents_md = project_root / "AGENTS.md"
    argv = [binary, "-p", prompt, "--output-format", "stream-json", "--verbose", "--permission-mode", permission_mode,
            "--max-turns", str(max_turns), "--allowedTools", ",".join(allowed_tools or DEFAULT_ALLOWED_TOOLS)]
    if agents_md.exists():
        argv += ["--append-system-prompt", agents_md.read_text(encoding="utf-8")[:60000]]
    if project_root != campaign.root:
        argv += ["--add-dir", str(campaign.root)]
    if model:
        argv += ["--model", model]
    argv += extra_args or []
    env = dict(os.environ)
    env.update({"FK_AGENT": agent_name, "FAST_KERNEL_CAMPAIGN": str(campaign.root), "FAST_KERNEL_HOME": str(project_root),
                "FK_HEADLESS": "1"})   # tells the Stop hook that this driver owns the iteration count
    store = campaign.store
    store.set_agent(agent_name, "running", "starting claude -p")
    started = time.perf_counter()
    summary: dict[str, Any] = {"agent": agent_name, "started_at": now_iso(), "turns": 0, "tool_uses": 0, "text": []}
    proc = subprocess.Popen(argv, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                            start_new_session=True)

    def _kill_tree():
        # start_new_session put the child in its own process group; kill the whole group so the
        # node/tool subprocesses claude spawned die with it instead of leaking (GPU/CPU held).
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()

    # Watchdog: kills the tree even if the child goes completely silent (readline would otherwise
    # block forever and the in-loop timeout check would never run).
    timed_out = {"v": False}

    def _on_timeout():
        timed_out["v"] = True
        _kill_tree()

    watchdog = threading.Timer(timeout, _on_timeout)
    watchdog.daemon = True
    watchdog.start()

    # Drain stderr concurrently: a child that writes more than the ~64 KB pipe buffer to stderr while
    # we are still reading stdout would otherwise deadlock (it blocks on write, we block on read).
    stderr_chunks: list[str] = []

    def _drain_stderr():
        if proc.stderr:
            for chunk in proc.stderr:
                stderr_chunks.append(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()
    try:
        assert proc.stdout
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            _handle_stream_message(campaign, agent_name, msg, summary, verbose)
        proc.wait(timeout=30)
    except Exception as exc:  # noqa: BLE001
        summary["error"] = str(exc)
        _kill_tree()
    finally:
        watchdog.cancel()
    if timed_out["v"]:
        summary["error"] = "iteration timeout"
    stderr_thread.join(timeout=5)
    stderr = "".join(stderr_chunks)
    summary["returncode"] = proc.returncode
    summary["seconds"] = round(time.perf_counter() - started, 1)
    if proc.returncode not in (0, None) and stderr:
        summary["stderr"] = stderr[-2000:]
    store.set_agent(agent_name, "idle", f"iteration done in {summary['seconds']} s ({summary['turns']} turns"
                    + (f", {summary['subtype']}" if summary.get("subtype") else "") + ")")
    store.event("agent.iteration", **{k: v for k, v in summary.items() if k not in ("text",)})
    return summary


def _handle_stream_message(campaign: Campaign, agent: str, msg: dict[str, Any], summary: dict[str, Any], verbose: bool) -> None:
    kind = msg.get("type")
    store = campaign.store
    if kind == "assistant":
        summary["turns"] += 1
        for block in (msg.get("message") or {}).get("content") or []:
            if block.get("type") == "text" and block.get("text"):
                text = block["text"].strip()
                summary["text"].append(text[:500])
                store.event("agent.text", agent=agent, text=text[:800])
                if verbose:
                    print(f"[{agent}] {text[:300]}", flush=True)
            elif block.get("type") == "tool_use":
                summary["tool_uses"] += 1
                name = block.get("name", "tool")
                inp = block.get("input") or {}
                detail = inp.get("command") or inp.get("file_path") or inp.get("pattern") or json.dumps(inp)[:120]
                store.set_agent(agent, "running", f"{name}: {str(detail)[:100]}")
                store.event("agent.tool", agent=agent, tool=name, detail=str(detail)[:400])
                if verbose:
                    print(f"[{agent}] {name}: {str(detail)[:160]}", flush=True)
    elif kind == "result":
        summary["result"] = (msg.get("result") or "")[:2000]
        summary["cost_usd"] = msg.get("total_cost_usd")
        summary["duration_ms"] = msg.get("duration_ms")
        summary["num_turns"] = msg.get("num_turns")
        summary["is_error"] = msg.get("is_error")
        summary["subtype"] = msg.get("subtype")          # e.g. "success" or "error_max_turns"
        store.event("agent.result", agent=agent, is_error=msg.get("is_error"), subtype=msg.get("subtype"), num_turns=msg.get("num_turns"),
                    cost_usd=msg.get("total_cost_usd"), result=(msg.get("result") or "")[:600])


def process_inbox(campaign: Campaign, *, quiet: bool = False) -> list[dict[str, Any]]:
    """Apply worker proposals one at a time on top of the incumbent and evaluate them."""
    from ..harness.evaluate import run_experiment
    inbox = campaign.state_dir / "inbox"
    done_dir = inbox / "processed"
    failed_dir = inbox / "failed"
    processed: list[dict[str, Any]] = []
    if not inbox.exists():
        return processed
    for meta_path in sorted(inbox.glob("*.json")):
        patch_path = meta_path.with_suffix(".diff")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not patch_path.exists():
            continue
        if campaign.candidate_dirty():
            if not quiet:
                print("inbox: candidate/ has uncommitted changes; skipping proposals until it is clean")
            break
        check = run(["git", "apply", "--check", str(patch_path)], cwd=campaign.root)
        if not check.ok:
            failed_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(patch_path), failed_dir / patch_path.name)
            shutil.move(str(meta_path), failed_dir / meta_path.name)
            campaign.store.event("inbox.rejected", worker=meta.get("worker"), reason="patch does not apply on the incumbent",
                                 description=meta.get("description"))
            processed.append({"meta": meta, "status": "conflict"})
            continue
        run(["git", "apply", str(patch_path)], cwd=campaign.root)
        record = run_experiment(campaign, f"[{meta.get('worker', 'worker')}] {meta.get('description', 'proposal')}",
                                techniques=meta.get("techniques") or [], target=meta.get("target"), agent=meta.get("worker"), quiet=quiet)
        done_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(patch_path), done_dir / patch_path.name)
        shutil.move(str(meta_path), done_dir / meta_path.name)
        processed.append({"meta": meta, "status": (record or {}).get("status", "error")})
    return processed


def run_auto(campaign: Campaign, *, iterations: int | None = None, model: str | None = None, max_turns: int = 80,
             permission_mode: str = "acceptEdits", agents: int = 0, sleep_between: float = 2.0, worker_iterations: int | None = None) -> None:
    """The endless loop. With agents > 0, spawn parallel worktree workers and promote their proposals."""
    from .worker import spawn_workers
    store = campaign.store
    campaign.set_flag("loop.active", f"auto pid={os.getpid()} started={now_iso()}")
    campaign.clear_flag("stop")
    workers = spawn_workers(campaign, agents, model=model, max_turns=max_turns, permission_mode=permission_mode,
                            iterations=worker_iterations) if agents > 0 else []
    store.event("loop.started", mode="auto", agents=agents, iterations=iterations, model=model)
    done = 0
    no_keep = 0
    try:
        while True:
            if campaign.has_flag("stop"):
                print("stop flag set; leaving the loop")
                break
            if no_keep >= CONVERGE_AFTER:
                print(f"optimization exhausted: {CONVERGE_AFTER} consecutive experiments with no accepted improvement; stopping.")
                store.event("loop.converged", mode="auto", no_keep=no_keep, completed=done)
                break
            if campaign.has_flag("paused"):
                time.sleep(2.0)
                continue
            if iterations is not None and done >= iterations:
                break
            goal = campaign.goal
            if not goal.continuous and goal.max_iterations and store.next_experiment_number() > goal.max_iterations:
                print("max_iterations reached (GOAL.md is not continuous)")
                break
            if workers:
                process_inbox(campaign)
                if all(w.poll() is not None for w in workers):
                    print("all workers finished")
                    if not process_inbox(campaign):
                        break
                time.sleep(sleep_between)
                continue
            done += 1
            before = store.next_experiment_number()
            prompt = iteration_prompt(campaign.root, iteration=before)
            summary = run_iteration(campaign, prompt=prompt, model=model, max_turns=max_turns, permission_mode=permission_mode)
            experiments = store.list_experiments()
            last = experiments[-1] if experiments else {}
            if last.get("number", -1) >= before and last.get("status") == "keep":
                no_keep = 0
            elif last.get("number", -1) >= before:
                no_keep += 1   # an experiment ran but was not accepted
            if summary.get("error") or (summary.get("is_error") and summary.get("subtype") != "error_max_turns"):
                print(f"iteration problem: {summary.get('error') or summary.get('result', '')[:300]}")
                time.sleep(10.0)
            elif summary.get("subtype") == "error_max_turns":
                print(f"iteration hit --max-turns ({max_turns}); the campaign state is intact, continuing")
            time.sleep(sleep_between)
    finally:
        for w in workers:
            if w.poll() is None:
                w.terminate()
        campaign.clear_flag("loop.active")
        store.event("loop.stopped", mode="auto", completed=done)

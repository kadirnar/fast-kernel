"""Parallel workers: each worker owns a git worktree of the campaign, leases one hotspot target,
runs the headless one-experiment loop there, and submits accepted patches to the main campaign's
inbox. The main loop re-evaluates proposals serially on top of the incumbent, so the lineage stays a
single chain of measured improvements even with many agents exploring at once.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..campaign import Campaign
from ..util import now_iso, read_json, run
from .driver import run_iteration
from .prompts import iteration_prompt


def worktree_path(campaign: Campaign, name: str) -> Path:
    return campaign.state_dir / "worktrees" / name


def create_worktree(campaign: Campaign, name: str) -> Campaign:
    path = worktree_path(campaign, name)
    campaign.ensure_git()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        campaign.git("worktree", "prune")
        campaign.git("worktree", "add", "-B", f"worker/{name}", str(path), "HEAD", check=True)
    else:
        run(["git", "checkout", "-q", "-B", f"worker/{name}", campaign.head()], cwd=path)
        run(["git", "reset", "-q", "--hard"], cwd=path)
        run(["git", "clean", "-fdq", "--", "candidate"], cwd=path)
    # ignored context files are not part of the tree: copy them
    for rel in ("PLAN.md", "hotspots.json", "capabilities.json", "KNOWLEDGE.md", "results.tsv"):
        src = campaign.root / rel
        if src.exists():
            shutil.copy2(src, path / rel)
    (path / ".fast-kernel").mkdir(exist_ok=True)
    inc = campaign.load_incumbent().to_dict()
    (path / ".fast-kernel" / "incumbent.json").write_text(json.dumps(inc, indent=2), encoding="utf-8")
    wt = Campaign(path)
    if not wt.store.list_experiments():
        base = campaign.store.get_experiment(0)
        if base:
            wt.store.save_experiment({**base, "dir": str(path / "experiments" / "0000-baseline")})
        latest = campaign.store.get_experiment(inc.get("number", 0))
        if latest and latest.get("number") != 0:
            wt.store.save_experiment(latest)
    return wt


def lease_target(campaign: Campaign, worker: str) -> dict[str, Any] | None:
    hotspots = read_json(campaign.hotspots_path, {}) or {}
    leased = {lease["target_id"] for lease in campaign.store.leases() if lease["state"] == "active" and lease["worker"] != worker}
    for target in hotspots.get("targets") or []:
        if target["id"] in leased:
            continue
        if campaign.store.acquire_lease(target["id"], worker):
            return target
    return None


def find_main_campaign(path: Path) -> Campaign:
    """The campaign a worktree belongs to: <main>/.fast-kernel/worktrees/<name> (or FAST_KERNEL_MAIN_CAMPAIGN)."""
    env = os.environ.get("FAST_KERNEL_MAIN_CAMPAIGN")
    if env and (Path(env) / "GOAL.md").exists():
        return Campaign(Path(env))
    here = Path(path).resolve()
    for candidate in [here, *here.parents]:
        if candidate.parent.name == "worktrees" and candidate.parents[1].name == ".fast-kernel":
            return Campaign(candidate.parents[2])
    raise RuntimeError(f"{here} is not a campaign worktree (expected <campaign>/.fast-kernel/worktrees/<name>)")


def submit_proposal(main: Campaign, wt: Campaign, record: dict[str, Any], worker: str) -> Path | None:
    inbox = main.state_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    # diff against the point where the worktree branched off the main lineage, so the patch contains only the proposal
    base = wt.git("merge-base", "HEAD", main.head()) or record.get("parent_commit") or main.load_incumbent().commit
    diff = wt.git("diff", f"{base}..HEAD", "--", "candidate", raw=True) if base else wt.git("show", "--format=", "HEAD", "--", "candidate", raw=True)
    if not diff.strip():
        return None
    diff = Campaign._patch_text(diff)
    stamp = f"{worker}-{int(time.time())}-{record.get('number', 0)}"
    (inbox / f"{stamp}.diff").write_text(diff, encoding="utf-8")
    meta = {"worker": worker, "description": record.get("description"), "techniques": record.get("techniques"),
            "target": record.get("target"), "local_value": record.get("primary_value"), "local_improvement": record.get("improvement"),
            "submitted_at": now_iso(), "worktree_commit": wt.head()}
    (inbox / f"{stamp}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    main.store.event("inbox.submitted", **meta)
    return inbox / f"{stamp}.diff"


def run_worker(main: Campaign, name: str, *, iterations: int | None = None, model: str | None = None, max_turns: int = 80,
               permission_mode: str = "acceptEdits") -> None:
    wt = create_worktree(main, name)
    os.environ["FAST_KERNEL_MAIN_CAMPAIGN"] = str(main.root)
    main.store.set_agent(name, "running", f"worktree {wt.root}")
    done = 0
    try:
        while iterations is None or done < iterations:
            if main.has_flag("stop"):
                break
            if main.has_flag("paused"):
                time.sleep(2)
                continue
            # rebase the worktree on the newest incumbent so proposals apply cleanly
            head = main.head()
            run(["git", "checkout", "-q", "-B", f"worker/{name}", head], cwd=wt.root)
            run(["git", "clean", "-fdq", "--", "candidate"], cwd=wt.root)
            (wt.root / ".fast-kernel" / "incumbent.json").write_text(json.dumps(main.load_incumbent().to_dict(), indent=2), encoding="utf-8")
            for rel in ("PLAN.md", "hotspots.json", "KNOWLEDGE.md"):
                if (main.root / rel).exists():
                    shutil.copy2(main.root / rel, wt.root / rel)
            target = lease_target(main, name)
            before = wt.store.next_experiment_number()
            prompt = iteration_prompt(wt.root, target=(target or {}).get("id"), worker=name, iteration=before)
            run_iteration(wt, prompt=prompt, model=model, max_turns=max_turns, permission_mode=permission_mode, agent_name=name)
            done += 1
            latest = wt.store.list_experiments(limit=1)
            if latest and latest[0].get("number", -1) >= before and latest[0].get("status") == "keep":
                submit_proposal(main, wt, latest[0], name)
                main.store.set_agent(name, "running", f"proposal submitted from #{latest[0]['number']}")
            if target:
                main.store.release_lease(target["id"], name, "released")
    finally:
        main.store.set_agent(name, "idle", f"worker finished after {done} iterations")


def spawn_workers(campaign: Campaign, count: int, *, model: str | None, max_turns: int, permission_mode: str,
                  iterations: int | None) -> list[subprocess.Popen]:
    procs = []
    for i in range(count):
        name = f"w{i + 1}"
        argv = [sys.executable, "-m", "fastkernel.cli", "worker", "--name", name, "--max-turns", str(max_turns),
                "--permission-mode", permission_mode]
        if model:
            argv += ["--model", model]
        if iterations:
            argv += ["--iterations", str(iterations)]
        log = (campaign.state_dir / f"worker-{name}.log").open("a", encoding="utf-8")
        procs.append(subprocess.Popen(argv, cwd=str(campaign.root), stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        campaign.store.event("worker.spawned", name=name, pid=procs[-1].pid)
    return procs


def propose_from_worktree(worktree: Path, description: str, techniques: list[str] | None = None, target: str | None = None,
                          worker: str | None = None) -> Path:
    """Commit the worktree's candidate/ changes and submit them to the main campaign's inbox (in-session parallel agents)."""
    wt = Campaign(Path(worktree))
    main = find_main_campaign(wt.root)
    worker = worker or wt.root.name
    if not wt.candidate_dirty() and wt.git("log", "--oneline", f"{wt.git('merge-base', 'HEAD', main.head()) or 'HEAD'}..HEAD") == "":
        raise RuntimeError("nothing to propose: candidate/ is identical to the incumbent")
    if wt.candidate_dirty():
        wt.commit_candidate(f"proposal: {description[:72]}")
    record = {"number": wt.store.next_experiment_number(), "description": description, "techniques": list(techniques or []),
              "target": target, "parent_commit": main.load_incumbent().commit}
    path = submit_proposal(main, wt, record, worker)
    if path is None:
        raise RuntimeError("nothing to propose: the diff against the incumbent is empty")
    return path


def list_worktrees(campaign: Campaign) -> list[Path]:
    root = campaign.state_dir / "worktrees"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def remove_worktree(campaign: Campaign, name: str) -> bool:
    path = worktree_path(campaign, name)
    if not path.exists():
        return False
    campaign.git("worktree", "remove", "--force", str(path))
    campaign.git("branch", "-D", f"worker/{name}")
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return True

import json
import subprocess
from pathlib import Path

import pytest

from fastkernel import cli
from fastkernel.agents.worker import create_worktree, lease_target, submit_proposal, worktree_path
from fastkernel.campaign import Campaign, Incumbent
from fastkernel.util import write_json


def _git_ok() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _git_ok(), reason="git required")
def test_worktree_lease_and_proposal(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "mimi", "--dir", str(tmp_path / "c")])
    main = Campaign(tmp_path / "c")
    main.store.save_experiment({"number": 0, "name": "baseline", "status": "baseline", "description": "baseline", "primary_value": 10.0})
    main.save_incumbent(Incumbent(number=0, commit=main.head(), value=10.0, noise_floor=0.01))
    write_json(main.hotspots_path, {"targets": [{"id": "t_a", "rank": 1, "title": "A"}, {"id": "t_b", "rank": 2, "title": "B"}]})
    main.plan_path.write_text("# plan\n", encoding="utf-8")

    wt = create_worktree(main, "w1")
    assert wt.root == worktree_path(main, "w1") and (wt.root / "candidate" / "__init__.py").exists()
    assert (wt.root / ".fast-kernel" / "incumbent.json").exists() and (wt.root / "PLAN.md").exists()
    assert wt.load_incumbent().number == 0 and wt.store.get_experiment(0) is not None
    assert "worker/w1" in wt.git("branch", "--show-current")

    # leases: two workers get different targets, a third gets none
    assert lease_target(main, "w1")["id"] == "t_a"
    assert lease_target(main, "w2")["id"] == "t_b"
    assert lease_target(main, "w3") is None
    main.store.release_lease("t_a", "w1")
    assert lease_target(main, "w3")["id"] == "t_a"

    # a local keep in the worktree becomes a proposal in the main inbox
    (wt.root / "candidate" / "kernels" / "fast.py").write_text("FAST = True\n", encoding="utf-8")
    commit = wt.commit_candidate("exp 1: fast kernel", tag="exp-1")
    record = {"number": 1, "status": "keep", "description": "fast kernel", "techniques": ["cuda-graphs"], "target": "t_a",
              "primary_value": 5.0, "improvement": 0.5, "parent_commit": main.head()}
    patch = submit_proposal(main, wt, record, "w1")
    assert patch is not None and patch.exists()
    meta = json.loads(patch.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["worker"] == "w1" and meta["target"] == "t_a" and meta["worktree_commit"] == commit
    assert "fast.py" in patch.read_text(encoding="utf-8")
    # the main campaign can apply it cleanly on the incumbent
    check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=main.root, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    events = [e["kind"] for e in main.store.events_after(0, limit=500)]
    assert "inbox.submitted" in events and "lease" in events

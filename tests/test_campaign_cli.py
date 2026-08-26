import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fastkernel import cli
from fastkernel.campaign import Campaign
from fastkernel.dashboard.data import campaign_state
from fastkernel.dashboard.report import build_report

ROOT = Path(__file__).resolve().parents[1]


def _git_ok() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _git_ok(), reason="git required")
def test_init_status_report(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "mimi", "--dir", str(tmp_path / "c"), "--set", "model_args.seconds=2", "--precision", "tolerant"])
    campaign = Campaign(tmp_path / "c")
    assert campaign.exists and campaign.goal.model == "mimi"
    assert campaign.goal.model_args["seconds"] == 2 and campaign.goal.gates.precision == "tolerant"
    assert (campaign.root / "candidate" / "__init__.py").exists() and (campaign.root / ".git").exists()
    assert campaign.head()
    assert not campaign.candidate_dirty()
    (campaign.root / "candidate" / "kernels" / "k.py").write_text("x = 1\n", encoding="utf-8")
    assert campaign.candidate_dirty() and "k.py" in campaign.candidate_diff()
    campaign.restore_candidate()
    assert not campaign.candidate_dirty() and not (campaign.root / "candidate" / "kernels" / "k.py").exists()
    assert campaign.is_protected(campaign.root / "GOAL.md") and campaign.is_protected(campaign.root / "experiments" / "x")
    assert not campaign.is_protected(campaign.root / "candidate" / "__init__.py")
    cli.main(["--campaign", str(campaign.root), "status"])
    out = capsys.readouterr().out
    assert "campaign c (mimi)" in out
    campaign.store.save_experiment({"number": 0, "name": "baseline", "status": "baseline", "description": "baseline", "primary_value": 10.0,
                                    "kernel_count": 1000, "top_targets": [{"id": "t", "title": "Whole workload", "fraction": 0.8, "amdahl_gain": 0.5, "category": "launch-bound"}]})
    campaign.store.save_experiment({"number": 1, "name": "graphs", "status": "keep", "description": "cuda graphs", "primary_value": 4.0,
                                    "parent": 0, "improvement": 0.6, "speedup_vs_baseline": 2.5, "kernel_count": 3})
    state = campaign_state(campaign)
    assert len(state["experiments"]) == 2 and state["experiments"][1]["speedup_vs_baseline"] == 2.5
    out = build_report(campaign, tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")
    assert "window.__FK_DATA__" in html and "cuda graphs" in html
    assert Campaign.discover_all(tmp_path)[0].root == campaign.root
    found = Campaign.find(campaign.root / "candidate")
    assert found is not None and found.root == campaign.root


def test_init_custom_no_git(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "custom", "--dir", str(tmp_path / "custom"), "--no-git"])
    assert (tmp_path / "custom" / "spec.py").exists()
    assert json.loads((tmp_path / "custom" / ".fast-kernel" / "state.db").exists() and "true")


def test_hooks_protect_and_loop_guard(tmp_path: Path):
    hooks = ROOT / ".claude" / "hooks"
    campaign = tmp_path / "campaigns" / "demo"
    (campaign / "candidate").mkdir(parents=True)
    (campaign / "GOAL.md").write_text("---\nmodel: custom\n---\n", encoding="utf-8")
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(tmp_path))

    def run_hook(name, payload):
        return subprocess.run([sys.executable, str(hooks / name)], input=json.dumps(payload), capture_output=True, text=True, env=env)

    blocked = run_hook("protect_paths.py", {"tool_input": {"file_path": str(campaign / "GOAL.md")}})
    assert blocked.returncode == 2 and "protected" in blocked.stderr
    allowed = run_hook("protect_paths.py", {"tool_input": {"file_path": str(campaign / "candidate" / "__init__.py")}})
    assert allowed.returncode == 0
    # no loop flag -> stop allowed
    res = run_hook("loop_guard.py", {"cwd": str(tmp_path)})
    assert res.returncode == 0 and res.stdout.strip() == ""
    (campaign / ".fast-kernel").mkdir()
    (campaign / ".fast-kernel" / "loop.active").write_text("1")
    res = run_hook("loop_guard.py", {"cwd": str(tmp_path)})   # 1st stop without progress -> block
    assert json.loads(res.stdout)["decision"] == "block"
    res = run_hook("loop_guard.py", {"cwd": str(tmp_path)})   # 2nd -> block again
    assert json.loads(res.stdout)["decision"] == "block"
    res = run_hook("loop_guard.py", {"cwd": str(tmp_path)})   # 3rd consecutive stop without progress -> allowed to end
    assert res.stdout.strip() == "" and res.returncode == 0
    from fastkernel.store import Store
    store = Store(campaign / ".fast-kernel" / "state.db")
    store.save_experiment({"number": 0, "name": "baseline", "status": "baseline", "description": "baseline"})
    store.close()
    res = run_hook("loop_guard.py", {"cwd": str(tmp_path)})   # progress since last stop -> block with the new experiment in the reason
    payload = json.loads(res.stdout)
    assert payload["decision"] == "block" and "#0" in payload["reason"]
    (campaign / ".fast-kernel" / "stop").write_text("1")
    res = run_hook("loop_guard.py", {"cwd": str(tmp_path)})   # stop flag -> never block
    assert res.stdout.strip() == ""
    (campaign / ".fast-kernel" / "stop").unlink()
    headless = subprocess.run([sys.executable, str(hooks / "loop_guard.py")], input=json.dumps({"cwd": str(tmp_path)}), capture_output=True,
                              text=True, env=dict(env, FK_HEADLESS="1"))
    assert headless.stdout.strip() == "" and headless.returncode == 0   # headless drivers own the loop
    session = run_hook("session_start.py", {"cwd": str(tmp_path)})
    assert "demo" in session.stdout

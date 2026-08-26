from pathlib import Path

from fastkernel.agents.driver import _handle_stream_message, find_project_root
from fastkernel.campaign import Campaign


def test_stream_messages_are_recorded(tmp_path: Path):
    (tmp_path / "candidate").mkdir()
    (tmp_path / "GOAL.md").write_text("---\nmodel: custom\n---\n", encoding="utf-8")
    campaign = Campaign(tmp_path)
    summary = {"agent": "claude", "started_at": "now", "turns": 0, "tool_uses": 0, "text": []}
    messages = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "reading state"},
                                                      {"type": "tool_use", "name": "Bash", "input": {"command": "fast-kernel status --brief"}}]}},
        {"type": "user", "message": {"content": []}},
        {"type": "result", "result": "done", "total_cost_usd": 0.5, "duration_ms": 1000, "num_turns": 3, "is_error": True, "subtype": "error_max_turns"},
    ]
    for msg in messages:
        _handle_stream_message(campaign, "claude", msg, summary, verbose=False)
    assert summary["turns"] == 1 and summary["tool_uses"] == 1 and summary["subtype"] == "error_max_turns"
    kinds = [e["kind"] for e in campaign.store.events_after(0)]
    assert kinds.count("agent.text") == 1 and kinds.count("agent.tool") == 1 and kinds.count("agent.result") == 1
    # the iteration summary must be loggable even though it carries an "agent" key
    campaign.store.event("agent.iteration", **{k: v for k, v in summary.items() if k != "text"})
    assert campaign.store.agents()[0]["state"] == "running"


def test_find_project_root(tmp_path: Path):
    root = tmp_path / "repo"
    (root / ".claude" / "skills" / "fk-experiment").mkdir(parents=True)
    campaign = root / "campaigns" / "x"
    campaign.mkdir(parents=True)
    assert find_project_root(campaign) == root
    lonely = tmp_path / "elsewhere"
    lonely.mkdir()
    assert find_project_root(lonely) == lonely

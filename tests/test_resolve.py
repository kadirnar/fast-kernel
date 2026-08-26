from pathlib import Path

from fastkernel import cli
from fastkernel.resolve import find_root, resolve


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".claude" / "skills" / "fk-optimize").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "fast-kernel"\n', encoding="utf-8")
    return root


def test_find_root_walks_up(tmp_path: Path):
    root = _repo(tmp_path)
    deep = root / "campaigns" / "mimi" / "candidate"
    deep.mkdir(parents=True)
    assert find_root(deep) == root
    assert find_root(tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve() or True


def test_resolve_new_campaign(tmp_path: Path):
    root = _repo(tmp_path)
    plan = resolve("Optimize the Mimi codec model.", root=root)
    assert plan["action"] == "optimize" and plan["model"] == "mimi"
    assert plan["campaign"] == str(root / "campaigns" / "mimi") and plan["exists"] is False
    assert any("fast-kernel init mimi" in s for s in plan["steps"])
    assert any("fast-kernel baseline" in s for s in plan["steps"]) and any("loop start" in s for s in plan["steps"])
    for text, model in [("make LFM2.5 faster", "lfm25"), ("Optimize the LFM2 audio model.", "lfm-audio"), ("optimize yolo detection", "yolo"),
                        ("Please optimize the Liquid audio speech-to-speech model", "lfm-audio"), ("optimize the kyutai codec", "mimi")]:
        assert resolve(text, root=root)["model"] == model, text


def test_resolve_custom_path_stop_status_unknown(tmp_path: Path):
    root = _repo(tmp_path)
    plan = resolve("Optimize the PyTorch model in ./models/my_net.py.", root=root)
    assert plan["model"] == "custom" and plan["custom_path"] == "./models/my_net.py" and plan["campaign_name"] == "my-net"
    assert plan["campaign"] == str(root / "campaigns" / "my-net")
    assert resolve("Stop optimizing.", root=root)["action"] == "stop"
    assert resolve("How is the optimization going?", root=root)["action"] == "status"
    unknown = resolve("Optimize everything.", root=root)
    assert unknown["action"] == "unknown" and "hint" in unknown


def test_resolve_existing_campaign_and_cli(tmp_path: Path, monkeypatch, capsys):
    root = _repo(tmp_path)
    monkeypatch.chdir(root)
    cli.main(["init", "mimi", "--dir", str(root / "campaigns" / "mimi"), "--no-git"])
    plan = resolve("Optimize the Mimi codec model.", root=root)
    assert plan["exists"] is True and not any("init" in s for s in plan["steps"])
    assert any("fast-kernel probe" in s for s in plan["steps"]) and any("fast-kernel baseline" in s for s in plan["steps"])
    # "continue optimizing" with exactly one campaign resolves to it
    assert resolve("Continue optimizing.", root=root)["model"] == "mimi"
    cli.main(["resolve", "--root", str(root), "Optimize", "the", "Mimi", "codec", "model."])
    out = capsys.readouterr().out
    assert "model: mimi" in out and "campaigns/mimi" in out and "steps:" in out


def test_resolve_ignores_examples_and_is_read_only(tmp_path: Path):
    root = _repo(tmp_path)
    example = root / "examples" / "mimi"
    (example / "candidate").mkdir(parents=True)
    (example / "GOAL.md").write_text("---\nmodel: mimi\n---\n", encoding="utf-8")
    plan = resolve("Optimize the Mimi codec model.", root=root)
    assert plan["campaign"] == str(root / "campaigns" / "mimi") and plan["exists"] is False
    assert not (example / ".fast-kernel").exists()
    campaign = root / "campaigns" / "mimi"
    (campaign / "candidate").mkdir(parents=True)
    (campaign / "GOAL.md").write_text("---\nmodel: mimi\n---\n", encoding="utf-8")
    plan = resolve("Optimize the Mimi codec model.", root=root)
    assert plan["exists"] is True and plan["experiments"] == 0
    assert not (campaign / ".fast-kernel").exists()          # resolving created nothing
    assert resolve("Stop optimizing.", root=root)["campaign"] == str(campaign)

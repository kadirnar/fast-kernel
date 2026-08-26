from pathlib import Path

from fastkernel import results
from fastkernel.config import load_goal
from fastkernel.store import Store


def test_load_goal_defaults(tmp_path: Path):
    goal = tmp_path / "GOAL.md"
    goal.write_text("---\nmodel: yolo\nmin_improvement: 0.02\ngates:\n  precision: tolerant\n---\n# Goal\n", encoding="utf-8")
    cfg = load_goal(goal)
    assert cfg.model == "yolo" and cfg.min_improvement == 0.02 and cfg.gates.precision == "tolerant"
    assert cfg.minimize and cfg.continuous and cfg.bench.repeats == 50
    assert cfg.gates.stages == ["smoke", "shapes", "numerical", "determinism", "edge"]
    assert "GOAL.md" in cfg.protected


def test_results_tsv(tmp_path: Path):
    path = tmp_path / "results.tsv"
    results.append_row(path, exp=0, commit="abc1234", status="baseline", metric="latency_ms", value=19.3, speedup=1.0,
                       peak_vram_gb=0.5, gates="5/5", description="baseline\twith tab")
    results.append_row(path, exp=1, commit="def", status="keep", metric="latency_ms", value=9.1, speedup=2.12, peak_vram_gb=None,
                       gates="5/5", description="graphs")
    rows = results.read_rows(path)
    assert len(rows) == 2 and rows[0]["status"] == "baseline" and rows[1]["speedup"] == "2.120"
    assert "\t" not in rows[0]["description"]


def test_store_roundtrip(tmp_path: Path):
    store = Store(tmp_path / "state.db")
    assert store.next_experiment_number() == 0
    store.save_experiment({"number": 0, "name": "baseline", "status": "baseline", "description": "b"})
    store.save_experiment({"number": 1, "name": "x", "status": "keep", "parent": 0, "description": "x"})
    assert store.next_experiment_number() == 2
    assert [e["number"] for e in store.list_experiments()] == [0, 1]
    store.save_experiment({"number": 1, "name": "x", "status": "discard", "description": "x"})
    assert store.get_experiment(1)["status"] == "discard"
    eid = store.event("experiment.finished", number=1, status="discard")
    assert store.events_after(eid - 1)[0]["payload"]["status"] == "discard"
    store.set_agent("w1", "running", "hello")
    assert store.agents()[0]["state"] == "running"
    assert store.acquire_lease("t_1", "w1") and not store.acquire_lease("t_1", "w2") and store.acquire_lease("t_1", "w1")
    store.release_lease("t_1", "w1")
    assert store.acquire_lease("t_1", "w2")
    store.set("k", {"a": 1})
    assert store.get("k") == {"a": 1}
    store.close()

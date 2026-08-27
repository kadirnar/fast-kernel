from pathlib import Path

from fastkernel import cli, memory
from fastkernel.campaign import Campaign, Incumbent


def test_classify_failure():
    assert memory.classify_failure("RuntimeError: CUDA out of memory. Tried to allocate")["class"] == "oom"
    assert memory.classify_failure("an illegal memory access was encountered")["class"] == "illegal-memory"
    assert memory.classify_failure("ModuleNotFoundError: No module named 'triton'")["class"] == "import"
    assert memory.classify_failure("ptxas fatal : Unresolved extern function")["class"] == "compile"
    assert memory.classify_failure("RuntimeError: The size of tensor a (3) must match")["class"] == "shape"
    # a failed numerical gate outranks the raw log
    gates = {"passed": False, "stages": {"numerical": {"passed": False, "skipped": False}}}
    assert memory.classify_failure("some unrelated log", gates)["class"] == "numerical"


def test_reflexion():
    keep = {"number": 5, "status": "keep", "primary_value": 8.0, "target": "t_a", "techniques": ["x"], "description": "faster"}
    r = memory.reflexion(keep, incumbent_value=10.0, minimize=True)
    assert r["status"] == "keep" and r["delta_pct"] > 19 and "improved" in r["outcome"]
    crash = {"number": 6, "status": "crash", "primary_value": None, "target": "t_a", "techniques": ["y"], "description": "z"}
    rc = memory.reflexion(crash, incumbent_value=10.0, minimize=True, run_log="an illegal memory access")
    assert rc["failure_class"] == "illegal-memory" and "illegal-memory" in rc["outcome"]


def test_record_and_retrieve(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "mimi", "--dir", str(tmp_path / "c")])
    campaign = Campaign(tmp_path / "c")
    target = {"id": "t_a", "category": "gemm", "boundness": "compute", "class": "Linear"}
    memory.record_outcome(campaign, memory.reflexion(
        {"number": 1, "status": "keep", "primary_value": 5.0, "target": "t_a", "techniques": ["triton"],
         "description": "fused", "_target_obj": target}, incumbent_value=10.0, minimize=True))
    memory.record_outcome(campaign, memory.reflexion(
        {"number": 2, "status": "crash", "primary_value": None, "target": "t_a", "techniques": ["cuda-cpp"],
         "description": "oops", "_target_obj": target}, incumbent_value=5.0, minimize=True, run_log="illegal memory access"))
    mem = memory.retrieve(campaign, target)
    assert mem["signature"] == "gemm:compute:Linear"
    assert any(e["number"] == 1 and e["status"] == "keep" for e in mem["similar"])
    assert any(e["failure_class"] == "illegal-memory" for e in mem["repair_chain"])
    assert "illegal-memory" in memory.render_memory(mem)


def test_beam(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "mimi", "--dir", str(tmp_path / "c")])
    campaign = Campaign(tmp_path / "c")
    for n, (val, commit) in enumerate([(10.0, "c0"), (7.0, "c1"), (5.0, "c2"), (5.0, "c2")], start=0):
        campaign.store.save_experiment({"number": n, "status": "keep" if n else "baseline",
                                        "commit": commit, "primary_value": val, "description": f"exp{n}"})
    beam = campaign.beam(3)
    # minimize: best (smallest) first, distinct commits only
    assert [e["number"] for e in beam][:2] == [2, 1]
    assert len({e["commit"] for e in beam}) == len(beam)

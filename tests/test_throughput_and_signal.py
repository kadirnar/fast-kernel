"""Throughput and signal: shapes under CUDA graphs, sequential anchor refinement, the GPU lock, the
plateau streak, `fast-kernel brief`, the 3-way inbox and the headless idle cap."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from fastkernel import cli
from fastkernel.agents.driver import _apply_proposal, _progress_since
from fastkernel.campaign import Campaign, Incumbent
from fastkernel.config import load_goal
from fastkernel.harness import bench
from fastkernel.profiling.rank import build_targets
from fastkernel.util import GpuLock, write_json

DEVICE = {"name": "test-gpu", "measured_bandwidth_gbs": 800.0, "measured_bf16_tflops": 200.0, "launch_latency_us": 4.0}


def _git_ok() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


# ---- ranking: unknown is not "0 % of peak" ---------------------------------------------------

def test_target_without_shapes_reports_unknown_sol_not_zero():
    profile = {"wall_ms": 2.0, "gpu_busy_ms": 1.9, "kernel_count": 20, "avg_kernel_us": 95.0,
               "modules": [{"path": "layers.0.fc", "class": "Linear", "gpu_us": 1000.0, "kernel_count": 10, "category": "gemm",
                            "shapes": {"class": "Linear", "inputs": [{"shape": [1, 512, 1024], "dtype": "float32"}],
                                       "output": {"shape": [1, 512, 4096], "dtype": "float32"}, "params": {"weight": [4096, 1024]}, "calls": 1}},
                           {"path": "Codebook.quantize", "class": "Codebook", "gpu_us": 900.0, "kernel_count": 10, "category": "quantizer",
                            "shapes": {}}]}
    targets = {t["class"]: t for t in build_targets(profile, DEVICE)}
    assert targets["Linear"]["sol_efficiency"] is not None and targets["Linear"]["sol_efficiency"] > 0
    assert targets["Codebook"]["sol_efficiency"] is None and targets["Codebook"]["headroom"] is None
    # ranked by raw share when nothing could be estimated -- still a valid, non-degenerate score
    assert targets["Codebook"]["score"] == pytest.approx(targets["Codebook"]["fraction"])


# ---- config: GOAL.md's bench.anchor* / max_banked are honoured ----------------------------

def test_load_goal_reads_anchor_and_banking_fields(tmp_path: Path):
    goal = tmp_path / "GOAL.md"
    goal.write_text("---\nmodel: mimi\nbench:\n  anchor: false\n  anchor_pairs: 8\n  anchor_max_pairs: 40\n  max_banked: 3\n---\n",
                    encoding="utf-8")
    cfg = load_goal(goal)
    assert cfg.bench.anchor is False and cfg.bench.anchor_pairs == 8
    assert cfg.bench.anchor_max_pairs == 40 and cfg.bench.max_banked == 3
    assert load_goal(tmp_path / "GOAL.md").bench.anchor_max_pairs == 40


# ---- the GPU lock: two measurements never overlap -----------------------------------------

def test_gpu_lock_serialises_measurements(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FAST_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.delenv("FK_GPU_LOCK", raising=False)
    assert GpuLock.path().name == "gpu-7.lock"
    hold = textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
        from fastkernel.util import GpuLock
        with GpuLock():
            open({str(tmp_path / "held")!r}, "w").write("1")
            time.sleep(1.5)
    """)
    proc = subprocess.Popen([sys.executable, "-c", hold], env=dict(os.environ, FAST_KERNEL_CACHE=str(tmp_path), CUDA_VISIBLE_DEVICES="7"))
    try:
        for _ in range(200):
            if (tmp_path / "held").exists():
                break
            time.sleep(0.02)
        assert (tmp_path / "held").exists()
        messages: list[str] = []
        started = time.perf_counter()
        with GpuLock(timeout=10, poll=0.05, log=messages.append) as lock:
            waited = time.perf_counter() - started
            assert lock.acquired and proc.poll() is not None      # the holder finished before we got in
        assert lock.waited >= 1.0 and waited >= 1.0
        assert any("waiting for the GPU" in m for m in messages)
    finally:
        proc.kill()
        proc.wait()


def test_gpu_lock_can_be_disabled(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FAST_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setenv("FK_GPU_LOCK", "0")
    with GpuLock() as lock:
        assert not lock.acquired and lock.waited == 0.0


# ---- the plateau streak ----------------------------------------------------------------------

def test_streak_counts_since_the_last_measured_improvement(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "mimi", "--dir", str(tmp_path / "c"), "--no-git"])
    campaign = Campaign(tmp_path / "c")
    rows = [(0, "baseline", None), (1, "keep", "t_a"), (2, "discard", "t_a"), (3, "remeasure", None), (4, "crash", "t_b"),
            (5, "discard", "t_b"), (6, "discard", "t_b")]
    for n, status, target in rows:
        campaign.store.save_experiment({"number": n, "status": status, "target": target, "description": f"e{n}",
                                        "failure_class": "compile" if status == "crash" else None})
    streak = campaign.streak()
    assert streak["no_improvement"] == 4          # 2, 4, 5, 6 (the re-measurement is neither)
    assert streak["target"] == "t_b" and streak["on_target"] == 3
    assert streak["failure_classes"] == ["compile"]
    campaign.store.save_experiment({"number": 7, "status": "bank", "target": "t_b", "description": "e7"})
    assert campaign.streak()["no_improvement"] == 0


# ---- `fast-kernel brief`: one screen, everything the iteration needs -------------------------

def test_brief_prints_state_streak_targets_memory_and_insights(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "mimi", "--dir", str(tmp_path / "c"), "--no-git"])
    campaign = Campaign(tmp_path / "c")
    campaign.store.save_experiment({"number": 0, "status": "baseline", "description": "baseline", "primary_value": 10.0})
    for n in range(1, 7):
        campaign.store.save_experiment({"number": n, "status": "discard", "target": "t_a", "description": f"idea {n}",
                                        "primary_value": 10.0, "improvement": -0.01, "reason": "no improvement",
                                        "gates": {"passed": True}})
    campaign.save_incumbent(Incumbent(number=0, commit="abc", value=10.0, anchor_ratio=1.0, anchor_uncertainty=0.004, banked=2))
    write_json(campaign.hotspots_path, {"experiment": 0, "summary": {"wall_ms": 10.0, "kernel_count": 900, "gpu_busy_ratio": 0.3},
                                        "targets": [{"id": "t_a", "rank": 1, "title": "Linear (gemm, latency-bound)", "class": "Linear",
                                                     "category": "gemm", "boundness": "latency", "fraction": 0.4, "sol_efficiency": 0.12,
                                                     "kernel_count": 200, "techniques": []},
                                                    {"id": "t_b", "rank": 2, "title": "Conv1d (conv, memory-bound)", "class": "Conv1d",
                                                     "category": "conv", "boundness": "memory", "fraction": 0.3, "sol_efficiency": None,
                                                     "kernel_count": 30, "techniques": []}]})
    from fastkernel import memory
    memory.record_outcome(campaign, memory.reflexion(
        {"number": 3, "status": "crash", "primary_value": None, "target": "t_a", "techniques": ["cuda-cpp"], "description": "boom",
         "_target_obj": {"id": "t_a", "category": "gemm", "boundness": "latency", "class": "Linear"}},
        incumbent_value=10.0, minimize=True, run_log="ptxas fatal error"))
    cli.main(["--campaign", str(campaign.root), "note", "fc1 epilogue is FFMA-bound at 1.2x", "--tags", "gemm"])
    capsys.readouterr()
    cli.main(["--campaign", str(campaign.root), "brief"])
    out = capsys.readouterr().out
    assert "incumbent #0" in out and "2 banked" in out
    assert "6 consecutive experiment(s) without a measured improvement (6 in a row on target t_a)" in out and "PLATEAU" in out
    assert "SOL 12%" in out and "SOL n/a" in out and "id=t_a" in out
    assert "repair: exp #3 failed with compile" in out
    assert "#6   discard" in out and "FFMA-bound" in out
    assert "fast-kernel eval -m" in out


# ---- sequential refinement of the paired comparison ------------------------------------------

torch = pytest.importorskip("torch")


def _spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_compare_callables_adds_pairs_only_while_the_verdict_is_unresolved():
    calls = {"n": 0}

    def never(ratio, unc, cand_ms=float('nan')):
        calls["n"] += 1
        return False

    out = bench.compare_callables(lambda: _spin(0.0005), lambda: _spin(0.0005), warmup=1, pairs=4, ramp_seconds=0.01,
                                  max_pairs=12, resolved=never)
    assert out["pairs"] == 12 and out["rounds"] == 3 and calls["n"] == 2

    def at_once(ratio, unc, cand_ms=float('nan')):
        return True

    out = bench.compare_callables(lambda: _spin(0.0005), lambda: _spin(0.0005), warmup=1, pairs=4, ramp_seconds=0.01,
                                  max_pairs=12, resolved=at_once)
    assert out["pairs"] == 4 and out["rounds"] == 1
    # without a rule the comparison is exactly one batch, whatever max_pairs says
    out = bench.compare_callables(lambda: _spin(0.0005), lambda: _spin(0.0005), warmup=1, pairs=4, ramp_seconds=0.01, max_pairs=12)
    assert out["pairs"] == 4 and out["rounds"] == 1


def test_decision_rule_is_resolved_only_away_from_both_boundaries(tmp_path: Path):
    from fastkernel.config import GoalConfig
    from fastkernel.harness.run import _decision_resolved
    (tmp_path / ".fast-kernel").mkdir()
    write_json(tmp_path / ".fast-kernel" / "incumbent.json", {"anchor_ratio": 2.0, "anchor_uncertainty": 0.004})
    goal = GoalConfig()
    goal.min_improvement = 0.01
    resolved = _decision_resolved(tmp_path, goal)
    assert resolved is not None
    assert resolved(2.0 * 1.05, 0.004)          # a clear 5 % win
    assert resolved(2.0 * 0.95, 0.004)          # a clear 5 % loss
    assert not resolved(2.0 * 1.002, 0.004)     # +0.2 %: could be 0 -> keep measuring
    assert not resolved(2.0 * 1.011, 0.006)     # +1.1 % vs a ~1 % threshold: could be a bank -> keep measuring
    assert resolved(2.0 * 1.005, 0.001)         # +0.5 %, well inside (0, 1 %): a certain bank, stop
    write_json(tmp_path / ".fast-kernel" / "incumbent.json", {"anchor_ratio": None})
    assert _decision_resolved(tmp_path, goal) is None


# ---- shapes are captured through CUDA-graph replays ------------------------------------------

def test_profile_captures_shapes_through_the_attribution_context():
    from fastkernel.backends import graphs
    from fastkernel.profiling.trace import profile_workload
    model = torch.nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    cached = model(x).detach()

    def run_fn(m, inputs):
        # a graph-replaying candidate: outside eager mode the module's forward is never entered
        return m(inputs["x"]) if graphs.EAGER else cached * 1.0

    prof = profile_workload(model, run_fn, {"x": x}, model, warmup=1, attribution_context=graphs.eager_mode)
    assert prof["module_shapes_count"] >= 1
    prof_without = profile_workload(model, run_fn, {"x": x}, model, warmup=1, attribution_context=None)
    assert prof_without["module_shapes_count"] == 0


# ---- headless progress accounting and the 3-way inbox ----------------------------------------

def test_progress_since_classifies_an_iteration():
    assert _progress_since([], 3) == "nothing"
    assert _progress_since([{"number": 2, "status": "keep"}], 3) == "nothing"
    assert _progress_since([{"number": 3, "status": "bank"}], 3) == "improved"
    assert _progress_since([{"number": 3, "status": "discard"}], 3) == "tried"
    assert _progress_since([{"number": 3, "status": "remeasure"}], 3) == "remeasure"


@pytest.mark.skipif(not _git_ok(), reason="git required")
def test_inbox_merges_a_proposal_three_way_when_the_incumbent_moved(tmp_path: Path, monkeypatch):
    from fastkernel.agents.worker import create_worktree, submit_proposal
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "mimi", "--dir", str(tmp_path / "c")])
    main = Campaign(tmp_path / "c")
    init = main.root / "candidate" / "__init__.py"
    # three neighbouring lines: any edit's context covers the other two, so a plain apply fails as soon
    # as the incumbent touched one of them -- even though the edits themselves do not overlap
    init.write_text("A = 1\nB = 1\nC = 1\n\n\ndef apply(model, ctx):\n    return model\n", encoding="utf-8")
    main.commit_candidate("exp 0: baseline", tag="exp-0")
    main.save_incumbent(Incumbent(number=0, commit=main.head(), value=10.0))
    wt = create_worktree(main, "w1")
    wt_init = wt.root / "candidate" / "__init__.py"
    wt_init.write_text(wt_init.read_text().replace("A = 1", "A = 2"), encoding="utf-8")
    wt.commit_candidate("exp 1: A")
    patch = submit_proposal(main, wt, {"number": 1, "description": "A", "techniques": [], "target": "t_a"}, "w1")
    # meanwhile the incumbent moved: another keep changed the neighbouring line
    init.write_text(init.read_text().replace("C = 1", "C = 2"), encoding="utf-8")
    main.commit_candidate("exp 2: C", tag="exp-2")
    assert subprocess.run(["git", "apply", "--check", str(patch)], cwd=main.root, capture_output=True).returncode != 0
    assert _apply_proposal(main, patch) == "ok"
    text = init.read_text()
    assert "A = 2" in text and "C = 2" in text
    assert main.candidate_dirty() and "A = 2" in main.candidate_diff()     # unstaged, so the harness sees the diff
    main.restore_candidate()
    assert "A = 1" in init.read_text()
    # a genuine overlap is rejected and the tree is left clean on the incumbent
    wt_init.write_text(wt_init.read_text().replace("C = 1", "C = 3"), encoding="utf-8")
    wt.commit_candidate("exp 3: C too")
    init.write_text(init.read_text().replace("C = 2", "C = 4"), encoding="utf-8")
    main.commit_candidate("exp 4: C again")
    conflicting = submit_proposal(main, wt, {"number": 3, "description": "clash", "techniques": [], "target": "t_a"}, "w1")
    reason = _apply_proposal(main, conflicting)
    assert reason != "ok" and "rebase" in reason and not main.candidate_dirty()
    assert "C = 4" in init.read_text() and "A = 1" in init.read_text()
    meta = json.loads(conflicting.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["worker"] == "w1"

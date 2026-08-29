"""The accept/bank/reject rule and the paired measurement it is built on."""
from __future__ import annotations

import time

import pytest

from fastkernel.campaign import Incumbent
from fastkernel.config import GoalConfig
from fastkernel.harness import bench
from fastkernel.harness.evaluate import decide_improvement


def _goal(**kw) -> GoalConfig:
    g = GoalConfig()
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def _anchor(ratio: float, unc: float = 0.0) -> dict:
    return {"anchor": {"ratio": ratio, "ratio_uncertainty": unc}}


# ---- what gets compared -----------------------------------------------------------------

def test_anchored_ratio_cancels_between_session_drift():
    """A candidate measured while the machine ran 10 % slower is still recognised as 5 % faster."""
    inc = Incumbent(number=1, value=10.0, anchor_ratio=2.0, anchor_uncertainty=0.001)
    # every absolute time in this session is 10 % worse, so raw ms says the candidate regressed...
    slow_value = 10.0 * 1.10 / 1.05
    raw = decide_improvement(_goal(), inc, {}, slow_value)
    assert raw["improvement"] < 0 and raw["status"] == "discard"
    # ...but the reference was equally slow, so the anchored ratio still shows the real +5 %
    anchored = decide_improvement(_goal(), inc, _anchor(2.0 * 1.05, 0.001), slow_value)
    assert anchored["decision_basis"].startswith("anchored")
    assert anchored["improvement"] == pytest.approx(0.05, abs=1e-9)
    assert anchored["status"] == "keep"


def test_threshold_comes_from_measurement_uncertainty_not_a_frozen_baseline_number():
    """A one-off unlucky baseline pair must not set the bar for the whole campaign."""
    inc = Incumbent(number=1, value=10.0, noise_floor=0.06,      # the old, inflated floor
                    anchor_ratio=2.0, anchor_uncertainty=0.004)
    d = decide_improvement(_goal(min_improvement=0.001), inc, _anchor(2.0 * 1.02, 0.003), 9.8)
    # threshold is hypot(0.003, 0.004) = 0.5 %, not the stale 6 %
    assert d["threshold"] == pytest.approx(0.005, abs=1e-9)
    assert d["improvement"] == pytest.approx(0.02, abs=1e-9)
    assert d["status"] == "keep"


def test_falls_back_to_raw_milliseconds_without_an_anchor():
    inc = Incumbent(number=1, value=10.0, noise_floor=0.02)
    d = decide_improvement(_goal(), inc, {}, 9.0)
    assert d["decision_basis"].startswith("raw milliseconds")
    assert d["improvement"] == pytest.approx(0.10) and d["status"] == "keep"


# ---- banking: small real wins are accumulated, not thrown away --------------------------

def test_real_gain_below_the_floor_is_banked_not_discarded():
    inc = Incumbent(number=1, value=10.0, anchor_ratio=2.0, anchor_uncertainty=0.004)
    d = decide_improvement(_goal(min_improvement=0.001), inc, _anchor(2.0 * 1.004, 0.004), 9.96)
    assert d["status"] == "bank" and 0 < d["improvement"] < d["threshold"]


def test_a_regression_is_never_banked():
    inc = Incumbent(number=1, value=10.0, anchor_ratio=2.0, anchor_uncertainty=0.004)
    d = decide_improvement(_goal(min_improvement=0.001), inc, _anchor(2.0 * 0.998, 0.004), 10.02)
    assert d["status"] == "discard"


def test_banking_stops_at_max_banked_so_the_tree_cannot_drift_forever():
    goal = _goal(min_improvement=0.001)
    goal.bench.max_banked = 3
    metrics, value = _anchor(2.0 * 1.004, 0.004), 9.96
    assert decide_improvement(goal, Incumbent(value=10.0, anchor_ratio=2.0, anchor_uncertainty=0.004,
                                              banked=2), metrics, value)["status"] == "bank"
    assert decide_improvement(goal, Incumbent(value=10.0, anchor_ratio=2.0, anchor_uncertainty=0.004,
                                              banked=3), metrics, value)["status"] == "discard"


def test_banked_gains_accumulate_until_they_clear_the_floor_together():
    """0.2 % wins are individually unmeasurable; three of them together are a keep.

    This is the mimi campaign's failure mode: each of them was discarded on its own, so the work
    was done, measured, and then thrown away. The incumbent stays put while they bank, so the
    comparison is always against the last number the campaign can actually defend.
    """
    inc = Incumbent(number=1, value=10.0, anchor_ratio=2.0, anchor_uncertainty=0.004)
    goal = _goal(min_improvement=0.001)
    ratio, banked = 2.0, 0
    for _ in range(2):
        ratio *= 1.002
        d = decide_improvement(goal, inc, _anchor(ratio, 0.004), None)
        assert d["status"] == "bank", d
        banked += 1
        inc = Incumbent(number=inc.number, value=inc.value, anchor_ratio=inc.anchor_ratio,
                        anchor_uncertainty=inc.anchor_uncertainty, banked=banked)
    ratio *= 1.002
    final = decide_improvement(goal, inc, _anchor(ratio, 0.004), None)
    assert final["status"] == "keep"
    assert final["improvement"] == pytest.approx(1.002 ** 3 - 1, abs=1e-9)


def test_simpler_still_keeps_equal_speed_code():
    inc = Incumbent(number=1, value=10.0, noise_floor=0.02)
    assert decide_improvement(_goal(), inc, {}, 10.05, simpler=True)["status"] == "keep"


# ---- the measurement primitives ---------------------------------------------------------

torch = pytest.importorskip("torch")


def _spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_compare_callables_recovers_a_known_ratio_under_drift():
    """`b` takes half as long as `a`; a machine that keeps slowing down must not change that.

    This is the whole point of pairing: the drift lands on both members of every pair, so it
    divides out of the ratio instead of being charged to whichever ran later.
    """
    state = {"n": 0}

    def work(seconds: float):
        def fn():
            state["n"] += 1
            drift = 1.0 + 0.5 * min(state["n"], 300) / 300.0   # up to 50 % slower over the run
            _spin(seconds * drift)
        return fn

    out = bench.compare_callables(work(0.002), work(0.001), warmup=2, pairs=16, ramp_seconds=0.02)
    assert out["pairs"] == 16 and out["inner_a"] >= 1 and out["inner_b"] >= 1
    assert 1.7 < out["ratio"] < 2.3
    assert out["ratio_uncertainty"] < 0.15
    # the raw medians drifted upward; only the ratio is trustworthy across the run
    assert out["a_median_ms"] > 2.0 and out["b_median_ms"] > 1.0


def test_self_noise_reports_a_ratio_of_one_and_a_small_floor():
    """The harness measured against itself: the true answer is 1.0, so anything else is error."""
    out = bench.self_noise(lambda: _spin(0.002), warmup=2, pairs=16, ramp_seconds=0.02)
    assert out["ratio"] == pytest.approx(1.0, abs=0.1)
    assert 0.0 <= out["noise"] < 0.2


def test_self_noise_is_not_fooled_by_an_unequal_warm_up():
    """The old estimator timed a warm run against a cold one and called the difference noise.

    `self_noise` runs both sides interleaved under identical conditions, so a workload that is
    slow only for its first few calls no longer inflates the campaign's acceptance threshold.
    """
    state = {"n": 0}

    def cold_start():
        state["n"] += 1
        _spin(0.006 if state["n"] <= 4 else 0.002)   # first calls are 3x slower

    out = bench.self_noise(cold_start, warmup=5, pairs=16, ramp_seconds=0.02)
    assert out["noise"] < 0.2      # the cold calls are absorbed by warm-up, not booked as noise


def test_auto_inner_batches_short_calls_and_leaves_long_ones_alone():
    assert bench.auto_inner(lambda: _spin(0.00005), target_ms=5.0) > 1
    assert bench.auto_inner(lambda: _spin(0.01), target_ms=5.0) == 1


# ---- the two bases must agree, and when they do not the run says so ----------------------

def test_a_contested_reading_is_named_in_the_verdict_not_hidden():
    """Anchored and raw disagreeing on the SIGN is the campaign's worst failure mode, so it is said.

    In campaigns/mimi a change was kept at +4.21 % anchored that the profiler put at +18.7 us of
    gpu_busy and five absolute readings put at 1.45 % slower; its revert was then discarded at
    -1.07 % anchored while the same run measured 1.406 against an incumbent of 1.432. Anchoring
    still decides -- it exists so drift cannot veto a real gain -- but the disagreement is now
    printed, and `_decision_resolved` spends more pairs on it before the verdict is taken.
    """
    inc = Incumbent(number=1, value=10.0, anchor_ratio=2.0, anchor_uncertainty=0.001)
    # anchored says +5 % while the candidate is slower in raw milliseconds
    d = decide_improvement(_goal(), inc, _anchor(2.0 * 1.05, 0.001), 10.5)
    assert d["contested"] is True
    assert d["raw_improvement"] < 0 < d["improvement"]
    assert "CONTESTED" in d["reason"]
    # and when they agree, nothing is flagged
    ok = decide_improvement(_goal(), inc, _anchor(2.0 * 1.05, 0.001), 9.5)
    assert ok["contested"] is False and "CONTESTED" not in ok["reason"]


def test_gpu_busy_ratio_uses_the_benchmarked_latency_not_the_profiler_s_own_wall():
    """The profile times its own call for a denominator; the benchmark times it properly."""
    from fastkernel.harness.evaluate import _busy_ratio
    prof = {"gpu_busy_ms": 1.3418, "gpu_busy_ratio": 0.836}      # 0.836 came from a 1.605 ms wall
    assert _busy_ratio(prof, 1.3504) == pytest.approx(0.9936, abs=1e-3)
    assert _busy_ratio({"gpu_busy_ratio": 0.836}, None) == 0.836  # no benchmark yet -> unchanged

"""A build lock left behind by a killed run must not hang every later run."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from fastkernel.backends.cuda_cpp import _clear_stale_lock, _live_build_elsewhere, _own_process_tree


def _build(tmp_path: Path) -> Path:
    build = tmp_path / "campaign" / ".fast-kernel" / "build" / "fk_conv"
    build.mkdir(parents=True)
    return build


def test_no_lock_is_a_no_op(tmp_path: Path):
    assert _clear_stale_lock(_build(tmp_path)) is None


def test_stale_lock_is_broken_and_reported(tmp_path: Path):
    build = _build(tmp_path)
    lock = build / "lock"
    lock.touch()
    message = _clear_stale_lock(build)
    assert message and "stale build lock" in message
    assert not lock.exists()


def test_our_own_wrapper_processes_are_not_mistaken_for_a_competing_build(tmp_path: Path):
    """`uv run ... harness.run --campaign <root>` is an ancestor of the build, not a rival.

    Getting this wrong is worse than having no guard at all: the lock is never broken and the run
    hangs exactly as it did before.
    """
    build = _build(tmp_path)
    (build / "lock").touch()
    mine = _own_process_tree()
    assert len(mine) > 1                      # the test runner plus at least one ancestor
    campaign = str(build.parents[2].resolve())
    assert not _live_build_elsewhere(campaign, mine)
    assert _clear_stale_lock(build, wait_seconds=1.0) is not None


def test_a_live_build_is_given_time_before_the_lock_is_broken(tmp_path: Path):
    """The guard must not steal the lock from a genuine concurrent build straight away."""
    build = _build(tmp_path)
    (build / "lock").touch()
    campaign = str(build.parents[2].resolve())
    # a process whose command line looks like the harness building this very campaign
    proc = subprocess.Popen([sys.executable, "-c",
                             f"import time; time.sleep(30)  # harness.run --campaign {campaign}"])
    try:
        for _ in range(100):         # it must be visible in /proc before we look
            if _live_build_elsewhere(campaign, _own_process_tree()):
                break
            time.sleep(0.02)
        assert _live_build_elsewhere(campaign, _own_process_tree())
        started = time.time()
        message = _clear_stale_lock(build, wait_seconds=3.0)
        waited = time.time() - started
        assert waited >= 2.0                  # it waited for the other build instead of barging in
        assert message and "stale build lock" in message   # then broke it rather than hanging
    finally:
        proc.kill()
        proc.wait()

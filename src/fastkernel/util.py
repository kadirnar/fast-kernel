from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def ensure_package(module: str, pip_spec: str | None = None) -> bool:
    """Import `module`; if it is missing, auto-install `pip_spec` (default: `module`) into the active
    venv and retry. Returns True if the module is importable afterwards. The agent never has to ask a
    human to install anything; install time is not part of any measured latency (compile/setup is
    excluded from the benchmark)."""
    import importlib
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        pass
    spec = pip_spec or module
    commands = []
    uv = shutil.which("uv")
    if uv:
        commands.append([uv, "pip", "install", spec])
    commands.append([sys.executable, "-m", "pip", "install", spec])
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except (subprocess.SubprocessError, OSError):
            continue
        if proc.returncode == 0:
            break
    importlib.invalidate_caches()
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


class CommandResult:
    def __init__(self, argv: list[str], returncode: int, stdout: str, stderr: str, seconds: float, timed_out: bool):
        self.argv = argv
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.seconds = seconds
        self.timed_out = timed_out

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run(argv: list[str], cwd: Path | None = None, timeout: float | None = None, env: dict[str, str] | None = None,
        input_text: str | None = None) -> CommandResult:
    """Run argv (never a shell string) with a timeout; kill the whole process group on timeout."""
    started = time.perf_counter()
    merged = dict(os.environ)
    if env:
        merged.update(env)
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout,
            env=merged, input=input_text, start_new_session=True,
        )
        return CommandResult(argv, proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - started, False)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(argv, -1, out, err, time.perf_counter() - started, True)
    except FileNotFoundError as exc:
        return CommandResult(argv, 127, "", str(exc), time.perf_counter() - started, False)


def which(name: str) -> str | None:
    return shutil.which(name)


class GpuLock:
    """Machine-wide mutual exclusion for GPU *measurements*.

    Agents think in parallel, but the GPU can only be measured by one harness at a time: a second
    benchmark, gate run or autotune sharing the device inflates every number of the first one, and a
    worker would then discard a real win -- or submit an imaginary one -- on contaminated timings.
    Held around the whole harness subprocess (build, gates, anchored comparison, benchmark, profile)
    by `fast-kernel eval` / `baseline` / `profile` / `probe`, so waiting never counts against the
    experiment's own timeout. `FK_GPU_LOCK=0` disables it (single-agent runs lose nothing either way).

    Implemented with an advisory `flock` on a file under ~/.cache/fast-kernel/locks, keyed by
    CUDA_VISIBLE_DEVICES, so campaigns, worktrees and headless workers on the same device all agree.
    """

    def __init__(self, timeout: float | None = None, poll: float = 0.5, log=None):
        self.timeout = timeout
        self.poll = poll
        self.log = log or (lambda message: print(message, flush=True))
        self.waited = 0.0
        self.acquired = False
        self._fh = None
        self.enabled = os.environ.get("FK_GPU_LOCK", "1") not in ("0", "false", "no")

    @staticmethod
    def path() -> Path:
        base = Path(os.environ.get("FAST_KERNEL_CACHE") or Path.home() / ".cache" / "fast-kernel") / "locks"
        base.mkdir(parents=True, exist_ok=True)
        devices = (os.environ.get("CUDA_VISIBLE_DEVICES") or "all").replace(",", "_").replace("/", "_")
        return base / f"gpu-{devices}.lock"

    def __enter__(self) -> GpuLock:
        if not self.enabled:
            return self
        try:
            import fcntl
        except ImportError:          # not a POSIX host: no locking, no failure
            return self
        try:
            self._fh = open(self.path(), "a+", encoding="utf-8")
        except OSError:
            return self
        started = time.perf_counter()
        announced = False
        while True:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.acquired = True
                break
            except OSError:
                pass
            waited = time.perf_counter() - started
            if self.timeout is not None and waited >= self.timeout:
                self.log(f"[fast-kernel] GPU still busy after {waited:.0f} s; measuring without the lock")
                break
            if not announced:
                holder = ""
                try:
                    self._fh.seek(0)
                    holder = self._fh.read().strip()[:120]
                except OSError:
                    pass
                self.log("[fast-kernel] waiting for the GPU: another measurement is running"
                         + (f" ({holder})" if holder else "") + " -- this wait is not charged to the experiment")
                announced = True
            time.sleep(self.poll)
        self.waited = time.perf_counter() - started
        if self.acquired:
            try:
                self._fh.seek(0)
                self._fh.truncate()
                self._fh.write(f"pid={os.getpid()} cwd={os.getcwd()} since={now_iso()}")
                self._fh.flush()
            except OSError:
                pass
        return self

    def __exit__(self, *exc) -> bool:
        if self._fh is not None:
            try:
                if self.acquired:
                    import fcntl
                    self._fh.seek(0)
                    self._fh.truncate()
                    fcntl.flock(self._fh, fcntl.LOCK_UN)
            except (OSError, ImportError):
                pass
            self._fh.close()
            self._fh = None
        return False


def python_executable() -> str:
    return sys.executable


def human_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    if seconds < 90:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.{digits}g}"
    return str(value)


def slugify(text: str, max_len: int = 48) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")[:max_len] or "exp"


def tail_text(text: str, lines: int = 60) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:])

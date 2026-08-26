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

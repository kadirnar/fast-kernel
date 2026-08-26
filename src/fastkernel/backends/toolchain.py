"""Self-contained CUDA toolchains from pip wheels (no system install, no sudo).

The CUDA runtime that torch ships is fixed (e.g. 13.0 for torch cu130) but its nvcc frontend may not
accept the machine's host compiler (a newer nvcc minor version usually does). A toolchain is the wheel set
{nvcc, cccl, crt, nvvm, runtime headers} of one CUDA version installed into an isolated directory:

    fast-kernel toolchain install --cuda 13.3      -> ~/.cache/fast-kernel/toolchains/cuda-13.3/nvidia/cu13

`find_nvcc()` prefers installed toolchains (newest first) over the venv wheels and the system, so
TileLang / CUDA C++ compile against a frontend that understands the host compiler while the kernels
still run on torch's runtime (same major version).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ..util import run, which

WHEELS = ["nvidia-cuda-nvcc", "nvidia-cuda-cccl", "nvidia-cuda-crt", "nvidia-nvvm", "nvidia-cuda-runtime"]


def toolchain_root() -> Path:
    env = os.environ.get("FAST_KERNEL_TOOLCHAINS")
    root = Path(env) if env else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "fast-kernel" / "toolchains"
    return root


def toolchain_home(version: str) -> Path:
    return toolchain_root() / f"cuda-{version}" / "nvidia" / "cu13"


def list_toolchains() -> list[tuple[str, Path]]:
    root = toolchain_root()
    if not root.exists():
        return []
    found = []
    for entry in sorted(root.iterdir(), reverse=True):
        if entry.is_dir() and entry.name.startswith("cuda-"):
            for cu in sorted(entry.glob("nvidia/cu*"), reverse=True):
                if (cu / "bin" / "nvcc").exists():
                    found.append((entry.name[len("cuda-"):], cu))
    found.sort(key=lambda item: tuple(int(p) if p.isdigit() else 0 for p in item[0].split(".")), reverse=True)
    return found


def install_cuda_toolchain(version: str = "13.3", python: str | None = None, quiet: bool = False) -> Path:
    """Install the wheel set of one CUDA minor version into an isolated target directory."""
    target = toolchain_root() / f"cuda-{version}"
    target.mkdir(parents=True, exist_ok=True)
    specs = [f"{name}=={version}.*" for name in WHEELS]
    python = python or sys.executable
    if which("uv"):
        argv = ["uv", "pip", "install", "--python", python, "--target", str(target), *specs]
    else:
        argv = [python, "-m", "pip", "install", "--target", str(target), *specs]
    if not quiet:
        print("installing:", " ".join(argv), flush=True)
    result = run(argv, timeout=1800)
    if not result.ok:
        raise RuntimeError(f"toolchain install failed: {result.stderr[-1500:] or result.stdout[-1500:]}")
    homes = sorted(target.glob("nvidia/cu*"))
    home = next((h for h in homes if (h / "bin" / "nvcc").exists()), None)
    if home is None:
        raise RuntimeError(f"no nvcc found under {target} after install")
    from .cuda_cpp import _ensure_link_layout
    _ensure_link_layout(str(home))
    for bin_name in ("nvcc", "ptxas", "cudafe++", "fatbinary", "nvlink"):
        path = home / "bin" / bin_name
        if path.exists():
            path.chmod(path.stat().st_mode | 0o111)
    if not quiet:
        print(f"toolchain ready: {home}", flush=True)
    return home


def remove_toolchain(version: str) -> bool:
    target = toolchain_root() / f"cuda-{version}"
    if target.exists():
        shutil.rmtree(target)
        return True
    return False

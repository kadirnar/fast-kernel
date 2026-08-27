"""CUDA C++ via torch.utils.cpp_extension.load_inline, with automatic nvcc discovery.

nvcc is frequently *not* on PATH but *is* installed by pip (torch's cu13x wheels depend on
`nvidia-cuda-nvcc`; it lives under site-packages/nvidia/cu13/bin/nvcc or nvidia/cuda_nvcc/bin/nvcc).
`ensure_cuda_home()` finds it and exports CUDA_HOME/PATH so torch, TileLang and CuTe can use it.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path
from typing import Any

NOTES = (
    "Write kernels as CUDA C++ strings compiled with torch.utils.cpp_extension.load_inline (build dir under "
    ".fast-kernel/build). Use fastkernel.backends.cuda_cpp.ensure_cuda_home() first. Tensor cores via wmma/mma.sync; "
    "set TORCH_CUDA_ARCH_LIST to the measured compute capability to avoid building many archs."
)


def find_nvcc() -> tuple[str | None, str | None]:
    """Return (nvcc_path, cuda_home). Order: FAST_KERNEL_CUDA_HOME, installed toolchains (newest first),
    CUDA_HOME/CUDA_PATH, PATH, pip wheels in the venv, /usr/local/cuda*, /opt/cuda*."""
    forced = os.environ.get("FAST_KERNEL_CUDA_HOME")
    if forced and Path(forced, "bin", "nvcc").exists():
        return str(Path(forced, "bin", "nvcc")), forced
    from .toolchain import list_toolchains
    for _version, home in list_toolchains():
        return str(home / "bin" / "nvcc"), str(home)
    env_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env_home and Path(env_home, "bin", "nvcc").exists():
        return str(Path(env_home, "bin", "nvcc")), env_home
    on_path = shutil.which("nvcc")
    if on_path:
        return on_path, str(Path(on_path).resolve().parent.parent)
    candidates: list[str] = []
    for site in _site_dirs():
        candidates += glob.glob(os.path.join(site, "nvidia", "cu*", "bin", "nvcc"))
        candidates += glob.glob(os.path.join(site, "nvidia", "cuda_nvcc", "bin", "nvcc"))
    candidates += glob.glob("/usr/local/cuda*/bin/nvcc") + glob.glob("/opt/cuda*/bin/nvcc")
    for path in candidates:
        if os.access(path, os.X_OK):
            return path, str(Path(path).parent.parent)
    return None, None


def _site_dirs() -> list[str]:
    dirs = []
    try:
        import site
        dirs += site.getsitepackages()
        user = site.getusersitepackages()
        if user:
            dirs.append(user)
    except Exception:  # noqa: BLE001
        pass
    dirs += [p for p in sys.path if p.endswith("site-packages")]
    return list(dict.fromkeys(d for d in dirs if d and os.path.isdir(d)))


def _ensure_link_layout(home: str) -> None:
    """pip CUDA wheels ship lib/libcudart.so.13 but linkers want -lcudart (libcudart.so) and torch looks in lib64/."""
    root = Path(home)
    lib = root / "lib"
    if not lib.is_dir():
        return
    try:
        for versioned in lib.glob("lib*.so.*"):
            name = versioned.name.split(".so.")[0] + ".so"
            unversioned = lib / name
            if not unversioned.exists():
                unversioned.symlink_to(versioned.name)
        lib64 = root / "lib64"
        if not lib64.exists():
            lib64.symlink_to("lib")
    except OSError:
        pass   # read-only install; the agent can point FAST_KERNEL_CUDA_HOME at a writable toolchain


def ensure_cuda_home() -> str | None:
    # tools installed into the venv (ninja, nvcc wrappers) must be visible even when the venv is not activated
    for venv_bin in {str(Path(sys.executable).parent), str(Path(sys.prefix) / "bin")}:
        if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    nvcc, home = find_nvcc()
    if home:
        # newer host compilers than nvcc officially supports still work for our kernels; do not stall on the version check
        flags = os.environ.get("NVCC_APPEND_FLAGS", "")
        if "-allow-unsupported-compiler" not in flags:
            os.environ["NVCC_APPEND_FLAGS"] = (flags + " -allow-unsupported-compiler").strip()
        os.environ["CUDA_HOME"] = home
        os.environ["CUDA_PATH"] = home
        _ensure_link_layout(home)
        bin_dir = str(Path(nvcc).parent)
        if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            import torch
            if torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
                major, minor = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:  # noqa: BLE001
            pass
    return home


def build_dir(campaign_root: Path | None = None) -> Path:
    root = Path(campaign_root) if campaign_root else Path.cwd()
    path = root / ".fast-kernel" / "build"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_cuda_inline(name: str, cuda_src: str, cpp_src: str, functions: list[str], campaign_root: Path | None = None,
                     extra_cuda_cflags: list[str] | None = None, verbose: bool = False):
    """Compile CUDA C++ once (cached by torch on source hash) and return the extension module."""
    ensure_cuda_home()
    from torch.utils.cpp_extension import load_inline
    build = build_dir(campaign_root) / name
    build.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=name, cpp_sources=[cpp_src], cuda_sources=[cuda_src], functions=functions,
        # No forced --use_fast_math: the strict quality contract needs exact rsqrt/div/fmad and no
        # denormal flush. A kernel that genuinely wants fast-math opts in via extra_cuda_cflags.
        extra_cuda_cflags=["-O3", *(extra_cuda_cflags or [])],
        build_directory=str(build), verbose=verbose,
    )


def probe(compile_test: bool = True) -> dict[str, Any]:
    nvcc, home = find_nvcc()
    result: dict[str, Any] = {"available": nvcc is not None, "nvcc": nvcc, "cuda_home": home, "version": None, "compiled": False}
    if not nvcc:
        result["error"] = ("nvcc not found. Fix: `uv pip install nvidia-cuda-nvcc` (pip wheel, no system install) or install the "
                           "CUDA toolkit and set CUDA_HOME.")
        return result
    from ..util import run
    ver = run([nvcc, "--version"], timeout=30)
    if ver.ok:
        for line in ver.stdout.splitlines():
            if "release" in line:
                result["version"] = line.strip().split("release")[-1].strip().split(",")[0]
    if not compile_test:
        return result
    try:
        import torch
        if not torch.cuda.is_available():
            result["error"] = "no CUDA device"
            return result
        ensure_cuda_home()
        module = load_cuda_inline(
            "fk_probe_add",
            cuda_src=r"""
            #include <torch/extension.h>
            __global__ void add_kernel(const float* a, const float* b, float* c, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) c[i] = a[i] + b[i];
            }
            torch::Tensor fk_add(torch::Tensor a, torch::Tensor b) {
                auto c = torch::empty_like(a);
                int n = a.numel();
                add_kernel<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
                return c;
            }
            """,
            cpp_src="torch::Tensor fk_add(torch::Tensor a, torch::Tensor b);",
            functions=["fk_add"],
        )
        a = torch.randn(1024, device="cuda")
        b = torch.randn(1024, device="cuda")
        ok = torch.allclose(module.fk_add(a, b), a + b)
        result["compiled"] = bool(ok)
        if not ok:
            result["error"] = "probe kernel produced wrong results"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {_error_summary(str(exc))}"
        if _host_compiler_problem(str(exc)):
            result["fix"] = "host compiler too new for this nvcc: run `fast-kernel toolchain install --cuda 13.3` (self-contained wheels) and re-probe"
    return result


def _error_summary(message: str, limit: int = 700) -> str:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    errors = [line for line in lines if "error" in line.lower() and "warning" not in line.lower()]
    text = " | ".join(errors[:6]) if errors else " | ".join(lines[-4:])
    return text[:limit]


def _host_compiler_problem(message: str) -> bool:
    return any(key in message for key in ("unsupported GNU version", "cudafe++' died", "is not allowed", "__builtin_is_virtual_base_of",
                                          "expected a type specifier", "exception specification is incompatible"))

# Backends

**CUDA C++ is the only implementation backend.** Every kernel a candidate ships is hand-written CUDA
compiled through `torch.utils.cpp_extension.load_inline`, captured with CUDA graphs. Triton, TileLang
and CuTe are not offered — that is a measured policy, and `AGENTS.md` carries the numbers behind it:
the wins left in a mature campaign are *fusion granularity* (a whole resnet block or a whole RVQ stage
in one launch, so its intermediates never reach global memory), which a tile DSL's automatic pipeline
cannot express. Leaving an op on stock torch is always a legitimate answer.

| backend | probe | helper | skill |
|---|---|---|---|
| CUDA C++ | `load_inline` compiles and runs a vector add with an auto-discovered nvcc | `load_cuda_inline(...)`, template bias+ELU epilogue | `/cuda-cpp-kernels` |
| CUDA graphs | capture + replay | `Graphed`, `ShapeBucketedGraphs` | `/cuda-graphs` |
| hub kernels | import + cached kernels | `kernels.get_kernel` | `/hub-kernels` |

`fast-kernel probe` writes exactly these three into `capabilities.json` (`backends.base.BACKENDS`); the
playbook adds `torch` as a fourth *answer* — "leave this op stock" — that needs no probe.

## nvcc discovery and toolchains

`fastkernel.backends.cuda_cpp.find_nvcc()` looks, in order, at `FAST_KERNEL_CUDA_HOME`, installed
toolchains (newest first), `CUDA_HOME`/`CUDA_PATH`, `PATH`, the pip wheels inside the venv
(`nvidia/cu13/bin/nvcc`), `/usr/local/cuda*`, `/opt/cuda*`. `ensure_cuda_home()` exports CUDA_HOME /
PATH (venv `bin/` for ninja), `TORCH_CUDA_ARCH_LIST` for the measured GPU, `NVCC_APPEND_FLAGS=
-allow-unsupported-compiler`, and creates the `libcudart.so` / `lib64` links pip wheels omit.

torch's CUDA wheels bundle one runtime version; its matching nvcc frontend may not accept a very new
host compiler. `fast-kernel toolchain install --cuda 13.3` installs
`nvidia-cuda-nvcc/cccl/crt/nvvm/runtime==13.3.*` into `~/.cache/fast-kernel/toolchains/cuda-13.3/`
(no sudo, nothing system-wide); CUDA C++ then compiles and the kernels run on the driver as usual.
`fast-kernel probe` prints a `fix:` hint when it detects this situation.

## Environment notes

- CUDA C++, CUDA graphs and hub kernels need nothing beyond `uv sync --extra cuda`.
- When the host compiler is newer than the nvcc bundled with torch supports (the probe reports
  "unsupported GNU version" or a `cudafe++` crash), `fast-kernel toolchain install --cuda <version>` with a
  newer CUDA minor version fixes the build; the kernels still run on torch's runtime.
- Fast-math is deliberately not forced (`-O3` only): under the strict policy a kernel needs exact
  `rsqrt`/`div`/`fmad` and no denormal flush. Opt in per kernel via `extra_cuda_cflags`.
- `fast-kernel probe` is the source of truth for what works on a given machine; nothing in this
  repository assumes a particular GPU.

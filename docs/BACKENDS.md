# Backends

| backend | probe | helper | skill |
|---|---|---|---|
| Triton | compile + run a vector add | templates: rmsnorm, silu*mul, autotuned GEMM w/ epilogue, causal depthwise conv1d, codebook argmin | `/triton-kernels` |
| TileLang | `tilelang.compile(..., target="cuda", execution_backend="tvm_ffi")` | template GEMM | `/tilelang-kernels` |
| CuTe DSL | `@cute.kernel` / `@cute.jit` + `from_dlpack` | template elementwise | `/cute-dsl-kernels` |
| CUDA C++ | `torch.utils.cpp_extension.load_inline` with auto-discovered nvcc | `load_cuda_inline(...)`, template bias+ELU | `/cuda-cpp-kernels` |
| torch.compile | compile + run | — | `/torch-compile` |
| CUDA graphs | capture + replay | `Graphed`, `ShapeBucketedGraphs` | `/cuda-graphs` |
| hub kernels | import + cached kernels | `kernels.get_kernel` | `/hub-kernels` |

## nvcc discovery and toolchains

`fastkernel.backends.cuda_cpp.find_nvcc()` looks, in order, at `FAST_KERNEL_CUDA_HOME`, installed
toolchains (newest first), `CUDA_HOME`/`CUDA_PATH`, `PATH`, the pip wheels inside the venv
(`nvidia/cu13/bin/nvcc`), `/usr/local/cuda*`, `/opt/cuda*`. `ensure_cuda_home()` exports CUDA_HOME /
PATH (venv `bin/` for ninja), `TORCH_CUDA_ARCH_LIST` for the measured GPU, `NVCC_APPEND_FLAGS=
-allow-unsupported-compiler`, and creates the `libcudart.so` / `lib64` links pip wheels omit.

torch's CUDA wheels bundle one runtime version; its matching nvcc frontend may not accept a very new
host compiler. `fast-kernel toolchain install --cuda 13.3` installs
`nvidia-cuda-nvcc/cccl/crt/nvvm/runtime==13.3.*` into `~/.cache/fast-kernel/toolchains/cuda-13.3/`
(no sudo, nothing system-wide); TileLang and CUDA C++ then compile and the kernels run on the driver
as usual. `fast-kernel probe` prints a `fix:` hint when it detects this situation.

## Environment notes

- Triton, torch.compile, CUDA graphs and hub kernels need nothing beyond `uv sync --extra cuda`.
- CuTe DSL needs `--extra cute`; TileLang needs `--extra tilelang`; both compile through nvcc.
- When the host compiler is newer than the nvcc bundled with torch supports (the probe reports
  "unsupported GNU version" or a `cudafe++` crash), `fast-kernel toolchain install --cuda <version>` with a
  newer CUDA minor version fixes TileLang and CUDA C++; the kernels still run on torch's runtime.
- `fast-kernel probe` is the source of truth for what works on a given machine; nothing in this
  repository assumes a particular GPU.

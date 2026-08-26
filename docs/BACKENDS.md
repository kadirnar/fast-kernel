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

torch's cu13x wheels bundle a 13.0 runtime; its matching nvcc frontend does not parse very new
host-compiler headers (gcc 16 on this machine). `fast-kernel toolchain install --cuda 13.3` installs
`nvidia-cuda-nvcc/cccl/crt/nvvm/runtime==13.3.*` into `~/.cache/fast-kernel/toolchains/cuda-13.3/`
(no sudo, nothing system-wide); TileLang and CUDA C++ then compile and the kernels run on the driver
as usual. `fast-kernel probe` prints a `fix:` hint when it detects this situation.

## Measured on the development machine (RTX 5070 Ti, sm_120, gcc 16, torch 2.13 cu130)

- Triton 3.7.1, torch.compile, CUDA graphs, hub kernels: ready out of the box.
- CuTe DSL 4.7.1: ready (probe kernel compiles and runs).
- TileLang 0.1.13: ready with the 13.3 toolchain (`execution_backend="tvm_ffi"`); the venv's 13.0 nvcc
  rejects gcc 16.
- CUDA C++ (`load_inline`): ready with the 13.3 toolchain (`ensure_cuda_home()` also creates the `libcudart.so` / `lib64` links the wheels omit and puts the venv's `ninja` on PATH).

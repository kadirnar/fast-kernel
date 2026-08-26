"""TileLang GEMM template (bf16 in, fp32 accumulate) with a pipelined shared-memory loop."""
from __future__ import annotations

import torch

from fastkernel.backends.cuda_cpp import ensure_cuda_home

ensure_cuda_home()
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402


def make_gemm(M: int, N: int, K: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, dtype: str = "bfloat16",
              accum_dtype: str = "float32"):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_l = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_s)
                T.copy(B[k * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * block_M, bx * block_N])
    return tilelang.compile(gemm, out_idx=[2])


if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    kernel = make_gemm(M, N, K)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = kernel(a, b)
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    print("max rel diff", ((c.float() - ref.float()).abs().max() / ref.float().abs().max()).item())

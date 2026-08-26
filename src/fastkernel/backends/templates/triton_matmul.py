"""Autotuned bf16 GEMM with a fused epilogue (bias + activation + residual). Starting point for
linear layers, merged QKV / gate-up projections, and implicit-GEMM convolutions."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP": 8}, num_stages=s, num_warps=w)
        for bm, bn, bk, s, w in [(128, 128, 32, 3, 8), (128, 64, 32, 4, 4), (64, 128, 32, 4, 4), (64, 64, 32, 4, 4),
                                 (64, 64, 64, 3, 4), (32, 64, 64, 3, 4), (64, 32, 64, 3, 2), (32, 32, 64, 3, 2)]
    ]


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, res_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                   ACT: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_in_group = GROUP * num_pid_n
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    pid_m = first_pid_m + (pid % num_in_group) % group_size_m
    pid_n = (pid % num_in_group) // group_size_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BK), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K - k * BK) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    if ACT == 1:      # silu
        acc = acc * tl.sigmoid(acc)
    elif ACT == 2:    # gelu (tanh approx)
        acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
    elif ACT == 3:    # elu
        acc = tl.where(acc > 0, acc, tl.exp(acc) - 1.0)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    if HAS_RES:
        acc += tl.load(res_ptr + rm[:, None] * scm + rn[None, :] * scn, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


ACTS = {None: 0, "none": 0, "silu": 1, "gelu": 2, "elu": 3}


def matmul(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None, act: str | None = None,
           residual: torch.Tensor | None = None, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """a: (M, K), b: (K, N) -> (M, N) with optional fused bias/activation/residual."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=out_dtype or a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)  # noqa: E731
    _matmul_kernel[grid](a, b, c, bias if bias is not None else c, residual if residual is not None else c,
                         M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                         ACT=ACTS[act], HAS_BIAS=bias is not None, HAS_RES=residual is not None)
    return c


if __name__ == "__main__":
    a = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    out = matmul(a, b, bias, act="silu")
    ref = torch.nn.functional.silu(a.float() @ b.float() + bias.float())
    print("max rel diff", ((out.float() - ref).abs().max() / ref.abs().max()).item())

"""Fused nearest-codebook search: argmin_j ||x_i - c_j||^2 without materializing cdist.

Uses -2 x.c + |c|^2 (|x|^2 is constant per row). fp32 by default; for tensor-core speed run the
coarse pass in bf16 and re-rank the top-k exactly (see fastkernel skill triton-kernels).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _argmin_kernel(x_ptr, c_ptr, cn_ptr, out_ptr, N, J, D: tl.constexpr, BN: tl.constexpr, BJ: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    rmask = rows < N
    best = tl.full((BN,), float("inf"), tl.float32)
    best_idx = tl.zeros((BN,), tl.int32)
    for j0 in range(0, J, BJ):
        js = j0 + tl.arange(0, BJ)
        jmask = js < J
        acc = tl.zeros((BN, BJ), tl.float32)
        for d0 in range(0, D, BD):
            ds = d0 + tl.arange(0, BD)
            dmask = ds < D
            xv = tl.load(x_ptr + rows[:, None] * D + ds[None, :], mask=rmask[:, None] & dmask[None, :], other=0.0)
            cv = tl.load(c_ptr + js[:, None] * D + ds[None, :], mask=jmask[:, None] & dmask[None, :], other=0.0)
            acc = tl.dot(xv, tl.trans(cv), acc)
        dist = tl.load(cn_ptr + js, mask=jmask, other=float("inf"))[None, :] - 2.0 * acc
        dist = tl.where(jmask[None, :], dist, float("inf"))
        local = tl.min(dist, axis=1)
        local_idx = tl.argmin(dist, axis=1) + j0
        upd = local < best
        best = tl.where(upd, local, best)
        best_idx = tl.where(upd, local_idx.to(tl.int32), best_idx)
    tl.store(out_ptr + rows, best_idx, mask=rmask)


def codebook_argmin(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """x: (N, D) fp32/bf16, codebook: (J, D) -> (N,) int32 nearest indices."""
    N, D = x.shape
    J = codebook.shape[0]
    x = x.contiguous()
    c = codebook.contiguous().to(x.dtype)
    cn = (c.float() ** 2).sum(-1)
    out = torch.empty(N, dtype=torch.int32, device=x.device)
    BN, BJ, BD = 16, 64, min(64, triton.next_power_of_2(D))
    _argmin_kernel[(triton.cdiv(N, BN),)](x, c, cn, out, N, J, D=D, BN=BN, BJ=BJ, BD=BD, num_warps=4)
    return out


if __name__ == "__main__":
    x = torch.randn(64, 256, device="cuda")
    cb = torch.randn(2048, 256, device="cuda")
    ref = torch.cdist(x, cb).argmin(-1)
    out = codebook_argmin(x, cb)
    print("match", (out.long() == ref).float().mean().item())

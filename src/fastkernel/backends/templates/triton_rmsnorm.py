"""Fused RMSNorm (optionally + residual add) in Triton. Row per program, fp32 statistics."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, res_ptr, stride, n_cols, eps, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        x += tl.load(res_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    y = x * inv * w
    tl.store(out_ptr + row * stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, residual: torch.Tensor | None = None) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    out = torch.empty_like(x2)
    n_rows, n_cols = x2.shape
    block = triton.next_power_of_2(n_cols)
    num_warps = 4 if block <= 2048 else 8
    res = residual.reshape(-1, shape[-1]).contiguous() if residual is not None else x2
    _rmsnorm_kernel[(n_rows,)](x2, weight, out, res, x2.stride(0), n_cols, eps, HAS_RES=residual is not None,
                               BLOCK=block, num_warps=num_warps)
    return out.reshape(shape)


if __name__ == "__main__":
    x = torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)
    out = rmsnorm(x, w)
    print("max diff", (out.float() - ref.float()).abs().max().item())

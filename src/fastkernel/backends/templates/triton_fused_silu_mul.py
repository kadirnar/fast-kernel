"""Fused SwiGLU gate: silu(a) * b in one pass (the MLP hot elementwise op in LFM2 / Llama-style MLPs)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * tl.sigmoid(a) * b
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK = 2048
    _silu_mul_kernel[(triton.cdiv(n, BLOCK),)](a, b, out, n, BLOCK=BLOCK, num_warps=8)
    return out


if __name__ == "__main__":
    a = torch.randn(1 << 20, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1 << 20, device="cuda", dtype=torch.bfloat16)
    print((silu_mul(a, b).float() - (torch.nn.functional.silu(a.float()) * b.float())).abs().max().item())

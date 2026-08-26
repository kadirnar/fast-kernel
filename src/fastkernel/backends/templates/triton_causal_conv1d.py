"""Depthwise causal conv1d (short kernel, e.g. LFM2 ShortConv with L_cache=3) fused with gating.

y[b, c, t] = sum_k w[c, k] * x[b, c, t - (K-1) + k]   (zero padded), optional out = gate[b,c,t] * y
Layout: (B, C, T) contiguous. One program per (b, c-block, t-block).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dwconv_kernel(x_ptr, w_ptr, gate_ptr, out_ptr, C, T, KS: tl.constexpr, HAS_GATE: tl.constexpr,
                   BC: tl.constexpr, BT: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)
    cs = pid_c * BC + tl.arange(0, BC)
    ts = pid_t * BT + tl.arange(0, BT)
    cmask = cs < C
    base = pid_b * C * T
    acc = tl.zeros((BC, BT), dtype=tl.float32)
    for k in tl.static_range(KS):
        tt = ts[None, :] - (KS - 1) + k
        mask = cmask[:, None] & (tt >= 0) & (tt < T)
        xv = tl.load(x_ptr + base + cs[:, None] * T + tt, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + cs * KS + k, mask=cmask, other=0.0).to(tl.float32)
        acc += xv * wv[:, None]
    omask = cmask[:, None] & (ts[None, :] < T)
    if HAS_GATE:
        g = tl.load(gate_ptr + base + cs[:, None] * T + ts[None, :], mask=omask, other=0.0).to(tl.float32)
        acc = acc * g
    tl.store(out_ptr + base + cs[:, None] * T + ts[None, :], acc.to(out_ptr.dtype.element_ty), mask=omask)


def causal_dwconv1d(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
    """x: (B, C, T), weight: (C, K) [or (C, 1, K)], gate: optional (B, C, T)."""
    if weight.dim() == 3:
        weight = weight[:, 0, :]
    B, C, T = x.shape
    K = weight.shape[-1]
    x = x.contiguous()
    out = torch.empty_like(x)
    BC, BT = 32, 64
    grid = (B, triton.cdiv(C, BC), triton.cdiv(T, BT))
    _dwconv_kernel[grid](x, weight.contiguous(), gate.contiguous() if gate is not None else x, out, C, T, KS=K,
                         HAS_GATE=gate is not None, BC=BC, BT=BT, num_warps=4)
    return out


if __name__ == "__main__":
    B, C, T, K = 2, 2048, 128, 3
    x = torch.randn(B, C, T, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(C, 1, K, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.conv1d(x.float(), w.float(), padding=K - 1, groups=C)[..., :T]
    out = causal_dwconv1d(x, w)
    print("max diff", (out.float() - ref).abs().max().item())

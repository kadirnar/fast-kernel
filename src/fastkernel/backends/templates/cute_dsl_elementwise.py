"""CuTe DSL starter: fused elementwise kernel (silu(a) * b). Layouts via cute.Tensor / from_dlpack."""

import torch

from fastkernel.backends.cuda_cpp import ensure_cuda_home

ensure_cuda_home()
import cutlass  # noqa: E402,F401
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402


@cute.kernel
def silu_mul_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    if idx < cute.size(gA):
        a = gA[idx]
        gC[idx] = a / (1.0 + cute.math.exp(-a)) * gB[idx]


@cute.jit
def silu_mul_launch(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    n = cute.size(mA)
    silu_mul_kernel(mA, mB, mC).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1))


def silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a)
    silu_mul_launch(from_dlpack(a.contiguous()), from_dlpack(b.contiguous()), from_dlpack(c))
    return c


if __name__ == "__main__":
    a = torch.randn(1 << 16, device="cuda")
    b = torch.randn(1 << 16, device="cuda")
    print("max diff", (silu_mul(a, b) - torch.nn.functional.silu(a) * b).abs().max().item())

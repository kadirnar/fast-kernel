"""CUDA C++ starter via load_inline: a fused bias+ELU epilogue kernel. Compiles with the pip nvcc."""
from __future__ import annotations

import torch

from fastkernel.backends.cuda_cpp import load_cuda_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
__global__ void bias_elu_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ y,
                                int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    float v = x[idx] + bias[idx % cols];
    y[idx] = v > 0.f ? v : (__expf(v) - 1.f);
}
torch::Tensor bias_elu(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be fp32 CUDA");
    auto y = torch::empty_like(x);
    int rows = x.numel() / x.size(-1), cols = x.size(-1);
    int n = rows * cols;
    bias_elu_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    return y;
}
"""
CPP_SRC = "torch::Tensor bias_elu(torch::Tensor x, torch::Tensor bias);"


def get_module(campaign_root=None):
    return load_cuda_inline("fk_bias_elu", CUDA_SRC.replace("#include <torch/extension.h>",
                                                            "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>"),
                            CPP_SRC, ["bias_elu"], campaign_root=campaign_root)


if __name__ == "__main__":
    mod = get_module()
    x = torch.randn(256, 512, device="cuda")
    b = torch.randn(512, device="cuda")
    print("max diff", (mod.bias_elu(x, b) - torch.nn.functional.elu(x + b)).abs().max().item())

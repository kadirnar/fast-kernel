"""Custom model spec. `build_model()` must return an nn.Module in eval mode (weights loaded/seeded).

Override `Spec` methods for model-specific workloads, comparisons or derived metrics; see
fastkernel.models.spec.ModelSpec and the built-in specs (mimi, hf_causal_lm, yolo) for examples.
"""
from __future__ import annotations

import torch
from torch import nn

from fastkernel.models.torch_module import TorchModuleSpec


def build_model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1000),
    )
    return model.eval()


class Spec(TorchModuleSpec):
    name = "custom"
    display_name = "My model"
    notes = "Describe the architecture and known hot paths here; the agent reads this in PLAN.md."

    def load_reference(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return build_model().to(device).eval()

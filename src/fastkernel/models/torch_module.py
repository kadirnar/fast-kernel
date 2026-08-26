"""Generic spec for any torch.nn.Module: point at a loader function and describe inputs.

model_args (GOAL.md frontmatter):
    loader: "my_package.models:build"      # callable returning an nn.Module (eval mode)
    input_shapes: [[1, 3, 224, 224]]       # positional tensor inputs (random normal, seeded)
    input_dtype: float32
    batch_sweep: [1, 4]                    # optional extra workloads with different batch sizes
"""
from __future__ import annotations

import importlib
from typing import Any

from .spec import ModelSpec, Workload


class TorchModuleSpec(ModelSpec):
    name = "custom"
    display_name = "Custom torch module"
    notes = "Generic torch module: outputs compared with allclose against the fp32 reference."

    def _loader(self):
        target = self.args.get("loader")
        if not target:
            raise RuntimeError("model_args.loader is required, e.g. 'my_pkg.models:build'")
        module_name, _, attr = target.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr or "build")

    def load_reference(self) -> Any:
        import torch
        model = self._loader()()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device).eval()

    def _make_inputs(self, shapes: list[list[int]]):
        def make(device, seed):
            import torch
            g = torch.Generator(device="cpu").manual_seed(seed)
            dtype = getattr(torch, str(self.args.get("input_dtype", "float32")))
            return {"args": [torch.randn(s, generator=g).to(device=device, dtype=dtype) for s in shapes]}
        return make

    @staticmethod
    def _run(model, inputs):
        return model(*inputs["args"])

    def workloads(self) -> list[Workload]:
        shapes = self.args.get("input_shapes") or [[1, 3, 224, 224]]
        items = [Workload("forward", self._make_inputs(shapes), self._run, primary=True, describe=f"forward with shapes {shapes}",
                          units={"samples": float(shapes[0][0])})]
        for batch in self.args.get("batch_sweep") or []:
            swept = [[int(batch), *s[1:]] for s in shapes]
            items.append(Workload(f"forward_b{batch}", self._make_inputs(swept), self._run, tags=("sweep",),
                                  describe=f"batch {batch}", units={"samples": float(batch)}))
        return items

    def edge_workloads(self) -> list[Workload]:
        shapes = self.args.get("input_shapes") or [[1, 3, 224, 224]]
        odd = [[s[0], *[max(1, d - 1) if i == len(s) - 2 else d for i, d in enumerate(s[1:])]] for s in shapes]
        return [Workload("edge_odd", self._make_inputs(odd), self._run, tags=("edge",), bench=False, describe="odd spatial size")]

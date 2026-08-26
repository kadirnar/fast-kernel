"""Ultralytics YOLO (YOLO26 by default; YOLO11/YOLOv8 weights work the same) as a plain torch module.

The DetectionModel underneath `ultralytics.YOLO(...)` is fused (Conv+BN) and run in eval mode on seeded
random images. End-to-end (NMS-free) heads output (B, 300, 6) = xyxy, conf, cls; older heads output raw
(B, 4 + nc, anchors). Gates compare boxes/scores/classes with pixel tolerances.

model_args: weights (default yolo26n.pt), imgsz (640), batch (1), batch_sweep ([8]), half (false)
"""
from __future__ import annotations

from typing import Any

from .spec import GateCheck, ModelSpec, Workload, compare_trees


class YOLOSpec(ModelSpec):
    name = "yolo"
    display_name = "Ultralytics YOLO (torch module)"
    hub_id = "ultralytics/yolo26n"
    notes = """\
YOLO26n/11n are tiny CNNs (~2.6M params): at batch 1 and 640x640 the GPU is idle most of the time
(hundreds of small conv/act/concat launches), so CUDA graphs and conv+SiLU epilogue fusion dominate;
at larger batches conv GEMMs (cuDNN, channels-last, bf16/fp16) and the C2PSA attention block matter.
The fused model (Conv+BN folded) is the reference; keep the (B, 300, 6) end-to-end output contract.
"""
    default_rtol = {"strict": 1e-3, "tolerant": 2e-2}
    default_atol = {"strict": 1e-3, "tolerant": 5e-2}

    def load_reference(self) -> Any:
        import torch
        from ultralytics import YOLO
        weights = str(self.args.get("weights", "yolo26n.pt"))
        wrapper = YOLO(weights)
        model = wrapper.model
        try:
            model = model.fuse()
        except Exception:  # noqa: BLE001
            pass
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).float().eval()
        for m in model.modules():
            if hasattr(m, "export"):
                m.export = False
        return model

    def _make_images(self, batch: int, size: int):
        def make(device, seed):
            import torch
            g = torch.Generator(device="cpu").manual_seed(seed)
            x = torch.rand((batch, 3, size, size), generator=g)
            # add a few bright rectangles so detections are not empty
            for i in range(batch):
                x[i, :, 100 + 40 * i: 300 + 40 * i, 120: 360] = torch.rand(3, 1, 1, generator=g)
            return {"images": x.to(device)}
        return make

    @staticmethod
    def _run(model, inputs):
        out = model(inputs["images"])
        return out[0] if isinstance(out, (tuple, list)) else out

    def workloads(self) -> list[Workload]:
        size = int(self.args.get("imgsz", 640))
        batch = int(self.args.get("batch", 1))
        items = [Workload(f"detect_b{batch}", self._make_images(batch, size), self._run, primary=True,
                          describe=f"batch {batch} @ {size}x{size}", units={"images": float(batch)})]
        for b in self.args.get("batch_sweep") or [8]:
            items.append(Workload(f"detect_b{int(b)}", self._make_images(int(b), size), self._run, tags=("sweep",),
                                  describe=f"batch {b} @ {size}x{size}", units={"images": float(b)}))
        return items

    def edge_workloads(self) -> list[Workload]:
        return [Workload("edge_small_320", self._make_images(1, 320), self._run, tags=("edge",), bench=False, describe="320x320"),
                Workload("edge_batch3", self._make_images(3, int(self.args.get("imgsz", 640))), self._run, tags=("edge",), bench=False,
                         describe="batch 3")]

    def compare(self, workload: Workload, reference: Any, candidate: Any) -> list[GateCheck]:
        strict = self.policy.precision == "strict"
        cand = candidate.to(reference.device)
        if tuple(reference.shape) != tuple(cand.shape):
            return [GateCheck(f"{workload.name}/shape", False, detail=f"{tuple(cand.shape)} != {tuple(reference.shape)}")]
        if reference.dim() == 3 and reference.shape[-1] == 6:   # end-to-end: xyxy, conf, cls
            conf_thr = 0.25
            keep = reference[..., 4] > conf_thr
            box_tol = 0.5 if strict else 2.0
            conf_tol = 1e-3 if strict else 1e-2
            boxes = (reference[..., :4] - cand[..., :4]).abs()[keep]
            confs = (reference[..., 4] - cand[..., 4]).abs()[keep]
            cls_ok = (reference[..., 5] == cand[..., 5])[keep]
            n = int(keep.sum().item())
            checks = [
                GateCheck(f"{workload.name}/boxes_px", (boxes.max().item() if n else 0.0) <= box_tol, boxes.max().item() if n else 0.0, box_tol,
                          f"max box coordinate diff over {n} detections (conf>{conf_thr})"),
                GateCheck(f"{workload.name}/confidence", (confs.max().item() if n else 0.0) <= conf_tol, confs.max().item() if n else 0.0, conf_tol,
                          "max confidence diff"),
                GateCheck(f"{workload.name}/classes", bool(cls_ok.all().item()) if n else True, cls_ok.float().mean().item() if n else 1.0, 1.0,
                          "class ids identical for confident detections"),
            ]
            if n == 0:
                checks.append(GateCheck(f"{workload.name}/no_detections", True, 0.0, None, "no confident detections in reference (informational)"))
            return checks
        rtol, atol = self.tolerances(workload)
        return compare_trees(reference, cand, rtol=rtol, atol=atol, prefix=workload.name)

    def hotspot_hints(self) -> list[dict[str, Any]]:
        return [
            {"symbol": "Conv", "category": "conv", "note": "Conv2d(+BN folded)+SiLU: channels-last, fp16/bf16 cuDNN, fuse SiLU epilogue"},
            {"symbol": "C3k2", "category": "conv", "note": "bottleneck stacks: many small convs -> launch bound at batch 1"},
            {"symbol": "C2PSA", "category": "attention", "note": "position-sensitive attention block"},
            {"symbol": "Detect", "category": "elementwise", "note": "head decode (DFL, box transform, one-to-one top-k)"},
            {"symbol": "Concat", "category": "memory-movement", "note": "avoid copies with pre-allocated buffers / fusion"},
            {"symbol": "Upsample", "category": "memory-movement", "note": "nearest upsample: fuse into the following conv's loads"},
        ]

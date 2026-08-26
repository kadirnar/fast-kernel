---
model: yolo
objective: "Minimise YOLO26n batch-1 640x640 detection latency (and keep batch 8 fast) while keeping boxes, confidences and classes identical within pixel tolerances."
target_metric: latency_ms
direction: minimize
min_improvement: 0.01
continuous: true
primary_workload: detect_b1
gates:
  precision: strict
  determinism: exact
  stages: [smoke, shapes, numerical, determinism, edge]
bench:
  warmup: 10
  repeats: 100
  ramp_seconds: 1.0
  timeout_seconds: 1200
  profile_every_experiment: true
model_args:
  weights: yolo26n.pt
  imgsz: 640
  batch: 1
  batch_sweep: [8]
protected: [GOAL.md, spec.py, harness/**, .fast-kernel/**, experiments/**, results.tsv]
---

# Goal

Speed up the fused Ultralytics YOLO26n `DetectionModel` (plain torch module, end-to-end NMS-free head,
output (B, 300, 6)). Reference is fp32 eval. Any `weights` (yolo11n.pt, yolov8n.pt, custom) work.

# Policy

- strict: boxes within 0.5 px, confidences within 1e-3, identical classes for detections with conf > 0.25.
- tolerant: 2 px / 1e-2 (allows fp16/bf16 convolutions).

# Where to start

1. Batch 1 is launch bound (hundreds of tiny convs): CUDA-graph the whole forward (static 640x640 input).
2. channels-last + cuDNN benchmark, then fp16 convs in tolerant mode.
3. Fuse Conv+SiLU epilogues and Concat/Upsample copies (Triton or torch.compile max-autotune).

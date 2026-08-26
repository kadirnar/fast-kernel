---
name: fk-add-model
description: Onboard a new model into fast-kernel (any torch module, a Transformers model, ultralytics, custom repo) - write spec.py (loader, workloads, oracle comparison, hints), GOAL.md, candidate stub, then baseline. Use for "add model X", "optimize my own model", "support <architecture>".
argument-hint: <name-or-hub-id>
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Bash(python *), Read, Edit, Write, Glob, Grep, WebFetch
---

1. `fast-kernel init custom --name <name>` (or start from the closest built-in: `mimi`, `lfm25`, `yolo`).
2. Edit `spec.py`: subclass `fastkernel.models.ModelSpec` (or `HFCausalLMSpec` / `TorchModuleSpec`):
   - `load_reference()` → deterministic eval model on CUDA (the oracle; fp32 or the deployment dtype);
   - `workloads()` → `Workload(name, make_inputs(device, seed) -> dict, run(model, inputs) -> outputs,
     primary=..., units={"audio_seconds"|"tokens"|"images": ...})`, plus sweeps (`tags=("sweep",)`) and
     `edge_workloads()` (short/odd/batched inputs);
   - `compare(workload, ref, cand) -> [GateCheck]` for model-specific semantics (exact discrete codes,
     top-1 agreement, IoU), otherwise the default allclose applies; `compare_determinism` if atomics are expected;
   - `notes` (architecture + known hot paths) and `hotspot_hints()` (symbol → category) for PLAN.md.
3. Edit `GOAL.md` frontmatter: `model: custom`, `target_metric`, `gates.precision`, `bench`, `model_args`.
4. `fast-kernel probe && fast-kernel baseline` — the baseline must pass its own gates (it always should; if
   not, the reference is non-deterministic: fix seeds/TF32/cuDNN settings in `load_reference`).
5. Optionally register it in `fastkernel/models/registry.py` and add a template under
   `fastkernel/templates/models/<name>/` so `fast-kernel init <name>` works for everyone.

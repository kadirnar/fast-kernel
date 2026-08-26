# Examples

Each directory is the campaign template that `fast-kernel init <name>` copies into `campaigns/<name>/`:
`GOAL.md` (objective, metric, quality policy, benchmark protocol), the agent-owned `candidate/` package
(identity `apply(model, ctx)` to start from), `PROMPT.md` (the one sentence you type) and model notes.

| directory | you type | model |
|---|---|---|
| `mimi/` | Optimize the Mimi codec model. | `kyutai/mimi` through `transformers` |
| `lfm25/` | Optimize the LFM2.5 model. | `LiquidAI/LFM2.5-1.2B-Instruct` through `transformers` |
| `lfm-audio/` | Optimize the LFM2 audio model. | `LiquidAI/LFM2.5-Audio-1.5B` through `liquid-audio` |
| `yolo/` | Optimize the YOLO model. | Ultralytics YOLO26n |
| `custom/` | Optimize the PyTorch model in ./spec.py. | any `torch.nn.Module` (edit `spec.py`) |

Every template keeps the default quality policy: outputs must stay identical to the original model.
The model loaders, reference oracles, workloads and correctness checks live in
`src/fastkernel/models/<name>.py`; the campaign only carries the goal and the candidate code.

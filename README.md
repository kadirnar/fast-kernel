# fast-kernel

fast-kernel makes a model run faster **without changing what it produces**.

You write one sentence. An agent (Claude Code) then profiles the model, finds the slow parts,
tries an optimization, measures it against the original model, keeps it only if it is faster
*and* the outputs still match, and repeats — for as long as you let it. Every experiment appears
on a live graph.

## Setup (once)

```bash
git clone https://github.com/kadirnar/fast-kernel && cd fast-kernel
uv sync --extra cuda
claude
```

You need an NVIDIA GPU, Python 3.12+, [uv](https://docs.astral.sh/uv/) and
[Claude Code](https://code.claude.com).

## Use

Type this in Claude Code:

```
Optimize the Mimi codec model.
```

That is all. Start Claude Code inside the `fast-kernel` folder: the sentence is turned into a plan
(`fast-kernel resolve` maps "Mimi codec" to the model and to the folder `campaigns/mimi/`, creating
it the first time and reusing it later). The agent measures the original model there, opens the
dashboard (it prints the address, usually http://127.0.0.1:8765) and starts experimenting.

Other models work the same way:

```
Optimize the LFM2.5 model.
Optimize the LFM2 audio model.
Optimize the YOLO model.
Optimize the PyTorch model in ./path/to/my_model.py.
```

To stop, type:

```
Stop optimizing.
```

Typing the first sentence again later continues from where it left off.

## Quality

Speed never comes at the cost of quality:

- Every experiment is compared with the **unmodified original model** on the same inputs.
- The outputs must match: Mimi must produce the **same audio codes** and the same waveform,
  language models the **same tokens**, YOLO the **same boxes and classes**. An experiment that
  changes them is thrown away automatically, even if it is faster.
- Results must be **deterministic** and must hold on short, odd-length and batched inputs too.
- The agent cannot loosen these checks. They live outside the code it is allowed to edit.

## What you get

Inside `campaigns/mimi/` (or the model you chose):

- `results.tsv` — one line per experiment: kept, discarded or crashed, with the numbers
- `experiments/` — metrics, correctness checks, profile, patch and log of every experiment
- `KNOWLEDGE.md` — what the agent learned
- `report.html` — the graph as a single file you can open anywhere (`uv run fast-kernel report`)

Example from an RTX 5070 Ti: Mimi encode+decode of one second of audio went from 18.9 ms to
3.1 ms (6×) in five experiments, with identical codes throughout.

## Models

| say | model |
|---|---|
| Mimi codec | `kyutai/mimi` through `transformers` |
| LFM2.5 | `LiquidAI/LFM2.5-1.2B-Instruct` through `transformers` |
| LFM2 audio | `LiquidAI/LFM2.5-Audio-1.5B` through `liquid-audio` (`uv sync --extra audio`) |
| YOLO | Ultralytics YOLO26n (`uv sync --extra yolo`) |
| a PyTorch model file | any `torch.nn.Module` |

## More

`docs/` explains how it works: the pipeline, the correctness checks, the dashboard, the
backends (Triton, TileLang, CuTe DSL, CUDA C++, CUDA graphs, torch.compile) and running
several agents at once. MIT license.

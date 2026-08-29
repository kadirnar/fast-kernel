# fast-kernel

fast-kernel makes a model run faster **without changing what it produces**. You give an agent
(Claude Code) the prompt below; it profiles the model, finds the slow parts, writes and tests
hand-written CUDA C++ kernels captured with CUDA graphs, measures each attempt against the original
model, keeps only what is faster *and* identical in output, and repeats until the optimization is
exhausted — with several agents working at once and every experiment on a live graph.

## Setup (once)

```bash
git clone https://github.com/kadirnar/fast-kernel && cd fast-kernel
uv sync --extra cuda          # add --extra audio for LFM2-Audio, --extra yolo for YOLO
claude                        # start Claude Code inside this folder
```

You need an NVIDIA GPU, Python 3.12+, [uv](https://docs.astral.sh/uv/) and [Claude Code](https://code.claude.com).

## The prompt (Mimi codec)

Paste this into Claude Code. The first line alone already works — `CLAUDE.md` routes it to the
`fk-optimize` skill, which reads the same files and runs the same steps — but the full text tells the
agent exactly what to read, how the library works, how to use the other agents, and what quality means.

```text
Optimize the Mimi codec model.

Optimize the Mimi neural audio codec (`kyutai/mimi`, loaded through `transformers.MimiModel`) with
fast-kernel: make encode + decode of one second of 24 kHz audio (batch 1, the primary workload
`roundtrip_1s`) as fast as possible on this machine, and keep improving it until the optimization is exhausted.
Speed is only accepted without any loss of quality: the optimized model must produce exactly the same
audio codes and the same waveform as the original.

## What to read before doing anything (in this order)

1. `README.md`, `AGENTS.md` (the research program — every rule below is binding) and `CLAUDE.md`.
2. `.claude/agents/*.md` (the roles you may delegate to) and these skills in `.claude/skills/`:
   `fk-optimize`, `fk-experiment`, `fk-parallel`, `hotspot-analysis`, `numerical-verification`, and the
   backend skills `cuda-cpp-kernels` (the implementation backend), `cuda-graphs`, `hub-kernels`.
3. The harness you must never edit but must understand: `src/fastkernel/harness/gates.py` (the five
   correctness stages), `src/fastkernel/harness/evaluate.py` (keep/revert logic), `src/fastkernel/harness/run.py`,
   `src/fastkernel/profiling/` (how targets are ranked by measured headroom), `src/fastkernel/memory.py`
   (the campaign's measured memory: reflexions, repair chains, failure classes),
   `src/fastkernel/backends/templates/` (starter kernels) and `src/fastkernel/backends/graphs.py`.
4. The model spec: `src/fastkernel/models/mimi.py` — it defines the reference oracle, the workloads, the exact
   correctness checks and the hotspot hints. Read the model's own source too (`transformers/models/mimi/modeling_mimi.py` in site-packages).
5. The campaign, once it exists: `campaigns/mimi/GOAL.md` (objective, metric, quality policy),
   `PLAN.md` (ranked targets measured on this machine: share of GPU time, roofline efficiency (SOL),
   shapes), `KNOWLEDGE.md` (insights and the experiment log), `results.tsv`, and
   `experiments/NNNN-*/` (metrics, gates,
   profile, patch, log of every experiment).

## How the library works (do it in this order; every step is idempotent)

1. `uv run fast-kernel resolve "<this sentence>" --json` — maps the sentence to the model, to the
   absolute campaign folder `campaigns/mimi` (created the first time, reused later) and to the
   remaining steps. Run every later command as `cd <campaign> && uv run fast-kernel ...`.
2. `uv run fast-kernel doctor` — torch/CUDA/backends/claude present; apply its fix hints
   (`uv sync --extra cuda`).
3. `uv run fast-kernel init mimi` (only if the campaign does not exist).
4. `uv run fast-kernel probe` — measures this GPU (bandwidth, TFLOPS, launch latency) and compiles one
   probe kernel per backend (CUDA C++ — the implementation backend — CUDA graphs, hub kernels) →
   `capabilities.json`. A backend that fails to compile is fixed (`uv run fast-kernel toolchain
   install --cuda <version>`, `uv pip install ...`) and probed again — never recorded as a limitation.
5. `uv run fast-kernel baseline` — experiment #0: the unmodified reference model passes its own five
   gates, is benchmarked with the fixed protocol (warm-up, clock ramp, median of N synchronised runs),
   the noise floor is measured, and the profile ranks the targets into `PLAN.md`.
6. `uv run fast-kernel dashboard --root campaigns` in the background — the live graph of every
   experiment (it prints the URL; tell me once).
7. `uv run fast-kernel loop start` — the Stop hook keeps this session iterating.
8. The loop: `fast-kernel brief` (state, plateau streak, ranked targets with their measured memory, last experiments, insights) → PLAN.md / KNOWLEDGE.md for depth →
   `fast-kernel memory --target <id>` (what was already measured on this target: what worked, which
   failures not to repeat) → one hypothesis for the target with the most measured headroom
   (share x (1 - roofline efficiency)); which technique to use is yours to discover from the measurements,
   nothing prescribes it (the backend is CUDA C++) → implement it only under `candidate/`
   (`candidate/__init__.py: apply(model, ctx)`, kernels in `candidate/kernels/`) → self-test on the real
   shapes → `fast-kernel eval -m "<one line>" --technique <ids> --target <id>` → read the verdict →
   `fast-kernel note "<insight with numbers>"` → next. Each experiment builds on the accepted incumbent
   (git branch `fast-kernel/mimi`, tags `exp-N`); rejected ones are reverted, their patches kept.

## Discover, do not assume

Nothing about the hardware or the model is given to you in advance, and no measurement is handed to you.
Where the time goes is measured by `fast-kernel profile` on this machine; whether the toolchain builds is
measured by `fast-kernel probe`; whether an idea helps is measured by `fast-kernel eval`. Every hypothesis
comes from those numbers and from what earlier experiments taught (KNOWLEDGE.md, `fast-kernel memory`).
Any technique may be tried in any order; ideas that failed here are facts about this model on this machine,
recorded with numbers, never generalised. The one thing that is *not* yours to choose is the implementation
backend: every kernel is hand-written CUDA C++ captured with CUDA graphs (stock torch and pre-built hub
kernels stay legitimate answers), for the measured reasons in AGENTS.md. Never write a device or vendor name
into notes, code or reports, and never conclude that something is unsupported — fix the environment and
re-probe.

## Multi-agent structure (use it)

You are the orchestrator (`fk-orchestrator`). Delegate with the Agent tool:
- `fk-profiler` after the baseline and after every surprising result: explain where the time goes
  (launch-bound vs compute vs memory), re-rank, name the next hypothesis.
- `fk-kernel-engineer`, one per target, in parallel: `uv run fast-kernel worktree create eng-<target>`
  gives each a private worktree branched from the incumbent; each engineer writes its kernel there,
  self-tests, and ends with `fast-kernel propose -m "..." --technique <id> --target <id>`. Then you run
  `uv run fast-kernel inbox`, which applies and measures every proposal serially on the incumbent
  (explore in parallel, measure serially — one GPU, one lineage). Details: `/fk-parallel`.
- `fk-verifier` on every failed gate (root cause and the minimal exact fix), `fk-reviewer` before an
  expensive evaluation (API preserved? hidden host syncs? simpler version?), `fk-benchmarker` when
  numbers look noisy, `fk-librarian` for references, templates and prior experiments.
- Alternatively run headless workers: `uv run fast-kernel auto --agents 3 --islands 2` (worktrees,
  hotspot leases, island populations that explore different bands of the ranked targets, inbox
  promotion) and keep watching the dashboard. `fast-kernel beam` shows the top-k accepted candidates —
  the search population, not just the single incumbent.

## Rules (from AGENTS.md; the harness enforces them, you follow them)

- Edit only `candidate/`. Never touch `GOAL.md`, `spec.py`, `results.tsv`, `experiments/`,
  `.fast-kernel/` or the `fastkernel` package (a hook blocks it). Never weaken, skip or shrink a gate,
  a workload or the policy; never hand-edit measurements.
- Quality contract: outputs must match the original model — identical discrete audio codes, decoded waveform allclose (rtol 2e-4, atol 2e-5), identical results on the 0.25 s / 5 s / noise inputs and on the 50 ms, odd-length and batch-2 edge cases. Under the default `strict`
  policy this is checked on every workload, deterministically, and on the edge inputs. You never change
  the policy; if an idea needs looser numerics, write "needs the human's decision" in KNOWLEDGE.md and move on.
- Crashes: `fast-kernel show <N> --log` also prints a failure class (compile, import, shape,
  illegal-memory, oom, timeout, numerical, determinism, …) — route by it; trivial (typo, import, stride,
  meta tensor) → fix and rerun once; fundamental → note it and take the next idea; the same crash three
  times → abandon the approach.
- A missing library is never a blocker and never a reason to ask me: install it yourself
  (`uv pip install ...`; the harness also auto-installs a backend's package on probe) and note it.
- Plateaus (5 discards on a target): change approach, then backend, then target; then widen the
  scope (combine kept kernels, remove copies between them); re-profile. The list of ideas is never empty.
- Verdicts are keep / **bank** / discard / crash. A bank means the gain was real but smaller than this
  machine can resolve in one measurement: the harness commits it and leaves it in `candidate/`, and the
  next experiment builds on top of it until the pile is jointly large enough to promote. Treat a bank as
  progress, keep going on the same lineage, and do not bundle unrelated edits to clear the threshold.
- Speed is judged by timing your candidate and the reference model interleaved in one process, so drift
  between sessions cannot be mistaken for a speedup. The threshold is that measurement's own uncertainty.
- Two or more targets with real headroom in `PLAN.md` means run them **in parallel** (one
  `fk-kernel-engineer` per target, or `uv run fast-kernel auto --agents 3 --islands 2`) rather than
  walking the list one experiment at a time. Serial is for when one target dominates.
- Report as short lines: `#N kept/banked/discarded/crashed · <metric> (Δ) · <speedup>x vs baseline · next: <idea>`.
  Never end a turn with a question. Never ask whether to continue.

## Stopping

Never ask me whether to continue and never ask me to run a stop command. The loop ends itself when the
optimization is exhausted — a measured signal: 15 consecutive experiments that ran but improved nothing
(the `loop.active` flag is then cleared automatically). I may still stop it early by writing
"Stop optimizing.", which is your cue to run `fast-kernel loop stop` (and `fast-kernel stop` if headless
workers run). The campaign is persisted either way; the same sentence later continues it.
```

The loop stops itself once nothing measurable is left. To stop it earlier: type `Stop optimizing.` —
the campaign is saved; the same prompt later continues it.

## Other models

The same prompt with a different first paragraph, ready to paste:
[`examples/lfm25/PROMPT.md`](examples/lfm25/PROMPT.md) (LFM2.5),
[`examples/lfm-audio/PROMPT.md`](examples/lfm-audio/PROMPT.md) (LFM2-Audio),
[`examples/yolo/PROMPT.md`](examples/yolo/PROMPT.md) (YOLO),
[`examples/custom/PROMPT.md`](examples/custom/PROMPT.md) (your own PyTorch model).

## What you will see

- `campaigns/mimi/results.tsv` — one line per experiment (kept / discarded / crashed, with numbers)
- `campaigns/mimi/experiments/NNNN-*/` — metrics, correctness checks, profile, patch and log of each
- `campaigns/mimi/KNOWLEDGE.md` — what the agents learned; `PLAN.md` — the current ranked targets
- the dashboard (URL printed at start; `uv run fast-kernel report` writes the same graph to one HTML file)

Results depend on your GPU and model; `results.tsv` and the dashboard show what was measured on your machine.

## Quality

Every experiment is compared with the unmodified original model on the same inputs: Mimi must give the
same codes and waveform, language models the same tokens, YOLO the same boxes and classes; results
must be deterministic and hold on short, odd-length and batched inputs. An experiment that changes
outputs is discarded automatically, however fast it is. The checks live outside the code the agents
may edit, and the agents may not change the policy.

## How it works

`docs/ARCHITECTURE.md`, `docs/PIPELINE.md`, `docs/MULTI_AGENT.md`, `docs/DASHBOARD.md`,
`docs/MODELS.md`, `docs/BACKENDS.md`. MIT license.

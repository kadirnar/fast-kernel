# The prompt for the LFM2 audio model

Type this in Claude Code, started inside the fast-kernel folder. The first line alone is enough — the
skills already contain the rest — but the full text makes every expectation explicit.

```text
Optimize the LFM2 audio model.

Optimize LiquidAI's LFM2.5-Audio-1.5B (the `liquid_audio` LFM2AudioModel built on the Transformers Lfm2
backbone, a FastConformer audio encoder and a Mimi-compatible 8-codebook decoder) with fast-kernel:
make text-to-speech generation of one sentence (the primary workload `tts`) and ASR of 2 s of audio as
fast as possible, and keep improving for as long as I let you run. Speed is only accepted without any
loss of quality: the optimized model must produce exactly the same tokens and audio codes as the original.

## What to read before doing anything (in this order)

1. `README.md`, `AGENTS.md` (the research program — every rule below is binding) and `CLAUDE.md`.
2. `.claude/agents/*.md` (the roles you may delegate to) and these skills in `.claude/skills/`:
   `fk-optimize`, `fk-experiment`, `fk-parallel`, `hotspot-analysis`, `numerical-verification`, and the
   backend skills `cuda-graphs`, `triton-kernels`, `tilelang-kernels`, `cute-dsl-kernels`,
   `cuda-cpp-kernels`, `torch-compile`, `hub-kernels`.
3. The harness you must never edit but must understand: `src/fastkernel/harness/gates.py` (the five
   correctness stages), `src/fastkernel/harness/evaluate.py` (keep/revert logic), `src/fastkernel/harness/run.py`,
   `src/fastkernel/profiling/` (how targets are ranked), `src/fastkernel/playbook.py` (technique ids),
   `src/fastkernel/backends/templates/` (starter kernels) and `src/fastkernel/backends/graphs.py`.
4. The model spec: `src/fastkernel/models/lfm_audio.py` — it defines the reference oracle, the workloads, the exact
   correctness checks and the hotspot hints. Read the model's own source too (the `liquid_audio` package and `transformers/models/lfm2/modeling_lfm2.py` in site-packages).
5. The campaign, once it exists: `campaigns/lfm-audio/GOAL.md` (objective, metric, quality policy),
   `PLAN.md` (ranked targets with shapes and technique status), `RECIPES.md` (measured, ordered recipes),
   `KNOWLEDGE.md` (insights and the experiment log), `results.tsv`, and `experiments/NNNN-*/` (metrics,
   gates, profile, patch, log of every experiment).

## How the library works (do it in this order; every step is idempotent)

1. `uv run fast-kernel resolve "<this sentence>" --json` — maps the sentence to the model, to the
   absolute campaign folder `campaigns/lfm-audio` (created the first time, reused later) and to the
   remaining steps. Run every later command as `cd <campaign> && uv run fast-kernel ...`.
2. `uv run fast-kernel doctor` — torch/CUDA/backends/claude present; apply its fix hints
   (`uv sync --extra cuda`, `uv sync --extra cuda --extra audio` for liquid-audio).
3. `uv run fast-kernel init lfm-audio` (only if the campaign does not exist).
4. `uv run fast-kernel probe` — GPU roofline numbers and one compiled probe kernel per backend
   (Triton, TileLang, CuTe DSL, CUDA C++, torch.compile, CUDA graphs, hub kernels) → `capabilities.json`.
   A backend that fails with a host-compiler/nvcc error is fixed with
   `uv run fast-kernel toolchain install --cuda 13.3` and probed again — never recorded as a hardware limit.
5. `uv run fast-kernel baseline` — experiment #0: the unmodified reference model passes its own five
   gates, is benchmarked with the fixed protocol (warm-up, clock ramp, median of N CUDA-synchronised
   runs), the noise floor is measured, and the profile ranks the targets into `PLAN.md`.
6. `uv run fast-kernel dashboard --root campaigns` in the background — the live graph of every
   experiment (it prints the URL; tell me once).
7. `uv run fast-kernel loop start` — the Stop hook keeps this session iterating.
8. The loop, forever: `fast-kernel status --brief` → `fast-kernel ideas` → read PLAN.md / KNOWLEDGE.md /
   RECIPES.md → one hypothesis with the largest end-to-end gain (share x (1 - 1/expected), untried
   first, recipe order, lower tier first) → implement it only under `candidate/`
   (`candidate/__init__.py: apply(model, ctx)`, kernels in `candidate/kernels/`) → self-test on the real
   shapes → `fast-kernel eval -m "<one line>" --technique <ids> --target <id>` → read the verdict →
   `fast-kernel note "<insight with numbers>"` → next. Each experiment builds on the accepted incumbent
   (git branch `fast-kernel/lfm-audio`, tags `exp-N`); rejected ones are reverted, their patches kept.

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
- Alternatively run headless workers: `uv run fast-kernel auto --agents 3` (worktrees, hotspot leases,
  inbox promotion) and keep watching the dashboard.

## Rules (from AGENTS.md; the harness enforces them, you follow them)

- Edit only `candidate/`. Never touch `GOAL.md`, `spec.py`, `results.tsv`, `experiments/`,
  `.fast-kernel/` or the `fastkernel` package (a hook blocks it). Never weaken, skip or shrink a gate,
  a workload or the policy; never hand-edit measurements.
- Quality contract: outputs must match the original model — identical text tokens and audio codes for the seeded TTS and ASR workloads, decoded waveform SNR >= 40 dB against the original. Under the default `strict`
  policy this is checked on every workload, deterministically, and on the edge inputs. You never change
  the policy; if an idea needs looser numerics, write "needs the human's decision" in KNOWLEDGE.md and move on.
- No hardware limitations: `capabilities.json` is evidence. Missing packages, headers, nvcc, weights →
  install/fix (`uv pip install`, `fast-kernel toolchain install`), re-probe, continue.
- Crashes: `fast-kernel show <N> --log`; trivial (typo, import, stride, meta tensor) → fix and rerun
  once; fundamental → note it and take the next idea; the same crash three times → abandon the approach.
- Plateaus (5 discards on a target): change technique tier, then backend
  (Triton ↔ TileLang ↔ CuTe DSL ↔ CUDA C++ ↔ hub kernels), then target; then widen the scope (combine
  kept kernels, remove copies between them); re-profile. The list of ideas is never empty.
- Report as short lines: `#N kept/discarded/crashed · <metric> (Δ) · <speedup>x vs baseline · next: <idea>`.
  Never end a turn with a question. Never ask whether to continue.

## Stopping

Only I stop this: when I write "Stop optimizing." run `fast-kernel loop stop` (and `fast-kernel stop`
if headless workers run). The campaign is persisted; the same sentence later continues it.

## Where the time goes and what to try

Every generated frame runs the 1.2B backbone once and the depthformer 8 times (one per codebook):
launch-bound GEMV chains. In order (see RECIPES.md): CUDA graphs on the per-step backbone forward (static
cache) and on the depthformer loop; the backbone kernels shared with the LFM2.5 campaign (fused RMSNorm,
merged gate/up + silu*mul, fused ShortConv); the Mimi decoder kernels shared with the Mimi campaign
(implicit-GEMM convs, fused codebook lookup); for ASR the FastConformer encoder (fused rel-pos attention,
channels-last convs with fused activations).
```

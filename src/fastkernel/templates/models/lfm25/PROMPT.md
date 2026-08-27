# The prompt for the LFM2.5 model

Type this in Claude Code, started inside the fast-kernel folder. The first line alone is enough — the
skills already contain the rest — but the full text makes every expectation explicit.

```text
Optimize the LFM2.5 model.

Optimize LiquidAI's LFM2.5-1.2B-Instruct (`transformers.AutoModelForCausalLM`, `Lfm2ForCausalLM`, bf16)
with fast-kernel: make greedy decoding of 64 tokens after a 64-token prompt (batch 1, the primary
workload `decode`) as fast as possible on this machine, keep prefill of 512 tokens fast too, and keep
improving until the optimization is exhausted. Speed is only accepted without any loss of quality: the
optimized model must generate exactly the same tokens as the original.

## What to read before doing anything (in this order)

1. `README.md`, `AGENTS.md` (the research program — every rule below is binding) and `CLAUDE.md`.
2. `.claude/agents/*.md` (the roles you may delegate to) and these skills in `.claude/skills/`:
   `fk-optimize`, `fk-experiment`, `fk-parallel`, `hotspot-analysis`, `numerical-verification`, and the
   backend skills `cuda-graphs`, `triton-kernels`, `tilelang-kernels`, `cute-dsl-kernels`,
   `cuda-cpp-kernels`, `torch-compile`, `hub-kernels`.
3. The harness you must never edit but must understand: `src/fastkernel/harness/gates.py` (the five
   correctness stages), `src/fastkernel/harness/evaluate.py` (keep/revert logic), `src/fastkernel/harness/run.py`,
   `src/fastkernel/profiling/` (how targets are ranked by measured headroom), `src/fastkernel/memory.py`
   (the campaign's measured memory: reflexions, repair chains, failure classes),
   `src/fastkernel/backends/templates/` (starter kernels) and `src/fastkernel/backends/graphs.py`.
4. The model spec: `src/fastkernel/models/lfm25.py (and hf_causal_lm.py it extends)` — it defines the reference oracle, the workloads, the exact
   correctness checks and the hotspot hints. Read the model's own source too (`transformers/models/lfm2/modeling_lfm2.py` in site-packages).
5. The campaign, once it exists: `campaigns/lfm25/GOAL.md` (objective, metric, quality policy),
   `PLAN.md` (ranked targets measured on this machine: share of GPU time, roofline efficiency (SOL),
   shapes), `KNOWLEDGE.md` (insights and the experiment log), `results.tsv`, and
   `experiments/NNNN-*/` (metrics, gates,
   profile, patch, log of every experiment).

## How the library works (do it in this order; every step is idempotent)

1. `uv run fast-kernel resolve "<this sentence>" --json` — maps the sentence to the model, to the
   absolute campaign folder `campaigns/lfm25` (created the first time, reused later) and to the
   remaining steps. Run every later command as `cd <campaign> && uv run fast-kernel ...`.
2. `uv run fast-kernel doctor` — torch/CUDA/backends/claude present; apply its fix hints
   (`uv sync --extra cuda`).
3. `uv run fast-kernel init lfm25` (only if the campaign does not exist).
4. `uv run fast-kernel probe` — measures this GPU (bandwidth, TFLOPS, launch latency) and compiles one
   probe kernel per backend (Triton, TileLang, CuTe DSL, CUDA C++, torch.compile, CUDA graphs, hub
   kernels) → `capabilities.json`. A backend that fails to compile is fixed (`uv run fast-kernel toolchain
   install --cuda <version>`, `uv pip install ...`) and probed again — never recorded as a limitation.
5. `uv run fast-kernel baseline` — experiment #0: the unmodified reference model passes its own five
   gates, is benchmarked with the fixed protocol (warm-up, clock ramp, median of N synchronised runs),
   the noise floor is measured, and the profile ranks the targets into `PLAN.md`.
6. `uv run fast-kernel dashboard --root campaigns` in the background — the live graph of every
   experiment (it prints the URL; tell me once).
7. `uv run fast-kernel loop start` — the Stop hook keeps this session iterating.
8. The loop: `fast-kernel status --brief` → `fast-kernel ideas` → read PLAN.md / KNOWLEDGE.md →
   `fast-kernel memory --target <id>` (what was already measured on this target: what worked, which
   failures not to repeat) → one hypothesis for the target with the most measured headroom
   (share x (1 - roofline efficiency)); which technique and backend to use is yours to discover from the
   measurements, nothing prescribes it → implement it only under `candidate/`
   (`candidate/__init__.py: apply(model, ctx)`, kernels in `candidate/kernels/`) → self-test on the real
   shapes → `fast-kernel eval -m "<one line>" --technique <ids> --target <id>` → read the verdict →
   `fast-kernel note "<insight with numbers>"` → next. Each experiment builds on the accepted incumbent
   (git branch `fast-kernel/lfm25`, tags `exp-N`); rejected ones are reverted, their patches kept.

## Discover, do not assume

Nothing about the hardware or the model is given to you in advance and nothing constrains what you may
try. Where the time goes is measured by `fast-kernel profile` on this machine; which backends compile is
measured by `fast-kernel probe`; whether an idea helps is measured by `fast-kernel eval`. Every hypothesis
comes from those numbers and from what earlier experiments taught (KNOWLEDGE.md, `fast-kernel memory`).
Any technique and any backend may be tried in any order; ideas that failed here are facts about this
model on this machine, recorded with numbers, never generalised. Never write a device or vendor name into
notes, code or reports, and never conclude that something is unsupported — fix the environment, re-probe,
or take another backend.

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
- Quality contract: outputs must match the original model — identical greedy tokens on the decode workload, top-1 agreement >= 99.5 % and top-5 overlap >= 0.9 on the prefill logits, identical results on the odd-length and short edge cases. Under the default `strict`
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
- Report as short lines: `#N kept/discarded/crashed · <metric> (Δ) · <speedup>x vs baseline · next: <idea>`.
  Never end a turn with a question. Never ask whether to continue.

## Stopping

Never ask me whether to continue and never ask me to run a stop command. The loop ends itself when the
optimization is exhausted — a measured signal: 15 consecutive experiments that ran but improved nothing
(the `loop.active` flag is then cleared automatically). I may still stop it early by writing
"Stop optimizing.", which is your cue to run `fast-kernel loop stop` (and `fast-kernel stop` if headless
workers run). The campaign is persisted either way; the same sentence later continues it.
```

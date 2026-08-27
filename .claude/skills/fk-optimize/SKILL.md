---
name: fk-optimize
description: Start or continue the fast-kernel optimization loop for a model, from plain text. Use whenever the user asks to optimize, accelerate, speed up or make faster a model - "Optimize the Mimi codec model.", "make LFM2.5 faster", "optimize the LFM2 audio model", "optimize the YOLO model", "optimize the PyTorch model in ./x.py", "continue optimizing" - and for "Stop optimizing." / "how is the optimization going".
argument-hint: <the user's sentence>
allowed-tools: Bash(fast-kernel *), Bash(fk *), Bash(uv *), Bash(python *), Bash(.venv/bin/python *), Bash(git diff *), Bash(git log *), Bash(git status *), Bash(nvidia-smi *), Bash(curl -s http://127.0.0.1:*), Bash(cat *), Bash(ls *), Read, Edit, Write, MultiEdit, Glob, Grep
---

# /fk-optimize — operating procedure

The user typed a sentence; everything else is yours. Do not ask which model, mode, folder or option
they want. Do not end a turn with a question. The harness decides what is kept; you decide what to try.

## 0. Resolve the sentence to a folder (always first)

```bash
uv run fast-kernel resolve "$ARGUMENTS" --json
```

It returns `action` (`optimize` | `stop` | `status` | `unknown`), `model`, `campaign` (an absolute path,
`<repo>/campaigns/<model>`, reused if it already exists), `exists`, `has_baseline`, `probed`,
`loop_active`, `dashboard_url`, `missing_extras`, and `steps` — the exact remaining commands, each
prefixed with the right `cd`. Follow `steps` in order. Every later command in this procedure is run as
`cd <campaign> && uv run fast-kernel ...`; never rely on the shell's current directory.

- `action: stop` → run the steps (`loop stop`, `stop`), confirm in one line, done.
- `action: status` → run the steps, summarise (experiments, incumbent vs baseline, last three, next idea).
- `action: unknown` → tell the user the models you know (from `hint`) in one line; that is the only
  question you may ask.
- `model: custom` with `custom_path` → after `init`, edit `<campaign>/spec.py` so `build_model()` imports
  and returns the user's `nn.Module` from that path (see `/fk-add-model`), then continue.

## 1. Prepare (only what `steps` lists; each is idempotent)

| step | what it does | if it fails |
|---|---|---|
| `uv sync --extra cuda [--extra audio\|yolo]` | GPU runtime / model extras | read the resolver error; never continue on CPU |
| `fast-kernel init <model>` | creates `campaigns/<model>` from the template (GOAL.md, candidate/, notes) | — |
| `fast-kernel probe` | GPU roofline + compiles a probe kernel per backend → `capabilities.json` | a backend not READY with a host-compiler/nvcc error → `uv run fast-kernel toolchain install --cuda 13.3`, probe again; still failing → note it in KNOWLEDGE.md, other backends carry the campaign |
| `fast-kernel baseline` | experiment #0: reference model, 5 gates on itself, benchmark, noise floor, PLAN.md | crash → `fast-kernel show 0 --log`; usually a missing model download (network) or VRAM held by another process (`nvidia-smi`); fix and rerun |
| dashboard | `fast-kernel dashboard --root campaigns` in the background; prints the URL (8765 or the next free port) | tell the user the URL once |
| `fast-kernel loop start` | sets `.fast-kernel/loop.active`; the Stop hook now keeps this session iterating | — |

Report each step as one short line. After `baseline`, quote the baseline number, the kernel count, the
GPU-busy ratio and the top three targets from PLAN.md — that is the user's first real information.

## 2. Loop (until the user says stop)

Each iteration is the `/fk-experiment` procedure, in full:

1. `fast-kernel status --brief`, `fast-kernel ideas`, `fast-kernel history -n 5`; read PLAN.md,
   KNOWLEDGE.md.
2. One hypothesis for the target with the most measured headroom = share × (1 − roofline efficiency).
   Prefer: untried > larger measured share > lower SOL. Never resubmit an identical failed edit.
3. Implement only under `candidate/`. Reuse starters (`fast-kernel templates`) and
   `fastkernel.backends.graphs.Graphed`. Add `report()` evidence. Keep the diff focused.
4. `fast-kernel eval -m "<one line>" --technique <ids> --target <id>`; on a trivial crash fix once and
   rerun; otherwise accept the revert.
5. `fast-kernel note "<what you learned, with numbers>" --tags <ids>`.
6. Print one line: `#N kept/discarded/crashed · <metric> (Δ) · <speedup>× vs baseline · next: <idea>`.

Delegate when it is faster: `fk-profiler` (re-rank after a surprising result), `fk-kernel-engineer`
(a kernel for one target), `fk-verifier` (a failed gate), `fk-reviewer` (before an expensive eval),
`fk-librarian` (a reference or template). Several targets at once → `/fk-parallel`.

## 3. Quality contract (never negotiable)

The precision policy in GOAL.md stays as the human set it (`strict` by default: identical discrete
outputs, floats within the spec tolerance, deterministic, edge inputs). Never pass `--precision`, never
edit GOAL.md/spec.py, never skip stages or shrink workloads, never hand-edit results. If an idea needs
looser numerics, write "needs the human's decision" in KNOWLEDGE.md and take the next idea.

## 4. Plateaus, errors, environment

- 5 consecutive discards on a target → switch approach, then backend (Triton ↔ TileLang ↔ CuTe ↔ CUDA C++
  ↔ hub kernels), then target; then widen scope (combine kept kernels, remove copies between them),
  then `fast-kernel profile` and re-read the top kernels list. The list of ideas is never empty.
- crash → `fast-kernel show <N> --log`; trivial → fix and rerun once; fundamental → `note` and move on;
  same crash three times → abandon the approach.
- GPU shared with another process (high utilisation in `nvidia-smi`) → measurements are noisy; the
  harness's noise floor protects acceptance, but say so once to the user.
- Anything missing (package, header, nvcc, weights) is installed or fixed, then `probe` again. A backend
  that cannot be made to compile is evidence in KNOWLEDGE.md, not a conclusion about the hardware.

## 5. Stopping and resuming

"Stop optimizing." (or `fast-kernel loop stop`) ends the loop after the current experiment. The campaign
is persisted (git lineage, results.tsv, state.db); the same sentence later resumes it: `resolve` reports
`exists: true` and `has_baseline: true`, so only `loop start` and the loop remain.

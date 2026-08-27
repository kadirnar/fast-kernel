# The optimization pipeline

```
capabilities  ->  trace  ->  attribute  ->  classify  ->  rank (headroom)  ->  PLAN.md
   (probe)      profiler   modules+frames   roofline     share x (1 - SOL)    + hotspots.json
```

## 1. Capabilities (`fast-kernel probe`)

Device facts (name, compute capability, SMs, memory, L2, smem) plus measured numbers the classifier
needs: device-to-device bandwidth (256 MB copy), bf16 and fp32 GEMM TFLOPS (4096^3), and launch
latency (2000 tiny kernels). Then every backend compiles and runs a probe kernel: Triton, TileLang,
CuTe DSL, CUDA C++ (torch `load_inline` with an auto-discovered nvcc), torch.compile, CUDA graphs,
hub kernels. Failures are recorded with the first error lines and a `fix` hint — evidence, never a
verdict. `fast-kernel toolchain install --cuda X.Y` installs a self-contained nvcc/CCCL/NVVM/runtime
wheel set when the system compiler is newer than torch's bundled nvcc supports.

## 2. Trace

`torch.profiler` (CPU + CUDA activities, `with_stack=True`) around one call of the primary workload
after warm-up, with a `record_function("fk::<module path>::<Class>")` scope pushed by forward
pre/post hooks on every `nn.Module`. The chrome trace is parsed: each GPU kernel is joined to its
launching CPU op (External id / correlation), the op to the innermost `fk::` scope, and — when the
op ran outside any `forward()` (Mimi's `quantizer.encode`, `MimiEuclideanCodebook.quantize`,
`generate()` loops) — to the innermost Python frame (`python_function` events) whose file/line falls
inside a model class's source range. Result: complete attribution of GPU time; per-module GPU
time, kernel count, top ops, plus wall time, GPU-busy time (union of kernel intervals), launch count.

Wall time is re-measured without the profiler; `gpu_busy_ratio < 0.6` flags a launch/overhead-bound
workload.

## 3. Classify (roofline)

Per module: FLOPs and bytes estimated from recorded shapes and parameters (Linear, Conv/ConvTranspose,
attention, norms, embeddings, MLPs), arithmetic intensity vs the measured ridge point, achieved
TFLOPS / GB/s and % of peak. Boundness: `latency` when the average kernel is shorter than ~2.5x the
launch latency, else `compute` above the ridge, else `memory`. Categories come from op names
(gemm, conv, attention, norm, elementwise, reduction, indexing, memory-movement, quantizer, sequential,
other) with spec hints overriding by class name.

## 4. Rank (measured headroom)

Instances are grouped by (class, category) because one kernel fixes all layers of a kind. For each
group: `share = group GPU time / total GPU time`, `sol_efficiency` = achieved bandwidth or FLOP/s over
this machine's measured Speed-of-Light peak, `headroom = share x (1 - sol_efficiency)`. A launch-bound workload adds a whole-workload target
with `share = GPU idle fraction of wall time`. The technique matrix from `results` (accepted / rejected /
crash per target x technique) orders techniques (untried first) and mildly demotes targets whose
ideas all failed. The ranking is recomputed after every kept experiment — shares move.

## 5. Technique catalogue (internal only)

PLAN.md and `fast-kernel ideas` deliberately show measured facts only — never a technique to use or a
predicted speedup; the agent discovers the method itself. The catalogue below is used internally (for
the experiment matrix and the dashboard) and is not rendered to the agent.

`playbook.py` holds the catalogue: tier 0 structure (CUDA graphs, torch.compile, kernel-count
reduction, weight pre-packing, dtype policy), tiers 1-2 block/memory tuning, tier 3 fusion (epilogues,
implicit-GEMM conv, fused attention/norm/quantizer/elementwise), tier 4 advanced (split-K, persistent
kernels, warp specialization), tier 5 architecture tuning, tier 6 kernel-specific tricks (indexing
fusion, decode-step fusion, hub kernels). Each technique names its backends and the skill that
documents how to do it.

## 6. Gates, benchmark, decision

Gates (`harness/gates.py`): smoke, shapes, numerical (spec-specific: exact codes, top-1 agreement,
box tolerances, allclose), determinism (candidate vs itself), edge (short/odd/batched inputs). Benchmark
(`harness/bench.py`): warm-up, 1 s clock ramp, N CUDA-synchronised repeats, median/min/p90/std, peak
VRAM, derived rtf / tokens/s / fps. Decision (`harness/evaluate.py`): crash -> revert; gates FAIL ->
revert; improvement >= max(`min_improvement`, baseline noise floor) -> keep; otherwise revert unless
`--simpler`. Every path records results.tsv, KNOWLEDGE.md, SQLite events, `experiments/NNNN-*/`.

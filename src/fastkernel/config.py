"""Campaign configuration: GOAL.md frontmatter + defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .frontmatter import split_frontmatter

DEFAULT_PROTECTED = ["GOAL.md", "spec.py", "harness/**", ".fast-kernel/**", "experiments/**", "results.tsv"]


@dataclass
class BenchPolicy:
    warmup: int = 5
    repeats: int = 50
    ramp_seconds: float = 1.0
    timeout_seconds: float = 900.0
    profile_every_experiment: bool = True
    # Accept/reject is decided on a paired, same-process comparison against the reference model
    # (see harness/bench.compare_callables): absolute milliseconds drift between sessions, the
    # ratio does not. Set anchor=false to fall back to comparing raw milliseconds across runs.
    anchor: bool = True
    anchor_pairs: int = 20
    # A measured improvement that is real but smaller than the noise floor used to be thrown away.
    # Instead it is *banked*: the candidate tree is left in place so the next experiment builds on
    # it, and the incumbent only moves once the accumulated tree clears the floor. At most this
    # many banks may accumulate before the next experiment has to settle the question.
    max_banked: int = 8


@dataclass
class GatePolicy:
    precision: str = "strict"      # strict | tolerant  (model specs map this to thresholds)
    determinism: str = "exact"     # exact | tolerant
    rtol: float | None = None      # optional explicit overrides for generic tensor comparison
    atol: float | None = None
    stages: list[str] = field(default_factory=lambda: ["smoke", "shapes", "numerical", "determinism", "edge"])


@dataclass
class GoalConfig:
    model: str = "custom"
    objective: str = "Make the primary workload as fast as possible without changing its outputs."
    target_metric: str = "latency_ms"
    direction: str = "minimize"    # minimize | maximize
    min_improvement: float = 0.01  # relative; the noise floor measured at baseline may raise it
    continuous: bool = True
    max_iterations: int | None = None
    primary_workload: str | None = None
    workloads: list[str] | None = None   # subset of spec workloads to run (None = all)
    model_args: dict[str, Any] = field(default_factory=dict)
    bench: BenchPolicy = field(default_factory=BenchPolicy)
    gates: GatePolicy = field(default_factory=GatePolicy)
    protected: list[str] = field(default_factory=lambda: list(DEFAULT_PROTECTED))
    body: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def minimize(self) -> bool:
        return self.direction.lower().startswith("min")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "objective": self.objective, "target_metric": self.target_metric,
            "direction": self.direction, "min_improvement": self.min_improvement, "continuous": self.continuous,
            "max_iterations": self.max_iterations, "primary_workload": self.primary_workload,
            "workloads": self.workloads, "model_args": self.model_args,
            "bench": self.bench.__dict__, "gates": self.gates.__dict__, "protected": self.protected,
        }


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_goal(path: Path) -> GoalConfig:
    text = Path(path).read_text(encoding="utf-8")
    data, body = split_frontmatter(text)
    cfg = GoalConfig(body=body.strip(), raw=data)
    cfg.model = str(data.get("model") or cfg.model)
    cfg.objective = str(data.get("objective") or cfg.objective)
    cfg.target_metric = str(data.get("target_metric") or cfg.target_metric)
    cfg.direction = str(data.get("direction") or cfg.direction)
    cfg.min_improvement = _as_float(data.get("min_improvement", data.get("minimum_improvement")), cfg.min_improvement)
    cfg.continuous = bool(data.get("continuous", True))
    max_iter = data.get("max_iterations")
    cfg.max_iterations = int(max_iter) if isinstance(max_iter, (int, float)) and max_iter else None
    cfg.primary_workload = data.get("primary_workload") or None
    workloads = data.get("workloads")
    cfg.workloads = [str(w) for w in workloads] if isinstance(workloads, list) else None
    model_args = data.get("model_args")
    cfg.model_args = dict(model_args) if isinstance(model_args, dict) else {}
    bench = data.get("bench") if isinstance(data.get("bench"), dict) else {}
    cfg.bench = BenchPolicy(
        warmup=int(bench.get("warmup", cfg.bench.warmup)),
        repeats=int(bench.get("repeats", cfg.bench.repeats)),
        ramp_seconds=_as_float(bench.get("ramp_seconds"), cfg.bench.ramp_seconds),
        timeout_seconds=_as_float(bench.get("timeout_seconds"), cfg.bench.timeout_seconds),
        profile_every_experiment=bool(bench.get("profile_every_experiment", True)),
    )
    gates = data.get("gates") if isinstance(data.get("gates"), dict) else {}
    cfg.gates = GatePolicy(
        precision=str(gates.get("precision", data.get("precision", "strict"))),
        determinism=str(gates.get("determinism", data.get("determinism", "exact"))),
        rtol=gates.get("rtol"), atol=gates.get("atol"),
        stages=[str(s) for s in gates.get("stages", cfg.gates.stages)],
    )
    protected = data.get("protected")
    if isinstance(protected, list) and protected:
        cfg.protected = [str(p) for p in protected]
    return cfg

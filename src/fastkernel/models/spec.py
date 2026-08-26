"""The ModelSpec contract every example implements.

A spec knows how to load the frozen *reference* model (the numerical oracle), how to build the
*candidate* (same weights + the agent's `candidate.apply`), which *workloads* to run (each with a
seeded input factory), how to *compare* candidate outputs to the reference, and which derived
metrics matter (rtf, tokens/s, fps). The harness is generic; everything model-specific lives here.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import GatePolicy


@dataclass
class Workload:
    name: str
    make_inputs: Callable[[Any, int], dict[str, Any]]      # (device, seed) -> kwargs for run()
    run: Callable[[Any, dict[str, Any]], Any]              # (model, inputs) -> outputs
    primary: bool = False
    bench: bool = True                                     # benchmark it?
    tags: tuple[str, ...] = ()                             # "sweep" | "edge" | "decode" ...
    describe: str = ""
    units: dict[str, float] = field(default_factory=dict)  # e.g. {"audio_seconds": 1.0, "tokens": 512}
    compare_policy: dict[str, Any] = field(default_factory=dict)

    @property
    def is_edge(self) -> bool:
        return "edge" in self.tags


@dataclass
class GateCheck:
    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": bool(self.passed), "value": _finite(self.value),
                "threshold": _finite(self.threshold), "detail": self.detail}


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


@dataclass
class CandidateContext:
    """What candidate.apply(model, ctx) receives."""
    campaign_root: Path
    device: Any
    spec_name: str
    capabilities: dict[str, Any]
    policy: GatePolicy
    model_args: dict[str, Any]
    workload_names: list[str]
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.logs.append(str(message))
        print(f"[candidate] {message}", flush=True)

    @property
    def strict(self) -> bool:
        return self.policy.precision == "strict"


class ModelSpec:
    """Base spec. Subclasses override load_reference / workloads / compare / hints."""

    name = "custom"
    display_name = "Custom torch module"
    hub_id: str | None = None
    notes = ""                          # architecture notes surfaced to the agent (PLAN.md)
    default_rtol = {"strict": 1e-4, "tolerant": 2e-2}
    default_atol = {"strict": 1e-5, "tolerant": 2e-2}

    def __init__(self, campaign_root: Path, args: dict[str, Any] | None = None, policy: GatePolicy | None = None):
        self.campaign_root = Path(campaign_root)
        self.args = dict(args or {})
        self.policy = policy or GatePolicy()

    # ---- loading -----------------------------------------------------------------------
    def load_reference(self) -> Any:
        raise NotImplementedError

    def load_candidate(self, ctx: CandidateContext) -> Any:
        model = self.load_reference()
        return apply_candidate(model, ctx)

    def hooks_root(self, model: Any) -> Any:
        """The nn.Module whose submodules get profiling hooks (default: the model itself)."""
        return model

    def free(self, model: Any) -> None:
        del model

    # ---- workloads ---------------------------------------------------------------------
    def workloads(self) -> list[Workload]:
        raise NotImplementedError

    def edge_workloads(self) -> list[Workload]:
        return []

    def primary_workload(self, preferred: str | None = None) -> Workload:
        items = self.workloads()
        if preferred:
            for item in items:
                if item.name == preferred:
                    return item
        for item in items:
            if item.primary:
                return item
        return items[0]

    # ---- comparison --------------------------------------------------------------------
    def tolerances(self, workload: Workload | None = None) -> tuple[float, float]:
        key = self.policy.precision if self.policy.precision in self.default_rtol else "strict"
        rtol = self.policy.rtol if self.policy.rtol is not None else self.default_rtol[key]
        atol = self.policy.atol if self.policy.atol is not None else self.default_atol[key]
        if workload and workload.compare_policy:
            rtol = workload.compare_policy.get("rtol", rtol)
            atol = workload.compare_policy.get("atol", atol)
        return float(rtol), float(atol)

    def compare(self, workload: Workload, reference: Any, candidate: Any) -> list[GateCheck]:
        rtol, atol = self.tolerances(workload)
        return compare_trees(reference, candidate, rtol=rtol, atol=atol, prefix=workload.name)

    def compare_determinism(self, workload: Workload, first: Any, second: Any) -> list[GateCheck]:
        if self.policy.determinism == "exact":
            return compare_trees(first, second, rtol=0.0, atol=0.0, prefix=f"{workload.name}/determinism", exact=True)
        return compare_trees(first, second, rtol=1e-5, atol=1e-6, prefix=f"{workload.name}/determinism")

    # ---- metrics -----------------------------------------------------------------------
    def derived_metrics(self, workload: Workload, latency_ms: float) -> dict[str, float]:
        out: dict[str, float] = {}
        units = workload.units
        if latency_ms <= 0:
            return out
        if "audio_seconds" in units:
            out["rtf"] = (latency_ms / 1000.0) / units["audio_seconds"]
            out["audio_x_realtime"] = units["audio_seconds"] / (latency_ms / 1000.0)
        if "tokens" in units:
            out["tokens_per_s"] = units["tokens"] / (latency_ms / 1000.0)
        if "images" in units:
            out["fps"] = units["images"] / (latency_ms / 1000.0)
        if "samples" in units:
            out["samples_per_s"] = units["samples"] / (latency_ms / 1000.0)
        return out

    # ---- hints -------------------------------------------------------------------------
    def hotspot_hints(self) -> list[dict[str, Any]]:
        """Static hints: [{"symbol": "MimiEuclideanCodebook", "category": "quantizer", "note": "..."}]"""
        return []


# ---- helpers -----------------------------------------------------------------------------
def apply_candidate(model: Any, ctx: CandidateContext) -> Any:
    module = load_candidate_module(ctx.campaign_root)
    apply_fn = getattr(module, "apply", None)
    if apply_fn is None:
        raise RuntimeError("candidate/__init__.py must define apply(model, ctx) -> model")
    out = apply_fn(model, ctx)
    return model if out is None else out


def load_candidate_module(campaign_root: Path):
    root = Path(campaign_root).resolve()
    init = root / "candidate" / "__init__.py"
    if not init.exists():
        raise RuntimeError(f"missing candidate package: {init}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for key in [k for k in sys.modules if k == "candidate" or k.startswith("candidate.")]:
        del sys.modules[key]
    return importlib.import_module("candidate")


def candidate_report(campaign_root: Path) -> dict[str, Any]:
    try:
        module = sys.modules.get("candidate") or load_candidate_module(campaign_root)
        report_fn = getattr(module, "report", None)
        data = report_fn() if report_fn else {}
        return dict(data) if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def load_spec(campaign_root: Path, model_name: str, args: dict[str, Any], policy: GatePolicy) -> ModelSpec:
    """Load `spec.py` from the campaign if present, else a built-in spec by name."""
    spec_file = Path(campaign_root) / "spec.py"
    if spec_file.exists():
        module_name = f"fk_spec_{abs(hash(str(spec_file)))}"
        spec = importlib.util.spec_from_file_location(module_name, spec_file)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        cls = getattr(module, "Spec", None)
        if cls is None:
            raise RuntimeError("spec.py must define a class named Spec (subclass of fastkernel.models.ModelSpec)")
        return cls(campaign_root, args, policy)
    from .registry import get_spec_class
    return get_spec_class(model_name)(campaign_root, args, policy)


def _flatten(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    import torch
    if isinstance(tree, torch.Tensor):
        return [(prefix or "out", tree)]
    if isinstance(tree, dict):
        out = []
        for key, value in tree.items():
            out.extend(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(tree, (list, tuple)):
        out = []
        for idx, value in enumerate(tree):
            out.extend(_flatten(value, f"{prefix}[{idx}]" if prefix else f"[{idx}]"))
        return out
    if hasattr(tree, "__dict__") and not isinstance(tree, (str, bytes, int, float)):
        # transformers ModelOutput and dataclasses
        items = tree.items() if hasattr(tree, "items") else vars(tree).items()
        out = []
        for key, value in items:
            if value is None or key.startswith("_"):
                continue
            out.extend(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return [(prefix or "out", tree)]


def compare_trees(reference: Any, candidate: Any, *, rtol: float, atol: float, prefix: str = "",
                  exact: bool = False) -> list[GateCheck]:
    import torch
    ref_items = _flatten(reference, prefix)
    cand_items = _flatten(candidate, prefix)
    checks: list[GateCheck] = []
    if len(ref_items) != len(cand_items):
        checks.append(GateCheck(f"{prefix}/structure", False, float(len(cand_items)), float(len(ref_items)),
                                f"reference has {len(ref_items)} leaves, candidate has {len(cand_items)}"))
        return checks
    for (name, ref), (_, cand) in zip(ref_items, cand_items, strict=True):
        if isinstance(ref, torch.Tensor) and isinstance(cand, torch.Tensor):
            if tuple(ref.shape) != tuple(cand.shape):
                checks.append(GateCheck(f"{name}/shape", False, detail=f"{tuple(cand.shape)} != {tuple(ref.shape)}"))
                continue
            ref_f = ref.detach()
            cand_f = cand.detach().to(ref_f.device)
            if not ref_f.is_floating_point():
                mismatch = (ref_f != cand_f).float().mean().item() if ref_f.numel() else 0.0
                checks.append(GateCheck(f"{name}/exact", mismatch == 0.0, 1.0 - mismatch, 1.0,
                                        f"{mismatch * 100:.3f}% elements differ"))
                continue
            ref_f = ref_f.float()
            cand_f = cand_f.float()
            finite = bool(torch.isfinite(cand_f).all().item())
            if not finite:
                checks.append(GateCheck(f"{name}/finite", False, detail="candidate has NaN/Inf"))
                continue
            if exact:
                equal = bool(torch.equal(ref_f, cand_f))
                diff = (ref_f - cand_f).abs().max().item() if ref_f.numel() else 0.0
                checks.append(GateCheck(f"{name}/bitwise", equal, diff, 0.0, f"max|diff|={diff:.3e}"))
                continue
            diff = (ref_f - cand_f).abs()
            tol = atol + rtol * ref_f.abs()
            bad = (diff > tol).float().mean().item() if ref_f.numel() else 0.0
            max_abs = diff.max().item() if ref_f.numel() else 0.0
            rel = max_abs / (ref_f.abs().max().item() + 1e-12)
            checks.append(GateCheck(f"{name}/allclose", bad == 0.0, rel, rtol,
                                    f"max|diff|={max_abs:.3e} rel={rel:.3e} violating={bad * 100:.4f}% (rtol={rtol:g}, atol={atol:g})"))
        else:
            same = ref == cand
            checks.append(GateCheck(f"{name}/equal", bool(same), detail=f"{type(cand).__name__}"))
    return checks


def snr_db(reference: Any, candidate: Any) -> float:
    import torch
    ref = reference.detach().float()
    cand = candidate.detach().float().to(ref.device)
    noise = (ref - cand).pow(2).mean()
    if noise.item() == 0:
        return float("inf")
    return float(10 * torch.log10(ref.pow(2).mean() / (noise + 1e-20)))

from __future__ import annotations

import importlib

BUILTIN = {
    "custom": ("fastkernel.models.torch_module", "TorchModuleSpec"),
    "torch-module": ("fastkernel.models.torch_module", "TorchModuleSpec"),
    "hf-causal-lm": ("fastkernel.models.hf_causal_lm", "HFCausalLMSpec"),
    "mimi": ("fastkernel.models.mimi", "MimiSpec"),
    "lfm25": ("fastkernel.models.lfm25", "LFM25Spec"),
    "lfm2.5": ("fastkernel.models.lfm25", "LFM25Spec"),
    "lfm-audio": ("fastkernel.models.lfm_audio", "LFMAudioSpec"),
    "lfm2-audio": ("fastkernel.models.lfm_audio", "LFMAudioSpec"),
    "yolo": ("fastkernel.models.yolo", "YOLOSpec"),
}


def get_spec_class(name: str):
    key = name.lower().strip()
    if key not in BUILTIN:
        raise KeyError(f"unknown built-in model spec '{name}'. Known: {', '.join(sorted(BUILTIN))}")
    module_name, class_name = BUILTIN[key]
    return getattr(importlib.import_module(module_name), class_name)


def builtin_names() -> list[str]:
    return sorted(set(BUILTIN))

"""Generic Transformers causal LM spec: prefill + decode workloads with logits/token gates.

model_args:
    hub_id: LiquidAI/LFM2.5-1.2B-Instruct
    dtype: bfloat16            # reference dtype (bf16 is the deployment dtype for these models)
    prefill_tokens: 512
    decode_tokens: 64
    batch: 1
    attn_implementation: sdpa
"""
from __future__ import annotations

from typing import Any

from .spec import GateCheck, ModelSpec, Workload, compare_trees


class HFCausalLMSpec(ModelSpec):
    name = "hf-causal-lm"
    display_name = "Transformers causal LM"
    default_hub_id = "LiquidAI/LFM2.5-1.2B-Instruct"
    prompt = ("Explain, step by step, how a residual vector quantizer turns a waveform into discrete codes, and why "
              "the search over codebooks is sequential. Then list three ways to make it faster on a GPU.")
    # bf16 reference => tolerant tolerances even in strict mode are meaningful; argmax agreement is the real gate.
    default_rtol = {"strict": 2e-2, "tolerant": 6e-2}
    default_atol = {"strict": 2e-2, "tolerant": 8e-2}
    top1_threshold = {"strict": 0.995, "tolerant": 0.97}
    greedy_match_threshold = {"strict": 1.0, "tolerant": 0.9}

    @property
    def hub_id(self) -> str:  # type: ignore[override]
        return str(self.args.get("hub_id") or self.default_hub_id)

    def _dtype(self):
        import torch
        return getattr(torch, str(self.args.get("dtype", "bfloat16")))

    def load_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.hub_id)

    def load_reference(self) -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        kwargs: dict[str, Any] = {"dtype": self._dtype()}
        if self.args.get("attn_implementation"):
            kwargs["attn_implementation"] = self.args["attn_implementation"]
        try:
            model = AutoModelForCausalLM.from_pretrained(self.hub_id, **kwargs)
        except TypeError:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(self.hub_id, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        return model

    # ---- inputs ------------------------------------------------------------------------
    def _prompt_ids(self, device, n_tokens: int, batch: int, seed: int):
        import torch
        tok = self.load_tokenizer()
        try:
            text = tok.apply_chat_template([{"role": "user", "content": self.prompt}], add_generation_prompt=True, tokenize=False)
        except Exception:  # noqa: BLE001
            text = self.prompt
        ids = tok(text, return_tensors="pt").input_ids[0]
        if ids.numel() < n_tokens:
            g = torch.Generator(device="cpu").manual_seed(seed)
            vocab = int(getattr(tok, "vocab_size", 32000) or 32000)
            filler = torch.randint(0, vocab, (n_tokens - ids.numel(),), generator=g)
            ids = torch.cat([ids[:-1], filler, ids[-1:]]) if ids.numel() else filler
        ids = ids[-n_tokens:].unsqueeze(0).repeat(batch, 1)
        return {"input_ids": ids.to(device), "attention_mask": torch.ones_like(ids).to(device)}

    def _make_prefill(self, n_tokens: int, batch: int):
        def make(device, seed):
            return self._prompt_ids(device, n_tokens, batch, seed)
        return make

    @staticmethod
    def _run_prefill(model, inputs):
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        return out.logits[:, -8:, :].float()

    def _make_decode(self, n_prompt: int, n_new: int, batch: int):
        def make(device, seed):
            inputs = self._prompt_ids(device, n_prompt, batch, seed)
            inputs["max_new_tokens"] = n_new
            return inputs
        return make

    @staticmethod
    def _run_decode(model, inputs):
        import torch
        with torch.inference_mode():
            out = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                 max_new_tokens=inputs["max_new_tokens"], min_new_tokens=inputs["max_new_tokens"],
                                 do_sample=False, num_beams=1, pad_token_id=model.config.eos_token_id
                                 if isinstance(model.config.eos_token_id, int) else None)
        return out[:, inputs["input_ids"].shape[1]:]

    def workloads(self) -> list[Workload]:
        n_prefill = int(self.args.get("prefill_tokens", 512))
        n_new = int(self.args.get("decode_tokens", 64))
        batch = int(self.args.get("batch", 1))
        n_prompt = int(self.args.get("decode_prompt_tokens", 64))
        return [
            Workload("decode", self._make_decode(n_prompt, n_new, batch), self._run_decode, primary=True, tags=("decode",),
                     describe=f"greedy generation of {n_new} tokens after a {n_prompt}-token prompt (batch {batch})",
                     units={"tokens": float(n_new * batch)}),
            Workload("prefill", self._make_prefill(n_prefill, batch), self._run_prefill, tags=("prefill",),
                     describe=f"prefill logits for {n_prefill} tokens (batch {batch})", units={"tokens": float(n_prefill * batch)}),
        ]

    def edge_workloads(self) -> list[Workload]:
        return [
            Workload("edge_prefill_odd", self._make_prefill(37, 1), self._run_prefill, tags=("edge",), bench=False,
                     describe="odd prompt length"),
            Workload("edge_decode_short", self._make_decode(5, 4, 1), self._run_decode, tags=("edge",), bench=False,
                     describe="tiny prompt, 4 new tokens"),
        ]

    def compare(self, workload: Workload, reference: Any, candidate: Any) -> list[GateCheck]:
        import torch
        key = self.policy.precision if self.policy.precision in self.top1_threshold else "strict"
        if "decode" in workload.tags or workload.name.startswith("edge_decode"):
            ref_ids, cand_ids = reference, candidate
            if tuple(ref_ids.shape) != tuple(cand_ids.shape):
                return [GateCheck(f"{workload.name}/shape", False, detail=f"{tuple(cand_ids.shape)} != {tuple(ref_ids.shape)}")]
            match = (ref_ids == cand_ids.to(ref_ids.device)).float().mean().item()
            # tokens after the first divergence are meaningless; report prefix match too
            eq = (ref_ids == cand_ids.to(ref_ids.device)).all(0)
            prefix = int(eq.cumprod(0).sum().item())
            thr = self.greedy_match_threshold[key]
            return [GateCheck(f"{workload.name}/greedy_tokens_match", match >= thr, match, thr,
                              f"{match * 100:.1f}% tokens identical, identical prefix {prefix}/{ref_ids.shape[1]}")]
        checks = compare_trees(reference, candidate, rtol=self.tolerances(workload)[0], atol=self.tolerances(workload)[1],
                               prefix=workload.name)
        # allclose on bf16 logits is noisy; the decisive gate is top-1 agreement (+ top-5 overlap)
        ref_top = reference.argmax(-1)
        cand_top = candidate.to(reference.device).argmax(-1)
        top1 = (ref_top == cand_top).float().mean().item()
        thr = self.top1_threshold[key]
        ref5 = reference.topk(5, dim=-1).indices
        cand5 = candidate.to(reference.device).topk(5, dim=-1).indices
        overlap = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / 5 for a, b in zip(ref5.flatten(0, 1), cand5.flatten(0, 1), strict=True)]).mean().item()
        checks = [c for c in checks if not c.name.endswith("/allclose")] + [
            GateCheck(f"{workload.name}/top1_agreement", top1 >= thr, top1, thr, f"top-1 agreement {top1 * 100:.2f}%"),
            GateCheck(f"{workload.name}/top5_overlap", overlap >= 0.9, overlap, 0.9, f"mean top-5 overlap {overlap:.3f}"),
            GateCheck(f"{workload.name}/logits_rel_err", True, *(_rel_err(reference, candidate)), "informational"),
        ]
        return checks

    def hotspot_hints(self) -> list[dict[str, Any]]:
        return [
            {"symbol": "Lfm2ShortConv", "category": "conv", "note": "in_proj (3x) -> chunk B,C,x -> B*x -> causal depthwise conv1d (L_cache) -> C*y -> out_proj: fuse gating+conv; cache path for decode"},
            {"symbol": "Lfm2Attention", "category": "attention", "note": "GQA with q/k RMSNorm; decode is GEMV shaped -> launch bound"},
            {"symbol": "Lfm2MLP", "category": "gemm", "note": "w2(silu(w1 x) * w3 x): merge w1/w3, fuse silu*mul epilogue"},
            {"symbol": "Lfm2RMSNorm", "category": "norm", "note": "fuse with the following projection input cast / residual add"},
        ]


def _rel_err(reference, candidate) -> tuple[float, float]:
    ref = reference.float()
    cand = candidate.to(reference.device).float()
    err = ((ref - cand).abs().max() / (ref.abs().max() + 1e-12)).item()
    return err, 1.0

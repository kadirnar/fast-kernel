"""LiquidAI LFM2-Audio / LFM2.5-Audio (liquid_audio on top of transformers' Lfm2 backbone).

Transformers 5.x has no native Lfm2AudioForConditionalGeneration; the `liquid-audio` package provides
LFM2AudioModel (an nn.Module built from the Transformers Lfm2 backbone, a FastConformer audio encoder
and a Mimi-compatible 8-codebook RQ decoder). Workloads: TTS (text -> audio codes -> waveform) and ASR
(audio -> text tokens), both greedy/seeded for reproducibility.

model_args: hub_id (default LiquidAI/LFM2.5-Audio-1.5B), tts_text, tts_max_new_tokens, asr_seconds, asr_audio (wav path)
"""
from __future__ import annotations

import inspect
import math
from typing import Any

from .spec import GateCheck, ModelSpec, Workload, snr_db

SR_IN = 16_000
SR_OUT = 24_000


class LFMAudioSpec(ModelSpec):
    name = "lfm-audio"
    display_name = "LiquidAI LFM2-Audio (liquid_audio + transformers Lfm2)"
    default_hub_id = "LiquidAI/LFM2.5-Audio-1.5B"
    notes = """\
LFM2-Audio-1.5B = LFM2(.5) 1.2B hybrid conv/attention backbone + FastConformer audio encoder (115M, 17 layers,
d=512, mel 128 @ 16 kHz) + Mimi-compatible audio decoder (8 codebooks) driven by a 6-layer depthformer
(RQ-transformer, d=1024). Generation is autoregressive over interleaved text/audio tokens: every step runs
the backbone once (GEMV shaped, launch bound) and the depthformer 8x for the codebooks, then Mimi decodes
the codes. Hot paths: the per-step backbone (static-cache CUDA graph / reduce-overhead compile), the
depthformer loop (fuse the 8 codebook steps), the FastConformer encoder for ASR (conv subsampling +
relative-position attention), and the Mimi decoder (see the Mimi example).
"""
    greedy_match_threshold = {"strict": 1.0, "tolerant": 0.9}

    def __init__(self, campaign_root, args=None, policy=None):
        super().__init__(campaign_root, args, policy)
        self.processor = None

    @property
    def hub_id(self) -> str:  # type: ignore[override]
        return str(self.args.get("hub_id") or self.default_hub_id)

    def load_reference(self) -> Any:
        import torch
        try:
            from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
        except ImportError as exc:
            raise RuntimeError("liquid_audio is required: uv pip install liquid-audio") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = LFM2AudioProcessor.from_pretrained(self.hub_id)
        try:
            self.processor = self.processor.to(device)
        except Exception:  # noqa: BLE001
            pass
        model = LFM2AudioModel.from_pretrained(self.hub_id)
        model = model.to(device).eval()
        return model

    def hooks_root(self, model: Any) -> Any:
        return model

    # ---- inputs ------------------------------------------------------------------------
    def _chat(self, role_texts: list[tuple[str, str]], audio=None, sr: int | None = None):
        from liquid_audio import ChatState
        chat = ChatState(self.processor)
        for role, text in role_texts:
            chat.new_turn(role)
            chat.add_text(text)
            chat.end_turn()
        if audio is not None:
            chat.new_turn("user")
            chat.add_audio(audio, sr)
            chat.end_turn()
        chat.new_turn("assistant")
        return chat

    def _gen_kwargs(self, model, max_new_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        try:
            params = inspect.signature(model.generate_sequential).parameters
        except (TypeError, ValueError):
            params = {}
        if "audio_temperature" in params:
            kwargs["audio_temperature"] = 1e-4     # greedy-equivalent sampling
        if "audio_top_k" in params:
            kwargs["audio_top_k"] = 1
        if "text_temperature" in params:
            kwargs["text_temperature"] = 1e-4
        if "text_top_k" in params:
            kwargs["text_top_k"] = 1
        if "do_sample" in params:
            kwargs["do_sample"] = False
        return kwargs

    def _make_tts(self):
        def make(device, seed):
            text = str(self.args.get("tts_text", "The quick brown fox jumps over the lazy dog near the river bank."))
            return {"text": text, "seed": seed, "max_new_tokens": int(self.args.get("tts_max_new_tokens", 96))}
        return make

    def _run_tts(self, model, inputs):
        import torch
        torch.manual_seed(inputs["seed"])
        chat = self._chat([("system", "Perform TTS. Use the UK male voice."), ("user", inputs["text"])])
        tokens = list(model.generate_sequential(**chat, **self._gen_kwargs(model, inputs["max_new_tokens"])))
        audio_tokens = [t for t in tokens if t.numel() > 1]
        text_tokens = [t.reshape(-1) for t in tokens if t.numel() == 1]
        out: dict[str, Any] = {}
        if len(audio_tokens) > 1:
            codes = torch.stack(audio_tokens[:-1], 1).unsqueeze(0)
            out["codes"] = codes
            out["audio"] = self.processor.decode(codes).float()
        elif audio_tokens:
            out["codes"] = torch.stack(audio_tokens, 1).unsqueeze(0)
        if text_tokens:
            out["text_tokens"] = torch.cat(text_tokens)
        return out

    @staticmethod
    def synthetic_speech(seconds: float, seed: int):
        import torch
        n = int(SR_IN * seconds)
        t = torch.arange(n) / SR_IN
        g = torch.Generator(device="cpu").manual_seed(seed)
        f0 = 120 + 30 * torch.sin(2 * math.pi * 0.7 * t)
        phase = torch.cumsum(2 * math.pi * f0 / SR_IN, 0)
        wave = torch.zeros(n)
        for k, amp in enumerate([1.0, 0.6, 0.4, 0.25, 0.15, 0.1], start=1):
            wave += amp * torch.sin(k * phase)
        env = (0.5 + 0.5 * torch.sin(2 * math.pi * 2.5 * t)).clamp(min=0.05)
        wave = wave * env / wave.abs().max()
        wave += 0.01 * torch.randn(n, generator=g)
        return wave

    def _make_asr(self):
        def make(device, seed):
            seconds = float(self.args.get("asr_seconds", 2.0))
            path = self.args.get("asr_audio")
            if path:
                wav, sr = _load_wav(path)
            else:
                wav, sr = self.synthetic_speech(seconds, seed), SR_IN
            return {"audio": wav, "sr": sr, "seed": seed, "max_new_tokens": int(self.args.get("asr_max_new_tokens", 64))}
        return make

    def _run_asr(self, model, inputs):
        import torch
        torch.manual_seed(inputs["seed"])
        chat = self._chat([("system", "Perform ASR.")], audio=inputs["audio"], sr=inputs["sr"])
        tokens = [t.reshape(-1) for t in model.generate_sequential(**chat, **self._gen_kwargs(model, inputs["max_new_tokens"])) if t.numel() == 1]
        return {"text_tokens": torch.cat(tokens) if tokens else torch.zeros(0, dtype=torch.long)}

    def workloads(self) -> list[Workload]:
        n_tts = int(self.args.get("tts_max_new_tokens", 96))
        return [
            Workload("tts", self._make_tts(), self._run_tts, primary=True, describe=f"text -> up to {n_tts} audio frames -> waveform",
                     units={"tokens": float(n_tts)}),
            Workload("asr", self._make_asr(), self._run_asr, describe="speech -> text tokens (greedy)",
                     units={"audio_seconds": float(self.args.get("asr_seconds", 2.0))}),
        ]

    def derived_metrics(self, workload: Workload, latency_ms: float) -> dict[str, float]:
        out = super().derived_metrics(workload, latency_ms)
        if workload.name == "tts" and latency_ms > 0:
            frames = float(workload.units.get("tokens", 0))
            if frames:
                out["audio_seconds_generated"] = frames / 12.5
                out["rtf"] = (latency_ms / 1000.0) / (frames / 12.5)
        return out

    def compare(self, workload: Workload, reference: Any, candidate: Any) -> list[GateCheck]:
        key = self.policy.precision if self.policy.precision in self.greedy_match_threshold else "strict"
        thr = self.greedy_match_threshold[key]
        checks: list[GateCheck] = []
        for name in ("text_tokens", "codes"):
            if name in reference:
                ref, cand = reference[name], candidate.get(name)
                if cand is None:
                    checks.append(GateCheck(f"{workload.name}/{name}", False, detail="missing in candidate output"))
                    continue
                cand = cand.to(ref.device)
                n = min(ref.numel(), cand.numel())
                if ref.shape != cand.shape:
                    detail = f"length {tuple(cand.shape)} vs {tuple(ref.shape)}"
                else:
                    detail = ""
                match = (ref.reshape(-1)[:n] == cand.reshape(-1)[:n]).float().mean().item() if n else 0.0
                same_len = ref.shape == cand.shape
                checks.append(GateCheck(f"{workload.name}/{name}_match", match >= thr and (same_len or key == "tolerant"), match, thr,
                                        f"{match * 100:.1f}% of {n} compared tokens identical {detail}".strip()))
        if "audio" in reference and "audio" in candidate:
            ref_audio, cand_audio = reference["audio"], candidate["audio"].to(reference["audio"].device)
            n = min(ref_audio.shape[-1], cand_audio.shape[-1])
            snr = snr_db(ref_audio[..., :n], cand_audio[..., :n]) if n else 0.0
            need = 40.0 if key == "strict" else 20.0
            checks.append(GateCheck(f"{workload.name}/audio_snr", snr >= need, snr, need, f"{snr:.1f} dB over {n} samples"))
        return checks or [GateCheck(f"{workload.name}/outputs", False, detail="no comparable outputs")]

    def hotspot_hints(self) -> list[dict[str, Any]]:
        return [
            {"symbol": "Lfm2ShortConv", "category": "conv", "note": "gated causal depthwise conv; fuse gating; cached decode step"},
            {"symbol": "Lfm2Attention", "category": "attention", "note": "GQA + q/k norm; decode is GEMV shaped"},
            {"symbol": "Lfm2MLP", "category": "gemm", "note": "SwiGLU: merge w1/w3, fuse silu*mul"},
            {"symbol": "Lfm2RMSNorm", "category": "norm", "note": "fuse into next projection"},
            {"symbol": "ConformerEncoder", "category": "attention", "note": "FastConformer: conv subsampling + rel-pos attention + depthwise conv modules"},
            {"symbol": "MimiEuclideanCodebook", "category": "quantizer", "note": "Mimi decoder path: see the Mimi example"},
        ]


def _load_wav(path: str):
    try:
        import soundfile as sf
        import torch
        data, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(data)
        return (wav.mean(-1) if wav.dim() > 1 else wav), int(sr)
    except ImportError:
        import torchaudio
        wav, sr = torchaudio.load(path)
        return wav.mean(0), int(sr)

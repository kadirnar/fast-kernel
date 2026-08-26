"""Kyutai Mimi neural audio codec (transformers.MimiModel): encode / decode / roundtrip workloads.

Gates (strict): discrete codes identical to the fp32 oracle, decoded waveform allclose.
Gates (tolerant): >= 85 % code match, decode(reference codes) SNR >= 35 dB, reconstruction SNR vs the
input within 0.5 dB of the reference (what a bf16 tensor-core implementation achieves).
"""
from __future__ import annotations

import math
from typing import Any

from .spec import GateCheck, ModelSpec, Workload, compare_trees, snr_db

SR = 24_000


class MimiSpec(ModelSpec):
    name = "mimi"
    display_name = "Kyutai Mimi codec (transformers MimiModel)"
    hub_id = "kyutai/mimi"
    default_rtol = {"strict": 2e-4, "tolerant": 5e-2}
    default_atol = {"strict": 2e-5, "tolerant": 5e-2}
    notes = """\
Mimi = SEANet conv encoder (causal Conv1d + ELU residual blocks, strided downsampling) -> 8-layer causal
transformer (RoPE, sliding window, LayerScale) -> downsample -> 32-stage residual vector quantizer (one
semantic + 31 acoustic codebooks, each a 2048x256 Euclidean codebook search) -> transformer -> SEANet
decoder with ConvTranspose1d upsampling. 12.5 Hz frames, 24 kHz audio, batch 1 is the deployment case.

Measured facts on an RTX 5070 Ti (fast-mimi, 2026-08): the stock transformers path launches ~1250
kernels for 1 s of audio (~19 ms round trip) while the GPU is busy only a few hundred microseconds ->
overwhelmingly launch/overhead bound. A rewrite into ~100 fused Triton kernels captured in CUDA graphs
(bf16 tensor-core GEMMs, fp32 residual stream, exact two-stage RVQ search: fp16 coarse distances on tensor
cores + exact fp32 re-rank of the top-2, grid barrier between the 32 sequential codebook stages) reached
~0.8 ms (~24x). Ideas that did NOT pay off there (re-measure before assuming): persistent single-launch
transformer (2x slower than 4 kernels), split-K one-tile-per-CTA transformer, INT8 weights (latency-bound,
no gain), tf32x3 exact RVQ (slower than fp16-coarse + fp32 re-rank), fused conv0+first residual block.
Ordering that worked: CUDA graphs on the stock model first (biggest single win), then fuse the RVQ
search, then the transformer blocks (LN+QKV+RoPE, attention+O+residual, LN+FC1+GELU, FC2), then the
SEANet convs as implicit GEMMs with fused ELU epilogues.

Graph-capture gotchas in transformers' Mimi: `MimiConv1d` keeps kernel_size/stride/padding_total as
int64 *buffers on the GPU* and computes `extra_padding` as a CUDA scalar that `F.pad` converts with
`.item()` -> a host sync per conv layer (illegal under CUDA-graph capture, wasteful in eager mode);
`padding_left/padding_right` are derived from those buffers at init and stay *meta* tensors after
from_pretrained (never call int() on them -- recompute from the int padding_total). Replace the
padding arithmetic with Python ints before capturing.
"""

    def __init__(self, campaign_root, args=None, policy=None):
        super().__init__(campaign_root, args, policy)
        self.reference_model = None
        self._codes_cache: dict[tuple, Any] = {}
        self.seconds = float(self.args.get("seconds", 1.0))
        self.sweep = [float(s) for s in (self.args.get("sweep") or [0.25, 5.0])]

    # ---- loading -----------------------------------------------------------------------
    def load_reference(self) -> Any:
        import torch
        from transformers import MimiModel
        kwargs: dict[str, Any] = {}
        if self.args.get("attn_implementation"):
            kwargs["attn_implementation"] = self.args["attn_implementation"]
        try:
            model = MimiModel.from_pretrained(self.hub_id, dtype=torch.float32, **kwargs)
        except TypeError:
            model = MimiModel.from_pretrained(self.hub_id, torch_dtype=torch.float32, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        self.reference_model = model
        return model

    # ---- inputs ------------------------------------------------------------------------
    @staticmethod
    def signal(seconds: float, seed: int, batch: int = 1, kind: str = "mix", samples: int | None = None):
        import torch
        n = samples or int(SR * seconds)
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((batch, 1, n), generator=g)
        t = torch.arange(n) / SR
        sweep = (0.5 * torch.sin(2 * math.pi * (200 + 3000 * t / max(seconds, 1e-6)) * t) * torch.exp(-t)).view(1, 1, n)
        if kind == "noise":
            return noise
        if kind == "sweep":
            return sweep.repeat(batch, 1, 1)
        return 0.1 * noise + sweep

    def _make_audio(self, seconds: float, kind: str = "mix", batch: int = 1, samples: int | None = None):
        def make(device, seed):
            import torch
            x = self.signal(seconds, seed, batch, kind, samples).to(device)
            return {"audio": x, "mask": torch.ones_like(x, dtype=torch.bool)}
        return make

    def _make_codes(self, seconds: float, kind: str = "mix"):
        def make(device, seed):
            import torch
            key = (seconds, kind, seed)
            if key not in self._codes_cache:
                x = self.signal(seconds, seed, 1, kind).to(device)
                mask = torch.ones_like(x, dtype=torch.bool)
                model = self.reference_model
                if model is None:
                    model = self.load_reference()
                with torch.inference_mode():
                    self._codes_cache[key] = model.encode(x, mask).audio_codes.clone()
            codes = self._codes_cache[key]
            return {"codes": codes, "mask": torch.ones((1, 1, int(SR * seconds)), dtype=torch.bool, device=device)}
        return make

    @staticmethod
    def run_encode(model, inputs):
        return model.encode(inputs["audio"], inputs["mask"]).audio_codes

    @staticmethod
    def run_decode(model, inputs):
        return model.decode(inputs["codes"], inputs["mask"]).audio_values

    @staticmethod
    def run_roundtrip(model, inputs):
        codes = model.encode(inputs["audio"], inputs["mask"]).audio_codes
        audio = model.decode(codes, inputs["mask"]).audio_values
        return {"codes": codes, "audio": audio}

    def workloads(self) -> list[Workload]:
        s = self.seconds
        items = [
            Workload(f"roundtrip_{s:g}s", self._make_audio(s), self.run_roundtrip, primary=True,
                     describe=f"encode + decode of {s:g} s of 24 kHz audio, batch 1", units={"audio_seconds": s}),
            Workload(f"encode_{s:g}s", self._make_audio(s), self.run_encode, describe=f"encode {s:g} s", units={"audio_seconds": s}),
            Workload(f"decode_{s:g}s", self._make_codes(s), self.run_decode, describe=f"decode {s:g} s of reference codes",
                     units={"audio_seconds": s}),
        ]
        for sec in self.sweep:
            items.append(Workload(f"roundtrip_{sec:g}s", self._make_audio(sec), self.run_roundtrip, tags=("sweep",),
                                  describe=f"round trip {sec:g} s", units={"audio_seconds": sec}))
        items.append(Workload(f"roundtrip_noise_{s:g}s", self._make_audio(s, "noise"), self.run_roundtrip, tags=("sweep",),
                              bench=False, describe="white-noise input (stresses deep codebooks)", units={"audio_seconds": s}))
        return items

    def edge_workloads(self) -> list[Workload]:
        return [
            Workload("edge_short_0.05s", self._make_audio(0.05), self.run_roundtrip, tags=("edge",), bench=False,
                     describe="50 ms input (single frame)"),
            Workload("edge_odd_length", self._make_audio(1.0, samples=SR + 123), self.run_roundtrip, tags=("edge",), bench=False,
                     describe="non-multiple-of-frame length"),
            Workload("edge_batch2", self._make_audio(0.5, batch=2), self.run_roundtrip, tags=("edge",), bench=False,
                     describe="batch of 2"),
        ]

    # ---- gates -------------------------------------------------------------------------
    def compare(self, workload: Workload, reference: Any, candidate: Any) -> list[GateCheck]:
        import torch
        strict = self.policy.precision == "strict"
        rtol, atol = self.tolerances(workload)
        checks: list[GateCheck] = []
        if isinstance(reference, dict):  # roundtrip
            ref_codes, cand_codes = reference["codes"], candidate["codes"].to(reference["codes"].device)
            if tuple(ref_codes.shape) != tuple(cand_codes.shape):
                return [GateCheck(f"{workload.name}/codes_shape", False, detail=f"{tuple(cand_codes.shape)} != {tuple(ref_codes.shape)}")]
            match = (ref_codes == cand_codes).float().mean().item()
            per_cb = (ref_codes == cand_codes).float().mean(dim=(0, 2))
            thr = 1.0 if strict else 0.85
            checks.append(GateCheck(f"{workload.name}/code_match", match >= thr, match, thr,
                                    f"{match * 100:.2f}% codes identical; worst codebook {per_cb.min().item() * 100:.1f}%"))
            ref_audio, cand_audio = reference["audio"], candidate["audio"].to(reference["audio"].device)
            if tuple(ref_audio.shape) != tuple(cand_audio.shape):
                checks.append(GateCheck(f"{workload.name}/audio_shape", False, detail=f"{tuple(cand_audio.shape)} != {tuple(ref_audio.shape)}"))
                return checks
            if not torch.isfinite(cand_audio).all():
                checks.append(GateCheck(f"{workload.name}/audio_finite", False, detail="NaN/Inf in decoded audio"))
                return checks
            if strict:
                checks += compare_trees(ref_audio, cand_audio, rtol=rtol, atol=atol, prefix=f"{workload.name}/audio")
            else:
                # different codes => different waveform; judge reconstruction quality vs the input instead
                x = self._last_inputs.get(workload.name)
                if x is not None:
                    length = min(x.shape[-1], ref_audio.shape[-1])
                    ref_rec = snr_db(x[..., :length], ref_audio[..., :length])
                    cand_rec = snr_db(x[..., :length], cand_audio[..., :length])
                    checks.append(GateCheck(f"{workload.name}/reconstruction_snr", cand_rec >= ref_rec - 0.5, cand_rec, ref_rec - 0.5,
                                            f"reference {ref_rec:.2f} dB vs candidate {cand_rec:.2f} dB (vs input)"))
                snr = snr_db(ref_audio, cand_audio)
                checks.append(GateCheck(f"{workload.name}/audio_snr_own_codes", True, snr, None, f"{snr:.1f} dB (informational)"))
            return checks
        if reference.dtype in (torch.int64, torch.int32):  # encode
            cand = candidate.to(reference.device)
            if tuple(reference.shape) != tuple(cand.shape):
                return [GateCheck(f"{workload.name}/codes_shape", False, detail=f"{tuple(cand.shape)} != {tuple(reference.shape)}")]
            match = (reference == cand).float().mean().item()
            thr = 1.0 if strict else 0.85
            return [GateCheck(f"{workload.name}/code_match", match >= thr, match, thr, f"{match * 100:.2f}% codes identical")]
        # decode of reference codes: waveform must match closely in both policies
        cand = candidate.to(reference.device)
        if tuple(reference.shape) != tuple(cand.shape):
            return [GateCheck(f"{workload.name}/audio_shape", False, detail=f"{tuple(cand.shape)} != {tuple(reference.shape)}")]
        if strict:
            return compare_trees(reference, cand, rtol=rtol, atol=atol, prefix=f"{workload.name}/audio")
        snr = snr_db(reference, cand)
        return [GateCheck(f"{workload.name}/audio_snr", snr >= 35.0, snr, 35.0, f"{snr:.1f} dB vs fp32 reference")]

    # the harness calls this before compare() so tolerant mode can measure reconstruction vs the input
    _last_inputs: dict[str, Any] = {}

    def observe_inputs(self, workload: Workload, inputs: dict[str, Any]) -> None:
        if "audio" in inputs:
            self._last_inputs[workload.name] = inputs["audio"]

    def hotspot_hints(self) -> list[dict[str, Any]]:
        return [
            {"symbol": "MimiEuclideanCodebook", "category": "quantizer", "note": "cdist+argmin per codebook, 32 sequential stages: fuse distance+argmin, keep residual on chip, one kernel for all stages"},
            {"symbol": "MimiVectorQuantization", "category": "quantizer", "note": "projection + codebook lookup around each stage"},
            {"symbol": "MimiSplitResidualVectorQuantizer", "category": "sequential", "note": "semantic (1) + acoustic (31) RVQ chain"},
            {"symbol": "MimiAttention", "category": "attention", "note": "8 layers, causal sliding window, RoPE; T<=13 frames for 1 s -> launch bound; fuse LN+QKV+RoPE, attn+O+residual"},
            {"symbol": "MimiMLP", "category": "gemm", "note": "fc1+GELU+fc2; fuse GELU into fc1 epilogue, split-K fc2"},
            {"symbol": "MimiConv1d", "category": "conv", "note": "causal padding + conv1d; implicit GEMM with fused ELU; channels-last"},
            {"symbol": "MimiConvTranspose1d", "category": "conv", "note": "decoder upsampling; implicit GEMM"},
            {"symbol": "MimiResnetBlock", "category": "conv", "note": "ELU->conv(k=3)->ELU->conv(k=1) + residual: fuse the chain"},
            {"symbol": "MimiLayerScale", "category": "elementwise", "note": "per-channel scale + residual add: fold into the previous kernel's epilogue"},
        ]

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.11",
#   "torchvision>=0.26",
#   "transformers==5.10.2",
#   "accelerate>=1.0",
#   "safetensors>=0.4",
#   "huggingface_hub>=1.0",
#   "numpy",
#   "pillow",
#   "av",
#   "fastapi",
#   "uvicorn[standard]",
#   "auto-round==0.16.0",
# ]
# ///
"""Serve Clef-Omni (Cloudflare) or one of its quantizations as a SystemOne decision API.

    uv run serve_clef.py --model dbirks/clef-omni-nvfp4            # HTTP server on :8000
    uv run serve_clef.py --model dbirks/clef-omni-int4 --demo      # one demo decision, then exit
    uv run serve_clef.py --model Cloudflare/clef-omni --request req.json

Endpoints: POST /v1/systemone ({"model", "state", "questions", optional "images"/"audio"/"videos"}),
GET /healthz (200 once the model is loaded). Answers come from Cloudflare's own `systemone()` in the
repo's joint_schema_model.py: one forward pass, per-option probabilities from the BF16 joint head.

Checkpoint kinds, detected from the repo's config:
  * BF16 original (Cloudflare/clef-omni): Cloudflare's load_release_model, unchanged.
  * AutoRound INT4 (quant_method auto-round): transformers loads it with real int4 kernels.
  * ModelOpt NVFP4 / NVFP4A16 (hf_quant_config.json) and AutoRound NVFP4 / NVFP4A16 (auto-round,
    data_type nv_fp): transformers cannot load packed NVFP4 (AutoRound has no backend for its own
    weight-only nv_fp), so the FP4 experts stay PACKED in VRAM (~22 GB for Clef) and each MoE layer is
    dequantized just in time to BF16 (bit-exact quantized values) for transformers' grouped_mm, then freed;
    --no-packed-nvfp4 dequantizes everything on load instead (~60 GB, BF16 speed). For W4A4 checkpoints,
    --simulate-fp4-activations also rounds the expert inputs to NVFP4 exactly as the quantizer calibrated
    them (per-16 block scales in FP8 E4M3 under the exported per-tensor input scale), reproducing the
    quantized model's answers; it is slow, so it is off by default (and it uses the up-front dequant).
    Native FP4 execution needs an engine such as vLLM, plus a hidden-state path into the head.
  * FP8 block (quant_method fp8, weight_block_size [128, 128], Qwen/DeepSeek layout from
    quant_fp8_block.py): experts stay FP8 + fp32 per-tile scales in VRAM (~32 GiB for Clef) and are
    dequantized per layer like NVFP4 (W8A16); --no-packed-fp8 dequantizes on load instead.
    --simulate-fp8-activations adds the dynamic 1x128 per-token-group FP8 rounding of expert inputs (W8A8).
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
E2M1_PAIRS = torch.stack((E2M1[torch.arange(256) & 15], E2M1[torch.arange(256) >> 4]), dim=-1)   # byte -> (lo, hi)


def log(*a):
    print(f"[{time.strftime('%FT%T')}]", *a, file=sys.stderr, flush=True)


def resolve(model, revision):
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, revision=revision))


def kind_of(path):
    if (path / "hf_quant_config.json").exists():
        algo = json.loads((path / "hf_quant_config.json").read_text())["quantization"]["quant_algo"]
        return f"modelopt:{algo}"
    qc = json.loads((path / "config.json").read_text()).get("quantization_config") or {}
    method = str(qc.get("quant_method", "")).replace("_", "-")
    if method == "fp8":
        block = qc.get("weight_block_size")
        if not block or len(block) != 2 or qc.get("fmt", "e4m3") != "e4m3":
            raise SystemExit(f"unsupported fp8 checkpoint (want e4m3 with weight_block_size): {qc}")
        return "fp8-block"
    if method == "auto-round":
        if "nv_fp" in str(qc.get("data_type", "")):
            if qc.get("bits") != 4 or qc.get("group_size") != 16:
                raise SystemExit(f"unsupported AutoRound nv_fp scheme: bits={qc.get('bits')} group_size={qc.get('group_size')}")
            return "auto-round:nvfp4" if (qc.get("act_bits") or 16) <= 4 else "auto-round:nvfp4a16"
        return f"auto-round:{qc.get('bits', '?')}bit"
    if method:
        raise SystemExit(f"unsupported quantization_config.quant_method={method!r}")
    return "bf16"


def unpack_e2m1(packed):
    """Two E2M1 codes per byte, low nibble first, bit 3 = sign (ModelOpt and AutoRound/llm-compressor alike)."""
    return E2M1_PAIRS.to(packed.device)[packed.int()].reshape(*packed.shape[:-1], -1)   # one gather per byte


def recip(t):
    return torch.where(t == 0, torch.zeros_like(t), 1.0 / t)


def dequant_nvfp4(packed, scale, scale_2, block=16):
    """ModelOpt NVFP4: e4m3 scale per 16, fp32 global scale is a multiplier."""
    vals = unpack_e2m1(packed)
    vals = vals.view(*vals.shape[:-1], -1, block) * (scale.float() * scale_2.float()).unsqueeze(-1)
    return vals.reshape(*packed.shape[:-1], -1).to(torch.bfloat16)


def dequant_nvfp4_ar(packed, scale, global_scale, block=16):
    """AutoRound NVFP4 (llm-compressor packing): the fp32 weight_global_scale is a DIVISOR. Same op order as
    AutoRound's own NVFP4QuantLinear._dequant_nvfp4_tensor: (fp4 * 1/global_scale) * e4m3 scale, in fp32."""
    if scale.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"expected FP8 E4M3 weight_scale, got {scale.dtype}")
    vals = unpack_e2m1(packed) * recip(global_scale.float())
    vals = vals.view(*vals.shape[:-1], -1, block) * scale.float().unsqueeze(-1)
    return vals.reshape(*packed.shape[:-1], -1).to(torch.bfloat16)


def dequant_fp8_block(w, scale_inv):
    """FP8 block checkpoints: e4m3 times the fp32 per-tile weight_scale_inv, tile size from the scale grid.
    Same math as transformers' Fp8Dequantize._dequantize_one (fp32 product, then BF16). Leading dims allowed."""
    *lead, rows, cols = w.shape
    sr, sc = scale_inv.shape[-2:]
    if rows % sr or cols % sc:
        raise RuntimeError(f"fp8 weight {tuple(w.shape)} does not tile by scale grid {tuple(scale_inv.shape)}")
    q = w.float().reshape(*lead, sr, rows // sr, sc, cols // sc)
    return (q * scale_inv.float()[..., :, None, :, None]).reshape(w.shape).to(torch.bfloat16)


def fp8_act_quant(x, group=128):
    """Dynamic per-token-group (1 x 128) FP8 E4M3 activation rounding, as the W8A8 block-FP8 kernels do
    (scale = group amax / 448, vLLM per_token_group_quant_fp8 / DeepSeek act_quant); returns BF16."""
    shape, dtype = x.shape, x.dtype
    xf = x.float().reshape(-1, shape[-1] // group, group)
    scale = xf.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448.0
    return ((xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale).reshape(shape).to(dtype)


def round_fp8_expert_inputs(experts):
    """Round both expert GEMM inputs (permuted hidden states, gated intermediate) to FP8 per 1x128 token group.
    Dynamic scales are per row, so rounding before the permutation equals rounding after it; works on the
    packed modules and on transformers' grouped_mm experts (both call self._apply_gate)."""
    inner, gate = experts.forward, experts._apply_gate
    experts._apply_gate = lambda gate_up: fp8_act_quant(gate(gate_up))
    experts.forward = lambda h, top_k_index, top_k_weights: inner(fp8_act_quant(h), top_k_index, top_k_weights)


def round_e2m1(a):
    """|x| to the nearest E2M1 magnitude, ties to even (same as AutoRound's cast_to_fp4)."""
    return torch.where(a <= 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a <= 1.25, 1.0, torch.where(
        a < 1.75, 1.5, torch.where(a <= 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)))))))


def fp4_fake_quant(x, global_scale, block=16):
    """ModelOpt's dynamic NVFP4 activation rounding (kernels/quantization/gemm/fp4_kernel_hopper.py)."""
    shape, dtype = x.shape, x.dtype
    xf = x.float().reshape(-1, shape[-1] // block, block)
    amax = xf.abs().amax(-1, keepdim=True)
    scale = (amax / (6.0 * global_scale)).clamp(max=448.0).to(torch.float8_e4m3fn).float() * global_scale
    scale = torch.where(scale >= 1e-5, scale, torch.ones_like(scale))
    return (torch.sign(xf) * round_e2m1(xf.abs() / scale) * scale).reshape(shape).to(dtype)


def fp4_fake_quant_ar(x, global_scale, block=16):
    """AutoRound's static-global-scale NVFP4 activation rounding (data_type/nvfp.py ref_nvfp4_quant), op for op.
    global_scale is the exported input_global_scale = 448*6/amax, the reciprocal of ModelOpt's input_scale."""
    shape, dtype = x.shape, x.dtype
    xf = x.float().reshape(-1, block)
    scale = (global_scale * (xf.abs().amax(-1, keepdim=True) * (1.0 / 6.0))).clamp(-448.0, 448.0)
    inv = recip(scale.to(torch.float8_e4m3fn).float() * recip(global_scale))
    xs = (xf * inv).clamp(-6.0, 6.0)
    return (torch.sign(xs) * round_e2m1(xs.abs()) * recip(inv)).reshape(shape).to(dtype)


def simulate_fp4_activations(experts, gu_quant, dn_quant):
    """Round expert inputs to NVFP4 inside a fused-experts module (eager path calls F.linear per expert).
    gu_quant / dn_quant: one rounding function per expert (AutoRound calibrates each expert separately)."""
    import torch.nn.functional as F
    linear, inner = F.linear, experts.forward
    gu, dn = experts.gate_up_proj, experts.down_proj

    def owner(w):
        p = w.data_ptr()
        for param, qs in ((gu, gu_quant), (dn, dn_quant)):
            start, step = param.data_ptr(), param[0].numel() * param.element_size()
            if start <= p < start + len(qs) * step:
                return qs[(p - start) // step]
        return None

    def q_linear(x, w, b=None):
        q = owner(w)
        return linear(q(x) if q is not None else x, w, b)

    def forward(*a, **k):
        F.linear = q_linear
        try:
            return inner(*a, **k)
        finally:
            F.linear = linear
    experts.forward = forward


class PackedNVFP4Experts(torch.nn.Module):
    """All experts of one MoE layer kept as packed NVFP4 in VRAM: gate|up and down as [E, out, in/2] uint8
    codes, [E, out, in/16] E4M3 block scales and a per-row fp32 global scale (gate and up rows each carry
    their own tensor's scale). forward dequantizes the layer into transient BF16 [E, out, in] weights (the
    transformers-5 parameter layout, bit-identical to the up-front dequant) and runs transformers'
    grouped_mm experts forward with is_transposed=False; the temporaries are freed on return."""

    def __init__(self, experts, divisor, chunk_bytes=256 << 20):
        super().__init__()
        self.num_experts, self.act_fn, self.divisor = experts.num_experts, experts.act_fn, divisor
        self.shapes = {"gate_up": (2 * experts.intermediate_dim, experts.hidden_dim),
                       "down": (experts.hidden_dim, experts.intermediate_dim)}
        self.chunk_bytes = chunk_bytes   # bounds the dequant transient (int32 index + fp32 values: ~32 B per packed byte)

    def set_weights(self, name, parts):
        """parts: per expert, the (packed, e4m3 scale, global scale) of each projection, concatenated by row."""
        rows, cols = self.shapes[name]
        packed = torch.stack([torch.cat([p for p, _, _ in ps]) for ps in parts])
        scale = torch.stack([torch.cat([s for _, s, _ in ps]) for ps in parts])
        gs = torch.stack([torch.cat([g.float().reshape(1, 1).expand(p.shape[0], 1) for p, _, g in ps]) for ps in parts])
        if (tuple(packed.shape) != (self.num_experts, rows, cols // 2) or packed.dtype != torch.uint8
                or tuple(scale.shape) != (self.num_experts, rows, cols // 16) or scale.dtype != torch.float8_e4m3fn):
            raise RuntimeError(f"unexpected packed NVFP4 {name}: {tuple(packed.shape)} {packed.dtype}, "
                               f"scales {tuple(scale.shape)} {scale.dtype}; want E={self.num_experts} [{rows}, {cols}]")
        self.register_buffer(f"{name}_packed", packed)
        self.register_buffer(f"{name}_scale", scale)
        self.register_buffer(f"{name}_global", gs.contiguous())

    def weight(self, name):
        packed, scale, gs = (getattr(self, f"{name}_{t}") for t in ("packed", "scale", "global"))
        out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, device=packed.device, dtype=torch.bfloat16)
        deq = dequant_nvfp4_ar if self.divisor else dequant_nvfp4
        step = max(1, self.chunk_bytes // (32 * packed[0].numel()))
        for e in range(0, self.num_experts, step):
            out[e:e + step] = deq(packed[e:e + step], scale[e:e + step], gs[e:e + step])
        return out

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from types import SimpleNamespace
        from transformers.integrations.moe import grouped_mm_experts_forward
        view = SimpleNamespace(num_experts=self.num_experts, has_gate=True, has_bias=False, is_transposed=False,
                               gate_up_proj=self.weight("gate_up"), down_proj=self.weight("down"),
                               _apply_gate=self._apply_gate)
        return grouped_mm_experts_forward(view, hidden_states, top_k_index, top_k_weights)


class PackedFP8Experts(torch.nn.Module):
    """All experts of one MoE layer kept as FP8 E4M3 in VRAM: gate|up and down as [E, out, in] float8 plus
    [E, out/128, in/128] fp32 weight_scale_inv (gate and up tiles concatenated by row). forward dequantizes
    the layer into transient BF16 weights (bit-identical to the up-front dequant) and runs transformers'
    grouped_mm experts forward, like PackedNVFP4Experts."""

    def __init__(self, experts, chunk_bytes=256 << 20):
        super().__init__()
        self.num_experts, self.act_fn = experts.num_experts, experts.act_fn
        self.shapes = {"gate_up": (2 * experts.intermediate_dim, experts.hidden_dim),
                       "down": (experts.hidden_dim, experts.intermediate_dim)}
        self.chunk_bytes = chunk_bytes   # bounds the dequant transient (fp32 values: 4 B per weight)

    def set_weights(self, name, parts):
        """parts: per expert, the (fp8 weight, fp32 scale_inv) of each projection, concatenated by row."""
        rows, cols = self.shapes[name]
        w = torch.stack([torch.cat([q for q, _ in ps]) for ps in parts])
        scale = torch.stack([torch.cat([s for _, s in ps]) for ps in parts]).float()
        if (tuple(w.shape) != (self.num_experts, rows, cols) or w.dtype != torch.float8_e4m3fn
                or rows % scale.shape[1] or cols % scale.shape[2]
                or any(p[0].shape[0] * scale.shape[1] != rows * p[1].shape[0] for p in parts[0])):
            raise RuntimeError(f"unexpected FP8 {name}: {tuple(w.shape)} {w.dtype}, scales {tuple(scale.shape)}; "
                               f"want E={self.num_experts} [{rows}, {cols}] float8_e4m3fn")
        self.register_buffer(f"{name}_weight", w)
        self.register_buffer(f"{name}_scale", scale)

    def weight(self, name):
        w, scale = getattr(self, f"{name}_weight"), getattr(self, f"{name}_scale")
        out = torch.empty(w.shape, device=w.device, dtype=torch.bfloat16)
        step = max(1, self.chunk_bytes // (4 * w[0].numel()))
        for e in range(0, self.num_experts, step):
            out[e:e + step] = dequant_fp8_block(w[e:e + step], scale[e:e + step])
        return out

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from types import SimpleNamespace
        from transformers.integrations.moe import grouped_mm_experts_forward
        view = SimpleNamespace(num_experts=self.num_experts, has_gate=True, has_bias=False, is_transposed=False,
                               gate_up_proj=self.weight("gate_up"), down_proj=self.weight("down"),
                               _apply_gate=self._apply_gate)
        return grouped_mm_experts_forward(view, hidden_states, top_k_index, top_k_weights)


def packed_expert_layers(path, shards, fp8=False):
    """Expert prefixes whose every gate/up/down weight is quantized (FP4: ModelOpt uint8 .weight or AutoRound
    .weight_packed; fp8: float8 .weight), read from the safetensors headers only. Layers with any BF16
    expert stay unpacked."""
    from safetensors import safe_open
    packed, plain = set(), set()
    for shard in shards:
        with safe_open(str(path / shard), "pt") as st:
            for k in st.keys():
                base, attr = k.rsplit(".", 1)
                if ".mlp.experts." in k and attr in ("weight", "weight_packed") and base.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj"):
                    prefix = base.rsplit(".", 2)[0]
                    dt = st.get_slice(k).get_dtype()
                    is_q = dt == "F8_E4M3" if fp8 else (attr == "weight_packed" or dt == "U8")
                    (packed if is_q else plain).add(prefix)
    return packed - plain


def load_nvfp4(path, device, simulate_activations=False, packed_experts=True, fp8=False):
    """ModelOpt or AutoRound NVFP4, or (fp8=True) FP8 block checkpoints, whose experts follow the same
    plan with PackedFP8Experts and .weight/.weight_scale_inv (FP8 activations are simulated by the caller).
    Build the BF16 backbone directly on the GPU, then fill it shard by shard
    (host RAM stays small). The two formats differ only in tensor names and global-scale direction: ModelOpt
    .weight/.weight_scale_2/.input_scale (multipliers), AutoRound .weight_packed/.weight_global_scale/
    .input_global_scale (divisors). packed_experts (default, ignored when simulating activations): the FP4
    experts stay packed (PackedNVFP4Experts, ~4.5 bits/weight) and are dequantized one layer at a time inside
    forward; the BF16 expert parameters are never allocated. Otherwise every expert is dequantized to BF16
    on load (Clef: ~60 GB instead of ~22 GB)."""
    from functools import partial
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as mq
    config = AutoConfig.from_pretrained(path)
    config.enable_audio_output = False
    if simulate_activations:   # per-expert F.linear is where the activation rounding hooks in (slow)
        for c in (config, config.thinker_config, config.thinker_config.text_config):
            c._experts_implementation = "eager"
    thinker = getattr(config, "thinker_config", None)
    for c in (config, thinker, getattr(thinker, "text_config", None)):   # else transformers tries AutoRound
        if c is not None and hasattr(c, "quantization_config"):
            delattr(c, "quantization_config")
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"] \
        if (path / "model.safetensors.index.json").exists() else None
    shards = sorted(set(index.values())) if index else ["model.safetensors"]
    packable = packed_expert_layers(path, shards, fp8) if packed_experts and not simulate_activations else set()
    experts_cls, experts_init = mq.Qwen3OmniMoeThinkerTextExperts, mq.Qwen3OmniMoeThinkerTextExperts.__init__
    if packable:   # experts' 3-D BF16 params go to 'meta': no 60 GB transient; materialized below if not packed
        def meta_init(self, *a, **k):
            with torch.device("meta"):
                experts_init(self, *a, **k)
        experts_cls.__init__ = meta_init
    try:
        with torch.device(device):
            model = Qwen3OmniMoeForConditionalGeneration._from_config(config, dtype=torch.bfloat16)
    finally:
        experts_cls.__init__ = experts_init
    model.eval()
    packed_mods = {}
    for name, block in list(model.named_modules()):
        if isinstance(getattr(block, "experts", None), experts_cls) and block.experts.gate_up_proj.is_meta:
            if f"{name}.experts" in packable:
                block.experts = packed_mods[f"{name}.experts"] = PackedFP8Experts(block.experts) if fp8 else \
                    PackedNVFP4Experts(block.experts, divisor=None)   # NVFP4 multiplier vs divisor: from the tensors
            else:
                block.experts.to_empty(device=device)
    params = dict(model.state_dict())
    filled, experts, in_scales = set(), {}, {}
    for shard in shards:
        with safe_open(str(path / shard), "pt", device=str(device)) as st:
            keys = list(st.keys())
            for k in keys:
                base, attr = k.rsplit(".", 1)
                if ".mlp.experts." in k and attr in ("weight", "weight_packed") and base.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj"):
                    w = st.get_tensor(k)
                    prefix, e, proj = base.rsplit(".", 2)          # ...mlp.experts, E, gate_proj
                    if prefix in packed_mods and fp8:    # keep FP8: (e4m3 weight, fp32 tile scale_inv)
                        experts.setdefault(prefix, {})[(int(e), proj)] = (w, st.get_tensor(base + ".weight_scale_inv"))
                        continue
                    if prefix in packed_mods:            # keep packed: (codes, e4m3 block scale, global scale)
                        ar = attr == "weight_packed"
                        mod = packed_mods[prefix]
                        if mod.divisor is not None and mod.divisor != ar:
                            raise RuntimeError(f"{prefix}: mixed ModelOpt / AutoRound expert tensors")
                        mod.divisor = ar
                        experts.setdefault(prefix, {})[(int(e), proj)] = (
                            w, st.get_tensor(base + ".weight_scale"),
                            st.get_tensor(base + (".weight_global_scale" if ar else ".weight_scale_2")))
                        continue
                    if w.dtype == torch.float8_e4m3fn:
                        w = dequant_fp8_block(w, st.get_tensor(base + ".weight_scale_inv"))
                    elif attr == "weight_packed":
                        w = dequant_nvfp4_ar(w, st.get_tensor(base + ".weight_scale"), st.get_tensor(base + ".weight_global_scale"))
                    elif w.dtype == torch.uint8:
                        w = dequant_nvfp4(w, st.get_tensor(base + ".weight_scale"), st.get_tensor(base + ".weight_scale_2"))
                    experts.setdefault(prefix, {})[(int(e), proj)] = w.to(torch.bfloat16)
                elif ".mlp.experts." in k and attr in ("input_scale", "input_global_scale"):
                    prefix, e, proj = base.rsplit(".", 2)
                    fq = fp4_fake_quant if attr == "input_scale" else fp4_fake_quant_ar
                    sc = st.get_tensor(k).float()
                    got = in_scales.setdefault(prefix, {}).setdefault((int(e), "down" if proj == "down_proj" else "gate_up"), (fq, sc))
                    if not torch.equal(got[1], sc):   # the fused gate_up_proj can take only one
                        raise RuntimeError(f"{base}: gate_proj and up_proj input scales differ")
                elif attr in ("weight_scale", "weight_scale_2", "input_scale", "weight_global_scale", "input_global_scale",
                              "weight_scale_inv"):
                    continue
                elif k in params:
                    w = st.get_tensor(k)
                    if w.dtype == torch.float8_e4m3fn:   # a non-expert FP8 linear (e.g. Qwen-style attention): to BF16
                        w = dequant_fp8_block(w, st.get_tensor(base + ".weight_scale_inv"))
                    params[k].copy_(w)
                    filled.add(k)
                else:
                    raise RuntimeError(f"checkpoint tensor {k} has no home in the model")
        # fuse completed expert layers into the transformers-5 3-D parameters as we go
        for prefix in list(experts):
            got = experts[prefix]
            if prefix in packed_mods:
                mod = packed_mods[prefix]
                if len(got) < 3 * mod.num_experts:
                    continue
                n = range(mod.num_experts)
                mod.set_weights("gate_up", [(got[(e, "gate_proj")], got[(e, "up_proj")]) for e in n])
                mod.set_weights("down", [(got[(e, "down_proj")],) for e in n])
                del experts[prefix]
                continue
            gu, dn = params[prefix + ".gate_up_proj"], params[prefix + ".down_proj"]
            n_exp, inter = gu.shape[0], gu.shape[1] // 2
            if len(got) < 3 * n_exp:
                continue
            for e in range(n_exp):
                gu[e, :inter].copy_(got[(e, "gate_proj")])
                gu[e, inter:].copy_(got[(e, "up_proj")])
                dn[e].copy_(got[(e, "down_proj")])
            filled.update({prefix + ".gate_up_proj", prefix + ".down_proj"})
            del experts[prefix]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log(f"loaded {shard}")
    missing = sorted(set(params) - filled)
    if experts or missing:
        raise RuntimeError(f"incomplete load: {len(missing)} params unfilled (e.g. {missing[:3]}), "
                           f"{len(experts)} expert layers incomplete")
    if packed_mods:
        nbytes = sum(b.numel() * b.element_size() for m in packed_mods.values() for b in m.buffers())
        log(f"packed {'FP8' if fp8 else 'NVFP4'} experts: {len(packed_mods)} layers (weights {nbytes / 2**30:.2f} GiB)")
    if simulate_activations:
        if not in_scales:
            raise RuntimeError("no expert input scale tensors: this checkpoint has no FP4 activations to simulate")
        modules = dict(model.named_modules())
        for prefix, sc in in_scales.items():
            fns = {part: [partial(sc[(e, part)][0], global_scale=sc[(e, part)][1].to(device))
                          for e in range(modules[prefix].num_experts)] for part in ("gate_up", "down")}
            simulate_fp4_activations(modules[prefix], fns["gate_up"], fns["down"])
        log(f"simulating NVFP4 activations in {len(in_scales)} expert layers")
    return model


def dequant_int4(qweight, scales, qzeros, g_idx):
    """AutoRound/GPTQ int4 (qweight [in/8, out], 8 codes per int32 low bits first, along `in`; qzeros
    [groups, out/8] packed along `out`): W[in, out] = scale * (q - (zp + 1)), in BF16. Triton on CUDA
    (AutoRound's own tritonv2_zp kernel, fp32 math); pure PyTorch elsewhere (CPU tests, ROCm safety)."""
    if qweight.is_cuda:
        try:
            from auto_round_extension.triton.triton_utils_zp.dequant import dequant248
            return dequant248(qweight, scales, qzeros, g_idx, 4, input_dtype=torch.bfloat16)
        except ImportError:
            pass
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    out = torch.empty(qweight.shape[0] * 8, qweight.shape[1], device=qweight.device, dtype=torch.bfloat16)
    rows = 8 * 128 * 16                   # bounded transients: 16 groups of 128 input rows at a time
    for r in range(0, out.shape[0], rows):
        q = ((qweight[r // 8:(r + rows) // 8].unsqueeze(1) >> shifts[:, None]) & 15).flatten(0, 1)
        g = g_idx[r:r + rows].long()
        z = ((qzeros[g].unsqueeze(-1) >> shifts) & 15).flatten(1) + 1
        out[r:r + rows] = (scales[g].float() * (q - z).float()).to(torch.bfloat16)
    return out


class PackedInt4Experts(torch.nn.Module):
    """All experts of one MoE layer as two stacked int4 matrices (gate|up, down), experts stacked along
    the input dim so one dequant yields the [E, in, out] BF16 layout transformers' grouped_mm wants."""

    def __init__(self, experts, act_fn):
        super().__init__()
        self.num_experts, self.act_fn = len(experts), act_fn
        for name, parts in (("gate_up", [(e.gate_proj, e.up_proj) for e in experts]),
                            ("down", [(e.down_proj,) for e in experts])):
            qw = torch.cat([torch.cat([p.qweight for p in ps], 1) for ps in parts], 0)
            qz = torch.cat([torch.cat([p.qzeros for p in ps], 1) for ps in parts], 0)
            sc = torch.cat([torch.cat([p.scales for p in ps], 1) for ps in parts], 0)
            self.register_buffer(f"{name}_qweight", qw)
            self.register_buffer(f"{name}_qzeros", qz)
            self.register_buffer(f"{name}_scales", sc)
            self.register_buffer(f"{name}_g_idx", torch.arange(qw.shape[0] * 8, device=qw.device, dtype=torch.int32) // 128)
            setattr(self, f"{name}_in", parts[0][0].infeatures)

    def weight(self, name):
        w = dequant_int4(*(getattr(self, f"{name}_{t}") for t in ("qweight", "scales", "qzeros", "g_idx")))
        return w.view(self.num_experts, getattr(self, f"{name}_in"), -1)

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """transformers' grouped_mm experts forward over per-layer BF16 temporaries (freed on return)."""
        from types import SimpleNamespace
        from transformers.integrations.moe import grouped_mm_experts_forward
        view = SimpleNamespace(num_experts=self.num_experts, has_gate=True, has_bias=False, is_transposed=True,
                               gate_up_proj=self.weight("gate_up"), down_proj=self.weight("down"),
                               _apply_gate=self._apply_gate)
        return grouped_mm_experts_forward(view, hidden_states, top_k_index, top_k_weights)


def fast_int4_moe(model):
    """Swap AutoRound's per-expert QuantLinear loop (~2.7 s/forward on Clef) for one dequant + grouped_mm
    per layer. Weights stay packed int4; only one layer's BF16 experts exist at a time. Anything
    unexpected: warn and keep the slow path for that layer."""
    done = 0
    # Snapshot NAMES only: a list of (name, module) pairs pins every old per-expert QuantLinear until
    # the loop ends, so peak = old + new packed copies (OOMed a 25000 MB HAMi slice after 15 of 48
    # layers, #156). Looked up lazily, each swapped layer's buffers free in time for the next one.
    names = [n for n, b in model.named_modules()
             if type(getattr(b, "experts", None)).__name__ == "SequentialQwen3OmniThinkerExperts"
             and callable(getattr(b, "experts_forward", None))]
    for name in names:
        block = model.get_submodule(name)
        experts = block.experts
        try:
            projs = [getattr(e, p) for e in experts for p in ("gate_proj", "up_proj", "down_proj")]
            for p in projs:
                gs, inf, outf = p.group_size, p.infeatures, p.outfeatures
                ok = (p.bits == 4 and gs == 128 and inf % gs == 0 and p.bias is None
                      and not getattr(p, "use_generic_bit_packing", False)
                      and p.qweight.dtype == p.qzeros.dtype == torch.int32
                      and tuple(p.qweight.shape) == (inf // 8, outf) and tuple(p.qzeros.shape) == (inf // gs, outf // 8)
                      and tuple(p.scales.shape) == (inf // gs, outf))
                g_idx = getattr(p, "g_idx", None)
                ok = ok and (g_idx is None or torch.equal(g_idx.long().cpu(), torch.arange(inf) // gs))
                if not ok:
                    raise ValueError(f"unsupported expert projection {type(p).__name__} "
                                     f"(bits={p.bits}, group_size={gs}, qweight {tuple(p.qweight.shape)})")
            if len({(e.gate_proj.infeatures, e.gate_proj.outfeatures, e.up_proj.outfeatures, e.down_proj.outfeatures)
                    for e in experts}) != 1 or experts[0].gate_proj.outfeatures != experts[0].down_proj.infeatures:
                raise ValueError("experts differ in shape")
            packed = PackedInt4Experts(experts, experts.act_fn)
        except Exception as e:
            log(f"WARNING: fast int4 MoE skipped for {name}: {e}")
            continue
        block.experts = packed                    # drops the per-expert QuantLinears
        block.experts_forward = packed.__call__   # the block calls experts_forward(h, top_k_index, top_k_weights)
        del experts, projs, p, g_idx, packed, block   # last local refs to the old layer
        done += 1
    log(f"fast grouped int4 MoE: {done} layers")
    return done


def load(path, device, simulate_fp4_activations=None, fast_moe=True, packed_nvfp4=True, packed_fp8=True,
         simulate_fp8_activations=False):
    """simulate_fp4_activations: off unless asked. On W4A4 checkpoints it reproduces the quantized model's
    answers exactly (for benchmarking), at a large speed cost (eager per-expert path).
    fast_moe: AutoRound INT4 g128 checkpoints run experts as dequant + grouped_mm (see fast_int4_moe).
    packed_nvfp4: NVFP4 checkpoints keep experts packed, dequantized per layer in forward (PackedNVFP4Experts).
    packed_fp8 / simulate_fp8_activations: the same for FP8 block checkpoints (PackedFP8Experts), plus
    optional dynamic FP8 rounding of expert inputs (cheap: no eager path needed)."""
    sys.path.insert(0, str(path))
    import joint_schema_model as jsm
    kind = kind_of(path)
    log(f"loading {path} ({kind}) on {device}")
    if kind == "bf16":
        model, processor = jsm.load_release_model(path, device=device)
        return model, processor, jsm, kind
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen3OmniMoeForConditionalGeneration
    if kind.startswith(("modelopt:", "auto-round:nvfp4")):
        simulate_fp4_activations = bool(simulate_fp4_activations) and kind in ("modelopt:NVFP4", "auto-round:nvfp4")
        backbone = load_nvfp4(path, device, simulate_fp4_activations, packed_nvfp4)
        kind += "+fp4-activations" if simulate_fp4_activations else ""
        kind += "+packed-experts" if any(isinstance(m, PackedNVFP4Experts) for m in backbone.modules()) else ""
    elif kind == "fp8-block":
        backbone = load_nvfp4(path, device, False, packed_fp8, fp8=True)
        if simulate_fp8_activations:
            from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as mq
            layers = [m for m in backbone.thinker.modules() if isinstance(m, (PackedFP8Experts, mq.Qwen3OmniMoeThinkerTextExperts))]
            for m in layers:
                round_fp8_expert_inputs(m)
            log(f"simulating dynamic FP8 activations in {len(layers)} expert layers")
            kind += "+fp8-activations"
        kind += "+packed-experts" if any(isinstance(m, PackedFP8Experts) for m in backbone.modules()) else ""
    else:
        config = AutoConfig.from_pretrained(path)
        config.enable_audio_output = False
        backbone = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            path, config=config, dtype=torch.bfloat16, device_map={"": str(device)})
        qc = config.quantization_config if isinstance(config.quantization_config, dict) else config.quantization_config.to_dict()
        if fast_moe and kind == "auto-round:4bit" and qc.get("group_size") == 128 \
                and "gptq" in str(qc.get("packing_format", "auto_round:auto_gptq")):
            kind += "+grouped-moe" if fast_int4_moe(backbone) else ""
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    head = jsm.JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
    head = head.to(device=device, dtype=torch.bfloat16)
    model = jsm.ClefModel(backbone.thinker, head).eval()
    return model, AutoProcessor.from_pretrained(path), jsm, kind


DEMO = {
    "model": "clef-omni",
    "state": {"goal": "Choose the next game action", "observation": "The player is facing a wall."},
    "questions": {"action": {"type": "choice", "instructions": "What is a sensible next action?",
                             "criteria": {"forward": "Move forward", "left": "Turn left", "right": "Turn right", "stop": "Stop"}},
                  "safe": {"type": "noul", "instructions": "Is it safe to keep moving forward?"}},
}


def main():
    ap = argparse.ArgumentParser(prog="serve_clef.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="dbirks/clef-omni-nvfp4", help="HF repo id or local directory")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--demo", action="store_true", help="answer one built-in request and exit")
    ap.add_argument("--request", help="answer the JSON request in this file and exit")
    ap.add_argument("--simulate-fp4-activations", action=argparse.BooleanOptionalAction, default=False,
                    help="NVFP4 W4A4 checkpoints: also round expert inputs to FP4, reproducing the quantized "
                         "model's answers exactly (slow; for benchmarking). Default: FP4 weights, BF16 activations")
    ap.add_argument("--no-fast-moe", dest="fast_moe", action="store_false",
                    help="AutoRound INT4: keep AutoRound's per-expert QuantLinear loop instead of dequant + grouped_mm")
    ap.add_argument("--no-packed-nvfp4", dest="packed_nvfp4", action="store_false",
                    help="NVFP4: dequantize every expert to BF16 on load (~60 GB on Clef) instead of keeping them "
                         "packed (~22 GB) and dequantizing one layer at a time in forward")
    ap.add_argument("--no-packed-fp8", dest="packed_fp8", action="store_false",
                    help="FP8 block: dequantize every expert to BF16 on load (~60 GB on Clef) instead of keeping "
                         "them FP8 (~32 GB) and dequantizing one layer at a time in forward")
    ap.add_argument("--simulate-fp8-activations", action=argparse.BooleanOptionalAction, default=False,
                    help="FP8 block: also round expert inputs to FP8 per 1x128 token group (W8A8-dynamic, as native "
                         "FP8 kernels run it). Default: FP8 weights, BF16 activations")
    args = ap.parse_args()

    path = resolve(args.model, args.revision)
    if args.demo or args.request:
        t0 = time.time()
        model, processor, jsm, kind = load(path, args.device, args.simulate_fp4_activations, args.fast_moe,
                                           args.packed_nvfp4, args.packed_fp8, args.simulate_fp8_activations)
        log(f"ready in {time.time() - t0:.0f}s")
        req = DEMO if args.demo else json.loads(Path(args.request).read_text())
        print(json.dumps(jsm.systemone(model, processor, req), indent=2))
        return

    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    state = {"ready": False, "requests": 0, "model_id": args.model}
    lock = threading.Lock()

    def _load():
        t0 = time.time()
        model, processor, jsm, kind = load(path, args.device, args.simulate_fp4_activations, args.fast_moe,
                                           args.packed_nvfp4, args.packed_fp8, args.simulate_fp8_activations)
        state.update(model=model, processor=processor, jsm=jsm, kind=kind, ready=True, load_s=round(time.time() - t0, 1))
        log(f"ready in {state['load_s']}s")

    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        body = {k: v for k, v in state.items() if k not in ("model", "processor", "jsm")}
        return JSONResponse(body, status_code=200 if state["ready"] else 503)

    @app.post("/v1/systemone")
    def systemone(body: dict):
        if not state["ready"]:
            raise HTTPException(503, "model loading")
        with lock:                       # one GPU: serve one decision at a time
            state["requests"] += 1
            try:
                return state["jsm"].systemone(state["model"], state["processor"], body)
            except torch.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                raise HTTPException(422, f"too many tokens: CUDA out of memory ({str(e).splitlines()[0][:200]})")
            except ValueError as e:
                msg = str(e)
                raise HTTPException(422, f"maximum context length: {msg}" if "maximum is" in msg else msg)

    threading.Thread(target=_load, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

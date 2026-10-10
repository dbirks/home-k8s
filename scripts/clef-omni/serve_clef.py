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
    weight-only nv_fp), so the FP4 expert weights are DEQUANTIZED to BF16 on load (bit-exact quantized
    values, BF16 memory and speed). For W4A4 checkpoints, --simulate-fp4-activations also rounds the
    expert inputs to NVFP4 exactly as the quantizer calibrated them (per-16 block scales in FP8 E4M3
    under the exported per-tensor input scale), reproducing the quantized model's answers; it is slow,
    so it is off by default.
    Native FP4 execution needs an engine such as vLLM, plus a hidden-state path into the head.
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
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(*packed.shape[:-1], -1).long()
    return E2M1.to(packed.device)[codes]


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


def load_nvfp4(path, device, simulate_activations=False):
    """ModelOpt or AutoRound NVFP4: build the BF16 backbone directly on the GPU, then fill it shard by shard
    (host RAM stays small), dequantizing the FP4 experts. The two formats differ only in tensor names and
    global-scale direction: ModelOpt .weight/.weight_scale_2/.input_scale (multipliers), AutoRound
    .weight_packed/.weight_global_scale/.input_global_scale (divisors)."""
    from functools import partial
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration
    config = AutoConfig.from_pretrained(path)
    config.enable_audio_output = False
    if simulate_activations:   # per-expert F.linear is where the activation rounding hooks in (slow)
        for c in (config, config.thinker_config, config.thinker_config.text_config):
            c._experts_implementation = "eager"
    thinker = getattr(config, "thinker_config", None)
    for c in (config, thinker, getattr(thinker, "text_config", None)):   # else transformers tries AutoRound
        if c is not None and hasattr(c, "quantization_config"):
            delattr(c, "quantization_config")
    with torch.device(device):
        model = Qwen3OmniMoeForConditionalGeneration._from_config(config, dtype=torch.bfloat16)
    model.eval()
    params = dict(model.state_dict())
    filled, experts, in_scales = set(), {}, {}
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"] \
        if (path / "model.safetensors.index.json").exists() else None
    shards = sorted(set(index.values())) if index else ["model.safetensors"]
    for shard in shards:
        with safe_open(str(path / shard), "pt", device=str(device)) as st:
            keys = list(st.keys())
            for k in keys:
                base, attr = k.rsplit(".", 1)
                if ".mlp.experts." in k and attr in ("weight", "weight_packed") and base.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj"):
                    w = st.get_tensor(k)
                    if attr == "weight_packed":
                        w = dequant_nvfp4_ar(w, st.get_tensor(base + ".weight_scale"), st.get_tensor(base + ".weight_global_scale"))
                    elif w.dtype == torch.uint8:
                        w = dequant_nvfp4(w, st.get_tensor(base + ".weight_scale"), st.get_tensor(base + ".weight_scale_2"))
                    prefix, e, proj = base.rsplit(".", 2)          # ...mlp.experts, E, gate_proj
                    experts.setdefault(prefix, {})[(int(e), proj)] = w.to(torch.bfloat16)
                elif ".mlp.experts." in k and attr in ("input_scale", "input_global_scale"):
                    prefix, e, proj = base.rsplit(".", 2)
                    fq = fp4_fake_quant if attr == "input_scale" else fp4_fake_quant_ar
                    sc = st.get_tensor(k).float()
                    got = in_scales.setdefault(prefix, {}).setdefault((int(e), "down" if proj == "down_proj" else "gate_up"), (fq, sc))
                    if not torch.equal(got[1], sc):   # the fused gate_up_proj can take only one
                        raise RuntimeError(f"{base}: gate_proj and up_proj input scales differ")
                elif attr in ("weight_scale", "weight_scale_2", "input_scale", "weight_global_scale", "input_global_scale"):
                    continue
                elif k in params:
                    params[k].copy_(st.get_tensor(k))
                    filled.add(k)
                else:
                    raise RuntimeError(f"checkpoint tensor {k} has no home in the model")
        # fuse completed expert layers into the transformers-5 3-D parameters as we go
        for prefix in list(experts):
            got = experts[prefix]
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
    for name, block in list(model.named_modules()):
        experts = getattr(block, "experts", None)
        if not (type(experts).__name__ == "SequentialQwen3OmniThinkerExperts" and callable(getattr(block, "experts_forward", None))):
            continue
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
        done += 1
    log(f"fast grouped int4 MoE: {done} layers")
    return done


def load(path, device, simulate_fp4_activations=None, fast_moe=True):
    """simulate_fp4_activations: off unless asked. On W4A4 checkpoints it reproduces the quantized model's
    answers exactly (for benchmarking), at a large speed cost (eager per-expert path).
    fast_moe: AutoRound INT4 g128 checkpoints run experts as dequant + grouped_mm (see fast_int4_moe)."""
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
        backbone = load_nvfp4(path, device, simulate_fp4_activations)
        kind += "+fp4-activations" if simulate_fp4_activations else ""
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
    args = ap.parse_args()

    path = resolve(args.model, args.revision)
    if args.demo or args.request:
        t0 = time.time()
        model, processor, jsm, kind = load(path, args.device, args.simulate_fp4_activations, args.fast_moe)
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
        model, processor, jsm, kind = load(path, args.device, args.simulate_fp4_activations, args.fast_moe)
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

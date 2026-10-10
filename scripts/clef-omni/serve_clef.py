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
  * ModelOpt NVFP4 / NVFP4A16 (hf_quant_config.json): transformers cannot load packed NVFP4, so the
    FP4 expert weights are DEQUANTIZED to BF16 on load (bit-exact quantized values, BF16 memory and
    speed). For the W4A4 checkpoint the expert inputs are also rounded to NVFP4 exactly as ModelOpt
    calibrated them (per-16 block scales in FP8 E4M3 under the exported input_scale), so answers
    match the quantized model; turn it off with --no-simulate-fp4-activations. Native FP4 execution
    needs an engine such as vLLM, plus a hidden-state path into the head.
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
        return f"auto-round:{qc.get('bits', '?')}bit"
    if method:
        raise SystemExit(f"unsupported quantization_config.quant_method={method!r}")
    return "bf16"


def dequant_nvfp4(packed, scale, scale_2, block=16):
    """ModelOpt NVFP4: two E2M1 codes per byte (low nibble first), e4m3 scale per 16, fp32 global scale."""
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(*packed.shape[:-1], -1).long()
    vals = E2M1.to(packed.device)[codes]
    vals = vals.view(*vals.shape[:-1], -1, block) * (scale.float() * scale_2.float()).unsqueeze(-1)
    return vals.reshape(*packed.shape[:-1], -1).to(torch.bfloat16)


def fp4_fake_quant(x, global_scale, block=16):
    """ModelOpt's dynamic NVFP4 activation rounding (kernels/quantization/gemm/fp4_kernel_hopper.py)."""
    shape, dtype = x.shape, x.dtype
    xf = x.float().reshape(-1, shape[-1] // block, block)
    amax = xf.abs().amax(-1, keepdim=True)
    scale = (amax / (6.0 * global_scale)).clamp(max=448.0).to(torch.float8_e4m3fn).float() * global_scale
    scale = torch.where(scale >= 1e-5, scale, torch.ones_like(scale))
    a = xf.abs() / scale
    q = torch.where(a <= 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a <= 1.25, 1.0, torch.where(
        a < 1.75, 1.5, torch.where(a <= 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)))))))
    return (torch.sign(xf) * q * scale).reshape(shape).to(dtype)


def simulate_fp4_activations(experts, gu_scale, dn_scale):
    """Round expert inputs to NVFP4 inside a fused-experts module (eager path calls F.linear per expert)."""
    import torch.nn.functional as F
    linear, inner = F.linear, experts.forward
    gu, dn = experts.gate_up_proj, experts.down_proj

    def owner(w):
        p = w.data_ptr()
        for param, sc in ((gu, gu_scale), (dn, dn_scale)):
            start = param.data_ptr()
            if start <= p < start + param.numel() * param.element_size():
                return sc
        return None

    def q_linear(x, w, b=None):
        sc = owner(w)
        return linear(fp4_fake_quant(x, sc) if sc is not None else x, w, b)

    def forward(*a, **k):
        F.linear = q_linear
        try:
            return inner(*a, **k)
        finally:
            F.linear = linear
    experts.forward = forward


def load_modelopt(path, device, simulate_activations=False):
    """Build the BF16 backbone directly on the GPU, then fill it shard by shard (host RAM stays small)."""
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration
    config = AutoConfig.from_pretrained(path)
    config.enable_audio_output = False
    for c in (config, config.thinker_config, config.thinker_config.text_config):
        c._experts_implementation = "eager"        # per-expert F.linear: needed for activation rounding
    for c in (config, getattr(config, "thinker_config", None)):
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
                if ".mlp.experts." in k and k.endswith(".weight") and k.rsplit(".", 2)[-2] in ("gate_proj", "up_proj", "down_proj"):
                    base = k[: -len(".weight")]
                    w = st.get_tensor(k)
                    if w.dtype == torch.uint8:
                        w = dequant_nvfp4(w, st.get_tensor(base + ".weight_scale"), st.get_tensor(base + ".weight_scale_2"))
                    prefix, e, proj = base.rsplit(".", 2)          # ...mlp.experts, E, gate_proj
                    experts.setdefault(prefix, {})[(int(e), proj)] = w.to(torch.bfloat16)
                elif k.endswith(".input_scale") and ".mlp.experts." in k:
                    prefix, _, proj = k[: -len(".input_scale")].rsplit(".", 2)
                    in_scales.setdefault(prefix, {})["down" if proj == "down_proj" else "gate_up"] = st.get_tensor(k).float()
                elif k.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
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
            raise RuntimeError("no expert input_scale tensors: this checkpoint has no FP4 activations to simulate")
        modules = dict(model.named_modules())
        for prefix, sc in in_scales.items():
            simulate_fp4_activations(modules[prefix], sc["gate_up"].to(device), sc["down"].to(device))
        log(f"simulating NVFP4 activations in {len(in_scales)} expert layers")
    return model


def load(path, device, simulate_fp4_activations=None):
    """simulate_fp4_activations: None = on exactly when the checkpoint quantized activations (W4A4)."""
    sys.path.insert(0, str(path))
    import joint_schema_model as jsm
    kind = kind_of(path)
    log(f"loading {path} ({kind}) on {device}")
    if kind == "bf16":
        model, processor = jsm.load_release_model(path, device=device)
        return model, processor, jsm, kind
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen3OmniMoeForConditionalGeneration
    if kind.startswith("modelopt:"):
        if simulate_fp4_activations is None:
            simulate_fp4_activations = kind == "modelopt:NVFP4"
        backbone = load_modelopt(path, device, simulate_fp4_activations)
        kind += "+fp4-activations" if simulate_fp4_activations else ""
    else:
        config = AutoConfig.from_pretrained(path)
        config.enable_audio_output = False
        backbone = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            path, config=config, dtype=torch.bfloat16, device_map={"": str(device)})
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
    ap.add_argument("--simulate-fp4-activations", action=argparse.BooleanOptionalAction, default=None,
                    help="NVFP4 W4A4 checkpoints: round expert inputs to FP4 like the quantized model (default: on for W4A4)")
    args = ap.parse_args()

    path = resolve(args.model, args.revision)
    if args.demo or args.request:
        t0 = time.time()
        model, processor, jsm, kind = load(path, args.device, args.simulate_fp4_activations)
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
        model, processor, jsm, kind = load(path, args.device, args.simulate_fp4_activations)
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

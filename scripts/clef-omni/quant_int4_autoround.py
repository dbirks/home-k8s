"""Clef-Omni quants with Intel AutoRound 0.16.0 (issue #152): INT4 W4A16 g128, NVFP4 (W4A4) and NVFP4A16.

AR_SCHEME selects the variant: W4A16 (int4, group 128, default), NVFP4 (FP4 weights + FP4 activations,
blocks of 16) or NVFP4A16 (FP4 weights, 16-bit activations). Output root defaults per variant.

Experts-only like the ModelOpt quants (router, attention, lm_head, embeddings, vision/audio towers and
the Clef head stay BF16). AutoRound's Qwen3-Omni handler unfuses the transformers-5 fused experts into
per-expert Linears, tunes each decoder block (SignRound, `iters` steps) on Clef's real forward path,
and writes the auto_round (auto_gptq-packed) format that transformers loads with int4 kernels.

Host RAM guard (46 GB, no swap): the BF16 model is loaded straight onto the GPU, and AutoRound's two
"move the whole model to CPU" calls are intercepted (orchestrator: after input caching; calibration
fallback: on a CUDA error). Tuned blocks still go to CPU one at a time and are freed by the shard writer.

    CLEF_REHEARSAL=1 python quant_int4_autoround.py   # tiny random model, same code path
    python quant_int4_autoround.py                    # real run -> /work/int4/export
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clef_records  # noqa: E402
from clef_records import log  # noqa: E402

REHEARSAL = os.environ.get("CLEF_REHEARSAL") == "1"
MODEL = Path(os.environ.get("CLEF_MODEL_DIR", "/work/clef-omni"))
SCHEME = os.environ.get("AR_SCHEME", "W4A16")
_ROOTS = {"W4A16": "int4", "NVFP4": "nvfp4-ar", "NVFP4A16": "nvfp4a16-ar"}
ROOT = Path(os.environ.get("CLEF_ROOT", f"/work/{_ROOTS[SCHEME]}{'-rehearsal' if REHEARSAL else ''}"))
SCHEME_ARG = {"bits": 4, "group_size": 16, "data_type": "nv_fp", "act_bits": 16} if SCHEME == "NVFP4A16" else SCHEME
# The whole BF16 model stays on the GPU (host-RAM guard), so tuning headroom is ~35 GB. INT4 peaked at 85 GB;
# NVFP4 W4A4 OOMed in block 0 at the defaults, so the NVFP4 variants keep cached inputs in host RAM and the
# W4A4 one also caps the sequence length.
LOW_GPU_MEM = os.environ.get("AR_LOW_GPU_MEM", "1" if SCHEME.startswith("NVFP4") else "0") == "1"
SEQLEN = int(os.environ.get("AR_SEQLEN", 4096 if SCHEME == "NVFP4" else 8192))
EXPORT, RES = ROOT / "export", ROOT / "results"
ITERS = int(os.environ.get("AR_ITERS", 2 if REHEARSAL else 200))
N_CALIB = int(os.environ.get("N_CALIB", 16 if REHEARSAL else 512))
sys.path.insert(0, str(MODEL))
import joint_schema_model as jsm  # noqa: E402


def avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return round(int(line.split()[1]) / 2**20, 1)


def rss_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return round(int(line.split()[1]) / 2**20, 2)


def mem():
    return (f"gpu {torch.cuda.memory_allocated() / 2**30:.1f} GiB (peak {torch.cuda.max_memory_allocated() / 2**30:.1f}), "
            f"rss {rss_gib()} GiB, host avail {avail_gib()} GiB")


def load_backbone():
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration
    config = AutoConfig.from_pretrained(MODEL)
    config.enable_audio_output = False
    if REHEARSAL:
        tc = config.thinker_config
        t = tc.text_config
        t.num_hidden_layers, t.hidden_size, t.intermediate_size, t.moe_intermediate_size = 2, 256, 128, 128
        t.num_experts, t.num_experts_per_tok, t.num_attention_heads, t.num_key_value_heads = 8, 2, 4, 2
        v = tc.vision_config
        v.depth, v.hidden_size, v.num_heads, v.intermediate_size, v.out_hidden_size = 3, 64, 2, 128, 256
        v.deepstack_visual_indexes = [0, 1, 2]
        a = tc.audio_config
        a.d_model, a.encoder_layers, a.encoder_attention_heads, a.encoder_ffn_dim, a.output_dim = 64, 1, 2, 128, 256
        torch.manual_seed(0)
        with torch.device("cuda"):
            return Qwen3OmniMoeForConditionalGeneration._from_config(config, dtype=torch.bfloat16).eval()
    return Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, config=config, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()


def guard_host_ram(top):
    """Keep AutoRound from copying the whole (GPU-resident) model into host RAM."""
    import auto_round.calibration.llm as cal
    import auto_round.compressors.orchestrator as orch
    original = orch.mv_module_from_gpu

    def keep_top_on_gpu(module):
        if module is top:
            log("guard: AutoRound asked to move the whole model to CPU; keeping it on the GPU")
            return module
        return original(module)

    def refuse_cpu_fallback(module):
        if module is top:
            raise RuntimeError("AutoRound tried its CPU calibration fallback (whole model to host RAM); "
                               "refusing: it would OOM the 46 GB node")
        return original(module)

    orch.mv_module_from_gpu = keep_top_on_gpu
    cal.mv_module_from_gpu = refuse_cpu_fallback


def calib_dicts(processor, records):
    out = []
    for r in records:
        enc = jsm.encode_record(processor.tokenizer, r, processor=processor)
        b = jsm.collate_records([enc], processor.tokenizer.pad_token_id, torch.device("cpu"))
        media = dict(b["media"] or {})
        uav = bool(media.pop("use_audio_in_video", False))
        for k in ("pixel_values", "pixel_values_videos", "input_features"):
            if k in media:
                media[k] = media[k].to(torch.bfloat16)
        d = {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"], "use_cache": False, **media}
        if uav:
            d["use_audio_in_video"] = True
        out.append(d)
    return out


def main():
    os.environ.setdefault("AR_DISABLE_COPY_MTP_WEIGHTS", "1")    # never copy talker weights into the export
    RES.mkdir(parents=True, exist_ok=True)
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL)
    calib, _ = clef_records.build(n_calib=512, n_eval=128)       # same split as every other quant
    calib = calib[:N_CALIB]
    data = calib_dicts(processor, calib)
    log(f"{len(data)} calibration records, {sum(int(d['input_ids'].shape[1]) for d in data)} tokens, "
        f"{sum(1 for d in data if 'pixel_values' in d or 'pixel_values_videos' in d or 'input_features' in d)} multimodal")

    t0 = time.time()
    backbone = load_backbone()
    log(f"backbone on GPU in {time.time() - t0:.0f}s;", mem())
    guard_host_ram(backbone)

    from auto_round import AutoRound
    ar = AutoRound(backbone, tokenizer=processor.tokenizer, processor=processor, scheme=SCHEME_ARG, dataset=data,
                   nsamples=len(data), seqlen=SEQLEN, iters=ITERS, batch_size=1, gradient_accumulate_steps=8,
                   device_map=0, low_gpu_mem_usage=LOW_GPU_MEM, ignore_layers="mlp.gate,lm_head,self_attn")
    log(f"AutoRound {type(ar).__name__}: scheme={SCHEME} iters={ITERS} nsamples={len(data)} "
        f"seqlen={SEQLEN} low_gpu_mem_usage={LOW_GPU_MEM}")
    t0 = time.time()
    _, out_dir = ar.quantize_and_save(output_dir=str(ROOT / "ar-out"), format="auto_round")
    log(f"quantize_and_save in {time.time() - t0:.0f}s -> {out_dir};", mem())

    shutil.move(str(out_dir), str(EXPORT))
    shutil.rmtree(ROOT / "ar-out", ignore_errors=True)
    for p in MODEL.iterdir():   # Clef head, loader, processor/tokenizer, license; never the BF16 weights
        if p.is_file() and not p.name.startswith((".", "model")) and not (EXPORT / p.name).exists():
            shutil.copy2(p, EXPORT / p.name)
    if REHEARSAL:   # tiny random head matching the tiny backbone, so the export loads end to end
        from safetensors.torch import save_file
        head = jsm.JointSchemaHead(hidden_size=256, width=64, routing_layers=1, layers=1, heads=2, feedforward=128)
        save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(EXPORT / "joint_head.safetensors"))
        (EXPORT / "joint_head_config.json").write_text(json.dumps(
            {"hidden_size": 256, "width": 64, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 128}))

    # artifact scan: int4 packed experts, BF16 everything else, no talker
    from safetensors import safe_open
    scan = {"files": {}, "qweight_tensors": 0, "expert_bf16_weights": 0, "talker_tensors": 0, "dtypes": {}}
    for f in sorted(EXPORT.glob("model*.safetensors")):
        scan["files"][f.name] = f.stat().st_size
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                if k.startswith(("talker.", "code2wav.")):
                    scan["talker_tensors"] += 1
                if ".mlp.experts." in k and k.endswith((".qweight", ".weight_packed")):
                    scan["qweight_tensors"] += 1
                if ".mlp.experts." in k and k.endswith(".weight"):
                    scan["expert_bf16_weights"] += 1
                dt = str(st.get_slice(k).get_dtype())
                scan["dtypes"][dt] = scan["dtypes"].get(dt, 0) + 1
    scan["export_bytes"] = sum(p.stat().st_size for p in EXPORT.iterdir() if p.is_file())
    scan["quantization_config"] = json.loads((EXPORT / "config.json").read_text()).get("quantization_config")
    json.dump(scan, open(RES / "artifact_scan.json", "w"), indent=1)
    log("ARTIFACT SCAN:", json.dumps({k: v for k, v in scan.items() if k not in ("files", "quantization_config")}))
    ok = scan["qweight_tensors"] > 0 and scan["expert_bf16_weights"] == 0 and scan["talker_tensors"] == 0
    log(f"{SCHEME} ARTIFACT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

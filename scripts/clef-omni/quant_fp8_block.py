"""Clef-Omni FP8 block-128 W8A8-dynamic (issue #154): model-free RTN of the thinker's routed experts.

No model load, no calibration, no GPU: the release stores every expert as its own 2-D tensor
(`thinker.model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight`), so each is rounded to
float8_e4m3fn with one fp32 `weight_scale_inv` per 128x128 tile, streaming shard by shard. The math is
transformers 5.10.2 `integrations/finegrained_fp8.py::Fp8Quantize._quantize_one` op for op (scale =
448/amax, clamp, cast; stored inverse = 1/scale), and the layout is Qwen's own `-FP8` releases
(`quant_method fp8`, `fmt e4m3`, `activation_scheme dynamic`, `weight_block_size [128, 128]`), which
transformers (FineGrainedFP8), vLLM and SGLang load natively.

Experts-only like the rest of the family: attention, router (mlp.gate), embeddings, lm_head, vision and
audio towers and the Clef head stay BF16 byte-for-byte (listed in modules_to_not_convert). talker.* and
code2wav.* are dropped.

Host RAM: about one tensor at a time. Shard layouts are planned from the source headers, so each output
shard's header is written first and tensors are streamed into it with plain reads/writes (no mmap, no
whole-shard buffer as safetensors.save_file would build). Peak RSS is the interpreter + the largest
single tensor (the BF16 embedding / lm_head, ~0.6 GB each) + an expert's fp32 temporaries.

    CLEF_REHEARSAL=1 CLEF_MODEL_DIR=/tmp/tiny-bf16 python quant_fp8_block.py   # same code, tiny checkpoint
    python quant_fp8_block.py                                                 # real run -> /work/fp8/export
"""
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch

REHEARSAL = os.environ.get("CLEF_REHEARSAL") == "1"
MODEL = Path(os.environ.get("CLEF_MODEL_DIR", "/work/clef-omni"))
ROOT = Path(os.environ.get("CLEF_ROOT", "/work/fp8-rehearsal" if REHEARSAL else "/work/fp8"))
EXPORT, RES = ROOT / "export", ROOT / "results"
SHARD_BYTES = int(float(os.environ.get("CLEF_SHARD_BYTES", 4 << 20 if REHEARSAL else 5e9)))   # tiny: force >1 shard
DEVICE = os.environ.get("FP8_DEVICE", "cpu")      # "cuda" only speeds up the rounding
BLOCK = (128, 128)
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max                    # 448
EXPERT = re.compile(r"^thinker\.model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")
DROP = ("talker.", "code2wav.")
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64, "F8_E4M3": FP8,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
CODES = {v: k for k, v in DTYPES.items()}


def log(*a):
    print(f"[{time.strftime('%FT%T')}]", *a, flush=True)


def peak_rss_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return round(int(line.split()[1]) / 2**20, 2)


def read_header(path):
    """safetensors header -> (data start offset, {key: {dtype, shape, data_offsets}})."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def read_tensor(f, start, meta):
    a, b = meta["data_offsets"]
    f.seek(start + a)
    buf = bytearray(b - a)
    f.readinto(buf)
    return torch.frombuffer(buf, dtype=torch.uint8).view(DTYPES[meta["dtype"]]).reshape(meta["shape"])


def write_shard(path, entries, produce):
    """entries: [(key, dtype code, shape)] in file order; produce() yields the tensors in that order."""
    header, off = {"__metadata__": {"format": "pt"}}, 0
    for k, dt, shape in entries:
        n = math.prod(shape) * torch.empty((), dtype=DTYPES[dt]).element_size()
        header[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for (k, dt, shape), t in zip(entries, produce(), strict=True):
            if CODES[t.dtype] != dt or list(t.shape) != list(shape):
                raise RuntimeError(f"{k}: produced {t.dtype} {tuple(t.shape)}, planned {dt} {shape}")
            f.write(memoryview(t.contiguous().reshape(-1).view(torch.uint8).numpy()))
        if f.tell() != 8 + len(hb) + off:
            raise RuntimeError(f"{path}: wrote {f.tell()} bytes, planned {8 + len(hb) + off}")


def quantize(w):
    """Fp8Quantize._quantize_one (transformers 5.10.2), block 128x128, float scales. Returns (fp8, fp32 inv)."""
    rows, cols = w.shape
    if rows % BLOCK[0] or cols % BLOCK[1]:
        raise ValueError(f"expert weight {tuple(w.shape)} is not a multiple of {BLOCK}")
    t = w.to(DEVICE, torch.float32).reshape(rows // BLOCK[0], BLOCK[0], cols // BLOCK[1], BLOCK[1])
    amax = t.abs().amax(dim=(-3, -1))
    scales = torch.where(amax > 0, FP8_MAX / torch.where(amax > 0, amax, torch.ones_like(amax)), torch.ones_like(amax))
    q = torch.clamp(t * scales.unsqueeze(-1).unsqueeze(-3), min=-FP8_MAX, max=FP8_MAX).to(FP8).reshape(rows, cols)
    return q.cpu(), (1.0 / scales).to(torch.float32).cpu()


def dequantize(q, inv):
    """Fp8Dequantize._dequantize_one: fp32 product per tile."""
    r, c = inv.shape
    return (q.float().reshape(r, q.shape[0] // r, c, q.shape[1] // c) * inv[:, None, :, None]).reshape(q.shape)


def transformers_crosscheck(samples):
    """Byte-compare our rounding with transformers' own Fp8Quantize when it is importable (the /work venv)."""
    try:
        from types import SimpleNamespace
        from transformers.integrations.finegrained_fp8 import Fp8Quantize
    except ImportError:
        return "skipped (transformers not importable)"
    op = Fp8Quantize(SimpleNamespace(quantization_config=SimpleNamespace(weight_block_size=list(BLOCK), scale_fmt="float")))
    for k, w in samples:
        ref = op.convert({k: w})
        q, inv = quantize(w)
        sk = k[:-len("weight")] + "weight_scale_inv"
        if not (torch.equal(ref[k].view(torch.uint8), q.view(torch.uint8)) and torch.equal(ref[sk], inv)):
            return f"MISMATCH on {k}"
    return f"byte-identical on {len(samples)} tensors"


def source_shards():
    idx = MODEL / "model.safetensors.index.json"
    return sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]


def plan(shards):
    """Source tensors in source order (talker/code2wav dropped), from the headers only: (shard, key, meta,
    output entries [(key, dtype code, shape)], output bytes)."""
    items = []
    for shard in shards:
        _, h = read_header(MODEL / shard)
        for k, m in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if k.startswith(DROP):
                continue
            if EXPERT.match(k):
                r, c = m["shape"]
                if r % BLOCK[0] or c % BLOCK[1]:
                    raise ValueError(f"{k}: {m['shape']} is not a multiple of {BLOCK}")
                outs = [(k, "F8_E4M3", [r, c]), (k[:-len("weight")] + "weight_scale_inv", "F32", [r // BLOCK[0], c // BLOCK[1]])]
                items.append((shard, k, m, outs, r * c + 4 * (r // BLOCK[0]) * (c // BLOCK[1])))
            else:
                a, b = m["data_offsets"]
                items.append((shard, k, m, [(k, m["dtype"], m["shape"])], b - a))
    return items


def not_converted(items):
    """Every non-expert module with a >=2-D weight (exact names, Qwen-FP8 style, which vLLM matches exactly)
    plus whole-subtree prefixes (transformers' should_convert_module: re.match prefix or endswith)."""
    mods = {k[:-len(".weight")] for _, k, m, _, _ in items
            if k.endswith(".weight") and not EXPERT.match(k) and len(m["shape"]) >= 2}
    return ["lm_head", "thinker.lm_head", "thinker.visual", "thinker.audio_tower", "thinker.model.embed_tokens",
            "talker", "code2wav"] + sorted(mods)


def should_convert(name, patterns):
    """transformers 5.10.2 quantizers_utils.should_convert_module, verbatim semantics."""
    return not any(re.match(f"{p}\\.", name) or re.match(f"{p}", name) or name.endswith(p) for p in patterns)


def write_export(items):
    total = sum(it[4] for it in items)
    groups, cur, size = [], [], 0
    for it in items:
        if cur and size + it[4] > SHARD_BYTES:
            groups.append(cur); cur, size = [], 0
        cur.append(it); size += it[4]
    groups.append(cur)
    names = [f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors" for i in range(len(groups))]
    log(f"{len(items)} tensors, {total / 1e9:.2f} GB -> {len(groups)} shards of <= {SHARD_BYTES / 1e9:.3g} GB")
    weight_map, samples, stats = {}, [], {"worst": 0.0, "n": 0}
    starts, files = {}, {}
    t0 = time.time()

    def produce(group):
        for shard, k, m, _, _ in group:
            if shard not in files:
                for f in files.values():
                    f.close()
                files.clear()
                files[shard], starts[shard] = open(MODEL / shard, "rb"), read_header(MODEL / shard)[0]
            w = read_tensor(files[shard], starts[shard], m)
            if not EXPERT.match(k):
                yield w
                continue
            q, inv = quantize(w)
            n = stats["n"]
            if n < 3:
                samples.append((k, w.clone()))
            if n < 8 or n % 997 == 0:   # spot-check the round trip against the E4M3 error bound
                err = (dequantize(q, inv) - w.float()).abs()
                ulp = inv.repeat_interleave(BLOCK[0], 0).repeat_interleave(BLOCK[1], 1) * 2.0 ** -10   # half subnormal step
                stats["worst"] = max(stats["worst"], float((err / torch.maximum(w.float().abs() / 16, ulp)).max()))
            stats["n"] += 1
            yield q
            yield inv

    for name, group in zip(names, groups, strict=True):
        entries = [e for it in group for e in it[3]]
        write_shard(EXPORT / name, entries, lambda g=group: produce(g))
        weight_map.update({k: name for k, _, _ in entries})
        log(f"wrote {name} ({len(weight_map)} tensors so far, {time.time() - t0:.0f}s, peak rss {peak_rss_gib()} GiB)")
    for f in files.values():
        f.close()
    worst = stats["worst"]
    (EXPORT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))
    return samples, worst


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def scan(items, modules_to_not_convert):
    cfg = json.loads((EXPORT / "config.json").read_text())
    t = cfg["thinker_config"]["text_config"]
    want = t["num_hidden_layers"] * t["num_experts"] * 3
    s = {"fp8_expert_weights": 0, "expected_fp8_expert_weights": want, "scales": 0, "bad_scales": 0,
         "bad_scale_shapes": 0, "fp8_nan_bytes": 0, "expert_bf16_weights": 0, "talker_tensors": 0,
         "non_expert_mismatch": 0, "dtypes": {}, "files": {}}
    src_other = {k: m for _, k, m, _, _ in items if not EXPERT.match(k)}
    seen_other, index = {}, json.loads((EXPORT / "model.safetensors.index.json").read_text())["weight_map"]
    for path in sorted(EXPORT.glob("model-*.safetensors")):
        s["files"][path.name] = path.stat().st_size
        start, h = read_header(path)
        with open(path, "rb") as f:
            for k, m in h.items():
                dt = m["dtype"]
                s["dtypes"][dt] = s["dtypes"].get(dt, 0) + 1
                s["index_mismatch"] = s.get("index_mismatch", 0) + int(index.get(k) != path.name)
                if k.startswith(DROP):
                    s["talker_tensors"] += 1
                elif EXPERT.match(k):
                    if dt != "F8_E4M3":
                        s["expert_bf16_weights"] += 1
                        continue
                    s["fp8_expert_weights"] += 1
                    q = read_tensor(f, start, m).view(torch.uint8)
                    s["fp8_nan_bytes"] += int(((q & 0x7F) == 0x7F).sum())
                elif k.endswith(".weight_scale_inv"):
                    s["scales"] += 1
                    inv = read_tensor(f, start, m)
                    w = h[k[:-len("_scale_inv")]]["shape"]
                    s["bad_scales"] += int((~torch.isfinite(inv) | (inv <= 0)).sum())
                    s["bad_scale_shapes"] += int(dt != "F32" or m["shape"] != [w[0] // BLOCK[0], w[1] // BLOCK[1]])
                else:
                    seen_other[k] = (dt, m["shape"])
    s["non_expert_mismatch"] = len(set(src_other) ^ set(seen_other)) + sum(
        seen_other[k] != (m["dtype"], m["shape"]) for k, m in src_other.items() if k in seen_other)
    s["index_mismatch"] = s.get("index_mismatch", 0) + abs(len(index) - sum(s["dtypes"].values()))
    s["non_expert_tensors"] = len(seen_other)
    experts = {k.rsplit(".", 3)[0] for _, k, _, _, _ in items if EXPERT.match(k)}
    others = set(modules_to_not_convert)   # exact names; re-checked through the pattern semantics
    s["experts_not_converted"] = sum(not should_convert(m, modules_to_not_convert) for m in experts)
    s["other_modules_converted"] = sum(should_convert(m, modules_to_not_convert) for m in others)
    s["head_sha256_match"] = sha256(EXPORT / "joint_head.safetensors") == sha256(MODEL / "joint_head.safetensors")
    s["export_bytes"] = sum(p.stat().st_size for p in EXPORT.iterdir() if p.is_file())
    return s


def main():
    RES.mkdir(parents=True, exist_ok=True)
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    EXPORT.mkdir(parents=True)
    torch.set_grad_enabled(False)
    t0 = time.time()
    items = plan(source_shards())
    samples, worst = write_export(items)
    log(f"quantized in {time.time() - t0:.0f}s; peak rss {peak_rss_gib()} GiB; "
        f"round-trip error / bound (max over checked tensors) = {worst:.3f}")
    crosscheck = transformers_crosscheck(samples)
    log("transformers Fp8Quantize cross-check:", crosscheck)

    for p in MODEL.iterdir():   # Clef head, loader, processor/tokenizer, license; never the BF16 weights
        if p.is_file() and not p.name.startswith((".", "model")) and p.name != "config.json":
            shutil.copy2(p, EXPORT / p.name)
    cfg = json.loads((MODEL / "config.json").read_text())
    cfg["enable_audio_output"] = False   # talker / code2wav are not in the export
    modules = not_converted(items)
    cfg["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                                  "weight_block_size": list(BLOCK), "modules_to_not_convert": modules}
    (EXPORT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    s = scan(items, modules)
    s.update(round_trip_ratio=round(worst, 4), transformers_crosscheck=crosscheck, peak_rss_gib=peak_rss_gib(),
             seconds=round(time.time() - t0, 1))
    json.dump(s, open(RES / "artifact_scan.json", "w"), indent=1)
    log("ARTIFACT SCAN:", json.dumps({k: v for k, v in s.items() if k != "files"}))
    ok = (s["fp8_expert_weights"] == s["expected_fp8_expert_weights"] == s["scales"] and s["bad_scales"] == 0
          and s["bad_scale_shapes"] == 0 and s["fp8_nan_bytes"] == 0 and s["expert_bf16_weights"] == 0
          and s["talker_tensors"] == 0 and s["non_expert_mismatch"] == 0 and s["experts_not_converted"] == 0
          and s["other_modules_converted"] == 0 and s["index_mismatch"] == 0 and s["head_sha256_match"] and worst <= 1.001
          and not crosscheck.startswith("MISMATCH"))
    log("FP8 ARTIFACT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

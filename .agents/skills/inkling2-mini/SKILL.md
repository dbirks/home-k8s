---
name: inkling2-mini
description: Operate Thinking Machines Inkling 2 Mini on the eGPU laptop's RTX 5090 (talos-8me-tcy) - the live vLLM FP8-KV 256K engine, the held SGLang DSpark config, what KV dtypes and long-context flags work on SM120 and which fail, how to test context, speed, prefix caching and parallel sessions, and how to switch engines. Use when changing, upgrading, resizing or debugging apps/inkling2-mini-*.yaml.
---

# Inkling 2 Mini on the RTX 5090 eGPU

**Files**
- `apps/inkling2-mini-models.yaml`: pinned + sha256-checked download onto talos-210-73x's HDD (source copy).
- `apps/inkling2-mini-laptop-weights.yaml`: the laptop NVMe PVC `inkling2-mini-models-laptop`, plus an HTTP server
  (talos-210-73x) and a copy Job (laptop) that move the 20 GB over the LAN and verify `SHA256SUMS`.
- `apps/inkling2-mini-vllm.yaml`: LIVE engine + the shared `inkling-2-mini` Service.
- `apps/inkling2-mini-sglang.yaml.hold`: tuned SGLang alternative (faster decode, half the context).

The model: 26.4B total / 3.4B active looped MoE, native NVFP4 + MXFP8 (~20 GB, no other "NVFP4 version" to pick),
bundled DSpark drafter in `mtp.safetensors`. Only 4 of 24 layers are full attention, but two sit in the 3x loop,
so 8 full-attention visits/token: about 32 KiB/token of BF16 KV. The 40 sliding-window visits are ~160 KiB/token but
only the last 1024 tokens per request are needed.

## Results (2026-11-21, measured in a lab pod)

| Engine / config | KV | pool | max ctx | 1 stream | 2 streams | notes |
|---|---|---|---|---|---|---|
| SGLang cookbook 5090 DSpark | BF16 | 49K | 32K | 172 tok/s | 263 | as shipped |
| SGLang tuned (held) | BF16 | 123K (0.85) / 141K (0.88) | ~pool | 176 | 283 | needle 120K/138K PASS |
| **vLLM (live)** | **FP8** | **277K** | **256K** | 158 | 230 | needle 245K PASS, no DSpark |

Prefix caching works in both (a repeated 23.5K-token system prompt: ~23.4K cached, TTFT 3-5x lower). That is the
"shared KV between sessions" lever; both engines have it on by default.

## What fails on SM120 (do not retry without a new image)
- **SGLang FP8 KV** (`fp8_e4m3`/`fp8_e5m2`): FA4's SM120 path has no FP8 (`q.dtype bfloat16 != k.dtype float8`).
  `mxfp8` KV: "FP8 is only supported on SM100". Triton attention: the model asserts "Inkling2 requires FA4", and the
  DSpark drafter "requires FA4 draft attention". So SGLang is BF16-only on the 5090.
- **SGLang loop knobs** (`--inkling2-loop-share-cache`, `--inkling2-loop-num-passes`): need `loop_config.is_varloop`
  true; this checkpoint is false (only `thinkingmachines/TML-d24-varloop-research`).
- **vLLM DSpark**: "no kernel image is available for execution on the device" in the drafter, with FP8 or BF16 KV.
- **vLLM KV dtypes** beyond fp8: Inkling2 accepts only auto/bfloat16/float16/fp8/fp8_e4m3 (nvfp4 refused).
- SGLang `--mem-fraction-static` >= 0.86 with prefill CUDA graphs, or 0.91 without: OOM.

## Gotchas
- **HAMi's allocator cap**: near the top of the card libvgpu refuses allocations (`HAMI-core ... cuMemoryAllocate
  failed`). Both engines set `CUDA_DISABLE_CONTROL=true` (safe: only GPU tenant on the laptop).
- SGLang's logged "avail mem" overstates real headroom: ~5 GB goes to non-PyTorch memory (CuTe JIT, workspaces).
- The 5090's power range is 400-600 W; the cluster's 300 W fallback is rejected, so `apps/gpu-power-limit.yaml`
  has `GPU_POWER_MAP: "RTX 5090=400"`.
- vLLM's first requests after start are slow (JIT); bench warm.
- vLLM on SM12x: keep `--max-num-batched-tokens 2048` until vLLM PR #69021.
- Temperature 1.0 (model-card sampling) occasionally gives an empty answer on a needle; rerun before concluding.
- **Never hold/remove a file that defines the weights PVC**: Flux prunes it and local-path deletes the data. That is
  why the PVC lives in its own file. A re-copy is ~15 min (set the weights server to 1, delete the copy Job if done).

## Lab loop (fast flag experiments without git round-trips)
1. Set the live Deployment to `replicas: 0` in git.
2. `kubectl apply` a throwaway pod with the same image/PVC/GPU limits and `command: ["sleep","infinity"]`.
3. `kubectl exec` a launcher that kills the old server, starts `sglang serve` / `vllm serve` with extra flags in the
   background, waits for "ready to roll" / "Application startup complete", and greps the KV-pool lines.
4. Port-forward and run: a math check, 1 vs 2 parallel 800-token streams, a repeated long system prompt (cached
   tokens), needle-in-haystack at ~95% of the pool (filler "The grass is green. ..." is ~15 tokens per sentence).
5. Put the winner in git, delete the lab pod, set replicas back.

## Switching engines
One commit: vLLM `replicas: 0` and rename the SGLang `.hold` to `.yaml` (or the reverse). Both pod templates carry
`app.kubernetes.io/name=inkling-2-mini`, so the Service follows whichever runs. Collective is not wired yet.

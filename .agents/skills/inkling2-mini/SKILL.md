---
name: inkling2-mini
description: Operate Thinking Machines Inkling 2 Mini on the eGPU laptop's RTX 5090 (talos-8me-tcy) - the live vLLM DSpark + FP8-KV 256K engine (with the SM120 DSpark patch), the held SGLang DSpark config, what KV dtypes and long-context flags work on SM120 and which fail, how to test context, speed, prefix caching and parallel sessions, and how to switch engines. Use when changing, upgrading, resizing or debugging apps/inkling2-mini-*.yaml.
---

# Inkling 2 Mini on the RTX 5090 eGPU

**Files**
- `apps/inkling2-mini-models.yaml`: pinned + sha256-checked download onto talos-210-73x's HDD (source copy).
- `apps/inkling2-mini-laptop-weights.yaml`: the laptop NVMe PVC `inkling2-mini-models-laptop`, plus an HTTP server
  (talos-210-73x) and a copy Job (laptop) that move the 20 GB over the LAN and verify `SHA256SUMS`.
- `apps/inkling2-mini-vllm.yaml`: LIVE engine, the shared `inkling-2-mini` Service, and the ConfigMap
  `inkling2-mini-vllm-patches` (vLLM PR #69096, applied with `patch` at container start).
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
| vLLM, no DSpark (until 2026-11-21 pm) | FP8 | 277K | 256K | 158 | 233 | needle 245K PASS |
| **vLLM + DSpark (live)** | **FP8** | **314K** | **256K** | 244 code / 151 prose | 340 / 265 (4: 620 / 440) | needle 250K PASS, 4 seqs |

Prefix caching works in both (a repeated 23.5K-token system prompt: ~23.4K cached, TTFT 3-5x lower). That is the
"shared KV between sessions" lever; both engines have it on by default.

## What fails on SM120 (do not retry without a new image)
- **SGLang FP8 KV** (`fp8_e4m3`/`fp8_e5m2`): FA4's SM120 path has no FP8 (`q.dtype bfloat16 != k.dtype float8`).
  `mxfp8` KV: "FP8 is only supported on SM100". Triton attention: the model asserts "Inkling2 requires FA4", and the
  DSpark drafter "requires FA4 draft attention". So SGLang is BF16-only on the 5090.
- **SGLang loop knobs** (`--inkling2-loop-share-cache`, `--inkling2-loop-num-passes`): need `loop_config.is_varloop`
  true; this checkpoint is false (only `thinkingmachines/TML-d24-varloop-research`).
- **vLLM DSpark on the stock image**: "no kernel image is available". Cause (issue #69081): the image is built for
  sm_100/103 only and the drafter's KV write calls `reshape_and_cache_flash`, which has no SM120 kernel. FIXED here by
  applying PR #69096 (Triton KV write, Python only) at start. Remove the patch once a pinned image contains it.
- **vLLM KV dtypes** beyond fp8: Inkling2 accepts only auto/bfloat16/float16/fp8/fp8_e4m3 (nvfp4 refused).
- SGLang `--mem-fraction-static` >= 0.86 with prefill CUDA graphs, or 0.91 without: OOM.

## Gotchas
- **HAMi's allocator cap**: near the top of the card libvgpu refuses allocations (`HAMI-core ... cuMemoryAllocate
  failed`). Both engines set `CUDA_DISABLE_CONTROL=true` (safe: only GPU tenant on the laptop).
- SGLang's logged "avail mem" overstates real headroom: ~5 GB goes to non-PyTorch memory (CuTe JIT, workspaces).
- The 5090's power range is 400-600 W; the cluster's 300 W fallback is rejected, so `apps/gpu-power-limit.yaml`
  has `GPU_POWER_MAP: "RTX 5090=400"`.
- vLLM's first requests after start are slow (JIT); bench warm.
- vLLM on SM12x: `--max-num-batched-tokens` must stay below 2048-ish until PR #69021; we use 1024, which is also what
  makes DSpark + 256K fit (2048 chunks reserve more activation/SWA memory: 177K max).
- Leave `--gpu-memory-utilization` UNSET: the extensible KV cache sizes itself after warmup/graph capture (11.6 GiB of
  KV vs 8.7 GiB at 0.91). Startup is ~4 min. Recipe credit: github.com/mxfp0/Inkling-2-Mini-RTX-5090.
- DSpark speeds up code ~1.5x (acceptance ~2.8) but prose barely (~1.85, can be slightly slower at 1 stream).
- Temperature 1.0 (model-card sampling) occasionally gives an empty answer on a needle; rerun before concluding.
- **Never hold/remove a file that defines the weights PVC**: Flux prunes it and local-path deletes the data. That is
  why the PVC lives in its own file. A re-copy is ~15 min (set the weights server to 1, delete the copy Job if done).

## NVFP4 KV (our fork, tested 2026-11-21, not live yet)
- Code: github.com/dbirks/vllm branch `inkling2-mini-sm120` (PR #69028 head eb7981b + #69096 + NVFP4 KV). Image:
  `build/vllm-inkling/Dockerfile` = the pinned day-0 image + the 9 patched .py files fetched at a fork commit (one
  ~500 kB layer), built by `gh workflow run build-vllm-inkling.yaml -f tag=... -f vllm_ref=<fork sha>`.
- Format: 72-byte rows per (head, token) = 64 B packed E2M1 + 8 E4M3 scales per 16 values; reference
  `vllm/models/inkling2/nvidia/ops/nvfp4_kv_ref.py`. Writer fused into `ops/sconv.py` (attention_prologue) via
  `ops/nvfp4_kv_write.py`; reader in `ops/attention_sm90.py` (`_load_nvfp4_kv`). CPU tests run under TRITON_INTERPRET.
- Results on the 5090 (DSpark on, 2 seqs, 1024-token chunks, no util cap): pool 410,604 tokens at max-model-len 262144
  (fp8 313,818), 466,416 at 393216, 484,639 at 458752. 20/20 arithmetic, needles PASS at 250K x3, 380K x2, 445K.
  Decode same as fp8 (code ~264, prose ~154 tok/s 1 stream); prefill ~15-25% slower (250K: ~65 s vs 54 s).
- Cost: NVFP4 blocks are 512 tokens (see below) and vLLM's scheduler block becomes 18,432 tokens, so prefix-cache hits
  round DOWN to 18,432-token multiples (a repeated 23.5K prompt reused 18,432 tokens: TTFT 5.4 s vs 0.4 s on fp8).
- Gotchas found on the GPU (the CPU interpreter hides both): compiled @triton.jit code can only read module globals
  wrapped in `tl.constexpr`; vLLM unifies KV page sizes across groups by integer block-size ratios else PADS to the
  largest page, and the BF16 DSpark draft (512 B/head-token) vs NVFP4 (144) is 3.56x, so block-16 NVFP4 pages were
  padded and used more memory than fp8. Fix: NVFP4 block = 32 x block_size (page = 9x the draft page).
- 524K (Justin's number) does not fit with DSpark on (484K pool max at 448K context).

## Research notes (2026-11-21 pi wave)
- SGLang FP8 KV on SM120 would need PR #45871 ported into Inkling's rel-bias attention.
- Other engines (llama.cpp mainline is broken on CUDA SM120 as of b12606; ingine is 1 day old, single-flight, text-only;
  Basalt and NInfer-all exist) are not better than vLLM + DSpark here.
- HiCache/host KV tier: not worth it with a 20 Gi pod limit (host tier smaller than the device pool).

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
`app.kubernetes.io/name=inkling-2-mini`, so the Service follows whichever runs.

## Collective
Model `inkling-2-mini` (backend `inkling-2-mini` in both `apps/llm-collective-contributor*.yaml`), concurrency prod 2 =
vLLM `--max-num-seqs 2`, dev 1 on top (dev is rarely used; vLLM queues any overflow). If you change max-num-seqs (or
switch to SGLang `--max-running-requests`), keep prod equal to it. Context 262144 (on SGLang it would be ~123K: lower `limits.context`). openai + anthropic protocols, text +
image. Reasoning: the template maps none/minimal/low/medium/high/xhigh/max to an effort number (default 0.9 = high); there is
no off switch, so "none" still emits a short thought. Server default is medium (`--default-chat-template-kwargs`); request values override it.

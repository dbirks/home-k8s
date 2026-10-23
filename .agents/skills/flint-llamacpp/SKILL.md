---
name: flint-llamacpp
description: Run Recursive Flint (40B-A2.5B MoE coding model, NVFP4 GGUF) on the RTX PRO 6000 (SM120) as the Collective `flint` model - vLLM with an SM120 arch gate (live) or llama.cpp (parked) - image builds, flags, KV sizing and why, switch/rollback with NInfer, and how to test it (curl, parallel, pi). Use when operating, resizing, upgrading or rolling back apps/flint-vllm.yaml or apps/flint-llamacpp.yaml.hold.
---

# Flint (#158): vLLM live, llama.cpp parked

## LIVE since 2026-10-22: vLLM (apps/flint-vllm.yaml)
Image `ghcr.io/dbirks/vllm-openai:flint-sm120-*` from `build/vllm-flint/Dockerfile` via
`.github/workflows/build-vllm-flint.yaml`: day-0 `vllm/vllm-openai:flint` + three fixes from the PR #62938
thread: (1) gate `_flint_kernels()` on `is_device_capability_family(100)` so SM120 takes stock layers
(Marlin NVFP4 linear + MoE, FlashInfer attention with sinks; without it: "No supported CUDA architectures
found for major versions [9, 10]"); (2) `pip install gguf>=0.19.0`; (3) delete `/usr/local/cuda-*/compat`.
- KV is FP8 since 2026-10-22 (image `flint-sm120-fp8kv-*`, patch `build/vllm-flint/fp8_kv.py`). Stock
  `FlintAttention` asserts a BF16 cache because its fused Triton `qk_norm_rope_cache` writes the cache itself.
  The patch passes no cache views to that kernel when `--kv-cache-dtype fp8` (it still norms/ropes q and k,
  k in place) and calls vLLM's `unified_kv_cache_update` (reshape_and_cache_flash quantizes) before
  `unified_attention_with_output`. FlashInfer on SM120 resolves Q=bf16 for fa2 prefill + XQA decode with an
  e4m3 cache, sinks + sliding window OK. Scales are uncalibrated 1.0 (vLLM warns; the 159K needle still
  passes). The SM100 Flint decode kernels read BF16, so the patch refuses fp8 there. Rollback = drop the flag.
- Memory: slice 56800 MiB (every MiB HAMi has left beside CLEF 25000 + 2x TTS 8000) with
  `--kv-cache-memory 30 GiB` instead of `--gpu-memory-utilization`: 1,882,584 tokens (7.2x full 262K
  requests), pod at 55,233 of 56,800 MiB. vLLM's own startup line "`--kv-cache-memory=N` to fully utilize gpu
  memory" is the ceiling to size against; leave ~1.5 GiB under it.
- Why it mattered: under real agent load (avg prompt ~98K tokens) the old 790K BF16 pool held only 6-8
  requests; the rest waited ~45 s and evicted prefixes got re-prefilled (hit rate fell from ~75% to ~6%).
  Diagnose with the engine log (`GPU KV cache usage` ~90%+, `Waiting`/`Deferred` > 0) and
  `/metrics` `vllm:request_queue_time_seconds`, `vllm:prefix_cache_hits_total / _queries_total`.
- The 300 W cap shows as `clocks_throttle_reasons.active 0x4` (SW power cap) with SM ~1.2 GHz of 3.09:
  prefill is compute-bound, so it is roughly proportionally slower. Stability trade-off, see gpu-power-management.
- After FP8 + 30 GiB (light live load): 236 tok/s single, 787/1286/1928 tok/s aggregate at 4/8/16;
  159K needle found in 21.6 s; tool calls and `/v1/messages` thinking blocks OK.
- First BF16 measurements (55000 slice, util 0.90, 25.18 GiB = 790,158 tokens), with live traffic on top:
  101 tok/s single, 425/676/995 tok/s at 4/8/16, 159K needle in 21 s.
- vLLM logs stats only while busy: a silent log is idle, not hung (check `/health`).

## Why llama.cpp was parked
Under real agent load (16 slots, 30-170K prompts) it was ~17 tok/s per request with `--kv-unified`: one KV
stream, and `get_n_kv` = highest used cell, so every decode step scanned ALL cached sessions (~1.3M
cells). `--no-kv-unified` (8 x 229K) doubled that to ~30 tok/s, still ~150 tok/s total at 5 busy. Its
batched MoE decode is the ceiling. The notes below are kept for reviving it.

# llama.cpp notes (apps/flint-llamacpp.yaml.hold)

Files: `apps/flint-models.yaml` (HDD PVC `flint-models-hdd` + pinned, SHA-checked download Job),
`apps/flint-llamacpp.yaml` (Deployment + Service `flint:8080`), `apps/flint-vllm.yaml.hold` (fallback),
`.github/workflows/build-llama-flint.yaml` (image), Collective backend `flint` in
`apps/llm-collective-contributor.yaml`.

## Why llama.cpp, not vLLM
- vLLM PR #62938 (`vllm/vllm-openai:flint`): `_flint_kernels()` only checks TP==1 + quant type, never the
  GPU arch, and its MoE grouped GEMM / small-M GEMV are FlashInfer CuTe DSL **SM100 `tcgen05`** kernels
  (GB200-validated). SM120 has no tcgen05. Escape hatch if ever needed: patch `_flint_kernels` to return
  False (stock vLLM layers, ~2.7x slower batch-1 decode on GB200) as an overlay image like build-vllm-dflash2.
- SGLang: BF16 only (80 GB), H200/B200/B300/GB300/Spark. Ollama: macOS only at 0.40.8-rc0. Recursive `rec`:
  Apple Silicon only. NInfer/Basalt/Pennyroyal: no Flint. The NVFP4 GGUF card lists RTX PRO 6000 as tested on llama.cpp.

## Image
llama.cpp Flint support is PR #30928 (unmerged as of 2026-10-21); it needs #30861 + #30929 (both merged by
b11799). The workflow merges the PR head onto a release tag and builds `.devops/cuda.Dockerfile --target
server`, CUDA 12.8.1, `CUDA_DOCKER_ARCH=120a-real` (~30 min on a GitHub runner, 2.46 GB image):
```bash
gh workflow run build-llama-flint.yaml -f base=b11799 -f pr_sha=<PR head>   # fails fast if the PR head moved
gh run view <id> --log | grep -oE 'ghcr.io/dbirks/llama-cpp-flint@sha256:[0-9a-f]{64}'   # pin this
```
Once #30928 merges, any upstream `ghcr.io/ggml-org/llama.cpp:server-cuda` release containing it will do.

## Flags that matter (long names; b11799)
- `--cache-type-k/v q8_0` + `--ctx-size 1310720 --kv-unified --parallel 16`: one shared q8_0 pool; KV is
  only on the 14 full-attention layers (14,336 values/token: 28 KiB f16, ~14.9 KiB q8_0). llama-server caps
  each slot at the 262,144 training context itself (log: `n_ctx_slot = 262144`; the `n_ctx_seq > n_ctx_train`
  warning is expected). No FP8 KV type exists in llama.cpp; q8_0 is the near-lossless choice. The card uses
  f16; rollback = f16 + `--ctx-size 655360` (same memory).
- Leave `--swa-full` unset (compact 513-token SWA cache).
- `--load-mode none` (no mmap; `--no-mmap` no longer exists), `--cache-ram 0` (no host prompt cache: RAM-poor node, ZDR).
- `--jinja --reasoning-format deepseek`: reasoning lands in `reasoning_content`; reasoning-preserve is on by default.
- Sampling per card: `--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0`, `--n-predict 65536`.

## Switch / rollback
Never run beside NInfer (55,000 + 50,000 MB do not fit with CLEF 25,000 + 2x TTS 8,000). One commit flips
`ninfer` replicas and `flint` replicas and swaps the Collective backend block; rollback is the reverse.
Pre-pull the image first with a throwaway CPU Job (7 min pull) so the gap is only the ~2 min model load
from the HDD.

## Verify
```bash
kubectl port-forward svc/flint 18080:8080 &
curl -s localhost:18080/props | jq '.total_slots, .build_info'
python3 -I scripts/flint_test.py http://localhost:18080 all   # tools, /v1/messages, 1/4/8/16 parallel, 39K/159K needle
```
Measured 2026-10-21 (350 W cap, beside CLEF + 2x TTS): decode 200 tok/s single (NInfer Qwen3.8-27B MTP3 was
~110), aggregate 494 / 679 / 713 tok/s at 4 / 8 / 16 parallel (45 tok/s each at 16); prefill 4.35k tok/s at
39K, 2.8k at 159K (decode 65 tok/s there); needle found at 159K; 16/16 correct arithmetic at 16 parallel;
tool calls + tool-result turn OK; `/v1/messages` returns thinking + text blocks. Card total 82.6 GB used
with all four services (was ~94 with NInfer). Startup: model load ~2 min.

Through the Collective: `curl https://llm-collective.e-gineering.com/v1/models -H "Authorization: Bearer $LLMC_API_KEY"`
lists `flint`. pi: `~/.pi/agent/models.json` providers `collective` / `collective-msgs` have a `flint`
entry (reasoning, 262144 ctx, 65536 out); their `apiKey` is the literal `${LLMC_API_KEY}` that pi expands,
so test curl with the env var, not the file value. `pi --provider collective --model flint`.

## Gotchas
- `ERROR: init 250 result=11` is printed once at container start, before llama-server's own logging; source
  not pinned down (the TTS pods don't print it). Harmless so far: the server loads and serves normally.
- The Collective needs `protocols: [openai, anthropic]`; most live agent traffic arrives as Anthropic `/v1/messages`.

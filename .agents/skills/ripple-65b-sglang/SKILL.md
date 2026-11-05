---
name: ripple-65b-sglang
description: Run Ricursive Ripple-65B Preview NVFP4 on stock SGLang 0.5.19 on the single RTX PRO 6000 (SM120) as the Collective model ripple-65b - image build, weight staging, flags and why, KV/SWA sizing, how to test correctness and concurrency, and the switch/rollback with qwen38-27b. Use when operating, resizing, upgrading, debugging or rolling back apps/ripple-65b-sglang.yaml (issue 168).
---

# Ripple-65B NVFP4 on SGLang (issue #168)

Files: `apps/ripple-65b-models.yaml` (weights), `apps/ripple-65b-sglang.yaml` (Deployment + Service `ripple-65b:8000`),
`build/ripple-sglang/` + `.github/workflows/build-ripple-sglang.yaml` (image), `scripts/` here (tests).
Issue #168 is the system of record (comments carry the measurements).

## The model, in numbers that matter
- Dense 65B, 80 layers: 64 sliding-window (1024 tokens, RoPE) + 16 full attention (NoPE), GQA 80 q / 8 kv heads x 128,
  value residuals + sigmoid output gate (`g_proj`). Vocab 100,352. Supports 2Mi positions.
- NVFP4 (ModelOpt W4A4, group 16) on the MLPs only; attention, embeddings, lm_head BF16. 57.2 GB on disk,
  ~53 GiB on the card. It is NOT a 4-bit 65B.
- KV per token (BF16): full layers 16 x 2 x 8 x 128 x 2 B = **64 KiB**. Sliding layers would be 256 KiB/token but
  live in SGLang's separate hybrid-SWA pool, which only needs each sequence's last 1024 tokens.
- Chat template always thinks (`<think>...</think>`); no thinking switch, no effort levels. Tool calls are
  hermes-like `<tool_call>\n{json}\n</tool_call>`. Use Ricursive's `ripple` parsers, never hermes/deepseek.

## Image (build/ripple-sglang)
- `FROM lmsysorg/sglang:v0.5.19-cu130@sha256:d6e72886...` = the exact SGLang Ricursive pins (torch 2.13.0+cu130,
  CUDA 13.0, fine on the node's 580.167.08 driver). `/usr/local/cuda*/compat` removed.
- `/opt/ripple/model.py` = NVFP4 repo `inference/local/model.py` (quant_config plumbed through) + the BF16 server
  copy's load-gap check (raises on unused checkpoint tensors / unloaded params; `RIPPLE_STRICT_LOAD=0` logs instead).
- `/opt/ripple/ripple_format.py` = BF16 repo `inference/server/ripple_format.py` (newer than the local copy:
  reconciles tool-call deltas). `chat_template.jinja` = repo root (sha 877c2dbf...). `serve.py` = registration only,
  every flag from the Deployment. Registration must happen at import of `__main__` because SGLang spawns workers.
- Deliberately NOT carried over from `inference/local/spark.py`: the SM121 PDL-softmax patch (DFlash accept kernel),
  the k/v-scale reset (this checkpoint has no KV scales), the cc==(12,1) gate, max-running-requests 1,
  disable-radix-cache, mmap-off loader flags (those are for Spark's unified memory).
- Build: `gh workflow run build-ripple-sglang.yaml -f tag=ripple-sglangXXXX`, then pin the digest printed by the
  push step. It pushes into the existing PUBLIC `ghcr.io/dbirks/vllm-openai` package: GITHUB_TOKEN gets
  `denied` creating a new package (and a new one would be private). The build's check step imports the launcher on
  the CPU runner and asserts the architecture + both parsers registered ("ripple registered" in the log).
- Pre-pull: all 70 base layers are byte-identical to the Docker Hub base, so
  `talosctl ... image pull --namespace cri docker.io/lmsysorg/sglang:v0.5.19-cu130@sha256:d6e72886...` first (14 GB,
  loop it, containerd resumes), then the ghcr image only adds ~4 small layers.

## Weights (apps/ripple-65b-models.yaml)
HDD PVC `ripple-65b-models-hdd` (80Gi request, room for the ~8 GB DFlash2 drafter later). Job pins the HF commit,
checks sha256 of every file, writes `/models/.verified-<rev>`. The node's WAN is ~100 Mbit/s: 57 GB took hours.
Use `HF_TOKEN` (secret `hf-token`) + `HF_XET_HIGH_PERFORMANCE=1`; a Job's pod template is immutable, so to change
it delete the Job and let Flux recreate it (and watch for Flux recreating the OLD spec first if the source was not
yet refreshed - annotate the GitRepository, then the Kustomization).

## Flags and why (apps/ripple-65b-sglang.yaml)
| flag | why |
|---|---|
| `--attention-backend flashinfer` | trtllm_mha (Ricursive's B200 pick) is SM100-only; FA3 is Hopper-only. Supports the hybrid-SWA pool. |
| `--fp4-gemm-backend flashinfer_cutlass` | SGLang's own SM120 auto choice; pinned so an upgrade cannot swap it silently. |
| `--kv-cache-dtype bfloat16` | the only KV dtype Ricursive validated; fp8 is a later A/B. |
| `--context-length 65536` | conservative start. |
| `--swa-full-tokens-ratio 0.2` | default 0.8 sizes the SWA pool at 80% of the full pool's tokens x 4 sliding layers per full layer = most of the bytes; SWA only needs ~window + chunk per live request. |
| `--max-running-requests 16`, `--chunked-prefill-size 8192`, `--schedule-policy lpm` | multi-agent batching; lpm admits the request with the longest cached prefix first. |
| `--cuda-graph-backend-prefill disabled` | as both Ricursive recipes. |
| radix cache ON (default), no HiCache | host RAM (46 GB, no swap) is the binding constraint here, not VRAM. |
| `--tool-call-parser ripple --reasoning-parser ripple --chat-template /opt/ripple/chat_template.jinja` | Ricursive's. |

HAMi: `nvidia.com/gpumem: 97800` = whole card; qwen38-27b, CLEF, both Qwen3-TTS must be 0. Strategy `Recreate`.

## Switch and rollback
One commit flips `apps/ripple-65b-sglang.yaml` replicas 0->1, `apps/qwen38-27b-vllm.yaml` 1->0, and swaps the
Collective backends (ripple-65b active, qwen38-27b commented as PARKED) in both contributor ConfigMaps.
**Rollback = `git revert <switch commit>`** (or do the reverse by hand), push, reconcile:
`kubectl -n flux-system annotate gitrepository flux-system reconcile.fluxcd.io/requestedAt="$(date +%s)" --overwrite`
then the same on `kustomization apps`. Qwen comes back in ~5-10 min (weights already on its PVC, image on the node).

## Test
```bash
kubectl port-forward svc/ripple-65b 18000:8000 &
python3 -I .agents/skills/ripple-65b-sglang/scripts/ripple_test.py http://localhost:18000           # chat/stream/tools/reasoning/messages
python3 -I .agents/skills/ripple-65b-sglang/scripts/bench.py http://localhost:18000 ripple-65b 1,4,8,16 8000 256
curl -s localhost:18000/get_server_info | jq '.max_total_num_tokens, .internal_states[0]'
curl -s localhost:18000/metrics | grep -E "^sglang:(cache_hit_rate|token_usage|num_running_reqs|num_queue_reqs|gen_throughput)"
pi -p --no-session --model collective/ripple-65b "Say hi" < /dev/null    # the < /dev/null is required or pi hangs
```

## Measured on this node
### Boot (2026-11-04/05)
- Weights 53.52 GB on the card. Load off the HDD 696-808 s (~70 MB/s; sequential vs multithread made no real
  difference, the HDD is the limit). FlashInfer fp4_gemm autotune ~25 s, decode CUDA graphs 3 s. Ready ~14 min after start.
- Host RAM: pod ~6.3 GiB RSS, node MemAvailable ~24 GB with it running. No HiCache.
- Startup probe MUST have `timeoutSeconds` > 1: /health does a 1-token generate polled in 1 s steps, so the 1 s default
  never passes and the pod stays 0/1 (Service has no endpoints -> Collective "connection refused") until it is killed.
- `ERROR: init 250 result=11` at the top of the log is HAMi's shim, harmless. "Tokenizer ... still TokenizersBackend"
  is harmless too: ripple_format's probe check found no mismatch (it logs "Restored ..." when it fixes one).

### BF16 KV, context 65,536, mem-fraction 0.88, swa ratio 0.2, max-running 16
- Pool: full 269,151 tokens (16.4 GB) + SWA 53,830 tokens (13.1 GB); 10.4 GB VRAM left after graphs.
- Correctness: ripple_test.py 7/7 (chat, streaming reasoning/content deltas, tool call with id, tool round trip,
  tool_choice=required strict, /v1/messages). kv_ab.py: math 3/4 (the "fail" is the checker taking the last number of
  a listed-primes answer), code ran, tool OK, needles 5/5 at 31K and 59K prompt tokens.
- Ripple decides by itself whether to think: "reply with only the number" prompts often skip `<think>` and guess
  (209 vs 215, 28 vs 21); with thinking it is right. Not an engine bug. Same template has no effort levels.
- Speed (shared 9K prefix, 256 out): single stream 21 tok/s, aggregate 51 / 86 / 175 tok/s at 4 / 8 / 16 (~19 tok/s
  per stream at 16). Cold 9K prefill TTFT 4.5 s (~2K tok/s prefill), warm re-send 0.17 s; ~98% device cache hits.
  Same through the Collective at 1/4/8: TTFT 0.35-0.7 s warm, 95% hits. Dense 65B at the 300 W cap: ~20 tok/s/stream
  is the ceiling (vs Qwen3.8-27B 85 tok/s single, 537 aggregate at 24).
- Capacity (distinct 38K-token agents, 3 turns each): 8 agents (307K > 269K pool) -> 73% / 83% hits on turns 2/3,
  p95 TTFT 57 s / 31 s; 12 agents -> ~50% hits, p95 TTFT 140-160 s = the eviction spiral. Cold prefill ~1.6K tok/s.
  So BF16 KV holds ~6 agents of ~40K each; that is why FP8 KV was tried next.

## Status + FP8 KV A/B (2026-11-05)
- **PARKED** at `replicas: 0` (commit dedacc9); Pennyroyal Flash-Next is primary again. Revival = flip
  `apps/ripple-65b-sglang.yaml` and `apps/pennyroyal-flashnext.yaml` replicas, uncomment the PARKED `ripple-65b`
  blocks in both contributor configs and comment out `pennyroyal`. Consider pointing the `models` volume at the
  NVMe copy `ripple-65b-models-ssd` (apps/ripple-65b-models-ssd.yaml, same path) to cut the ~12-14 min HDD load.
- Last live config: `--kv-cache-dtype fp8_e4m3 --context-length 131072 --max-running-requests 12
  --swa-full-tokens-ratio 0.07`, Collective prod 8 / dev 2, context 131072.
- FP8 vs BF16 at temperature 0 (`scripts/kv_ab.py`): correctness suite 7/7 both; math, code-vs-tests and tool
  call identical; needles 5/5 at 31K, 59K and **119K** (FP8). No sign of silent corruption despite the checkpoint
  having no k/v scales (1.0 default). Pool: BF16 269,151 full + 53,830 SWA tokens -> FP8 755,989 full + 52,919 SWA.
- FP8 load/concurrency test was cut short by the park: no FP8 throughput numbers yet. BF16 numbers above stand
  (~21 tok/s single, 175 tok/s at 16; ~6 agents of ~40K before the eviction spiral at 12).
- Not yet tried: NVFP4 KV (needs prefill flashinfer + decode trtllm_mha, page-size 64; trtllm_mha XQA does run on
  SM120 in 0.5.19 despite older comments), DFlash2 drafter.

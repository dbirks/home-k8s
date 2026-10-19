---
name: ninfer-serving-memory
description: Size NInfer (qwen3.8-27b-ninfer, apps/ninfer-standalone.yaml) VRAM KV and host RAM on this RAM-poor, VRAM-rich node, upgrade its image/artifact pair, and roll it out without hanging. Use when NInfer's host memory looks too big, when fitting other models (CLEF, Qwen3-TTS) beside it, when changing its HAMi slice, KV dtype or context-cache flags, or when rebuilding the dbirks/ninfer image.
---

# NInfer serving memory (VRAM KV vs pinned host tier)

## Mental model
- Active KV is ALWAYS in VRAM: one shared "Main Text KV" pool, sized at startup by `--kv-capacity auto`
  (fills the HAMi slice after weights + runtime, minus 1 GiB headroom). Retained prefixes live there too.
- A host context tier pins host RAM for inactive continuations / pause snapshots. It is pinned
  (`/dev/zero (deleted)` in smaps, `shmem` in memory.stat) and cannot be reclaimed.
- This node is 46GB RAM, no swap, 96GB VRAM: the host tier is the wrong trade. Zero it, buy VRAM:
  - **current engine (2026-10+):** `--host-context-mib 0 --device-state-slots 16`
  - old v2-era engine (20260908): `--host-kv-mib 0 --host-state-slots 0 --device-state-slots 16`
  Those two old flags were REMOVED upstream (folded into `--host-context-mib`); passing them to the new
  image fails startup. With 0 host budget, a preempted request REPLAYS (recompute) on resume; the
  `ninfer_{preemptions,snapshot_restores,replay_restores}_total` metrics show how often.

## Numbers
| date | engine / KV | HAMi slice | extras | VRAM KV tokens | pod host RAM |
|---|---|---|---|---:|---:|
| 2026-09-24 | v2, int8 (33,792 B/tok) | 40000 MB | host tier 8 GiB+8 | 289,728 | ~11.7GB |
| 2026-09-24 | v2, int8 | 50000 MB | host 0 | 553,600 | ~2GB |
| 2026-10-16 | v3, **nvfp4** (18,432 B/tok) | 55000 MB | host 0, dflash2 | **1,519,424** | ~2.3GB |
| 2026-10-17 | v3, nvfp4 | 55000 MB | + `--vision` | **1,463,424** | ~2.3GB |
| 2026-10-17 | **Sept build** 20260908 + v2, nvfp4, 8 lanes | 55000 MB | + `--vision` | **1,245,056** (4.75 GiB free) | |
Weights 21.4 GiB (21.7 with vision), runtime 30.5 GiB, engine ready in ~9.5 s. Vision costs only ~56K
tokens. Single-stream decode ~177 tok/s with `--spec dflash2 --draft-tokens 7 --lm-head-draft`.
Verify on the `capacity |` startup log line; `GET /metrics` exposes running/waiting/paused gauges.

## Current build: dbirks fork, crash fixed, 16 lanes (2026-10-19)
`ghcr.io/dbirks/ninfer:20261019@sha256:6c4b2b6e...23d5` = upstream 64d10fbf + two fork commits, with the v3
artifact on `qwen38-ninfer-model-v3`, `--host-context-mib 0`, `--max-concurrency 16`; Collective backend 16.
- `20075061` fixes the 2026-10-17 engine fault `std::logic_error "KV write exceeds its page reservation"`
  (`src/models/common/context/kv_history.cpp` `prepare_write`). Cause: `settle_growth()` sizes the copy-on-write
  reservation one boundary early; with 2+ requests another history (e.g. a new prompt matching the same
  chat-template prefix) starts reading/protecting/pinning a page before the write, so more copies are needed
  than reserved. Fix: `can_resize_reservation`/`resize_reservation` from free pool pages; only throw if full.
- `8213c792`: `kMaximumConcurrency` 8 -> 16 (resident lanes) plus `kMaximumRoundBatch = 8`. Upstream's 8 is
  baked into ~25 ops (`batch > 8` checks), kernel tables (candidate selector, `GdnReplayFoldKernelRow row[8]`),
  per-batch CUDA graphs and workspace plans, so the fork does NOT raise the GPU batch. `reserve_resident_units`
  licenses at most 8 ready lanes per cycle (the oldest or a protected recovery always, the rest
  least-recently-executed first) and the others wait one cycle. The program uses `round_batch_limit()` for graph
  capture, GDN replay record capacity and per-batch workspace loops; lane buffers stay at max_concurrency
  (ops get `.slice(0,0,B)` views; KV/state go through `table_rows[b]`/`state_slots[b]`). qwen4_exp (Flash-Next)
  stays at 8. Effect: 16 admitted with no queue, but total decode speed is about the 8-wide rate shared by all 16.
- Verified on the card: bursts of 2/3/8/16 all 200; 16 concurrent 400-token generations 16/16 correct with
  `running 16 (decode-ready 16)`; vision OK; KV pool 1,461,568 tokens; no faults in early production traffic.
- Rollback (one commit): `ninfer:20260908@sha256:0a4bb65a...f6e` + `qwen38-ninfer-model` (v2) +
  `--host-kv-mib 0 --host-state-slots 0` + `--max-concurrency 8`, Collective back to 8.
- The fork has no CI tests run (Dockerfile builds with BUILD_TESTING=OFF); CI takes ~27 min. No upstream issue
  was filed (owner's choice).

## Speculative decoding (dflash2) acceptance, measured 2026-10-18
Each `req#N done` log line carries `dflash2 accepted A/P`. Over 7,229 production requests (agentic tool
calls, 100k-150k-token prompts, batch ~4): 35.0% of proposed drafts accepted, per-request p10/median/p90
27/37/51%, mean accept length (output / verify steps) **3.71** with `--draft-tokens 7`. z-lab publishes
4.10 (MT-Bench) to 5.46 (GSM8K) for this drafter, so we run below the benchmark but not broken. Thinking on vs off
and short vs long prompts made no difference (3.71 vs 3.68; 37% vs 36%). With ~2.7 of 7 drafts accepted per step, a
smaller K (4-5) might give more throughput at batch 4. Untested: measure with the bench before changing it.
Aggregate: `kubectl logs deploy/ninfer | grep -o 'dflash2 accepted [0-9,]*/[0-9,]*'`.
History: `--spec mtp --draft-tokens 3` until 2026-09-08, switched to dflash2 K=7 with the #121 image rebuild,
never A/B'd on our traffic. Upstream docs/performance/qwen3.8-27b.md (nvfp4, 75-request corpus): C=1 DFlash2
K=7 207.9 tok/s at 36.6% acceptance vs MTP3 168.6 tok/s at 59.6%, so our 35% is NORMAL for dflash2. MTP3 scales
to 440 tok/s at C=4 and 491 at C=8; DFlash2 is published only at C=1. Under load (batch 3-4) MTP3 may win: A/B it.
**2026-10-19 06:10 UTC: switched to MTP3 and kept it.** Compare with `scripts/spec-decode-stats.py`.
DFlash2 K=7 baseline (2,385 requests, 10-18 13:32 .. 10-19 06:09): 34.0% accepted, 3.75 tok/step, per-request
decode median 138 tok/s (p10 103, p90 185), TTFT median 1.40 s (corrected; light load, mostly batch 1-3), engine 128/208/281 tok/s at batch 1/2/3 (an earlier
sample showed 318-399 at batch 4). Raw log: spruce-goose ~/backups/ninfer-ab/. MTP3 frees VRAM: KV pool
1,490,688 tokens (vs 1,245,056 with dflash2) on the same 55000 MB slice.

## Prefill vs decode: strict 1:1 alternation (researched 2026-10-19)
Both the Sept build (`scheduler.h` `choose_execution`) and master (`engine_core.h`: one decode round, then one
`next_prefill` unit) alternate ONE prefill chunk with ONE decode round, one prefilling request at a time. No
knob sets the ratio, so `--prefill-chunk` (multiple of 128, default 1024) is the only lever. A 1024 chunk at 45k+
context takes ~0.3-0.4 s against a ~50 ms decode round, so during a big uncached prefill all 7 other lanes drop
to ~40-55 tok/s total. Measured: a prefill was running in 79% (DFlash2 log) and 98% (MTP log) of busy 5-s
windows; mostly small cache-hit suffixes costing ~25% (MTP at 7-8 decoding: 616 pure vs 474 mixed).
Upstream #277 (RTX 5090, 2.3k prompt): 512 = 2.91k, 1024 = 2.99k, 2048 = 3.02k prefill tok/s, ~10 ms per chunk
boundary, math-neutral. So 512 costs ~3% prefill speed and roughly doubles decode's share during big prefills.
APPLIED 2026-10-19 06:26 UTC: `--prefill-chunk 512`, MTP3 kept (no further A/B wanted), Collective ninfer
concurrency 15 (8 lanes + 7 queued). First minutes, >=4 decoding during a big prefill: decode 77 -> 217 tok/s,
prefill 3,070 -> 2,865 tok/s (-7%); small sample, re-check.
13 h later (9,389 requests, 06:26-19:41 UTC 10-19): acceptance 59.5%, 3.07 tok/step; per-request decode median 75
tok/s (p10 58, p90 111). Per-request speed by lanes decoding (no prefill / prefill running): 1: 110/70, 4: 83/18,
8: 58/32. A prefill was running in 90% of decode windows. Queue (concurrency 15 > 8 lanes): 42% of the time
someone waits; queue p50 0.05 s, p90 23 s, p95 46 s, max 284 s; TTFT median 2.2 s, p90 28 s.
NOTE: the stats script mis-parsed TTFT before 2026-10-19 20:00 (only "N ms" values counted); fixed, and old
logs recomputed. Logs in ~/backups/ninfer-ab/ on spruce-goose. Related open upstream: #303 (multi-chunk units when no decode
is waiting, i.e. faster TTFT when idle), #356/PR #355 (NVFP4 tiled prefill attention under-occupies the GPU; the
split-KV prefill work, 7f6aafed on 09-28, is NOT in our Sept build).

## Reasoning effort (Qwen3.8 v3 artifact template)
Accepted: `none` (engine thinking-off), `low`, `medium`, `xhigh` (template default). `minimal`, `high`,
`max` return 400 `invalid_prompt` ("Unexpected reasoning effort"). Declare only the accepted four.
Thinking BUDGET (runaway reasoning to max_tokens in long agent contexts): server-wide only via
`--default-thinking-budget N` (both builds; force-closes `</think>` with Qwen's early-close text, counted in
max_tokens). Chat Completions has NO per-request field (`thinking_budget` etc. silently ignored,
`chat_template_kwargs.thinking_budget` rejected); per-request only via Anthropic Messages
`thinking.budget_tokens`. Not set today. VERIFIED 2026-10-17 through the Collective gateway: POST
/v1/messages with `thinking: {type: enabled, budget_tokens: 1024}` capped at ~1,048 then answered
(the gateway rejects budget_tokens below 1024). Prefer that over the server-wide flag.

## Image + artifact are a matched pair
- Upstream Neroued/ninfer publishes NO image. ghcr.io/dbirks/ninfer is built by the fork's
  `.github/workflows/publish-ghcr.yml` on every push to master (tags `latest`, `YYYYMMDD`, `sha-*`).
- Update: clone dbirks/ninfer, `git remote add upstream https://github.com/Neroued/ninfer.git`,
  `git rebase upstream/master` (fork carries only CI + Dockerfile commits), push (force-with-lease).
  Commits are SSH-signed: if signing/push fails with no agent, `export SSH_AUTH_SOCK=/run/user/1000/ssh-agent.socket`.
- Upstream's Dockerfile has lagged its CMake deps before (2026-10: `liburing>=2.5` required, not
  installed -> build fails at cmake configure in ~90 s). Diff `cmake/Dependencies.cmake`
  `pkg_check_modules` lines against the Dockerfile apt lists; the fork's fix is `505ee46a`.
- Current master reads ONLY v3 `.ninfer` artifacts. v3 lives on its own PVC
  (`qwen38-ninfer-model-v3`, apps/qwen38-ninfer-weights-v3.yaml, HF rev a107ba1, SHA-pinned); the v2
  artifact stays on `qwen38-ninfer-model` as the rollback pair for `ninfer:20260908`. Never swap one
  without the other. Pin the image by digest.

## VRAM budget, four co-resident models (#156, 2026-10-17)
Card 97887 MiB (HAMi `nvidia.com/gpumem` units are MiB: a 9000 slice shows as 8.79 GiB in-pod).
NInfer 55000 + CLEF INT4 25000 + Qwen3-TTS Base 8000 + VoiceDesign 8000 = 96000; measured total ~89.7 GiB.
Bring big loaders up one at a time (host RAM bottomed at ~12.3 GB free with all four).

## Rollout gotcha: NO SURGE
A surge pod asks HAMi for a second full slice that cannot fit, so a default RollingUpdate hangs Pending.
Use `strategy: {type: RollingUpdate, rollingUpdate: {maxSurge: 0, maxUnavailable: 1}}` (~15s gap).
Do NOT switch to `type: Recreate` via Flux: SSA keeps the live `rollingUpdate` block, the dry-run fails
"spec.strategy.rollingUpdate: Forbidden ... Recreate", and `rollingUpdate: null` is stripped by kustomize.
When another whole-card pod is terminating, the new pod shows `FilteringFailed ... CardInsufficientMemory`
until the old slice is released (~grace period); that is expected, not a fault.

## Tools
- Upstream docs: github.com/Neroued/ninfer `docs/serving.md` (flag table, context-cache, metrics).
- Image has no curl; test from another pod: `kubectl run x --rm -i --restart=Never --image=curlimages/curl:8.11.1 -- curl http://ninfer.default.svc.cluster.local:8000/v1/models`.

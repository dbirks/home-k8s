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
Weights 21.4 GiB (21.7 with vision), runtime 30.5 GiB, engine ready in ~9.5 s. Vision costs only ~56K
tokens. Single-stream decode ~177 tok/s with `--spec dflash2 --draft-tokens 7 --lm-head-draft`.
Verify on the `capacity |` startup log line; `GET /metrics` exposes running/waiting/paused gauges.

## Concurrency is a compile-time 8
`--max-concurrency` accepts 1..8 only: `kMaximumConcurrency = 8` (`include/ninfer/types.h`) sizes fixed
arrays in the scheduler/round buffers/graph capture, and the Gated-DeltaNet replay kernel has its own
`kMaximumRows = 8`. Raising it is kernel work, not a flag. To admit more clients, raise the LLM
Collective backend `concurrency` above 8: extras wait in NInfer's pending FIFO (`--max-pending-requests`).

## Reasoning effort (Qwen3.8 v3 artifact template)
Accepted: `none` (engine thinking-off), `low`, `medium`, `xhigh` (template default). `minimal`, `high`,
`max` return 400 `invalid_prompt` ("Unexpected reasoning effort"). Declare only the accepted four.

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

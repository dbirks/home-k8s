---
name: ninfer-serving-memory
description: Size NInfer (qwen3.8-27b-ninfer, apps/ninfer-standalone.yaml) VRAM KV and host RAM on this RAM-poor, VRAM-rich node, and roll it out without hanging. Use when NInfer's host memory looks too big, when fitting a second model (e.g. Muse) beside it, or when changing its HAMi slice or context-cache flags.
---

# NInfer serving memory (VRAM KV vs pinned host tier)

## Mental model
- Active KV is ALWAYS in VRAM: one shared "Main Text KV" pool, sized at startup by `--kv-capacity auto`
  (fills the HAMi slice after weights + runtime, minus 1 GiB headroom). Retained prefixes live there too.
- The DEFAULT context cache ALSO pins host RAM as an overflow tier for inactive continuations:
  `Host KV = 8192 MiB` + `Host State = 8 slots` (~187MB each, the DeltaNet recurrent state) ~= 9.5GB.
  It is pinned (`/dev/zero (deleted)` shared mapping in smaps, `shmem` in memory.stat) and cannot be
  reclaimed. That is why the pod "uses 11GB": ~2GB engine anon + ~9.5GB pinned tier.
- This node is 46GB RAM, no swap, 96GB VRAM: the host tier is the wrong trade. Zero it and buy VRAM:
  ```
  --host-kv-mib 0 --host-state-slots 0 --device-state-slots 16
  ```
  (0 is legal for both; they only cannot be combined with `--no-prefix-reuse`.) Extra device state
  slots keep retained prefixes' recurrent state on-card. Evicted continuations recompute instead.

## Numbers (qwen3.8-27b nvfp4, int8 KV, 2026-09-24)
| HAMi slice | host tier | VRAM KV tokens | pod host RAM |
|---|---|---:|---:|
| 40000 MB | 8 GiB + 8 slots | 289,728 | ~11.7GB |
| 50000 MB | 0 | 553,600 | ~2GB |
~32KB/token int8 (16 full-attention layers). Verify on the `capacity |` and `context cache |` startup
log lines. A repeated prompt should log `cache N (99.9%, turn closure)` with TTFT ~30ms.

## VRAM budget with Muse co-resident
Card 97887 MB (HAMi register annotation `hami.io/node-nvidia-register`). NInfer 50000 + Muse 44000 =
94000. Leave a few GB slack; do not exceed the card or the second pod stays Pending.

## Rollout gotcha: NO SURGE
A surge pod asks HAMi for a second full slice that cannot fit, so a default RollingUpdate hangs Pending.
Use `strategy: {type: RollingUpdate, rollingUpdate: {maxSurge: 0, maxUnavailable: 1}}` (~15s gap).
Do NOT switch to `type: Recreate` via Flux: SSA keeps the live `rollingUpdate` block, the dry-run fails
"spec.strategy.rollingUpdate: Forbidden ... Recreate", and `rollingUpdate: null` is stripped by kustomize.

## Tools
- Upstream docs: github.com/Neroued/ninfer `docs/serving.md` (flag table, context-cache section).
- Image has no curl; test from another pod: `curl http://ninfer.default.svc.cluster.local:8000/v1/...`.

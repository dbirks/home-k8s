---
name: pennyroyal-flashnext
description: Deploy and operate Pennyroyal (jpezzulli/sglang-rtxpro6000, an SGLang fork) serving Qwen3.8-Flash-Next with NVMe-backed PLE and HiCache/NIXL prefix persistence on this 46GB-RAM, single RTX PRO 6000 node. Use when staging the RadixArk checkpoint, building the NVMe-PLE overlay, booting or tuning the pennyroyal-flashnext Deployment, or benchmarking agentic prefix reuse (issue 135).
---

# Pennyroyal Flash-Next on home-k8s

Manifests: `apps/pennyroyal-*.yaml(.hold)`, `apps/agentic-prefix-bench.yaml.hold`,
`apps/flashnext-radixark-*`. Issue #135 is the system of record (body = runbook, comments = priority).

## Facts that are easy to get wrong
- **Pin by index digest and verify it yourself.** The issue body quoted `sha256:1d71b02...` for v2.5.2;
  that is actually v2.5.0. Check with an anonymous GHCR token:
  `curl -s "https://ghcr.io/token?scope=repository:jpezzulli/sglang-rtxpro6000:pull"`, then
  `curl -sI -H "Authorization: Bearer $TOK" -H "Accept: application/vnd.oci.image.index.v1+json" https://ghcr.io/v2/jpezzulli/sglang-rtxpro6000/manifests/<tag>`
  and read `docker-content-digest`. Compare releases with `git diff pennyroyal-vA pennyroyal-vB -- configs/pennyroyal scripts/pennyroyal`.
- **Profile is the first ARG** (`next`), not an env var; the entrypoint rejects any extra args. All tuning
  is env. Hard-coded by the recipe: port 8001, served name `pennyroyal`, context 524,288, no API key.
- **Image USER is the name `penny`**: set `runAsUser/runAsGroup: 1000` or `runAsNonRoot` refuses to start.
- **local-path is hostPath**, so `fsGroup` is ignored: chown `/cache` `/nixl` (and the models tree) to
  1000 in an initContainer or the prep Job.
- **The NVMe-PLE overlay is absolute symlinks into the source.** Source and overlay must share one PVC
  mounted at the SAME path (`/models`) in the prep Job and the server, or the preflight fails.
  `prepare_ple_nvme.py --source ... --output ...` needs no GPU, streams 8 MiB chunks, refuses an existing
  output (skip when `ssd-stream.json` exists).
- **Keep `.cache/huggingface/download/*.metadata`** when copying the checkpoint (`rsync -a`). Without it
  the NIXL namespace helper re-hashes every safetensors file on every start. Moving the mount path or
  touching mtimes creates a new NIXL namespace (cold cache).
- **NIXL cleanup is whole-filesystem** (85%/80% watermarks in `nixl-posix-frspec.toml`, overridable via
  `NIXL_CONFIG`) plus the soft `SGLANG_HICACHE_NIXL_MAX_CACHE_GB` budget. On shared local-path that means
  it trims itself when the node disk passes 85%, before kubelet's 90% eviction.
- **No prompt data on disk (our ZDR requirement)**: the recipe hard-codes `--hicache-storage-backend
  nixl` (restart-persistent prefix cache on /nixl) with no env switch. The Deployment runs a /tmp copy of
  `configs/pennyroyal` with the three `--hicache-storage*` flags sed-stripped and grep-verified (fails
  closed). In-process GPU radix + host-RAM HiCache reuse still work; restart persistence is given up.
  /nixl is then a 64Mi RAM emptyDir (only namespace-identity.json lands there).
- **Host RAM**: upstream default = 47.68GiB pinned PLE + 32GB pinned HiCache, impossible at 46GB. Use
  `PENNY_PLE_BACKEND=nvme` and a modest `PENNY_HICACHE_SIZE_GB` (12 since 2026-10-26; ~10.4 GiB node MemAvailable left).
- **Driver**: toolchain is CUDA 13.3 (upstream qualified on driver 610.57.04) but the torch wheels are
  cu130, and the smoke Job PASSED on this node's 580.167.08 (2026-09-28): torch/matmul, sgl_kernel,
  flashinfer, triton imports, and a triton JIT kernel. Real-load nvcc-13.3 JIT is still unproven.
  Re-run `apps/pennyroyal-cuda-smoke.yaml.hold` after any driver or image bump (rename the Job each run;
  triton `@jit` needs a real .py file, not a stdin heredoc).
- **Pre-pull the 9.1GB image on the node.** GHCR's CDN stalls/resets single streams from this network,
  so the kubelet pull of the 5.4GB + 3.2GB layers times out (a Job can burn its whole deadline in
  ImagePullBackOff). Loop `talosctl image pull --namespace cri <image@digest>`: containerd resumes the
  partial layers and it converges (took 4 attempts).
- Whole card: the recipe uses `--mem-fraction-static 0.981`; scale ninfer and muse to 0 first.

## Measured on this node (2026-09-28, v2.5.3, A0: NVMe PLE, HiCache 8GB, NIXL off)
- Cold boot to ready ~7 min (weights 175s). KV 824,384 fp8 tokens; HiCache host pool 6.66GB + 1.4GB.
- Host RAM: node MemAvailable dips to ~8.3GB during weight load (pod WS 17.6GB peak), settles ~13.7GB.
  Do not co-schedule anything memory-heavy during a Pennyroyal boot.
- Pod memlock is 8192 KiB (containerd default); fine with NIXL off, revisit if io_uring/NIXL returns.
- Agentic bench (70K prefix, 6 turns): cold TTFT 19.0s, warm TTFT 0.43s median, decode 165-217 tok/s.
- Staging: rsync HDD to NVMe ~114MB/s (19 min for 137GB); prepare_ple_nvme.py took 2.5 min (48GB).

## Prefix-cache capacity under multi-agent load (2026-10-26)
Symptom: several agents with 190K-410K-token contexts (~1.17M total) take turns; cache hit collapses and
every turn re-prefills from the shared 13,440-token system prompt (ReqTimeStats `cached_input_len=13440`).
When the working set fits, hits are ~93% and TTFT ~0.2 s. Eviction is LRU, so it always drops the
context needed next. Analyse a log with ReqTimeStats (token-weighted `cached_input_len / input_len`,
queue/prefill ms) and `/metrics` deltas of `sglang:cached_tokens_total{cache_source=device|host}`.

What is and is not the bottleneck:
- **Mamba slots DO cap concurrency** (corrected 2026-11-02; this note used to say they were not).
  `resolve_max_num_reqs()` in `python/sglang/srt/mem_cache/kv_cache_configurator.py` sets
  `effective = min(MAX_RUNNING_REQUESTS, MAX_MAMBA_CACHE_SIZE // ratio)`. With
  `--mamba-radix-cache-strategy extra_buffer` + the overlap scheduler the ratio is **5** (3 slots per
  request + 2 ping-pong buffers); `extra_buffer_lazy` makes it 4 but is only qualified on Pennyroyal's
  27B profile, not Flash-Next. So `6 / 24` silently ran **4** (`/get_server_info` ->
  `internal_states[0].effective_max_running_requests_per_dp`; the boot log says "max_running_requests is
  capped to N by the mamba state cache"). `mamba usage` in decode logs reads low because live requests
  hold ~4 slots; admission reserves 5. Always set `MAX_MAMBA_CACHE_SIZE >= 5 x MAX_RUNNING_REQUESTS`.
  Cost: ~57 MB VRAM per slot (36 linear layers x 48 v-heads x 128 x 128 bf16 + conv state), ~7.5K fp8 KV
  tokens per slot out of the GPU pool. Measured: 6/24 -> token_capacity 855,616 (kvcache 10.405 GB);
  8/40 -> 722,496 (-133K, ~8.3K tokens per extra slot incl. per-request spec buffers; mamba ssm 2.16 GB +
  conv 0.08 GB) and then **FAILED TO BOOT (2026-11-02, reverted)**: `--hicache-size` is split by GPU pool bytes,
  so the host mamba pool grew 1.42 -> 2.36 GB (host KV 9.99 -> 9.08 GB), and `host_memory_budget_bytes()`
  (`pool_host/base.py`: `psutil.virtual_memory().available - 10 GiB` reserve, checked after the host KV pool is
  pinned) only had 1.35 GB -> `ValueError: Not enough host memory available. Requesting 2.36 GB`. 24 slots
  passed with ~0.07 GB to spare. Any raise needs ~1 GB more node MemAvailable at boot (stop another
  talos-210-73x pod) or a smaller `PENNY_HICACHE_SIZE_GB` (~10 for 8/40, costs host-L2 tokens), or
  `extra_buffer_lazy` (ratio 4, same 24 slots -> 6; unqualified on Flash-Next). Upstream: jpezzulli/sglang-rtxpro6000#5 (closed, by design);
  Pennyroyal v3.1.4 (2026-10-29) still defaults to 4 / 24 with the same ratio. Do NOT use
  `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` to shave the ratio: sgl-project/sglang#43204 (engine dies on
  "Can not alloc mamba cache").
- **The GPU KV pool is.** 824,384 fp8 tokens is a recipe **cap** (`configure_max_total_tokens 824384`,
  overridable with `MAX_TOTAL_TOKENS`), not the memory limit. Boot leaves ~4.06 GB VRAM free after CUDA
  graphs, and upstream qualified `MAX_TOTAL_TOKENS=1000000`. fp8 costs ~12.4 KB/token (target 9.44 GB +
  MTP draft 0.78 GB for 824,384), so 1,000,000 needs ~2.2 GB more and leaves ~1.9 GB. That is the
  quality-neutral next lever. Never lower it.
- **The host HiCache tier is inclusive, not additive.** Write-through backs up every GPU node, and only
  GPU-evicted leaves are host-evictable (`_is_host_leaf` requires `node.evicted`). So distinct cached
  tokens are roughly max(device, host), not device + host. The boot log warns when the host is smaller
  than the device pool. Host KV is ~13.3 KB/token (incl. the MTP layer) and gets ~83% of the setting, so
  parity with 824K device tokens is `PENNY_HICACHE_SIZE_GB` ~13, and covering a ~1.2M-token working set
  needs ~20. That is not affordable on 46 GB of RAM. Growing the GPU pool is the cheaper lever.
  `--hicache-size` is split across host KV (+QSA) and host mamba in proportion to their GPU pool bytes
  (`_split_hicache_size`). At 12: KV 750,656 tokens / 9.99 GB, mamba 1.42 GB, QSA 0.62 GB.
- **Scheduling:** `--schedule-policy lpm` (sed-injected, fails closed) admits the waiting request with
  the longest cached prefix first, instead of fcfs. Collective concurrency 8 (was 12) keeps fewer
  contexts in flight.

Applied 2026-10-26 (one restart): HiCache 8 to 12, lpm, contributor concurrency 12 to 8. Measured:
- Boot ~6 min to "fired up". KV still 824,384 fp8. Server args showed `schedule_policy: lpm`,
  `hicache_size: 12` and `hicache_storage_backend: None`.
- Node MemAvailable was 13.1 GiB steady before. It rose to ~27 GiB while the old pod was gone, never
  dipped during weight load, and fell to a 9.9 GiB minimum, ~10.4 GiB steady, once the pinned host pool
  was allocated. Shmem went from 8.6 to 12.25 GiB. The 2026-09-28 "dip to 8.3 GB" was measured with
  other pods resident.
- fp8 quality probe: math 8/8, executed code 4/4, tools 3/3 (incl. round trip), and a 3-needle test at
  256K tokens 3/3. Cold prefill 44 s; second turn 1.9 s with 256,000 cached. Spec accept length ~2.9-3.3.

**NVFP4 KV: evaluated, NOT applied (do not enable on this fork without a fix).**
- The access registry allows nvfp4 KV only with prefill=`flashinfer` (fp8 dequant workspace) and
  decode=`trtllm_mha` (XQA native FP4 on SM120). The default backend here is flashinfer for both, which
  fails `_handle_kv4_compatibility` and the decode access check, so you would need
  `--decode-attention-backend trtllm_mha`. It also needs `MAX_TOTAL_TOKENS` raised (~1,465,536 at the
  same bytes), or the pool stays capped at 824,384.
- The blocker: `pool_host/mha.py` (MHATokenToKVPoolHost) assumes unpacked KV. It sizes tokens as
  `head_num*head_dim*itemsize`, while FP4 rows are `head_dim/2` wide. `jit_transfer_hicache_one_layer`
  moves only `k_buffer/v_buffer`, never `k_scale_buffer/v_scale_buffer`. With write-through HiCache,
  every backup reads two tokens' worth and every host restore overwrites neighbours with stale scales:
  silent KV corruption. Quick quality probes only hit GPU-resident prefixes, so they would not catch it.
  Revisit only if upstream makes the HiCache host pool FP4-aware, or HiCache is disabled.

## Order of operations
1. RadixArk download to HDD staging (`flashnext-radixark-download-*`), then retire that Job to `.hold`.
2. Free NVMe space (user approves the list), unhold `pennyroyal-pvcs`.
3. Unhold `pennyroyal-stage-model` (rsync HDD to NVMe, tokenizer sha check, overlay prep), retire after.
4. CUDA smoke Job. 5. Scale ninfer + muse to 0, unhold `pennyroyal-flashnext`, watch `memlock=` line,
   `/health`, and `The server is fired up and ready to roll!`. 6. `agentic-prefix-bench` per arm.

## Exposure (post-teardown, 2026-10-10)
The llm.birks.dev KServe/Envoy/KEDA gateway stack is **gone** (history: issues #95/#135, teardown
research #144). Consumers connect to the Service directly: the LLM Collective contributor runs
`url: http://pennyroyal-flashnext.default.svc.cluster.local:8001` (protocols openai+anthropic,
concurrency 8 since 2026-10-26; SGLang runs 4 (24 mamba slots // 5) and queues the rest, the dev contributor runs 2). Do not write Backend/AIServiceBackend/KEDA manifests — those
CRDs no longer exist on the cluster.

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
  `PENNY_PLE_BACKEND=nvme` and `PENNY_HICACHE_SIZE_GB` in single digits.
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
concurrency 4 = MAX_RUNNING_REQUESTS). Do not write Backend/AIServiceBackend/KEDA manifests — those
CRDs no longer exist on the cluster.

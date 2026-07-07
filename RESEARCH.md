# Home K8s Cluster — Research Notes

Running notes from sessions researching and tuning this cluster. Most recent topics at the top.

---

## DSpark Speculator Training

### What is DSpark?

DSpark (Dynamic Sparse Approximation for Reasoning Knowledge) is a speculative decoding draft model
architecture from vllm-project/speculators. It's a successor to DFlash with a significantly higher
token acceptance rate — measured at ~34% better than DFlash on Qwen3-8B benchmarks
(DFlash: 0.309 → DSpark: 0.415 acceptance rate).

DSpark merged into the speculators repo on June 28, 2026, and into vLLM main July 1-2, 2026.
As of this writing, no pre-built Qwen3.6-27B DSpark checkpoint exists — only DFlash
(`z-lab/Qwen3.6-27B-DFlash`). Training one is the goal of `apps/dspark-train.yaml.hold`.

### Feasibility on Single RTX PRO 6000 (96GB)

Confirmed viable — VRAM breakdown:
- vLLM serving Qwen3.6-27B-NVFP4 at `--gpu-memory-utilization 0.35`: ~33.6 GB
- DSpark 3-layer draft model training (BF16 + Adam optimizer states): ~24–33 GB
- Total: ~57–67 GB / 96 GB, leaving ~29 GB headroom

Online training mode runs both vLLM and the trainer as separate processes sharing one GPU via
CUDA time-slicing. No CUDA MPS required.

### Training Job Design

See `apps/dspark-train.yaml.hold` for the full manifest. Key design decisions:

**No custom Docker image needed.** Uses `vllm/vllm-openai:v0.24.0` (already cached on the node)
as the base, installs `speculators` via pip at job start (~30s).

**Two concurrent processes in one container:**
- vLLM server (background): serves the target model, extracts hidden states
- DSpark trainer (foreground): reads hidden states, trains the draft model

**vLLM hidden state extraction** is enabled by `scripts/launch_vllm.py`, which wraps vllm serve
with these extra flags:
```
--speculative_config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[2,32,61,64]}}}'
--kv_transfer_config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/hidden_states"}}'
--no-enable-chunked-prefill
```

**Target layer IDs for Qwen3.6-27B (64 layers):** `2 32 61 64`
Formula: `[2, num_layers//2, num_layers-3, num_layers]`

**Online vs offline training modes:**
- Online (`--on-missing generate --on-generate delete`): trainer requests completions from vLLM,
  vLLM generates hidden states to `--hidden-states-path`, trainer reads and deletes immediately.
  No disk buildup. This is the mode in the job.
- Offline: generate all hidden states first (huge disk requirement), stop vLLM, train separately.
  Avoids VRAM contention but requires 500GB–3TB depending on dataset size.

### Confirmed Speculators Flags

```
pip install speculators           # PyPI package name (NOT vllm-speculators)

python scripts/train.py \
  --verifier-name-or-path unsloth/Qwen3.6-27B-NVFP4   # pre-quantized model works fine
  --vllm-endpoint http://localhost:8001/v1
  --speculator-type dspark
  --num-layers 3                   # draft model depth; 3=~1.4B params, 5=~2B params
  --block-size 8                   # tokens generated per speculative step
  --max-anchors 3072               # max hidden state sequence length
  --target-layer-ids 2 32 61 64   # must match vLLM launch exactly
  --draft-vocab-size 32000         # draft head vocabulary size
  --markov-rank 256                # rank of Markov transition approximation
  --markov-head-type vanilla       # vanilla | lm
  --enable-confidence-head         # adds a scoring head for token acceptance
  --confidence-head-with-markov    # confidence head uses Markov context
  --loss-fn '{"ce": 0.1, "tv": 0.9}'   # cross-entropy + total variation loss mix
  --confidence-head-alpha 1.0
  --total-seq-len 4096             # max sequence length during training
  --epochs 5
  --lr 3e-4
  --on-missing generate
  --on-generate delete
  --data-path /output              # root dir; hidden states go to /output/hidden_states
  --hidden-states-path /output/hidden-states
  --save-path /output/checkpoints
  --checkpoint-freq 0.25           # save every 25% of epoch (4 saves/epoch)
```

**Checkpoint resume:** automatic — train.py looks in `--save-path` on startup and resumes from
the latest checkpoint. `--no-resume-from-checkpoint` disables this.

### Qwen3.6-27B Architecture Notes for DSpark

- 64 transformer layers, hybrid DeltaNet+Attention architecture
- `Qwen3NextForCausalLM` implements `SupportsEagle3` ✅ (hidden state extraction confirmed working)
- Hidden states extracted at each transformer block output, regardless of layer type (DeltaNet or attention)
- `hidden_size = 5120`, `intermediate_size = 17408`
- Draft model at 3 layers: ~1.4B params; at 5 layers: ~2B params (matching z-lab's DFlash model)

### Estimated Training Time (Single RTX PRO 6000)

| Samples | Estimated Time | Notes |
|---------|---------------|-------|
| 5K      | 1.5–3 hrs     | Sanity check only (5.9% acceptance rate) |
| 50K     | 15–30 hrs     | Usable quality |
| 200K    | 60–120 hrs    | Production-grade |

Bottleneck is vLLM throughput for hidden state generation, not GPU compute for training.

---

## vLLM Semantic Router v0.3 (Themis)

### Architecture

- **extproc**: Envoy External Processor (ExtProc) — intercepts every request before it reaches vLLM
- **mmBERT-32K classifiers**: multimodal BERT variant, classifies request domain and complexity
- **~50ms P50 overhead** per request for classification
- Routing signals: domain (mmBERT), complexity (easy/medium/hard), keywords, embeddings, PII, jailbreak
- Admin dashboard at port 8700

### Our Setup

`vllm.hoam.lan` → nginx ingress → `semantic-router:8080` → either:
- `vllm:80` (Qwen3.6-27B, 0.60 GPU util) — default/complex requests
- `vllm-small:80` (Qwen3.5-4B, 0.28 GPU util) — future routing target

Dashboard: `semantic-router.hoam.lan`

Current routing: all traffic to large model (safe default). Add complexity/domain
routing decisions via the dashboard to route simple queries to Qwen3.5-4B.

### Key Container Images (v0.3.0)

```
ghcr.io/vllm-project/semantic-router/extproc:v0.3.0         # router
ghcr.io/vllm-project/semantic-router/dashboard:v0.3.0       # admin UI
ghcr.io/vllm-project/semantic-router/operator:v0.3.0        # K8s operator (not used)
```

Helm chart: `oci://ghcr.io/vllm-project/charts/semantic-router` version `0.3.0`

### Config Structure

The chart config is pure YAML rendered into a ConfigMap. Key sections:

```yaml
config:
  providers:
    models:
      - name: "main"           # model name clients use
        provider_model_id: "main"  # forwarded to backend
        backend_refs:
          - name: "qwen27"
            endpoint: "vllm:80"    # in-cluster service
            protocol: "http"
  routing:
    signals:
      complexity:
        - name: "my_rule"
          threshold: 0.15
          hard: { candidates: ["...complex examples..."] }
          easy: { candidates: ["...simple examples..."] }
    decisions:
      - name: "route-simple"
        priority: 90
        rules:
          operator: "AND"
          conditions:
            - type: "complexity"
              name: "my_rule"
        modelRefs:
          - model: "qwen4b"
```

---

## vLLM v0.24.0 Configuration (Qwen3.6-27B)

### Key Flags

```yaml
- "--gpu-memory-utilization=0.60"     # was 0.90; reduced for semantic router + small model
- "--kv-cache-dtype=fp8_e4m3"         # e5m2 does NOT work (incompatible with compressed-tensors)
- "--mamba-cache-dtype=float16"        # DeltaNet recurrent state must be float16, not FP8
- "--mamba-ssm-cache-dtype=float16"    # same
- "--compilation-config"
- '{"cudagraph_mode":"PIECEWISE"}'     # FULL_DECODE_ONLY incompatible with FlashInfer + spec-decode
- "--speculative-config"
- '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":16}'
```

### Key Environment Variables

```yaml
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: "1"    # FlashInfer CUTLASS NVFP4 MoE kernel (vs Marlin fallback)
VLLM_FLASHINFER_MOE_BACKEND: latency          # latency-optimized selection
FLASHINFER_CUDA_ARCH_LIST: 12.0f              # explicit SM120 targeting for JIT
FLASHINFER_FORCE_SM: 120f                      # same
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True  # reduces allocator fragmentation
```

### Known Working Configurations

- **Quantization**: NVFP4 (`unsloth/Qwen3.6-27B-NVFP4`) — leverages SM120 FP4 tensor cores
- **KV cache**: fp8_e4m3 works; fp8_e5m2 does NOT; TurboQuant does NOT (hybrid attention)
- **Mamba dtype**: must be float16 (DeltaNet recurrent state incompatible with FP8)
- **Compilation**: PIECEWISE required; FULL_DECODE_ONLY breaks FlashInfer+speculative
- **Speculative decoding**: DFlash with 16 tokens currently; DSpark pending training

### Pitfalls Encountered

- `cudagraph_capture_size` is NOT a valid CompilationConfig field in v0.24.0 → pydantic crash
- `"dynamic": true` is NOT a valid SpeculativeConfig field in v0.24.0 → pydantic crash
- NVFP4 on SM120 has known kernel bugs with MoE models — use dense models only
- TurboQuant KV cache does NOT work with DeltaNet/Mamba hybrid models
- `enableServiceLinks: false` required on vLLM pods (K8s service named "vllm" injects VLLM_PORT)

---

## Hardware: NVIDIA RTX PRO 6000 Blackwell

- Architecture: SM120 (Blackwell)
- VRAM: 96GB
- vBIOS: 98.02.81.00.07
- Features: FP4 tensor cores (NVFP4), MIG support (confirmed by vBIOS ≥ 98.02.55.00.00)

### MIG Profiles Available

| Profile | Instances |
|---------|-----------|
| 1g.24gb | ×4        |
| 2g.48gb | ×2        |
| 4g.96gb | ×1        |

MIG is available but complex to set up on Talos. Not currently configured.

### Talos Config

- Schematic: `036d341b186bfa76a1c0a545125bbd667908a09a50dfe5e7ab32cc93901b84a2`
- Uses `nvidia-open-gpu-kernel-modules` (required for Blackwell)
- Version: Talos Linux v1.12.6

---

## vllm-project Organization — Other Notable Projects

### speculators

`github.com/vllm-project/speculators` — trains speculative decoding draft models.
Implements DFlash (attention-pattern caching) and DSpark (dynamic sparse approximation).
DSpark merged June 28, 2026. Training a DSpark checkpoint for Qwen3.6-27B is the goal.

### llm-d

Kubernetes-native LLM inference distribution system. Key features:
- **Disaggregated prefill/decode**: separate pods for prefill (CPU-bound) and decode (GPU-bound)
- **KV cache sharing** between pods via RDMA/NVLink
- **Inference gateway** with request scheduling

Not relevant for single-node setup — designed for multi-node inference clusters.

### KServe

Production ML serving platform for Kubernetes. Supports multiple frameworks (vLLM, TGI, Triton).
Adds: autoscaling, canary deployments, inference graphs, A/B testing.
Overkill for single-node home lab but useful at scale.

### vllm-router (AIBrix)

Request-level load balancer for multiple vLLM instances. Routes based on:
- KV cache affinity (send to the instance most likely to have the prefix cached)
- Load metrics (GPU utilization, queue depth)
- Model availability

Less sophisticated than Semantic Router but lower overhead (~0ms vs ~50ms).
AIBrix is the broader project (Bytedance); vllm-router is the routing component.

### ClawOS

Experimental multi-agent orchestration with KV cache sharing between agents.
Not production-ready. Research project exploring cooperative inference.

---

## llama.cpp MoE CPU Offload Flags

For running MoE models (Qwen3 235B, DeepSeek-V3, etc.) with CPU-offloaded experts:

```bash
--cpu-moe                 # offload ALL expert layers to CPU
--n-cpu-moe N             # offload first N MoE layers to CPU
-ot "blk\.([0-9]+)\.ffn_.*=CPU"  # regex: offload expert layers by pattern
```

The `-ot` flag is the most flexible — matches tensor names by regex, lets you offload
specific layers or expert indices. Useful when GPU VRAM is insufficient for the full MoE
weight set but you want some experts on GPU.

### LongCat-2.0 (48B active on 96GB)

48B active parameters from a 235B total model. On 96GB VRAM:
- Full BF16: ~88GB for active params alone — too tight with overhead
- Requires 4-bit quantization (Q4_K_M) or NVFP4 to fit comfortably
- llama.cpp with `-ot` to offload cold experts to CPU is a viable approach
- vLLM support: check if NVFP4 quantization is available for the specific checkpoint

---

## Omnigent SDK

`github.com/omnigent-ai/omnigent` — orchestration platform for coding agents.

- **REST API**: `POST /v1/sessions` returns session ID immediately (fire-and-forget)
- **CLI**: `omnigent run` blocks until completion; no built-in detach flag
- **Languages**: Python SDK official; Rust/Go/JS via raw REST API
- **Fire-and-forget from agents**: Use `POST /v1/sessions` directly, capture session ID,
  poll `/v1/sessions/{id}` for status, OR use `omnigent host` which runs persistently

Current cluster version: v0.4.0 (`ghcr.io/omnigent-ai/omnigent-server:v0.4.0`)

---

## Model Notes

### Qwen3.6-27B

- Hybrid architecture: DeltaNet (linear attention) + standard attention layers
- 77.2% SWE-bench Verified — outperforms much larger models including Qwen3-Coder-Next 80B
- NVFP4 quantization: leverages Blackwell FP4 tensor cores natively
- DFlash speculative decoding: `z-lab/Qwen3.6-27B-DFlash`, 16 speculative tokens
- DeltaNet recurrent state requires float16 KV cache (not FP8)

### Qwen3.5-4B-Instruct (small model)

- Standard transformer (no hybrid layers) — simpler than Qwen3.6
- BF16, 28% GPU util (~26.9GB VRAM)
- Serves as fast backend for the semantic router's "easy" routing decisions
- `--max-model-len=32768`, `--max-num-seqs=8`

### DSpark Acceptance Rate vs DFlash

| Model | Method | Acceptance Rate |
|-------|--------|----------------|
| Qwen3-8B | DFlash  | 0.309 |
| Qwen3-8B | DSpark  | 0.415 |
| Improvement | +34% | — |

No published DSpark checkpoint for Qwen3.6-27B exists yet (as of July 2026).

---

## Networking / DNS

- DNS: Pi-hole at 10.0.0.202 (MetalLB LoadBalancer)
- external-dns watches Ingresses → creates Pi-hole DNS records (v6 API)
- Domain: `*.hoam.lan`
- Tailscale subnet router: advertises 10.0.0.0/24 for remote access

### Active Service Endpoints

| URL | Service |
|-----|---------|
| vllm.hoam.lan | Semantic router → Qwen3.6-27B or Qwen3.5-4B |
| speech.hoam.lan | granite-speech sidecar (STT) |
| semantic-router.hoam.lan | Router dashboard |
| omnigent.hoam.lan | Omnigent orchestration server |
| openhands.hoam.lan | OpenHands agent canvas |
| pihole-webui.hoam.lan | Pi-hole admin |
| uptime.hoam.lan | Uptime Kuma |
| jellyfin.hoam.lan | Jellyfin media server |

---

## Flux GitOps Notes

### Reconcile Chain

```
prereqs → infra → apps
```

To force a full reconcile after pushing:
```bash
kubectl annotate gitrepository -n flux-system flux-system \
  reconcile.fluxcd.io/requestedAt="$(date -u +%Y-%m-%dT%H:%M:%SZ)" --overwrite
kubectl annotate kustomization -n flux-system prereqs infra apps \
  reconcile.fluxcd.io/requestedAt="$(date -u +%Y-%m-%dT%H:%M:%SZ)" --overwrite
```

### OCI HelmRepository (Flux v2.8+)

Must use API v1 (not v1beta2):
```yaml
apiVersion: source.toolkit.fluxcd.io/v1
kind: HelmRepository
spec:
  type: oci
  url: oci://ghcr.io/...
```

### Suspended / Held Apps

- `.yaml.hold` suffix: Flux ignores the file entirely
- `replicas: 0`: Flux deploys but scales to zero
- `dspark-train.yaml.hold`: training job, apply manually when ready

---

## SOPS / Secrets

- Encryption: SOPS + age
- Config: `.sops.yaml` in repo root
- Encrypted files: `.enc.yaml` suffix
- Age key: `~/.config/sops/age/keys.txt`

After cluster wipe:
```bash
kubectl create secret generic sops-age --namespace=flux-system \
  --from-file=age.agekey=$HOME/.config/sops/age/keys.txt
```

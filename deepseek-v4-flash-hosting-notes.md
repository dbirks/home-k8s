# DeepSeek V4 Flash: Hosting Notes for Single High-VRAM GPU

> Research compiled July 2026. Covers model identity, quantizations, vLLM/SGLang configs,
> Blackwell SM120 caveats, performance vs Qwen3.6-27B, and speculative decoding support.
> Targeted at single-card 96 GB deployments (RTX PRO 6000 Blackwell / H100 SXM5).

---

## TL;DR for This Setup (Single RTX PRO 6000 Blackwell, 96 GB, SM120)

**Short answer: Not viable as a drop-in replacement for Qwen3.6-27B on a single 96 GB card.**

- Native checkpoint is ~170–175 GB; a single 96 GB card cannot hold it.
- Q4 quantization (~86–96 GB weights) technically fits, but leaves almost no room for KV
  cache and incurs measurable quality regression.
- SM120 (Blackwell workstation) is not officially supported by vLLM or SGLang for
  DeepSeek V4; community patches exist but are not merged upstream as of July 2026.
- On coding benchmarks Qwen3.6-27B outperforms V4 Flash substantially — the model you
  are already running wins on the tasks you care about.

---

## 1. What Is DeepSeek V4 Flash?

| Property | Value |
|---|---|
| Release date | 24 April 2026 |
| Developer | DeepSeek (deepseek-ai) |
| License | MIT |
| Architecture | Sparse Mixture-of-Experts (MoE) |
| Total parameters | 284 B |
| Active parameters per token | 13 B |
| Expert configuration | 256 routed + 1 shared; top-6 routing |
| Context window | 1 M tokens |
| Max output tokens | 384 K |
| Native precision | FP4 (MoE expert weights) + FP8 (attention, router, norms) |
| Disk size (instruct checkpoint) | ~158 GB |

DeepSeek V4 Flash is the efficiency tier of the V4 family, released alongside the larger
V4 Pro (1.6 T parameters). "Flash" does not mean a small model — at 284 B total params it
is a substantial MoE that requires server-class memory at any reasonable precision.

### Architecture Innovations

- **Hybrid Compressed Sparse Attention (CSA + HCA):** Reduces single-token FLOPs to 27%
  and KV cache to ~10% of V3.2 at 1 M-token context. This is what enables the 1 M context
  window without prohibitive memory cost.
- **Manifold-Constrained Hyper-Connections:** Replaces standard residual connections to
  improve signal stability across layers.
- **Muon Optimizer:** Enables faster convergence during training.
- **Training corpus:** 32 T+ tokens; post-training uses GRPO + on-policy distillation.

### Relationship to Other DeepSeek Models

```
DeepSeek-V3  (671 B MoE, 37 B active, 128 K ctx)   — predecessor
DeepSeek-V4-Flash (284 B MoE, 13 B active, 1 M ctx) — efficiency variant
DeepSeek-V4-Pro  (1.6 T MoE, 1 M ctx)               — flagship
```

Sources:
- [HuggingFace model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Spheron deployment guide](https://www.spheron.network/blog/deploy-deepseek-v4-flash-gpu-cloud/)
- [Codersera complete guide](https://codersera.com/blog/deepseek-v4-complete-guide-2026/)

---

## 2. Quantizations and VRAM Requirements

### V4 Flash VRAM Table

| Quantization | Weight Size | Minimum VRAM | Comfortable VRAM | Notes |
|---|---|---|---|---|
| BF16 | ~568 GB | ~640 GB | — | Not practical |
| FP8 full | ~284 GB | ~320 GB | ~384 GB | 4× H100 80 GB minimum |
| **FP4+FP8 native** | **~158 GB** | **~170–175 GB** | **~210 GB** | **Official checkpoint format** |
| Q6 GGUF | ~120 GB | ~160 GB | ~192 GB | 2× H100 or 1× H200 |
| Q5 GGUF | ~100 GB | ~128 GB | ~160 GB | |
| Q4_K_M GGUF / INT4 AWQ | ~80 GB | **~86–96 GB** | ~128 GB | Single 96 GB card — tight |
| Q3 GGUF | ~60 GB | ~80 GB | ~96 GB | Significant quality loss |
| Q2 GGUF | ~40 GB | ~48 GB | ~64 GB | Not recommended |

**Critical note on active parameters vs memory:** Despite only 13 B active parameters
per forward pass, all 284 B weights must reside in VRAM simultaneously — the router can
select any expert, so no weights can be left off-card.

### Single 96 GB Card Feasibility

The native FP4+FP8 checkpoint (~158 GB on disk, ~170–175 GB runtime) does not fit in 96 GB.

At Q4_K_M (~86–96 GB), the weights technically fit but:
- Almost no headroom remains for KV cache (1–8 GB max).
- 128 K context is the practical ceiling; 1 M context is impossible.
- Performance: community reports 45–60 tok/s at Q4_K_M on a single 96 GB card.
- Quality: approximately 5% degradation vs native FP4+FP8 across reasoning tasks;
  larger gaps on math and structured code.

The minimum sensible configuration for native quality is **2× RTX PRO 6000 (192 GB)**
running the FP4+FP8 checkpoint at TP=2.

Sources:
- [Knightli VRAM table](https://knightli.com/en/2026/05/01/deepseek-v4-local-vram-quantization-table/)
- [Codersera VRAM guide](https://codersera.com/blog/deepseek-v4-vram-gpu-requirements-2026/)
- [Spheron GPU recommender](https://www.spheron.network/tools/gpu-recommender/deepseek-ai/DeepSeek-V4-Flash/)
- [Compute Market hardware guide](https://www.compute-market.com/blog/deepseek-v4-flash-local-hardware-guide-2026)

---

## 3. vLLM and SGLang Configurations

### Requirements

| Framework | Minimum Version |
|---|---|
| vLLM | ≥ 0.9.0 |
| SGLang | ≥ 0.4.4 |
| CUDA | 12.4+ |
| Python | 3.10+ |

### Official vLLM Recipe (Single Large GPU, e.g. MI355X or GB200 288 GB)

From [recipes.vllm.ai](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash). Note: the
`--tensor-parallel-size 1` here targets datacenter chips with ≥ 288 GB VRAM, not a
workstation RTX PRO 6000.

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --block-size 256 \
  --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-cudagraph-capture-size 128 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

### vLLM Multi-GPU (4× H100/H200, Production)

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --dtype fp8 \
  --kv-cache-dtype fp8_e5m2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill
```

For full 1 M context on 4× H200:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --dtype fp8 \
  --kv-cache-dtype fp8_e5m2 \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --max-num-seqs 16
```

### SGLang (4× GPUs)

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --enable-ep \
  --dtype fp8
```

Or with explicit quantization flag:

```bash
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 \
  --quantization fp4_mixed \
  --context-length 32768 \
  --port 8100
```

### Recommended Sampling Parameters

From the official model card: `temperature=1.0, top_p=1.0`.
For maximum reasoning mode: `--max-model-len >= 393216` (384 K tokens minimum).

Sources:
- [Official vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
- [Spheron deployment blog](https://www.spheron.network/blog/deploy-deepseek-v4-flash-gpu-cloud/)
- [Codersera local setup guide](https://codersera.com/blog/run-deepseek-v4-flash-locally-full-2026-setup-guide/)

---

## 4. Blackwell SM120 Caveats (RTX PRO 6000 / RTX 5090)

This is the most important section for this setup.

### Root Cause

DeepSeek V4 Flash depends on two kernel families that have no SM120 implementation:

1. **DeepGEMM** (`hyperconnection.hpp:56`): `Assertion error: Unsupported architecture`
   - The `tf32_hc_prenorm_gemm` and `paged_mqa_logits` kernels only target SM90/SM100
     (H100/H200). Setting `VLLM_USE_DEEP_GEMM=0` does not resolve the issue — the
     failure extends beyond the DeepGEMM opt-in path.
   - DeepGEMM developers indicated no near-term SM120 plans as of mid-2026.

2. **FlashMLA sparse-decode kernels**: Stock SGLang image ships SM90/SM100 binaries only.
   Error: `RuntimeError: Unsupported architecture for sparse decode fwd`

3. **torch.compile inductor failure** (vLLM vanilla): `auto_functionalized was not removed`
   during the `profile_run` phase. Warning also logged: `Device capability 12.0 not
   supported, communicator is not available`.

### Official Status

DeepSeek V4 Flash officially supports: A100, H100, H200, B300 (Blackwell datacenter).
SM120 workstation cards (RTX PRO 6000, RTX 5090) are **not** officially supported.

Relevant upstream tracking:
- [vLLM issue #40802](https://github.com/vllm-project/vllm/issues/40802) — SM120 cannot run DeepSeek V4
- [vLLM issue #40928](https://github.com/vllm-project/vllm/issues/40928) — Triton fallback feature request
- [DeepGEMM issue #317](https://github.com/deepseek-ai/DeepGEMM/issues/317) — SM120 kernel gaps

### Community Workaround 1: vLLM Community Fork (jasl/vllm, `ds4-sm120-preview` branch)

vLLM PR [#41834](https://github.com/vllm-project/vllm/pull/41834) implements SM12x support
but was not merged upstream as of late June 2026. Use the preview branch:

```bash
git clone https://github.com/jasl/vllm.git
cd vllm
git checkout ds4-sm120-preview

export DEEPGEMM_SRC_DIR=/path/to/DeepGEMM
MAX_JOBS=64 pip install --no-build-isolation -e . --verbose
```

Launch with required env vars (CUDA graphs are essential — without them throughput
collapses from ~30 tok/s to ~5 tok/s):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_TRITON_MLA_SPARSE=1 \
VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=256 \
VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=128 \
VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1 \
  vllm serve /mnt/models/deepseek-v4-flash \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.93 \
    --max-model-len 262144
```

What PR #41834 fixes under the hood:
- Sparse MLA Triton kernels for SM120/SM121 (avoids unreleased FlashInfer dependency)
- Persistent top-k streaming path replacing cooperative-radix (SM120 has <128 KB shared mem)
- Triton recompilation memory-leak fix (per-shape `constexpr` → runtime args)
- FP8 einsum alignment restoring ~24% decode performance at long context

Optional gate env vars:
- `VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE` — decode optimization toggle
- `VLLM_DEEPSEEK_V4_FLASHINFER_SM120_PREFILL` — prefill optimization (default off)
- DSpark spec decoding: add `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`

**Caveat:** The fork is not synced with upstream vLLM releases. Any upstream DeepGEMM
change may break it.

### Community Workaround 2: SGLang Docker Patch (0xSero/deepseek-v4-flash-sm120)

Repository: [github.com/0xSero/deepseek-v4-flash-sm120](https://github.com/0xSero/deepseek-v4-flash-sm120)

Build the kernel extension patch inside the official SGLang Docker image:

```bash
git clone https://github.com/0xSero/deepseek-v4-flash-sm120.git
cd deepseek-v4-flash-sm120
scripts/build_in_sglang_docker.sh
# Generates kernel extension and sitecustomize.py in build-docker/
```

Launch (model from sgl-project/DeepSeek-V4-Flash-FP8):

```bash
MODEL_DIR=/mnt/llm_models/DeepSeek-V4-Flash-FP8 PORT=8000 \
  scripts/launch_dsv4_flash_sm120.sh
```

Or manually with Docker:

```bash
docker run --gpus all \
  --shm-size=64g --ipc=host \
  -v /mnt/models/deepseek-v4-flash:/workspace/model:ro \
  -v ./build-docker:/dsv4:ro \
  -e PYTHONPATH=/dsv4 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -p 8000:8000 \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --model-path /workspace/model \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend compressed \
    --cuda-graph-max-bs 32 \
    --tool-call-parser deepseekv4 \
    --max-total-tokens 524288 \
    --port 8000
```

Key SGLang flags for SM120:
- `--attention-backend compressed` — engages CSA/HCA kernels, avoids flash-attention SM120 crash
- `--kv-cache-dtype fp8_e4m3` — note: FP8 e4m3 (not e5m2, which is incompatible with compressed-tensors)
- `--cuda-graph-max-bs 32` — enables CUDA graphs (required for acceptable throughput)

Optional EAGLE speculative decoding with this patch:

```bash
--speculative-algorithm EAGLE \
--speculative-num-steps 1 \
--speculative-topk 1 \
--speculative-num-draft-tokens 2
```

**Do not** use custom Jinja chat templates — SGLang has built-in DeepSeek V4 encoding.
**Do not** disable CUDA graphs.

### Known Invalid Flags (Do Not Use on SM120)

```
VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION  # does not exist
VLLM_ATTENTION_BACKEND=FLASH_ATTN_2          # breaks SM120
```

Do not allow `support_deep_gemm()` to return `True` for SM120 — it should only activate
for SM90/SM100.

### SM120 Performance Numbers (Community-Reported)

| Hardware | Config | Single-stream decode | Prefill | Notes |
|---|---|---|---|---|
| 2× RTX PRO 6000 (TP=2) | Community vLLM fork, FP8 | ~60 tok/s | 1638 tok/s | PR #41834 validated |
| 4× RTX PRO 6000 (TP=4) | SGLang patch | 25–35 tok/s | 800–1400 tok/s | 2K–32K prompt |
| 4× RTX PRO 6000 (TP=4) | SGLang + EAGLE spec decode | ~45 tok/s avg | — | 0xSero patch |
| 8× RTX PRO 6000 (TP=8) | Community vLLM fork + CUDA graphs | 30–35 tok/s | — | Without CUDA graphs: ~5 tok/s |
| 4× RTX PRO 6000 @ 300K ctx | SGLang patch, max-thinking | — | 646 tok/s | 15 decode tok/s (long context) |

**Note:** None of these benchmarks use a single 96 GB card — they all use 2–8× RTX PRO 6000.

Sources:
- [HuggingFace discussion #15 (DEEP_GEMM error)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/discussions/15)
- [HuggingFace discussion #28 (SM120 vLLM)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/discussions/28)
- [vLLM PR #41834](https://github.com/vllm-project/vllm/pull/41834)
- [vLLM issue #40802](https://github.com/vllm-project/vllm/issues/40802)
- [DeepGEMM issue #317](https://github.com/deepseek-ai/DeepGEMM/issues/317)
- [0xSero SM120 patch repo](https://github.com/0xSero/deepseek-v4-flash-sm120)
- [Codersera 4× RTX PRO 6000 guide](https://codersera.com/blog/deepseek-v4-flash-rtx-pro-6000-blackwell-benchmarks-2026/)
- [NVIDIA Developer Blog (Blackwell)](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/)

---

## 5. Speculative Decoding

### MTP (Multi-Token Prediction) — Built In

V4 Flash includes MTP heads natively. No draft model needed.

```json
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

This is the lowest-friction option and is included in the official vLLM recipe.

### DSpark — Released 27 June 2026

DeepSeek open-sourced DSpark under MIT license on June 27, 2026. It is a new speculative
decoding framework specifically built for DeepSeek V4.

**How it works:** Combines a parallel draft backbone (DFlash) with a lightweight sequential
Markov head (rank-256 low-rank factorization) to generate 5-token draft blocks before
verification.

**Claimed speed gains over MTP-1 baseline:**
- V4 Flash: 60–85% faster per-user generation
- V4 Pro: 57–78% faster per-user generation
- Acceptance length vs Eagle3: +26–31%
- Acceptance length vs DFlash: +16–18%
- Code generation acceptance: naturally high (code is structured and predictable)

**Integration with vLLM (via PR #41834 / community fork):**

```json
--speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

**Training framework:** DeepSpec (also MIT-licensed). Training assumes 8 GPUs minimum.
The Qwen3-4B draft model cache requires ~38 TB for training — self-training is not
practical for home setups.

**Caveat:** All performance numbers are from DeepSeek's own benchmarks as of late June
2026. No independent third-party verification has been published yet. Real-world gains
on single-user workstation workloads may differ from DeepSeek's multi-user production
benchmarks.

**DSpark vs DFlash:** DSpark supersedes DFlash. The "DeepSpec" framework bundles all
three algorithms (DSpark, DFlash, Eagle3) but DSpark is the recommended default.

Sources:
- [MarkTechPost DSpark announcement](https://www.marktechpost.com/2026/06/27/deepseek-releases-dspark-a-speculative-decoding-framework-that-accelerates-deepseek-v4-per-user-generation-60-85-over-mtp-1/)
- [ChatForest DSpark builder guide](https://chatforest.com/builders-log/deepseek-dspark-speculative-decoding-v4-builder-inference-guide/)
- [ExplainX DSpark guide](https://explainx.ai/blog/deepseek-dspark-v4-speculative-decoding-deepspec-guide-2026)

---

## 6. Performance vs Qwen3.6-27B for Coding

| Benchmark | DeepSeek V4 Flash | Qwen3.6-27B | Winner |
|---|---|---|---|
| SWE-bench Verified | 79.0 | 77.2 | V4 Flash (+1.8) |
| SWE-bench Pro | 52.6 | 53.5 | Qwen3.6 (+0.9) |
| LiveCodeBench | 55.2% | 83.9% | **Qwen3.6 (+28.7)** |
| Coding category avg | 57.1 | 70.6 | **Qwen3.6 (+13.5)** |
| Math avg | 40.8 | 89.2 | **Qwen3.6 (+48.4)** |
| Agentic avg | 49.1 | 59.3 | **Qwen3.6 (+10.2)** |
| Knowledge avg | 39.2 | 53.6 | **Qwen3.6 (+14.4)** |

**Summary:** Qwen3.6-27B wins on every coding benchmark except SWE-bench Verified (where
V4 Flash leads by 1.8 points). The LiveCodeBench gap (28.7 points) is especially notable
for interactive coding. V4 Flash's only meaningful advantage is its 1 M-token context
window vs Qwen3.6-27B's 262 K.

For a single-card coding assistant, Qwen3.6-27B with DFlash speculative decoding
(already running) is the better model. V4 Flash would only make sense if long-context
tasks (>262 K tokens) become a requirement — which then needs a second 96 GB card anyway.

Sources:
- [BenchLM comparison](https://benchlm.ai/compare/deepseek-v4-flash-vs-qwen3-6-27b)
- [LLMReference comparison](https://www.llmreference.com/compare/deepseek-v4-flash/qwen3.6-27b)
- [Kilo.ai open coding models](https://kilo.ai/open-source-models)

---

## 7. Tensor Parallelism Requirements

| Config | VRAM Required | Min Hardware | Notes |
|---|---|---|---|
| TP=1 (single card) | ~96 GB | 1× RTX PRO 6000 | Q4_K_M only; SM120 patch required; no headroom for KV |
| TP=2 | ~192 GB | 2× RTX PRO 6000 | Native FP4+FP8 fits; 256 K–512 K context viable |
| TP=4 | ~384 GB | 4× RTX PRO 6000 | Recommended for 512 K–1 M context |
| TP=4 | ~320 GB | 4× H100 SXM5 | FP8, max 32 K context (only ~36 GB free for KV) |
| TP=4 | ~456 GB | 4× H200 SXM5 | FP8, full 1 M context; recommended production tier |

The RTX PRO 6000 has no NVLink — all multi-GPU communication goes over PCIe. This causes
a 30–40% throughput reduction vs NVLink setups (e.g. H200 SXM5). Expert-parallelism
(`--enable-expert-parallel` in vLLM) partially mitigates this by localizing MoE routing
traffic but cannot overcome the PCIe bandwidth ceiling.

---

## 8. Practical Decision Guide for This Setup

### Current Setup
- Single RTX PRO 6000 Blackwell (96 GB, SM120)
- Running Qwen3.6-27B-NVFP4 with DFlash speculative decoding on vLLM v0.24.0
- Talos Linux K8s cluster

### Can I Run DeepSeek V4 Flash Today?

| Question | Answer |
|---|---|
| Fits in 96 GB natively? | No — needs ~170–175 GB |
| Fits in 96 GB at Q4? | Barely — ~86–96 GB weights, almost no KV cache |
| SM120 officially supported? | No — community patches only |
| Community patches stable? | Moderately — SGLang patch works; vLLM PR pending merge |
| Better than Qwen3.6-27B for coding? | No — significantly worse on LiveCodeBench and all other coding metrics |
| Worth replacing Qwen3.6-27B? | No — unless 1 M context is specifically needed |

### What Would Be Required

To run V4 Flash meaningfully on this hardware:

1. **Add a second RTX PRO 6000 (96 GB)** → 192 GB total → run native FP4+FP8 at TP=2
2. **Apply SM120 patches**: Either vLLM `jasl/ds4-sm120-preview` branch or `0xSero/deepseek-v4-flash-sm120` SGLang patch
3. **Use SGLang** (more stable than patched vLLM on SM120 for this model)
4. **Accept PCIe multi-GPU penalty**: Expect ~30–40% lower throughput than an equivalent NVLink rig

### Recommendation

Keep Qwen3.6-27B-NVFP4 + DFlash for coding. DeepSeek V4 Flash is a larger, more
expensive-to-serve model that currently trails on every coding benchmark except SWE-bench
Verified (and by only 1.8 points). Wait for:
- Upstream SM120 support in vLLM (PR #41834 merge)
- A second 96 GB card before attempting V4 Flash
- Community Q4 GGUF availability via llama.cpp (CSA+HCA support is still patchy there)

---

## 9. Reference Links

| Resource | URL |
|---|---|
| HuggingFace model card | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash |
| Official vLLM recipe | https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash |
| vLLM PR #41834 (SM120 fix) | https://github.com/vllm-project/vllm/pull/41834 |
| vLLM issue #40802 (SM120 original) | https://github.com/vllm-project/vllm/issues/40802 |
| vLLM issue #40928 (Triton fallback) | https://github.com/vllm-project/vllm/issues/40928 |
| DeepGEMM issue #317 (SM120 kernels) | https://github.com/deepseek-ai/DeepGEMM/issues/317 |
| HuggingFace discussion #15 (DEEP_GEMM) | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/discussions/15 |
| HuggingFace discussion #28 (SM120 vLLM) | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/discussions/28 |
| 0xSero SGLang SM120 patch | https://github.com/0xSero/deepseek-v4-flash-sm120 |
| jasl vLLM SM120 fork | https://github.com/jasl/vllm (branch: ds4-sm120-preview) |
| Spheron deployment guide | https://www.spheron.network/blog/deploy-deepseek-v4-flash-gpu-cloud/ |
| Codersera 4× RTX PRO 6000 guide | https://codersera.com/blog/deepseek-v4-flash-rtx-pro-6000-blackwell-benchmarks-2026/ |
| DSpark announcement (MarkTechPost) | https://www.marktechpost.com/2026/06/27/deepseek-releases-dspark-a-speculative-decoding-framework-that-accelerates-deepseek-v4-per-user-generation-60-85-over-mtp-1/ |
| NVIDIA Blackwell blog | https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/ |
| VRAM requirements table | https://knightli.com/en/2026/05/01/deepseek-v4-local-vram-quantization-table/ |
| BenchLM V4 Flash vs Qwen3.6 | https://benchlm.ai/compare/deepseek-v4-flash-vs-qwen3-6-27b |

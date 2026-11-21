# vLLM day-0 Inkling 2 Mini image + our SM120 patches, as one small overlay layer (no CUDA compile).
# The patched files come from branch inkling2-mini-sm120 of github.com/dbirks/vllm, pinned by commit
# (VLLM_REF), on top of vllm-project/vllm PR #69028 head eb7981b (byte-identical to the image's Inkling files):
#   1. vllm-project/vllm#69096: DSpark draft KV write without reshape_and_cache_flash (no SM120 kernel in this
#      image, issue #69081). Replaces the patch-at-start ConfigMap in apps/inkling2-mini-vllm.yaml.
#   2. NVFP4 KV cache for Inkling2 on SM12x (--kv-cache-dtype nvfp4): 72-byte rows (64 B E2M1 + 8 E4M3 scales).
# Built by .github/workflows/build-vllm-inkling.yaml.
FROM vllm/vllm-openai:inkling-2-mini@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e

ARG VLLM_REF
ARG SITE=/usr/local/lib/python3.14/dist-packages
ARG RAW=https://raw.githubusercontent.com/dbirks/vllm

RUN set -eux; \
    test -n "${VLLM_REF}"; \
    cd "${SITE}"; \
    for f in \
      vllm/models/inkling2/nvidia/attention.py \
      vllm/models/inkling2/nvidia/dspark.py \
      vllm/models/inkling2/nvidia/ops/attention.py \
      vllm/models/inkling2/nvidia/ops/attention_sm90.py \
      vllm/models/inkling2/nvidia/ops/dspark.py \
      vllm/models/inkling2/nvidia/ops/nvfp4_kv_ref.py \
      vllm/models/inkling2/nvidia/ops/nvfp4_kv_write.py \
      vllm/models/inkling2/nvidia/ops/sconv.py \
      vllm/models/inkling2/speculator_config.py; do \
        python3 -c "import urllib.request,sys; open(sys.argv[2],'wb').write(urllib.request.urlopen(sys.argv[1]).read())" \
          "${RAW}/${VLLM_REF}/${f}" "${f}"; \
        python3 -m py_compile "${f}"; \
    done; \
    grep -q '"nvfp4")' vllm/models/inkling2/nvidia/attention.py; \
    grep -q 'KV_NVFP4' vllm/models/inkling2/nvidia/ops/attention_sm90.py; \
    grep -q 'nvfp4_store_kv_rows' vllm/models/inkling2/nvidia/ops/sconv.py; \
    grep -q 'write_kv_paged(k, v' vllm/models/inkling2/nvidia/dspark.py; \
    echo "${VLLM_REF}" > /opt/inkling-sm120-ref
LABEL org.opencontainers.image.source="https://github.com/dbirks/vllm/tree/inkling2-mini-sm120"

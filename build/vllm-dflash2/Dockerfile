# vLLM + DFlash2 (PR #52816) — pure-Python overlay, no CUDA compile.
# Base = Aug-18-2026 upstream nightly (plain nightly-<sha>, the variant our SM120 node already runs).
# PR #52816 touches ZERO csrc/*.cu, so we keep the base image's compiled kernels and overlay only the
# PR's Python tree (whole tree, for internal consistency — the base commit isn't the exact PR base).
FROM vllm/vllm-openai:nightly-aa9903490c616dc6871e5acc62cec7bb1e5e9434

ARG PR=52816
RUN set -eux; \
    git clone --filter=blob:none https://github.com/vllm-project/vllm.git /tmp/vllm; \
    cd /tmp/vllm; git fetch origin "pull/${PR}/head"; git checkout FETCH_HEAD; \
    V="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"; \
    echo "installed vllm at: $V"; \
    # overlay PR .py over the installed package; KEEP base compiled artifacts (.so/.pyd) since csrc is unchanged
    tar -C /tmp/vllm/vllm --exclude='*.so' --exclude='*.pyd' --exclude='__pycache__' -cf - . \
      | tar -C "$V" -xf -; \
    find "$V" -name '__pycache__' -type d -prune -exec rm -rf {} + || true; \
    rm -rf /tmp/vllm; \
    python3 -c "import vllm; import vllm.model_executor.models.qwen3_dflash2; import vllm.v1.worker.gpu.spec_decode.dflash2.speculator; from vllm.model_executor.models.registry import ModelRegistry; print('DFlash2 overlay import OK')"

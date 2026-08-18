# vLLM + DFlash2 (PR #52816) — pure-Python overlay, no CUDA compile.
# Base = Aug-18-2026 upstream nightly (plain nightly-<sha>, the variant our SM120 node already runs).
# PR #52816 touches ZERO csrc/*.cu, so we keep the base image's compiled kernels and overlay only the
# PR's Python tree. The PR source (vllm/ package) is staged into the build context as `vllm-pr` by the
# CI workflow (the runner has git; the vLLM runtime image does NOT), then COPY'd in — no git in-image.
FROM vllm/vllm-openai:nightly-aa9903490c616dc6871e5acc62cec7bb1e5e9434

COPY vllm-pr /tmp/vllm-pr
RUN set -eux; \
    V="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"; \
    echo "installed vllm at: $V"; \
    # overlay PR .py over the installed package; KEEP base compiled artifacts (.so/.pyd) since csrc is unchanged
    tar -C /tmp/vllm-pr --exclude='*.so' --exclude='*.pyd' --exclude='__pycache__' -cf - . \
      | tar -C "$V" -xf -; \
    find "$V" -name '__pycache__' -type d -prune -exec rm -rf {} + || true; \
    rm -rf /tmp/vllm-pr; \
    python3 -c "import vllm; import vllm.model_executor.models.qwen3_dflash2; import vllm.v1.worker.gpu.spec_decode.dflash2.speculator; from vllm.model_executor.models.registry import ModelRegistry; print('DFlash2 overlay import OK')"

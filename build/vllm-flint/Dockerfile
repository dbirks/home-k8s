# vLLM day-0 Flint image + the SM120 fixes from vllm PR #62938's comment thread (#158).
#  1. `_flint_kernels()` has no arch gate, so on SM120 (RTX PRO 6000 / 5090) it picks the Flint MoE, whose
#     FlashInfer CuTe DSL / trtllm-gen kernels exist only for SM90/SM100 ("No supported CUDA architectures
#     found for major versions [9, 10]"). Gate it on SM100, so SM120 takes vLLM's stock layers: Marlin W4A16
#     NVFP4 linears + MoE, FlashInfer attention with sinks (verified on an RTX PRO 6000 by DataCraftsmanAU:
#     LiveCodeBench v6 82.2%, SWE-bench Verified 80%; zflowai on a 5090: 2,572 tok/s at 32 concurrent).
#  2. The day-0 image lacks `gguf` (needed by the NVFP4 GGUF loader, >= 0.19.0).
#  3. CUDA forward-compat libs in the image clash with the node's 580 driver: drop them.
#  4. fp8_kv.py: allow --kv-cache-dtype fp8 on the stock path (vLLM writes the fp8 cache, not the fused
#     BF16-only Triton kernel). Doubles the KV pool. See the header of fp8_kv.py.
FROM vllm/vllm-openai:flint@sha256:be337371b88c770a25a7c029ecc1096dde132889853609b28093ddabc002b0bd

RUN set -eux; \
    pip install --no-cache-dir "gguf>=0.19.0"; \
    rm -rf /usr/local/cuda-*/compat; \
    python3 - <<'PY'
import os, re, vllm
p = os.path.join(os.path.dirname(vllm.__file__), "models/flint/model.py")
s = open(p).read()
sig = "def _flint_kernels("
assert s.count(sig) == 1, "signature not found"
i = s.index(sig)
j = s.index('"""', s.index('"""', i) + 3) + 3   # end of the docstring
gate = ("\n    from vllm.platforms import current_platform\n\n"
        "    # dbirks/home-k8s#158: Flint kernels are SM100-only (CuTe DSL / trtllm-gen); SM120 takes stock layers.\n"
        "    if not current_platform.is_device_capability_family(100):\n"
        "        return False")
s = s[:j] + gate + s[j:]
open(p, "w").write(s)
print(s[i:j + len(gate) + 200])
PY
COPY fp8_kv.py /tmp/fp8_kv.py
RUN python3 /tmp/fp8_kv.py && rm /tmp/fp8_kv.py
RUN python3 -c "import gguf, vllm.models.flint.model as m; print('gguf', gguf.__version__ if hasattr(gguf,'__version__') else 'ok'); import inspect; assert 'is_device_capability_family(100)' in inspect.getsource(m._flint_kernels); assert 'unified_kv_cache_update(k, v' in inspect.getsource(m.FlintAttention.forward)"

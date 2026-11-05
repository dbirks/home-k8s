# Ripple-65B Preview NVFP4 on stock SGLang 0.5.19 (CUDA 13.0) for one RTX PRO 6000 (SM120), issue #168.
#
# Base = the exact SGLang release Ricursive pins (inference/server/serve.py: sglang==0.5.19, torch cu130).
# Overlay, no compile, no runtime pip: Ricursive's model code + parsers + chat template in /opt/ripple.
#   model.py         ricursiveAI/Ripple-65B-Preview-NVFP4 @ 99eea447 inference/local/model.py (quant-aware), plus
#                    the load-gap check from the BF16 server copy (RIPPLE_STRICT_LOAD=0 downgrades it to a log)
#   ripple_format.py ricursiveAI/Ripple-65B-Preview @ e27e1717 inference/server/ripple_format.py (newer than the
#                    local copy: reconciles tool-call deltas a speculative step can leave buffered)
#   chat_template.jinja  repo-root template (same sha256 in both repos)
#   serve.py         our launcher: registration only, all flags from the Deployment
# NOT taken from inference/local/spark.py: its SM121 (GB10) hacks - PDL-off softmax patch for the DFlash
# accept kernel, k/v_scale reset (this checkpoint has no KV scales), the cc==(12,1) gate, and its
# max-running-requests 1 / disable-radix-cache defaults.
FROM lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9

# CUDA forward-compat libs must never shadow the node's 580 driver (same fix as build/vllm-flint).
RUN rm -rf /usr/local/cuda-*/compat /usr/local/cuda/compat || true

COPY ripple/ /opt/ripple/
ENV PYTHONDONTWRITEBYTECODE=1

COPY check.py /tmp/check.py
# Registration check. The CI runner has no GPU/libcuda, so a CUDA-only import failure is reported and
# downgraded to a byte-compile; anything else (bad sha, missing parser hook) fails the build.
RUN python3 /tmp/check.py && rm /tmp/check.py

ENTRYPOINT ["python3", "/opt/ripple/serve.py"]
